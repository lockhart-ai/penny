"""The conversation state machine — its classifier machinery and its durable
half (#1706).

The machine's structural invariants are pinned as data assertions (the edge
table: break-out from every classifying state, no learn edge out of idle, no
out-edges at all from apply) and pure-function contracts (fail → stay in
``next_state``; the apply edge withheld when no skill candidates exist).  The
classifier itself — micro-context customer #3 — is pinned by whole-render
literals of everything the model sees (system prompt, the rendered slice, the
per-edge state meanings) and by the draw mechanics: membership-validated tag
parse, one reroll on a contract violation, poison discard-and-reroll, honest
enumerated failures.

The persistence half (``ConversationMachine``) is pinned on what a machine must
do that a decision alone cannot: hold state across turns, keep the ANCHOR
through a parked round and drop it on break-out, move structurally where the
edge table has no out-edges, and record EVERY draw — including the held ones,
without which per-edge accuracy over the ledger is unmeasurable.

Deterministic mock model responses throughout — the live-model contract is the
eval suite's job (beat 1 onward), not this file's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

import pytest

from penny.constants import PennyConstants, TransitionCause
from penny.conversation_machine import (
    OUT_EDGES,
    STATE_INSTRUCTIONS,
    CandidateParameter,
    ConversationMachine,
    ConversationState,
    MachineSnapshot,
    SkillCandidate,
    StateClassifier,
    build_snapshot,
    conversation_prompt,
    next_state,
    presented_edges,
    render_classifier_content,
)
from penny.database.skills import SkillDraft, SkillStep
from penny.llm.models import LlmMessage, LlmResponse
from penny.prompts import Prompt
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.micro_context import (
    STATE_CLASSIFIER_SYSTEM_PROMPT,
    StateDrawOutcome,
)

# ── Fictional conversation fixtures ───────────────────────────────────────────

_ASK = "hey can you keep an eye on the harbor ferry timetable for me?"
_TEACH_QUESTION = (
    "I don't know how to do that yet — can you teach me? "
    "What should I read, look for, and remember?"
)
_STEPS = "sure — read harborferries.example/timetable and remember the first morning departure"
_SKILL = SkillCandidate(
    name="watch a listing price for changes",
    description="checks a page and records the current price",
    parameters=[CandidateParameter(name="url", description="the listing page to watch")],
)

_IDLE_SNAPSHOT = MachineSnapshot(state=ConversationState.IDLE)
_ELICIT_SNAPSHOT = MachineSnapshot(
    state=ConversationState.ELICIT,
    penny_last_turn=_TEACH_QUESTION,
    task_anchor=_ASK,
)


def _responds_per_call(reply: Callable[[int], str]) -> MockLlmClient:
    """A mock model client whose Nth chat returns ``reply(N)``.

    This file's whole subject IS the classifier, so its handler CLAIMS those
    calls (``answers_state_classifier``) rather than letting the mock's default
    intercept answer them — that intercept exists for the many flow tests that
    say nothing about the machine and must not have it eat their responses."""
    model = MockLlmClient()

    def handler(request: dict, count: int) -> LlmResponse:
        return LlmResponse(message=LlmMessage(role="assistant", content=reply(count)))

    model.set_response_handler(handler, answers_state_classifier=True)
    return model


def _responds(content: str) -> MockLlmClient:
    """A mock model client whose every chat returns ``content``."""
    return _responds_per_call(lambda count: content)


def _classifier(model: MockLlmClient) -> StateClassifier:
    return StateClassifier(cast(Any, model))


# ── The edge table: structural invariants as data assertions ──────────────────


def test_edge_table_invariants():
    """Every state that classifies carries the break-out edge → idle; learn is
    reachable from idle (teaching can arrive unprompted); apply has NO
    out-edges — its reset is structural, never a classifier call."""
    for state, edges in OUT_EDGES.items():
        if edges:
            assert ConversationState.IDLE in edges, f"{state} lacks the break-out edge"
    # Teaching can arrive unprompted, so learn IS reachable from idle.
    assert ConversationState.LEARN in OUT_EDGES[ConversationState.IDLE]
    assert OUT_EDGES[ConversationState.APPLY] == ()
    # learn exits to apply (the demonstrated round ends by OFFERING to set the
    # routine running, so the acceptance is answerable from where that offer was
    # made), stays on further instructions, or falls to the idle default — never
    # back to elicit, which exists to GET instructions already given.
    assert OUT_EDGES[ConversationState.LEARN] == (
        ConversationState.APPLY,
        ConversationState.LEARN,
        ConversationState.IDLE,
    )
    assert ConversationState.ELICIT not in OUT_EDGES[ConversationState.LEARN]


def test_presented_edges_withholds_apply_without_candidates():
    """The SKILL-GATED edges (apply, request) are offered only when the
    snapshot carries skill candidates — an empty registry never renders an
    option whose contract demands naming a skill (the structural false-apply
    guard, and its request twin)."""
    assert presented_edges(_IDLE_SNAPSHOT) == (
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    )
    with_skills = MachineSnapshot(state=ConversationState.IDLE, skill_candidates=[_SKILL])
    assert presented_edges(with_skills) == (
        ConversationState.APPLY,
        ConversationState.REQUEST,
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    )


# ── Whole-render literals: everything the classifier model sees ───────────────


def test_system_prompt_whole_render():
    """Whole-render literal of the dispatch contract: the frame + the
    execution-context guard, the given-inputs list (current state and
    transitions among them), the numbered decision steps (pick the transition
    whose CONDITION is met, else the default), and the output contract."""
    assert STATE_CLASSIFIER_SYSTEM_PROMPT == (
        "You are a dispatch step for a conversation between a user and their "
        "assistant. The assistant has real tools (reading pages, saving values), and a "
        "separate context carries out whatever you decide — NEVER judge whether an "
        "action is possible; your only job is the state.\n"
        "\n"
        "You are given:\n"
        "- The assistant's last message\n"
        "- The task being worked on (when there is one)\n"
        "- Known skills — the assistant's existing routines ((none) when it has none)\n"
        "- The user's newest message\n"
        "- Current state — where the conversation stands right now\n"
        "- Transitions — the states you may move to, each with the condition that "
        "selects it; the last one is the default\n"
        "\n"
        "Do this:\n"
        "1. In your reasoning, note what the user's newest message is doing in the "
        "conversation, judging only from what the messages say.\n"
        "2. Pick the ONE transition whose condition the newest message meets. When "
        "none of the conditions is met, pick the default.\n"
        "3. Check whether the chosen transition directs you to add a SKILL: line.\n"
        "\n"
        "Respond with exactly one line:\n"
        "STATE: <name>\n"
        "The name must be one of the listed transitions, copied EXACTLY. When the "
        "chosen transition directs it, add exactly one more line — SKILL: <the skill's "
        "name, exactly as quoted in Known skills> — and nothing more.\n"
        "IMPORTANT: write nothing else — no preamble, no explanation, no restating the "
        "messages."
    )


def test_render_idle_slice_whole():
    """The idle render, whole: the slice sections, then WHERE the machine
    stands (current state + its canonical definition) and WHAT MOVES IT (one
    line per transition with its condition; idle last as the declared
    default).  Apply is absent — no candidates."""
    assert render_classifier_content(_IDLE_SNAPSHOT, _ASK) == (
        "## The assistant's last message\n"
        "(none)\n"
        "\n"
        "## Known skills\n"
        "(none)\n"
        "\n"
        "## The user's newest message\n"
        "hey can you keep an eye on the harbor ferry timetable for me?\n"
        "\n"
        "## Current state\n"
        "idle — ordinary conversation — chat, questions, passing mentions, or anything "
        "put off for later; no task is being given or taught right now\n"
        "\n"
        "## Transitions\n"
        "- learn — the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, including "
        "corrections to previous steps\n"
        "- elicit — they are asking to set up an ongoing task or routine and no known "
        "skill covers it\n"
        "- idle — in all other cases"
    )


def test_render_parked_elicit_slice_whole():
    """The parked-elicit render, whole: teach question + instigating ask as
    their own sections, elicit named as the current state, and its three
    transitions carrying the conditions that select them."""
    assert render_classifier_content(_ELICIT_SNAPSHOT, _STEPS) == (
        "## The assistant's last message\n"
        "I don't know how to do that yet — can you teach me? What should I read, look "
        "for, and remember?\n"
        "\n"
        "## The task being worked on\n"
        "hey can you keep an eye on the harbor ferry timetable for me?\n"
        "\n"
        "## Known skills\n"
        "(none)\n"
        "\n"
        "## The user's newest message\n"
        "sure — read harborferries.example/timetable and remember the first morning "
        "departure\n"
        "\n"
        "## Current state\n"
        "elicit — the user wants a task done that no known skill covers, and the "
        "assistant is asking to be taught the steps\n"
        "\n"
        "## Transitions\n"
        "- learn — the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, including "
        "corrections to previous steps\n"
        "- elicit — they are still working the task out with the assistant — a "
        "question back, or a clarification about the task itself\n"
        "- idle — in all other cases"
    )


def test_render_parked_learn_slice_whole():
    """The parked-learn render after a FAILED round, whole: with nothing in the
    registry the skill-gated apply edge is withheld, so the union narrows to
    learn (the user provided instructions) and idle (everything else, the
    declared default).  There is no path back to elicit: elicit exists to GET
    the instructions, and they have been given."""
    parked_learn = MachineSnapshot(
        state=ConversationState.LEARN,
        penny_last_turn=(
            "I tried, but the timetable page wouldn't load, so I couldn't save "
            "anything. Should I try again, or is there a different page I should read?"
        ),
        task_anchor=_ASK,
    )
    assert render_classifier_content(parked_learn, "try again — the page should load now") == (
        "## The assistant's last message\n"
        "I tried, but the timetable page wouldn't load, so I couldn't save anything. "
        "Should I try again, or is there a different page I should read?\n"
        "\n"
        "## The task being worked on\n"
        "hey can you keep an eye on the harbor ferry timetable for me?\n"
        "\n"
        "## Known skills\n"
        "(none)\n"
        "\n"
        "## The user's newest message\n"
        "try again — the page should load now\n"
        "\n"
        "## Current state\n"
        "learn — the user's message gives instructions to follow — what to read, look "
        "for, or remember; a plain command counts, and a message without instructions "
        "is never learn\n"
        "\n"
        "## Transitions\n"
        "- learn — the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, including "
        "corrections to previous steps\n"
        "- idle — in all other cases"
    )


def test_render_parked_learn_with_candidates_whole():
    """The parked-learn render after a round that RAN, whole: the round ends by
    offering to set the routine running, the skill it just taught is in the
    registry, so apply joins the union with the SKILL: directive — the edge the
    acceptance takes.  Nothing about the values it needs is asked for: the
    round that just ran supplied them."""
    taught = MachineSnapshot(
        state=ConversationState.LEARN,
        penny_last_turn=(
            "Read the timetable and saved the first morning departure. "
            "Want me to keep it up to date on its own?"
        ),
        task_anchor=_ASK,
        skill_candidates=[_SKILL],
    )
    assert render_classifier_content(taught, "yeah, check it every morning") == (
        "## The assistant's last message\n"
        "Read the timetable and saved the first morning departure. Want me to keep it "
        "up to date on its own?\n"
        "\n"
        "## The task being worked on\n"
        "hey can you keep an eye on the harbor ferry timetable for me?\n"
        "\n"
        "## Known skills\n"
        '- "watch a listing price for changes" — checks a page and records the current '
        "price (needs: url — the listing page to watch)\n"
        "\n"
        "## The user's newest message\n"
        "yeah, check it every morning\n"
        "\n"
        "## Current state\n"
        "learn — the user's message gives instructions to follow — what to read, look "
        "for, or remember; a plain command counts, and a message without instructions "
        "is never learn\n"
        "\n"
        "## Transitions\n"
        "- apply — they are asking for the routine just demonstrated to run on its own. "
        "How often it runs, how long it keeps running, and whether it tells them are the "
        "job's terms; naming any of those is expected here and does not make the message "
        "instructions. Only a change to the routine's own steps — what to read, what to "
        "look for, what to remember, where to save it — is instructions, even when it "
        "also sounds like a yes. Add a second line naming that skill: SKILL: <its name, "
        "exactly as quoted in Known skills>\n"
        "- learn — the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, including "
        "corrections to previous steps\n"
        "- idle — in all other cases"
    )


def test_render_idle_with_candidates_whole():
    """The idle render with a ranked skill candidate, whole: full skill
    metadata in Known skills, and the apply transition joining the list with
    its coverage condition + the SKILL: directive.  A parameterless candidate
    renders without the needs tail, byte-identical."""
    assert SkillCandidate(name="x", description="y").render() == '"x" — y'
    with_skills = MachineSnapshot(state=ConversationState.IDLE, skill_candidates=[_SKILL])
    assert render_classifier_content(with_skills, "what's the ferry price at today?") == (
        "## The assistant's last message\n"
        "(none)\n"
        "\n"
        "## Known skills\n"
        '- "watch a listing price for changes" — checks a page and records the current '
        "price (needs: url — the listing page to watch)\n"
        "\n"
        "## The user's newest message\n"
        "what's the ferry price at today?\n"
        "\n"
        "## Current state\n"
        "idle — ordinary conversation — chat, questions, passing mentions, or anything "
        "put off for later; no task is being given or taught right now\n"
        "\n"
        "## Transitions\n"
        "- apply — one of the known skills does what they are asking for AND their "
        "message supplies everything that skill needs — mere resemblance to a skill is "
        "not coverage — add a second line naming that skill: SKILL: <its name, exactly "
        "as quoted in Known skills>\n"
        "- request — a known skill looks like it covers what they are asking "
        "for, but something that skill needs is missing from their message — add a "
        "second line naming that skill: SKILL: <its name, exactly as quoted in Known "
        "skills>\n"
        "- learn — the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, including "
        "corrections to previous steps\n"
        "- elicit — they are asking to set up an ongoing task or routine and no known "
        "skill covers it\n"
        "- idle — in all other cases"
    )


# ── The classifier draw: membership, rerolls, attribution, fail → stay ────────


@pytest.mark.asyncio
async def test_classify_decides_with_attribution_and_exact_model_input():
    """A tagged in-union draw decides the transition, and the single call
    carries the classifier's own ledger attribution plus exactly the dispatch
    system prompt and the rendered slice — the whole model input, pinned."""
    model = _responds("STATE: elicit")
    decision = await _classifier(model).classify(_IDLE_SNAPSHOT, _ASK, run_target="chat")
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.ELICIT
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request["agent_name"] == PennyConstants.STATE_CLASSIFIER_AGENT_NAME
    assert request["prompt_type"] == PennyConstants.STATE_CLASSIFIER_PROMPT_TYPE
    assert request["run_target"] == "chat"
    assert request["messages"][0]["content"] == STATE_CLASSIFIER_SYSTEM_PROMPT
    # The classifier's user turn is the bare rendered situation — no
    # Instruction:/Content: wrapper (the system prompt owns the ask).
    assert request["messages"][1]["content"] == render_classifier_content(_IDLE_SNAPSHOT, _ASK)


@pytest.mark.asyncio
async def test_classify_out_of_union_draw_is_rerolled_then_stays():
    """A drawn state OUTSIDE the offered union is a contract violation exactly
    like an untagged draw: one reroll of the unchanged context, then an honest
    INVALID the machine holds its state on — apply is WITHHELD from a
    candidate-less idle snapshot, so a flaky draw can never conjure an apply
    against an empty registry."""
    model = _responds("STATE: apply")
    decision = await _classifier(model).classify(_IDLE_SNAPSHOT, _ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert decision.state is None
    assert len(model.requests) == 2  # the draw + exactly one reroll
    assert next_state(ConversationState.IDLE, decision) == ConversationState.IDLE


@pytest.mark.asyncio
async def test_classify_untagged_draw_is_rerolled_then_stays():
    """Untagged (but clean) output takes the same path: one reroll, then
    INVALID — prose is never promoted to a transition."""
    model = _responds("sure, sounds good")
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert len(model.requests) == 2
    assert next_state(ConversationState.ELICIT, decision) == ConversationState.ELICIT


@pytest.mark.asyncio
async def test_classify_tolerates_a_decorated_or_quoted_draw():
    """The declared shape's tolerance (#1814) reaches the classifier too: a draw
    whose tag arrives bolded or list-marked, or whose state name arrives wrapped in
    quotes, is the model getting the contract RIGHT in a cosmetically different way.
    Membership is exact, so without tolerance a correct decision would have been
    read as out-of-union, rerolled, and then held the machine where it stood."""
    model = _responds('- **STATE:** "learn"')
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.LEARN
    assert len(model.requests) == 1  # no reroll — the draw was valid all along


@pytest.mark.asyncio
async def test_classify_reroll_can_recover():
    """The one contract-violation reroll re-draws on the unchanged context — a
    valid second draw decides the transition."""
    model = _responds_per_call(lambda count: "hmm, let me think" if count == 1 else "STATE: learn")
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.LEARN
    assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_classify_poison_is_discarded_then_stays():
    """Poison output (a degeneration collapse) is discarded and re-drawn on the
    unchanged context up to the reroll budget, then fails honestly — and the
    machine holds its state (a poisoned draw can never eject a parked teach
    loop)."""
    model = _responds("...???...")
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.POISON_REROLL_FAILED
    assert decision.state is None
    assert len(model.requests) == 3
    assert next_state(ConversationState.ELICIT, decision) == ConversationState.ELICIT


@pytest.mark.asyncio
async def test_classify_from_apply_refuses():
    """Apply has no out-edges — its reset to idle is a post-turn structural
    fact.  Asking the classifier to run there is a programming error, refused
    loudly rather than classified into nonsense."""
    with pytest.raises(ValueError, match="structural"):
        await _classifier(_responds("STATE: idle")).classify(
            MachineSnapshot(state=ConversationState.APPLY), "great, thanks!"
        )


# ── The skill-gated apply draw (#1706 beat 2) ─────────────────────────────────

_WITH_SKILL = MachineSnapshot(state=ConversationState.IDLE, skill_candidates=[_SKILL])
_PRICE_ASK = "can you watch the price on ridgelinefoxes.example/den-camera-kit?"


async def test_classify_apply_draw_binds_a_listed_skill():
    """An apply draw carrying a SKILL: line naming a listed candidate decides
    apply WITH the skill bound — the machine never receives a dangling apply."""
    model = _responds("STATE: apply\nSKILL: watch a listing price for changes")
    decision = await _classifier(model).classify(_WITH_SKILL, _PRICE_ASK)
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.APPLY
    assert decision.skill == "watch a listing price for changes"
    assert len(model.requests) == 1


async def test_classify_apply_without_skill_line_is_rerolled_then_stays():
    """Drawing the gated state WITHOUT its SKILL: line is a contract violation
    exactly like an untagged draw: one reroll, then INVALID — fail → stay."""
    model = _responds("STATE: apply")
    decision = await _classifier(model).classify(_WITH_SKILL, _PRICE_ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert decision.skill is None
    assert len(model.requests) == 2
    assert next_state(ConversationState.IDLE, decision) == ConversationState.IDLE


async def test_classify_apply_with_unlisted_skill_is_rerolled_then_stays():
    """A SKILL: payload outside the offered candidates is the same violation —
    the bound skill is membership-validated, never a free-text guess."""
    model = _responds("STATE: apply\nSKILL: fold the laundry")
    decision = await _classifier(model).classify(_WITH_SKILL, _PRICE_ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert len(model.requests) == 2


async def test_classify_ungated_draw_ignores_a_stray_skill_line():
    """A stray SKILL: line on a NON-gated draw binds nothing and does not
    invalidate the decision — only the gated state demands (or reads) it."""
    model = _responds("STATE: idle\nSKILL: watch a listing price for changes")
    decision = await _classifier(model).classify(_WITH_SKILL, "morning! how's it going?")
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.IDLE
    assert decision.skill is None
    assert len(model.requests) == 1


# ── The production snapshot builder ───────────────────────────────────────────


def _seed_skill(db, name: str, description: str) -> None:
    draft = SkillDraft(
        name=name,
        intent=description,
        description=description,
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="browse",
                arguments={"queries": ["https://example.test"]},
            )
        ],
        parameters=[],
        source_run_id="test-seed",
    )
    db.skills.upsert(draft, author="test-seed", description_embedding=None)


def test_build_snapshot_offers_every_skill_with_no_relevance_gate(db):
    """EVERY skill is offered — no ranking, no cap, no embedding (the code-owner
    ruling): a relevance gate here would hide the covering skill exactly like
    the gates #1471 removed, and this list is strictly smaller than the full
    recipes chat already carries every turn.  A skill with no description vector
    is offered like any other, since nothing is compared."""
    _seed_skill(db, "watch a listing price for changes", "record a listing's price")
    _seed_skill(db, "collect daily cafe specials", "save a cafe's daily specials")
    snapshot = build_snapshot(db, state=ConversationState.IDLE, message=_PRICE_ASK)
    assert [candidate.name for candidate in snapshot.skill_candidates] == [
        "collect daily cafe specials",
        "watch a listing price for changes",
    ]
    assert ConversationState.APPLY in presented_edges(snapshot)


def test_build_snapshot_on_an_empty_registry_withholds_the_gated_states(db):
    """An empty registry offers no candidates, so the SKILL-GATED states are
    structurally withheld — the cold-start shape, reached by there being nothing
    to offer rather than by anything failing."""
    snapshot = build_snapshot(db, state=ConversationState.IDLE, message=_PRICE_ASK)
    assert snapshot.skill_candidates == []
    assert ConversationState.APPLY not in presented_edges(snapshot)


# ── The durable half: state held across turns, every move recorded ────────────

_KAYAK_ASK = "keep an eye on the price of the harbor kayak rental page"


def _machine(db, model: MockLlmClient) -> ConversationMachine:
    """A machine over the real store, driven by a mock model."""
    return ConversationMachine(db, _classifier(model))


def _log(db, content: str) -> int:
    message_id = db.messages.log_message(direction="incoming", sender="tester", content=content)
    assert message_id is not None
    return message_id


async def test_machine_cold_starts_idle_and_records_the_move(db):
    """First read creates the row at idle (no seeded state), and a decided draw
    both moves the machine and lands one classifier transition carrying its
    outcome, message, run and bound skill."""
    _seed_skill(db, "watch a listing price for changes", "record a listing's price")
    machine = _machine(db, _responds("STATE: apply\nSKILL: watch a listing price for changes"))
    assert db.machine.latest_transition() is None  # cold start = no history, not a seeded row
    assert machine.state() is ConversationState.IDLE

    message_id = _log(db, _KAYAK_ASK)
    decision = await machine.advance(_KAYAK_ASK, message_id=message_id, run_id="run-1")

    assert decision.state is ConversationState.APPLY
    assert machine.state() is ConversationState.APPLY
    (transition,) = db.machine.recent_transitions(10)
    assert transition.from_state == ConversationState.IDLE.value
    assert transition.to_state == ConversationState.APPLY.value
    assert transition.cause == TransitionCause.CLASSIFIER.value
    assert transition.outcome == StateDrawOutcome.DECIDED.value
    assert transition.message_id == message_id
    assert transition.run_id == "run-1"
    assert transition.skill_name == "watch a listing price for changes"


async def test_held_draw_stays_put_but_is_still_recorded(db):
    """Fail → stay, with the non-decision RECORDED: a ledger that logged only
    successful moves would report a perfect classifier by construction, so the
    held draw lands as a self-edge carrying its honest outcome."""
    machine = _machine(db, _responds("I think we should probably elicit here"))
    decision = await machine.advance(_ASK, message_id=_log(db, _ASK))

    assert decision.state is None
    assert machine.state() is ConversationState.IDLE
    (transition,) = db.machine.recent_transitions(10)
    assert transition.from_state == transition.to_state == ConversationState.IDLE.value
    assert transition.outcome == StateDrawOutcome.INVALID.value
    assert transition.skill_name is None


async def test_anchor_is_set_on_entry_kept_while_parked_and_cleared_on_break_out(db):
    """The anchor lifecycle end to end: the instigating ask is captured entering
    elicit, SURVIVES the parked round (a later message never overwrites it — a
    reply is classified against the ask it answers, not against itself), and is
    dropped on the break-out to idle."""
    anchor_id = _log(db, _ASK)
    machine = _machine(db, _responds("STATE: elicit"))
    await machine.advance(_ASK, message_id=anchor_id)
    assert machine.state() is ConversationState.ELICIT
    assert db.machine.latest_transition().anchor_message_id == anchor_id

    staying = _machine(db, _responds("STATE: elicit"))
    await staying.advance("wait — what exactly do you need?", message_id=_log(db, "wait — what?"))
    assert db.machine.latest_transition().anchor_message_id == anchor_id

    bailing = _machine(db, _responds("STATE: idle"))
    await bailing.advance("never mind, forget it", message_id=_log(db, "never mind"))
    assert bailing.state() is ConversationState.IDLE
    assert db.machine.latest_transition().anchor_message_id is None


async def test_parked_anchor_reaches_the_classifier_as_the_task(db):
    """The stored anchor is READ back into the snapshot as the task being worked
    on — the whole point of persisting it — resolved from the message row rather
    than a copy the machine keeps."""
    anchor_id = _log(db, _ASK)
    await _machine(db, _responds("STATE: elicit")).advance(_ASK, message_id=anchor_id)

    model = _responds("STATE: learn")
    await _machine(db, model).advance(_STEPS, message_id=_log(db, _STEPS))
    assert _ASK in model.requests[-1]["messages"][1]["content"]


async def test_apply_resets_structurally_then_classifies_the_new_message(db):
    """A state with no out-edges cannot be classified, so the machine settles it
    FIRST: the reset lands as its own structural row (no model, no outcome) and
    the message is then classified from idle — two rows, in causal order."""
    _seed_skill(db, "watch a listing price for changes", "record a listing's price")
    db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=ConversationState.APPLY.value,
        cause=TransitionCause.CLASSIFIER,
    )
    machine = _machine(db, _responds("STATE: idle"))
    await machine.advance("thanks!", message_id=_log(db, "thanks!"))

    assert machine.state() is ConversationState.IDLE
    _parked, reset, classified = reversed(db.machine.recent_transitions(10))
    assert reset.from_state == ConversationState.APPLY.value
    assert reset.to_state == ConversationState.IDLE.value
    assert reset.cause == TransitionCause.STRUCTURAL.value
    assert reset.outcome is None
    assert classified.cause == TransitionCause.CLASSIFIER.value
    assert classified.from_state == ConversationState.IDLE.value


async def test_state_is_a_fold_over_the_log_with_no_materialized_twin(db):
    """The whole state IS the newest row — state, anchor and last-moved time —
    so nothing can drift out of step with the audit trail.  Appending a move by
    hand moves the machine, which is the property that makes the second table
    unnecessary."""
    anchor_id = _log(db, _ASK)
    db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=ConversationState.ELICIT.value,
        cause=TransitionCause.CLASSIFIER,
        anchor_message_id=anchor_id,
        outcome=StateDrawOutcome.DECIDED.value,
    )
    machine = _machine(db, _responds("STATE: idle"))
    assert machine.state() is ConversationState.ELICIT

    latest = db.machine.latest_transition()
    assert latest is not None
    assert (latest.to_state, latest.anchor_message_id) == (
        ConversationState.ELICIT.value,
        anchor_id,
    )


# ── Per-state chat instructions (#1706) ─────────────────────────────────────


def test_every_state_has_an_instruction_and_there_is_no_fallback():
    """TOTAL by construction: the machine always has a state, so a turn always
    has exactly one instruction.  A default would mean the state failed to
    determine the prompt — the one thing the machine exists to fix — so a
    missing entry must RAISE rather than quietly compose someone else's."""
    assert set(STATE_INSTRUCTIONS) == set(ConversationState)
    for state in ConversationState:
        prompt = conversation_prompt(state)
        assert prompt.startswith(Prompt.CONVERSATION_HEAD)
        assert prompt.endswith(Prompt.CONVERSATION_TAIL)
        assert STATE_INSTRUCTIONS[state] in prompt


def test_each_state_carries_only_its_own_instruction():
    """No union: a state's prompt contains ITS instruction and no other's.  This
    is what the machine buys — #1687's four-case block existed only so the model
    could work out which case applied, and that is answered in Python now."""
    for state in ConversationState:
        prompt = conversation_prompt(state)
        for other, instruction in STATE_INSTRUCTIONS.items():
            if other is not state:
                assert instruction not in prompt, f"{state} leaked {other}'s instruction"


def test_no_instruction_names_the_machine():
    """Nothing renders the machine to chat — no state name, no transitions, no
    hint a classifier ran.  Where the conversation stands is already decided;
    what the turn needs is what to do."""
    machine_words = ("idle", "elicit", "learn", "apply", "request-details", "transition", "classif")
    for state, instruction in STATE_INSTRUCTIONS.items():
        for word in machine_words:
            assert word not in instruction.lower(), f"{state} instruction names '{word}'"


def test_the_un_stated_prompt_is_idle():
    """With no machine wired (or a classifier failure) the chat agent falls back
    to ``CONVERSATION_PROMPT`` — which IS idle's composition, not a second
    definition that could drift from it."""
    assert conversation_prompt(ConversationState.IDLE) == Prompt.CONVERSATION_PROMPT


def test_elicit_instruction_whole_render():
    """The whole instruction, verbatim — pinned so an edit is a visible diff.

    Generically and minimally sufficient to enact the state: no task shape, no
    example phrasing, and no guard written against a particular failed sample.
    (Both of those crept in from #1687 and had to come back out — one of them
    quoted an eval fixture's own words, which is the contamination the
    definitions-are-product-semantics rule exists to catch.)"""
    assert Prompt.ELICIT_INSTRUCTION == (
        "The user has asked for a task you have no skill for. Your job this turn "
        "is to get the instructions from them.\n\n"
        "In ONE message, ask them to walk you through doing it once: what to "
        "read, what to do with it, and what to remember afterwards. Ask in the "
        "terms they used — describing the task is theirs, working out how to "
        "carry it out is yours. Never ask them to define keywords, terms, "
        "matching rules, css or selectors, or anything about how a page is "
        "built.\n\n"
        "Don't attempt the task, don't do part of it, and don't record anything. "
        "Nothing exists yet, so don't say or imply that it does.\n\n"
    )


def test_apply_instruction_whole_render():
    """The whole instruction, verbatim — pinned so an edit is a visible diff.

    ONE coherent statement, not a boundary policed twice: the call to make, then the
    scope of the turn (configuring is all of it, the routine runs itself afterwards),
    then what to report and the one condition on claiming it is running.  It was
    patched by accretion once — a third paragraph restating the reporting boundary from
    another angle — which is the drift this literal makes visible."""
    assert Prompt.APPLY_INSTRUCTION == (
        "A skill you already know does what the user is asking, and they have "
        "given you everything it needs. Set it up now, in one `collection_set` "
        "call, binding what they told you.\n\n"
        "Configuring it is the whole turn — you are not carrying the routine out "
        "yourself. Once it is set up it runs itself on the schedule they just "
        "gave you, and its first run is the first thing they'll hear about.\n\n"
        "Then tell them what you set up and what will happen. Say it is running "
        "only if the call came back confirming it.\n\n"
    )


def test_no_instruction_carries_a_task_shape_or_an_example_phrasing():
    """An instruction describes the STATE, never a kind of task or a form of
    words a user might use.  Both leaked in from #1687 — an example filter quoted
    verbatim from an eval fixture, and a numbered template mirroring that
    fixture's teach turn — which is a prompt describing its own test pool."""
    for state, instruction in STATE_INSTRUCTIONS.items():
        # A quoted phrase (an apostrophe opening after whitespace) is an example
        # of what a user might say — contractions don't match, quoted specimens do.
        assert not re.search(r"(?<=\s)'[^']+'", instruction), f"{state} quotes an example phrasing"
        assert not re.search(r"\d\.\s+(go to|pull out|visit)", instruction), (
            f"{state} models a task-shaped template"
        )
