"""The conversation state machine — its classifier machinery and its durable
half (#1706).

The machine's structural invariants are pinned as data assertions (the edge
table: break-out from every classifying state, no learn edge out of idle, no
out-edges at all from apply) and pure-function contracts (fail → stay in
``next_state``; the apply edge withheld when no skill candidates exist).  The
classifier itself — micro-context customer #3 — is pinned by whole-render
literals of everything the model sees (system prompt, the rendered slice, the
per-edge state meanings) and by the draw mechanics: membership-validated tag
parse, discard-and-reroll on a contract violation, poison discard-and-reroll, honest
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
    ROUND_LINES,
    STATE_INSTRUCTIONS,
    CandidateParameter,
    ConversationMachine,
    ConversationState,
    MachineSnapshot,
    RoundFraming,
    RoundShortfall,
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
    FramedParameter,
    SkillSignature,
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
    learn and idle (everything else, the declared default).  There is no path
    back to elicit: elicit exists to GET the instructions, and they have been
    given.

    Once a round has run, learn's condition is the CORRECTION shape in two
    cases: the steps restated with changes, or a plain ask to run it again after
    a hiccup — which is this sample's message, and which restates no steps at
    all.  Either way the correction has to be IN the message: saying only that
    the round was wrong, or promising new instructions later, carries none,
    which is what keeps a bail on the idle default rather than reading as a
    correction because it sounds dissatisfied."""
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
        "- learn — the user is correcting their previous instructions — this message "
        "itself restates the steps with changes: what to read, what to look for, what "
        "to remember, where to save it — or asks to simply run it again after a "
        "hiccup. A message that only says the round was wrong, or promises new "
        "instructions later, carries no correction.\n"
        "- idle — in all other cases"
    )


def test_render_parked_learn_with_candidates_whole():
    """The parked-learn render after a round that RAN, whole: the round ends by
    offering to set the routine running, the skill it just taught is in the
    registry, so apply joins the union with the SKILL: directive — the edge the
    acceptance takes.  Nothing about the values it needs is asked for: the
    round that just ran supplied them.

    The two live edges are a CHOICE MENU, each stating only its own shape: apply
    is an acceptance of what was just demonstrated, with the job's terms welcome
    but not required; learn is a correction the message itself carries — steps
    restated with changes, or an ask to just run it again.  Neither argues
    against the other — the sibling is in the same list saying what it is."""
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
        "- apply — the user signals positively — accepting what was just demonstrated: "
        "a yes, a great, a go-ahead. They often add how the job should run — its "
        "timing, how long it keeps going, or whether to tell them — but a plain "
        "acceptance is enough. Add a second line naming that skill: SKILL: <its name, "
        "exactly as quoted in Known skills>\n"
        "- learn — the user is correcting their previous instructions — this message "
        "itself restates the steps with changes: what to read, what to look for, what "
        "to remember, where to save it — or asks to simply run it again after a "
        "hiccup. A message that only says the round was wrong, or promises new "
        "instructions later, carries no correction.\n"
        "- idle — in all other cases"
    )


def test_render_idle_with_candidates_whole():
    """The idle render with a ranked skill candidate, whole: full skill
    metadata in Known skills, and the apply transition joining the list with
    its coverage condition + the SKILL: directive.  A parameterless candidate
    renders without the needs tail, byte-identical.

    The apply condition states the SKILL-DOES-ONCE fact (code-owner authored): a skill
    carries out its task a single time, and a schedule and notifications are added when
    it is set up — so a routine described as doing the job once covers an ask to do it
    repeatedly.  The measured class it answers is a cold ask for a recurring watch drawn
    as elicit or request because the covering routine's own description reads as a
    one-shot; stating where cadence LIVES is what makes the two comparable.

    A WATCHED DELETION rides with it: the old "mere resemblance to a skill is not
    coverage" clause is gone.  It argued against a sibling condition from inside this
    one, which the choice-menu discipline above ``TRANSITIONS`` forbids — request is
    right there in the same list saying what it is.  What it guarded (a skill that
    merely looks related being applied) is the classifier suite's idle-apply-hold cases'
    to gate, and they are its gate now."""
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
        "- apply — one of the known skills does what they are asking for, and their "
        "message contains all the information for the skill's parameters. A skill does "
        "the task once. The schedule and notifications are added when it is set up, so "
        "a skill that does the task once covers an ask to do it repeatedly. Add a "
        "second line naming that skill: SKILL: <its name, exactly as quoted in Known "
        "skills>\n"
        "- request — a known skill looks like it covers what they are asking "
        "for, but something that skill needs is missing from their message. A skill "
        "does the task once. The schedule and notifications are added when it is set "
        "up, so a skill that does the task once covers an ask to do it repeatedly. Add "
        "a second line naming that skill: SKILL: <its name, exactly as quoted in Known "
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
    like an untagged draw: discarded and re-drawn on the unchanged context for the
    WHOLE budget — the same patience poison gets, since membership is DETECTED
    against the offered union rather than judged — then an honest INVALID the
    machine holds its state on.  Apply is WITHHELD from a candidate-less idle
    snapshot, so a flaky draw can never conjure an apply against an empty
    registry."""
    model = _responds("STATE: apply")
    decision = await _classifier(model).classify(_IDLE_SNAPSHOT, _ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert decision.state is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert next_state(ConversationState.IDLE, decision) == ConversationState.IDLE


@pytest.mark.asyncio
async def test_classify_untagged_draw_is_rerolled_then_stays():
    """Untagged (but clean) output takes the same path: the whole budget, then
    INVALID — prose is never promoted to a transition."""
    model = _responds("sure, sounds good")
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
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
    """A contract-violation reroll re-draws on the unchanged context — a valid draw
    decides the transition, and it can arrive anywhere inside the budget: TWO
    violations still leave a third draw, which is the whole point of spending the
    poison budget on a detectably-invalid one."""
    model = _responds_per_call(lambda count: "hmm, let me think" if count == 1 else "STATE: learn")
    decision = await _classifier(model).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert decision.outcome == StateDrawOutcome.DECIDED
    assert decision.state == ConversationState.LEARN
    assert len(model.requests) == 2

    late = _responds_per_call(lambda count: "hmm, let me think" if count <= 2 else "STATE: learn")
    recovered = await _classifier(late).classify(_ELICIT_SNAPSHOT, _STEPS)
    assert recovered.outcome == StateDrawOutcome.DECIDED
    assert recovered.state == ConversationState.LEARN
    assert len(late.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


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
    exactly like an untagged draw: the whole budget, then INVALID — fail → stay."""
    model = _responds("STATE: apply")
    decision = await _classifier(model).classify(_WITH_SKILL, _PRICE_ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert decision.skill is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert next_state(ConversationState.IDLE, decision) == ConversationState.IDLE


async def test_classify_apply_with_unlisted_skill_is_rerolled_then_stays():
    """A SKILL: payload outside the offered candidates is the same violation —
    the bound skill is membership-validated, never a free-text guess."""
    model = _responds("STATE: apply\nSKILL: fold the laundry")
    decision = await _classifier(model).classify(_WITH_SKILL, _PRICE_ASK)
    assert decision.outcome == StateDrawOutcome.INVALID
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


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
    entered = await machine.advance(_KAYAK_ASK, message_id=message_id, run_id="run-1")

    assert entered.decision.state is ConversationState.APPLY
    assert entered.state is ConversationState.APPLY
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
    entered = await machine.advance(_ASK, message_id=_log(db, _ASK))

    assert entered.decision.state is None
    assert entered.shortfall is None
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


def _framing() -> RoundFraming:
    """One round's framing, shared by every render pin below — the same signature and the
    same container, so what a state says about ONE framing is the only thing that varies
    between the states that say anything at all."""
    return RoundFraming(
        signature=SkillSignature(
            name="watch-rental-price",
            description="keep a rental page's current day rate up to date",
            parameters=(
                FramedParameter(
                    name="url", description="the rental page to read", value="harborkayak.example"
                ),
            ),
        ),
        container="watch-rental-price-harborkayak-example",
    )


def _shortfall() -> RoundShortfall:
    """One round's shortfall, shared by the render pins below — the SAME routine the
    framing above is for, so what request says about a round and what learn says about it
    differ only in which half of the entry each reads.

    Two parameters on purpose: one the words settled and one they did not, which is the
    only shape that can show both halves of the render at once."""
    return RoundShortfall(
        skill="watch-rental-price",
        description="keep a rental page's current day rate up to date",
        bound={"keyword": "the weekend rate"},
        missing=(CandidateParameter(name="url", description="the rental page to read"),),
    )


def _composed(state: ConversationState) -> str:
    """A state's prompt as that state can actually be composed: apply names the round
    inside its own instruction and has no unframed form (#1875), so it is the one state
    that cannot be asked for a prompt without one."""
    return conversation_prompt(state, _framing() if state is ConversationState.APPLY else None)


def test_every_state_has_an_instruction_and_there_is_no_fallback():
    """TOTAL by construction: the machine always has a state, so a turn always
    has exactly one instruction.  A default would mean the state failed to
    determine the prompt — the one thing the machine exists to fix — so a
    missing entry must RAISE rather than quietly compose someone else's.

    Apply's instruction is a TEMPLATE — it names the round's container and routine — so
    what its prompt carries is that instruction rendered, which is the same claim read
    through the framing it is composed with."""
    assert set(STATE_INSTRUCTIONS) == set(ConversationState)
    for state in ConversationState:
        prompt = _composed(state)
        assert prompt.startswith(Prompt.CONVERSATION_HEAD)
        assert prompt.endswith(Prompt.CONVERSATION_TAIL)
        rendered = STATE_INSTRUCTIONS[state].format(
            skill=_framing().skill, container=_framing().container
        )
        assert rendered in prompt


def test_each_state_carries_only_its_own_instruction():
    """No union: a state's prompt contains ITS instruction and no other's.  This
    is what the machine buys — #1687's four-case block existed only so the model
    could work out which case applied, and that is answered in Python now."""
    for state in ConversationState:
        prompt = _composed(state)
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


def test_learn_instruction_whole_render():
    """The whole instruction, verbatim — pinned so an edit is a visible diff.

    Two of its clauses answer measured thinking (#1838), and each states the RULE
    rather than the sample that found it: a step is done when its tool call has run —
    finding a value is not remembering it, the write is (a demonstrated round reported
    a price it had only browsed); and the job terms an anchored watch ask carries
    (a cadence, being told about changes) are somebody else's turn, said as permission
    to leave them alone rather than as one more prohibition.

    The third paragraph is the stop-and-report caveat (#1855): a step the world cannot
    support ends the round where it stopped, reported honestly, rather than being routed
    around or written up as done.  It names the STATE that stops a round — the page
    doesn't have it, the value never showed up — never a tool or a task shape, so it
    reaches a round built out of tools nobody has written yet.  Permission first ("it's
    okay to stop there"), prohibition second, which is the order the #1838 clauses
    already use.  The over-correction it must NOT cause: a call that failed with an
    actionable error is still fixed and retried — that is a step that CAN be done."""
    assert Prompt.LEARN_INSTRUCTION == (
        "The user has given you the steps for a task. Follow them now, once, "
        "exactly as given — this turn is that one run.\n\n"
        "Do what each step says, with your real tools — a step is done when its "
        "tool call has run. Where a step says to remember something, write it "
        "with a real call, and record what you ACTUALLY found — never a "
        "placeholder, an example, or a description of what you would have found. "
        "Finding a value is not remembering it; the write is.\n\n"
        "Follow the steps as they gave them. If a step can't be done — the page "
        "doesn't have what you're looking for, or a value you need never showed "
        "up — it's okay to stop there. Tell them what you did find and which step "
        "stopped you, and they'll adjust the instructions. Don't take a different "
        "route to the goal, and never report a step as done that didn't happen.\n\n"
        "If the task also mentions a schedule or being told about changes, leave "
        "that part alone for now — it is set up in a later turn, after they ask "
        "for it. Your job this turn is the steps, nothing else.\n\n"
        "Then tell them what you did: each step and what it produced, including "
        "anything that failed or came back empty. Say what you now know how to "
        "do, and offer to set it up to run on its own.\n\n"
        "Don't set it up yourself. Offering is where this turn ends — they will "
        "tell you if they want it running.\n\n"
    )


def test_a_framed_round_names_its_routine_and_its_container():
    """The round's framing renders as the learn instruction's closing paragraph (#1868),
    verbatim — pinned so an edit is a visible diff.

    Both anchors render EXACTLY as they are stored, which is the whole point: a
    destination the model would otherwise have to invent a name for is now a name it
    copies, so "one job, one container" stops being a judgment anybody makes.  It says
    what the round IS — the routine being taught, the collection its results are kept in —
    and leaves what to do about it to the instruction above, which already says a step
    that remembers something is a real write.  Naming a tool here would key the sentence to
    one way of keeping a result, and a routine is an arbitrary sequence of tool calls.

    It sits INSIDE the instruction, between head and tail, so a framed learn turn is the
    unframed one plus one paragraph and nothing else moves."""
    framed = conversation_prompt(ConversationState.LEARN, _framing())

    assert framed == (
        Prompt.CONVERSATION_HEAD
        + Prompt.LEARN_INSTRUCTION
        + "The routine you are being taught this round is called `watch-rental-price`, "
        "and `watch-rental-price-harborkayak-example` is the collection set up to hold "
        "what it produces. Where a step says to remember something, that collection is "
        "where it goes — it is already there, so there is nothing to set up.\n\n"
        + Prompt.CONVERSATION_TAIL
    )


def test_an_unframed_round_composes_the_prompt_it_always_did():
    """A state with no framing composes the byte-identical prompt this function has always
    composed, so the framing is additive rather than a new shape every turn has to absorb
    — and a round whose entry draw failed reads exactly like a round from before the draw
    existed.  Apply is the one exception, and it is an exception by design: it has no
    unframed form at all (below)."""
    for state in ConversationState:
        if state is ConversationState.APPLY:
            continue
        assert conversation_prompt(state, None) == conversation_prompt(state)


def test_an_apply_turn_has_no_unframed_form():
    """There IS no unframed apply prompt (#1875): apply configures the collection its own
    round set up, so with no framing there is no container to name, no routine to state,
    and nothing to instruct — composing one anyway would mean falling back to a turn that
    binds the routine itself off values it cannot see.

    So this raises rather than composing something, and the caller fails the turn honestly
    before it ever gets here."""
    with pytest.raises(ValueError, match="no unframed form"):
        conversation_prompt(ConversationState.APPLY)


def test_an_apply_turn_enters_with_the_container_already_known():
    """The round renders INSIDE the apply instruction (#1875), verbatim — pinned so an
    edit is a visible diff.

    Learn and apply say DIFFERENT things about the same framing, which is why apply's is
    not the closing line learn gets: learn is being taught the routine, so its line says
    the container is already there and there is nothing to set up; apply is standing the
    routine up, so the container is not a note appended to its instruction — it is the
    subject of every sentence in it.

    Both anchors render EXACTLY as they are stored, so the collection the turn configures
    is a name it copies rather than one it works out."""
    framed = conversation_prompt(ConversationState.APPLY, _framing())

    assert framed == (
        Prompt.CONVERSATION_HEAD
        + "A collection has been set up for this task from what the user asked. It is "
        "named `watch-rental-price-harborkayak-example`, it runs the routine "
        "`watch-rental-price`, and it is already pointed at what they gave. Configure "
        "its schedule now, in one `collection_set` call on "
        "`watch-rental-price-harborkayak-example`: when it runs, when it stops if they "
        "gave an end, and whether to tell them — all from the user's own words.\n\n"
        "Configuring it is the whole turn — you are not carrying the routine out "
        "yourself. Once it is set up it runs itself on the schedule they just "
        "gave you, and its first run is the first thing they'll hear about.\n\n"
        "Then tell them what you set up and what will happen. Say it is running "
        "only if the call came back confirming it.\n\n" + Prompt.CONVERSATION_TAIL
    )


def test_only_learn_renders_the_round_as_a_closing_line():
    """The framing is CARRIED on every non-idle move — it is the round link a later turn
    reads — but only learn renders it as its own closing paragraph.

    Elicit is still asking for the task and request is still asking for a detail, so
    neither has anything to say about a container that may not exist yet; idle never
    carries one at all.  Apply is absent from the map for the opposite reason: it renders
    the round inside its own instruction rather than after it."""
    assert set(ROUND_LINES) == {ConversationState.LEARN}
    for state in ConversationState:
        if state in ROUND_LINES or state is ConversationState.APPLY:
            continue
        assert conversation_prompt(state, _framing()) == conversation_prompt(state), state


def test_apply_instruction_whole_render():
    """The whole instruction, verbatim — pinned so an edit is a visible diff.

    ONE coherent statement, not a boundary policed twice: what already exists and what is
    left to say, then the scope of the turn (configuring is all of it, the routine runs
    itself afterwards), then what to report and the one condition on claiming it is
    running.  It was patched by accretion once — a third paragraph restating the reporting
    boundary from another angle — which is the drift this literal makes visible.

    The opening paragraph is the round's own STATE (code-owner authored): a collection
    exists, this is its name, this is the routine it runs, and it is already pointed at
    what the user gave — so the three terms asked for are the only things the turn has to
    decide, and the container it configures is a name it copies.  Both anchors are
    placeholders because they are read off the round rather than worked out here.

    What it no longer carries is the end-date rule.  That belongs with the field it
    governs — ``expires_at``'s own description says when to set it and when to leave it
    out — so the instruction says what the turn IS and the field says how the field
    works, each in one place."""
    assert Prompt.APPLY_INSTRUCTION == (
        "A collection has been set up for this task from what the user asked. It is "
        "named `{container}`, it runs the routine `{skill}`, and it is already pointed "
        "at what they gave. Configure its schedule now, in one `collection_set` call on "
        "`{container}`: when it runs, when it stops if they gave an end, and whether to "
        "tell them — all from the user's own words.\n\n"
        "Configuring it is the whole turn — you are not carrying the routine out "
        "yourself. Once it is set up it runs itself on the schedule they just "
        "gave you, and its first run is the first thing they'll hear about.\n\n"
        "Then tell them what you set up and what will happen. Say it is running "
        "only if the call came back confirming it.\n\n"
    )


def test_request_instruction_whole_render():
    """The whole instruction, verbatim — pinned so an edit is a visible diff.

    Its own state and nothing else: they want something already known how to do, part of
    what it needs is unsaid, so the turn asks.  What it does NOT carry, deliberately, is
    elicit's "never ask about keywords, matching rules or selectors" clause — in elicit
    nothing is known and every such question is the assistant asking the user to do its
    job, while HERE the routine declares what it needs and one of those declared things
    may well be the phrase to look for on a page.  A blanket prohibition would forbid
    exactly the ask this state exists to make."""
    assert Prompt.REQUEST_INSTRUCTION == (
        "The user wants something done that you already know how to do, but they "
        "have not said everything it needs. Your job this turn is to ask them for "
        "the part that is missing.\n\n"
        "In ONE message, say in plain words what you would do, then ask for what "
        "is missing. Ask for that and nothing else.\n\n"
        "Ask in their own words, the way they would say it. Don't guess the "
        "missing part, don't use a value you happen to know, and don't do any of "
        "the task yet.\n\n"
    )


def test_a_request_turn_names_the_routine_and_what_is_still_missing():
    """The round's shortfall renders as the request instruction's closing paragraph
    (#1885), verbatim — pinned so an edit is a visible diff.

    Everything the ask has to work from is HERE: the routine, what it is for, each value
    the words already settled, and each missing detail with the registry's own line of what
    to supply.  So the reply is written at n=0 — no lookup, and nothing on it invented —
    and the already-settled list is what stops the turn asking twice for something the user
    has already said.

    It sits INSIDE the instruction, between head and tail, so a request turn with a
    shortfall is the plain one plus this paragraph and nothing else moves."""
    composed = conversation_prompt(ConversationState.REQUEST, None, _shortfall())

    assert composed == (
        Prompt.CONVERSATION_HEAD
        + Prompt.REQUEST_INSTRUCTION
        + "The routine for this is `watch-rental-price` — keep a rental page's current "
        "day rate up to date\n\n"
        "What they have already given you, which you must not ask for again:\n"
        "- keyword: the weekend rate\n\n"
        "What is still missing, which is what to ask them for:\n"
        "- url — the rental page to read\n\n"
        "Ask for the missing part by what it IS, using the plain description above — "
        "never by the short name it is listed under.\n\n" + Prompt.CONVERSATION_TAIL
    )


def test_a_request_turn_that_has_none_of_it_yet_says_so():
    """The empty case is a STATED line, not an absent section: an ask whose words settled
    nothing still renders the already-given list, saying there is nothing in it.

    Inferring "they have given me none of it" from a section that is not there is exactly
    the read a rendered state exists to remove — and it is the common shape, since a
    routine asking for one thing that the words did not supply settles nothing at all."""
    composed = conversation_prompt(
        ConversationState.REQUEST, None, _shortfall().model_copy(update={"bound": {}})
    )

    assert composed == (
        Prompt.CONVERSATION_HEAD
        + Prompt.REQUEST_INSTRUCTION
        + "The routine for this is `watch-rental-price` — keep a rental page's current "
        "day rate up to date\n\n"
        "What they have already given you, which you must not ask for again:\n"
        "- nothing yet — they have given you none of it\n\n"
        "What is still missing, which is what to ask them for:\n"
        "- url — the rental page to read\n\n"
        "Ask for the missing part by what it IS, using the plain description above — "
        "never by the short name it is listed under.\n\n" + Prompt.CONVERSATION_TAIL
    )


def test_a_missing_detail_with_no_description_renders_as_its_bare_name():
    """A parameter carrying no description renders as its name alone — no dash, no empty
    clause after one.

    A description is optional on the row (a labelling fallback leaves none), so an absent
    one is a real shape rather than empty text to render: a trailing dash would read as a
    description somebody wrote and left blank, which is the state presenting a fact that is
    not true."""
    bare = _shortfall().model_copy(
        update={"missing": (CandidateParameter(name="url", description=None),)}
    )
    composed = conversation_prompt(ConversationState.REQUEST, None, bare)

    assert "What is still missing, which is what to ask them for:\n- url\n\n" in composed, composed
    assert "- url —" not in composed


def test_a_request_turn_without_a_shortfall_composes_the_prompt_it_always_did():
    """A request turn the classifier parked on its own carries no shortfall — nothing ran
    the binder — so it composes the byte-identical prompt this function always composed.

    The rendered state is ADDITIVE, like the framing before it: a turn that has it is
    better instructed, and a turn that does not is instructed exactly as it was."""
    assert conversation_prompt(ConversationState.REQUEST, None, None) == conversation_prompt(
        ConversationState.REQUEST
    )
    assert Prompt.ROUND_SHORTFALL_LINE.split("{")[0] not in conversation_prompt(
        ConversationState.REQUEST
    )


def test_only_request_renders_the_round_s_shortfall():
    """The shortfall is request's alone: no other state's prompt moves when one is passed.

    Learn reads the FRAMING and request the SHORTFALL because those are the two halves of a
    round's entry and each state has exactly one of them — so handing a shortfall to a
    state that does not read it changes nothing, which is what keeps the two paragraphs
    from ever both appearing."""
    for state in ConversationState:
        if state is ConversationState.REQUEST:
            continue
        framing = _framing() if state is ConversationState.APPLY else None
        assert conversation_prompt(state, framing, _shortfall()) == conversation_prompt(
            state, framing
        ), state


def test_the_applied_configuration_narration_whole_render():
    """The frame a turn that CONFIGURED the round's routine is handed, verbatim (#1869) —
    pinned so an edit is a visible diff, like every other model-facing surface.

    It is the skill-learned frame's sibling one step further along: that one carries what
    a run LEARNED, this one what it SET RUNNING, and this one exists because the turn no
    longer supplies the routine or the values it is pointed at.  So its last sentence is
    the honesty clause — say only what is above — since the record is now the only place
    those facts are, and a reply going beyond it would be stating a configuration the turn
    never made."""
    assert Prompt.CONFIGURATION_APPLIED_NARRATION == (
        "The routine is now set up, and here is exactly what was configured:\n\n"
        "{configuration}\n\n"
        "Reply to the user now. Tell them in your own words what is running: what it "
        "watches, how often it runs, when it stops if it stops, and whether they will "
        "hear from it. Say only what is above — anything not there was not set."
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
