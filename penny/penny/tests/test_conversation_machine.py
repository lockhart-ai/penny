"""The conversation state machine's classifier machinery (#1706, beat 0).

The machine's structural invariants are pinned as data assertions (the edge
table: break-out from every classifying state, no learn edge out of idle, no
out-edges at all from apply) and pure-function contracts (fail → stay in
``next_state``; the apply edge withheld when no skill candidates exist).  The
classifier itself — micro-context customer #3 — is pinned by whole-render
literals of everything the model sees (system prompt, the rendered slice, the
per-edge state meanings) and by the draw mechanics: membership-validated tag
parse, one reroll on a contract violation, poison discard-and-reroll, honest
enumerated failures.

Deterministic mock model responses throughout — the live-model contract is the
eval suite's job (beat 1 onward), not this file's.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.conversation_machine import (
    OUT_EDGES,
    ConversationState,
    MachineSnapshot,
    StateClassifier,
    next_state,
    presented_edges,
    render_classifier_content,
)
from penny.llm.models import LlmMessage, LlmResponse
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
_SKILL_LINE = "watch a listing price for changes — checks a page and records the current price"

_IDLE_SNAPSHOT = MachineSnapshot(state=ConversationState.IDLE)
_ELICIT_SNAPSHOT = MachineSnapshot(
    state=ConversationState.ELICIT,
    penny_last_turn=_TEACH_QUESTION,
    task_anchor=_ASK,
)


def _responds(content: str) -> MockLlmClient:
    """A mock model client whose every chat returns ``content``."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


def _classifier(model: MockLlmClient) -> StateClassifier:
    return StateClassifier(cast(Any, model))


# ── The edge table: structural invariants as data assertions ──────────────────


def test_edge_table_invariants():
    """Every state that classifies carries the break-out edge → idle; learn is
    unreachable from idle (steps can only arrive after an ask); apply has NO
    out-edges — its reset is structural, never a classifier call."""
    for state, edges in OUT_EDGES.items():
        if edges:
            assert ConversationState.IDLE in edges, f"{state} lacks the break-out edge"
    assert ConversationState.LEARN not in OUT_EDGES[ConversationState.IDLE]
    assert OUT_EDGES[ConversationState.APPLY] == ()


def test_presented_edges_withholds_apply_without_candidates():
    """The apply edge is offered only when the snapshot carries skill
    candidates — an empty registry never renders an apply option (the
    structural false-apply guard)."""
    assert presented_edges(_IDLE_SNAPSHOT) == (
        ConversationState.IDLE,
        ConversationState.ELICIT,
    )
    with_skills = MachineSnapshot(state=ConversationState.IDLE, skill_candidates=[_SKILL_LINE])
    assert presented_edges(with_skills) == (
        ConversationState.IDLE,
        ConversationState.APPLY,
        ConversationState.ELICIT,
    )


# ── Whole-render literals: everything the classifier model sees ───────────────


def test_system_prompt_whole_render():
    """Whole-render literal of the dispatch contract: one tagged STATE: line,
    the name copied exactly from the listed states, nothing else."""
    assert STATE_CLASSIFIER_SYSTEM_PROMPT == (
        "You are a dispatch step. You are given a small slice of a conversation "
        "between a user and their assistant — the assistant's last message, the task "
        "being worked on (when there is one), and the user's newest message — plus a "
        "closed list of states, each with a one-line meaning. Decide which ONE state "
        "the user's newest message puts the conversation in. Respond with exactly "
        "one line:\n"
        "STATE: <name>\n"
        "The name must be one of the listed states, copied exactly. Judge only from "
        "what the messages say, and write nothing else — no preamble, no "
        "explanation, no restating the messages."
    )


def test_render_idle_slice_whole():
    """The idle render, whole: no last turn yet renders the (none) placeholder,
    the skills section renders an explicit (none) — the elicit meaning's "no
    known skill covers it" must be a READ, never an inference from a missing
    section — and the offered states are idle + elicit only (apply withheld
    with no candidates)."""
    assert render_classifier_content(_IDLE_SNAPSHOT, _ASK) == (
        "The assistant's last message: (none)\n"
        "Known skills: (none)\n"
        "The user's newest message: hey can you keep an eye on the harbor ferry "
        "timetable for me?\n"
        "\n"
        "States:\n"
        "- idle: ordinary conversation — chat, a question, or a passing mention; "
        "they are not asking for a task to be set up\n"
        "- elicit: they are asking for a task or routine to be done and no known "
        "skill covers it — the assistant would need to be taught how"
    )


def test_render_parked_elicit_slice_whole():
    """The parked-elicit render, whole: the assistant's teach question and the
    instigating ask are both present (a reply is only classifiable against what
    it answers), and the union is the elicit out-edges with their per-edge
    meanings — including the break-out edge."""
    assert render_classifier_content(_ELICIT_SNAPSHOT, _STEPS) == (
        "The assistant's last message: I don't know how to do that yet — can you "
        "teach me? What should I read, look for, and remember?\n"
        "The task being worked on: hey can you keep an eye on the harbor ferry "
        "timetable for me?\n"
        "Known skills: (none)\n"
        "The user's newest message: sure — read harborferries.example/timetable "
        "and remember the first morning departure\n"
        "\n"
        "States:\n"
        "- learn: their message gives the steps — it tells the assistant how to do "
        "the task it asked to be taught\n"
        "- elicit: still working out the task — the assistant's question is not "
        "answered yet\n"
        "- idle: they changed the topic or called the task off"
    )


def test_render_idle_with_candidates_whole():
    """The idle render with a ranked skill candidate, whole: the Known skills
    section appears and the apply edge joins the union."""
    with_skills = MachineSnapshot(state=ConversationState.IDLE, skill_candidates=[_SKILL_LINE])
    assert render_classifier_content(with_skills, "what's the ferry price at today?") == (
        "The assistant's last message: (none)\n"
        "Known skills:\n"
        "- watch a listing price for changes — checks a page and records the "
        "current price\n"
        "The user's newest message: what's the ferry price at today?\n"
        "\n"
        "States:\n"
        "- idle: ordinary conversation — chat, a question, or a passing mention; "
        "they are not asking for a task to be set up\n"
        "- apply: they are asking for something one of the known skills already "
        "covers\n"
        "- elicit: they are asking for a task or routine to be done and no known "
        "skill covers it — the assistant would need to be taught how"
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
    assert request["messages"][1]["content"] == (
        "Instruction: Pick the one listed state the user's newest message puts "
        "the conversation in.\n"
        "\n"
        "Content:\n" + render_classifier_content(_IDLE_SNAPSHOT, _ASK)
    )


@pytest.mark.asyncio
async def test_classify_out_of_union_draw_is_rerolled_then_stays():
    """A drawn state OUTSIDE the offered union is a contract violation exactly
    like an untagged draw: one reroll of the unchanged context, then an honest
    INVALID the machine holds its state on — learn is not an idle out-edge, so
    a flaky draw can never conjure a teach round from ordinary chat."""
    model = _responds("STATE: learn")
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
async def test_classify_reroll_can_recover():
    """The one contract-violation reroll re-draws on the unchanged context — a
    valid second draw decides the transition."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="hmm, let me think" if count == 1 else "STATE: learn",
            )
        )
    )
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
