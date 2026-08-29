"""Envelope leak recovery: the turn is answered cleanly after a raw envelope leaks.

Ported to the cohort structure under #2009; the contract is `docs/eval-case-design.md`.

Mid-turn, right after the model's first real tool call, the harness hands it back a whole
Harmony tool-call envelope as literal control-token text — what some remote
OpenAI-compatible backends serving gpt-oss do instead of parsing the call into
``tool_calls``, leaving a chat text turn to finalise the raw envelope as the reply.  The
decision (#1501) is REJECT + REROLL, never reconstruct the call from the envelope grammar,
and the LIVE model must recover on the unchanged context.

Sibling of ``test_chat_call_recovery`` — same world, same question, same end state — and
deliberately a SEPARATE case: the fault is different, and the two are caught by different
machinery.  A leaked envelope is a TRANSPORT artifact checked on every draw whatever its
shape, where call-as-text is checked only on a draw carrying no tool calls, so a cohort
pooling them would average two mechanisms into one score.

  * THE FAULT IS FIXED, the WORDING varies — five wordings of one question, pooled.
  * THE INJECTION IS HARNESS MACHINERY: a sample the leak never fired on ran an unbroken
    turn and leaves the cohort as a named exclusion, never as a behavioural failure — at
    the cost #2018 records, since the leak fires after the first tool call and a turn that
    read nothing leaves here rather than failing the provenance claim.
  * THE FAULT IS AIMED AT THE CHAT TURN.  An agent's model client is shared with every
    microcontext built from it, so the sabotage is confined to the chat agent by name —
    without that, the bail lands in `browse-extract` and the turn under test runs clean.
  * the tool calls MEASURED rather than asserted.

REPORT-ONLY (``min_pass_rate=None``).  Nothing here is the user's: the ask is a
general-knowledge question and the page is public reference content served from an
``example`` domain, because the repo is public.

The deterministic mechanism (detect the leak on the raw output, discard, re-roll, abort if
it persists) is pinned in ``tests/agents/test_agentic_loop.py``; this owns the live
model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, _InjectTextBail
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.worlds import (
    DEEPEST_LAKE,
    DEEPEST_LAKE_ASK,
    DEEPEST_LAKE_PHRASINGS,
)

pytestmark = pytest.mark.eval

_CASE_ID = "chat-harmony-envelope-leak-recovery"

# The family every chat recovery case reports under — what Penny does once something has
# already gone wrong.  A second spelling would split one bucket across two headings.
_FAMILY = "chat-recovery"

# A whole Harmony tool-call envelope the injector emits as plain text after the model's
# first real tool call — the exact leak shape seen on non-Ollama gpt-oss backends, naming
# a generic tool so the fixture carries no real-world specifics.
_HARMONY_LEAK = "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_leaked_envelope_is_caught_and_the_turn_is_answered_cleanly(
    chat_eval: ChatEval, model: str
) -> None:
    """A leaked call envelope is thrown away, and the question is still answered from the
    page rather than from the model's own head."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        model=model,
        world=DEEPEST_LAKE,
        ask=DEEPEST_LAKE_ASK,
        also_phrased=DEEPEST_LAKE_PHRASINGS,
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectTextBail(
            real, _HARMONY_LEAK, target_agent=PennyConstants.CHAT_AGENT_NAME
        ),
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.assert_no_delivered_message_is_an_unusable_draw()
    cohort.assert_every_delivered_message_is_whole()
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
