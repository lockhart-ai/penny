"""Chat call-as-text recovery: the turn is completed after her own call comes back as text.

Ported to the cohort structure under #2009; the contract is `docs/eval-case-design.md`.

Mid-turn, right after the model's first real tool call, the harness hands it back a
well-formed browse call emitted as plain TEXT — gpt-oss's Harmony call-as-text fallback,
observed at ~50% on retry-heavy searches in production, where the raw
``{"queries": [...], "reasoning": "..."}`` blob was delivered to the user verbatim.  The
loop must reject that draw (#1839) and the LIVE model must recover: the question gets
answered, out of the page, and nothing unusable reaches the user.

  * THE FAULT IS FIXED, the WORDING varies.  Five wordings of one question, pooled into
    one variance score.  A fault that varied would be a different case.
  * THE INJECTION IS HARNESS MACHINERY, not model output.  A sample the sabotage never
    fired on ran an unbroken turn and exercised no recovery, so it leaves the cohort as a
    named exclusion rather than counting as a behavioural failure.  What that costs is
    #2018: the sabotage fires after the model's first tool call, so a turn that answered
    without reading anything leaves here too — and that turn is one this case's provenance
    claim would have failed.
  * THE FAULT IS AIMED AT THE CHAT TURN.  An agent's model client is shared with every
    microcontext built from it, so the sabotage is confined to the chat agent by name —
    without that, the bail lands in `browse-extract` and the turn under test runs clean.
  * the tool calls MEASURED rather than asserted — many routes reach one end state, and a
    recovery route is exactly where they differ.

REPORT-ONLY (``min_pass_rate=None``): the floors and ceilings this run proposes are the
code owner's to accept once the numbers have been read.  Nothing here is the user's: the
ask is a general-knowledge question and the page is public reference content served from
an ``example`` domain, because the repo is public.

The deterministic mechanism (detect call-as-text, discard the draw, re-roll on the
unchanged context) is pinned in ``tests/agents/test_agentic_loop.py``; this owns the live
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

_CASE_ID = "chat-call-as-text-recovery"

# The family every chat recovery case reports under — what Penny does once something has
# already gone wrong.  A second spelling would split one bucket across two headings.
_FAMILY = "chat-recovery"

# A well-formed browse call the injector emits as plain text after the model's first real
# tool call — the exact Harmony call-as-text shape seen in production.  Its arguments are
# the framework-injected ``reasoning`` field plus a query, which is what makes it a call
# rather than prose.
_CALL_AS_TEXT = (
    '{"queries": ["deepest lake in the world"], '
    '"reasoning": "Look up which lake is the deepest and read the details."}'
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_call_as_text_is_caught_and_the_turn_still_completes(
    chat_eval: ChatEval, model: str
) -> None:
    """A tool call that came back as text is thrown away, and the question is still
    answered from the page rather than from the model's own head."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        model=model,
        world=DEEPEST_LAKE,
        ask=DEEPEST_LAKE_ASK,
        also_phrased=DEEPEST_LAKE_PHRASINGS,
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectTextBail(
            real, _CALL_AS_TEXT, target_agent=PennyConstants.CHAT_AGENT_NAME
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
