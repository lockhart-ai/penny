"""Chat call-as-text recovery contract — when a chat reply is really a tool call
emitted as a JSON text object (gpt-oss's Harmony call-as-text fallback), the loop
must NOT send that JSON blob to the user: it is an INVALID DRAW, discarded and
re-rolled on the unchanged context (#1839), and the live model recovers to a real
reply.

Production failure this pins (narration-design probe, July 2026): on the
loop-stressed give-up path — a fruitless search the model keeps rewording — gpt-oss
emits a well-formed browse call as *text content* instead of routing it through the
tool channel. Chat replies inline via a text turn and had no run-shape guard, so
that raw ``{"queries": [...], "reasoning": "..."}`` blob was delivered to the user
verbatim (observed ~50% on retry-heavy searches, even on the stock prompt). The
collector already rejects the sibling shape; this adds the chat equivalent.

The slip is stochastic, so we FORCE one call-as-text response right after the
model's first real tool call (``_InjectTextBail`` with a JSON call as the bail text)
and let the REAL model drive the redraw. The contract is STRUCTURAL, never wording:

  PASS = the reply is NOT a serialized tool call (the JSON never reached the user)
         and it's substantive prose — the model either re-issued the real call and
         answered, or gave an honest "couldn't find it".

The mechanism is current: ``ChatAgent.invalid_draw_conditions`` declares
``(CALL_AS_TEXT, is_call_as_text_bail)`` as its first entry, and
``Agent._unusable_output_condition`` consults it on any draw carrying no tool
calls — so this scorer reads the reply through the SAME predicate production
rejects it with.

Report-only (``min_pass_rate=None``), the canonical convention — stated rather
than inherited from ``chat_eval``'s 0.75 default.  The deterministic mechanism
(detect call-as-text, discard the draw, re-roll) is pinned in
``tests/agents/test_agentic_loop.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    _InjectTextBail,
    gave_up_mid_run,
    outgoing_replies,
    tool_call_sequence,
)
from penny.tests.eval.utils.fixtures import TOPIC_PAGES
from penny.text_validity import is_call_as_text_bail

pytestmark = pytest.mark.eval

# Enough letters that the reply is something a person could have received, rather than a
# fragment the loop finalized.  A floor, not a quality bar — what the case is about is
# the JSON never arriving, and a bare "ok" would leave that unanswered either way.
_SUBSTANTIVE_LETTERS = 15

# A well-formed browse call the injector emits as plain text after the model's first
# real tool call — the exact Harmony call-as-text shape seen in production.
_CALL_AS_TEXT = (
    '{"queries": ["deepest lake in the world"], '
    '"reasoning": "Look up which lake is the deepest and read the details."}'
)


def _score_recovered(db: Database, before: set[str], reply: str) -> list[Check]:
    """Graded: the forced call-as-text bail reached NO delivered message, and the reply the
    user got is substantive prose — the model recovered into a real answer or an honest
    dead-end rather than the loop finalizing the JSON blob.

    The first check reads the DELIVERED messages off the messagelog rather than the last
    reply the runner happened to hold: what the contract forbids is the blob reaching the
    user at all, and a turn that delivers two messages would otherwise be scored on one of
    them.  The 'forced bail fired — contract exercised' guard is PREPENDED by ``chat_eval``'s
    graded path (#1697) — so a run that never triggered the bail can't pass on a normal
    answer — and this scorer owns only the recovery outcome."""
    delivered = outgoing_replies(db)
    leaked = [message for message in delivered if is_call_as_text_bail(message)]
    letters = sum(1 for character in reply if character.isalpha())
    gave_up = gave_up_mid_run(db)
    return [
        Check(
            "nothing delivered to the user was a serialized tool call",
            not leaked,
            kind="reply",
            rationale=None
            if not leaked
            else (
                f"{len(leaked)} of {len(delivered)} delivered message(s) were a serialized "
                f"call — the discarded draw reached the user: {leaked[0][:120]!r}"
            ),
        ),
        Check(
            "the reply reads as something a person would receive",
            letters >= _SUBSTANTIVE_LETTERS,
            kind="reply",
            rationale=None
            if letters >= _SUBSTANTIVE_LETTERS
            else (
                f"the reply carries {letters} letters, under the {_SUBSTANTIVE_LETTERS} a "
                f"real answer or an honest dead-end would: {reply[:120]!r}"
            ),
        ),
        Check(
            "answered rather than apologising its way out",
            not gave_up,
            scored=False,
            kind="proc",
            rationale=(
                "the reply is a defeatist give-up — a clean redraw was available and the "
                "model stopped instead"
            )
            if gave_up
            else f"calls made: {tool_call_sequence(db) or 'none'}",
        ),
    ]


async def test_call_as_text_is_caught_and_recovers(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-call-as-text-recovery",
        family="chat-recovery",
        message="what's the deepest lake in the world?",
        browse=list(TOPIC_PAGES),
        wrap_client=lambda real: _InjectTextBail(real, _CALL_AS_TEXT),
        score=_score_recovered,
        min_pass_rate=None,
    )
