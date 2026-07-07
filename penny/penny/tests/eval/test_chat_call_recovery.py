"""Chat call-as-text recovery contract — when a chat reply is really a tool call
emitted as a JSON text object (gpt-oss's Harmony call-as-text fallback), the loop
must NOT send that JSON blob to the user; ``CallAsTextValidator`` nudges and the
live model recovers to a real reply.

Production failure this pins (narration-design probe, July 2026): on the
loop-stressed give-up path — a fruitless search the model keeps rewording — gpt-oss
emits a well-formed browse call as *text content* instead of routing it through the
tool channel. Chat replies inline via a text turn and had no run-shape guard, so
that raw ``{"queries": [...], "reasoning": "..."}`` blob was delivered to the user
verbatim (observed ~50% on retry-heavy searches, even on the stock prompt). The
collector already guards the sibling shape (``DoneJsonBailValidator``); this adds
the chat equivalent.

The slip is stochastic, so we FORCE one call-as-text response right after the
model's first real tool call (``_InjectTextBail`` with a JSON call as the bail text)
and let the REAL model drive the recovery through the production nudge. The contract
is STRUCTURAL, never wording:

  PASS = the reply is NOT a serialized tool call (the JSON never reached the user)
         and it's substantive prose — the model either re-issued the real call and
         answered, or gave an honest "couldn't find it".

The deterministic mechanism (detect call-as-text on the text branch, nudge, don't
finalize) is pinned in ``tests/agents/test_agentic_loop.py``; this owns the live
model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, _InjectTextBail
from penny.tests.eval.fixtures import TOPIC_PAGES
from penny.validation.response_validators import is_call_as_text_bail

pytestmark = pytest.mark.eval

# A well-formed browse call the injector emits as plain text after the model's first
# real tool call — the exact Harmony call-as-text shape seen in production.
_CALL_AS_TEXT = (
    '{"queries": ["deepest lake in the world"], '
    '"reasoning": "Look up which lake is the deepest and read the details."}'
)


def _score_recovered(db: Database, before: set[str], reply: str) -> list[str]:
    """Pass iff the forced call-as-text bail did NOT reach the user as raw JSON and
    the reply is substantive prose (the model recovered into a real answer or an
    honest dead-end, rather than the loop finalizing the JSON blob)."""
    fails: list[str] = []
    if is_call_as_text_bail(reply):
        fails.append(
            f"reply is a serialized tool call, not prose — bail reached the user: {reply[:120]!r}"
        )
    if sum(1 for character in reply if character.isalpha()) < 15:
        fails.append(f"reply is not substantive prose: {reply[:120]!r}")
    return fails


async def test_call_as_text_is_caught_and_recovers(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-call-as-text-recovery",
        message="what's the deepest lake in the world?",
        browse=list(TOPIC_PAGES),
        wrap_client=lambda real: _InjectTextBail(real, _CALL_AS_TEXT),
        score=_score_recovered,
        min_pass_rate=0.75,
    )
