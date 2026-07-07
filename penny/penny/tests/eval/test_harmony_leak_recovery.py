"""Harmony-envelope-leak recovery contract — when a chat reply is really a tool
call that the backend failed to parse, so the whole Harmony envelope leaked into
``message.content`` as literal control-token text, the loop must NOT deliver that
raw envelope to the user; the agent-loop reroll guard discards it and the live
model recovers on the unchanged context to a real reply.

Production failure this pins: on some remote OpenAI-compatible backends serving
gpt-oss (non-Ollama runners), the Harmony tool-call envelope leaks into the text
channel instead of being parsed into ``tool_calls`` — e.g.
``<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>``
with ``tool_calls`` empty. Chat replies inline via a text turn, so that raw
envelope string would be finalized as the reply and delivered verbatim. Stock
Ollama parses the envelope, so this is defensive robustness for leaky backends,
not a fix for our own runner. The decision (issue #1501) is REJECT + REROLL — do
not reconstruct the call from the envelope grammar — reusing the discard-and-reroll
machinery the punctuation-collapse degeneracy guard already owns.

The leak is intermittent (the same runner parses correctly on other turns), so we
FORCE one leaked-envelope response right after the model's first real tool call
(``_InjectTextBail`` with the envelope as the bail text) and let the REAL model
drive the recovery through the production reroll. The contract is STRUCTURAL, never
wording:

  PASS = the reply carries NO raw Harmony tokens (the envelope never reached the
         user) and it's substantive prose — the model re-drew cleanly and answered.

The deterministic mechanism (detect the leaked envelope on the raw output, discard,
re-roll on unchanged context, abort if it persists) is pinned in
``tests/agents/test_agentic_loop.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, _InjectTextBail
from penny.tests.eval.fixtures import TOPIC_PAGES
from penny.text_validity import has_leaked_harmony_envelope

pytestmark = pytest.mark.eval

# A whole Harmony tool-call envelope the injector emits as plain text after the
# model's first real tool call — the exact leak shape seen on non-Ollama gpt-oss
# backends (generic ``browse`` tool, no real-world specifics).
_HARMONY_LEAK = "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"


def _score_recovered(db: Database, before: set[str], reply: str) -> list[str]:
    """Pass iff the forced Harmony-envelope leak did NOT reach the user as raw
    control-token text and the reply is substantive prose (the model re-drew into a
    real answer rather than the loop finalizing the leaked envelope)."""
    fails: list[str] = []
    if has_leaked_harmony_envelope(reply):
        fails.append(
            f"reply carries a raw Harmony envelope — leak reached the user: {reply[:120]!r}"
        )
    if sum(1 for character in reply if character.isalpha()) < 15:
        fails.append(f"reply is not substantive prose: {reply[:120]!r}")
    return fails


async def test_harmony_envelope_leak_is_caught_and_recovers(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-harmony-envelope-leak-recovery",
        message="what's the deepest lake in the world?",
        browse=list(TOPIC_PAGES),
        wrap_client=lambda real: _InjectTextBail(real, _HARMONY_LEAK),
        score=_score_recovered,
        min_pass_rate=0.75,
    )
