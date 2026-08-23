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

The mechanism is current, and is checked on EVERY draw rather than only on a
call-less one: ``Agent._unusable_output_condition`` treats ``TOOL_CALL_LEAK`` as a
transport artifact (alongside ``DEGENERATE_OUTPUT``), so it does not ride an
agent's declared ``invalid_draw_conditions`` and applies to a collector cycle
exactly as it does to a chat turn.  This scorer reads the reply through the SAME
``has_leaked_harmony_envelope`` predicate the guard rejects it with.

Report-only (``min_pass_rate=None``), the canonical convention — stated rather
than inherited from ``chat_eval``'s 0.75 default.  The deterministic mechanism
(detect the leaked envelope on the raw output, discard, re-roll on unchanged
context, abort if it persists) is pinned in
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
from penny.tests.eval.fixtures import TOPIC_PAGES
from penny.text_validity import has_leaked_harmony_envelope

pytestmark = pytest.mark.eval

# Enough letters that the reply is something a person could have received, rather than a
# fragment the loop finalized.  A floor, not a quality bar — what the case is about is
# the envelope never arriving, and a bare "ok" would leave that unanswered either way.
_SUBSTANTIVE_LETTERS = 15

# A whole Harmony tool-call envelope the injector emits as plain text after the
# model's first real tool call — the exact leak shape seen on non-Ollama gpt-oss
# backends (generic ``browse`` tool, no real-world specifics).
_HARMONY_LEAK = "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"


def _score_recovered(db: Database, before: set[str], reply: str) -> list[Check]:
    """Graded: the forced Harmony-envelope leak reached NO delivered message, and the reply
    the user got is substantive prose — the model re-drew into a real answer rather than the
    loop finalizing the leaked envelope.

    The first check reads the DELIVERED messages off the messagelog rather than the last
    reply the runner happened to hold: what the contract forbids is raw control tokens
    reaching the user at all, and a turn that delivers two messages would otherwise be
    scored on one of them.  The 'forced bail fired — contract exercised' guard is PREPENDED
    by ``chat_eval``'s graded path (#1697) — so a run that never triggered the leak can't
    pass on a normal answer — and this scorer owns only the recovery outcome."""
    delivered = outgoing_replies(db)
    leaked = [message for message in delivered if has_leaked_harmony_envelope(message)]
    letters = sum(1 for character in reply if character.isalpha())
    gave_up = gave_up_mid_run(db)
    return [
        Check(
            "nothing delivered to the user carried a raw Harmony envelope",
            not leaked,
            kind="reply",
            rationale=None
            if not leaked
            else (
                f"{len(leaked)} of {len(delivered)} delivered message(s) carried raw control "
                f"tokens — the leak reached the user: {leaked[0][:120]!r}"
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
                f"real answer would: {reply[:120]!r}"
            ),
        ),
        Check(
            "answered rather than apologising its way out",
            not gave_up,
            scored=False,
            kind="proc",
            rationale=(
                "the reply is a defeatist give-up — the re-roll runs on the unchanged "
                "context and a clean draw was available"
            )
            if gave_up
            else f"calls made: {tool_call_sequence(db) or 'none'}",
        ),
    ]


async def test_harmony_envelope_leak_is_caught_and_recovers(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-harmony-envelope-leak-recovery",
        family="chat-recovery",
        message="what's the deepest lake in the world?",
        browse=list(TOPIC_PAGES),
        wrap_client=lambda real: _InjectTextBail(real, _HARMONY_LEAK),
        score=_score_recovered,
        min_pass_rate=None,
    )
