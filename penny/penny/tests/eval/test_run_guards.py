"""Runtime-guard contracts — the collector refuses a coherent-but-wrong tool call
with an ERROR TOOL RESPONSE and the live model recovers.

Two unhandled bail shapes pulled from production promptlogs, each forced
deterministically (the slip is stochastic, so we inject it once) and then driven
through the REAL collector loop so the production guard's error tool response has
to carry the recovery on the live model:

  premature-done   — the cycle's FIRST tool call is done() (opened with "no new
                     matches" before reading anything).  The guard refuses it; the
                     model must read its inputs, then do the work.
  half-formed-send — a send_message goes out with no real content
                     ("Hi there! ......???").  The send gate refuses it; the model
                     must resend a COMPLETE message.

These differ from the text-step nudge (``test`` ``nudge_eval``): there the model
emitted plain text (no tool call) and we nudge via a user turn; here the model
made a coherent tool call, so the correction comes back in that call's tool-result
field.  The deterministic mechanism (refuse + don't stop the loop) is pinned in
``tests/agents/test_agentic_loop.py`` / ``tests/tools/test_send_message.py``; these
own the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import half_formed_send_reason
from penny.tests.eval.conftest import (
    _InjectDoneBail,
    _InjectSendBail,
    collection_entries,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    HALF_FORMED_SEND,
    SEND_DIGEST,
    SEND_DIGEST_EXTRACTION_PROMPT,
    WEEKLY_DIGEST,
    WEEKLY_DIGEST_EXTRACTION_PROMPT,
    WEEKLY_DIGEST_MESSAGES,
)

pytestmark = pytest.mark.eval

_INCOMING = PennyConstants.MessageDirection.INCOMING


def _seed_digest_with_messages(db: Database) -> None:
    """Numbered summary collector + clearly-summarizable seeded messages.

    The messages give the cycle real work, so a recovered run MUST write a summary
    entry — a no-write after the forced first-move done() is a failure to recover."""
    db.memories.create_collection(
        WEEKLY_DIGEST.name,
        WEEKLY_DIGEST.description,
        extraction_prompt=WEEKLY_DIGEST_EXTRACTION_PROMPT,
        collector_interval_seconds=1200,
    )
    for message in WEEKLY_DIGEST_MESSAGES:
        db.messages.log_message(_INCOMING, "user", message)


def _seed_send_digest(db: Database) -> None:
    """A read-then-send collector + seeded messages — the send step the injector
    hijacks with a half-formed body."""
    db.memories.create_collection(
        SEND_DIGEST.name,
        SEND_DIGEST.description,
        extraction_prompt=SEND_DIGEST_EXTRACTION_PROMPT,
        collector_interval_seconds=1200,
    )
    for message in WEEKLY_DIGEST_MESSAGES:
        db.messages.log_message(_INCOMING, "user", message)


def _score_recovered_with_work(db: Database, sent: list[str]) -> list[str]:
    """Pass iff the cycle recovered from the forced first-move done(): it read its
    inputs and wrote the summary entry the seeded messages clearly warrant."""
    if collection_entries(db, WEEKLY_DIGEST.name):
        return []
    read = tool_was_called(db, "log_read")
    wrote = tool_was_called(db, "collection_write")
    return [
        "did not recover after the refused first-move done() — no summary written "
        f"(log_read={read}, collection_write={wrote})"
    ]


def _score_resent_complete_message(db: Database, sent: list[str]) -> list[str]:
    """Pass iff a COMPLETE message was sent and NO half-formed one slipped through.

    Pre-fix the forced ``"Hi there! ......???"`` was enqueued (it shows up in
    ``sent``); post-fix the gate refuses it and the model resends a real message."""
    half_formed = [body for body in sent if half_formed_send_reason(body) is not None]
    if half_formed:
        return [f"a half-formed message was sent (gate let it through): {half_formed[0]!r}"]
    if not sent:
        return ["no message sent after the gate refused the half-formed body — no recovery"]
    return []


async def test_premature_done_is_refused_and_recovers(guard_recovery_eval) -> None:
    """A first-move done() is refused via an error tool response; the live model
    then reads its inputs and writes the summary."""
    await guard_recovery_eval(
        case_id="guard-premature-done",
        collection=WEEKLY_DIGEST.name,
        seed=_seed_digest_with_messages,
        wrap_client=lambda real: _InjectDoneBail(real),
        score=_score_recovered_with_work,
        min_pass_rate=0.75,
    )


async def test_half_formed_send_is_refused_and_recovers(guard_recovery_eval) -> None:
    """A half-formed send is refused via an error tool response; the live model
    then resends a complete message."""
    await guard_recovery_eval(
        case_id="guard-half-formed-send",
        collection=SEND_DIGEST.name,
        seed=_seed_send_digest,
        wrap_client=lambda real: _InjectSendBail(real, HALF_FORMED_SEND),
        score=_score_resent_complete_message,
        min_pass_rate=0.75,
    )
