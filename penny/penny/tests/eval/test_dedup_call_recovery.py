"""Duplicate-CALL recovery contract — when the agent-loop dedup guard rejects a
byte-identical repeat, the reworked message must move the model ON without it
over-generalizing "no repeated calls" and suppressing the work it still owes.

Production failure this pins (July 2026 tool-failure audit): the terse
``"You already made this exact tool call. Try a different query or tool."``
rejection moved the model on ~83% of the time, but the runs containing it failed
at ~8x the baseline rate — traces show the model concluding the policy forbids
repeated calls and then dropping legitimate follow-up work (a verify re-read after
a write) for the rest of the run.  The message now states the why-now (this exact
call already ran; its result is above) AND the legitimate path (reuse that result;
this flags only the identical repeat, not reusing a tool at all).

The slip is a model DECISION on a visible tool result, but a natural cycle only
rarely repeats an exact call, so we force ONE byte-identical repeat of the model's
first tool call (``_InjectDuplicateCall``) and let the REAL model drive the recovery
off the production rejection.  The contract is STRUCTURAL, never wording:

  PASS = the cycle RECOVERED — it reused the earlier read and still wrote the
         summary the seeded messages clearly warrant — rather than freezing after
         the rejection (the over-generalization) or spiraling to the step ceiling.

The guard blocks a byte-identical repeat for the whole run, so this measures the
real harm — owed follow-up work being suppressed — via the write completing, not by
forcing a literal re-read (which the unchanged guard would itself refuse).  The
deterministic mechanism (reject in place, don't stop the loop) is pinned in
``tests/agents/test_agentic_loop.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import Inclusion, RecallMode
from penny.tests.eval.conftest import (
    _InjectDuplicateCall,
    collection_entries,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    WEEKLY_DIGEST,
    WEEKLY_DIGEST_EXTRACTION_PROMPT,
    WEEKLY_DIGEST_INTENT,
    WEEKLY_DIGEST_MESSAGES,
)

pytestmark = pytest.mark.eval

_INCOMING = PennyConstants.MessageDirection.INCOMING


def _seed_digest_with_messages(db: Database) -> None:
    """Numbered summary collector + clearly-summarizable seeded messages.

    The messages give the cycle real work, so a recovered run MUST write a summary
    entry — a no-write after the forced duplicate read is a failure to recover (the
    model froze on the rejection instead of reusing the read result)."""
    db.memories.create_collection(
        WEEKLY_DIGEST.name,
        WEEKLY_DIGEST.description,
        Inclusion(WEEKLY_DIGEST.inclusion),
        RecallMode.RECENT,
        extraction_prompt=WEEKLY_DIGEST_EXTRACTION_PROMPT,
        intent=WEEKLY_DIGEST_INTENT,
        collector_interval_seconds=1200,
    )
    for message in WEEKLY_DIGEST_MESSAGES:
        db.messages.log_message(_INCOMING, "user", message)


def _score_recovered_with_work(db: Database, sent: list[str]) -> list[str]:
    """Pass iff the cycle recovered from the forced duplicate call: it reused the
    earlier read and wrote the summary entry the seeded messages clearly warrant."""
    if collection_entries(db, WEEKLY_DIGEST.name):
        return []
    read = tool_was_called(db, "log_read")
    wrote = tool_was_called(db, "collection_write")
    return [
        "did not recover after the duplicate-call rejection — no summary written "
        f"(log_read={read}, collection_write={wrote}); the model likely over-"
        "generalized the rejection and suppressed the owed write"
    ]


async def test_duplicate_call_is_rejected_and_recovers(guard_recovery_eval) -> None:
    """A byte-identical repeat of the first tool call is rejected in place; the live
    model reuses the earlier result and still writes the summary it owes."""
    await guard_recovery_eval(
        case_id="duplicate-call-recovery",
        collection=WEEKLY_DIGEST.name,
        seed=_seed_digest_with_messages,
        wrap_client=lambda real: _InjectDuplicateCall(real),
        score=_score_recovered_with_work,
        min_pass_rate=0.75,
    )
