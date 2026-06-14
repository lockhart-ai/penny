"""Quality-collector contracts — the graduated self-correcting collector.

The ``quality`` collection is seeded by migration 0055, so it exists in every
DB (prod and test).  These cases drive the REAL seeded extraction_prompt via
``run_for("quality")``: seed one synthetic suspect collection (its intent + a
drifted prompt) plus the logs of its misbehaviour, then check the persisted
correction.  The collector gates the real ``prompt_test`` dry-run tool into the
quality cycle's surface, so the corrective cases also assert the model dry-ran
its fix before applying it.

  rebroadcast  — intent "one fresh thought, never repeat"; a digest re-sent →
                 rewrite the prompt (loose bar: any material corrective change).
  silent-drift — intent "never ping me"; still sends updates → drop send_message.
  healthy      — behaviour matches intent → change nothing.
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import Inclusion, LogEntryInput, RecallMode
from penny.tests.eval.conftest import CollectorScorer, tool_was_called

pytestmark = pytest.mark.eval

# The quality flow is the hardest multi-hop cycle (read two logs → diagnose →
# read metadata → dry-run → rewrite → notify).  gpt-oss clears it only some of
# the time per sample, so the corrective cases use a looser bar than the suite
# default — this is the surface you iterate the seeded prompt against.
_QUALITY_PASS_RATE = 0.6

# ── Synthetic suspect collections (intent + a drifted extraction_prompt) ─────

_DIGEST_PROMPT = (
    "Share one fresh daily digest thought.\n"
    '1. log_read_recent("penny-messages") — re-read what you sent so you do not '
    "repeat yourself.\n"
    "2. Compose a short digest of the latest items.\n"
    "3. send_message the digest.\n"
    "4. done()."
)
_SILENT_DRIFT_PROMPT = (
    "Collect espresso equipment worth considering.\n"
    "1. browse(...) for new espresso gear; read actual pages.\n"
    '2. collection_write("espresso-gear", entries=[...]).\n'
    "3. If the write succeeded, send_message: one-sentence 'found a new item' + URL.\n"
    "4. done()."
)
_HEALTHY_PROMPT = (
    "Collect houseplant care tips.\n"
    "1. browse(...) for fresh houseplant-care advice; read pages.\n"
    '2. collection_write("houseplant-care", entries=[...]).\n'
    "3. If a genuinely new tip was written, send_message one sentence + URL.\n"
    "4. done()."
)


def _seed(*, suspect: str, description: str, intent: str, prompt: str, runs, penny_msgs):
    """Seeder: one suspect collection (drifted) + its log trail.

    The ``quality`` collection itself is already present from migration 0055.
    """

    def _apply(db: Database) -> None:
        db.memories.create_collection(
            suspect,
            description,
            Inclusion.RELEVANT,
            RecallMode.RECENT,
            extraction_prompt=prompt,
            intent=intent,
        )
        db.memories.append(
            PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
            [LogEntryInput(content=entry) for entry in runs],
            author="collector",
        )
        db.memories.append(
            PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
            [LogEntryInput(content=entry) for entry in penny_msgs],
            author="penny",
        )

    return _apply


def _snapshot(suspect: str):
    def _take(db: Database) -> str:
        memory = db.memories.get(suspect)
        return (memory.extraction_prompt or "") if memory else ""

    return _take


def _score_update(suspect: str, forbidden: str | None) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        fails = []
        if not tool_was_called(db, "prompt_test"):
            fails.append("did not dry-run the fix with prompt_test before applying")
        if new_prompt == original:
            fails.append(f"did not change {suspect!r}'s extraction_prompt")
        elif forbidden is not None and forbidden in new_prompt:
            fails.append(f"corrected prompt still contains the offending {forbidden!r} step")
        elif len(new_prompt) < 80:
            fails.append(f"corrected prompt looks gutted ({len(new_prompt)} chars)")
        if not sent:
            fails.append("did not message the user about the change")
        return fails

    return _score


def _score_no_op(suspect: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        if new_prompt != original:
            return [f"over-corrected a healthy collection ({suspect!r})"]
        return []

    return _score


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_rebroadcast(collector_eval) -> None:
    suspect = "daily-digest"
    await collector_eval(
        case_id="quality-rebroadcast",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A once-daily digest of fresh items worth a heads-up.",
            intent="Once per cycle, share exactly one fresh thought I haven't seen "
            "before, and never resend something you've already sent me.",
            prompt=_DIGEST_PROMPT,
            runs=[
                "[daily-digest] Sent the daily digest and marked it shared.",
                "[daily-digest] Sent the daily digest.",
            ],
            penny_msgs=[
                "Daily digest — new co-op title announced, a reprint, and a sale.",
                "Daily digest — new co-op title announced, a reprint, and a sale.",
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden=None),
        min_pass_rate=_QUALITY_PASS_RATE,
    )


async def test_silent_drift(collector_eval) -> None:
    suspect = "espresso-gear"
    await collector_eval(
        case_id="quality-silent-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A quiet running list of espresso equipment worth considering.",
            intent="Keep a quiet running list of espresso equipment worth considering "
            "— never ping me about it, I'll check the list myself.",
            prompt=_SILENT_DRIFT_PROMPT,
            runs=["[espresso-gear] wrote 2 entries and sent an update about a new grinder."],
            penny_msgs=["Found a new espresso grinder: the Niche Zero clone, $300."],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden="send_message"),
        min_pass_rate=_QUALITY_PASS_RATE,
    )


async def test_healthy(collector_eval) -> None:
    suspect = "houseplant-care"
    await collector_eval(
        case_id="quality-healthy",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A list of houseplant-care tips, with a ping on genuinely new ones.",
            intent="Keep a list of houseplant-care tips and ping me when you find a "
            "genuinely new one.",
            prompt=_HEALTHY_PROMPT,
            runs=["[houseplant-care] wrote 1 new entry and sent one update about watering."],
            penny_msgs=["New houseplant tip: bottom-water pothos weekly to avoid root rot."],
        ),
        snapshot=_snapshot(suspect),
        score=_score_no_op(suspect),
    )
