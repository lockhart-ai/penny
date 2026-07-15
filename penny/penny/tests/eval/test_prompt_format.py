"""Prompt-format contracts — numbered instructions vs prose, the bailout effect.

We found gpt-oss follows a NUMBERED instruction/tool-call recipe far more reliably
than the SAME task written as PROSE.  Replaying real failing collector prompts, a
prose task in the system prompt bailed (jumped to ``done()`` without doing the
work) ~60% of the time on the empty collector user turn; the numbered rewrite of
the identical task bailed ~5%.  Format dominated placement.

These cases pin that effect through the REAL collector loop (``run_for``) so it's a
regression guard, not just a one-off replay:

  numbered-engages   — a numbered collector reliably does its work (GATED).
  prose-bails        — the SAME task as prose does it less often (report-only; the
                       gap to ``numbered-engages`` is the format effect to watch).
  reads-empty-first  — even on an empty log, a numbered collector READS before
                       concluding no-work (the production bailout was the model
                       assuming "the user said nothing" and never checking).

Collector prompts are now deterministic renders of taught skills (#1590/#1591), so
the numbered shape is produced by the render itself — there is no longer a
model-judgment reviewer that rewrites prose into numbered steps (the ``quality``
collector was retired by #1569).  These cases guard that the render's numbered
recipe keeps engaging the model.
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.tests.eval.conftest import CollectorScorer, collection_entries, tool_was_called
from penny.tests.eval.fixtures import (
    WATCHLIST,
    WATCHLIST_MESSAGES,
    WATCHLIST_NUMBERED_PROMPT,
    WATCHLIST_PROSE_PROMPT,
    WEEKLY_DIGEST,
    WEEKLY_DIGEST_EXTRACTION_PROMPT,
)

pytestmark = pytest.mark.eval

_INCOMING = PennyConstants.MessageDirection.INCOMING


# ── Seeders ───────────────────────────────────────────────────────────────────


def _seed_watchlist(prompt: str):
    """Create the watchlist collector with ``prompt`` + seed clearly-watchable msgs.

    The messages name two titles the user wants to watch, so a working cycle MUST
    write an entry — a no-write is a bailout, not a defensible no-op.
    """

    def _apply(db: Database) -> None:
        db.memories.create_collection(
            WATCHLIST.name,
            WATCHLIST.description,
            extraction_prompt=prompt,
            collector_interval_seconds=300,
        )
        for message in WATCHLIST_MESSAGES:
            db.messages.log_message(_INCOMING, "user", message)

    return _apply


def _seed_digest_empty(db: Database) -> None:
    """Numbered digest collector, but NO seeded messages (empty log).

    Correct behaviour is to ``log_read`` first, see it's empty, and no-op.  The
    bailout is concluding "the user said nothing" from the empty user turn and
    calling ``done()`` without ever reading.
    """
    db.memories.create_collection(
        WEEKLY_DIGEST.name,
        WEEKLY_DIGEST.description,
        extraction_prompt=WEEKLY_DIGEST_EXTRACTION_PROMPT,
        collector_interval_seconds=1200,
    )


def _snapshot(name: str):
    def _take(db: Database) -> dict[str, str]:
        return collection_entries(db, name)

    return _take


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_wrote(name: str) -> CollectorScorer:
    """Pass iff the cycle wrote a new entry; on miss, surface the bailout shape."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        if set(collection_entries(db, name)) - set(before_entries):
            return []
        read = tool_was_called(db, "log_read")
        wrote = tool_was_called(db, "collection_write")
        done = tool_was_called(db, "done")
        if done and not read and not wrote:
            return ["bailout: called done() without reading the log (empty-user-turn)"]
        return [f"no entry written (log_read={read}, collection_write={wrote}, done={done})"]

    return _score


def _score_read_before_giving_up(db: Database, before: object, sent: list[str]) -> list[str]:
    """Pass iff the cycle READ the log before concluding no-work (process, not outcome)."""
    if tool_was_called(db, "log_read"):
        return []
    return ["bailout: concluded no-work and called done() without ever reading the log"]


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_numbered_prompt_engages(collector_eval) -> None:
    """GATED: a numbered collector reliably does its work (the contract to hold)."""
    await collector_eval(
        case_id="format-numbered-engages",
        collection=WATCHLIST.name,
        seed=_seed_watchlist(WATCHLIST_NUMBERED_PROMPT),
        snapshot=_snapshot(WATCHLIST.name),
        score=_score_wrote(WATCHLIST.name),
        min_pass_rate=0.75,
    )


async def test_prose_prompt_bails(collector_eval) -> None:
    """REPORT-ONLY: the SAME task as prose does the work less reliably.

    The gap between this rate and ``test_numbered_prompt_engages`` is the format
    effect — watch it shrink to ~0 once prose prompts are rewritten to numbered.
    """
    await collector_eval(
        case_id="format-prose-bails",
        collection=WATCHLIST.name,
        seed=_seed_watchlist(WATCHLIST_PROSE_PROMPT),
        snapshot=_snapshot(WATCHLIST.name),
        score=_score_wrote(WATCHLIST.name),
        min_pass_rate=None,
    )


async def test_numbered_reads_empty_log_first(collector_eval) -> None:
    """REPORT-ONLY: faithful reproduction of the production bailout on an empty log."""
    await collector_eval(
        case_id="format-reads-empty-first",
        collection=WEEKLY_DIGEST.name,
        seed=_seed_digest_empty,
        snapshot=_snapshot(WEEKLY_DIGEST.name),
        score=_score_read_before_giving_up,
        min_pass_rate=None,
    )
