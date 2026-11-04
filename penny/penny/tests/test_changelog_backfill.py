"""Tests for ``penny.changelog_backfill`` (#2962) — the one-off script that
backfills the `changelog` log for PRs merged before the log existed.

``WINDOW_ENTRIES`` itself (the real 76-row table) is never exercised here —
these tests run ``build_entries``/``run_backfill`` over tiny, made-up
windows, the same way every other changelog test uses synthetic PR numbers
instead of real ones."""

from __future__ import annotations

from datetime import UTC, datetime

from penny.changelog import ChangelogEntry, record_new_entries
from penny.changelog_backfill import build_entries, run_backfill
from penny.constants import PennyConstants
from penny.tests.conftest import require_memory

_WINDOW = [
    (9501, "2026-10-19T04:00:00Z", "an old PR's own first sentence"),
    (9502, "2026-10-20T05:00:00Z", "a second old PR's own first sentence"),
]


def test_build_entries_pairs_each_row_with_its_parsed_created_at():
    """Pure: no DB, no fetch — just the window's own tuples turned into
    ``(ChangelogEntry, created_at)`` pairs, ``created_at`` parsed off each
    row's own ``merged_at``."""
    pairs = build_entries(_WINDOW)
    assert pairs == [
        (
            ChangelogEntry(pr=9501, text="an old PR's own first sentence"),
            datetime(2026, 10, 19, 4, 0, tzinfo=UTC),
        ),
        (
            ChangelogEntry(pr=9502, text="a second old PR's own first sentence"),
            datetime(2026, 10, 20, 5, 0, tzinfo=UTC),
        ),
    ]


def test_run_backfill_records_every_row_then_a_rerun_is_a_no_op(db):
    """The whole script's flow: every window row lands once, each stamped
    with ITS OWN merge time (never "now") — and running it again against the
    same log adds nothing, so re-running the script is safe."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)

    recorded = run_backfill(changelog, _WINDOW)
    assert [e.render() for e in recorded] == [
        "PR #9501: an old PR's own first sentence",
        "PR #9502: a second old PR's own first sentence",
    ]
    rows = changelog.read_all()
    assert [row.content for row in rows] == [e.render() for e in recorded]
    assert [row.created_at.replace(tzinfo=UTC) for row in rows] == [
        datetime(2026, 10, 19, 4, 0, tzinfo=UTC),
        datetime(2026, 10, 20, 5, 0, tzinfo=UTC),
    ]

    again = run_backfill(changelog, _WINDOW)
    assert again == []
    assert len(changelog.read_all()) == 2


def test_run_backfill_sorts_before_an_entry_already_recorded_for_real(db):
    """A live deploy already recorded an entry (stamped with the deploy's own
    "now") before the backfill ever runs — the exact order #2962 happened
    in prod. The backfilled, older PRs still read BEFORE it: entries are
    read in ``created_at`` order, so the log stays truthful about when
    things actually happened, not when they were typed in."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)
    live_entry = ChangelogEntry(pr=9601, text="a live deploy's own entry")
    record_new_entries(changelog, [live_entry], created_at=datetime(2026, 11, 1, tzinfo=UTC))

    recorded = run_backfill(changelog, _WINDOW)
    assert len(recorded) == 2

    rows = changelog.read_all()
    assert [row.content for row in rows] == [
        "PR #9501: an old PR's own first sentence",
        "PR #9502: a second old PR's own first sentence",
        "PR #9601: a live deploy's own entry",
    ]
