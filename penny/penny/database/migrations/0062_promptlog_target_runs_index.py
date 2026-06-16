"""Index completed runs by ``run_target`` — fix the per-collection run panel.

The addon's memory-detail view loads one collection's collector activity via
``RunLog`` scoped to ``run_target = <name>``:

    WHERE run_outcome IS NOT NULL AND run_target = ? ORDER BY timestamp DESC LIMIT 50

The only partial index over completed runs (``ix_promptlog_completed_runs``,
migration 0059) is keyed on ``timestamp`` alone, so this scoped query can't seek
to the target — it walks every completed run newest-first, fetching each row to
test ``run_target``.  A sparse collection (a handful of runs among tens of
thousands) never fills the 50-row limit, so the walk scans the whole
completed-run history: a multi-second freeze on every memory click.

This adds a composite partial index keyed on ``(run_target, timestamp)`` so the
scoped read seeks straight to the target's runs and reads them in completion
order.  The existing single-column partial index stays — it still serves the
unscoped ``collector-runs`` log (``run_target IS NOT NULL``), whose ordering the
target-leading index can't satisfy.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" in tables:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_promptlog_target_runs "
            "ON promptlog (run_target, timestamp) WHERE run_outcome IS NOT NULL"
        )
