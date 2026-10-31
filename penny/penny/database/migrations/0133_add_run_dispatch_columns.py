"""Record why each collector run started when it did (#2862).

Type: schema

Each collector cycle's dispatch reason is decided ONCE, by the same code that
already decides whether a collection is ready — never re-derived at render
time.  This adds three nullable ``promptlog`` columns, stamped on the run's
last prompt row alongside ``run_outcome``:

* ``run_due_at`` — the occurrence the dispatcher picked this cycle up for,
  read off the same schedule computation ``Collector._schedule_due`` already
  makes.  NULL for an on-demand run (``run_for`` bypasses readiness, so there
  is no occurrence) and for a chat run.
* ``run_dispatch_reason`` — a ``DispatchReason`` value: ``on_time``,
  ``catch_up``, ``catch_up_offline``, ``retry_preempted``, ``retry_failed``,
  or ``on_demand``.
* ``run_dispatch_attempt`` — the attempt number at this occurrence, set only
  for the two retry reasons.

Universal: all three are nullable columns on a table every deployment has; no
row is targeted by key.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "run_due_at" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_due_at DATETIME")
    if "run_dispatch_reason" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_dispatch_reason TEXT")
    if "run_dispatch_attempt" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_dispatch_attempt INTEGER")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
