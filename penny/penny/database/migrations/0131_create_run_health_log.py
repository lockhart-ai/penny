"""Create the ``run-health`` log memory (#2843).

One line per run — across chat AND every collector, unlike ``collector-runs``
(collector-only by design: the seeded ``quality`` prompt reads it expecting
only collector records).  Each line carries its calls/output-tokens/time,
discarded draws by reason, failed tool calls by tool, and whether a message
went out and carried a link — read off existing columns only.  Read via
``log_read`` (cursor / day); the same facts also render on ``run_list``/
``run_get`` (``render_run_calls``), which filter by source.

Stores nothing of its own (a read facade over ``promptlog``, like
``collector-runs``) — the row here is only the marker the store map names and
``log_read`` dispatches on.  ``recall`` was retired by migration 0093, so this
seeds only the columns the current ``memory`` table actually has (migration
0119 is the most recent seed of this shape).

Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

NAME = "run-health"
DESCRIPTION = (
    "One line per run: cost, discards, tool errors, and what was sent — chat and every job"
)


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory (name, type, description, archived, created_at) "
        "VALUES (?, 'log', ?, 0, ?)",
        (NAME, DESCRIPTION, now),
    )
    conn.commit()
