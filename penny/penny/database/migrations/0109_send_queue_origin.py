"""Add ``send_queue.origin`` — which lane a queued message is delivered in (#1939).

Type: schema

Every queued message rode one delivery lane: the autonomous-send cooldown, 600 seconds
since Penny's last outgoing message.  That is the right rule for a message nobody asked
for, and the wrong one for a message the user is sitting in front of waiting for —
observed live, a user pressed "run this now" in the addon and waited ~10.5 minutes for
the run's notification, because an unrelated chat reply seconds earlier had started the
cooldown.

This column carries what set the queuing cycle running (``CycleTrigger``): ``cadence``,
the schedule's own dispatch, or ``on_demand``, the user's own trigger.  The drainer reads
it and delivers an on-demand row on the next tick regardless of the cooldown, while a
cadence row waits it out exactly as before.

One NOT NULL column defaulting to ``cadence``: every existing row was queued by a
scheduled cycle, so the backfill is the truth rather than a guess, and an unstamped row
is in the conservative lane — the bypass has to be claimed.  Plus the partial index the
lane-scoped read seeks on, since the sparse case is the one the feature exists for: one
on-demand row behind a backlog of cadence ones.  Schema only, universal — a fresh
``create_tables()`` DB already carries both from the model (the guards skip them); a
production copy that predates them gets them added.
"""

from __future__ import annotations

import sqlite3

# A FROZEN COPY of ``CycleTrigger.CADENCE``: a migration states the value it wrote at
# the time it ran, so a later rename of the enum member cannot change history.
_CADENCE = "cadence"


def up(conn: sqlite3.Connection) -> None:
    _add_send_queue_origin(conn)
    _index_the_pending_lane(conn)
    conn.commit()


def _add_send_queue_origin(conn: sqlite3.Connection) -> None:
    """Add the NOT NULL ``origin`` column to ``send_queue`` (idempotent)."""
    if not _table_exists(conn, "send_queue"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(send_queue)").fetchall()}
    if "origin" in columns:
        return
    conn.execute(f"ALTER TABLE send_queue ADD COLUMN origin TEXT NOT NULL DEFAULT '{_CADENCE}'")


def _index_the_pending_lane(conn: sqlite3.Connection) -> None:
    """Index the pending tail by lane, after the column it leads with exists.

    The 0061 index serves the ORDER BY alone, which is enough for the unscoped read
    and not for the lane-scoped one — that filter is an equality on a sparse value, so
    without a leading column the ``LIMIT 1`` walks the whole ordered pending partition
    to find it."""
    if not _table_exists(conn, "send_queue"):
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_send_queue_pending_lane "
        "ON send_queue (origin, created_at) "
        "WHERE sent_at IS NULL AND cancelled_at IS NULL"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
