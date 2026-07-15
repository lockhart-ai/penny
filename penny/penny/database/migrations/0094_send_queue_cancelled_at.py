"""Add ``send_queue.cancelled_at`` — visible cancellation of pending sends (#1634).

Type: schema

Archiving a collection cancels its still-pending queued sends so a torn-down
mechanism can't speak from the grave through the send queue (epic #1554, journey
gap 3, #1570).  Cancellation is VISIBLE, not deletion: a pending row is stamped
``cancelled_at`` (kept as an audit trail, like a delivered row) and excluded from
the drainer's pending query structurally.  ``sent_at`` stays NULL on a cancelled
row — it was never sent — so the delivered/cancelled distinction is unambiguous
and ``sent_at`` remains the single source of truth for "was it delivered".

One nullable column: NULL for every existing row (pending or delivered).  Schema
only, universal, idempotent — a fresh ``create_tables()`` DB already carries it
from the model (the guard skips it); a production copy that predates it gets it
added.  The 0061 partial index (``WHERE sent_at IS NULL``) still serves the
pending read — a cancelled row is a filtered subset of it — so no index change.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_send_queue_cancelled_at(conn)
    conn.commit()


def _add_send_queue_cancelled_at(conn: sqlite3.Connection) -> None:
    """Add the nullable ``cancelled_at`` column to ``send_queue`` (idempotent)."""
    if not _table_exists(conn, "send_queue"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(send_queue)").fetchall()}
    if "cancelled_at" not in columns:
        conn.execute("ALTER TABLE send_queue ADD COLUMN cancelled_at TIMESTAMP")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
