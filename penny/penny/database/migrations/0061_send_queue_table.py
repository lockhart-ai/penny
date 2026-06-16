"""Add the `send_queue` table — durable outbound message queue.

``send_message`` no longer drops a message when the autonomous-send cooldown
hasn't elapsed; it enqueues the message here and a background drain schedule
delivers it once the cooldown clears.  ``sent_at IS NULL`` is the single
source of truth for "still pending" — no separate boolean flag to desync.
``collection`` records which collector queued the message (the bound target),
so the queue is attributable the same way ``promptlog.run_target`` is.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "send_queue" not in tables:
        conn.execute("""
            CREATE TABLE send_queue (
                id INTEGER PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                collection TEXT NOT NULL,
                sent_at TIMESTAMP
            )
        """)
        # The drain reads the oldest pending row each tick — index the pending
        # tail (sent_at IS NULL) by arrival order so the read never scans sent rows.
        conn.execute(
            "CREATE INDEX ix_send_queue_pending ON send_queue (created_at) WHERE sent_at IS NULL"
        )
