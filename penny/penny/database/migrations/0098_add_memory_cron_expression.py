"""Add ``memory.cron_expression`` — the cron trigger's 5-field schedule (#1684).

Type: schema

Completes the trigger union (epic #1554 via #1570) with a fourth form: a collection
can carry a 5-field cron expression (e.g. ``0 8,20 * * *``) so the model can encode a
time-of-day recurrence a stated schedule states in words ("morning and evening",
"weekdays at 9") — inexpressible in the ``every`` / ``once at`` / ``on advance of``
forms.  ``croniter`` computes the next fire time; the collector gates readiness on it.

One nullable column: NULL for every collection on the recurring, once-shaped, or
on_advance trigger (the whole existing set), set only for a collection created with the
model-facing ``cron`` trigger form.  Schema only, universal — a fresh
``create_tables()`` DB already carries it from the model (the guard skips it); a
production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_memory_cron_expression(conn)
    conn.commit()


def _add_memory_cron_expression(conn: sqlite3.Connection) -> None:
    """Add the nullable ``cron_expression`` column to ``memory`` (idempotent)."""
    if not _table_exists(conn, "memory"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "cron_expression" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN cron_expression TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
