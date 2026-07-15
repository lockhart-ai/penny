"""Add ``memory.source_log`` — the on_advance trigger's declared source (#1604).

Type: schema

Completes the trigger union (epic #1554 via mini-epic #1562): a collection can
declare a *source log* so it wakes when that log's high-water mark passes the
collection's read cursor — the ``on_advance`` trigger, the declared-input variant
of the existing (inferred) cursor gate.  ``source_log`` names the log; the
collector reads the frontier structurally in Python (source head > cursor), never
a model judgment.

One nullable column: NULL for every collection on the recurring or once-shaped
trigger (the whole existing set), set only for a collection created with the
model-facing ``on_advance`` arg.  Schema only, universal — a fresh
``create_tables()`` DB already carries it from the model (the guard skips it); a
production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_memory_source_log(conn)
    conn.commit()


def _add_memory_source_log(conn: sqlite3.Connection) -> None:
    """Add the nullable ``source_log`` column to ``memory`` (idempotent)."""
    if not _table_exists(conn, "memory"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "source_log" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN source_log TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
