"""Add ``promptlog.trailing_messages`` — the run tail no model call carried (#1778).

Type: schema

A tool RESULT is never written as a record of its own: it is persisted only as a side
effect of being fed back to the model, as a ``role:"tool"`` turn inside the NEXT call's
``messages``.  A run that ends immediately after executing a tool — a write-gate STOP,
``max_steps`` reached on a tool step, a reroll abort, an exception — has no next call,
so its terminal call's outcome was written nowhere and the run record silently omitted
it.  Those are exactly the runs worth reading afterwards.

This column holds that tail: the turns the agent loop appended after its final model
call, in the SAME wire shape ``messages`` uses (it is literally what the next call's
``messages`` would have ended with).  Stamped on the run's last prompt row; NULL for a
run that ended on a text reply (every result already round-tripped), for every row that
isn't its run's last, and for every historical row — so the run renderers read
``messages`` + this tail as one sequence and pre-existing runs render byte-identically.

One nullable column.  Schema only, universal — a fresh ``create_tables()`` DB already
carries it from the model (the guard skips it); a production copy that predates it gets
it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_promptlog_trailing_messages(conn)
    conn.commit()


def _add_promptlog_trailing_messages(conn: sqlite3.Connection) -> None:
    """Add the nullable ``trailing_messages`` column to ``promptlog`` (idempotent)."""
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "trailing_messages" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN trailing_messages TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
