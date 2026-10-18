"""Add the failed call's own fault classification to ``promptlog`` (#2526).

Type: schema

``LlmError`` already classifies a response error by status the moment it raises
(``fault_for_status`` in ``penny.llm.models`` — 429 is ``RATE_LIMITED``, 5xx is
``SERVER_ERROR``, 4xx is ``CLIENT_ERROR``, a 200 with no completion is ``NO_CHOICES``),
but migration 0114 persisted only the exception's class name and message — the
classification itself lived on the in-memory exception and was gone once the row was
written. A reader of the ``promptlog`` (the eval harness, chiefly) had no way to tell a
provider-side refusal (429/5xx — the provider's own doing) from Penny's own malformed or
oversized request (4xx) without re-parsing the error message's free text.

One nullable column, added once: ``call_error_fault`` — the failed call's own
``LlmFault`` value, set only on "timed_out"/"raised" (NULL on every row that predates
this migration, and on "completed"/"cancelled" rows, exactly like its two siblings).

A fresh ``create_tables()`` DB already carries this column from the model (the guard
below is a no-op there); a production copy that predates this migration gets it added.
"""

from __future__ import annotations

import sqlite3

_NEW_COLUMN = "call_error_fault"


def up(conn: sqlite3.Connection) -> None:
    _add_promptlog_call_error_fault_column(conn)
    conn.commit()


def _add_promptlog_call_error_fault_column(conn: sqlite3.Connection) -> None:
    """Add the one nullable column to ``promptlog`` (idempotent)."""
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if _NEW_COLUMN not in columns:
        conn.execute(f"ALTER TABLE promptlog ADD COLUMN {_NEW_COLUMN} TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
