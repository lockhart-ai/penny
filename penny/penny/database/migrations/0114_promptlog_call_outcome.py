"""Add the columns a model call that does not finish needs to be logged (#2355 part A).

Type: schema

``LlmClient.chat`` wrote its ``promptlog`` row only once ``chat.completions.create``
had returned a usable response, so a call cancelled by a tool's own timeout, or one
that exhausted its retries and raised, left no row at all: no input, no duration, no
agent name — the only trace of it was a DEBUG line in the runtime log, which rotates.

Five nullable columns, added once:

- ``call_outcome`` — how the call ended: "completed" | "cancelled" | "timed_out" |
  "raised" (``penny.llm.models.CallOutcome``).  NULL on every pre-existing row, read as
  "completed" by convention (the old code path only ever logged a call that had
  succeeded).
- ``call_error_class`` / ``call_error_message`` — the failed call's own exception class
  and message, set only on "timed_out"/"raised".
- ``parent_run_id`` / ``parent_tool_call_id`` — the enclosing chat/collector run (and,
  for a draw made from inside a dispatched tool call, that call's own id) a
  micro-context draw was made on behalf of.  A micro-context mints its own ``run_id``
  for ledger attribution, so without this a turn's own record could not list the model
  calls made for it.

``response`` is left exactly as it is (``TEXT NOT NULL``) rather than widened to
nullable: SQLite has no ``ALTER COLUMN ... DROP NOT NULL``, so that would mean a full
table rebuild (and re-creating the FTS5 triggers from migration 0051) on what is often
the largest table in the database, for no reader that needs the distinction — every
existing reader already treats a falsy ``response`` as an empty dict
(``json.loads(prompt.response) if prompt.response else {}``), so a call that never got
one writes the literal ``"{}"`` and every reader sees exactly what it already expects.

A fresh ``create_tables()`` DB already carries these columns from the model (the guard
below is a no-op there); a production copy that predates this migration gets them
added.
"""

from __future__ import annotations

import sqlite3

_NEW_COLUMNS = (
    "call_outcome",
    "call_error_class",
    "call_error_message",
    "parent_run_id",
    "parent_tool_call_id",
)


def up(conn: sqlite3.Connection) -> None:
    _add_promptlog_call_outcome_columns(conn)
    _add_parent_run_id_index(conn)
    conn.commit()


def _add_promptlog_call_outcome_columns(conn: sqlite3.Connection) -> None:
    """Add the five nullable columns to ``promptlog`` (idempotent, per column)."""
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    for column in _NEW_COLUMNS:
        if column not in columns:
            conn.execute(f"ALTER TABLE promptlog ADD COLUMN {column} TEXT")


def _add_parent_run_id_index(conn: sqlite3.Connection) -> None:
    """Index ``parent_run_id`` so ``run_get`` can list the draws made for a run without
    a full table scan — the same shape as the existing ``ix_promptlog_run_id``."""
    if not _table_exists(conn, "promptlog"):
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_promptlog_parent_run_id ON promptlog (parent_run_id)"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
