"""Record which wire shape served each prompt log call (#2829).

Type: schema

Penny now has two transports for a model call: the OpenAI chat-completions
API, and — for a model with a vendored chat template — rendering that
template itself and sending the result as a raw completions prompt.  This
adds one nullable ``transport`` column ("chat" | "raw_template",
``penny.llm.models.PromptTransport``), stamped by ``LlmClient`` on every new
call.  NULL on every row logged before this column existed, read as "chat"
by convention — the only transport that existed then, so no backfill is
needed.

A fresh ``create_tables()`` DB already carries the column from the model (the
guard below is a no-op there); a production copy that predates this migration
gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "transport" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN transport TEXT")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
