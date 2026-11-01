"""Record the exact request sent for every model call (#2875).

Type: schema

Today's ``promptlog`` stores a call's ``messages`` and ``tools`` — the inputs a
replay re-renders through the CURRENT code's template, with whatever sampling the
replaying client happens to use.  This adds one nullable ``request`` BLOB column:
``LlmClient`` stamps it with a zlib-compressed JSON dump of its own
``SentRequest`` — the literal kwargs handed to the OpenAI SDK's ``create()``
(chat transport), or those plus the already-rendered raw-template prompt string
— so a prod failure replays byte for byte instead of through a reconstruction.

Compressed because the dump duplicates most of the SAME row's ``messages``/
``tools`` and its JSON text compresses well.  Carries configured extra-header
NAMES only (never their values — ``LLM_EXTRA_HEADERS`` can carry anything,
credentials included) and never the API key.

NULL on every row logged before this column existed, and on a cancelled call,
which never got far enough to leave a request behind.

Universal: one nullable column on a table every deployment has; no row is
targeted by key.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "request" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN request BLOB")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
