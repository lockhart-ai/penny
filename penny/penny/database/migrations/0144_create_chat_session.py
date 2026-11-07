"""Add ``chat_session`` — the chat session's stored boundary (#3125).

Type: schema

The session's start is now a stored fact, not a timestamp-gap calculation:
the idle rotation task (``ChatSessionRotationTask``) and a real turn's own
``MessageStore.record_idle_rotation``/``record_cap_rotation`` write a row
here once the user has gone idle long enough, or a session's history has
grown past ``CHAT_SESSION_MAX_TURNS`` — the instant every reader
(``_dispatch_turn``, the chat pre-warm, the conversation map, related
exchanges) reads off ``MessageStore.session_anchor`` as the one shared
boundary. One row per sender, upserted in place — a session only ever has
ONE current start.

Schema only, universal — a fresh ``create_tables()`` DB already carries the
table from the model (the guard skips it); a production copy that predates
it gets it created. No boundary row yet means "everything stored so far is
one session" (``MessageStore.session_anchor``'s own no-row fallback) — the
first idle gap or cap-crossing writes the first real row, so there is
nothing to backfill here.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "chat_session"):
        conn.execute(
            """
            CREATE TABLE chat_session (
                sender VARCHAR NOT NULL PRIMARY KEY,
                started_at DATETIME NOT NULL
            )
            """
        )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name", {"name": name}
    ).fetchone()
    return row is not None
