"""Add ``messagelog.exchange_embedding`` — one exchange as one vector (#3010).

Type: schema

Each exchange is a user message plus Penny's reply to it, embedded as one text
with the same model as ``messagelog.embedding`` and stored on the REPLY row.
The related-exchanges retrieval scores the live user message against these
vectors and renders the top matches in full in the per-turn dynamic system
message, so an agreed plan stays visible after it has left the history window
and the conversation map.  The exchange — not the user message alone — is the
unit, because the user's later message often refers to the plan by a pronoun
("do those") while the plan's words sit in her reply.

Vectors are computed when the reply is logged (``_deliver_agent_response``) and
backfilled for existing exchanges by the startup message-embedding backfill
(``Penny._backfill_message_embeddings``), which reconstructs both sides off the
reply row's run stamps.  A migration cannot embed, so this ships the column
empty and the backfill fills it on the next restart — the same shape every
other embedding-bearing column uses.

One nullable column plus the partial index the retrieval query scans
(``ix_messagelog_exchange_time``, over exchange-bearing rows only).  Schema
only, universal — a fresh ``create_tables()`` DB already carries both from the
model (the guard skips them); a production copy that predates them gets them
added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_messagelog_exchange_embedding(conn)
    _add_exchange_time_index(conn)
    conn.commit()


def _add_messagelog_exchange_embedding(conn: sqlite3.Connection) -> None:
    """Add the nullable ``exchange_embedding`` column to ``messagelog`` (idempotent)."""
    if not _table_exists(conn, "messagelog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
    if "exchange_embedding" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN exchange_embedding BLOB")


def _add_exchange_time_index(conn: sqlite3.Connection) -> None:
    """Add the retrieval scan's partial index (idempotent; dropped with the column absent)."""
    if not _table_exists(conn, "messagelog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
    if "exchange_embedding" not in columns:
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_exchange_time "
        "ON messagelog (timestamp, id) WHERE exchange_embedding IS NOT NULL"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
