"""Tests for migration 0027 — data migration into the memory framework.

The surviving block of the migration (messages → the user/penny facades) gets a
focused test that seeds the old table, runs the FULL migration chain, and verifies
the resulting rows.

THE WATCHED DELETION (#1911's soft reboot): the three tests that pinned 0027's
PREFERENCE split — the valence partition, its idempotency, and its
already-populated guard — are GONE with their subject.  0027 wrote into ``likes``
and ``dislikes``; 0097 nuked the first and 0108 now drops the second, so at
end-of-chain the split has no observable state left to assert on, and the
end-state that replaced it is
``test_0108_leaves_no_seeded_collection_at_all`` in ``test_migrations.py``.  The
message logs stay, because they are facades over ``messagelog`` rather than
seeded collections — nothing dropped them.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from penny.database import Database
from penny.database.migrate import migrate
from penny.llm.embeddings import serialize_embedding


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — migrations off so we control timing."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _seed_message(
    conn: sqlite3.Connection,
    direction: str,
    content: str,
    timestamp: datetime,
    embedding: bytes | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messagelog"
        " (direction, sender, content, timestamp, is_reaction, processed, embedding)"
        " VALUES (?, '+15551234567', ?, ?, 0, 0, ?)",
        (direction, content, timestamp.isoformat(), embedding),
    )


def _seed_preference(
    conn: sqlite3.Connection,
    content: str,
    valence: str,
    created_at: datetime,
    embedding: bytes | None = None,
) -> None:
    """Insert a legacy ``preference`` row, creating the pre-0097 table if needed.

    The ``Preference`` model is gone (0097 drops the table), so ``create_tables``
    no longer materialises it — recreate the legacy shape 0027 reads from, exactly
    as an old deployment would carry it (migration 0001's CREATE IF NOT EXISTS
    then leaves it alone, and 0097 drops it at end-of-chain).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS preference ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL,"
        " content TEXT NOT NULL, valence TEXT NOT NULL, embedding BLOB,"
        " created_at TIMESTAMP NOT NULL, last_thought_at TIMESTAMP,"
        " mention_count INTEGER NOT NULL DEFAULT 1,"
        " source TEXT NOT NULL DEFAULT 'extracted')"
    )
    conn.execute(
        "INSERT INTO preference"
        " (user, content, valence, embedding, created_at, mention_count, source)"
        " VALUES ('+15551234567', ?, ?, ?, ?, 1, 'extracted')",
        (content, valence, embedding, created_at.isoformat()),
    )


def _entries(conn: sqlite3.Connection, name: str) -> list[tuple]:
    """Return rows from memory_entry for a memory in chronological order."""
    return conn.execute(
        "SELECT key, content, author, key_embedding, content_embedding"
        " FROM memory_entry WHERE memory_name = ? ORDER BY created_at ASC, id ASC",
        (name,),
    ).fetchall()


# ── Happy path: each source table populates its target memory ──────────────


def test_messages_split_into_user_and_penny_logs(tmp_path):
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    incoming_vec = serialize_embedding([1.0, 0.0, 0.0])

    with sqlite3.connect(db.db_path) as conn:
        _seed_message(conn, "incoming", "hey penny", base, embedding=incoming_vec)
        _seed_message(conn, "outgoing", "hey back", base + timedelta(seconds=1))
        _seed_message(conn, "outgoing", "thinking about jazz", base + timedelta(seconds=2))
        conn.commit()

    migrate(db.db_path)

    # ``user-messages`` / ``penny-messages`` are read facades over ``messagelog``
    # (the 0027 memory_entry replica is dropped by 0059), so read them through the
    # facade.  A message has two authors — the user (incoming) or Penny (outgoing).
    user_messages = db.memory("user-messages")
    penny_messages = db.memory("penny-messages")
    assert user_messages is not None and penny_messages is not None
    user_rows = user_messages.read_all()
    penny_rows = penny_messages.read_all()

    assert [(e.content, e.author) for e in user_rows] == [("hey penny", "user")]
    assert [(e.content, e.author) for e in penny_rows] == [
        ("hey back", "penny"),
        ("thinking about jazz", "penny"),
    ]
    # The incoming message's embedding survives the facade (read from messagelog).
    assert user_rows[0].content_embedding == incoming_vec
