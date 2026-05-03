"""Tests for migration 0027 — data migration into the memory framework.

Each block of the migration (messages, preferences, thoughts, knowledge)
gets a focused test that seeds the relevant old table(s), runs the
migration, and verifies the resulting ``memory_entry`` rows.  A separate
test pair confirms idempotency and the empty-target guard.
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
    thought_id: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messagelog"
        " (direction, sender, content, timestamp, is_reaction, processed,"
        "  embedding, thought_id)"
        " VALUES (?, '+15551234567', ?, ?, 0, 0, ?, ?)",
        (direction, content, timestamp.isoformat(), embedding, thought_id),
    )


def _seed_preference(
    conn: sqlite3.Connection,
    content: str,
    valence: str,
    created_at: datetime,
    embedding: bytes | None = None,
) -> None:
    conn.execute(
        "INSERT INTO preference"
        " (user, content, valence, embedding, created_at, mention_count, source)"
        " VALUES ('+15551234567', ?, ?, ?, ?, 1, 'extracted')",
        (content, valence, embedding, created_at.isoformat()),
    )


def _seed_thought(
    conn: sqlite3.Connection,
    content: str,
    title: str | None,
    notified_at: datetime | None,
    created_at: datetime,
    embedding: bytes | None = None,
    title_embedding: bytes | None = None,
) -> None:
    conn.execute(
        "INSERT INTO thought"
        " (user, content, title, notified_at, created_at, embedding, title_embedding)"
        " VALUES ('+15551234567', ?, ?, ?, ?, ?, ?)",
        (
            content,
            title,
            notified_at.isoformat() if notified_at else None,
            created_at.isoformat(),
            embedding,
            title_embedding,
        ),
    )


def _seed_knowledge(
    conn: sqlite3.Connection,
    url: str,
    title: str,
    summary: str,
    created_at: datetime,
    embedding: bytes | None = None,
) -> int:
    """Insert a knowledge row.  Requires a promptlog row first for FK."""
    cur = conn.execute(
        "INSERT INTO promptlog (timestamp, model, messages, response)"
        " VALUES (?, 'test-model', '[]', '{}')",
        (created_at.isoformat(),),
    )
    prompt_id = cur.lastrowid
    conn.execute(
        "INSERT INTO knowledge"
        " (url, title, summary, embedding, source_prompt_id, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, title, summary, embedding, prompt_id, created_at.isoformat(), created_at.isoformat()),
    )
    return prompt_id or 0


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
        _seed_message(
            conn, "outgoing", "thinking about jazz", base + timedelta(seconds=2), thought_id=None
        )
        # An outgoing message tied to a thought → author "notify"
        conn.execute(
            "INSERT INTO thought"
            " (user, content, title, created_at)"
            " VALUES ('+15551234567', 'jazz musings', 'jazz', ?)",
            (base.isoformat(),),
        )
        thought_id = conn.execute("SELECT MAX(id) FROM thought").fetchone()[0]
        _seed_message(
            conn,
            "outgoing",
            "did you see this jazz article?",
            base + timedelta(seconds=3),
            thought_id=thought_id,
        )
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        user_rows = _entries(conn, "user-messages")
        penny_rows = _entries(conn, "penny-messages")

    assert user_rows == [(None, "hey penny", "user", None, incoming_vec)]
    assert penny_rows == [
        (None, "hey back", "chat", None, None),
        (None, "thinking about jazz", "chat", None, None),
        (None, "did you see this jazz article?", "notify", None, None),
    ]


def test_preferences_split_by_valence(tmp_path):
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    coffee_vec = serialize_embedding([1.0, 0.0, 0.0])

    with sqlite3.connect(db.db_path) as conn:
        _seed_preference(conn, "dark roast coffee", "positive", base, embedding=coffee_vec)
        _seed_preference(conn, "country music", "negative", base + timedelta(seconds=1))
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        likes = _entries(conn, "likes")
        dislikes = _entries(conn, "dislikes")

    assert likes == [("dark roast coffee", "dark roast coffee", "history", None, coffee_vec)]
    assert dislikes == [("country music", "country music", "history", None, None)]


def test_thoughts_split_by_notified_status(tmp_path):
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    content_vec = serialize_embedding([1.0, 0.0, 0.0])
    title_vec = serialize_embedding([0.0, 1.0, 0.0])

    with sqlite3.connect(db.db_path) as conn:
        _seed_thought(
            conn,
            "pending insight",
            "Quantum Gravity",
            None,
            base,
            embedding=content_vec,
            title_embedding=title_vec,
        )
        _seed_thought(
            conn,
            "shared insight",
            "Black Holes",
            base + timedelta(hours=1),
            base + timedelta(seconds=1),
        )
        # No-title thought is skipped — collections require a key.
        _seed_thought(conn, "untitled musing", None, None, base + timedelta(seconds=2))
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        unnotified = _entries(conn, "unnotified-thoughts")
        notified = _entries(conn, "notified-thoughts")

    assert unnotified == [
        ("Quantum Gravity", "pending insight", "thinking", title_vec, content_vec),
    ]
    assert notified == [("Black Holes", "shared insight", "thinking", None, None)]


def test_knowledge_collection_populated_with_url_in_content(tmp_path):
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    knowledge_vec = serialize_embedding([1.0, 0.0, 0.0])

    with sqlite3.connect(db.db_path) as conn:
        _seed_knowledge(
            conn,
            "https://example.com/quantum-gravity",
            "Quantum Gravity Primer",
            "Loop quantum gravity reconciles GR with QM via spin networks.",
            base,
            embedding=knowledge_vec,
        )
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        rows = _entries(conn, "knowledge")

    expected_body = (
        "URL: https://example.com/quantum-gravity\n\n"
        "Loop quantum gravity reconciles GR with QM via spin networks."
    )
    assert rows == [("Quantum Gravity Primer", expected_body, "history", None, knowledge_vec)]


# ── Idempotency / skip-when-populated guards ──────────────────────────────


def test_running_migration_twice_does_not_duplicate_entries(tmp_path):
    """Each block guards on the target memory being empty, so re-running
    the migration after a manual revert is safe."""
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    with sqlite3.connect(db.db_path) as conn:
        _seed_message(conn, "incoming", "first", base)
        _seed_preference(conn, "tea", "positive", base)
        conn.commit()

    migrate(db.db_path)
    # Force-re-run by clearing the migration record so the runner re-applies.
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM _migrations WHERE name = '0027_memory_data_migration'")
        conn.commit()
    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        assert len(_entries(conn, "user-messages")) == 1
        assert len(_entries(conn, "likes")) == 1


def test_skips_block_when_target_memory_already_populated(tmp_path):
    """If the target memory already has entries, the migration leaves it alone."""
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    # Pre-seed an entry directly into likes (simulating a partial earlier run
    # or manual fix-up).  The migration should leave it intact and not append
    # the seeded preference row alongside it.
    with sqlite3.connect(db.db_path) as conn:
        _seed_preference(conn, "tea", "positive", base)
        conn.execute(
            "INSERT INTO memory"
            " (name, type, description, recall, archived, created_at, updated_at)"
            " VALUES ('likes', 'collection', 'x', 'relevant', 0, ?, ?)",
            (base.isoformat(), base.isoformat()),
        )
        conn.execute(
            "INSERT INTO memory_entry"
            " (memory_name, key, content, author, created_at)"
            " VALUES ('likes', 'pre-existing', 'pre-existing', 'manual', ?)",
            (base.isoformat(),),
        )
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        rows = _entries(conn, "likes")
    assert rows == [("pre-existing", "pre-existing", "manual", None, None)]
