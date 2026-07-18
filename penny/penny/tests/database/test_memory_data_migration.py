"""Tests for migration 0027 — data migration into the memory framework.

Each surviving block of the migration (messages → the user/penny facades,
preferences → ``dislikes``) gets a focused test that seeds the relevant old
table(s), runs the FULL migration chain, and verifies the resulting rows.  A
separate test pair confirms idempotency and the empty-target guard.

The thoughts/knowledge/likes blocks of 0027 still run, but migration 0097 (#1676)
nukes those generic catch-all collections entirely downstream — so their
end-of-chain state is verified by ``test_0097_nukes_generic_seeded_collections``
in ``test_migrations.py``, and the surviving witness of 0027's split here is
``dislikes`` (narrow + specific, deliberately retained).

The legacy ``preference`` table 0027 reads from is itself dropped by 0097 (and its
model removed, so ``create_tables`` no longer materialises it) — the seeding
helper recreates it as it existed pre-0097, which is honest: 0027's input IS a
legacy table from an old deployment.
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

    # 0027 splits by valence into likes/dislikes; migration 0097 (#1676) then nukes
    # the generic ``likes`` catch-all, leaving ``dislikes`` (narrow + specific) as
    # the surviving preference collection at end-of-chain.  (``coffee_vec`` still
    # rides the positive→likes path 0027 exercises; it just doesn't survive 0097.)
    assert likes == []
    assert dislikes == [("country music", "country music", "history", None, None)]


# ── Idempotency / skip-when-populated guards ──────────────────────────────


def test_running_migration_twice_does_not_duplicate_entries(tmp_path):
    """Each block guards on the target memory being empty, so re-running
    the migration after a manual revert is safe."""
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    with sqlite3.connect(db.db_path) as conn:
        _seed_message(conn, "incoming", "first", base)
        # A NEGATIVE preference lands in ``dislikes`` — the surviving preference
        # collection after migration 0097 (#1676) nukes the ``likes`` catch-all.
        _seed_preference(conn, "tea", "negative", base)
        conn.commit()

    migrate(db.db_path)
    # Force-re-run 0027 by clearing its migration record so the runner re-applies
    # it.  0027 reads columns later migrations have since DROPPED — it seeds into
    # the legacy ``recall`` column (dropped by 0091, #1583) and selects
    # ``messagelog.thought_id`` (dropped by 0097, #1676) — so re-provision both
    # (matching the pre-drop window) so the isolated re-run exercises 0027's
    # block-empty guards, the point of this test.
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM _migrations WHERE name = '0027_memory_data_migration'")
        conn.execute("ALTER TABLE memory ADD COLUMN recall TEXT NOT NULL DEFAULT 'recent'")
        conn.execute("ALTER TABLE messagelog ADD COLUMN thought_id INTEGER")
        conn.commit()
    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        # dislikes survives 0097, so the re-run hits 0027's populated-target SKIP
        # guard (not a re-create) — proving the block is idempotent, no duplicate.
        assert len(_entries(conn, "user-messages")) == 1
        assert len(_entries(conn, "dislikes")) == 1


def test_skips_block_when_target_memory_already_populated(tmp_path):
    """If the target memory already has entries, the migration leaves it alone."""
    db = _make_db(tmp_path)
    base = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    # Pre-seed an entry directly into dislikes (simulating a partial earlier run
    # or manual fix-up).  The migration should leave it intact and not append
    # the seeded preference row alongside it.  ``dislikes`` is the surviving
    # preference collection after migration 0097 (#1676) — a NEGATIVE preference
    # targets it.
    with sqlite3.connect(db.db_path) as conn:
        _seed_preference(conn, "tea", "negative", base)
        conn.execute(
            "INSERT INTO memory"
            " (name, type, description, archived, created_at, updated_at)"
            " VALUES ('dislikes', 'collection', 'x', 0, ?, ?)",
            (base.isoformat(), base.isoformat()),
        )
        conn.execute(
            "INSERT INTO memory_entry"
            " (memory_name, key, content, author, created_at)"
            " VALUES ('dislikes', 'pre-existing', 'pre-existing', 'manual', ?)",
            (base.isoformat(),),
        )
        conn.commit()

    migrate(db.db_path)

    with sqlite3.connect(db.db_path) as conn:
        rows = _entries(conn, "dislikes")
    assert rows == [("pre-existing", "pre-existing", "manual", None, None)]
