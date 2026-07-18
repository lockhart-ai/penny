"""Nuke the generic catch-all seeded collections ENTIRELY — no tombstones.

Type: data

Issue #1676 (epic #1554 / #1570).  Code-owner decision, quoted: "those were all
seeded to automate the discovery of new information without the user's explicit
direction because we didn't have a good way for the user to direct the collection
of information, but now we do [the skills/teach loop] and so those generic
catch-alls only get in the way.  now the model can reason about creating
topic-specific collections so that's what we'll do instead."

Eight generic catch-all collections are removed **entirely** — their rows, their
entries, and the read cursors they own — **not** archived as tombstones (unlike
the 0086/0089/0092 retirements, which left a visible archived shell).  "No
tombstones even" applies to the already-archived shells too (``notifier`` /
``quality`` / ``skills`` and the retired ``unnotified-thoughts`` /
``notified-thoughts`` pair): a generic catch-all only gets in the way, archived or
not.

Scope extension (code owner: "let's add thought and preference to the deletion"):
the legacy ``thought`` and ``preference`` TABLES drop too.  They were the
pre-memory-framework stores the thoughts pipeline ran on — 0027 long ago migrated
their contents into the collections this migration deletes, and nuking that
pipeline killed their last real consumers (the startup preference-embedding
backfill and the profile onboarding check, both removed with this migration).
``messagelog.thought_id`` — the FK into the dropped ``thought`` table — drops
with them (its index first; historical rows lose only that link).  Row counts
are logged before each drop.

What deliberately STAYS:
  * ``dislikes`` — its collector + entries (code owner: "very narrow and specific
    — still holds water").
  * All four logs — ``user-messages`` / ``penny-messages`` / ``browse-results`` /
    ``collector-runs``.
  * ``messagelog`` / ``mutation_event`` / ``send_queue`` (history) — untouched
    beyond the ``thought_id`` column drop.

Every name below is a MIGRATION-SEEDED row referenced by its known key, so this is
universal (present identically on every deployment) and safe per the house
migration rules; a user's own chat-created collection is never in this set.  The
three deletes are idempotent — a re-run deletes nothing.  Per-table counts are
logged so the removal is diagnosable, never silent.

The removal set is ONE module-level constant, so extending it (should the code
owner confirm more names) is a one-line change.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# The generic catch-all seeded collections, by known key.  ONE constant consumed
# by all three deletes below (memory_entry, agent_cursor, memory).
REMOVED_COLLECTIONS = (
    "likes",
    "knowledge",
    "thoughts",
    "notifier",
    "quality",
    "unnotified-thoughts",
    "notified-thoughts",
    "skills",
)


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" in tables:
        _delete_collections(conn, tables)
    _drop_legacy_thought_preference(conn, tables)
    conn.commit()


def _delete_collections(conn: sqlite3.Connection, tables: set[str]) -> None:
    placeholders = ", ".join("?" for _ in REMOVED_COLLECTIONS)

    # 1. Entries — every stored row scoped to a removed collection.
    entries = conn.execute(
        f"DELETE FROM memory_entry WHERE memory_name IN ({placeholders})",
        REMOVED_COLLECTIONS,
    ).rowcount

    # 2. Read cursors — these collections OWN read cursors into the logs (e.g.
    #    likes/knowledge cursor into user-messages/browse-results).  The cursor
    #    reader is the bound collection name, so match either side of the
    #    ``(agent_name, memory_name)`` pair to catch cursors the collection owns
    #    AND any cursor pointed AT one of these (defensive; there are none today).
    cursors = 0
    if "agent_cursor" in tables:
        cursors = conn.execute(
            f"DELETE FROM agent_cursor "
            f"WHERE agent_name IN ({placeholders}) OR memory_name IN ({placeholders})",
            REMOVED_COLLECTIONS + REMOVED_COLLECTIONS,
        ).rowcount

    # 3. The collection rows themselves.
    memories = conn.execute(
        f"DELETE FROM memory WHERE name IN ({placeholders})",
        REMOVED_COLLECTIONS,
    ).rowcount

    logger.info(
        "0097 nuked generic seeded collections %s: %d memory rows, %d entries, %d cursors deleted",
        list(REMOVED_COLLECTIONS),
        memories,
        entries,
        cursors,
    )


def _drop_legacy_thought_preference(conn: sqlite3.Connection, tables: set[str]) -> None:
    """Drop the legacy pre-memory-framework tables + the FK column into them.

    Row counts are logged BEFORE each drop — the deletion is diagnosable, never
    silent.  Guarded on existence at every step, so a re-run (or a fresh DB whose
    ``create_tables`` no longer materialises these models) is a no-op.
    """
    # The FK column first: its index blocks SQLite's DROP COLUMN, so the index
    # (created by 0006) goes, then the column.  Historical rows lose only this
    # link — the messages themselves are untouched.
    if "messagelog" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
        if "thought_id" in columns:
            conn.execute("DROP INDEX IF EXISTS ix_messagelog_thought_id")
            conn.execute("ALTER TABLE messagelog DROP COLUMN thought_id")
            logger.info("0097 dropped messagelog.thought_id (the FK into the dropped tables)")

    for table in ("thought", "preference"):
        if table not in tables:
            continue
        rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DROP TABLE {table}")
        logger.info("0097 dropped legacy table %r (%d rows)", table, rows)
