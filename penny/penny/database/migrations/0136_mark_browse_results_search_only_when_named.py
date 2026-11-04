"""Mark ``browse-results`` as searched only when named (#2956).

Type: schema + data

Adds ``memory.search_only_when_named`` (default 0) — the structural mark
``find``'s and ``synthesize``'s shared retrieval reads to exclude a memory's
entries from an UNNAMED search (``MemoryStore.resolve_objects``); naming the
memory directly (``find(memory=...)``, ``synthesize(memories=[...])``) still
reaches it. Then sets it on the one row every deployment already has by this
exact seeded key (migration 0026): the ``browse-results`` log. A raw browse
page embeds very close to a question about the same topic, so unmarked it
can crowd out the curated collection built FROM those pages — the retrieval
spread (#2956's other half) only helps once the dominant source is excluded
by default.

Universal: the column is schema on a table every deployment has; the data
update targets one row by its known, migration-seeded name, not anything a
user created at runtime.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "memory"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "search_only_when_named" not in columns:
        conn.execute(
            "ALTER TABLE memory ADD COLUMN search_only_when_named INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute("UPDATE memory SET search_only_when_named = 1 WHERE name = 'browse-results'")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
