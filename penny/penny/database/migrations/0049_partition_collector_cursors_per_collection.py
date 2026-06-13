"""Partition collector read-cursors per collection instead of per dispatcher.

The dispatcher drove every collection under one ``agent_name`` ("collector"),
and ``agent_cursor`` keys on ``(agent_name, memory_name)`` — so every
collection that reads the same log (the 11 that read ``user-messages``, plus
``knowledge`` reading ``browse-results``) shared a single cursor row.
Whichever collection ran first consumed the new entries and advanced the
shared cursor; the rest saw "(no entries)" and no-op'd.  The code fix keys the
cursor on the bound collection name; this migration seeds the new
per-collection rows so nothing re-processes history on the first post-fix run.

Seeding rule (per the "set them all to current, catch up forward" decision):
for each existing ``(collector, <log>)`` cursor and each collection whose
``extraction_prompt`` references ``<log>``, copy the shared cursor's
``last_read_at`` to a new ``(collection, <log>)`` row.  Collections that read a
log with no existing shared cursor are left unseeded — their first read
bootstraps from the most-recent-N path, which is the intended new-collection
behaviour.

Then drop the now-dead rows: the old ``(collector, *)`` cursors (superseded by
the per-collection rows) and the pre-dispatcher ``knowledge-extractor`` /
``preference-extractor`` cursors (those agent classes no longer exist).

Idempotent: re-running finds no ``(collector, *)`` rows to seed from and the
per-collection rows already present, so it's a no-op on the second pass.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "agent_cursor" not in tables or "memory" not in tables:
        return
    _seed_per_collection_cursors(conn)
    _drop_dead_cursors(conn)
    conn.commit()


def _seed_per_collection_cursors(conn: sqlite3.Connection) -> None:
    """Copy each shared ``(collector, log)`` cursor onto every collection that reads that log."""
    shared = conn.execute(
        "SELECT memory_name, last_read_at, updated_at "
        "FROM agent_cursor WHERE agent_name = 'collector'"
    ).fetchall()
    collections = conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
    ).fetchall()
    for log_name, last_read_at, updated_at in shared:
        for name, extraction_prompt in collections:
            if name == log_name or log_name not in (extraction_prompt or ""):
                continue
            _insert_if_absent(conn, name, log_name, last_read_at, updated_at)


def _insert_if_absent(
    conn: sqlite3.Connection,
    agent_name: str,
    memory_name: str,
    last_read_at: str,
    updated_at: str,
) -> None:
    """Add a cursor row only if one doesn't already exist (keeps the migration idempotent)."""
    exists = conn.execute(
        "SELECT 1 FROM agent_cursor WHERE agent_name = ? AND memory_name = ?",
        (agent_name, memory_name),
    ).fetchone()
    if exists:
        return
    conn.execute(
        "INSERT INTO agent_cursor (agent_name, memory_name, last_read_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (agent_name, memory_name, last_read_at, updated_at),
    )


def _drop_dead_cursors(conn: sqlite3.Connection) -> None:
    """Remove cursors no agent reads anymore: the shared dispatcher rows + pre-dispatcher agents."""
    conn.execute(
        "DELETE FROM agent_cursor WHERE agent_name IN "
        "('collector', 'knowledge-extractor', 'preference-extractor')"
    )
