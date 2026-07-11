"""Ledger provenance closure — entry run-id stamps + the mutation-event table (#1560).

Type: schema

Wave 1 of the tool-surface hardening epic (#1554), the event half of the
operational spine (the entity half is #1566's registry columns, already on
main).  The ledger is a *property* over the stores that already exist, not a new
event-sourcing supertable: a run is a ``promptlog`` group (``run_target`` already
records which entity it served), a tool call is a ``promptlog.response`` entry
(stored verbatim as ``raw.model_dump()`` — round-trippable, unchanged here).  This
migration adds the two pieces that had no home:

1. **Entry run-id stamps** — ``memory_entry.created_by_run_id`` +
   ``memory_entry.last_written_by_run_id``.  A collection entry is *rewritten*
   (a watch's baseline changes every cycle), so the current value's writer differs
   from its creator; the full write history is the ledger's ``collection_write``
   tool-call records, and these two stamps are the read-path anchors that make
   ``read_run_calls(run_id)`` one guess-free hop from wherever an entry renders.
   Stamped at write time by the writing run, threaded as a parameter (no ambient
   state).  Both nullable — migration-seeded / pre-#1560 entries carry neither.

2. **The ``mutation_event`` table** — one durable row per create / update /
   archive / unarchive of a registry entity, carrying (entity, run, actor, what
   changed, when).  The one ledger table with no other home: a *system* archive
   (the scheduler's ``max_runs`` / ``expires_at`` retire, #1556) runs no model and
   logs no prompt, so without this row it is invisible.  Audit + provenance over
   the materialized ``memory`` row, NOT event sourcing — ``memory.archived`` stays
   the truth.  Indexed on ``(entity_name, created_at)`` so a mechanism's change
   history reads in time order.

Schema only — touches no deployment-specific rows.  A fresh ``create_tables()`` DB
already carries these from the models, so the guards below skip them; a production
copy that predates them gets them added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_entry_stamps(conn)
    _create_mutation_event_table(conn)
    conn.commit()


def _add_entry_stamps(conn: sqlite3.Connection) -> None:
    """Add the two nullable run-id stamp columns to ``memory_entry`` (idempotent)."""
    if not _table_exists(conn, "memory_entry"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_entry)").fetchall()}
    if "created_by_run_id" not in columns:
        conn.execute("ALTER TABLE memory_entry ADD COLUMN created_by_run_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_memory_entry_created_by_run_id "
            "ON memory_entry(created_by_run_id)"
        )
    if "last_written_by_run_id" not in columns:
        conn.execute("ALTER TABLE memory_entry ADD COLUMN last_written_by_run_id TEXT")


def _create_mutation_event_table(conn: sqlite3.Connection) -> None:
    """Create the ``mutation_event`` audit table + its per-entity time index."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mutation_event ("
        "  id INTEGER PRIMARY KEY,"
        "  entity_type TEXT NOT NULL,"
        "  entity_name TEXT NOT NULL,"
        "  action TEXT NOT NULL,"
        "  actor TEXT NOT NULL,"
        "  run_id TEXT,"
        "  detail TEXT,"
        "  created_at TIMESTAMP NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_mutation_event_entity_time "
        "ON mutation_event(entity_name, created_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_mutation_event_run_id ON mutation_event(run_id)")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
