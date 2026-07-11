"""Add provenance + lifecycle columns to ``memory`` (operational registry, #1566).

Type: schema

Makes the ``memory`` table the registry of operational entities: every
mechanism created from a chat request can answer *who* asked for it, *what*
run created it, and *when* it ends — by a read, not a reconstruction.

Three nullable columns (seeded / system rows have none, so all default NULL):

- ``source_message_id`` — FK to ``messagelog.id``: the user message that
  spawned this mechanism.
- ``created_by_run_id`` — the ``promptlog.run_id`` of the run that created it.
- ``expires_at`` — the mechanism's end condition (UTC datetime); consumed by
  #1562's lifecycle axis.  Declared ``TIMESTAMP`` to match the table's other
  datetime columns (``created_at`` / ``last_collected_at``), so a fresh
  ``create_tables()`` DB and a migrated production DB agree on affinity.

Schema only — touches no deployment-specific rows.  A fresh DB already carries
these columns from ``create_tables()`` (the model defines them), so the guard
below skips them; a production copy that predates the columns gets them added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "source_message_id" not in columns:
        conn.execute(
            "ALTER TABLE memory ADD COLUMN source_message_id INTEGER REFERENCES messagelog(id)"
        )
    if "created_by_run_id" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN created_by_run_id TEXT")
    if "expires_at" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN expires_at TIMESTAMP")
    conn.commit()
