"""The skill substrate table (#1590).

Type: schema

Stage ④ of the collector core (#1562 ⭐ rev. 3; epic #1554).  Adds the dedicated
``skill`` table — versionless, one row per name (a re-teach REPLACES the row;
collections carry the rendered TEXT snapshotted at creation, so a skill edit never
retroactively changes an instantiation, and the version pin had no remaining job).
A skill's ``steps`` are the ``LoggedToolCall`` shape (#1578) as JSON, its ``holes``
the declared parameters; #1591's ``collection_create`` renders steps + bound params
into a collection's numbered TEXT ``extraction_prompt``.

Table only — no seeded rows (code-owner decision on the #1599 revision): every
skill enters through the real ``skill_create`` teaching flow, so the
certified-by-execution invariant holds universally, with no hand-authored
exception.  A fresh install starts with an honestly-empty skill registry.

Idempotent (``CREATE TABLE IF NOT EXISTS`` — ``create_tables()`` runs first on a
fresh DB and materialises the table from the model, so this is a no-op there and
an add on a production copy that predates it).
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS skill ("
        "  name TEXT PRIMARY KEY,"
        "  steps TEXT NOT NULL,"
        "  holes TEXT NOT NULL,"
        "  intent TEXT NOT NULL,"
        "  description TEXT NOT NULL,"
        "  description_embedding BLOB,"
        "  source_run_id TEXT,"
        "  author TEXT NOT NULL,"
        "  created_at TIMESTAMP NOT NULL,"
        "  updated_at TIMESTAMP NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_author ON skill(author)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_source_run_id ON skill(source_run_id)")
    conn.commit()
