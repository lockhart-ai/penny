"""Collection skill provenance — a collection records the skill it was rendered
from + the bound params (#1603).

Type: schema

Part of the tool-surface hardening epic (#1554, mini-epic #1562): #1591's front
door renders a skill's steps into a collection's ``extraction_prompt`` but the row
kept no record of *which* skill made it or *with what*.  Two nullable columns close
that, so "which skill made this, and with what?" is a read off the collection's own
row and a future rebind/re-render has the current bindings as reachable input:

**``memory.skill_name``** — a by-name reference to ``skill.name`` (a plain column,
not a DB FK: a skill is re-teachable / REPLACE-able, so the reference must survive a
re-teach).  ``collection_create`` stamps it at instantiation.

**``memory.skill_params``** — the params bound into the skill's render, as a JSON
object.  NULL alongside ``skill_name`` for a hand-authored / seeded / migration
collection (no skill origin), so its catalog / metadata render is byte-identical to
the pre-provenance shape — the unmarked case stays the quiet default.

Schema only, universal (no deployment-specific rows).  A fresh ``create_tables()``
DB already carries these from the models (the guard below skips them); a production
copy that predates them gets them added.  No index — provenance is read on an
already-resolved row, never a filtered scan.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_skill_provenance_columns(conn)
    conn.commit()


def _add_skill_provenance_columns(conn: sqlite3.Connection) -> None:
    """Add ``skill_name`` + ``skill_params`` to ``memory`` (idempotent)."""
    if not _table_exists(conn, "memory"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "skill_name" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN skill_name TEXT")
    if "skill_params" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN skill_params TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
