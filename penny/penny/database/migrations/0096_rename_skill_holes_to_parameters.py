"""Rename ``skill.holes`` → ``skill.parameters`` — parameters are SKILL-level (#1668).

Type: schema + data

'Hole' is PL jargon that leaked into the design (#1665 already swapped the
MODEL-FACING vocabulary to 'parameters'; #1668 completes the code — the internal
identifiers rename too).  The declared-parameter column renames, and the per-leaf
``SkillSubstitution`` inside each row's ``steps`` JSON renames its ``hole`` key to
``parameter`` (the field rename ripples into the serialized shape) — so a
production skill demonstrated before this migration keeps rendering after it,
rather than losing its parameter names (a substitution whose ``hole`` key the new
model ignores would render an empty ``{}`` placeholder).

Both changes are UNIVERSAL — a column rename (DDL) and a generic column-wide JSON
rewrite keyed on the OLD key's presence, never a deployment-specific skill name.
Guarded/idempotent: the column rename fires only when ``holes`` exists and
``parameters`` does not (on a fresh DB ``create_tables()`` materialises
``parameters`` from the model first, so the rename is a no-op there); the steps
rewrite only touches substitutions that still carry the old ``hole`` key.
"""

from __future__ import annotations

import json
import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "skill" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(skill)").fetchall()]
    if "holes" in columns and "parameters" not in columns:
        conn.execute("ALTER TABLE skill RENAME COLUMN holes TO parameters")
    _rename_substitution_key(conn)
    conn.commit()


def _rename_substitution_key(conn: sqlite3.Connection) -> None:
    """Rewrite every stored skill's ``steps`` JSON so each substitution's ``hole``
    key becomes ``parameter`` — the field rename carried through the serialized
    shape, so existing skills still render their parameter names after #1668."""
    rows = conn.execute("SELECT name, steps FROM skill").fetchall()
    for name, steps_json in rows:
        steps = json.loads(steps_json)
        changed = False
        for step in steps:
            for sub in step.get("substitutions", []):
                if "hole" in sub:
                    sub["parameter"] = sub.pop("hole")
                    changed = True
        if changed:
            conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
