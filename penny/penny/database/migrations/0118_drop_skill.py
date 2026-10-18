"""Drop the ``skill`` table — the registry it stored is retired (#2538).

Type: schema

The naive-jobs branch (#2455 / #2537) deleted the skill registry, the skill
distillation pipeline, and the conversation machine that produced and read
skills; nothing has written or read the ``skill`` table since (migration 0116
already dropped the ``memory.skill_name`` / ``memory.skill_params`` columns
that pointed at it), and the ``Skill`` model is removed alongside this
migration. Guarded (table present? then drop) so it is a no-op on a fresh
``create_tables``-first DB — the current models no longer declare the table, so
``create_tables`` never materialises it and this drop finds nothing to remove.
On an upgraded prod DB the table exists (empty, since nothing has written to it
on this branch) and is dropped.

Universal — a plain table drop, no data to preserve — so it is safe on every
deployment.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "skill" not in tables:
        return
    conn.execute("DROP TABLE skill")
    conn.commit()
