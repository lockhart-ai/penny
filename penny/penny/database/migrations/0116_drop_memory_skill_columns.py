"""Drop the ``memory.skill_name`` / ``memory.skill_params`` columns — the skill
registry they pointed at is retired (#2538).

Type: schema

The naive-jobs branch (#2455 / #2537) deleted the skill registry, the skill
distillation pipeline, and the conversation machine that gated on them; nothing
has written these columns since, and the store code that read and rendered them
is removed alongside this migration. Guarded (table present? column present?
then drop) so it is a no-op on a fresh ``create_tables``-first DB — the current
``MemoryRow`` model no longer declares either column, so ``create_tables`` never
materialises them and this drop finds nothing to remove.  On an upgraded prod DB
both columns exist and are dropped.

Universal — a plain column drop — so it is safe on every deployment.
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
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "skill_name" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN skill_name")
    if "skill_params" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN skill_params")
    conn.commit()
