"""Rewrite the seeded thoughts prompt's browse call to the items shape.

Type: data

Browse takes items now (#2718): one call runs the search, picks a result with
the decision model and reads it, and each item carries its own "looking for".
Migration 0068 seeded the unified ``thoughts`` collection's extraction_prompt
teaching step 3 as ``browse(queries=["<seed topic>"])`` — a shape the tool's arg
gate rejects (queries are gone; each item needs its ``looking_for``).  The
prompt is seeded data created by an earlier migration, referenced by its known
key, so rewriting it here is universal — every fresh database has this row.

Only the call syntax changes; the step's instruction stays what it was.
"""

from __future__ import annotations

import sqlite3

_OLD = '3. browse(queries=["<seed topic>"]) — search the web and read one or two pages to find '
_NEW = (
    '3. browse(items=[{"query": "<seed topic>", "looking_for": "something timely and '
    'interesting grounded in the seed topic"}]) — search the web and read one or two pages to find '
)


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'thoughts' AND extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()
