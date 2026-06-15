"""Rename the read tools in stored extraction prompts to the clean taxonomy.

Type: data

The model-facing read surface was made shape-consistent:
  * ``read_latest`` → ``collection_read_latest`` (collections only; it errors on
    a log now, so logs are read strictly through ``log_read`` (streams, no get) —
    no newest-first scan bypassing the cursor).
  * ``collection_metadata`` → ``memory_metadata`` (it serves both shapes, so the
    ``collection_`` prefix was misleading).

This rewrites every live ``memory.extraction_prompt`` that names the old tools.
Pure rename — same behaviour, since the renamed tools wrap the same access-layer
calls.  (The old migration files keep their original text as historical record;
only the current rows are rewritten.)
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'read_latest(', 'collection_read_latest(') "
        "WHERE extraction_prompt LIKE '%read_latest(%'"
    )
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'collection_metadata(', 'memory_metadata(') "
        "WHERE extraction_prompt LIKE '%collection_metadata(%'"
    )
    conn.commit()
