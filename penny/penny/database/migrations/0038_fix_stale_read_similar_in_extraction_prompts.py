"""Fix stale tool names in extraction_prompts: replace old collection_read_similar
and log_read_similar with the unified read_similar.

Type: data

PR #1000 unified the shape-specific ``collection_read_similar`` and
``log_read_similar`` tools into a single shape-agnostic ``read_similar``
tool.  Any extraction_prompt authored before or shortly after that rename
(whether by the system seed migrations or by the user via chat) may still
reference the old names, causing the Collector to call a tool that no
longer exists and receive a "Tool not found" error.

Idempotent — rows that already use ``read_similar`` are unaffected by the
REPLACE call (the string just doesn't match).
"""

from __future__ import annotations

import sqlite3

_OLD_COLLECTION = "collection_read_similar"
_OLD_LOG = "log_read_similar"
_NEW = "read_similar"


def up(conn: sqlite3.Connection) -> None:
    """Replace stale tool names in all extraction_prompt rows."""
    conn.execute(
        "UPDATE memory "
        "SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD_COLLECTION, _NEW, f"%{_OLD_COLLECTION}%"),
    )
    conn.execute(
        "UPDATE memory "
        "SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD_LOG, _NEW, f"%{_OLD_LOG}%"),
    )
    conn.commit()
