"""Fix extraction prompts that reference the non-existent tool 'read_last'.

Type: data

The correct tool name is 'read_latest'.  User-created or LLM-generated
extraction prompts may contain 'read_last(' instead, causing the Collector
agent to produce a "Tool not found: read_last" error on every cycle.

This migration replaces every occurrence of 'read_last(' with 'read_latest('
across all non-NULL extraction_prompt values.  The parenthesis anchor ensures
only the tool call form is rewritten and not any prose description that happens
to use the phrase 'read last'.
"""

from __future__ import annotations

import sqlite3

_OLD = "read_last("
_NEW = "read_latest("


def up(conn: sqlite3.Connection) -> None:
    """Replace read_last( with read_latest( in all extraction prompts."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()
