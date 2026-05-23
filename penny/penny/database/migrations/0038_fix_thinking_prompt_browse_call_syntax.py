"""Fix thinking prompt: replace bare 'browse' label with explicit call syntax.

Type: data

Migration 0033 wrote _THINKING_PROMPT with step 3 as a bare label:
  "3. browse — search the web..."

All other steps in that prompt use explicit call syntax (e.g.
collection_read_random("likes", 1), done()).  The bare label caused the LLM
to hallucinate "search" as the tool name instead of the registered "browse".

This migration patches rows that still contain the old wording.
"""

from __future__ import annotations

import sqlite3

_OLD = "3. browse — search the web and read one or two pages to find "
_NEW = '3. browse(queries=["<seed topic>"]) — search the web and read one or two pages to find '


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'unnotified-thoughts' AND extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()
