"""Seed the ``rules`` collection — standing rules the user gives Penny (#2633).

A standing rule is a thing the user asked Penny to always or never do ("remember
to always X", "never do Y").  It is an ordinary entry: "remember to always X" is
one ``collection_write(collection="rules", key=<short name>, content=<the rule
in the user's words>)`` and "forget the rule about X" is one
``collection_remove`` — no new tools, no schedule, no extraction prompt.

The row is seeded so the collection exists on every deployment from the first
turn: the ``### Standing rules`` section (``agents/self_state.py``) renders its
entries in full wherever Penny acts — the chat self-state header, every
collector cycle's system prompt, and the notify composer's document — and its
empty form names this collection, so the anchor is on screen whether or not a
rule exists yet.  Nothing else seeds an entry: the rules begin empty, which is
the honest state.

A seeded row with a known key is universal, so the migration is allowed
(CLAUDE.md → Database Migrations).  Idempotent — INSERT OR IGNORE on the
collection row.  (The name is a constant, ``PennyConstants.MEMORY_RULES_COLLECTION``;
migrations stay standalone, so it is spelled here as well.)
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

RULES_NAME = "rules"

RULES_DESCRIPTION = "Standing instructions the user has given: things to always or never do"


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory (name, type, description, archived, created_at) "
        "VALUES (?, 'collection', ?, 0, ?)",
        (RULES_NAME, RULES_DESCRIPTION, now),
    )
    conn.commit()
