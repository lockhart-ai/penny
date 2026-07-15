"""Drop the ``memory.intent`` column — the surface collapse absorbs it into
``description`` (epic #1554, issue #1631).

Type: schema

``intent`` (added by migration 0050) recorded the user's stated goal for a
collection.  In practice it duplicated ``description`` byte-for-byte at birth —
``collection_create`` passed the same value to both — and its one distinct role,
the immutable anchor for the retired quality collector, is gone.  The minimal
collection surface makes ``description`` the single descriptor everywhere
(routing / dedup meaning anchor, #1558), so ``intent`` is removed: the
``collection_create`` arg, ``collection_update``'s accept-but-ignore ceremony
(migration 0050's protection dance), every ``intent:`` render line, and now the
column.

Guarded (present? drop) so it is safe whether the column exists (prod / an
upgraded DB) or not.  No fresh-DB re-provisioning is needed on the
``create_tables``-first path (unlike #1628's recall drop): migration 0050's own
guarded ``ADD COLUMN intent`` re-adds it before this drop removes it, and no
migration between 0050 and here reads the column in a way a fresh replay would
miss.  ``description`` (+ ``description_embedding``, #1558) stays.

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
    if "intent" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN intent")
    conn.commit()
