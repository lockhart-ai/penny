"""Drop the dead recall substrate: the ``inclusion`` + ``recall`` columns.

Type: schema + data

The ambient inversion (#1555 / PR #1581) removed speculative recall's *behaviour*
— the chat prompt no longer injects a recalled-content block; it opens with the
deterministic ``SelfStateHeader`` instead — and #1471 (PR #1621) re-homed
taught-skill / standing-rule firing onto that header's ``### Skills and rules``
section.  So the two-stage recall machinery (stage-1 ``inclusion`` routing,
stage-2 ``recall`` entry rendering, ``read_similar_hybrid`` + temporal-neighbour
expansion) has no remaining consumer.  This migration retires its schema (#1583).

This migration:

1. **Rewrites the migration-seeded skill entries that still teach the dropped
   flags** — the 0069-seeded research-notify / research-silent recipes carry an
   ``- inclusion: "relevant", recall: "relevant"`` line in their ``collection_create``
   step.  A surgical, idempotent substring removal over the ``skills`` collection
   (generic content shape, not a per-key list — the exact line is unique to these
   entries) so a standing rule can't teach a flag that no longer exists; the
   content embedding is nulled so the startup backfill re-vectorizes.  (The rest of
   those recipes is #1471's re-home to finish.)
2. **Drops the ``inclusion`` and ``recall`` columns** from ``memory``.  Guarded
   (present? drop) so it is safe whether the columns exist (prod / an upgraded DB)
   or not.

Universal — a migration-seeded content rewrite by generic shape + a column drop —
so it is safe on every deployment.  ``description_embedding`` STAYS (it now serves
resolve-by-meaning, #1558); ``notify`` STAYS (emission-as-property, #1557).
"""

from __future__ import annotations

import sqlite3

# The dropped-flag line in the 0069-seeded research skill entries (leading newline
# so the whole line is removed, leaving the surrounding steps intact).  Unique to
# those entries, so a generic REPLACE over the ``skills`` collection is exact.
_DROPPED_FLAG_LINE = '\n   - inclusion: "relevant", recall: "relevant"'


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    # 1. Strip the dropped-flag line from the seeded skill recipes (idempotent —
    #    a re-run finds nothing left to replace) before the columns disappear.
    conn.execute(
        "UPDATE memory_entry SET content = REPLACE(content, ?, ''), content_embedding = NULL "
        "WHERE memory_name = 'skills' AND content LIKE '%' || ? || '%'",
        (_DROPPED_FLAG_LINE, _DROPPED_FLAG_LINE),
    )

    # 2. Drop the retired recall columns (nothing reads inclusion/recall after #1583).
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "inclusion" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN inclusion")
    if "recall" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN recall")

    conn.commit()
