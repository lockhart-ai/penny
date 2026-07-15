"""Retire the ``skills`` collection entirely — one skills store, the ``skill`` table.

Type: data

Epic #1554, issue #1624 (as amended by the code owner: "why would we keep two
skills stores? that doesn't make any sense").  The ``skills`` collection (seeded
0043, regrounded 0069) carried prose recipes on two duties: a reconcile COLLECTOR
that folded recipe improvements into its entries by model judgment, and a
standing-rules feed the self-state ``### Skills and rules`` section rendered
ambiently (#1621).  Both are superseded by the structural path — skills are
**taught** (``skill_create``, certified-by-execution, #1590), **instantiated**
(``collection_create`` renders a taught skill, #1591), **fired ambiently** (the
taught-skill feed of the same section), and **re-rendered**
(``collection_update(skill=…)``, #1620).  There is exactly ONE skills store now:
the ``skill`` table.  A second, prose one has no job — the same disease the
``quality`` retirement cured (0089, #1569) and the ``notifier`` tombstone before
it (0086).

1. **Archive the ``skills`` collection** (visible tombstone, the 0086/0089
   pattern).  An archived collection never dispatches, drops out of the
   self-state store map (non-archived only), and ``SYSTEM_COLLECTIONS`` keeps the
   shell hidden from ``collection_catalog``.  Its ``extraction_prompt`` is left
   intact (0089's quality precedent — an archived collection never runs, and the
   prompt is the tombstone's historical record).

2. **Delete ALL the migration-seeded rule entries** (known keys, 0043/0070/0076/
   0078/0079 lineage as amended by 0066/0069/0086) plus the reconcile loop's own
   ``author='skills'`` output (generic criteria, the 0069 precedent).  The seeded
   rules can never enter the ``skill`` table — it is certified-by-execution with
   no seeds (0084), and these were never demonstrated.  Anything actually needed
   is re-taught live as a real table skill.  A user's own chat-authored entry is
   deployment-specific runtime data this migration never targets; archival keeps
   any such rows readable.

Touches only universal data — the migration-seeded ``skills`` row + seeded entries
by known key + one generic-criteria scrub — so it is safe on every deployment and
idempotent (re-archiving and re-deleting are no-ops).
"""

from __future__ import annotations

import sqlite3

_SKILLS = "skills"

# Every migration-seeded rule entry, by its current known key: the 0043 originals
# (as rewritten by 0066/0069/0086; ``Scheduled digest`` already deleted by 0069,
# the 0077 schedule skills by 0082) plus the later operate-the-system seeds
# (0070 source-change, 0076 mute, 0078 email, 0079 likes/dislikes).
_SEEDED_RULE_KEYS = [
    "Research collection — notify on new finds",
    "Research collection — silent",
    "Browse for a one-shot question",
    "Update collection scope",
    "Flip silent ↔ notify",
    "Change collection cadence",
    "Archive a collection",
    "Change collection source",
    "Mute or unmute notifications",
    "Look up email",
    "Record a like or dislike",
    "Forget a like or dislike",
    "List likes or dislikes",
]


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    # 1. Archive the retired skills collection (visible tombstone, 0086/0089
    #    pattern) — never dispatched, hidden from the catalog, out of the store map.
    conn.execute("UPDATE memory SET archived = 1 WHERE name = ?", (_SKILLS,))

    # 2. Delete the seeded rule entries (known keys) + the reconcile loop's own
    #    minted entries (generic criteria, 0069 precedent).  Never demonstrated,
    #    so they cannot move into the certified-by-execution skill table; needed
    #    behaviors get re-taught live.
    for key in _SEEDED_RULE_KEYS:
        conn.execute(
            "DELETE FROM memory_entry WHERE memory_name = ? AND key = ?",
            (_SKILLS, key),
        )
    conn.execute(
        "DELETE FROM memory_entry WHERE memory_name = ? AND author = ?",
        (_SKILLS, _SKILLS),
    )

    conn.commit()
