"""Retire the ``skills`` reconcile collector; prune superseded seeded build-recipes.

Type: data

Epic #1554, issue #1624.  The ``skills`` background collector (seeded 0043,
regrounded 0069) read ``collection_catalog`` each cycle and folded recipe
improvements into prose ``skills`` entries by model judgment.  That job is gone:
skills are now STRUCTURAL — taught (``skill_create``, certified by execution,
#1590), instantiated (``collection_create`` renders a taught skill, #1591), fired
ambiently (the self-state ``### Skills and rules`` section, #1621), and
re-rendered (``collection_update(skill=…)``, #1620).  A model-judgment loop
maintaining prose recipes has no prose-generation step left to review — the same
disease the ``quality`` retirement cured (0089, #1569) and the ``notifier``
tombstone before it (0086).

Unlike ``quality`` / ``notifier``, the ``skills`` COLLECTION is NOT archived: the
self-state ``### Skills and rules`` section reads its ENTRIES (the standing-rule
store, #1471), which must keep rendering.  So this retires the COLLECTOR while the
COLLECTION stays active:

1. **Clear the ``skills`` collector's dispatch fields** — ``extraction_prompt``
   plus the cadence columns (``collector_interval_seconds`` /
   ``base_interval_seconds`` / ``consecutive_idle_runs``).  The dispatcher gates on
   ``extraction_prompt IS NOT NULL`` (``Collector._is_ready``), so a NULL prompt
   means it never runs again, and the self-state Active-mechanisms section (which
   renders only rows WITH an ``extraction_prompt``) drops it.  The row stays
   ``archived = 0`` — an active store whose entries still render ambiently and
   whose line still appears in the store map.

2. **Prune the superseded seeded build-recipes** — the ``Research collection —
   notify on new finds`` / ``Research collection — silent`` entries (0069 lineage,
   known keys; 0086 rewrote only their bodies, ``published`` → ``notify``, leaving
   the keys unchanged).  They teach the model to hand-author a
   ``collection_create`` with a numbered ``extraction_prompt`` — the path the
   structural front door (#1591, instantiate-a-skill) replaced — so rendered
   ambiently they actively teach the WRONG path.  The operate-the-system rules
   (flip, cadence, archive, update-scope, one-shot, mute, email, likes/dislikes)
   STAY — they are live standing rules for existing collections and the chat tool
   surface.

Touches only universal data — the migration-seeded ``skills`` row + two seeded
entries by known key — so it is safe on every deployment and idempotent
(re-running clears already-cleared fields and deletes already-deleted keys).  A
user's own chat-authored ``skills`` entry is deployment-specific runtime data this
migration never targets.
"""

from __future__ import annotations

import sqlite3

_SKILLS = "skills"

# The superseded build-recipes — hand-authoring a collection with a numbered
# ``extraction_prompt``, the path #1591's structural front door replaced.  0069
# seeded these keys; 0086 rewrote only their bodies (``published`` → ``notify``),
# so the keys are unchanged.  The operate-the-system rules are deliberately NOT
# listed here — they teach live operations and stay.
_SUPERSEDED_BUILD_RECIPE_KEYS = [
    "Research collection — notify on new finds",
    "Research collection — silent",
]


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    # 1. Retire the COLLECTOR: clear the dispatch fields so it never runs again,
    #    while the COLLECTION stays active (archived = 0) — its entries still
    #    render in the self-state ``### Skills and rules`` section.
    conn.execute(
        "UPDATE memory SET extraction_prompt = NULL, collector_interval_seconds = NULL, "
        "base_interval_seconds = NULL, consecutive_idle_runs = 0 WHERE name = ?",
        (_SKILLS,),
    )

    # 2. Prune the superseded build-recipes (operate-the-system rules stay).
    for key in _SUPERSEDED_BUILD_RECIPE_KEYS:
        conn.execute(
            "DELETE FROM memory_entry WHERE memory_name = ? AND key = ?",
            (_SKILLS, key),
        )

    conn.commit()
