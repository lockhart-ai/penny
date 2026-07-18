"""Nuke the generic catch-all seeded collections ENTIRELY — no tombstones.

Type: data

Issue #1676 (epic #1554 / #1570).  Code-owner decision, quoted: "those were all
seeded to automate the discovery of new information without the user's explicit
direction because we didn't have a good way for the user to direct the collection
of information, but now we do [the skills/teach loop] and so those generic
catch-alls only get in the way.  now the model can reason about creating
topic-specific collections so that's what we'll do instead."

Eight generic catch-all collections are removed **entirely** — their rows, their
entries, and the read cursors they own — **not** archived as tombstones (unlike
the 0086/0089/0092 retirements, which left a visible archived shell).  "No
tombstones even" applies to the already-archived shells too (``notifier`` /
``quality`` / ``skills`` and the retired ``unnotified-thoughts`` /
``notified-thoughts`` pair): a generic catch-all only gets in the way, archived or
not.

What deliberately STAYS:
  * ``dislikes`` — its collector + entries (code owner: "very narrow and specific
    — still holds water").
  * All four logs — ``user-messages`` / ``penny-messages`` / ``browse-results`` /
    ``collector-runs``.
  * ``messagelog`` / ``mutation_event`` / ``send_queue`` / ``thought`` /
    ``preference`` (history; the ``preference`` table's fate is the separate #1301
    legacy) — untouched.

Every name below is a MIGRATION-SEEDED row referenced by its known key, so this is
universal (present identically on every deployment) and safe per the house
migration rules; a user's own chat-created collection is never in this set.  The
three deletes are idempotent — a re-run deletes nothing.  Per-table counts are
logged so the removal is diagnosable, never silent.

The removal set is ONE module-level constant, so extending it (should the code
owner confirm more names) is a one-line change.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# The generic catch-all seeded collections, by known key.  ONE constant consumed
# by all three deletes below (memory_entry, agent_cursor, memory).
REMOVED_COLLECTIONS = (
    "likes",
    "knowledge",
    "thoughts",
    "notifier",
    "quality",
    "unnotified-thoughts",
    "notified-thoughts",
    "skills",
)


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    placeholders = ", ".join("?" for _ in REMOVED_COLLECTIONS)

    # 1. Entries — every stored row scoped to a removed collection.
    entries = conn.execute(
        f"DELETE FROM memory_entry WHERE memory_name IN ({placeholders})",
        REMOVED_COLLECTIONS,
    ).rowcount

    # 2. Read cursors — these collections OWN read cursors into the logs (e.g.
    #    likes/knowledge cursor into user-messages/browse-results).  The cursor
    #    reader is the bound collection name, so match either side of the
    #    ``(agent_name, memory_name)`` pair to catch cursors the collection owns
    #    AND any cursor pointed AT one of these (defensive; there are none today).
    cursors = 0
    if "agent_cursor" in tables:
        cursors = conn.execute(
            f"DELETE FROM agent_cursor "
            f"WHERE agent_name IN ({placeholders}) OR memory_name IN ({placeholders})",
            REMOVED_COLLECTIONS + REMOVED_COLLECTIONS,
        ).rowcount

    # 3. The collection rows themselves.
    memories = conn.execute(
        f"DELETE FROM memory WHERE name IN ({placeholders})",
        REMOVED_COLLECTIONS,
    ).rowcount

    conn.commit()

    logger.info(
        "0097 nuked generic seeded collections %s: %d memory rows, %d entries, %d cursors deleted",
        list(REMOVED_COLLECTIONS),
        memories,
        entries,
        cursors,
    )
