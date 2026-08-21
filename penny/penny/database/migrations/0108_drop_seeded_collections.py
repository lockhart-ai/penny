"""Drop the last migration-seeded collection — no pre-seeded collections remain.

Type: data

Issue #1911.  Code-owner ruling, quoted: "i don't wanna even leave any intermediate
legacy structures around since everyone's just gonna reset their db so just drop any
pre seeded collections for the time being and we can add new ones later if we want.
this is kind of a soft reboot of penny."

Migration 0097 (#1676) already removed eight generic catch-alls entirely — ``likes``,
``knowledge``, ``thoughts``, ``notifier``, ``quality``, ``unnotified-thoughts``,
``notified-thoughts``, ``skills`` — leaving ``dislikes`` as the one deliberate
survivor ("very narrow and specific — still holds water").  The soft reboot retires
that exemption: a seeded collection is a legacy structure whether or not it is a good
one, and Penny now builds her collections from what the user teaches her.

So this is 0097's tail, not a new mechanism: the same three deletes (entries, the read
cursors the collection owns into the logs, then the row), the same idempotence (a
re-run deletes nothing), and the same logged counts so the removal is diagnosable
rather than silent.

DROPPED, not archived.  An archived shell is exactly the "intermediate legacy
structure" the ruling names, and 0097 already set that precedent for this family.

What deliberately STAYS:
  * All four logs — ``user-messages`` / ``penny-messages`` / ``browse-results`` /
    ``collector-runs``.  They are populated by Python side-effects and are how Penny
    perceives her own history; they are not collections a user would rebuild.
  * ``messagelog`` / ``mutation_event`` / ``send_queue`` (history) — untouched.
    History is never rewritten, which is also why ``PennyConstants``
    ``MEMORY_NOTIFIER_COLLECTION`` survives in code: it classifies historical
    notifier-sent message rows on the iOS surface, and names no live row.
  * Anything the USER created.  Every name below is a MIGRATION-SEEDED row referenced
    by its known key, so this is universal (present identically on every deployment)
    and safe per the house migration rules; a chat-created collection is never in this
    set.

The removal set is ONE module-level constant, mirroring 0097, so a name the code owner
adds later is a one-line change.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# The last migration-seeded collection, by known key.  ONE constant consumed by all
# three deletes below (memory_entry, agent_cursor, memory) — 0097's shape, kept so the
# two migrations read as the one decision they are.
REMOVED_COLLECTIONS = ("dislikes",)


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

    # 2. Read cursors — a seeded extractor OWNS a cursor into the log it reads
    #    (``dislikes`` into ``user-messages``).  The cursor's reader is the bound
    #    collection name, so match either side of the ``(agent_name, memory_name)``
    #    pair: the cursors it owns AND any pointed AT it (defensive; there are none).
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
        "0108 dropped the last seeded collections %s: %d memory rows, %d entries, "
        "%d cursors deleted",
        list(REMOVED_COLLECTIONS),
        memories,
        entries,
        cursors,
    )
