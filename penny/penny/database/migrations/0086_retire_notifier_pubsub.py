"""Retire the notifier consumer + the ``published`` pub/sub side-channel.

Type: schema + data

Emission is now a collection PROPERTY (#1557): a collection's ``notify`` flag
(migration 0085) drives run-time notify steps appended to its collector's
composed prompt, so the collector tells the user about a new/changed find in
the same cycle that produced it.  That replaces the whole pub/sub layer —
the ``notifier`` consumer that drained ``published`` collections via
``read_published_latest`` is gone, and so is the ``published`` column it keyed on
(#1583 assigned that column's death here).

This migration:

1. **Archives the ``notifier`` collection** — the migration-seeded consumer (0067,
   known key, universal).  Archived (not deleted) so the row stays a visible
   tombstone in the archived-inclusive catalog; ``notify`` supersedes it, and its
   ``read_published_latest`` step no longer names a live tool, so it must not
   dispatch.
2. **Drops the ``published`` column** from ``memory``.  ``notify`` (seeded from
   ``published`` by 0085) is the sole emission flag now; nothing reads
   ``published`` after this PR.
3. **Rewrites the migration-seeded skill entries that teach ``published``** — the
   research-notify / research-silent / flip recipes (0069, known keys) and the
   skills collector's own prompt — to teach ``notify`` instead, so a re-homed
   skill (#1471) can't teach a dropped flag.  Surgical substring replacements over
   the seeded text: they preserve any runtime refinement to the rest of the body
   and are naturally idempotent (a re-run finds nothing left to replace).  #1471
   owns the full re-home of these dark skills onto a firing channel.

Touches only universal data — a migration-seeded collection + skill rows by known
key, and a column drop — so it is safe on every deployment.  A fresh DB carries
the 0069-seeded text verbatim, so the eval (which runs migrations) validates
exactly what ships.
"""

from __future__ import annotations

import sqlite3

# (memory_name, key, old substring, new substring) — the ``published`` teaching in
# each 0069-seeded skill entry, rewritten to ``notify``.
_SKILL_ENTRY_REPLACEMENTS = [
    (
        "Research collection — notify on new finds",
        "published: true (a notifier delivers each new entry to the user)",
        "notify: true (Penny tells you about each new entry)",
    ),
    (
        "Research collection — silent",
        "published: false (the collector gathers; nothing pings the user)",
        "notify: false (the collector gathers; nothing pings the user)",
    ),
    (
        "Flip silent ↔ notify",
        'collection_update("[X]", published=true) to start notifying, or published=false '
        "to go silent.",
        'collection_update("[X]", notify=true) to start notifying, or notify=false to go silent.',
    ),
]

# The skills collector's own extraction_prompt (memory row name='skills') teaches
# the create step's emission flag — rewrite it to ``notify`` too.
_SKILLS_PROMPT_OLD = (
    "its create step sets published: true (a separate notifier delivers new entries)"
)
_SKILLS_PROMPT_NEW = "its create step sets notify: true (Penny tells the user about each new entry)"


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    # 1. Archive the retired notifier consumer (seeded key, universal).
    conn.execute("UPDATE memory SET archived = 1 WHERE name = 'notifier'")

    # 2. Rewrite the seeded skills that teach ``published`` → ``notify`` (surgical,
    #    idempotent) before the column disappears.
    for key, old, new in _SKILL_ENTRY_REPLACEMENTS:
        conn.execute(
            "UPDATE memory_entry SET content = REPLACE(content, ?, ?), content_embedding = NULL "
            "WHERE memory_name = 'skills' AND key = ? AND content LIKE '%' || ? || '%'",
            (old, new, key, old),
        )
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'skills' AND extraction_prompt LIKE '%' || ? || '%'",
        (_SKILLS_PROMPT_OLD, _SKILLS_PROMPT_NEW, _SKILLS_PROMPT_OLD),
    )

    # 3. Drop the retired pub/sub column (``notify`` is the sole emission flag now).
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "published" in columns:
        conn.execute("ALTER TABLE memory DROP COLUMN published")

    conn.commit()
