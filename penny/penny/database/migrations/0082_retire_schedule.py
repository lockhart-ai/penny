"""Retire the ``schedule`` mechanism and add the once-shaped trigger columns (#1556).

Type: schema + data

Wave 1 of the tool-surface hardening epic (#1554).  The ``schedule`` table was a
recurring-task mechanism separate from collectors, whose fire replayed a stored
``prompt_text`` as a synthetic *user turn* into the chat agent — a shape this epic
exists to kill.  Its tools, executor, table, and client surfaces are removed in
this PR; this migration handles the schema + seeded-data half:

1. **Drop the ``schedule`` table.**  Existing rows are NOT migrated to collectors
   (translating a free-text ``prompt_text`` into a collector recipe is model
   judgment a migration can't do, and empty placeholder collectors are exactly the
   half-alive mechanisms the epic removes — see the code-owner decision on #1556).
   Users re-create what they still want through the collection surface.  The drop
   is NOT silent: the count and the human cadence labels are logged first so the
   disappearance is diagnosable (visible-degradation).

2. **Delete the 0077-seeded schedule-dispatch skills** by their known seeded keys.
   They taught the chat agent to call the now-deleted ``schedule_create`` /
   ``schedule_delete`` / ``schedule_list`` tools; left live they would fire broken
   tool calls on every "every morning…" request.  Re-teaching the capability
   through the collection surface is deferred to #1471 — this migration only
   removes.  Targets universal seeded data (``author='system'`` + known keys),
   never deployment-specific rows.

3. **Add ``memory.run_at`` + ``memory.max_runs``** — the once-shaped trigger
   columns.  Store-level only: no model-facing ``collection_create`` args yet
   (#1562 owns the creation-surface exposure).  ``run_at`` delays a collector's
   first fire; ``max_runs`` retires it (archive) after that many completed cycles,
   so a one-shot reminder (``run_at`` + ``max_runs=1``) is expressible without a
   new bespoke mechanism.  Both nullable — an ordinary recurring collection carries
   neither.  A fresh ``create_tables()`` DB already has these columns (the model
   defines them), so the presence guard skips them; a production copy gets them
   added.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# The 0077-seeded schedule-dispatch skill keys (``author='system'`` rows in the
# ``skills`` collection).  Referenced by their known seeded keys — legitimate
# migration territory, unlike a deployment-specific chat-created row.
_SEEDED_SCHEDULE_SKILL_KEYS = (
    "Schedule a recurring task",
    "Stop a scheduled task",
    "List scheduled tasks",
)


def up(conn: sqlite3.Connection) -> None:
    _drop_schedule_table(conn)
    _delete_seeded_schedule_skills(conn)
    _add_trigger_columns(conn)
    conn.commit()


def _drop_schedule_table(conn: sqlite3.Connection) -> None:
    """Log the dropped schedules (count + cadence labels), then drop the table.

    The excerpts go only to the local runtime log so a user can see what
    disappeared — the mechanism is gone, but its disappearance is not silent.
    """
    if not _table_exists(conn, "schedule"):
        return
    rows = conn.execute(
        "SELECT timing_description, prompt_text FROM schedule ORDER BY created_at"
    ).fetchall()
    if rows:
        logger.info(
            "Retiring the schedule mechanism (#1556): dropping %d scheduled task(s) — "
            "NOT migrated to collectors; re-create any you still want via the "
            "collection surface.",
            len(rows),
        )
        for timing, prompt in rows:
            logger.info("  dropped schedule: %s — %.60s", timing, prompt or "")
    conn.execute("DROP TABLE IF EXISTS schedule")


def _delete_seeded_schedule_skills(conn: sqlite3.Connection) -> None:
    """Remove the seeded skills that dispatched to the deleted schedule tools."""
    if not _table_exists(conn, "memory_entry"):
        return
    placeholders = ", ".join("?" for _ in _SEEDED_SCHEDULE_SKILL_KEYS)
    cursor = conn.execute(
        f"DELETE FROM memory_entry WHERE memory_name = 'skills' AND author = 'system' "  # noqa: S608
        f"AND key IN ({placeholders})",
        _SEEDED_SCHEDULE_SKILL_KEYS,
    )
    if cursor.rowcount:
        logger.info("Deleted %d seeded schedule-dispatch skill(s)", cursor.rowcount)


def _add_trigger_columns(conn: sqlite3.Connection) -> None:
    """Add the nullable once-shaped trigger columns to ``memory`` (idempotent)."""
    if not _table_exists(conn, "memory"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "run_at" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN run_at TIMESTAMP")
    if "max_runs" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN max_runs INTEGER")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
