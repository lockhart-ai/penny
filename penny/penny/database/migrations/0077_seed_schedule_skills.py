"""Seed skills that dispatch scheduling requests to the schedule tools.

Type: data

The ``/schedule`` and ``/unschedule`` commands retired (epic #1445) in favour of
the ``schedule_create`` / ``schedule_delete`` / ``schedule_list`` tools the chat
agent drives from natural language.  These skills are the NL triggers that make the
dispatch reliable — a TRIGGER (intent + example phrasings) plus numbered tool-call
STEPS, in the one clean shape migration 0069 established.

They are operate-the-system skills (no source collection), so the skills reconcile
loop leaves them alone, exactly like the scope/cadence/flip/change-source skills.

Seeded with ``author='system'`` (like every 0043 seed), so the 0069
``author='skills'`` scrub never touches them.  Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_CREATE = """TRIGGER
User wants something to happen on a recurring cadence — a task Penny runs automatically on
a schedule (daily, weekly, every morning, hourly, etc.). Example phrasings:
- "every morning send me a summary of the local news"
- "each Monday remind me to water the plants"
- "can you check the weather for me every day at 7am?"
- "set up a daily digest of my unread email"

STEPS
1. schedule_create(request=<the user's whole request — the task AND the timing, verbatim>) —
   e.g. "every weekday at 8am summarize my unread email". The tool parses the cadence into a
   schedule and saves it.
2. The result echoes the parsed cadence (timing + cron + task). Confirm it back to the user in
   plain language — name the task and when it will run (e.g. "Got it — I'll summarize your
   unread email every weekday at 8am").
3. If the result says a timezone is missing, ask the user for their location or city, then retry."""

_DELETE = """TRIGGER
User wants to stop or remove a recurring scheduled task Penny runs for them. Example phrasings:
- "you can stop the morning summaries"
- "cancel the daily news digest"
- "stop reminding me to water the plants"
- "turn off the weekly report"

STEPS
1. schedule_delete(description=<the schedule to remove, in the user's words — its task or
   timing, e.g. "the morning news summary">) — the tool matches the closest schedule by meaning
   and deletes it. Never pick by number or position.
2. The result names exactly which task was removed. Confirm that back to the user (e.g. "Done —
   I've stopped the morning news summary")."""

_LIST = """TRIGGER
User asks what recurring tasks they have scheduled. Example phrasings:
- "what do you have scheduled?"
- "what are my scheduled tasks?"
- "list my schedules"
- "what's on my schedule?"

STEPS
1. schedule_list() — returns each scheduled task and its cadence, or that there are none.
2. Report the list back to the user in plain language, or tell them their schedule is empty."""

_SKILLS = {
    "Schedule a recurring task": _CREATE,
    "Stop a scheduled task": _DELETE,
    "List scheduled tasks": _LIST,
}


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    for key, content in _SKILLS.items():
        conn.execute(
            "INSERT OR IGNORE INTO memory_entry "
            "(memory_name, key, content, author, key_embedding, content_embedding, created_at) "
            "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
            (key, content, now),
        )
    conn.commit()
