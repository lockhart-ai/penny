"""Wire thinking + notify into the unified collector by backfilling their prompts.

Phase 6 of the collector refactor.  Two more system collections become
auto-collected:

  - ``unnotified-thoughts`` — populated by the inner monologue (was
    ThinkingAgent).  Picks a random like, browses for something timely,
    drafts a thought, writes to its own collection.  Interval 1200s
    (matches the legacy THINKING_INTERVAL).
  - ``notified-thoughts`` — populated by the notify cycle (was
    NotifyAgent).  Reads ``unnotified-thoughts``, sends one to the user
    via ``send_message``, moves the entry into its own collection.
    Interval 300s (matches the legacy NOTIFY_INTERVAL).

Per-collection ``collector_interval_seconds`` is backfilled for every
collection that has an extraction_prompt so the dispatcher Collector
can pick the most-overdue one each tick:

  - likes, dislikes, knowledge, notified-thoughts: 300s
  - unnotified-thoughts: 1200s

Idempotent — only writes when the row currently has NULL.

The legacy NotifyAgent's ``terminator_tool`` was ``send_message``; the
unified Collector exits via ``done()`` like all other agents, so the
NOTIFY prompt's step 6 ("done()") is now load-bearing.  The old prompt
already includes that step, so no behavior change.
"""

from __future__ import annotations

import sqlite3

_THINKING_PROMPT = (
    "You are Penny's thinking agent. Once per run, you find ONE specific, "
    "concrete thing worth knowing about — something the user would enjoy "
    "hearing — and store it as a thought.\n\n"
    "Sequence:\n"
    '1. collection_read_random("likes", 1) — pick one seed topic from '
    "the user's likes.\n"
    '2. read_latest("dislikes") — see what the user doesn\'t like.\n'
    '3. browse(queries=["<seed topic>"]) — search the web and read one or '
    "two pages to find something timely and interesting grounded in the "
    "seed topic.\n"
    "4. Draft ONE thought connecting what you found to the seed.  Write "
    "it conversationally, like you're texting a friend; include specific "
    "details (names, specs, dates), at least one source URL, and finish "
    "with an emoji.  Keep it under 300 words.\n"
    "5. Check the draft against the dislikes list.  If it conflicts with "
    "anything the user dislikes, call done() without writing.\n"
    '6. exists(["unnotified-thoughts", "notified-thoughts"], key, '
    "content) — if a similar thought already exists, call done() without "
    "writing.\n"
    '7. collection_write("unnotified-thoughts", entries=[{key: short '
    "topic name (3-10 words), content: the thought you drafted}]).\n"
    "8. done().\n\n"
    "The interesting stuff is ON the pages, not in search snippets — "
    "browse the URLs you find rather than searching forever.  If nothing "
    "noteworthy comes up, call done() without writing; quiet cycles are "
    "normal.  Troubleshooting guides, bug workarounds, and support "
    "articles are NOT interesting discoveries."
)


_NOTIFY_PROMPT = (
    "You are Penny's notify agent. Once per cycle, you reach out to "
    "your friend the user with ONE thought worth sharing.\n\n"
    "Sequence:\n"
    '1. read_latest("unnotified-thoughts") — list every '
    "fresh thought you have to share.\n"
    '2. log_read_recent("penny-messages", window_seconds=86400) — '
    "see what you've already said today; don't repeat yourself.\n"
    "3. Pick ONE unnotified thought you haven't already shared and "
    "still find interesting.\n"
    "4. send_message(content=<your message>) — deliver the thought to "
    "the user.  Write it conversationally, like you're texting a "
    "friend; open with a casual greeting, then write out every "
    "detail in full.  No ellipses ('...', '…'), no 'etc.', no 'and "
    "more', no teaser phrasing — finish every sentence and bullet "
    "you start.  The user only sees what you put in `content`; "
    "nothing else is attached.  Include the specific details from "
    "the thought (names, specs, dates), at least one source URL "
    "from the thought, and finish with an emoji.\n"
    '5. ONLY IF send_message returned "Message sent.": '
    'collection_move("unnotified-thoughts", "notified-thoughts", '
    "key=<chosen key>) — mark it as shared.  If send_message returned "
    "an error or refusal, DO NOT move the thought — leave it in "
    "unnotified-thoughts so a later cycle can retry.\n"
    "6. done().\n\n"
    "Every fact and URL in your message must come from the thought "
    "you read — do not invent information.  If no unnotified thought "
    "is worth sharing, call done() without sending anything."
)


_PROMPT_BACKFILL = [
    ("unnotified-thoughts", _THINKING_PROMPT),
    ("notified-thoughts", _NOTIFY_PROMPT),
]

_INTERVAL_BACKFILL = [
    ("likes", 300),
    ("dislikes", 300),
    ("knowledge", 300),
    ("notified-thoughts", 300),
    ("unnotified-thoughts", 1200),
]


def up(conn: sqlite3.Connection) -> None:
    for name, prompt in _PROMPT_BACKFILL:
        conn.execute(
            "UPDATE memory SET extraction_prompt = ? WHERE name = ? AND extraction_prompt IS NULL",
            (prompt, name),
        )
    for name, interval in _INTERVAL_BACKFILL:
        conn.execute(
            "UPDATE memory SET collector_interval_seconds = ? "
            "WHERE name = ? AND collector_interval_seconds IS NULL",
            (interval, name),
        )
    conn.commit()
