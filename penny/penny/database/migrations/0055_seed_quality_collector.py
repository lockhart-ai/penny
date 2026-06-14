"""Seed the self-correcting ``quality`` collector.

Type: data

Graduates the quality collector from an eval prototype into a real seeded
collection so every deployment gets it.  Each cycle it reviews Penny's own
recent behaviour (``collector-runs`` + ``penny-messages``) against each
collection's ``intent`` and rewrites whichever ``extraction_prompt`` has drifted
— dry-running the fix with ``prompt_test`` before applying it, then telling the
user what it changed (apply-then-notify).

The collector gates the ``prompt_test`` tool into the surface only for this
collection's cycles (see ``Collector.get_tools`` / ``MEMORY_QUALITY_COLLECTION``).

Seeded ``inclusion='never'`` (a background supervisor — never surfaces in chat
recall) at a daily cadence; the auto-throttle backs it off toward weekly on the
many quiet days and snaps it back when something actually drifts.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

QUALITY_DESCRIPTION = (
    "Reviews Penny's own runs and messages and corrects collection prompts that "
    "have drifted from their stated intent"
)

QUALITY_EXTRACTION_PROMPT = (
    "You are Penny's quality agent.  Each cycle you review your own recent "
    "behaviour and fix the ONE collection that has drifted most from what the "
    "user asked of it — then tell the user what you changed.\n\n"
    "A collection's `intent` is the user's own words for what it should do — the "
    "spec.  Its `extraction_prompt` is how it tries to do it.  When the prompt "
    "(or the behaviour it produces) no longer serves the intent, rewrite the "
    "prompt to match.  The intent is fixed — you can never change it; you change "
    "the prompt to honour it.\n\n"
    "Sequence:\n"
    '1. log_read_recent("collector-runs", window_seconds=86400) and '
    'log_read_recent("penny-messages", window_seconds=86400) — what your '
    "collectors actually did and what you actually sent the user.\n"
    "2. Look for ONE concrete problem: a message the user didn't ask for, the "
    "same thing sent twice, a collection acting against its stated intent.  If "
    "nothing looks wrong, call done() and change nothing — quiet cycles are "
    "normal and expected.\n"
    "3. collection_metadata(<the suspect collection>) — read its intent and its "
    "current extraction_prompt.\n"
    "4. Draft a corrected extraction_prompt: fix the one offending step and keep "
    "every other step intact.  Diagnosing the culprit:\n"
    "   - Unwanted pings (intent says stay silent / never notify): remove the "
    "`send_message` step from the body.\n"
    "   - Repeats (the same thing sent twice): the offender is a step that reads "
    "your OWN past output (`penny-messages`, things you already sent) and then "
    "sends it again.  That read is only for CHECKING what you've said so you can "
    "AVOID repeating it — its result must never itself be sent.  Drop the read, "
    "or make the prompt explicit that its result is only for avoiding repeats.\n"
    "5. prompt_test(collection=<the suspect collection>, extraction_prompt=<your "
    "draft>) — dry-run the fix.  Read the result: if the cycle would still "
    "violate the intent (e.g. still sends a message a silent collection "
    "shouldn't), revise the draft and prompt_test again.  Only proceed once the "
    "dry run is clean.\n"
    "6. collection_update(name=<the suspect collection>, extraction_prompt=<the "
    "dry-run-confirmed draft>) — apply the fix.\n"
    "7. send_message the user one or two sentences naming the collection you "
    "fixed, what was going wrong, and what you changed, so they can correct you "
    "if needed.\n"
    "8. done().\n\n"
    "Only act on a clear, current contradiction between behaviour and intent.  "
    "Never weaken an intent to excuse a prompt."
)

# Daily; base_interval_seconds is the snap-back target for the auto-throttle.
_INTERVAL_SECONDS = 86400


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory "
        "(name, type, description, inclusion, recall, archived, created_at, "
        "extraction_prompt, collector_interval_seconds, base_interval_seconds) "
        "VALUES ('quality', 'collection', ?, 'never', 'recent', 0, ?, ?, ?, ?)",
        (
            QUALITY_DESCRIPTION,
            now,
            QUALITY_EXTRACTION_PROMPT,
            _INTERVAL_SECONDS,
            _INTERVAL_SECONDS,
        ),
    )
    conn.commit()
