"""Switch the quality collector to cursor-based log reads.

Type: data

0055 had the quality collector review ``collector-runs`` + ``penny-messages``
with ``log_read_recent`` (a fixed 24h window).  The auto-throttle can stretch a
collector's interval past that window (toward the weekly cap on quiet cycles),
which would leave a blind spot — behaviour that happened between the window edge
and the (longer) interval would never be reviewed.

``log_read_next`` is cursor-based: it reads everything since the last cycle
regardless of how long that was, so the gap can't open no matter how far the
throttle backs quality off.  It also gives quality a real read cursor, which you
can roll back (addon cursor controls) to re-review a stretch of history on
demand.  (``log_read_recent`` stays in the toolbox for genuine fixed-lookback
needs like dedup, where "since last run" would be too short.)

Rewrites the seeded quality ``extraction_prompt`` in place — only step 1 of the
sequence changes (the two reads); the rest is identical to 0055.
"""

from __future__ import annotations

import sqlite3

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
    '1. log_read_next("collector-runs") and log_read_next("penny-messages") — '
    "the collector runs and messages since you last reviewed.\n"
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


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'",
        (QUALITY_EXTRACTION_PROMPT,),
    )
    conn.commit()
