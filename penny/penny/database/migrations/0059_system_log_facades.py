"""System logs become read facades over their canonical tables.

Type: data + schema

This is the single migration for the system-log facade refactor (one step on
top of 0058 — collector-runs, user-messages, penny-messages stop being
duplicated ``memory_entry`` rows and are read straight from ``promptlog`` /
``messagelog``).  It:

1. Renames the read tools in every stored extraction_prompt to the
   shape-specific taxonomy: ``read_latest(`` → ``collection_read_latest(`` and
   ``collection_metadata(`` → ``memory_metadata(`` (collection reads error on a
   log now; logs are read only via the cursored ``log_read`` — no ``log_get``).
2. Rewrites the ``quality`` prompt to review runs via plain
   ``log_read("collector-runs")`` (a facade over ``promptlog`` that renders each
   run as a record) — no ``log_get``, no ``penny-messages`` read.
3. Rewrites the ``notify`` prompt to pick an unshared thought with
   ``collection_read_random`` instead of newest-first + "pick one" (which let
   older thoughts sink and never get shared).
4. Drops the now-dead ``memory_entry`` rows for the three facade logs — their
   data lives in ``promptlog`` / ``messagelog``.  The log marker rows stay.
   ``messagelog.embedding`` (for ``read_similar`` over messages) is filled by
   the startup backfill + at write time; nothing is copied between tables.
5. Adds a partial index over the run-completion rows so listing/counting
   ``collector-runs`` is a bounded index read, not a scan of the whole
   ``promptlog``.
"""

from __future__ import annotations

import sqlite3

_QUALITY_PROMPT = (
    "You are Penny's quality agent.  Each cycle you review your collectors' "
    "recent runs and fix EVERY collection whose behaviour has drifted from what "
    "the user asked of it — then tell the user what you changed.\n\n"
    "A collection's `intent` is the user's own words for what it should do — the "
    "spec.  Its `extraction_prompt` is how it tries to do it.  When a run's "
    "actual behaviour no longer serves the intent, rewrite the prompt to match.  "
    "The intent is fixed — you can never change it; you change the prompt to "
    "honour it.\n\n"
    "Sequence:\n"
    '1. log_read("collector-runs") — the next batch of your collectors\' runs.  '
    "Each record is one run: a `[collection] summary` header and, for a run that "
    "did something, the exact tool calls it made (what it wrote, the message it "
    "sent).\n"
    "2. Review EVERY record.  A run showing only a header did nothing or failed "
    "— skip it; there's no behaviour to judge.  For each run that DID something, "
    "judge what it did against its collection's intent: did it message the user "
    "when the intent says stay quiet?  did it send the same thing twice?  did it "
    "write the wrong thing?  Read the intent with memory_metadata(<collection>) "
    "when you need it.  If everything honoured its intent, call done() and change "
    "nothing — quiet batches are normal and expected.\n"
    "3. For EACH collection that genuinely drifted, carry the fix all the way "
    "through — judging is not enough, you must apply it:\n"
    "   a. Draft a corrected extraction_prompt: fix the offending step, keep "
    "every other step intact.  Unwanted pings (intent says stay quiet): remove "
    "the send_message step.  Repeats: drop the step that reads past output and "
    "re-sends it — not-repeating is handled by the collection's own move/write, "
    "never by re-sending.\n"
    "   b. prompt_test(collection=<collection>, extraction_prompt=<draft>) — "
    "dry-run it; if the cycle would still violate the intent, revise and "
    "prompt_test again.  Only proceed once the dry run is clean.\n"
    "   c. collection_update(name=<collection>, extraction_prompt=<the "
    "dry-run-confirmed draft>).\n"
    "4. If you fixed anything, send_message the user one or two sentences naming "
    "each collection you fixed, what was going wrong, and what you changed.\n"
    "5. done().\n\n"
    "Only act on a clear, current contradiction between behaviour and intent.  "
    "Never weaken an intent to excuse a prompt."
)

_NOTIFY_PROMPT = (
    "You are Penny's notify agent.  Once per cycle, share ONE fresh thought with "
    "your friend the user.\n\n"
    "Sequence:\n"
    '1. collection_read_random("unnotified-thoughts", 1) — a thought you have '
    "NOT shared yet, picked at random so older thoughts get their turn too "
    "(sharing moves a thought out of this collection, so it never contains "
    "anything you've already sent — you don't need to check past messages).\n"
    "2. send_message(content=...) — deliver it conversationally: a greeting, all "
    "details from the thought (names, specs, dates), at least one source URL from "
    "the thought, and finish with an emoji.\n"
    '3. ONLY IF send_message returned "Message sent." then '
    'collection_move("unnotified-thoughts", "notified-thoughts", key=<the '
    "thought's key>) — this marks it shared so it can never be picked again.  If "
    "the send failed, leave it in place.\n"
    "4. done().  If there's nothing fresh to share, just done()."
)


def up(conn: sqlite3.Connection) -> None:
    # 1. Read-tool taxonomy rename across all stored extraction prompts.
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'read_latest(', 'collection_read_latest(') "
        "WHERE extraction_prompt LIKE '%read_latest(%'"
    )
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'collection_metadata(', 'memory_metadata(') "
        "WHERE extraction_prompt LIKE '%collection_metadata(%'"
    )
    # 2 + 3. Quality reviews the collector-runs facade; notify samples fairly.
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'",
        (_QUALITY_PROMPT,),
    )
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'notified-thoughts'",
        (_NOTIFY_PROMPT,),
    )
    # 4. Drop the dead replica rows for the three facade logs (markers stay).
    conn.execute(
        "DELETE FROM memory_entry WHERE memory_name IN "
        "('collector-runs', 'user-messages', 'penny-messages')"
    )
    # 5. Partial index over run-completion rows — bounded collector-runs reads.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "promptlog" in tables:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_promptlog_completed_runs "
            "ON promptlog (timestamp) WHERE run_outcome IS NOT NULL"
        )
    conn.commit()
