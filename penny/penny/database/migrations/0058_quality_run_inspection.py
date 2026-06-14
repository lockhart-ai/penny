"""Quality reviews runs by inspecting their traces, not a second log.

Type: data

Reworks the ``quality`` collector's extraction_prompt around the run trace:

1. **One cursor, not two.**  The old prompt read both ``collector-runs`` and
   ``penny-messages``.  Those are separately-cursored logs whose positions drift
   apart, so the agent compared runs against messages from unrelated windows.
   ``penny-messages`` is dropped — ``collector-runs`` already records every
   cycle, one entry per run, under a single cursor.
2. **Index, then detail.**  ``collector-runs`` is the index: each entry is a run
   summary tagged with its ``[run id]``.  For a run that looks off, the agent now
   calls ``log_get(<run id>)`` to pull that run's full trace — the actual entries
   it wrote, the exact message it sent — and judges *what it did* against the
   collection's intent (the right write? the right message? a message the intent
   never wanted?).  No more reasoning from a one-line summary alone.
3. **Run failures aren't drift.**  A ``❌`` run (max steps, a crash) is capacity,
   not a behaviour-vs-intent contradiction.  The prompt says to skip those — only
   a clean run whose actions contradict the intent warrants a prompt fix.
4. **Collections, not logs.**  The agent reviews goal *collections*; it never
   treats a system log as something to "fix".
"""

from __future__ import annotations

import sqlite3

_QUALITY_PROMPT = (
    "You are Penny's quality agent.  Each cycle you inspect your collectors' "
    "recent runs and fix EVERY collection whose behaviour has drifted from what "
    "the user asked of it — then tell the user what you changed.\n\n"
    "A collection's `intent` is the user's own words for what it should do — the "
    "spec.  Its `extraction_prompt` is how it tries to do it.  When a run's "
    "actual behaviour no longer serves the intent, rewrite the prompt to match.  "
    "The intent is fixed — you can never change it; you change the prompt to "
    "honour it.\n\n"
    "Sequence:\n"
    '1. log_read("collector-runs") — the next batch of your collectors\' runs.  '
    "Each entry is one run: a collection name, an outcome marker, a one-line "
    "summary, tagged with its run id in brackets.\n"
    "2. Only a ✅ worked run can have drifted — it's the only kind that DID "
    "something.  Skip 💤 idle runs (they did nothing) and ❌ failed runs (max "
    "steps or a crash — a capacity problem, never a prompt fix).  For EACH ✅ "
    "worked run in the batch, inspect it before forming any opinion:\n"
    "   a. log_get(<run id>) — its full trace: the exact entries it wrote and "
    "the exact message it sent.  The one-line summary is NOT enough to judge — "
    "you must read the trace.\n"
    "   b. collection_metadata(<collection>) — its intent + current prompt.\n"
    "   c. Judge the trace against the intent.  Drift is a clear contradiction: "
    "it messaged the user when the intent says stay quiet; it wrote the wrong "
    "thing.  If the run honoured the intent, change nothing and move to the next "
    "run.\n"
    "   d. When a collection has MORE THAN ONE ✅ worked run in this batch, "
    "compare their traces against each other: if two runs sent the user the "
    "same (or nearly the same) message, the collection is re-sending itself — "
    "that is a repeat drift even though each send looks fine on its own.\n"
    "3. For EACH collection that genuinely drifted, carry the fix ALL the way "
    "through — judging is not enough, you must apply it:\n"
    "   a. Draft a corrected extraction_prompt: fix the offending step, keep "
    "every other step intact.  Unwanted pings (intent says stay quiet): remove "
    "the send_message step.  Repeats: the offender is a step that reads past "
    "output and re-sends it — drop that read; not-repeating is handled by the "
    "collection's own move/write, never by re-sending.\n"
    "   b. prompt_test(collection=<collection>, extraction_prompt=<draft>) — "
    "dry-run it; if the cycle would still violate the intent, revise and "
    "prompt_test again.  Only proceed once the dry run is clean.\n"
    "   c. collection_update(name=<collection>, extraction_prompt=<the "
    "dry-run-confirmed draft>).\n"
    "4. If you fixed anything, send_message the user one or two sentences naming "
    "each collection you fixed, what was going wrong, and what you changed.\n"
    "5. done().  If no worked run contradicted its intent, just done() and "
    "change nothing — quiet batches are normal and expected.\n\n"
    "Only act on a clear, current contradiction between behaviour and intent.  "
    "Never weaken an intent to excuse a prompt."
)


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'",
        (_QUALITY_PROMPT,),
    )
    conn.commit()
