"""Make the quality agent reliably rewrite drifted prompts; enforce numbered format.

Type: data

Two findings drove this:

1. gpt-oss follows a NUMBERED instruction/tool-call recipe far more reliably than
   the same task as prose — replaying real collector prompts, a prose task in the
   system prompt bailed (jumped to ``done()`` without doing the work) ~60% of the
   time on the empty collector user turn, vs ~5% for the numbered rewrite.  So the
   collectors that author prompts must enforce numbering: ``quality`` when it
   rewrites a drifted ``extraction_prompt`` (and a prose prompt is itself drift),
   ``skills`` for the STEPS it writes (the chat agent follows those on recall).

2. Measuring the quality cycle exposed its real weakness: it fixes a clear drift
   well (silent-drift 4/6) but on the harder cross-run-repeat case it usually
   *rewrote the prompt correctly and then forgot to message the user* — the notify
   was a separate trailing step (4) it skipped after applying the fix in step 3.

   A first attempt also made quality proactively scan every reviewed prompt for
   prose and "fix" the format — that backfired hard (healthy collections rewritten
   5/6: it can't tell a fine numbered prompt from one needing reformatting, so it
   over-corrects).  So quality does NOT hunt for prose; retroactive format
   conversion of existing prose prompts is done out-of-band (direct edits), and
   new prompts are kept numbered at authoring time (CollectionCreateTool guidance).

So the quality change is minimal: (a) fold the user-notify INTO the per-collection
fix loop (apply → immediately message, never a trailing afterthought it skips), and
(b) require the rewrite it ALREADY makes for a behaviour drift to come out numbered.
``skills`` gets a surgical change requiring numbered STEPS.

Only these two code-managed system prompts are touched; user-created collections are
never migrated.
"""

from __future__ import annotations

import sqlite3

_QUALITY_PROMPT = """\
You are Penny's quality agent.  Each cycle you review your collectors' recent runs \
and fix EVERY collection whose behaviour has drifted from what the user asked of it \
— applying and announcing each fix as you go.

A collection's `intent` is the user's own words for what it should do — the spec.  \
Its `extraction_prompt` is how it tries to do it.  When a run's actual behaviour no \
longer serves the intent, rewrite the prompt to match.  The intent is fixed — you \
can never change it; you change the prompt to honour it.

Sequence:
1. log_read("collector-runs") — the next batch of your collectors' runs.  Each \
record is one run: a `[collection] summary` header and, for a run that did \
something, the exact tool calls it made (what it wrote, the message it sent).
2. Review EVERY record.  A run showing only a header did nothing or failed — skip \
it; there's no behaviour to judge.  For each run that DID something, judge what it \
did against its collection's intent: did it message the user when the intent says \
stay quiet?  did it send the same thing twice?  did it write the wrong thing?  Read \
the intent with memory_metadata(<collection>) when you need it.  If everything \
honoured its intent, call done() and change nothing — quiet batches are normal and \
expected.
3. For EACH collection that genuinely drifted, carry the fix all the way through, \
one collection at a time — judging is not enough, you must apply AND announce it:
   a. Draft a corrected extraction_prompt as a NUMBERED list of explicit steps and \
tool calls (1., 2., 3.), never flowing prose (gpt-oss follows a numbered recipe far \
more reliably): fix the offending step, keep every other step intact.  Unwanted \
pings (intent says stay quiet): remove the send_message step.  Repeats: drop the \
step that reads past output and re-sends it — not-repeating is handled by the \
collection's own move/write, never by re-sending.
   b. prompt_test(collection=<collection>, extraction_prompt=<draft>) — dry-run it; \
if the cycle would still violate the intent, revise and prompt_test again.  Only \
proceed once the dry run is clean.
   c. collection_update(name=<collection>, extraction_prompt=<the dry-run-confirmed \
draft>).
   d. send_message the user one sentence naming this collection, what was going \
wrong, and what you changed.  REQUIRED after every fix — never apply a change \
silently.
4. done().

Only act on a clear, current contradiction between behaviour and intent.  Never \
weaken an intent to excuse a prompt."""

_SKILLS_EDITS = [
    (
        "STEPS section (concrete tool composition)",
        "STEPS section written as a NUMBERED list of concrete tool calls and steps "
        "(1., 2., 3.), never prose — the chat agent follows a numbered recipe far "
        "more reliably than a paragraph",
    ),
]
_SKILLS_MARKERS = ["STEPS section written as a NUMBERED list of concrete tool calls"]


def _apply_skills_edits(conn: sqlite3.Connection) -> None:
    for old, new in _SKILLS_EDITS:
        conn.execute(
            "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
            "WHERE name = 'skills'",
            (old, new),
        )
    row = conn.execute("SELECT extraction_prompt FROM memory WHERE name = 'skills'").fetchone()
    prompt = row[0] if row else ""
    for marker in _SKILLS_MARKERS:
        if marker not in prompt:
            raise RuntimeError(
                f"0063: 'skills' enforcement not applied — missing marker {marker!r}. "
                "An anchor likely changed; update this migration's REPLACE pairs."
            )


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'", (_QUALITY_PROMPT,)
    )
    _apply_skills_edits(conn)
    conn.commit()
