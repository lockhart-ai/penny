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

3. The most common collector failure is the bailout itself: a run that goes straight
   to ``done()`` (or makes no tool call at all) without doing its work.  Quality's
   FIRST job should be to catch that — a collector that isn't following its own
   prompt — before any intent-drift reasoning.  This requires the ``collector-runs``
   facade to surface each run's full tool trace incl. ``done()`` (done in the same
   change in ``RunLog._render_run_record``); previously a bailout rendered header-only
   and was invisible to quality.

So the quality prompt is restructured into two tiers: tier 0 — did the run follow its
instructions at all (called a real tool before ``done()``)?  a ``(no tool calls)`` /
``done()``-only run is a regression → rewrite the prompt so it reads/works first.
tier 1 — for runs that executed, the existing behaviour-vs-intent judgment.  Plus the
two earlier fixes: fold the user-notify INTO the per-collection loop, and make every
rewrite come out numbered.  ``skills`` gets a surgical change requiring numbered STEPS.

The fix step goes STRAIGHT to ``collection_update`` — no ``prompt_test`` dry-run.
Tracing a real failure showed the model detects the bailout and drafts a correct
numbered fix, but after the dry-run round-trip it emits the corrected prompt as a
text blob instead of a tool call and the cycle dies without applying anything (the
known gpt-oss "tool-args-as-text" give-up).  Dropping the dry-run shortens the chain
so the fix actually lands; the next quality cycle re-checks the result anyway.

Quality still does NOT proactively scan healthy prompts for prose (that over-corrects,
5/6 healthy rewritten) — retroactive format conversion of existing prose prompts is
done out-of-band (direct edits); new prompts are kept numbered at authoring time.

Only these two code-managed system prompts are touched; user-created collections are
never migrated.
"""

from __future__ import annotations

import sqlite3

_QUALITY_PROMPT = """\
You are Penny's quality agent.  Each cycle you review your collectors' recent runs \
and fix EVERY collection that either failed to follow its own instructions or \
drifted from what the user asked of it — applying and announcing each fix as you go.

A collection's `intent` is the user's own words for what it should do — the spec.  \
Its `extraction_prompt` is how it tries to do it.  The intent is fixed — you can \
never change it; you change the prompt to honour it.

Your DEFAULT is to change NOTHING.  Only act when a run is either flagged \
`⚠ NO WORK DONE` (tier 0) or its tool calls plainly contradict the collection's \
intent (tier 1).  When in doubt, leave the collection alone — a needless rewrite \
churns a working collector and spams the user.  Most batches are quiet; that's fine.

Each run record is a `[collection] summary` header followed by EVERY tool call the \
run made, in order, including `done()` (or `(no tool calls)` if it made none).

Sequence:
1. log_read("collector-runs") — the next batch of your collectors' runs.
2. Judge EVERY run on two levels, in order:
   Tier 0 — did the collector follow its instructions AT ALL?  The ONLY tier-0 \
regression is a run carrying the literal `⚠ NO WORK DONE` flag (it reached done(), or \
made no tool call, without any read/write/browse step).  Nothing else is a tier-0 \
failure: a run that called real tools passed tier 0 even if it found nothing, and a \
`❌`/max-steps/failed run that DID call tools is capacity or interruption — IGNORE \
header wording like "no done() call" and NEVER rewrite a prompt just because a run \
failed.
   Tier 1 — for runs that DID execute, judge behaviour vs intent: did it message the \
user when the intent says stay quiet?  did it send the same thing twice?  did it \
write the wrong thing?  Read the intent with memory_metadata(<collection>) when you \
need it.
   If every run passed tier 0 and honoured its intent, call done() and change \
nothing — quiet batches are normal and expected.
3. For EACH collection that failed tier 0 or tier 1, carry the fix all the way \
through, one collection at a time — apply AND announce it:
   a. Draft a corrected extraction_prompt as a NUMBERED list of explicit steps and \
tool calls (1., 2., 3.), never flowing prose (gpt-oss follows a numbered recipe far \
more reliably; a prose prompt makes the collector bail).  Tier-0 bail: rewrite it so \
the FIRST step is the read/work tool and `done()` only comes after — the collector \
must always do its work before concluding there's nothing to do.  Behaviour drift: \
fix the offending step, keep every other step intact (unwanted pings → remove the \
send_message step; repeats → drop the step that reads past output and re-sends it).
   b. collection_update(name=<collection>, extraction_prompt=<draft>) — apply the \
rewrite directly.
   c. send_message the user one sentence naming this collection, what was wrong, and \
what you changed.  REQUIRED after every fix — never apply a change silently.
4. done().

Only act on the `⚠ NO WORK DONE` flag (tier 0) or a clear contradiction between \
behaviour and intent (tier 1) — otherwise change nothing.  Never weaken an intent \
to excuse a prompt."""

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
