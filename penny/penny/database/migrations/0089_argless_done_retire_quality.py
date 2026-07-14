"""Argless ``done()`` seeded-prompt cleanup + retire the ``quality`` collector.

Type: data

Epic #1554, issue #1569.  ``done()`` is now an argless sentinel — the run record
is GENERATED from the run's canonical ledger rows (its tool calls + write-gate
outcomes + structural counts), never from a model-authored ``success``/``summary``.
Three universal seeded-data consequences:

1. **Rewrite the live ``skills`` collector prompt's ``done(success=…, summary=…)``
   instructions to ``done()``** (seeded by 0069, known key ``skills``).  Two
   compound conditional steps survived 0087's terminal-bare-done strip; with an
   argless ``done`` they would tell the model to pass forbidden arguments (the
   ``done`` args model is now ``NoArgs`` / ``extra="forbid"``), so each such call
   would fail arg-validation.  Surgical, idempotent substring replacements —
   naturally no-ops on a re-run, and they preserve any runtime refinement to the
   rest of the body.  The archived ``notifier`` prompt (0067) keeps its stale
   ``done(success=…)`` text: it is archived and never dispatched, so nothing runs
   it.

2. **Rewrite the ``collector-runs`` log description** (seeded by 0034, known key)
   — it named the old "target + success marker + done() summary" shape, which the
   model reads in the self-state header's store map; the record is now a generated
   structural record, so the description says so.

3. **Archive the ``quality`` collection** (seeded by 0055, known key) — the same
   visible-tombstone pattern 0086 used for ``notifier``.  ``quality`` existed to
   correct drift in extraction_prompts GENERATED FROM PROSE; that authoring
   channel is gone (a collector's prompt is now a deterministic render of a taught
   skill, #1590/#1591 — a wrong prompt is fixed by the user re-teaching the skill,
   which REPLACES it and re-renders the collection).  With no prose-generation step
   left to review, the model-judgment reviewer retires with the failure mode.
   Archived (not deleted): the row stays enumerable in the archived-inclusive
   catalog, and ``SYSTEM_COLLECTIONS`` keeps the shell hidden from
   ``collection_catalog``.  Its ``extraction_prompt`` is untouched (an archived
   collection never dispatches); the shared ``render_run_record`` / ``classify_run``
   machinery it used to read STAYS — the self-state header and the addon consume
   it.

Touches only universal data — migration-seeded rows by known key — so it is safe
on every deployment and idempotent.
"""

from __future__ import annotations

import sqlite3

# (old substring, new substring) — the ``skills`` prompt's compound ``done``
# conditionals, rewritten to the argless sentinel.  Verbatim from the 0069 seed
# (0086 rewrote a different line, the ``published`` → ``notify`` teaching, not
# these), so they match the shipped text exactly.
_SKILLS_DONE_REPLACEMENTS = [
    ('done(success=true, summary="no collections to learn from")', "done()"),
    ('done(success=true, summary="skills already match the collections")', "done()"),
]

# The ``collector-runs`` log description, rewritten off the old done()-summary shape
# (seeded verbatim by 0034).
_COLLECTOR_RUNS_DESC_OLD = "One entry per Collector cycle: target + success marker + done() summary"
_COLLECTOR_RUNS_DESC_NEW = (
    "One entry per Collector cycle: the target + its generated structural record "
    "(outcome, counts, tool trace)"
)


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return

    # 1. Argless-done cleanup of the live ``skills`` collector prompt.
    for old, new in _SKILLS_DONE_REPLACEMENTS:
        conn.execute(
            "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
            "WHERE name = 'skills' AND extraction_prompt LIKE '%' || ? || '%'",
            (old, new, old),
        )

    # 2. Rewrite the ``collector-runs`` description off the old done()-summary shape.
    conn.execute(
        "UPDATE memory SET description = ? WHERE name = 'collector-runs' AND description = ?",
        (_COLLECTOR_RUNS_DESC_NEW, _COLLECTOR_RUNS_DESC_OLD),
    )

    # 3. Archive the retired ``quality`` consumer (visible tombstone).
    conn.execute("UPDATE memory SET archived = 1 WHERE name = 'quality'")

    conn.commit()
