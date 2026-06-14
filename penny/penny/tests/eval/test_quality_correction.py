"""Quality-collector contracts — the graduated self-correcting collector.

The ``quality`` collection is seeded by migration 0055 (prompt refined through
0058), so it exists in every DB.  These cases drive the REAL seeded
extraction_prompt via ``run_for("quality")``.

Quality reviews its collectors' runs the way the prompt describes: read the
``collector-runs`` index (one summary per run, tagged with its run id), pick the
runs that look off, and ``log_get(<run id>)`` each to see the run's full trace —
the actual entries it wrote, the exact message it sent — then judge that against
the collection's ``intent``.  So each case seeds a synthetic suspect collection
(its intent + a prompt) AND the promptlog run(s) behind it, plus the keyed
``collector-runs`` entries that index them.  There is no ``penny-messages`` read
— that second cursor drifted from ``collector-runs``; the message detail now
comes from the run's own trace.

  rebroadcast  — intent "one fresh thought, never repeat"; two runs re-send the
                 same digest → rewrite the prompt (any material corrective change).
  silent-drift — intent "never ping me"; a run sent an update → drop send_message.
  healthy      — a run's behaviour matches intent → change nothing.
  run-failure  — a ❌ run (max steps) is capacity, not drift → change nothing.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from penny.agents.collector import Collector
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory_store import Inclusion, LogEntryInput, RecallMode
from penny.tests.eval.conftest import CollectorScorer, tool_was_called

pytestmark = pytest.mark.eval

# These cases are REPORT-ONLY (``min_pass_rate=None``): each prints its X/Y pass
# rate but never fails the run.  The quality flow is the hardest multi-hop cycle
# (read the run index → inspect a suspect run's trace → read its intent → judge →
# dry-run → rewrite → notify), and gpt-oss clears it — especially the cross-run
# repeat case — only some of the time per sample.  A calibrated red/green
# threshold would add no signal beyond the printed rate, which is what you watch
# as you iterate the prompt.  ``make eval`` is run by hand, so nothing gates on it.

# ── Synthetic suspect collections (intent + a drifted extraction_prompt) ─────

_DIGEST_PROMPT = (
    "Share one fresh daily digest thought.\n"
    '1. log_read("penny-messages") — re-read what you sent so you do not '
    "repeat yourself.\n"
    "2. Compose a short digest of the latest items.\n"
    "3. send_message the digest.\n"
    "4. done()."
)
_SILENT_DRIFT_PROMPT = (
    "Collect espresso equipment worth considering.\n"
    "1. browse(...) for new espresso gear; read actual pages.\n"
    '2. collection_write("espresso-gear", entries=[...]).\n'
    "3. If the write succeeded, send_message: one-sentence 'found a new item' + URL.\n"
    "4. done()."
)
_HEALTHY_PROMPT = (
    "Collect houseplant care tips.\n"
    "1. browse(...) for fresh houseplant-care advice; read pages.\n"
    '2. collection_write("houseplant-care", entries=[...]).\n'
    "3. If a genuinely new tip was written, send_message one sentence + URL.\n"
    "4. done()."
)
# A correct notify-on-new prompt whose recent run merely FAILED (max steps) —
# the behaviour doesn't contradict the intent, so it must be left alone.
_OK_NEWS_PROMPT = (
    "Collect notable new developer tools.\n"
    "1. browse(...) for newly released or trending dev tools; read pages.\n"
    '2. collection_write("dev-tools", entries=[...]).\n'
    "3. If a genuinely new tool was written, send_message one sentence + URL.\n"
    "4. done()."
)


def _seed_run(
    db: Database,
    *,
    suspect: str,
    run_id: str,
    outcome: RunOutcome,
    summary: str,
    calls: list[tuple[str, dict]],
) -> None:
    """Seed one collector run: its promptlog trace + the keyed index entry.

    The promptlog row carries the run's tool calls (what it actually did) so
    ``log_get`` can render the trace; the ``collector-runs`` entry is keyed by
    ``run_id`` exactly as the real ``_log_run`` writes it, so the quality cycle's
    ``log_read`` surfaces the id it then passes back to ``log_get``.
    """
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"{run_id}-{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for index, (name, args) in enumerate(calls)
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=response,
        agent_name="collector",
        run_id=run_id,
        run_target=suspect,
    )
    db.messages.set_run_outcome(run_id, outcome.value, summary)
    marker = Collector._OUTCOME_MARKER[outcome]
    db.memories.append(
        PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
        [LogEntryInput(content=f"[{suspect}] {marker} {outcome.value} — {summary}", key=run_id)],
        author="collector",
    )


def _seed(*, suspect: str, description: str, intent: str, prompt: str, runs):
    """Seeder: one suspect collection (drifted) + the runs that exercised it.

    The ``quality`` collection itself is already present from migration 0055.
    """

    def _apply(db: Database) -> None:
        db.memories.create_collection(
            suspect,
            description,
            Inclusion.RELEVANT,
            RecallMode.RECENT,
            extraction_prompt=prompt,
            intent=intent,
        )
        for run in runs:
            _seed_run(db, suspect=suspect, **run)

    return _apply


def _snapshot(suspect: str):
    def _take(db: Database) -> str:
        memory = db.memories.get(suspect)
        return (memory.extraction_prompt or "") if memory else ""

    return _take


def _score_update(suspect: str, forbidden: str | None) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        fails = []
        if not tool_was_called(db, "log_get"):
            fails.append("did not inspect the run trace with log_get before judging")
        if not tool_was_called(db, "prompt_test"):
            fails.append("did not dry-run the fix with prompt_test before applying")
        if new_prompt == original:
            fails.append(f"did not change {suspect!r}'s extraction_prompt")
        elif forbidden is not None and forbidden in new_prompt:
            fails.append(f"corrected prompt still contains the offending {forbidden!r} step")
        elif len(new_prompt) < 80:
            fails.append(f"corrected prompt looks gutted ({len(new_prompt)} chars)")
        if not sent:
            fails.append("did not message the user about the change")
        return fails

    return _score


def _score_no_op(suspect: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        if new_prompt != original:
            return [f"over-corrected a healthy collection ({suspect!r})"]
        return []

    return _score


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_rebroadcast(collector_eval) -> None:
    suspect = "daily-digest"
    digest = "Daily digest — a new co-op title, a reprint, and a sale."
    await collector_eval(
        case_id="quality-rebroadcast",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A once-daily digest of fresh items worth a heads-up.",
            intent="Once per cycle, share exactly one fresh thought I haven't seen "
            "before, and never resend something you've already sent me.",
            prompt=_DIGEST_PROMPT,
            runs=[
                {
                    "run_id": "digest-run-1",
                    "outcome": RunOutcome.WORKED,
                    "summary": "sent the daily digest",
                    "calls": [
                        ("send_message", {"content": digest}),
                        ("done", {"success": True, "summary": "sent the daily digest"}),
                    ],
                },
                {
                    "run_id": "digest-run-2",
                    "outcome": RunOutcome.WORKED,
                    "summary": "sent the daily digest",
                    "calls": [
                        ("send_message", {"content": digest}),
                        ("done", {"success": True, "summary": "sent the daily digest"}),
                    ],
                },
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden=None),
        min_pass_rate=None,
    )


async def test_silent_drift(collector_eval) -> None:
    suspect = "espresso-gear"
    await collector_eval(
        case_id="quality-silent-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A quiet running list of espresso equipment worth considering.",
            intent="Keep a quiet running list of espresso equipment worth considering "
            "— never ping me about it, I'll check the list myself.",
            prompt=_SILENT_DRIFT_PROMPT,
            runs=[
                {
                    "run_id": "espresso-run-1",
                    "outcome": RunOutcome.WORKED,
                    "summary": "wrote 1 entry and sent an update about a new grinder",
                    "calls": [
                        (
                            "collection_write",
                            {
                                "memory": suspect,
                                "entries": [
                                    {
                                        "key": "niche-zero-clone",
                                        "content": "Niche Zero clone grinder, $300",
                                    }
                                ],
                            },
                        ),
                        (
                            "send_message",
                            {
                                "content": "Found a new espresso grinder: "
                                "the Niche Zero clone, $300."
                            },
                        ),
                        (
                            "done",
                            {
                                "success": True,
                                "summary": "wrote 1 entry and sent an update about a new grinder",
                            },
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden="send_message"),
        min_pass_rate=None,
    )


async def test_healthy(collector_eval) -> None:
    suspect = "houseplant-care"
    await collector_eval(
        case_id="quality-healthy",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A list of houseplant-care tips, with a ping on genuinely new ones.",
            intent="Keep a list of houseplant-care tips and ping me when you find a "
            "genuinely new one.",
            prompt=_HEALTHY_PROMPT,
            runs=[
                {
                    "run_id": "plant-run-1",
                    "outcome": RunOutcome.WORKED,
                    "summary": "wrote 1 new tip and pinged about watering",
                    "calls": [
                        (
                            "collection_write",
                            {
                                "memory": suspect,
                                "entries": [
                                    {
                                        "key": "bottom-water-pothos",
                                        "content": "Bottom-water pothos weekly to avoid root rot",
                                    }
                                ],
                            },
                        ),
                        (
                            "send_message",
                            {
                                "content": "New houseplant tip: bottom-water pothos weekly "
                                "to avoid root rot."
                            },
                        ),
                        (
                            "done",
                            {
                                "success": True,
                                "summary": "wrote 1 new tip and pinged about watering",
                            },
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_no_op(suspect),
        min_pass_rate=None,
    )


async def test_run_failure_is_not_drift(collector_eval) -> None:
    """A collector RUN that failed (❌ max steps) is not a behaviour-vs-intent
    drift — it's transient/capacity.  Quality must NOT rewrite the prompt of a
    collection just because its last run failed; only a clean run whose actions
    contradict the intent warrants a fix."""
    suspect = "dev-tools"
    await collector_eval(
        case_id="quality-run-failure-not-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="Notable new developer tools, with a ping on good ones.",
            intent="Track new developer tools and ping me when a good one shows up.",
            prompt=_OK_NEWS_PROMPT,
            runs=[
                {
                    "run_id": "dev-run-1",
                    "outcome": RunOutcome.FAILED,
                    "summary": "max steps exceeded, no done() call this cycle",
                    "calls": [
                        ("browse", {"queries": ["new developer tools 2026"]}),
                        (
                            "collection_write",
                            {
                                "memory": suspect,
                                "entries": [{"key": "zed-1.0", "content": "Zed editor hit 1.0"}],
                            },
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_no_op(suspect),
        min_pass_rate=None,
    )
