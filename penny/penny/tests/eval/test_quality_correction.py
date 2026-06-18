"""Quality-collector contracts — the graduated self-correcting collector.

The ``quality`` collection is seeded by migration 0055 (prompt refined since),
so it exists in every DB.  These cases drive the REAL seeded extraction_prompt
via ``run_for("quality")``.

Quality reviews its collectors' runs by ``log_read("collector-runs")`` — a read
facade over ``promptlog`` that renders each run as a record (``[target] summary``
+ the worked run's tool trace: the entries it wrote, the exact message it sent).
It judges each against the collection's ``intent``.  So each case seeds a
synthetic suspect collection (its intent + a prompt) AND the ``promptlog`` run(s)
behind it — which IS the ``collector-runs`` content, no separate log to seed.
There is no keyed ``log_get`` and no ``penny-messages`` read.

  rebroadcast  — intent "one fresh thought, never repeat"; two runs re-send the
                 same digest → rewrite the prompt (any material corrective change).
  silent-drift — intent "never ping me"; a run sent an update → drop send_message.
  healthy      — a run's behaviour matches intent → change nothing.
  run-failure  — a ❌ run (max steps) is capacity, not drift → change nothing.
  ends-with-done — a drift scenario must close the full read→fix→notify hop with done().
  triage       — a batch of several collections must still converge on one fix (weakest path).
  quiet        — a clean batch changes nothing AND stays off the channel (no self-leak).

Sends are observed off ``db.send_queue`` (a collector cycle enqueues; the drainer
that delivers to the channel doesn't run inside ``run_for``) — see ``collector_eval``.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import Inclusion, RecallMode
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
    """Seed one collector run as a ``promptlog`` row (+ its outcome).

    That row IS the ``collector-runs`` content — the facade renders it as a run
    record when the quality cycle calls ``log_read("collector-runs")``.  The
    response carries the run's tool calls (what it actually did) and
    ``set_run_outcome`` stamps the target/outcome/summary the record header uses.
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


def _score_called_done(db: Database, before: object, sent: list[str]) -> list[str]:
    """Done-discipline: the cycle must end by calling ``done()`` — whether it
    fixed something or gave up — never trail off with plain text.  Recreates the
    prod give-up (run e5a7c9e3 returned a text blob and never called done())."""
    if not tool_was_called(db, "done"):
        return ["cycle ended without calling done() — gave up with plain text"]
    return []


def _seed_many(specs):
    """Seeder: several suspect collections, each with its own run(s).

    Stresses the cycle the way a real batch does — multiple collections to
    triage at once.  Each spec is ``{suspect, description, intent, prompt,
    runs}``; ``runs`` reuses the ``_seed_run`` shape.
    """

    def _apply(db: Database) -> None:
        for spec in specs:
            db.memories.create_collection(
                spec["suspect"],
                spec["description"],
                Inclusion.RELEVANT,
                RecallMode.RECENT,
                extraction_prompt=spec["prompt"],
                intent=spec["intent"],
            )
            for run in spec["runs"]:
                _seed_run(db, suspect=spec["suspect"], **run)

    return _apply


def _snapshot_many(names: list[str]):
    def _take(db: Database) -> dict[str, str]:
        prompts = {}
        for name in names:
            memory = db.memories.get(name)
            prompts[name] = (memory.extraction_prompt or "") if memory else ""
        return prompts

    return _take


def _score_triage(drifted: list[str], healthy: list[str]) -> CollectorScorer:
    """Convergence under load: given a batch of collections to triage, the cycle
    must (a) close with ``done()`` and (b) actually land a fix on at least one
    drifted collection — and not touch a healthy one.

    Recreates a production non-convergence: faced with
    several collections, the model grazed ``memory_metadata`` across all of them,
    filled its context with "looks fine", and trailed off with text — fixing
    nothing.  One-drift-per-cycle should make it pick the worst and land it.
    """

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        prompts = cast(dict, before)
        fails: list[str] = []
        if not tool_was_called(db, "done"):
            fails.append("cycle ended without calling done() — gave up with plain text")
        fixed = [name for name in drifted if _changed(db, name, prompts)]
        if not fixed:
            fails.append("fixed none of the drifted collections this cycle")
        touched_healthy = [name for name in healthy if _changed(db, name, prompts)]
        if touched_healthy:
            fails.append(f"over-corrected healthy collection(s): {touched_healthy}")
        return fails

    return _score


def _changed(db: Database, name: str, before: dict) -> bool:
    memory = db.memories.get(name)
    now = (memory.extraction_prompt or "") if memory else ""
    return now != before.get(name, "")


def _score_quiet(suspect: str) -> CollectorScorer:
    """No self-leak on a quiet batch: when nothing drifted, the quality cycle
    must change nothing AND stay silent — it is not a notifier.

    Recreates a production self-leak: reviewing a notify collection, the
    model read that collection's "reach out with one thought" prompt and acted it
    out itself, sending the user an off-intent health fact.  ``_score_no_op`` only
    guards the prompt; this also guards the channel."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        fails: list[str] = []
        if new_prompt != original:
            fails.append(f"over-corrected a healthy collection ({suspect!r})")
        if sent:
            fails.append(f"sent the user a message on a quiet batch (self-leak): {sent!r}")
        return fails

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


async def test_ends_with_done(collector_eval) -> None:
    """The cycle must always close with ``done()`` — recreates the prod give-up
    where the agent dry-ran a fix, then trailed off with a text blob and never
    called done() (the dry-run cluster was the root cause; this guards the
    convergence the better feedback should now produce).  A drift scenario is
    used as the stressor: it forces the full read → dry-run → fix → notify → done
    hop, the hardest path to land cleanly."""
    suspect = "espresso-gear"
    await collector_eval(
        case_id="quality-ends-with-done",
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
                            {"content": "Found a new espresso grinder: the Niche clone, $300."},
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
        score=_score_called_done,
        min_pass_rate=None,
    )


async def test_triage_converges(collector_eval) -> None:
    """A batch with several collections to triage — two genuinely drifted, two
    fine — must still converge: land a fix on at least one drifted collection
    (the rest come round next tick) and leave the healthy ones alone.

    Documents the live multi-collection weakness:
    with several collections in view the model grazes metadata across all of
    them and often trails off having fixed nothing.  This is the quality agent's
    weakest path — it clears it only sometimes per sample (report-only); the
    single-collection cases above are far more reliable."""
    digest = "Daily digest — a new co-op title, a reprint, and a sale."
    drifted = ["daily-digest", "espresso-gear"]
    healthy = ["houseplant-care", "dev-tools"]
    await collector_eval(
        case_id="quality-triage-converges",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed_many(
            [
                {
                    "suspect": "daily-digest",
                    "description": "A once-daily digest of fresh items worth a heads-up.",
                    "intent": "Once per cycle, share exactly one fresh thought I haven't "
                    "seen before, and never resend something you've already sent me.",
                    "prompt": _DIGEST_PROMPT,
                    "runs": [
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
                },
                {
                    "suspect": "espresso-gear",
                    "description": "A quiet running list of espresso equipment worth considering.",
                    "intent": "Keep a quiet running list of espresso equipment worth "
                    "considering — never ping me about it, I'll check the list myself.",
                    "prompt": _SILENT_DRIFT_PROMPT,
                    "runs": [
                        {
                            "run_id": "espresso-run-1",
                            "outcome": RunOutcome.WORKED,
                            "summary": "wrote 1 entry and sent an update about a new grinder",
                            "calls": [
                                (
                                    "collection_write",
                                    {
                                        "memory": "espresso-gear",
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
                                    {"content": "Found a new espresso grinder: the clone, $300."},
                                ),
                                (
                                    "done",
                                    {
                                        "success": True,
                                        "summary": "wrote 1 entry and sent an update",
                                    },
                                ),
                            ],
                        }
                    ],
                },
                {
                    "suspect": "houseplant-care",
                    "description": "A list of houseplant-care tips, with a ping on new ones.",
                    "intent": "Keep a list of houseplant-care tips and ping me when you "
                    "find a genuinely new one.",
                    "prompt": _HEALTHY_PROMPT,
                    "runs": [
                        {
                            "run_id": "plant-run-1",
                            "outcome": RunOutcome.WORKED,
                            "summary": "wrote 1 new tip and pinged about watering",
                            "calls": [
                                (
                                    "collection_write",
                                    {
                                        "memory": "houseplant-care",
                                        "entries": [
                                            {
                                                "key": "bottom-water-pothos",
                                                "content": "Bottom-water pothos weekly",
                                            }
                                        ],
                                    },
                                ),
                                (
                                    "send_message",
                                    {"content": "New houseplant tip: bottom-water pothos weekly."},
                                ),
                                (
                                    "done",
                                    {"success": True, "summary": "wrote 1 new tip and pinged"},
                                ),
                            ],
                        }
                    ],
                },
                {
                    "suspect": "dev-tools",
                    "description": "Notable new developer tools, with a ping on good ones.",
                    "intent": "Track new developer tools and ping me when a good one shows up.",
                    "prompt": _OK_NEWS_PROMPT,
                    "runs": [
                        {
                            "run_id": "dev-run-1",
                            "outcome": RunOutcome.FAILED,
                            "summary": "max steps exceeded, no done() call this cycle",
                            "calls": [("browse", {"queries": ["new developer tools 2026"]})],
                        }
                    ],
                },
            ]
        ),
        snapshot=_snapshot_many(drifted + healthy),
        score=_score_triage(drifted=drifted, healthy=healthy),
        min_pass_rate=None,
    )


async def test_quiet_when_nothing_drifted(collector_eval) -> None:
    """A clean batch must stay clean: the quality cycle changes nothing AND sends
    nothing.  Recreates a production self-leak where, reviewing a
    notify collection, the model acted out that collection's own "share a thought"
    prompt and messaged the user an off-intent fact.  ``test_healthy`` guards only
    the prompt; this also guards that the cycle stayed off the channel."""
    suspect = "houseplant-care"
    await collector_eval(
        case_id="quality-quiet-when-healthy",
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
        score=_score_quiet(suspect),
        min_pass_rate=None,
    )
