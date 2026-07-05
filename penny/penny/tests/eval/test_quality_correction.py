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

Quality SUGGESTS, never applies (migration 0073): on real drift it ``send_message``s
a fix suggestion (Observed / Proposed fix / a written-out numbered New prompt) and
calls NO ``collection_update`` — the user approves and the chat agent makes the edit.
So the corrective scorer asserts a suggestion was SENT (naming the collection, with a
numbered new prompt) AND nothing was mutated; the leave-alone scorer asserts nothing
sent AND nothing mutated.  ``collection_update`` being called at all is the failure.

  rebroadcast  — intent "one fresh thought, never repeat"; two runs re-send the
                 same digest → suggest a fix (don't apply it).
  silent-drift — intent "never ping me"; a run sent an update → suggest dropping send.
  healthy      — a run's behaviour matches intent → suggest nothing, mutate nothing.
  run-failure  — a ❌ run (max steps) is capacity, not drift → leave it alone.
  ends-with-done — a drift scenario must close the read→suggest hop with done().
  triage       — a batch of several collections must still converge on one suggestion.
  quiet        — a clean batch suggests nothing AND stays off the channel (no self-leak).

Sends are observed off ``db.send_queue`` (a collector cycle enqueues; the drainer
that delivers to the channel doesn't run inside ``run_for``) — see ``collector_eval``.
"""

from __future__ import annotations

import json

import pytest

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import Inclusion, RecallMode
from penny.tests.eval.conftest import CollectorScorer, looks_numbered, tool_was_called
from penny.tests.eval.fixtures import WATCHLIST, WATCHLIST_INTENT, WATCHLIST_PROSE_PROMPT

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
# A notify prompt whose run sent the user a half-formed message before the real
# one — the prompt lacks a "compose the complete message first" step.
_NEWS_NOTIFY_PROMPT = (
    "Tell me about newly released board games.\n"
    "1. browse(...) for newly released board games; read pages.\n"
    '2. collection_write("board-game-news", entries=[...]).\n'
    "3. send_message about the new game.\n"
    "4. done()."
)


def _browse_fail_turns(errors: int) -> list[dict]:
    """Tool-result turns carrying ``errors`` browse-error sections — the
    ``## browse error:`` headers the run-health tally counts to flag ``⚠ NO
    WRITES`` (a browse failed AND the run wrote nothing).  Mirrors the run's
    final-prompt conversation, where the browse tool's result lands."""
    content = PennyConstants.SECTION_SEPARATOR.join(
        f"{PennyConstants.BROWSE_ERROR_HEADER}https://news.example.test/{index}\n"
        "Could not read this page"
        for index in range(errors)
    )
    return [{"role": "tool", "tool_call_id": "b", "content": content}]


def _seed_run(
    db: Database,
    *,
    suspect: str,
    run_id: str,
    outcome: RunOutcome,
    summary: str,
    calls: list[tuple[str, dict]],
    tool_failures: int = 0,
    messages: list[dict] | None = None,
) -> None:
    """Seed one collector run as a ``promptlog`` row (+ its outcome).

    That row IS the ``collector-runs`` content — the facade renders it as a run
    record when the quality cycle calls ``log_read("collector-runs")``.  The
    response carries the run's tool calls (what it actually did) and
    ``set_run_outcome`` stamps the target/outcome/summary the record header uses.
    ``messages`` is the run's conversation (the final prompt's turns) — pass
    browse-error turns (``_browse_fail_turns``) so the run-health tally can flag
    ``⚠ NO WRITES``; defaults to empty (no browse activity).
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
        messages=messages or [],
        response=response,
        agent_name="collector",
        run_id=run_id,
        run_target=suspect,
    )
    db.messages.set_run_outcome(run_id, outcome.value, summary, tool_failures)


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


def _applied_an_edit(db: Database) -> bool:
    """Quality must SUGGEST, never apply (migration 0073).  ``collection_update``
    is the apply tool — calling it at all, on ANY collection, breaks the
    suggest-only contract.  The sharpest, collection-agnostic 'it mutated' signal,
    and the gate that would catch a regression to self-applying."""
    return tool_was_called(db, "collection_update")


def _names_collection(sent: list[str], name: str) -> bool:
    """Did any sent message name this collection — is the suggestion specific
    enough for the user to act on?"""
    return any(name in message for message in sent)


def _score_suggests(suspect: str) -> CollectorScorer:
    """Drift → quality SENDS a fix suggestion and APPLIES NOTHING.  The user
    approves; the chat agent edits.  This is the suggest-not-apply contract
    (migration 0073) — it replaces the old apply-then-notify scorer that asserted
    the extraction_prompt changed.  The suggestion must be approve-as-is: it names
    the collection AND writes out a complete new prompt as a numbered recipe (the
    user-requested Observed / Proposed fix / New prompt shape — we score the two
    machine-checkable parts: the named collection and the numbered new prompt)."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        fails: list[str] = []
        if _applied_an_edit(db):
            fails.append("called collection_update — quality must suggest, not apply")
        if not sent:
            fails.append("did not send a fix suggestion to the user")
            return fails
        if not _names_collection(sent, suspect):
            fails.append(f"suggestion did not name the {suspect!r} collection (not actionable)")
        if not any(looks_numbered(message) for message in sent):
            fails.append("suggestion did not write out a numbered new prompt to approve")
        return fails

    return _score


def _score_healthy(suspect: str) -> CollectorScorer:
    """Healthy / quiet → quality MUTATES NOTHING and SENDS NOTHING.  An applied
    edit is the over-correction suggest-only exists to prevent; a needless message
    spams the user about a working collector.  Folds the old _score_no_op +
    _score_quiet — under suggest-only both reduce to 'silent + unchanged'."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        fails: list[str] = []
        if _applied_an_edit(db):
            fails.append(f"called collection_update on a healthy collection ({suspect!r})")
        if sent:
            fails.append(f"sent a needless suggestion on a healthy/quiet batch: {sent!r}")
        return fails

    return _score


def _score_called_done(db: Database, before: object, sent: list[str]) -> list[str]:
    """Done-discipline: the cycle must end by calling ``done()`` — whether it
    sent a suggestion or found nothing — never trail off with plain text.
    Recreates the prod give-up (a run returned a text blob and never called
    done())."""
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
    must (a) close with ``done()``, (b) SEND a suggestion naming at least one
    drifted collection, (c) apply NOTHING (no collection_update), and (d) not
    suggest about a healthy one.

    Recreates a production non-convergence: faced with several collections, the
    model grazed ``memory_metadata`` across all of them, filled its context with
    "looks fine", and trailed off with text — suggesting nothing.  One-suggestion-
    per-cycle should make it pick the worst and write it up.
    """

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        fails: list[str] = []
        if not tool_was_called(db, "done"):
            fails.append("cycle ended without calling done() — gave up with plain text")
        if _applied_an_edit(db):
            fails.append("called collection_update — quality must suggest, not apply")
        if not any(_names_collection(sent, name) for name in drifted):
            fails.append("did not send a suggestion naming any drifted collection")
        leaked = [name for name in healthy if _names_collection(sent, name)]
        if leaked:
            fails.append(f"suggested about healthy collection(s): {leaked}")
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
        score=_score_suggests(suspect),
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
        score=_score_suggests(suspect),
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
        score=_score_healthy(suspect),
        min_pass_rate=None,
    )


async def test_repairs_half_formed_send(collector_eval) -> None:
    """A worked run that ALSO sent a half-formed message (``⚠ HALF-FORMED SEND``)
    is a tier-1 regression — the user received junk.  Quality must SUGGEST a fix
    (compose the complete message before the one send) and announce it.

    The observed bad send is an ELLIPSIS-TRUNCATED tail ("Hey there, I wanted to
    tell you..."), which ``classify_run`` still flags ``⚠ HALF-FORMED SEND`` (via
    ``is_truncated``) but which — unlike a degeneration-collapse run ("......???")
    — is NOT an ``is_degenerate_run``.  That matters for what this case MEASURES:
    the quality suggestion may quote the evidence, and a truncation quote survives
    both the send gate AND the agent-loop reroll guard (``is_degenerate_run`` on
    tool-call args), so the case scores JUDGMENT + DELIVERY rather than the accident
    of whether the model happened to paraphrase (see #1386).  A suggestion quoting a
    genuine ``......???`` collapse is still discarded by the reroll guard upstream —
    that remaining gap is pinned in ``tests/agents/test_agentic_loop.py`` and tracked
    as a follow-up; here we hold the case to the shape it can measure cleanly."""
    suspect = "board-game-news"
    await collector_eval(
        case_id="quality-half-formed-send",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="News about newly released board games, delivered to me.",
            intent="Tell me about newly released board games.",
            prompt=_NEWS_NOTIFY_PROMPT,
            runs=[
                {
                    "run_id": "bg-run-1",
                    "outcome": RunOutcome.WORKED,
                    "summary": "delivered news about a new board game",
                    "calls": [
                        (
                            "collection_write",
                            {
                                "memory": suspect,
                                "entries": [
                                    {"key": "ark-nova-2", "content": "Ark Nova 2 announced"}
                                ],
                            },
                        ),
                        ("send_message", {"content": "Hey there, I wanted to tell you..."}),
                        ("send_message", {"content": "A new board game dropped: Ark Nova 2."}),
                        (
                            "done",
                            {"success": True, "summary": "delivered news about a new board game"},
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_suggests(suspect),
        min_pass_rate=None,
    )


async def test_incomplete_run_is_not_drift(collector_eval) -> None:
    """An ``⚠ INCOMPLETE`` run (work landed, hit the step ceiling without done())
    is capacity, NOT drift — quality must leave the prompt alone (over-correction
    guard for the new flag)."""
    suspect = "dev-tools"
    await collector_eval(
        case_id="quality-incomplete-not-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="Notable new developer tools, with a ping on good ones.",
            intent="Track new developer tools and ping me when a good one shows up.",
            prompt=_OK_NEWS_PROMPT,
            runs=[
                {
                    "run_id": "dev-incomplete-1",
                    "outcome": RunOutcome.INCOMPLETE,
                    "summary": "max steps exceeded after writing",
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
        score=_score_healthy(suspect),
        min_pass_rate=None,
    )


async def test_max_steps_no_calls_is_not_drift(collector_eval) -> None:
    """A run that recorded NO tool call and hit the step ceiling — the model spun
    on rejected premature-done()s until max-steps — is capacity, NOT drift, so
    quality must leave the prompt alone.  This is the real entry-#9 shape: a
    healthy collector (its other cycles wrote fine) whose one stalled cycle used to
    render ``⚠ NO WORK DONE`` and trigger a needless rewrite (over-correction
    guard; the classify_run fix renders it ``⚠ INCOMPLETE`` instead)."""
    suspect = "dev-tools"
    await collector_eval(
        case_id="quality-max-steps-no-calls-not-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="Notable new developer tools, with a ping on good ones.",
            intent="Track new developer tools and ping me when a good one shows up.",
            prompt=_OK_NEWS_PROMPT,
            runs=[
                {
                    "run_id": "dev-maxsteps-1",
                    "outcome": RunOutcome.FAILED,
                    "summary": "max steps exceeded — no done() call",
                    "calls": [],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_healthy(suspect),
        min_pass_rate=None,
    )


async def test_tool_failure_is_not_drift(collector_eval) -> None:
    """A ``⚠ TOOL FAILURES`` run (a tool errored and the run kept going) is
    transience, NOT drift — quality must leave the prompt alone."""
    suspect = "dev-tools"
    await collector_eval(
        case_id="quality-tool-failure-not-drift",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="Notable new developer tools, with a ping on good ones.",
            intent="Track new developer tools and ping me when a good one shows up.",
            prompt=_OK_NEWS_PROMPT,
            runs=[
                {
                    "run_id": "dev-toolfail-1",
                    "outcome": RunOutcome.WORKED,
                    "summary": "wrote one tool after a failed browse",
                    "tool_failures": 1,
                    "calls": [
                        ("browse", {"queries": ["new developer tools 2026"]}),
                        (
                            "collection_write",
                            {
                                "memory": suspect,
                                "entries": [{"key": "zed-1.0", "content": "Zed editor hit 1.0"}],
                            },
                        ),
                        (
                            "done",
                            {"success": True, "summary": "wrote one tool after a failed browse"},
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_healthy(suspect),
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
        score=_score_healthy(suspect),
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


async def test_rewrites_prose_to_numbered(collector_eval) -> None:
    """When quality rewrites a drifted collection whose prompt is PROSE, the fix must
    come out as a NUMBERED instruction/tool-call recipe — gpt-oss follows numbered
    lists far more reliably (a prose collector task bails ~60% on the empty user turn;
    the numbered rewrite ~5%).  This uses a CLEAR drift (intent says stay quiet, the
    run sent a ping) so the rewrite is reliably triggered — what we score is that the
    corrected prompt is numbered, not prose.  Quality is NOT asked to hunt for prose
    on its own (that over-corrects healthy collections); it only reformats the prompt
    it is already rewriting to fix a behaviour drift."""
    suspect = "espresso-gear"
    prose_prompt = (
        "Collect espresso equipment worth considering by browsing the web for new "
        "espresso gear and reading the actual pages, then writing what you find into "
        "the espresso-gear collection.  Whenever a write succeeds, send the user a "
        "one-sentence note about the new item with its URL, and finish by calling done."
    )
    await collector_eval(
        case_id="quality-rewrites-prose-to-numbered",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description="A quiet running list of espresso equipment worth considering.",
            intent="Keep a quiet running list of espresso equipment worth considering "
            "— never ping me about it, I'll check the list myself.",
            prompt=prose_prompt,
            runs=[
                {
                    "run_id": "espresso-prose-1",
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
                            {"success": True, "summary": "wrote 1 entry and sent an update"},
                        ),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_suggests(suspect),
        min_pass_rate=None,
    )


async def test_repairs_done_only_bailout(collector_eval) -> None:
    """Tier-0 compliance: a collector that jumped straight to done() WITHOUT calling
    any read/work tool isn't following its own instructions — quality must catch that
    and repair the prompt (a numbered recipe that actually reads) before any
    intent-drift reasoning.  Modeled on real done-only no_work runs — the prose
    collectors bail exactly this way (91 such runs in prod, mostly one prose collector).

    Baseline is RED today: a no_work run renders header-only, so quality is shown a
    clean-looking quiet cycle and does nothing.  Goes green once the facade surfaces
    the tool trace (incl. done) AND the prompt gains the tier-0 check."""
    suspect = WATCHLIST.name
    await collector_eval(
        case_id="quality-repairs-done-only",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description=WATCHLIST.description,
            intent=WATCHLIST_INTENT,
            prompt=WATCHLIST_PROSE_PROMPT,
            runs=[
                {
                    "run_id": "watchlist-bail-done",
                    "outcome": RunOutcome.NO_WORK,
                    "summary": "no new matches this cycle",
                    "calls": [
                        ("done", {"success": True, "summary": "no new matches this cycle"}),
                    ],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_suggests(suspect),
        min_pass_rate=None,
    )


async def test_repairs_zero_tool_bailout(collector_eval) -> None:
    """Tier-0 compliance, the other shape: a run that called NO tools at all (exited
    immediately, no done()) — it never did any work.  Modeled on real zero-tool failed
    runs (37 in prod).  Quality must repair the prompt so the collector actually reads
    before concluding there's nothing to do.

    Baseline RED for the same reason (failed runs render header-only and are skipped)."""
    suspect = WATCHLIST.name
    await collector_eval(
        case_id="quality-repairs-zero-tool",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description=WATCHLIST.description,
            intent=WATCHLIST_INTENT,
            prompt=WATCHLIST_PROSE_PROMPT,
            runs=[
                {
                    "run_id": "watchlist-bail-zero",
                    "outcome": RunOutcome.FAILED,
                    "summary": "exited without calling any tool",
                    "calls": [],
                }
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_suggests(suspect),
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
        score=_score_healthy(suspect),
        min_pass_rate=None,
    )


# ── Fruitless-run drift: persistent ⚠ NO WRITES (the phase-6 contract) ────────

_ROUNDUP_PROMPT = (
    "Collect fresh tech headlines worth a glance.\n"
    "1. browse a tech-news source for today's headlines; read the pages.\n"
    '2. collection_write("tech-roundup", entries=[...]) for each new headline.\n'
    "3. done()."
)
_ROUNDUP_INTENT = "Keep a list of fresh tech headlines — I'll check the list myself."
_ROUNDUP_DESCRIPTION = "A running list of fresh tech headlines."


def _fruitless_run(run_id: str) -> dict:
    """One run that browsed, every source failed, wrote NOTHING, then closed with a
    confabulated success summary — the production fruitless-run shape.  Renders
    ``⚠ NO WRITES`` (browses failed + 0 writes) under a summary that lies."""
    return {
        "run_id": run_id,
        "outcome": RunOutcome.WORKED,
        "summary": "wrote 3 new tech headlines",
        "calls": [
            ("browse", {"queries": ["https://news.example.test/tech"]}),
            ("done", {"success": True, "summary": "wrote 3 new tech headlines"}),
        ],
        "messages": _browse_fail_turns(3),
    }


async def test_persistent_no_writes_is_suggested(collector_eval) -> None:
    """A browse collector that has read NOTHING and written NOTHING cycle after
    cycle (persistent ``⚠ NO WRITES``, each closing with a confabulated "wrote 3"
    summary) is real drift — the source is unreachable.  Quality must read the
    collector's run history, confirm the pattern, and SUGGEST switching the source,
    without applying anything (no collection_update)."""
    suspect = "tech-roundup"
    await collector_eval(
        case_id="quality-persistent-no-writes",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description=_ROUNDUP_DESCRIPTION,
            intent=_ROUNDUP_INTENT,
            prompt=_ROUNDUP_PROMPT,
            runs=[
                _fruitless_run("roundup-1"),
                _fruitless_run("roundup-2"),
                _fruitless_run("roundup-3"),
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_suggests(suspect),
        min_pass_rate=None,
    )


async def test_single_no_writes_is_not_drift(collector_eval) -> None:
    """A SINGLE ``⚠ NO WRITES`` cycle is transience, not drift — a source can fail
    to load once.  With only one fruitless run in the collector's history, quality
    must read that history, see no pattern, and leave the collection alone (no
    suggestion, no mutation).  This is what the run-history check buys: telling a
    one-off from a pattern, so a transient browse failure doesn't spam the user —
    the over-correction guard for the new ``⚠ NO WRITES`` signal."""
    suspect = "tech-roundup"
    await collector_eval(
        case_id="quality-single-no-writes",
        collection=PennyConstants.MEMORY_QUALITY_COLLECTION,
        seed=_seed(
            suspect=suspect,
            description=_ROUNDUP_DESCRIPTION,
            intent=_ROUNDUP_INTENT,
            prompt=_ROUNDUP_PROMPT,
            runs=[_fruitless_run("roundup-1")],
        ),
        snapshot=_snapshot(suspect),
        score=_score_healthy(suspect),
        min_pass_rate=None,
    )
