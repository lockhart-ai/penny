"""Core extraction-collector contracts — the background collectors that make up
the bulk of production runs, driven against the REAL model via ``run_for`` on
their CANONICAL migration-seeded extraction prompts.

These collections (``likes``, ``dislikes``, ``knowledge``, ``thoughts``) already
exist with their prompts in a fresh eval DB (migrations 0027/0031/0033/0068), so
each case only seeds the collector's INPUT — the ``user-messages`` /
``browse-results`` logs, or prior entries — and checks the entry-level outcome on
the bound collection (diffing before/after).

Every collector is one of two shapes, both covered here:

  read memory/log → write          likes / dislikes / knowledge
  browse → extract → write          thoughts (inner-monologue) / research-watcher

These cases seed ``notify=False`` collections — pure gather-and-write, no
emission.  Emission is now a collection PROPERTY (#1557): a ``notify=True``
collection gets the run-time notify steps appended before its injected terminal
``done()`` and sends in the same cycle; that STOP/send mechanic is pinned in
``tests/agents/test_collector.py``.

Browse-driven cases inject query-aware canned pages (``browse=``) so the
*subsequent* call (the write, the send) is what gets scored.  Sends are read off
``db.send_queue`` (a cycle enqueues; the drainer doesn't run inside ``run_for``).
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.tests.eval.conftest import (
    CollectorScorer,
    _InjectEmptyResponse,
    collection_entries,
    seed_collection,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    COLLECTOR_DONE_JSON_BAIL,
    COLLECTOR_PROSE_BAIL,
    KNOWLEDGE_PAGE_CONTENT,
    RESEARCH_PAGES,
    RESEARCH_WATCHER,
    RESEARCH_WATCHER_EXTRACTION_PROMPT,
    THINKING_PAGES,
    WATCHLIST,
    WATCHLIST_MESSAGES,
    WATCHLIST_NUMBERED_PROMPT,
)

pytestmark = pytest.mark.eval

_INCOMING = PennyConstants.MessageDirection.INCOMING


# ── Seeders ──────────────────────────────────────────────────────────────────


def _seed_user_messages(*messages: str):
    """Seed incoming user messages (the ``user-messages`` log is a facade over
    ``messagelog`` — seed the canonical table)."""

    def _apply(db: Database) -> None:
        for message in messages:
            db.messages.log_message(_INCOMING, "user", message)

    return _apply


def _seed_browse_results(content: str):
    def _apply(db: Database) -> None:
        db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
            [LogEntryInput(content=content)], author="chat"
        )

    return _apply


def _seed_research_watcher(db: Database) -> None:
    # A silent gatherer (``notify=False``): it gathers and writes, never sends.
    # A ``notify=True`` collection would append the notify suffix and send in the
    # same cycle — that mechanic is pinned in tests/agents/test_collector.py.
    db.memories.create_collection(
        RESEARCH_WATCHER.name,
        RESEARCH_WATCHER.description,
        extraction_prompt=RESEARCH_WATCHER_EXTRACTION_PROMPT,
        collector_interval_seconds=3600,
        notify=False,
    )


def _seed_like(db: Database) -> None:
    db.memory("likes").write(
        [EntryInput(key="tabletop board games", content="I love tabletop board games")],
        author="history",
    )


def _snapshot(name: str):
    def _take(db: Database) -> dict[str, str]:
        return collection_entries(db, name)

    return _take


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_wrote_entry(name: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, name)
        if set(after) - set(before_entries):
            return []
        return [f"expected a new {name!r} entry, none added"]

    return _score


def _score_no_op(name: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        if collection_entries(db, name) != before_entries:
            return [f"wrote a {name!r} entry on a no-signal batch (false positive)"]
        return []

    return _score


def _score_knowledge(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, "knowledge")
    new_keys = set(after) - set(before_entries)
    if not new_keys:
        return ["no knowledge entry written from the browse-results page"]
    fails = []
    body = " ".join(after[key].lower() for key in new_keys)
    if "antikythera" not in body:
        fails.append("summary missing the page's subject (antikythera)")
    if "http" not in body:
        fails.append("summary missing the source URL (should lead with it)")
    # The cycle must close with a real done() call — a run that writes the entry
    # then narrates "Done. Summary: ..." as prose instead of calling done() is
    # marked failed and leaves its cursor uncommitted (re-run next tick).  The
    # text-step nudge exists to keep that slip from ending the cycle.
    if not tool_was_called(db, "done"):
        fails.append("wrote the entry but never closed the cycle with done()")
    return fails


def _score_research(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, RESEARCH_WATCHER.name)
    fails = []
    if not (set(after) - set(before_entries)):
        fails.append("did not write the browsed find to the collection")
    # A silent (``notify=False``) gatherer writes only — with no notify suffix in
    # its prompt it must NOT send.
    if sent:
        fails.append("silent collector sent a message — a notify=False cycle never emits")
    if not tool_was_called(db, "done"):
        fails.append("cycle did not close with done()")
    return fails


# ── Cases: read memory/log → write ───────────────────────────────────────────


async def test_extract_likes(collector_eval) -> None:
    await collector_eval(
        case_id="extract-likes",
        collection="likes",
        seed=_seed_user_messages(
            "honestly i've been obsessed with single-origin pour-over coffee lately",
            "what time is it in tokyo right now?",
        ),
        snapshot=_snapshot("likes"),
        score=_score_wrote_entry("likes"),
    )


async def test_extract_likes_quiet(collector_eval) -> None:
    await collector_eval(
        case_id="extract-likes-quiet",
        collection="likes",
        seed=_seed_user_messages(
            "what's the capital of australia?",
            "can you convert 30 celsius to fahrenheit?",
            "remind me what we talked about yesterday",
        ),
        snapshot=_snapshot("likes"),
        score=_score_no_op("likes"),
    )


async def test_extract_dislikes(collector_eval) -> None:
    await collector_eval(
        case_id="extract-dislikes",
        collection="dislikes",
        seed=_seed_user_messages(
            "ugh i really can't stand cilantro, it ruins every dish for me",
            "anyway, what's the weather tomorrow?",
        ),
        snapshot=_snapshot("dislikes"),
        score=_score_wrote_entry("dislikes"),
    )


async def test_extract_knowledge(collector_eval) -> None:
    await collector_eval(
        case_id="extract-knowledge",
        collection="knowledge",
        seed=_seed_browse_results(KNOWLEDGE_PAGE_CONTENT),
        snapshot=_snapshot("knowledge"),
        score=_score_knowledge,
    )


# ── Cases: browse → extract → write ──────────────────────────────────────────


async def test_collector_research_browse(collector_eval) -> None:
    await collector_eval(
        case_id="collector-research-browse",
        collection=RESEARCH_WATCHER.name,
        seed=_seed_research_watcher,
        snapshot=_snapshot(RESEARCH_WATCHER.name),
        browse=list(RESEARCH_PAGES),
        score=_score_research,
    )


def _seed_watchlist(db: Database) -> None:
    seed_collection(
        db,
        WATCHLIST,
        extraction_prompt=WATCHLIST_NUMBERED_PROMPT,
        interval=3600,
    )
    for message in WATCHLIST_MESSAGES:
        db.messages.log_message(_INCOMING, "user", message)


async def test_collector_recovers_from_text_bail(nudge_eval) -> None:
    """Contract: a collector that emits plain text mid-cycle (instead of a tool
    call) is nudged back to a tool call and recovers to a clean ``done()`` close
    — rather than the loop treating the text as a final answer and ending the
    cycle failed with an uncommitted cursor.

    The ~25% terminal slip can't be reproduced reliably by seeding, so the
    harness forces one plain-text bail right after the model's first tool call;
    the real model then drives the recovery through the production text-step
    nudge.  This is the durable, live-model definition of the nudge contract
    (the mechanism itself is covered deterministically by
    ``test_agentic_loop.TestCollectorTextNudge``)."""
    await nudge_eval(
        case_id="collector-text-bail-recovery",
        collection=WATCHLIST.name,
        seed=_seed_watchlist,
        bail_text=COLLECTOR_PROSE_BAIL,
    )


def _score_taught_real_done(db: Database, before: object, sent: list[str]) -> list[str]:
    """The teaching worked: the live model MADE the real done() tool call after the
    nudge (the promptlog records the model's own emission — the injected bail was
    text, so any logged done() call is the model's recovery move).  ``nudge_eval``
    separately asserts the cycle closed successfully; this pins that the close came
    from the model's own tool call, i.e. the taught behaviour, not any absorption
    by the system."""
    if tool_was_called(db, "done"):
        return []
    return ["no real done() call was logged — the model never re-emitted the taught call"]


async def test_collector_taught_out_of_args_only_json_bail(nudge_eval) -> None:
    """Contract: a collector that emits the argless done() call as a JSON text
    envelope (``{"name": "done", "arguments": {}}``) — gpt-oss's native
    Harmony-backend fallback and the dominant call-shaped text bail in production —
    receives the shape-specific TEACHING nudge (``COLLECTOR_DONE_JSON_NUDGE``: what
    it did, and the exact argless ``done()`` call to make) and recovers by MAKING
    the real done() tool call; the cycle completes.  One extra round-trip versus a
    repair is accepted and expected — the model, not the system, must emit the call
    (reject-and-teach: repairs are reserved for transport-mangled calls, not
    model-authored malformations).

    The harness forces one JSON bail right after the model's first tool call; the
    live model then drives the recovery through the production teaching nudge.
    ``_score_taught_real_done`` asserts the real done() call was logged (the
    mechanism itself is covered deterministically by
    ``test_agentic_loop.TestCollectorDoneJsonBailNudge``)."""
    await nudge_eval(
        case_id="collector-done-json-bail-teaching",
        collection=WATCHLIST.name,
        seed=_seed_watchlist,
        bail_text=COLLECTOR_DONE_JSON_BAIL,
        score=_score_taught_real_done,
    )


def _score_watchlist_recovered(db: Database, before: object, sent: list[str]) -> list[str]:
    """Pass iff the cycle recovered from the forced empty response with a REAL tool
    call — it wrote the watchlist entries the seeded messages clearly warrant, not
    just any close.  (``nudge_eval`` separately asserts the cycle closed with a
    genuine ``done()``; this proves the recovery move was real work, not prose.)"""
    if collection_entries(db, WATCHLIST.name) and tool_was_called(db, "collection_write"):
        return []
    return ["did not recover with a real write after the forced empty response"]


async def test_collector_recovers_from_empty_response(nudge_eval) -> None:
    """Contract: a collector that returns EMPTY content mid-cycle (no text, no tool
    call) is nudged for a TOOL CALL — the collector nudge demands one and names
    done(), not the chat 'Please provide your response.' that would invite an
    unparseable prose reply — and recovers to a real write + a genuine done().

    The empty slip is stochastic, so the harness forces one empty response right
    after the model's first tool call; the live model then drives the recovery
    through the production ``COLLECTOR_CONTINUE_NUDGE`` (the mechanism itself is
    covered deterministically by ``test_agentic_loop.TestCollectorEmptyNudge``)."""
    await nudge_eval(
        case_id="collector-empty-response-recovery",
        collection=WATCHLIST.name,
        seed=_seed_watchlist,
        wrap=_InjectEmptyResponse,
        score=_score_watchlist_recovered,
    )


async def test_thinking_generate(collector_eval) -> None:
    await collector_eval(
        case_id="thinking-generate",
        collection="thoughts",
        seed=_seed_like,
        snapshot=_snapshot("thoughts"),
        browse=list(THINKING_PAGES),
        score=_score_wrote_entry("thoughts"),
        min_pass_rate=None,  # report-only: read-like → browse → draft → dedup → write is long
    )
