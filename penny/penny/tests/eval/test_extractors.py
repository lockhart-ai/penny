"""Core extraction-collector contracts — the background collectors that make up
the bulk of production runs, driven against the REAL model via ``run_for`` on
their CANONICAL migration-seeded extraction prompts.

These collections (``likes``, ``dislikes``, ``knowledge``, ``unnotified-thoughts``,
``notified-thoughts``) already exist with their prompts in a fresh eval DB
(migrations 0027/0031/0033), so each case only seeds the collector's INPUT — the
``user-messages`` / ``browse-results`` logs, or prior thought entries — and
checks the entry-level outcome on the bound collection (diffing before/after).

Every collector is one of two shapes, both covered here:

  read memory/log → write          likes / dislikes / knowledge / notify
  browse → extract → write/notify   research-watcher / inner-monologue

Browse-driven cases inject query-aware canned pages (``browse=``) so the
*subsequent* call (the write, the send) is what gets scored.  Sends are read off
``db.send_queue`` (a cycle enqueues; the drainer doesn't run inside ``run_for``).
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, LogEntryInput, RecallMode
from penny.tests.eval.conftest import CollectorScorer, collection_entries, tool_was_called
from penny.tests.eval.fixtures import (
    KNOWLEDGE_PAGE_CONTENT,
    RESEARCH_PAGES,
    RESEARCH_WATCHER,
    RESEARCH_WATCHER_EXTRACTION_PROMPT,
    RESEARCH_WATCHER_INTENT,
    THINKING_PAGES,
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


def _seed_unnotified(entries: list[EntryInput]):
    def _apply(db: Database) -> None:
        db.memory("unnotified-thoughts").write(entries, author="thinking")

    return _apply


def _seed_research_watcher(db: Database) -> None:
    db.memories.create_collection(
        RESEARCH_WATCHER.name,
        RESEARCH_WATCHER.description,
        Inclusion(RESEARCH_WATCHER.inclusion),
        RecallMode.RELEVANT,
        extraction_prompt=RESEARCH_WATCHER_EXTRACTION_PROMPT,
        intent=RESEARCH_WATCHER_INTENT,
        collector_interval_seconds=3600,
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
    return fails


def _score_notify(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, "notified-thoughts")
    fails = []
    if not sent:
        fails.append("did not send a thought to the user")
    if not (set(after) - set(before_entries)):
        fails.append("did not move the shared thought into notified-thoughts")
    return fails


def _score_research(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, RESEARCH_WATCHER.name)
    fails = []
    if not (set(after) - set(before_entries)):
        fails.append("did not write the browsed find to the collection")
    if not sent:
        fails.append("did not ping the user about the new find")
    elif not any("http" in message for message in sent):
        fails.append("notification carried no source URL")
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


async def test_notify_send_and_move(collector_eval) -> None:
    await collector_eval(
        case_id="notify-send-and-move",
        collection="notified-thoughts",
        seed=_seed_unnotified(
            [
                EntryInput(
                    key="tidewatch co-op board game",
                    content="Found a neat new co-op board game: Tidewatch — modular ocean "
                    "board, 60-minute play time. https://bgnews.example.com/tidewatch 🌊",
                )
            ]
        ),
        snapshot=_snapshot("notified-thoughts"),
        score=_score_notify,
    )


# ── Cases: browse → extract → write/notify ───────────────────────────────────


async def test_collector_research_browse(collector_eval) -> None:
    await collector_eval(
        case_id="collector-research-browse",
        collection=RESEARCH_WATCHER.name,
        seed=_seed_research_watcher,
        snapshot=_snapshot(RESEARCH_WATCHER.name),
        browse=list(RESEARCH_PAGES),
        score=_score_research,
    )


async def test_thinking_generate(collector_eval) -> None:
    await collector_eval(
        case_id="thinking-generate",
        collection="unnotified-thoughts",
        seed=_seed_like,
        snapshot=_snapshot("unnotified-thoughts"),
        browse=list(THINKING_PAGES),
        score=_score_wrote_entry("unnotified-thoughts"),
        min_pass_rate=None,  # report-only: read-like → browse → draft → dedup → write is long
    )
