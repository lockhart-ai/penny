"""Collection-lifecycle contracts — the chat agent authoring/operating
collections, driven against the REAL model and scored on PERSISTED DB state.

This is the faithful replacement for scripts/prompt_validation/collection_lifecycle.py:
same behavioural cases, but exercising the production prompt + tool surface +
recall path end to end (no AST stubs, no hand-built tools), and asserting on
what actually landed in the DB rather than captured tool-call JSON.

    create  — notify, silent, cadence-in-request
    update  — broaden scope, flip notify→silent
    archive — done phrasing
    abstain — implicit trip-prep should NOT create a collection
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, new_collections, seed_collection
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    BOARD_GAMES_EXTRACTION_PROMPT,
    BOARD_GAMES_INTENT,
)

pytestmark = pytest.mark.eval


def _seed_board_games(db: Database) -> None:
    """A fully-formed board-games collection (prompt + goal + cadence + entries)."""
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        intent=BOARD_GAMES_INTENT,
        interval=3600,
        published=True,
    )


def _seed_board_games_silent(db: Database) -> None:
    """The same collection, but silent (not published) — for the notify-on flip."""
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        intent=BOARD_GAMES_INTENT,
        interval=3600,
        published=False,
    )


# ── Scorers (read persisted Memory rows) ────────────────────────────────────


def _created_collection(db: Database, before: set[str]):
    created = new_collections(db, before)
    return created[0] if created else None


def _score_create(
    db: Database, before: set[str], *, inclusion: str, published: bool, interval: int | None
) -> list[str]:
    memory = _created_collection(db, before)
    if memory is None:
        return ["no collection created"]
    fails = []
    if memory.inclusion != inclusion:
        fails.append(f"inclusion expected {inclusion!r}, got {memory.inclusion!r}")
    body = (memory.extraction_prompt or "").lower()
    if "browse" not in body:
        fails.append("extraction_prompt missing browse step")
    # Pub/sub model: notify-on-new is the ``published`` flag, NOT a send_message
    # step in the producer prompt.  The model must map "ping/tell me" onto the
    # flag — and producers never send, so no producer prompt should call it.
    if memory.published != published:
        fails.append(f"published expected {published}, got {memory.published}")
    if "send_message" in body:
        fails.append("producer prompt has send_message — notify is the published flag, not a send")
    if interval is not None and memory.collector_interval_seconds != interval:
        fails.append(f"interval expected {interval}, got {memory.collector_interval_seconds}")
    return fails


def _score_update_scope(db: Database, before: set[str], *, added: tuple[str, ...]) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    text = f"{memory.description}\n{memory.extraction_prompt or ''}".lower()
    if not any(term in text for term in added):
        return [f"scope not broadened — none of {added} in description/extraction_prompt"]
    return []


def _score_silent_flip(db: Database, before: set[str], reply: str) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    # "stop pinging me" = flip ``published`` off.  The collector keeps gathering;
    # only the notify side is silenced.
    return [] if not memory.published else ["still publishing — published not flipped to false"]


def _score_notify_flip(db: Database, before: set[str], reply: str) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    # "start telling me" = flip ``published`` on for an existing silent collection.
    return [] if memory.published else ["did not start publishing — published not flipped to true"]


def _score_archive(db: Database, before: set[str], reply: str) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    return [] if memory.archived else ["collection not archived"]


def _score_no_create(db: Database, before: set[str], reply: str) -> list[str]:
    created = new_collections(db, before)
    if created:
        return [f"created a collection on an ambiguous request: {[m.name for m in created]}"]
    return []


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_create_notify(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="create-notify",
        message="research heavier euro-style strategy board games for me, "
        "ping me when you find good ones",
        score=lambda db, before, reply: _score_create(
            db, before, inclusion="relevant", published=True, interval=None
        ),
    )


async def test_create_silent(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="create-silent",
        message="research fountain pens and inks for me — silent, i'll check the list myself",
        score=lambda db, before, reply: _score_create(
            db, before, inclusion="never", published=False, interval=None
        ),
    )


async def test_create_cadence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="create-cadence",
        message="research new sci-fi novels for me, check daily, ping me when good ones land",
        score=lambda db, before, reply: _score_create(
            db, before, inclusion="relevant", published=True, interval=86400
        ),
    )


async def test_update_add_scope(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="update-add-scope",
        message="add solo and co-op board games to the board games collection too",
        seed=_seed_board_games,
        score=lambda db, before, reply: _score_update_scope(
            db, before, added=("solo", "co-op", "cooperative")
        ),
    )


async def test_update_silent_flip(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="update-silent-flip",
        message="stop pinging me about new board game finds, i'll just check the collection myself",
        seed=_seed_board_games,
        score=_score_silent_flip,
    )


async def test_update_notify_flip(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="update-notify-flip",
        message="actually, start telling me when you find new board games",
        seed=_seed_board_games_silent,
        score=_score_notify_flip,
    )


async def test_archive_done(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="archive-done",
        message="i'm done collecting board games, archive that one",
        seed=_seed_board_games,
        score=_score_archive,
    )


async def test_abstain_implicit_prep(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="abstain-implicit-prep",
        message="booked a cabin trip for october, 10 days off-grid. starting to plan.",
        score=_score_no_create,
    )
