"""Collection-lifecycle contracts — the chat agent authoring/operating
collections, driven against the REAL model and scored on PERSISTED DB state.

This is the faithful replacement for scripts/prompt_validation/collection_lifecycle.py:
same behavioural cases, but exercising the production prompt + tool surface end to
end (no AST stubs, no hand-built tools), and asserting on what actually landed in
the DB rather than captured tool-call JSON.

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
)

pytestmark = pytest.mark.eval


def _seed_board_games(db: Database) -> None:
    """A fully-formed board-games collection (prompt + goal + cadence + entries)."""
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        interval=3600,
        notify=True,
    )


def _seed_board_games_silent(db: Database) -> None:
    """The same collection, but silent (not notify) — for the notify-on flip."""
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        interval=3600,
        notify=False,
    )


# ── Scorers (read persisted Memory rows) ────────────────────────────────────


def _created_collection(db: Database, before: set[str]):
    created = new_collections(db, before)
    return created[0] if created else None


def _score_create(
    db: Database, before: set[str], *, notify: bool, interval: int | None
) -> list[str]:
    memory = _created_collection(db, before)
    if memory is None:
        return ["no collection created"]
    fails = []
    body = (memory.extraction_prompt or "").lower()
    if "browse" not in body:
        fails.append("extraction_prompt missing browse step")
    # Emission is the ``notify`` flag, NOT a send_message step in the stored
    # extraction_prompt.  The model must map "ping/tell me" onto the flag — the
    # run-time notify suffix (#1557) does the sending, so the stored prompt itself
    # should never call send_message.
    if memory.notify != notify:
        fails.append(f"notify expected {notify}, got {memory.notify}")
    if "send_message" in body:
        fails.append("stored prompt has send_message — notify is the flag, not a send step")
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


def _score_update_source(db: Database, before: set[str], *, url_token: str) -> list[str]:
    """The new source URL must land in the collection's extraction_prompt.

    Changing where a collection gathers from (the source URL the collector browses)
    is a ``collection_update`` of the ``extraction_prompt`` — the same field a scope
    change rewrites.  The contract is the PERSISTED prompt now names the new source;
    a model that only says "done" (no tool call), confabulates the change, or rewrites
    the prompt while dropping the URL all fail this — exactly the production failure
    where three "all set!" replies never wrote the URL.  Match on the host+path token,
    not the verbatim ``https://`` string, so dropping the scheme isn't a false miss.
    """
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    body = memory.extraction_prompt or ""
    if url_token not in body:
        return [f"source URL not applied — {url_token!r} absent from extraction_prompt: {body!r}"]
    return []


def _score_silent_flip(db: Database, before: set[str], reply: str) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    # "stop pinging me" = flip ``notify`` off.  The collector keeps gathering;
    # only the notify side is silenced.
    return [] if not memory.notify else ["still notifying — notify not flipped to false"]


def _score_notify_flip(db: Database, before: set[str], reply: str) -> list[str]:
    memory = db.memories.get("board-games")
    if memory is None:
        return ["board-games disappeared"]
    # "start telling me" = flip ``notify`` on for an existing silent collection.
    return [] if memory.notify else ["did not start notifying — notify not flipped to true"]


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
        score=lambda db, before, reply: _score_create(db, before, notify=True, interval=None),
    )


async def test_create_silent(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="create-silent",
        message="research fountain pens and inks for me — silent, i'll check the list myself",
        score=lambda db, before, reply: _score_create(db, before, notify=False, interval=None),
    )


async def test_create_cadence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="create-cadence",
        message="research new sci-fi novels for me, check daily, ping me when good ones land",
        score=lambda db, before, reply: _score_create(db, before, notify=True, interval=86400),
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


# A distinctive synthetic source the model must browse instead of the generic query
# the seeded prompt currently uses.  Host+path token is what the scorer matches on.
_NEW_SOURCE_URL = "https://tabletop.example.com/hotness"
_NEW_SOURCE_TOKEN = "tabletop.example.com/hotness"


async def test_update_source_url(chat_eval: ChatEval) -> None:
    """Pointing an existing collection at a new source URL must persist into the
    extraction_prompt.

    Contract for the "Change collection source" skill (migration 0070).  The
    failure is an INTERPRETATION gap, not an inability to update: phrased
    explicitly ("update the prompt so it browses this url from now on") the model
    nails it; phrased the way users actually do ("for X you should browse this url
    to find good games") it reads the request as "go browse that now" — a one-shot
    action — and never reconfigures the collector.  Before the skill this case was
    0/8; the skill (seeded by the migration the eval DB runs, so this drives the
    SHIPPED text) frames "point X at this url" as a collection_update and lifts it.
    Only the board-games collection is seeded here — the skill comes from the
    migration, the single source of truth."""
    await chat_eval(
        case_id="update-source-url",
        message=(
            "actually, for the board games collection you should browse this url to "
            f"find good games: {_NEW_SOURCE_URL}"
        ),
        seed=_seed_board_games,
        score=lambda db, before, reply: _score_update_source(
            db, before, url_token=_NEW_SOURCE_TOKEN
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
