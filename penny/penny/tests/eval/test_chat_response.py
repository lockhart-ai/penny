"""Chat-response contracts — the chat agent's day-to-day replies (the #1
user-facing prompt type), driven against the REAL model and scored on
PERSISTED state + the reply text.

The collection-lifecycle suite already covers the chat agent *authoring*
collections; this covers the other branch — *responding*.  Every Penny reply is
either answered from system context (recall routed into the prompt) or reached
for the web (a ``browse`` call it then reasons over), so the cases span:

  chitchat       — casual turn → coherent reply, no spurious tool/collection.
  recall-answer  — a memory question → answered from the routed-in collection.
  browse-answer  — a factual question → browse → extract the fact → answer.
  browse-multihop— the fact lives one link deep → chain a second browse to it.

Browse cases inject query-aware canned pages (``browse=`` on chat_eval) so we
score the model's *subsequent* reasoning, not merely that it called browse.
Scoring is on BEHAVIOUR (tool called, fact surfaced, nothing spurious created),
never on exact wording — the model is stochastic.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    new_collections,
    seed_collection,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    MULTIHOP_PAGES,
    VERSION_PAGES,
)

pytestmark = pytest.mark.eval

# Seeded board-game entry names — a recall answer should name at least one.
_SEEDED_GAMES = ("brass", "ark nova", "twilight struggle", "spirit island")


def _has_emoji(text: str) -> bool:
    """True if the text carries an emoji — the chat voice ends every message
    with one, so its presence is a cheap voice signal."""
    return any(ord(char) >= 0x1F000 or 0x2600 <= ord(char) <= 0x27BF for char in text)


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_chitchat(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not reply.strip():
        fails.append("empty reply")
    if not _has_emoji(reply):
        fails.append("reply has no emoji (chat voice ends with one)")
    if new_collections(db, before):
        fails.append("created a collection on plain chitchat")
    if tool_was_called(db, "browse"):
        fails.append("browsed the web for a no-lookup chitchat turn")
    return fails


def _score_recall_answer(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not any(game in reply.lower() for game in _SEEDED_GAMES):
        fails.append("reply named no seeded game — did not answer from memory")
    if tool_was_called(db, "browse"):
        fails.append("browsed instead of answering from the routed-in collection")
    if new_collections(db, before):
        fails.append("created a collection when just answering a memory question")
    return fails


def _score_browse_answer(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not tool_was_called(db, "browse"):
        fails.append("did not browse for a current-info question")
    if "4.2" not in reply:
        fails.append("reply missing the version fact (4.2) from the browsed page")
    return fails


def _score_browse_multihop(db: Database, before: set[str], reply: str) -> list[str]:
    # The release year lives ONLY on the linked detail page, so a reply that
    # cites it proves the model chained a second browse to the URL it found.
    if "2031" not in reply:
        return ["reply missing the release year (2031) — did not chain to the detail page"]
    return []


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_chitchat(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-chitchat",
        message="hey! good morning, how's your day going so far?",
        score=_score_chitchat,
    )


async def test_recall_answer(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-recall-answer",
        message="remind me which board games we'd flagged as worth buying",
        seed=lambda db: seed_collection(db, BOARD_GAMES),
        score=_score_recall_answer,
    )


async def test_browse_answer(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-browse-answer",
        message="what's the latest stable version of the quillpad note-taking app?",
        browse=list(VERSION_PAGES),
        score=_score_browse_answer,
    )


async def test_browse_multihop(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-browse-multihop",
        message="what year did the strategy game mistforge tactics come out? "
        "check the official title page if you need to.",
        browse=list(MULTIHOP_PAGES),
        score=_score_browse_multihop,
        min_pass_rate=None,  # report-only: a two-hop browse chain is stochastic
    )
