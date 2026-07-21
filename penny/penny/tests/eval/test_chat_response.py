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
never on exact wording — the model is stochastic.  Each case scores as a graded
``Check`` list (partial credit per named expectation, with an observed-vs-expected
``rationale`` on a miss) and carries an explicit ``family`` tag
(chitchat / recall / browse-answer).
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    new_collections,
    seed_collection,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    MULTIHOP_PAGES,
    TOPIC_PAGES,
)

pytestmark = pytest.mark.eval

# Seeded board-game entry names — a recall answer should name at least one.
_SEEDED_GAMES = ("brass", "ark nova", "twilight struggle", "spirit island")


def _has_emoji(text: str) -> bool:
    """True if the text carries an emoji — the chat voice ends every message
    with one, so its presence is a cheap voice signal."""
    return any(ord(char) >= 0x1F000 or 0x2600 <= ord(char) <= 0x27BF for char in text)


# ── Scorers (graded Check lists — each expectation is one named check) ────────


def _score_chitchat(db: Database, before: set[str], reply: str) -> list[Check]:
    created = new_collections(db, before)
    return [
        Check("reply is non-empty", bool(reply.strip())),
        Check(
            "reply carries the chat voice (an emoji)",
            _has_emoji(reply),
            anchor=REPLY_ANCHOR,
            rationale=None if _has_emoji(reply) else "no emoji in the reply",
        ),
        Check(
            "no collection created on plain chitchat",
            not created,
            rationale=None if not created else f"created {[m.name for m in created]}",
        ),
        Check("no browse on a no-lookup chitchat turn", tool_not_called(db, "browse")),
    ]


def _score_recall_answer(db: Database, before: set[str], reply: str) -> list[Check]:
    named = any(game in reply.lower() for game in _SEEDED_GAMES)
    created = new_collections(db, before)
    return [
        Check(
            "named a seeded game (answered from memory)",
            named,
            anchor=REPLY_ANCHOR,
            rationale=None if named else f"none of {_SEEDED_GAMES} in the reply",
        ),
        Check("answered from the collection, did not browse", tool_not_called(db, "browse")),
        Check(
            "no collection created when just answering a memory question",
            not created,
            rationale=None if not created else f"created {[m.name for m in created]}",
        ),
    ]


def _score_browse_answer(db: Database, before: set[str], reply: str) -> list[Check]:
    surfaced = "baikal" in reply.lower()
    return [
        Check(
            "browsed for a current-info question", tool_was_called(db, "browse"), anchor="browse("
        ),
        Check(
            "reply surfaces the browsed fact (Lake Baikal)",
            surfaced,
            anchor=REPLY_ANCHOR,
            rationale=None if surfaced else "'baikal' absent from the reply",
        ),
    ]


def _score_browse_multihop(db: Database, before: set[str], reply: str) -> list[Check]:
    # The release year lives ONLY on the linked detail page, so a reply that
    # cites it proves the model chained a second browse to the URL it found.
    chained = "2031" in reply
    return [
        Check(
            "chained to the detail page (cites the release year 2031)",
            chained,
            anchor=REPLY_ANCHOR,
            rationale=None if chained else "release year 2031 absent — no second hop",
        ),
    ]


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_chitchat(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-chitchat",
        message="hey! good morning, how's your day going so far?",
        score=_score_chitchat,
        family="chitchat",
    )


async def test_recall_answer(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-recall-answer",
        message="remind me which board games we'd flagged as worth buying",
        seed=lambda db: seed_collection(db, BOARD_GAMES),
        score=_score_recall_answer,
        family="recall",
    )


async def test_browse_answer(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-browse-answer",
        message="what's the deepest lake in the world?",
        browse=list(TOPIC_PAGES),
        score=_score_browse_answer,
        family="browse-answer",
    )


async def test_browse_multihop(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-browse-multihop",
        message="what year did the strategy game mistforge tactics come out? "
        "check the official title page if you need to.",
        browse=list(MULTIHOP_PAGES),
        score=_score_browse_multihop,
        min_pass_rate=None,  # report-only: a two-hop browse chain is stochastic
        family="browse-answer",
    )
