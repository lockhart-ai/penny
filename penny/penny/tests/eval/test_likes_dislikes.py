"""NL-dispatch contracts for the likes/dislikes memory tools — the chat agent
routing a naive-register preference utterance onto ``collection_write`` /
``collection_delete_entry`` / ``collection_read_latest`` against the ``likes`` and
``dislikes`` collections, driven against the REAL model and scored STRUCTURALLY on
persisted collection state + the tool the model actually called (never wording).

This is the retirement contract for the ``/like`` + ``/unlike`` + ``/dislike`` +
``/undislike`` commands (epic #1445, issue #1451): the slash commands are gone, so
the intent must now dispatch from natural language onto the memory collections that
recall + the ambient extractor already share.  The legacy ``preference`` table is
untouched (its fate is #1301) — these cases exercise the collections only.

  add     — "I'm really into <topic>" → collection_write on "likes"; "I can't
            stand <topic>" → collection_write on "dislikes"; the entry lands in
            the right collection.
  remove  — "forget about <topic>" (a seeded entry) → collection_delete_entry on
            the matched key (matched BY MEANING, not exact text) + entry gone.
  list    — "what do you think I'm into?" → collection_read_latest("likes"); the
            seeded entries stay put (a read, not a mutation).
  no-fire — a passing opinion ("that movie was fine") must NOT write a preference:
            no collection_write / collection_delete_entry against likes/dislikes.
"""

from __future__ import annotations

import json

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import ChatEval, collection_entries
from penny.tests.eval.conftest import _response_tool_calls as response_tool_calls

pytestmark = pytest.mark.eval

_LIKES = "likes"
_DISLIKES = "dislikes"
_WRITE = "collection_write"
_DELETE = "collection_delete_entry"
_READ = "collection_read_latest"


# ── Promptlog scanners (the real record of what the model did this run) ────────


def _memory_args(db: Database, tool_name: str) -> list[str]:
    """Every ``memory`` argument the model passed to ``tool_name`` this run.

    Sourced from the persisted promptlog so a scorer can assert a write/delete/read
    targeted the right collection — the structural counterpart of ``tool_was_called``.
    """
    memories: list[str] = []
    for row in db.messages.recent_prompts(limit=200):
        for call in response_tool_calls(row):
            function = call.get("function", {})
            if function.get("name") != tool_name:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError, TypeError:
                continue
            memory = args.get("memory")
            if isinstance(memory, str):
                memories.append(memory)
    return memories


def _has_entry_mentioning(db: Database, memory: str, token: str) -> bool:
    """True when some entry in ``memory`` carries ``token`` (in its key or content)."""
    token = token.lower()
    return any(
        token in key.lower() or token in content.lower()
        for key, content in collection_entries(db, memory).items()
    )


# ── Seeders ───────────────────────────────────────────────────────────────────


def _seed_like(db: Database, key: str, content: str) -> None:
    db.memory(_LIKES).write([EntryInput(key=key, content=content)], author="user")


def _seed_dislike(db: Database, key: str, content: str) -> None:
    db.memory(_DISLIKES).write([EntryInput(key=key, content=content)], author="user")


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_add(memory: str, other: str, token: str):
    """The utterance must write the preference into ``memory`` (carrying the topic)
    and leave the opposite-valence collection untouched."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if memory not in _memory_args(db, _WRITE):
            fails.append(f"{_WRITE} was not called on '{memory}'")
        if not _has_entry_mentioning(db, memory, token):
            fails.append(f"'{memory}' has no entry mentioning '{token}' after the write")
        if collection_entries(db, other):
            fails.append(f"opposite collection '{other}' was written to")
        return fails

    return score


def _score_remove(memory: str, key: str, token: str):
    """The retraction must delete the seeded entry from ``memory`` by its key —
    matched by meaning (the user names the topic, not the exact key)."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if memory not in _memory_args(db, _DELETE):
            fails.append(f"{_DELETE} was not called on '{memory}'")
        if key in collection_entries(db, memory):
            fails.append(f"entry '{key}' still present in '{memory}' — not removed")
        return fails

    return score


def _score_list(db: Database, before: set[str], reply: str) -> list[str]:
    """A "what do I like?" query reads the likes collection and mutates nothing."""
    fails: list[str] = []
    if _LIKES not in _memory_args(db, _READ):
        fails.append(f"{_READ} was not called on '{_LIKES}'")
    if _memory_args(db, _WRITE) or _memory_args(db, _DELETE):
        fails.append("a listing request mutated the preference collections")
    if len(collection_entries(db, _LIKES)) != 2:
        fails.append("seeded likes entries changed on a listing request")
    return fails


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[str]:
    """A passing opinion must not write a preference to likes/dislikes."""
    fails: list[str] = []
    if _LIKES in _memory_args(db, _WRITE) or _DISLIKES in _memory_args(db, _WRITE):
        fails.append(f"{_WRITE} fired on a passing opinion")
    if _LIKES in _memory_args(db, _DELETE) or _DISLIKES in _memory_args(db, _DELETE):
        fails.append(f"{_DELETE} fired on a passing opinion")
    if collection_entries(db, _LIKES) or collection_entries(db, _DISLIKES):
        fails.append("a preference was recorded from a passing opinion")
    return fails


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_add_like(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="likes-add-like",
        message="I'm really into competitive kite flying these days, love it",
        score=_score_add(_LIKES, _DISLIKES, "kite"),
    )


async def test_add_like_fan_of(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="likes-add-like-fan",
        message="add matcha lattes to my likes, I'm a big fan",
        score=_score_add(_LIKES, _DISLIKES, "matcha"),
    )


async def test_add_dislike(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="dislikes-add-dislike",
        message="ugh, I really can't stand soggy cereal",
        score=_score_add(_DISLIKES, _LIKES, "cereal"),
    )


async def test_remove_like_by_meaning(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="likes-remove-by-meaning",
        message="eh, you can forget about the typewriters",
        seed=lambda db: _seed_like(
            db,
            "collecting vintage typewriters",
            "I've gotten really into collecting vintage typewriters",
        ),
        score=_score_remove(_LIKES, "collecting vintage typewriters", "typewriters"),
    )


async def test_remove_dislike_by_meaning(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="dislikes-remove-by-meaning",
        message="actually I don't mind being on hold anymore, drop that one",
        seed=lambda db: _seed_dislike(
            db, "waiting on hold", "I really hate waiting on hold with customer service"
        ),
        score=_score_remove(_DISLIKES, "waiting on hold", "hold"),
    )


async def test_list_likes(chat_eval: ChatEval) -> None:
    def seed(db: Database) -> None:
        _seed_like(db, "trail running", "I'm really into trail running")
        _seed_like(db, "collecting vinyl records", "big fan of collecting vinyl records")

    await chat_eval(
        case_id="likes-list",
        message="what do you think I'm into?",
        seed=seed,
        score=_score_list,
    )


async def test_no_fire_passing_opinion(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="likes-no-fire",
        message="the documentary last night was pretty forgettable, nothing worth noting",
        score=_score_no_fire,
    )
