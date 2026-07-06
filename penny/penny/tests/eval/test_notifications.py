"""NL-dispatch contracts for the notification mute/unmute tools — the chat agent
routing a naive-register utterance to ``notifications_mute`` /
``notifications_unmute``, driven against the REAL model and scored on the
PERSISTED ``MuteState`` row + the tool the model actually called.

This is the retirement contract for the ``/mute`` + ``/unmute`` commands (epic
#1445, issue #1447): the slash commands are gone, so the intent must now dispatch
from natural language.  Every case asserts STRUCTURALLY — which tool fired (from
the persisted promptlog) and whether the mute row is present/absent afterward —
never on the reply's wording, which is stochastic.

  mute      — "stop messaging me for a while", "quiet down" → notifications_mute
              called + MuteState present.
  unmute    — "you can message me again", "turn updates back on" (seeded muted)
              → notifications_unmute called + MuteState absent.
  no-fire   — a casual mention ("it's been a quiet day") must NOT call either
              tool and must leave the mute state untouched.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import ChatEval, tool_was_called

pytestmark = pytest.mark.eval

_MUTE = "notifications_mute"
_UNMUTE = "notifications_unmute"


def _seed_muted(db: Database) -> None:
    """Start the user already muted — the precondition for an unmute case."""
    db.users.set_muted(TEST_SENDER)


# ── Scorers (read the persisted MuteState row + the promptlog tool calls) ─────


def _score_mute(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not tool_was_called(db, _MUTE):
        fails.append(f"{_MUTE} not called")
    if tool_was_called(db, _UNMUTE):
        fails.append(f"{_UNMUTE} called on a mute request")
    if not db.users.is_muted(TEST_SENDER):
        fails.append("notifications not muted — MuteState row absent")
    return fails


def _score_unmute(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not tool_was_called(db, _UNMUTE):
        fails.append(f"{_UNMUTE} not called")
    if tool_was_called(db, _MUTE):
        fails.append(f"{_MUTE} called on an unmute request")
    if db.users.is_muted(TEST_SENDER):
        fails.append("notifications still muted — MuteState row present")
    return fails


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if tool_was_called(db, _MUTE):
        fails.append(f"{_MUTE} fired on a casual mention")
    if tool_was_called(db, _UNMUTE):
        fails.append(f"{_UNMUTE} fired on a casual mention")
    if db.users.is_muted(TEST_SENDER):
        fails.append("mute state changed on a casual mention")
    return fails


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_mute_stop_messaging(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-stop-messaging",
        message="hey, can you stop messaging me for a while? need some quiet",
        score=_score_mute,
    )


async def test_mute_quiet_down(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-quiet-down",
        message="quiet down please — no proactive updates for now",
        score=_score_mute,
    )


async def test_unmute_message_again(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-message-again",
        message="ok, you can start messaging me again",
        seed=_seed_muted,
        score=_score_unmute,
    )


async def test_unmute_turn_back_on(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-turn-back-on",
        message="go ahead and turn your updates back on",
        seed=_seed_muted,
        score=_score_unmute,
    )


async def test_no_fire_casual_mention(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="notifications-no-fire",
        message="it's been a quiet day today, not much going on honestly",
        score=_score_no_fire,
    )
