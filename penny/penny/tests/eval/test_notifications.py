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
from penny.tests.eval.conftest import ChatEval, Check, tool_not_called, tool_was_called

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "nl-dispatch"

_MUTE = "notifications_mute"
_UNMUTE = "notifications_unmute"


def _seed_muted(db: Database) -> None:
    """Start the user already muted — the precondition for an unmute case."""
    db.users.set_muted(TEST_SENDER)


# ── Scorers (read the persisted MuteState row + the promptlog tool calls) ─────


def _score_mute(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        Check(
            "notifications_mute called",
            tool_was_called(db, _MUTE),
            anchor=f"{_MUTE}(",
            kind="spine",
        ),
        Check(
            "notifications_unmute not called",
            tool_not_called(db, _UNMUTE),
            anchor=f"{_UNMUTE}(",
            kind="spine",
        ),
        Check(
            "notifications muted (MuteState present)",
            db.users.is_muted(TEST_SENDER),
            kind="state",
        ),
    ]


def _score_unmute(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        Check(
            "notifications_unmute called",
            tool_was_called(db, _UNMUTE),
            anchor=f"{_UNMUTE}(",
            kind="spine",
        ),
        Check(
            "notifications_mute not called",
            tool_not_called(db, _MUTE),
            anchor=f"{_MUTE}(",
            kind="spine",
        ),
        Check(
            "notifications unmuted (MuteState absent)",
            not db.users.is_muted(TEST_SENDER),
            kind="state",
        ),
    ]


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        Check(
            "notifications_mute not fired",
            tool_not_called(db, _MUTE),
            anchor=f"{_MUTE}(",
            kind="spine",
        ),
        Check(
            "notifications_unmute not fired",
            tool_not_called(db, _UNMUTE),
            anchor=f"{_UNMUTE}(",
            kind="spine",
        ),
        Check("mute state unchanged", not db.users.is_muted(TEST_SENDER), kind="state"),
    ]


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_mute_stop_messaging(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-stop-messaging",
        family=_FAMILY,
        message="hey, can you stop messaging me for a while? need some quiet",
        score=_score_mute,
    )


async def test_mute_quiet_down(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-quiet-down",
        family=_FAMILY,
        message="quiet down please — no proactive updates for now",
        score=_score_mute,
    )


async def test_unmute_message_again(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-message-again",
        family=_FAMILY,
        message="ok, you can start messaging me again",
        seed=_seed_muted,
        score=_score_unmute,
    )


async def test_unmute_turn_back_on(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-turn-back-on",
        family=_FAMILY,
        message="go ahead and turn your updates back on",
        seed=_seed_muted,
        score=_score_unmute,
    )


async def test_no_fire_casual_mention(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="notifications-no-fire",
        family=_FAMILY,
        message="it's been a quiet day today, not much going on honestly",
        score=_score_no_fire,
    )
