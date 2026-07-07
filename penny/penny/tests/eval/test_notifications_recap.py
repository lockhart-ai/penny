"""Survival contract for the notification mute/unmute tool narrations (part of
epic #1478, issue #1481).

The narration seam (#1479) makes each tool result lead with a first-person line
("You paused notifications:"), and the chat recap folds every tool call into
Penny's reply.  A unit test (``tests/tools/test_notifications.py``) proves the
narration STRING exists — but not the thing that actually matters: that the
summary **survives into Penny's reply to the user**.  The model can read "You
paused notifications:" and still answer without mentioning it.  These cases drive
the real chat loop and score the REPLY, so a pass means the summary genuinely
reached the user.

Scored STRUCTURALLY (the reply reflects the action — paused / resumed / already
on), never on exact wording, since the recap is composed fresh each turn.  Each
scorer prints a sample reply so the PR can report ``case | sample text | N score``.
The NL-dispatch contract (which tool fires) lives in ``test_notifications.py``;
here the concern is whether the ACTION is recapped honestly in the reply.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import ChatEval, tool_was_called

pytestmark = pytest.mark.eval

_MUTE = "notifications_mute"
_UNMUTE = "notifications_unmute"

# Action-reflection that proves the mute/unmute summary survived into the reply.
# The model recaps as "Got it—I'll hold off on messages" / "notifications are back
# on" / "they were already on", so match the ACTION SEMANTICS broadly (no leading
# "I" — the model often drops it), never exact wording.
_PAUSED = re.compile(
    r"\b(paus(e|ed|ing)|mut(e|ed|ing)|quiet(ed|ing|ed down)?|stop(ped|ping)? (messag|"
    r"ping|send|proactiv|updat)|hold(ing)? off|won'?t (message|ping|send|bug|bother|"
    r"reach)|no (more|proactive) (messag|ping|update|notif)|silenc(e|ed|ing)|"
    r"turn(ed|ing)? .*off|shush|dial(ed|ing)? .*down|give you (some )?(peace|quiet|"
    r"space))\b",
    re.I,
)
_RESUMED = re.compile(
    r"\b(back (on|in|up)|resum\w*|re-?enabl\w*|\benabl\w*|un-?mut\w*|"
    r"turn(ed|ing)? [^.]*on|switch(ed|ing)? [^.]*on|up and running|running again|"
    r"all ears again|"
    r"(ping|messag|send|updat|notif|hear|reach|nudge)\w*[^.]{0,30}\bagain\b|"
    r"\bagain\b[^.]{0,30}(ping|messag|send|updat|notif)|"
    r"(ping|messag|send|updat|notif)\w* (you|me)( |,)?(again|right away)|"
    r"are (active|on|up and running)|notif\w* are (back|on|up))\b",
    re.I,
)
# The honest no-op: unmute when NOT muted → the reply must say they were already on,
# not claim it flipped something.
_ALREADY_ON = re.compile(
    r"\b(already (on|active|enabled|running|going)|weren'?t (muted|paused|off)|"
    r"wasn'?t (muted|paused|off)|not (muted|paused|off)|never (muted|paused|off|"
    r"turned off)|nothing (to (turn on|change|do)|changed|was muted)|still (on|active)|"
    r"(are|were) already (coming|going out)|no change)\b",
    re.I,
)


def _seed_muted(db: Database) -> None:
    """Start the user already muted — the precondition for an unmute case."""
    db.users.set_muted(TEST_SENDER)


def _score(tool: str, pattern: re.Pattern, label: str):
    """Happy-path scorer: the tool must fire AND the reply must recap the action."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not tool_was_called(db, tool):
            fails.append(f"{tool} was not called — no {label} action to recap")
        if not pattern.search(reply):
            fails.append(
                f"reply did not recap the {label} — summary did not survive into the response"
            )
        print(f"[SURVIVAL {label}] tool={int(tool_was_called(db, tool))} :: {reply[:200]!r}")
        return fails

    return score


def _score_reply_only(pattern: re.Pattern, label: str):
    """No-op scorer: the honest summary must survive into the reply.  The tool may or
    may not be called (the model can recognise the no-op from state), so this scores
    ONLY that the reply honestly reflects the no-op — never a false claim of a flip."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not pattern.search(reply):
            fails.append(f"reply did not honestly recap the {label} — summary did not survive")
        print(f"[SURVIVAL {label}] :: {reply[:200]!r}")
        return fails

    return score


async def test_mute_summary_survives(chat_eval: ChatEval) -> None:
    """ "stop messaging me for a while" → the mute fires and the reply recaps pausing."""
    await chat_eval(
        case_id="notif-recap-mute",
        message="hey, can you stop messaging me for a while? need some quiet",
        score=_score(_MUTE, _PAUSED, "mute"),
        min_pass_rate=0.75,
    )


async def test_unmute_summary_survives(chat_eval: ChatEval) -> None:
    """ "you can message me again" (seeded muted) → the unmute fires and the reply
    recaps resuming."""
    await chat_eval(
        case_id="notif-recap-unmute",
        message="ok, you can start messaging me again",
        seed=_seed_muted,
        score=_score(_UNMUTE, _RESUMED, "unmute"),
        min_pass_rate=0.75,
    )


async def test_unmute_when_on_is_honest(chat_eval: ChatEval) -> None:
    """Not muted → the unmute finds nothing to flip; the reply must say notifications
    were already on, not claim a fresh resume."""
    await chat_eval(
        case_id="notif-recap-unmute-noop",
        message="go ahead and turn your updates back on",
        score=_score_reply_only(_ALREADY_ON, "already-on"),
        min_pass_rate=0.75,
    )
