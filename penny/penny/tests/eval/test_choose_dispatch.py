"""NL-dispatch contract for the `choose` tool (#1679/#1680).

The model is a biased chooser — asked to "pick one at random" it gravitates to
the first/most-salient option — so random choice belongs in Python.  The
dispatch was PRE-VALIDATED against a synthetic stub carrying this name +
description (fires 1.00 with options intact and the tool's pick reported;
no-fire guard 5/5 on judgment asks); these cases now run against the REAL
registered tool and are its standing contract:

  * "choose one of X, Y, Z" → the tool is CALLED with the options, and the
    reply reports the pick the TOOL returned (read from its persisted result —
    a reply naming a different option means she free-chose past the tool).
  * a JUDGMENT ask over the same options ("which do you think is best?") must
    NOT fire it — an opinion is hers to give, not a coin flip.
"""

from __future__ import annotations

import json
import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, Check, routing_clean

pytestmark = pytest.mark.eval

_OPTIONS = ["cedar", "maple", "birch"]

# The real tool's result body (whole-render pinned in its unit tests).
_PICK_PATTERN = re.compile(r"Chose '([^']+)' at random")


def _choose_calls(db: Database) -> list[dict]:
    """Every `choose` call's parsed arguments, from the persisted promptlog."""
    calls: list[dict] = []
    for row in db.messages.recent_prompts(limit=200):
        if not row.response:
            continue
        message = json.loads(row.response).get("choices", [{}])[0].get("message", {}) or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            if function.get("name") == "choose":
                try:
                    calls.append(json.loads(function.get("arguments") or "{}"))
                except ValueError:
                    calls.append({})
    return calls


_CHOOSE_TURN = (
    "choose one of cedar, maple, or birch at random for me, and tell me which one you picked."
)


def _tool_picks(db: Database) -> list[str]:
    """Every pick the REAL tool returned this sample, read from its persisted
    result frames in the prompt log."""
    picks: list[str] = []
    for row in db.messages.recent_prompts(limit=200):
        if row.messages:
            picks += _PICK_PATTERN.findall(row.messages)
    return picks


def _score_dispatch(db: Database, before: set[str], reply: str) -> list[Check]:
    calls = _choose_calls(db)
    options_ok = any(
        {option.lower() for option in call.get("options", [])} == set(_OPTIONS) for call in calls
    )
    picks = _tool_picks(db)
    return [
        Check("calls: the choose tool was called", bool(calls), kind="spine"),
        Check("calls: it was given all three options", options_ok, kind="spine"),
        Check(
            # SAID == DID on the pick itself: the reply must report the option
            # the TOOL returned — naming a different one means she free-chose.
            "reply: the reply reports the TOOL'S pick, not her own",
            bool(picks) and picks[-1].lower() in reply.lower(),
            kind="reply",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_choose_fires_on_a_random_pick_ask(chat_eval: ChatEval):
    """'choose one of X, Y, Z at random' → the real choose tool is called with
    the options and the reply reports the pick it returned."""
    await chat_eval(
        case_id="choose-dispatch-fires",
        message=_CHOOSE_TURN,
        score=_score_dispatch,
        min_pass_rate=None,  # report-only: pre-build dispatch validation
    )


_JUDGMENT_TURN = (
    "between cedar, maple, and birch, which do you think makes the best-sounding guitar top?"
)


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        Check(
            # An opinion ask is hers to answer — a coin flip here would be the
            # over-firing failure mode (the no-fire guard of the house pattern).
            "calls: choose was NOT called on a judgment ask",
            not _choose_calls(db),
            kind="spine",
        ),
        Check(
            "reply: she gave an opinion (names at least one wood)",
            any(option in reply.lower() for option in _OPTIONS),
            kind="reply",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_choose_does_not_fire_on_a_judgment_ask(chat_eval: ChatEval):
    """'which do you think is best?' over the same options must NOT flip a coin."""
    await chat_eval(
        case_id="choose-dispatch-no-fire",
        message=_JUDGMENT_TURN,
        score=_score_no_fire,
        min_pass_rate=None,  # report-only: pre-build dispatch validation
    )
