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

**Dispatch stands on the tool description alone.**  ``ChooseTool`` is registered on
every agent surface (``Agent.get_tools``) and no skill teaches this routing —
nothing is pre-seeded since migration 0108 — so these cases seed none: the world
they measure is a fresh deployment's.

**The conversation state machine fronts every driven turn** (#1706) — it classifies
before the chat agent runs, and a pick-one ask lands in whatever state it lands in.
What is scored here is the chat turn's dispatch, not the state, which is why these
cases are REPORT-ONLY (``min_pass_rate=None``), the same posture the sibling
dispatch modules carry.
"""

from __future__ import annotations

import json
import re

import pytest

from penny.database import Database
from penny.database.models import PromptLog
from penny.tests.eval.conftest import ChatEval, Check, live_prompts, routing_clean

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module — the same
# tag its sibling NL-dispatch modules (email, generate_image) carry, so the report's
# families rollup reads the dispatch keep-set as one group.
_FAMILY = "nl-dispatch"

_CHOOSE_TOOL = "choose"

_OPTIONS = ["cedar", "maple", "birch"]

# The real tool's result body (whole-render pinned in its unit tests).
_PICK_PATTERN = re.compile(r"Chose '([^']+)' at random")


def _row_tool_calls(row: PromptLog) -> list[dict]:
    """One promptlog row's emitted tool calls (empty when the draw carried none)."""
    response = json.loads(row.response) if row.response else {}
    choices = response.get("choices") or []
    if not choices:
        return []
    return choices[0].get("message", {}).get("tool_calls") or []


def _choose_calls(db: Database) -> list[dict]:
    """Every `choose` call's parsed arguments, from the persisted promptlog.

    Sourced through ``live_prompts`` — the harness's fetch chokepoint, which drops any
    seeded prior turn — so this reads what THIS sample's model did and can never count
    a seeded round's call as the live turn's."""
    calls: list[dict] = []
    for row in live_prompts(db):
        for call in _row_tool_calls(row):
            function = call.get("function", {})
            if function.get("name") == _CHOOSE_TOOL:
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
    result frames in the prompt log — through ``live_prompts``, so a seeded prior
    turn's pick can never stand in for this turn's.

    A tool result is durable as a side effect of being fed back, so the pick rides
    into the NEXT call's ``messages``: a run that ended AT the choose call carries
    no pick here and no reply reporting one either, which is the same verdict."""
    picks: list[str] = []
    for row in live_prompts(db):
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
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def test_choose_fires_on_a_random_pick_ask(chat_eval: ChatEval) -> None:
    """'choose one of X, Y, Z at random' → the real choose tool is called with
    the options and the reply reports the pick it returned."""
    await chat_eval(
        case_id="choose-dispatch-fires",
        family=_FAMILY,
        message=_CHOOSE_TURN,
        score=_score_dispatch,
        min_pass_rate=None,  # report-only: a live-model dispatch rate
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
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def test_choose_does_not_fire_on_a_judgment_ask(chat_eval: ChatEval) -> None:
    """'which do you think is best?' over the same options must NOT flip a coin."""
    await chat_eval(
        case_id="choose-dispatch-no-fire",
        family=_FAMILY,
        message=_JUDGMENT_TURN,
        score=_score_no_fire,
        min_pass_rate=None,  # report-only: a live-model dispatch rate
    )
