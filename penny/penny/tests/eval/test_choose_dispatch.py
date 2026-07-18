"""NL-dispatch pre-validation for the `choose` tool (built AFTER this passes).

The model is a biased chooser — asked to "pick one at random" it gravitates to
the first/most-salient option — so random choice belongs in Python (the v3
`pick()` design).  Before building the real tool, this case grafts a SYNTHETIC
stub carrying the REAL intended name + description onto the chat agent and
validates the dispatch contract both ways (the house NL-dispatch pattern):

  * "choose one of X, Y, Z" → the tool is CALLED with the options, and the
    reply reports the TOOL'S result — the stub returns a FIXED option that a
    free-choosing model would be unlikely to echo by luck, so a reply naming a
    different option means she picked herself instead of using the tool.
  * a JUDGMENT ask over the same options ("which do you think is best?") must
    NOT fire it — an opinion is hers to give, not a coin flip.

When the real tool lands, the stub registration is replaced by the real
surface and these cases become its standing contract.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from penny.database import Database
from penny.penny import Penny
from penny.tests.eval.conftest import ChatEval, Check
from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

pytestmark = pytest.mark.eval

# The stub always "randomly" lands on the SECOND option — a fixed sentinel so
# the scorer can tell tool-reported picks from the model free-choosing (a real
# random.choice is the built tool's concern; dispatch is this case's).
_OPTIONS = ["cedar", "maple", "birch"]
_STUB_PICK = "maple"


class _ChooseArgs(ToolArgs):
    options: list[str]
    reasoning: str | None = None


class _ChooseStub(Tool):
    """The synthetic `choose` — the NAME + DESCRIPTION are the real design under
    test (the dispatch lever); execute is a fixed-sentinel stand-in."""

    name = "choose"
    description = (
        "Pick ONE option at random from a list, fairly — a Python coin flip, not "
        "a judgment. Use this whenever the user asks you to choose, pick, or "
        "select randomly among options ('choose one of', 'pick one at random', "
        "'surprise me with one of these') — never pick yourself: you are a "
        "biased chooser and this tool is the fair one. Returns the chosen option; "
        "use it for whatever comes next. Not for questions asking your opinion "
        "or judgment about which option is best."
    )
    parameters = {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The options to choose among (2 or more).",
            },
        },
        "required": ["options"],
    }
    args_model = _ChooseArgs

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = _ChooseArgs(**kwargs)
        pick = _STUB_PICK if _STUB_PICK in args.options else args.options[0]
        return ToolResult(message=f"Chose '{pick}' at random from {len(args.options)} options.")


def _graft_choose(penny: Penny) -> None:
    """Register the stub on the chat agent for this sample (the per-turn tool
    build composes get_tools fresh, so wrap it)."""
    original = penny.chat_agent.get_tools

    def with_choose(run_id: str | None = None) -> list[Tool]:
        return [*original(run_id), _ChooseStub()]

    penny.chat_agent.get_tools = with_choose  # ty: ignore[invalid-assignment]


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


def _score_dispatch(db: Database, before: set[str], reply: str) -> list[Check]:
    calls = _choose_calls(db)
    options_ok = any(
        {option.lower() for option in call.get("options", [])} == set(_OPTIONS) for call in calls
    )
    return [
        Check("the choose tool was called", bool(calls)),
        Check("it was given all three options", options_ok),
        Check(
            # The stub's fixed sentinel: a reply naming a different option means
            # the model free-chose instead of using the tool's result.
            "the reply reports the TOOL'S pick (maple), not her own",
            _STUB_PICK in reply.lower(),
        ),
    ]


@pytest.mark.asyncio
async def test_choose_fires_on_a_random_pick_ask(chat_eval: ChatEval):
    """'choose one of X, Y, Z at random' → the choose tool is called with the
    options and the reply reports its result."""
    await chat_eval(
        case_id="choose-dispatch-fires",
        message=_CHOOSE_TURN,
        prepare=_graft_choose,
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
            "choose was NOT called on a judgment ask",
            not _choose_calls(db),
        ),
        Check(
            "she gave an opinion (names at least one wood)",
            any(option in reply.lower() for option in _OPTIONS),
        ),
    ]


@pytest.mark.asyncio
async def test_choose_does_not_fire_on_a_judgment_ask(chat_eval: ChatEval):
    """'which do you think is best?' over the same options must NOT flip a coin."""
    await chat_eval(
        case_id="choose-dispatch-no-fire",
        message=_JUDGMENT_TURN,
        prepare=_graft_choose,
        score=_score_no_fire,
        min_pass_rate=None,  # report-only: pre-build dispatch validation
    )
