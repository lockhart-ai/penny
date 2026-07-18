"""Tests for the choose tool — fair random selection in Python.

The model is a biased chooser, so "pick one at random" belongs in Python: ``execute``
runs ``random.choice`` over the validated options and reports the pick.  These pin the
deterministic mechanism for ``make check`` — the exact result body, the two arg-gate
refusals (fewer than two options; a blank option), and the result narration.  The
live-model NL-dispatch contract (a "choose one of X, Y, Z" phrasing reaches the tool
with options intact; a judgment ask does NOT) lives in ``tests/eval/test_choose_dispatch.py``.

RNG is never left live — the exact-pick case monkeypatches ``random.choice`` and the
fairness case seeds the module RNG, so no assertion depends on chance.
"""

from __future__ import annotations

import random

import pytest

from penny.constants import ProgressEmoji
from penny.tools import choose as choose_module
from penny.tools.choose import CHOSE_MESSAGE, ChooseTool
from penny.tools.models import ToolResult

_OPTIONS = ["red", "green", "blue"]

# The full, byte-exact description — the pre-validated NL-dispatch surface the eval
# stub carries.  Asserted verbatim so any drift (a stray space, a reworded alias)
# fails here rather than silently diverging from the contract the supervisor re-runs.
_EXPECTED_DESCRIPTION = (
    "Pick ONE option at random from a list, fairly — a Python coin flip, not a "
    "judgment. Use this whenever the user asks you to choose, pick, select, draw, "
    "roll for, or flip a coin between options — any phrasing like 'choose one of', "
    "'pick one at random', 'randomly select', 'surprise me with one of these', "
    "'you decide', 'dealer's choice', 'roll the dice between', 'flip a coin' — "
    "never pick yourself: you are a biased chooser and this tool is the fair one. "
    "Returns the chosen option; use it for whatever comes next. Not for questions "
    "asking your opinion or judgment about which option is best."
)

# The whole arg-gate refusal bodies, rendered by ``Tool._validation_error_message``
# (field descriptor + the validator reason + the retry instruction) — asserted whole.
_TOO_FEW_REFUSAL = (
    "options (array: The options to choose among (2 or more).): provide at least 2 "
    "options to choose among — a random pick needs two or more to pick between. "
    "Call choose(<valid arguments>) again."
)
_BLANK_REFUSAL = (
    "options (array: The options to choose among (2 or more).): every option must be "
    "real text — one or more options were blank or whitespace only. "
    "Call choose(<valid arguments>) again."
)


def test_description_matches_prevalidated_surface():
    """The dispatch surface the eval swaps in is byte-for-byte what ships."""
    assert ChooseTool.description == _EXPECTED_DESCRIPTION


@pytest.mark.asyncio
async def test_picks_an_option_and_reports_it(monkeypatch):
    """A valid call picks one of the options and reports it in the exact body.

    ``random.choice`` is monkeypatched to a fixed element so the pick — and thus the
    whole message literal — is deterministic; the tool changes no state (mutated=False).
    """
    monkeypatch.setattr(choose_module.random, "choice", lambda seq: seq[1])

    result = await ChooseTool().run(options=_OPTIONS)

    assert result.success is True
    assert result.mutated is False
    assert result.message == "Chose 'green' at random from 3 options."
    # Narration is the first-person twin of the body — the option count, no pick parse.
    assert ChooseTool.to_result_narration({"options": _OPTIONS}, result) == (
        "You flipped a coin among 3 options:"
    )


@pytest.mark.asyncio
async def test_pick_is_always_among_the_options(monkeypatch):
    """Over many seeded draws the pick is always one of the options — the body is
    exactly one of the well-formed ``Chose '<option>' …`` messages, never anything
    outside the list.  Seeded (not live) so the assertion is deterministic."""
    monkeypatch.setattr(choose_module, "random", random.Random(20260717))
    valid_messages = {CHOSE_MESSAGE.format(pick=option, n=len(_OPTIONS)) for option in _OPTIONS}

    tool = ChooseTool()
    for _ in range(30):
        result = await tool.run(options=_OPTIONS)
        assert result.message in valid_messages


@pytest.mark.asyncio
async def test_fewer_than_two_options_is_refused():
    """A single option is not a choice — refused at the arg gate, actionably."""
    result = await ChooseTool().run(options=["only one"])

    assert result.success is False
    assert result.mutated is False
    assert result.message == _TOO_FEW_REFUSAL
    # Empty list hits the same requirement (fewer than two).
    empty = await ChooseTool().run(options=[])
    assert empty.message == _TOO_FEW_REFUSAL


@pytest.mark.asyncio
async def test_blank_option_is_refused():
    """A blank / whitespace-only option can't be a real outcome — refused, actionably."""
    result = await ChooseTool().run(options=["red", "   "])

    assert result.success is False
    assert result.mutated is False
    assert result.message == _BLANK_REFUSAL
    # An empty-string option is the same defect.
    empty_string = await ChooseTool().run(options=["red", "", "blue"])
    assert empty_string.message == _BLANK_REFUSAL


def test_failure_narration_is_honest():
    """A failed result narrates the honest 'didn't work', never a false success."""
    failed = ToolResult(message="x", success=False)
    assert ChooseTool.to_result_narration({"options": _OPTIONS}, failed) == (
        "You tried to choose but it didn't work:"
    )


def test_progress_emoji_is_the_dice():
    """Choosing surfaces as the 🎲 progress reaction while it runs."""
    assert ChooseTool.to_progress_emoji({}) is ProgressEmoji.ROLLING
