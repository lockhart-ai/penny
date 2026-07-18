"""ChooseTool — a fair, Python-space random pick from a list.

The model is a **biased chooser**: asked to "pick one at random", gpt-oss:20b
gravitates to the salient option (first-listed, most-recently-mentioned, longest),
so "random" selection is not random in model-space.  Genuine uniform choice must be
a Python coin flip, not a model judgment (*Python-space over model-space*): this tool
runs ``random.choice`` over the validated options and reports the pick back, so the
model uses the result for whatever comes next and never picks for itself.

The NL-dispatch surface (this exact ``name`` + ``description``) was pre-validated
against the live model with a synthetic stub carrying the same name + description: it
fires 1.00 on "choose one of X, Y, Z at random" (options intact, result reported) and
never over-fires on a judgment / opinion ask.  The dispatch contract lives in
``tests/eval/test_choose_dispatch.py`` (against the live model); the deterministic
mechanism (seeded/monkeypatched RNG → exact pick + message) is pinned in
``tests/tools/test_choose.py`` for ``make check``.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import field_validator

from penny.constants import ProgressEmoji
from penny.text_validity import is_blank
from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

# A fair pick needs something to pick between — one option is not a choice.
MIN_OPTIONS = 2

# Actionable refusals (the diagnosis + the next move), raised by the arg validator so
# ``Tool.run`` surfaces them through the standard arg-validation envelope.
TOO_FEW_OPTIONS_MESSAGE = (
    "provide at least 2 options to choose among — a random pick needs two or more to pick between"
)
BLANK_OPTION_MESSAGE = (
    "every option must be real text — one or more options were blank or whitespace only"
)

# The result body (matches the pre-validated stub's shape — do NOT mutate).
CHOSE_MESSAGE = "Chose '{pick}' at random from {n} options."

# First-person result narration (the #1480 twin of ``to_action_str``), derived from
# the call arguments + ``result.success`` like every other override — never by parsing
# the body.  The chosen option itself renders in the body one line below.
NARRATION_SUCCESS = "You flipped a coin among {n} options:"
NARRATION_FAILURE = "You tried to choose but it didn't work:"


class ChooseArgs(ToolArgs):
    """Validated arguments for the choose tool.

    ``options`` must carry at least two non-blank entries: one option is not a
    choice, and a blank option can't be a real outcome, so both are rejected at the
    arg gate with an actionable message (the shared ``is_blank`` predicate — one
    definition of "empty" across the codebase) rather than reaching ``execute``.
    """

    options: list[str]
    reasoning: str | None = None

    @field_validator("options")
    @classmethod
    def _require_choosable_options(cls, options: list[str]) -> list[str]:
        if any(is_blank(option) for option in options):
            raise ValueError(BLANK_OPTION_MESSAGE)
        if len(options) < MIN_OPTIONS:
            raise ValueError(TOO_FEW_OPTIONS_MESSAGE)
        return options


class ChooseTool(Tool):
    """Pick one option at random from a list — the coin flip that belongs in Python.

    The model gravitates to the salient option when asked to pick at random, so a
    genuine uniform choice can't live in model-space.  ``execute`` does
    ``random.choice`` over the validated options and reports the pick; it changes no
    durable state, so the result is ``mutated=False`` (a read-shaped step).
    """

    name = "choose"
    description = (
        "Pick ONE option at random from a list, fairly — a Python coin flip, not a "
        "judgment. Use this whenever the user asks you to choose, pick, select, draw, "
        "roll for, or flip a coin between options — any phrasing like 'choose one of', "
        "'pick one at random', 'randomly select', 'surprise me with one of these', "
        "'you decide', 'dealer's choice', 'roll the dice between', 'flip a coin' — "
        "never pick yourself: you are a biased chooser and this tool is the fair one. "
        "Returns the chosen option; use it for whatever comes next. Not for questions "
        "asking your opinion or judgment about which option is best."
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
    args_model = ChooseArgs

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Pick one option uniformly at random and report it."""
        args = ChooseArgs(**kwargs)
        pick = random.choice(args.options)
        return ToolResult(message=CHOSE_MESSAGE.format(pick=pick, n=len(args.options)))

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Making a random choice"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person narration of the choice, branching on ``result.success``.

        Derived from the call arguments (the option count) + ``result.success``,
        the same way ``browse`` / ``generate_image`` narrate — never by parsing the
        body.  The chosen option renders in ``result.message`` directly below.
        """
        if not result.success:
            return NARRATION_FAILURE
        return NARRATION_SUCCESS.format(n=len(arguments.get("options", [])))

    @classmethod
    def to_progress_emoji(cls, arguments: dict) -> ProgressEmoji:
        return ProgressEmoji.ROLLING
