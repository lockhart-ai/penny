"""NL-dispatch contracts for command-retirement tools (epic #1445).

These cases prove that a natural-language request dispatches to the tool that
replaced a slash command — the model calls the right tool with faithful args —
plus a no-fire guard that a casual mention of the same topic must NOT trigger it.
Scoring is STRUCTURAL (the persisted tool call + its arguments), never wording.

`generate_image` (retired `/draw`): the image client is mocked at the system
boundary via the ``prepare`` hook, so no real image model is needed — the
contract is purely "did the utterance dispatch to generate_image with a faithful
description, and does an unrelated mention stay quiet?".
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from penny.penny import Penny
from penny.tests.conftest import ONE_PX_PNG_B64
from penny.tests.eval.conftest import ChatEval, Check, last_tool_args, tool_not_called

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "nl-dispatch"

_GENERATE_IMAGE = "generate_image"


def _mock_image_client(penny: Penny) -> None:
    """Wire a mocked image client so generate_image is registered and its
    boundary call is a no-op returning a canned PNG (no real image model)."""
    client = AsyncMock()
    client.generate_image.return_value = ONE_PX_PNG_B64
    penny.chat_agent._image_client = client


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_drew(subject_token: str):
    """The utterance must dispatch to generate_image with a description that
    faithfully carries the requested subject (a salient token of it)."""

    def score(db, before, reply) -> list[Check]:
        anchor = f"{_GENERATE_IMAGE}("
        args = last_tool_args(db, _GENERATE_IMAGE)
        if args is None:
            # No dispatch — the description checks have nothing to inspect (not-applicable).
            return [
                Check("generate_image called", False, anchor=anchor),
                Check.na("description is non-empty", anchor=anchor),
                Check.na(f"description carries the subject '{subject_token}'", anchor=anchor),
            ]
        description = str(args.get("description") or "")
        has_subject = subject_token in description.lower()
        return [
            Check("generate_image called", True, anchor=anchor),
            Check("description is non-empty", bool(description.strip()), anchor=anchor),
            Check(
                f"description carries the subject '{subject_token}'",
                has_subject,
                anchor=anchor,
                rationale=None
                if has_subject
                else f"description {description!r} dropped '{subject_token}'",
            ),
        ]

    return score


def _score_no_draw(db, before, reply) -> list[Check]:
    """A casual mention of art/drawing must NOT trigger image generation."""
    return [
        Check(
            "generate_image not fired on a casual mention",
            tool_not_called(db, _GENERATE_IMAGE),
            anchor=f"{_GENERATE_IMAGE}(",
        ),
    ]


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_draw_request_dispatches(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-generate-image-draw",
        family=_FAMILY,
        message="can you draw me a teal origami dragon perched on a coffee mug?",
        prepare=_mock_image_client,
        score=_score_drew("dragon"),
    )


async def test_make_a_picture_dispatches(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-generate-image-picture",
        family=_FAMILY,
        message="make a picture of a neon cactus wearing tiny sunglasses",
        prepare=_mock_image_client,
        score=_score_drew("cactus"),
    )


async def test_casual_art_mention_does_not_dispatch(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-generate-image-nofire",
        family=_FAMILY,
        message="i saw a really nice watercolor painting at the gallery today, it was lovely",
        prepare=_mock_image_client,
        score=_score_no_draw,
    )
