"""Survival contract for the image-tool narration (part of epic #1478 / #1481).

The narration seam (#1479) makes each tool result lead with a first-person line
(``You drew "a red fox":``), and the recap instruction (#1483) tells Penny to open
her reply with what she did.  A unit test can prove the narration STRING exists in
the tool result (see ``test_tool_reasoning.py``) — but not the thing that actually
matters: that the summary **survives into Penny's reply to the user**.  The model
can read ``You drew "a red fox":`` and still answer without recapping the drawing.
This case drives the real chat loop and scores the REPLY, so a pass means the
drawing recap genuinely reached the user.

Scored STRUCTURALLY (``generate_image`` was called AND the reply reflects the
drawing action), never on exact wording, since the recap is composed fresh each
turn.  The scorer prints a sample reply so the PR can report
``case | sample text | N score``.

The image tool is only registered when an image client is present, which the eval
config does NOT configure.  Rather than skip the case (a skipped survival eval
gives zero signal), this mocks the image client at the system boundary via the
``prepare`` hook — exactly as the ``generate_image`` NL-dispatch case in
``test_command_tools.py`` does — so ``generate_image`` IS registered and the live
model can call it with no real image model.  Only the tool's OWN failure branch
(``result.success is False``) can't be exercised live: ``GenerateImageTool.execute``
never returns ``success=False`` (a backend error surfaces as a framework-synthesised
exception narration, not the tool's own failure line), so that branch is covered by
the unit test in ``test_tool_reasoning.py`` instead.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import ONE_PX_PNG_B64
from penny.tests.eval.conftest import ChatEval, tool_was_called

pytestmark = pytest.mark.eval

_GENERATE_IMAGE = "generate_image"

# Action-reflection that proves the image summary survived into the reply — the
# model recaps as "here's the illustration I drew" / "made you a red fox" /
# "your fox is ready", so match the DRAWING ACTION (draw/sketch/paint/illustrate/
# make/render...) or a delivery phrasing (no leading "I" — the model often drops
# it; the apostrophe in "here's" may be straight or curly), never exact wording.
_DREW = re.compile(
    r"\b(drew|draw(ing|n)?|sketch(ed|ing)?|paint(ed|ing)?|illustrat(ed|ion|ing|e)|"
    r"made|make|making|whipped up|cooked up|created?|creating|generated?|generating|"
    r"rendered|rendering|conjured|put together|here\s?[’']?s|here it is|here you go|"
    r"(image|picture|drawing|illustration|fox) (is )?(ready|done|coming|for you|attached))\b",
    re.I,
)


def _mock_image_client(penny: Penny) -> None:
    """Wire a mocked image client so generate_image is registered and its boundary
    call is a no-op returning a canned PNG (no real image model)."""
    client = AsyncMock()
    client.generate_image.return_value = ONE_PX_PNG_B64
    penny.chat_agent._image_client = client


def _score_drew(subject_token: str):
    """The drawing must dispatch to generate_image AND the reply must recap it — so
    a pass proves the drawing summary survived into Penny's reply, not just that the
    tool fired."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not tool_was_called(db, _GENERATE_IMAGE):
            fails.append("generate_image was not called — no drawing action to recap")
        if not _DREW.search(reply):
            fails.append(
                "reply did not recap the drawing — the summary did not survive into the response"
            )
        elif subject_token not in reply.lower():
            fails.append(
                f"reply {reply!r} recapped a drawing but dropped the subject '{subject_token}'"
            )
        print(
            f"[SURVIVAL draw] tool={int(tool_was_called(db, _GENERATE_IMAGE))} :: {reply[:200]!r}"
        )
        return fails

    return score


async def test_draw_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="image-recap-draw",
        message="draw me a red fox",
        prepare=_mock_image_client,
        score=_score_drew("fox"),
        min_pass_rate=0.75,
    )
