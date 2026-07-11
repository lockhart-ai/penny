"""Peripheral prompt-type contracts — the single-shot LLM calls that aren't the
chat or collector agent loops but still ship in prod and matter when swapping
models.

  startup-announce— ``get_restart_message``: a git commit → a casual one-line
                   announcement.  Driven via the ``startup_eval`` runner.

Scored on structure/behaviour (non-fallback, length, voice), never exact wording.
"""

from __future__ import annotations

import pytest

from penny.responses import PennyResponse
from penny.tests.eval.conftest import StartupEval

pytestmark = pytest.mark.eval


def _score_startup(announcement: str) -> list[str]:
    fails = []
    if announcement == PennyResponse.RESTART_FALLBACK:
        fails.append("fell back to the canned message instead of generating one")
    if len(announcement) > 150:
        fails.append(f"announcement too long ({len(announcement)} chars)")
    lowered = announcement.lower()
    if "i " not in lowered and "i'" not in lowered:
        fails.append("not first-person (the announcement speaks as Penny)")
    return fails


async def test_startup_announcement(startup_eval: StartupEval) -> None:
    await startup_eval(
        case_id="startup-announcement",
        commit_message="feat: add /recap command to summarize the week's chats",
        score=_score_startup,
    )
