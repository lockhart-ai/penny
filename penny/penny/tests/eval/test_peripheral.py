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
from penny.tests.eval.conftest import Check, StartupEval

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "peripheral"


def _score_startup(announcement: str) -> list[Check]:
    """Graded: each structural expectation of the announcement is its own named check —
    non-fallback, one-line length, first-person voice — so a miss shows exactly which
    property failed instead of collapsing the whole sample to 0."""
    generated = announcement != PennyResponse.RESTART_FALLBACK
    within_length = len(announcement) <= 150
    lowered = announcement.lower()
    first_person = "i " in lowered or "i'" in lowered
    return [
        Check(
            "generated (not the canned fallback)",
            generated,
            rationale=None if generated else "fell back to the canned restart message",
        ),
        Check(
            "within one-line length",
            within_length,
            rationale=None if within_length else f"{len(announcement)} chars > 150",
        ),
        Check(
            "first-person voice (speaks as Penny)",
            first_person,
            rationale=None if first_person else "no first-person 'I' in the announcement",
        ),
    ]


async def test_startup_announcement(startup_eval: StartupEval) -> None:
    await startup_eval(
        case_id="startup-announcement",
        family=_FAMILY,
        commit_message="feat: add /recap command to summarize the week's chats",
        score=_score_startup,
    )
