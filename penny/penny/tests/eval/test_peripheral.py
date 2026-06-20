"""Peripheral prompt-type contracts — the single-shot LLM calls that aren't the
chat or collector agent loops but still ship in prod and matter when swapping
models.

  schedule-parse — the /schedule command's structured-output call: natural-
                   language timing → a 5-field cron + a prompt.  Driven through
                   the real command path (push ``/schedule …`` like any message)
                   and scored on the persisted ``Schedule`` row.
  startup-announce— ``get_restart_message``: a git commit → a casual one-line
                   announcement.  Driven via the ``startup_eval`` runner.

Scored on structure/behaviour (valid cron, non-empty prompt, non-fallback,
length, voice), never exact wording.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from penny.database import Database
from penny.database.models import Schedule
from penny.responses import PennyResponse
from penny.tests.eval.conftest import ChatEval, StartupEval

pytestmark = pytest.mark.eval


def _score_schedule(db: Database, before: set[str], reply: str) -> list[str]:
    with Session(db.engine) as session:
        schedules = list(session.exec(select(Schedule)).all())
    if not schedules:
        return ["no schedule created from the /schedule command"]
    schedule = schedules[0]
    fails = []
    if len(schedule.cron_expression.split()) != 5:
        fails.append(f"cron is not a 5-field expression: {schedule.cron_expression!r}")
    if not schedule.prompt_text.strip():
        fails.append("schedule has an empty prompt_text")
    return fails


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


async def test_schedule_parse(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="schedule-parse",
        message="/schedule every weekday at 8am summarize my unread email",
        score=_score_schedule,
    )


async def test_startup_announcement(startup_eval: StartupEval) -> None:
    await startup_eval(
        case_id="startup-announcement",
        commit_message="feat: add /recap command to summarize the week's chats",
        score=_score_startup,
    )
