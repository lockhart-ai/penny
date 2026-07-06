"""NL-dispatch contracts for the schedule tools (epic #1445, issue #1448).

Naive-register phrasings must dispatch to the right tool and persist the right
effect — asserted structurally (tool calls + ``Schedule`` rows), never on wording:

  create — "every morning send me a summary of <topic>" → ``schedule_create`` runs
           and a ``Schedule`` row exists with a 5-field cron in the user's timezone.
  delete — "you can stop the morning summaries" → ``schedule_delete`` runs and the
           MATCHING schedule (by meaning, not index) is the one removed.
  list   — "what do you have scheduled?" → ``schedule_list`` runs.
  no-fire — "my schedule is packed this week" must NOT create a schedule (a casual
           mention of one's calendar is not a scheduling request).

Synthetic topics only (the repo is public).  Extends the retired command-path
cron-parse peripheral case rather than duplicating it — the sane-cron assertion
lives here now, on the NL→tool path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlmodel import Session, select

from penny.database import Database
from penny.database.models import Schedule
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import ChatEval, tool_was_called

pytestmark = pytest.mark.eval

# The user's profile timezone (set by ``seed_user``) — a created schedule must
# carry it so the cron fires at the right wall-clock time.
_USER_TIMEZONE = "America/Los_Angeles"


def _schedules(db: Database) -> list[Schedule]:
    with Session(db.engine) as session:
        return list(session.exec(select(Schedule)))


def _seed_schedule(db: Database, *, prompt_text: str, timing_description: str, cron: str) -> None:
    """Insert one existing schedule for the delete/list cases."""
    with Session(db.engine) as session:
        session.add(
            Schedule(
                user_id=TEST_SENDER,
                user_timezone=_USER_TIMEZONE,
                cron_expression=cron,
                prompt_text=prompt_text,
                timing_description=timing_description,
                created_at=datetime.now(UTC),
            )
        )
        session.commit()


def _score_create(db: Database, before: set[str], reply: str) -> list[str]:
    fails: list[str] = []
    if not tool_was_called(db, "schedule_create"):
        fails.append("schedule_create was not called")
    schedules = _schedules(db)
    if not schedules:
        fails.append("no Schedule row was created")
        return fails
    schedule = schedules[0]
    if len(schedule.cron_expression.split()) != 5:
        fails.append(f"cron is not a 5-field expression: {schedule.cron_expression!r}")
    if not schedule.prompt_text.strip():
        fails.append("schedule has an empty prompt_text")
    if schedule.user_timezone != _USER_TIMEZONE:
        fails.append(f"schedule not in the user's timezone: {schedule.user_timezone!r}")
    return fails


def _score_delete(db: Database, before: set[str], reply: str) -> list[str]:
    fails: list[str] = []
    if not tool_was_called(db, "schedule_delete"):
        fails.append("schedule_delete was not called")
    remaining = {s.prompt_text for s in _schedules(db)}
    if "send me a summary of the morning news" in remaining:
        fails.append("the morning-news schedule was not removed")
    if "water the front garden every week" not in remaining:
        fails.append("the unrelated weekly schedule was wrongly removed")
    return fails


def _score_list(db: Database, before: set[str], reply: str) -> list[str]:
    if not tool_was_called(db, "schedule_list"):
        return ["schedule_list was not called"]
    return []


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[str]:
    fails: list[str] = []
    if tool_was_called(db, "schedule_create"):
        fails.append("schedule_create fired on a casual mention of a busy week")
    if _schedules(db):
        fails.append("a Schedule row was created from a non-scheduling message")
    return fails


async def test_schedule_create_dispatch(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="schedule-create-dispatch",
        message="every morning can you send me a summary of the neighborhood gardening club?",
        score=_score_create,
        min_pass_rate=0.6,
    )


async def test_schedule_delete_dispatch(chat_eval: ChatEval) -> None:
    def seed(db: Database) -> None:
        _seed_schedule(
            db,
            prompt_text="send me a summary of the morning news",
            timing_description="every morning at 8am",
            cron="0 8 * * *",
        )
        _seed_schedule(
            db,
            prompt_text="water the front garden every week",
            timing_description="every Monday at 7am",
            cron="0 7 * * 1",
        )

    await chat_eval(
        case_id="schedule-delete-dispatch",
        message="you can stop the morning summaries",
        score=_score_delete,
        seed=seed,
        min_pass_rate=0.6,
    )


async def test_schedule_list_dispatch(chat_eval: ChatEval) -> None:
    def seed(db: Database) -> None:
        _seed_schedule(
            db,
            prompt_text="send me a summary of the morning news",
            timing_description="every morning at 8am",
            cron="0 8 * * *",
        )

    await chat_eval(
        case_id="schedule-list-dispatch",
        message="what do you have scheduled?",
        score=_score_list,
        seed=seed,
        min_pass_rate=0.6,
    )


async def test_schedule_no_fire_guard(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="schedule-no-fire",
        message="my schedule is packed this week, so many meetings",
        score=_score_no_fire,
        min_pass_rate=0.75,
    )
