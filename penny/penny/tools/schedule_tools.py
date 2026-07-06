"""Schedule tools — the chat agent's natural-language surface for recurring tasks.

Replaces the retired ``/schedule`` and ``/unschedule`` commands (epic #1445): the
chat agent dispatches to these from natural language, so "every morning send me a
summary of X" creates a schedule and "you can stop the morning summaries" removes
one — no slash syntax, no positional index.

Three tools, all single-user (the recipient/owner is the primary sender, resolved
from ``db`` at execute time, exactly like ``SendMessageTool``):

- ``schedule_create`` — takes the user's natural-language request (task + timing)
  and reuses the same single-shot NL→cron parse the command used
  (``Prompt.SCHEDULE_PARSE_PROMPT`` grounded in ``current_datetime_line`` and the
  user's profile timezone), then writes a ``Schedule`` row.  The result mirrors the
  parsed cadence back in plain language so the chat agent can confirm it honestly.
- ``schedule_delete`` — removes a schedule **by meaning**, not by index: it embeds
  the caller's description and each existing schedule's timing+prompt, and deletes
  the nearest match.  The result names exactly what was removed.
- ``schedule_list`` — reports the user's active schedules (or that there are none).

Timezone is a hard prerequisite for creating a schedule (a cron with no zone fires
at the wrong wall-clock time), so a missing profile timezone returns an actionable
failure telling the user to set their location — never a silent default.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from similarity.embeddings import cosine_similarity
from sqlmodel import Session, select

from penny.database.models import Schedule
from penny.datetime_utils import current_datetime_line
from penny.llm.models import LlmError
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.models import NonBlankText, ToolArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# A parsed cron must be the standard 5 fields (minute hour day month weekday).
_CRON_FIELDS = 5


class ScheduleParseResult(BaseModel):
    """Structured output of the NL→cron parse."""

    timing_description: str = Field(description="Natural language timing description")
    prompt_text: str = Field(description="Prompt to execute")
    cron_expression: str = Field(description="Cron expression (5 fields)")


class ScheduleCreateArgs(ToolArgs):
    """Validated arguments for schedule_create."""

    request: NonBlankText


class ScheduleDeleteArgs(ToolArgs):
    """Validated arguments for schedule_delete."""

    description: NonBlankText


class ScheduleCreateTool(Tool):
    """Create a recurring scheduled task from a natural-language request."""

    name = "schedule_create"
    description = (
        "Create a recurring scheduled task from the user's own words — pass the whole "
        "request (what to do plus when) as `request` and this parses the timing into a "
        "schedule and saves it.  Use this when the user asks for something to happen on a "
        "cadence, e.g. 'every morning send me a summary of the local news' or 'each Monday "
        "remind me to water the plants'.  The result echoes the parsed cadence so you can "
        "confirm it back to the user in plain language."
    )
    parameters = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "The user's full scheduling request in their own words — the task to run "
                    "AND the timing, e.g. 'every weekday at 8am summarize my unread email'."
                ),
            }
        },
        "required": ["request"],
    }
    args_model = ScheduleCreateArgs

    _NO_RECIPIENT = (
        "Could not create the schedule: no user is registered yet. Ask the user to set up "
        "their profile first, then try again."
    )
    _NEED_TIMEZONE = (
        "Could not create the schedule: I don't know the user's timezone, so a recurring "
        "time would fire at the wrong hour. Ask the user for their location or city, then "
        "retry."
    )
    _PARSE_FAILED = (
        "Could not parse a timing from that request. Ask the user to state the cadence more "
        "plainly (e.g. 'daily at 9am', 'every Monday morning', 'hourly'), then retry."
    )

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to set up a scheduled task but it didn't work:"
        return "You set up a recurring task for the user:"

    def __init__(self, db: Database, model_client: LlmClient) -> None:
        self._db = db
        self._model_client = model_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = ScheduleCreateArgs(**kwargs)
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            return ToolResult(message=self._NO_RECIPIENT, success=False)
        user_info = self._db.users.get_info(recipient)
        if user_info is None or not user_info.timezone:
            return ToolResult(message=self._NEED_TIMEZONE, success=False)
        parsed = await self._parse(args.request, user_info.timezone)
        if parsed is None:
            return ToolResult(message=self._PARSE_FAILED, success=False)
        self._write_schedule(recipient, user_info.timezone, parsed)
        return ToolResult(message=self._confirmation(parsed), mutated=True)

    async def _parse(self, request: str, timezone: str) -> ScheduleParseResult | None:
        """Run the single-shot NL→cron parse, validating the cron field count."""
        prompt = Prompt.SCHEDULE_PARSE_PROMPT.format(
            today=current_datetime_line(self._db),
            timezone=timezone,
            command=request,
        )
        try:
            response = await self._model_client.generate(
                prompt=prompt, format="json", agent_name="chat", prompt_type="schedule-parse"
            )
            parsed = ScheduleParseResult.model_validate_json(response.message.content)
        except (LlmError, ValueError) as error:
            logger.warning("Failed to parse schedule request: %s", error)
            return None
        if len(parsed.cron_expression.split()) != _CRON_FIELDS:
            logger.warning("Invalid cron expression: %s", parsed.cron_expression)
            return None
        return parsed

    def _write_schedule(self, recipient: str, timezone: str, parsed: ScheduleParseResult) -> None:
        """Persist the parsed schedule for the user."""
        with Session(self._db.engine) as session:
            session.add(
                Schedule(
                    user_id=recipient,
                    user_timezone=timezone,
                    cron_expression=parsed.cron_expression,
                    prompt_text=parsed.prompt_text,
                    timing_description=parsed.timing_description,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    @staticmethod
    def _confirmation(parsed: ScheduleParseResult) -> str:
        """Mirror-back so the chat agent can confirm the cadence in plain language."""
        return (
            f"Scheduled '{parsed.prompt_text}' to run {parsed.timing_description} "
            f"(cron: {parsed.cron_expression}). Confirm this cadence back to the user."
        )


class ScheduleDeleteTool(Tool):
    """Remove a scheduled task by meaning — matches against existing schedules."""

    name = "schedule_delete"
    description = (
        "Remove a recurring scheduled task the user no longer wants. Describe the schedule "
        "to remove in `description` (what it does or when it runs, in the user's words, "
        "e.g. 'the morning news summary') — this finds the closest matching schedule by "
        "meaning and deletes it. The result names exactly what was removed so you can "
        "confirm it back."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "What the schedule to remove is about, in natural language — its task or "
                    "timing, e.g. 'the daily news digest' or 'the Monday plant reminder'."
                ),
            }
        },
        "required": ["description"],
    }
    args_model = ScheduleDeleteArgs

    _NO_RECIPIENT = "Could not remove a schedule: no user is registered yet."
    _NONE_SCHEDULED = (
        "There are no scheduled tasks to remove. Tell the user they have nothing scheduled."
    )

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to remove a scheduled task but it didn't work:"
        return "You removed a scheduled task:"

    def __init__(self, db: Database, embedding_client: LlmClient) -> None:
        self._db = db
        self._embedding_client = embedding_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = ScheduleDeleteArgs(**kwargs)
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            return ToolResult(message=self._NO_RECIPIENT, success=False)
        schedules = self._user_schedules(recipient)
        if not schedules:
            return ToolResult(message=self._NONE_SCHEDULED, success=False)
        match = await self._best_match(args.description, schedules)
        self._delete_schedule(match)
        return ToolResult(message=self._confirmation(match), mutated=True)

    def _user_schedules(self, recipient: str) -> list[Schedule]:
        """Every schedule owned by the user, oldest first."""
        with Session(self._db.engine) as session:
            return list(
                session.exec(
                    select(Schedule)
                    .where(Schedule.user_id == recipient)
                    .order_by(Schedule.created_at)  # ty: ignore[invalid-argument-type]
                )
            )

    async def _best_match(self, description: str, schedules: list[Schedule]) -> Schedule:
        """The schedule whose timing+prompt is nearest the description by embedding.

        By meaning, never by index — embeds the caller's description and each
        schedule's ``timing_description`` + ``prompt_text``, and returns the highest
        cosine match.  A single schedule is trivially the match without an embed.
        """
        if len(schedules) == 1:
            return schedules[0]
        texts = [description] + [f"{s.timing_description}: {s.prompt_text}" for s in schedules]
        vectors = await self._embedding_client.embed(texts)
        query_vector = vectors[0]
        scored = zip(
            (cosine_similarity(query_vector, vector) for vector in vectors[1:]),
            schedules,
            strict=True,
        )
        return max(scored, key=lambda pair: pair[0])[1]

    def _delete_schedule(self, schedule: Schedule) -> None:
        """Delete the matched schedule row."""
        with Session(self._db.engine) as session:
            session.delete(session.get(Schedule, schedule.id))
            session.commit()

    @staticmethod
    def _confirmation(schedule: Schedule) -> str:
        """Mirror-back naming exactly what was removed."""
        return (
            f"Removed the schedule '{schedule.prompt_text}' ({schedule.timing_description}). "
            f"Confirm to the user which task you stopped."
        )


class ScheduleListTool(Tool):
    """List the user's active scheduled tasks."""

    name = "schedule_list"
    description = (
        "List the user's active recurring scheduled tasks — use when the user asks what they "
        "have scheduled. Returns each task and its cadence, or that there are none."
    )
    parameters = {"type": "object", "properties": {}}

    _NO_RECIPIENT = "No user is registered yet, so there are no schedules."
    _NONE_SCHEDULED = "The user has no scheduled tasks. Tell them their schedule is empty."

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to check the scheduled tasks but it didn't work:"
        return "You checked the user's scheduled tasks:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            return ToolResult(message=self._NO_RECIPIENT)
        with Session(self._db.engine) as session:
            schedules = list(
                session.exec(
                    select(Schedule)
                    .where(Schedule.user_id == recipient)
                    .order_by(Schedule.created_at)  # ty: ignore[invalid-argument-type]
                )
            )
        if not schedules:
            return ToolResult(message=self._NONE_SCHEDULED)
        lines = [
            f"- {schedule.timing_description}: {schedule.prompt_text}" for schedule in schedules
        ]
        return ToolResult(message="\n".join(["The user's scheduled tasks:", *lines]))
