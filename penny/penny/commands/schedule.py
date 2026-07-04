"""The /schedule command — create and list recurring background tasks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Schedule, UserInfo
from penny.datetime_utils import current_datetime_line
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ScheduleParseResult(BaseModel):
    """Parsed schedule command."""

    timing_description: str = Field(description="Natural language timing description")
    prompt_text: str = Field(description="Prompt to execute")
    cron_expression: str = Field(description="Cron expression (5 fields)")


class ScheduleCommand(Command):
    """Create and list recurring background tasks."""

    name = "schedule"
    description = "Create and list recurring background tasks"
    help_text = (
        "Create recurring background tasks that run prompts automatically.\n\n"
        "**Usage**:\n"
        "• `/schedule` — List all your active schedules\n"
        "• `/schedule <timing> <prompt>` — Create a new schedule\n"
        "  (e.g., `/schedule daily 9am what's the news?`)\n\n"
        "Use `/unschedule` to delete a schedule."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute schedule command."""
        args = args.strip()

        if not args:
            return await self._list_schedules(context)

        return await self._create_schedule(args, context)

    async def _list_schedules(self, context: CommandContext) -> CommandResult:
        """List all schedules for the user."""
        with Session(context.db.engine) as session:
            schedules = list(
                session.exec(
                    select(Schedule).where(Schedule.user_id == context.user).order_by(Schedule.id)  # ty: ignore[invalid-argument-type]
                )
            )

            if not schedules:
                return CommandResult(text=PennyResponse.SCHEDULE_NO_TASKS)

            lines = ["**Your Schedules**", ""]
            for idx, sched in enumerate(schedules, start=1):
                lines.append(f"{idx}. **{sched.timing_description}**: {sched.prompt_text}")

            return CommandResult(text="\n".join(lines))

    async def _create_schedule(self, command: str, context: CommandContext) -> CommandResult:
        """Create a new schedule."""
        # Get user timezone
        with Session(context.db.engine) as session:
            user_info = session.exec(
                select(UserInfo).where(UserInfo.sender == context.user)
            ).first()

            if not user_info or not user_info.timezone:
                return CommandResult(text=PennyResponse.SCHEDULE_NEED_TIMEZONE)

            user_timezone = user_info.timezone

        # Parse command using LLM — ground it in today's date so relative
        # cadences ("every other friday") resolve against the right calendar day.
        prompt = Prompt.SCHEDULE_PARSE_PROMPT.format(
            today=current_datetime_line(context.db),
            timezone=user_timezone,
            command=command,
        )

        try:
            response = await context.model_client.generate(
                prompt=prompt,
                format="json",
            )

            # Parse JSON from response
            result = ScheduleParseResult.model_validate_json(response.message.content)

        except Exception as e:
            logger.warning("Failed to parse schedule command: %s", e)
            return CommandResult(text=PennyResponse.SCHEDULE_PARSE_ERROR)

        # Validate cron expression format (5 fields)
        cron_parts = result.cron_expression.split()
        if len(cron_parts) != 5:
            logger.warning("Invalid cron expression: %s", result.cron_expression)
            return CommandResult(text=PennyResponse.SCHEDULE_INVALID_CRON)

        # Create schedule in database
        with Session(context.db.engine) as session:
            new_schedule = Schedule(
                user_id=context.user,
                user_timezone=user_timezone,
                cron_expression=result.cron_expression,
                prompt_text=result.prompt_text,
                timing_description=result.timing_description,
                created_at=datetime.now(UTC),
            )
            session.add(new_schedule)
            session.commit()

        return CommandResult(
            text=PennyResponse.SCHEDULE_ADDED.format(
                timing=result.timing_description, prompt=result.prompt_text
            )
        )
