"""ScheduleExecutor — agent that executes user-created scheduled tasks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from croniter import croniter
from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.database.models import Schedule

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ScheduleExecutor(Agent):
    """Agent that executes user-created scheduled tasks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "schedule"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending responses."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Find and execute any due schedules.

        Returns:
            True if any schedules were executed, False otherwise
        """
        if not self._channel:
            logger.error("ScheduleExecutor: no channel set")
            return False

        with Session(self.db.engine) as session:
            schedules = session.exec(select(Schedule)).all()

            now_utc = datetime.now(UTC)
            executed_any = False

            for sched in schedules:
                try:
                    # Convert user timezone to ZoneInfo
                    tz = ZoneInfo(sched.user_timezone)
                    now_in_user_tz = now_utc.astimezone(tz)

                    # Check if a cron occurrence falls within the past 60 seconds.
                    # Strategy: find the *next* occurrence after (now - 60s) and check if
                    # it's <= now. This correctly handles the exact boundary case where
                    # 'now' equals the cron time — get_prev(now) skips the current second
                    # and returns yesterday, but get_next(now - 60s) returns today's time.
                    cron = croniter(sched.cron_expression, now_in_user_tz - timedelta(seconds=60))
                    next_occurrence = cron.get_next(datetime)

                    if next_occurrence <= now_in_user_tz:
                        logger.info(
                            "Executing schedule: user=%s, timing=%s, prompt=%s",
                            sched.user_id,
                            sched.timing_description,
                            sched.prompt_text,
                        )

                        # Execute the scheduled prompt as if the user sent it
                        await self._execute_scheduled_prompt(sched)
                        executed_any = True

                except Exception as e:
                    logger.exception(
                        "Failed to execute schedule id=%s: %s",
                        sched.id,
                        e,
                    )
                    continue

            return executed_any

    async def _execute_scheduled_prompt(self, schedule: Schedule) -> None:
        """Execute a scheduled prompt as if the user sent it.

        Drive it through ``ChatAgent.handle`` — the same entry point a real user
        message takes — so the agent installs its tools (browse, memory) and
        builds the recall-grounded system prompt.  Calling ``run`` directly here
        skipped ``_install_tools``, leaving the scheduled cycle to read whatever
        the *last* ``handle`` call left in the shared tool registry: tools after a
        recent chat, but EMPTY after a restart or a vision message (which installs
        ``[]``).  So a "fetch me the news" schedule worked only by accident — when
        a chat happened to precede it — and otherwise the model's browse call was
        stripped as a tool-less hallucination and Penny apologized she had nothing
        to report.  Going through ``handle`` installs tools deterministically.
        """
        if not self._channel:
            return

        message_agent = self._channel._message_agent
        response = await message_agent.handle(content=schedule.prompt_text, sender=schedule.user_id)

        answer = response.answer.strip() if response.answer else None
        if not answer:
            logger.warning("Schedule produced empty response: id=%s", schedule.id)
            return

        # Send the response to the user
        await self._channel.send_response(
            schedule.user_id,
            answer,
            parent_id=None,
            author=self.name,
            quote_message=None,
        )
