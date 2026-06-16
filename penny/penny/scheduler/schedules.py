"""Concrete schedule implementations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from penny.scheduler.base import Schedule, ScheduledTask

logger = logging.getLogger(__name__)


class PeriodicSchedule(Schedule):
    """Runs periodically, optionally gated on system idle state."""

    def __init__(
        self,
        agent: ScheduledTask,
        interval: Callable[[], float],
        requires_idle: bool = True,
    ):
        """
        Initialize periodic schedule.

        Args:
            agent: The agent to execute on each interval
            interval: Callable returning current interval in seconds (read each tick)
            requires_idle: If True, only runs when system is past the idle threshold.
                           If False, runs on its own timer regardless of user activity.
        """
        self.agent = agent
        self._interval = interval
        self._requires_idle = requires_idle
        # Independent schedules start their clock at boot so the full interval
        # elapses before the first run.  Idle-gated schedules fire on first idle
        # (None means "not yet run").
        self._last_run: float | None = None if requires_idle else time.monotonic()
        logger.info(
            "PeriodicSchedule created for %s with interval=%.0fs requires_idle=%s",
            agent.name,
            interval(),
            requires_idle,
        )

    def should_run(self, is_idle: bool) -> bool:
        """Check if interval has elapsed; also checks idle state when requires_idle=True."""
        if self._requires_idle and not is_idle:
            return False

        now = time.monotonic()
        if self._last_run is None:
            return True

        elapsed = now - self._last_run
        return elapsed >= self._interval()

    def reset(self) -> None:
        """Reset last run time on message arrival — no-op for idle-independent schedules."""
        if self._requires_idle:
            self._last_run = None

    def mark_complete(self) -> None:
        """Record completion time for next interval calculation."""
        self._last_run = time.monotonic()


class AlwaysRunSchedule(Schedule):
    """Runs periodically regardless of idle state."""

    def __init__(
        self,
        agent: ScheduledTask,
        interval: float,
    ):
        """
        Initialize always-run schedule.

        Args:
            agent: The agent to execute on each interval
            interval: Time in seconds between executions
        """
        self.agent = agent
        self._interval = interval
        self._last_run: float | None = None
        logger.info(
            "AlwaysRunSchedule created for %s with interval=%.0fs",
            agent.name,
            interval,
        )

    def should_run(self, is_idle: bool) -> bool:
        """Check if interval has elapsed since last run, regardless of idle state."""
        now = time.monotonic()
        if self._last_run is None:
            # First run immediately on startup
            return True

        elapsed = now - self._last_run
        return elapsed >= self._interval

    def reset(self) -> None:
        """No-op — this schedule ignores message arrivals."""
        pass

    def mark_complete(self) -> None:
        """Record completion time for next interval calculation."""
        self._last_run = time.monotonic()
