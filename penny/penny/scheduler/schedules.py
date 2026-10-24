"""Concrete schedule implementations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from penny.scheduler.base import Schedule, ScheduledTask

logger = logging.getLogger(__name__)


class PeriodicSchedule(Schedule):
    """Runs periodically once its OWN idle threshold has cleared.

    There is no single global idle gate: each schedule is handed its own
    ``idle_threshold`` and reads the scheduler's raw elapsed idle_seconds against
    it (#2779) — a schedule with no idle requirement of its own passes
    ``lambda: 0.0``, which every non-negative ``idle_seconds`` clears immediately.
    """

    def __init__(
        self,
        agent: ScheduledTask,
        interval: Callable[[], float],
        idle_threshold: Callable[[], float],
    ):
        """
        Initialize periodic schedule.

        Args:
            agent: The agent to execute on each interval
            interval: Callable returning current interval in seconds (read each tick)
            idle_threshold: Callable returning THIS schedule's own idle requirement in
                seconds (read each tick). Required — a schedule that runs regardless of
                foreground activity passes ``lambda: 0.0``, never ``None``, so every
                call site states its idle requirement explicitly.
        """
        self.agent = agent
        self._interval = interval
        self._idle_threshold = idle_threshold
        # Fires on its first eligible check (None means "not yet run").
        self._last_run: float | None = None
        logger.info(
            "PeriodicSchedule created for %s with interval=%.0fs idle_threshold=%.0fs",
            agent.name,
            interval(),
            idle_threshold(),
        )

    def should_run(self, idle_seconds: float) -> bool:
        """Check this schedule's own idle threshold has cleared and its interval elapsed."""
        if idle_seconds < self._idle_threshold():
            return False

        now = time.monotonic()
        if self._last_run is None:
            return True

        elapsed = now - self._last_run
        return elapsed >= self._interval()

    def reset(self) -> None:
        """Reset last run time on message arrival, so it fires again once idle."""
        self._last_run = None

    def mark_complete(self) -> None:
        """Record completion time for next interval calculation."""
        self._last_run = time.monotonic()
