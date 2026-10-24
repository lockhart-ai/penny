"""Tests for BackgroundScheduler."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from penny.scheduler.base import BackgroundScheduler, Schedule
from penny.scheduler.schedules import PeriodicSchedule
from penny.tests.conftest import TEST_SENDER, wait_until


class _SlowAgent:
    """Agent that blocks until an event is set or cancelled."""

    name = "slow_agent"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = False
        self.cancelled = False

    async def execute(self) -> bool:
        self.started.set()
        try:
            await self.release.wait()
            self.completed = True
            return True
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _SimpleAgent:
    """Agent that returns a fixed value from execute()."""

    def __init__(self, name: str, return_value: bool) -> None:
        self.name = name
        self.return_value = return_value
        self.execute_count = 0

    async def execute(self) -> bool:
        self.execute_count += 1
        return self.return_value


class _AlwaysRunSchedule(Schedule):
    """Schedule that always fires (idle-independent)."""

    def __init__(self, agent: _SlowAgent | _SimpleAgent) -> None:
        self.agent: Any = agent
        self._completed = False

    def should_run(self, idle_seconds: float) -> bool:
        return not self._completed

    def mark_complete(self) -> None:
        self._completed = True


class _AlwaysEligibleSchedule(Schedule):
    """Schedule that is always eligible (never marks complete internally)."""

    def __init__(self, agent: _SimpleAgent) -> None:
        self.agent: Any = agent
        self.mark_complete_count = 0

    def should_run(self, idle_seconds: float) -> bool:
        return True

    def mark_complete(self) -> None:
        self.mark_complete_count += 1


@pytest.mark.asyncio
async def test_foreground_cancels_active_background_task():
    """When foreground work starts, the active background task is cancelled."""
    agent = _SlowAgent()
    schedule = _AlwaysRunSchedule(agent)
    scheduler = BackgroundScheduler(
        schedules=[schedule],
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Wait for the agent to start executing
        await asyncio.wait_for(agent.started.wait(), timeout=2.0)

        # Simulate a message arriving — should cancel the background task
        scheduler.notify_foreground_start()

        await wait_until(lambda: agent.cancelled, timeout=2.0)

        assert not agent.completed, "Agent should not have completed normally"
        assert scheduler._active_task is None, "Active task should be cleared"
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_foreground_during_idle_prevents_task_start():
    """Background tasks don't start while foreground is active."""
    agent = _SlowAgent()
    schedule = _AlwaysRunSchedule(agent)
    scheduler = BackgroundScheduler(
        schedules=[schedule],
        tick_interval=0.01,
    )

    # Block background tasks before starting
    scheduler.notify_foreground_start()

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Negative assertion: verify immediately. The agent must not have
        # started yet, since the scheduler hasn't had time for any tick.
        assert not agent.started.is_set(), "Agent should not have started while foreground active"

        # Release foreground — background should start
        scheduler.notify_foreground_end()
        await asyncio.wait_for(agent.started.wait(), timeout=2.0)

        assert agent.started.is_set(), "Agent should start after foreground ends"
    finally:
        agent.release.set()
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_command_does_not_cancel_background_task(
    signal_server, make_config, mock_llm, running_penny
):
    """Commands don't use Ollama, so they should not interrupt background tasks."""
    config = make_config(IDLE_SECONDS=0.0)

    async with running_penny(config) as penny:
        # Spy on the scheduler's notify_foreground_start
        original = penny.scheduler.notify_foreground_start
        calls: list[bool] = []

        def tracking_notify() -> None:
            calls.append(True)
            original()

        penny.scheduler.notify_foreground_start = tracking_notify

        # Send a command (doesn't need Ollama)
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")
        await signal_server.wait_for_message(timeout=5.0)

        assert not calls, "Command should not trigger notify_foreground_start"


@pytest.mark.asyncio
async def test_scheduler_skips_agents_with_no_work():
    """When a higher-priority agent returns False, lower-priority agents get a turn."""
    agent_a = _SimpleAgent("agent_a", return_value=False)
    agent_b = _SimpleAgent("agent_b", return_value=True)

    schedule_a = _AlwaysEligibleSchedule(agent_a)
    schedule_b = _AlwaysEligibleSchedule(agent_b)

    scheduler = BackgroundScheduler(
        schedules=[schedule_a, schedule_b],
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await wait_until(
            lambda: agent_a.execute_count > 0 and agent_b.execute_count > 0,
            timeout=2.0,
        )

        # Agent A (no work) was called, but agent B (has work) ran too
        assert agent_a.execute_count > 0, "Higher-priority agent should be called"
        assert agent_b.execute_count > 0, (
            "Lower-priority agent should run when higher returns False"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_scheduler_mark_complete_always_called():
    """mark_complete is called after every execution, even when agent has no work.

    Regression guard for the hot-loop bug: if mark_complete only runs on
    True returns, no-work agents stay perpetually eligible and burn a
    model call on every tick to rediscover the queue is empty.
    """
    agent_no_work = _SimpleAgent("no_work", return_value=False)
    agent_has_work = _SimpleAgent("has_work", return_value=True)

    schedule_no_work = _AlwaysEligibleSchedule(agent_no_work)
    schedule_has_work = _AlwaysEligibleSchedule(agent_has_work)

    scheduler = BackgroundScheduler(
        schedules=[schedule_no_work, schedule_has_work],
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await wait_until(lambda: schedule_has_work.mark_complete_count > 0, timeout=2.0)

        assert schedule_no_work.mark_complete_count > 0, (
            "mark_complete should be called even when agent returns False"
        )
        assert schedule_has_work.mark_complete_count > 0, (
            "mark_complete should be called when agent returns True"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_scheduler_breaks_when_agent_does_work():
    """When an agent does work, lower-priority agents don't run in the same tick."""
    agent_a = _SimpleAgent("agent_a", return_value=True)
    agent_b = _SimpleAgent("agent_b", return_value=True)

    schedule_a = _AlwaysEligibleSchedule(agent_a)
    schedule_b = _AlwaysEligibleSchedule(agent_b)

    scheduler = BackgroundScheduler(
        schedules=[schedule_a, schedule_b],
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await wait_until(lambda: agent_a.execute_count >= 3, timeout=2.0)

        # Agent A always does work and breaks — B never gets a turn
        assert agent_a.execute_count >= 3
        assert agent_b.execute_count == 0, (
            "Lower-priority agent should not run when higher-priority does work"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_schedules_gate_on_their_own_idle_threshold():
    """Each PeriodicSchedule reads its OWN idle_threshold against the scheduler's raw
    idle_seconds (#2779) — there is no single global gate any more. A schedule with a
    much higher threshold (like the collector's COLLECTOR_IDLE_SECONDS) stays
    ineligible for as long as its threshold hasn't cleared, while another schedule
    with a low threshold (like the send-queue drainer's IDLE_SECONDS) becomes eligible
    almost immediately — in the SAME tick, from the SAME elapsed idle time.
    """
    high_threshold_agent = _SimpleAgent("high_threshold", return_value=True)
    low_threshold_agent = _SimpleAgent("low_threshold", return_value=True)

    # Checked FIRST each tick, so its own should_run is actually exercised and not
    # merely skipped by priority order.
    high_threshold_schedule = PeriodicSchedule(
        agent=high_threshold_agent, interval=lambda: 0.01, idle_threshold=lambda: 999.0
    )
    low_threshold_schedule = PeriodicSchedule(
        agent=low_threshold_agent, interval=lambda: 0.01, idle_threshold=lambda: 0.0
    )

    scheduler = BackgroundScheduler(
        schedules=[high_threshold_schedule, low_threshold_schedule],
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await wait_until(lambda: low_threshold_agent.execute_count > 0, timeout=2.0)

        assert low_threshold_agent.execute_count > 0
        assert high_threshold_agent.execute_count == 0, (
            "A schedule gated on a much larger idle threshold should stay ineligible "
            "while another schedule's own (lower) threshold has already cleared"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task
