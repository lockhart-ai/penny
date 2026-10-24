"""Unit tests for PeriodicSchedule."""

from unittest.mock import MagicMock

import pytest

from penny.scheduler.schedules import PeriodicSchedule


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "test_agent"
    return agent


def test_periodic_schedule_first_run_once_idle_threshold_cleared(mock_agent):
    """PeriodicSchedule fires on its first check once idle_seconds clears its threshold."""
    schedule = PeriodicSchedule(
        agent=mock_agent, interval=lambda: 60.0, idle_threshold=lambda: 60.0
    )

    assert schedule.should_run(idle_seconds=60.0)


def test_periodic_schedule_not_run_below_its_idle_threshold(mock_agent):
    """PeriodicSchedule does not fire while idle_seconds is below its OWN threshold."""
    schedule = PeriodicSchedule(
        agent=mock_agent, interval=lambda: 60.0, idle_threshold=lambda: 60.0
    )

    assert not schedule.should_run(idle_seconds=10.0)


def test_periodic_schedule_respects_interval(mock_agent):
    """Test that PeriodicSchedule respects the interval between runs."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1, idle_threshold=lambda: 0.0)

    # First run should fire
    assert schedule.should_run(idle_seconds=0.0)
    schedule.mark_complete()

    # Immediately after, should not fire
    assert not schedule.should_run(idle_seconds=0.0)

    # Advance past the interval by backdating _last_run
    assert schedule._last_run is not None
    schedule._last_run = schedule._last_run - 0.2

    # Should fire again after interval
    assert schedule.should_run(idle_seconds=0.0)


def test_periodic_schedule_reset_clears_last_run(mock_agent):
    """Test that reset() clears the last run time."""
    schedule = PeriodicSchedule(
        agent=mock_agent, interval=lambda: 60.0, idle_threshold=lambda: 60.0
    )

    # Fire once and mark complete
    schedule.should_run(idle_seconds=60.0)
    schedule.mark_complete()

    # After reset, should fire again immediately once its threshold clears
    schedule.reset()
    assert schedule.should_run(idle_seconds=60.0)


def test_zero_idle_threshold_runs_with_no_idle_at_all(mock_agent):
    """A schedule with no idle requirement passes idle_threshold=lambda: 0.0 (#2779) — the
    bool flag retired; a schedule is either gated on a positive idle requirement or not,
    stated as the threshold value itself, never a separate ``requires_idle`` switch."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1, idle_threshold=lambda: 0.0)

    # idle_seconds=0.0 already clears a zero threshold.
    assert schedule.should_run(idle_seconds=0.0)
    schedule.mark_complete()

    # Backdate to simulate the interval elapsing — still gated only by the interval,
    # never by idle state, since the threshold is 0.
    assert schedule._last_run is not None
    schedule._last_run -= 0.2
    assert schedule.should_run(idle_seconds=0.0)


def test_reset_clears_last_run_regardless_of_threshold(mock_agent):
    """reset() always clears last_run on message arrival, whatever the schedule's own
    idle threshold is — there is no longer a distinct "idle-independent" shape that
    skips the reset."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1, idle_threshold=lambda: 0.0)
    schedule.mark_complete()
    assert schedule._last_run is not None
    schedule._last_run -= 0.2  # Simulate interval elapsed

    schedule.reset()

    assert schedule._last_run is None
    assert schedule.should_run(idle_seconds=0.0)
