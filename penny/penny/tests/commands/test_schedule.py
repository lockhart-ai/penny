"""Integration tests for /schedule command."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from penny.constants import PennyConstants
from penny.database.models import Schedule, UserInfo
from penny.tests.conftest import TEST_SENDER, wait_until
from penny.tools.browse import BrowseTool


def _find_request(mock_llm, needle: str) -> str:
    """The content of the first captured LLM request containing needle."""
    for request in mock_llm.requests:
        for message in request["messages"]:
            if needle in message["content"]:
                return message["content"]
    raise AssertionError(f"No LLM request containing {needle!r}")


def _has_message(server, text: str) -> bool:
    """Check if any outgoing message contains text."""
    return any(text in msg.get("message", "") for msg in server.outgoing_messages)


def _find_message(server, text: str) -> dict:
    """Find the first outgoing message containing text. Must exist."""
    for msg in server.outgoing_messages:
        if text in msg.get("message", ""):
            return msg
    raise AssertionError(f"No message containing {text!r}")


def _is_schedule_due(cron_expression: str, now: datetime) -> bool:
    """Helper that mirrors the fixed ScheduleExecutor firing logic."""
    from croniter import croniter

    cron = croniter(cron_expression, now - timedelta(seconds=60))
    next_occurrence = cron.get_next(datetime)
    return next_occurrence <= now


def test_schedule_fires_at_exact_cron_time():
    """Schedule must fire when checked at the exact scheduled second.

    Regression test for the bug where croniter.get_prev(now) returned
    yesterday's occurrence when 'now' exactly equalled the cron time,
    causing the schedule to silently miss its tick.
    """
    tz = ZoneInfo("America/Los_Angeles")
    # Exactly at 9:30:00 — this is the problematic case
    now = datetime(2026, 2, 24, 9, 30, 0, tzinfo=tz)
    assert _is_schedule_due("30 9 * * *", now), "Schedule should fire at the exact cron second"


def test_schedule_fires_within_60_second_window():
    """Schedule should fire for any check within the 60-second window."""
    tz = ZoneInfo("America/Los_Angeles")
    for offset_seconds in [0, 1, 30, 59]:
        now = datetime(2026, 2, 24, 9, 30, offset_seconds, tzinfo=tz)
        assert _is_schedule_due("30 9 * * *", now), (
            f"Schedule should fire at +{offset_seconds}s past cron time"
        )


def test_schedule_does_not_fire_before_cron_time():
    """Schedule must not fire before the cron time."""
    tz = ZoneInfo("America/Los_Angeles")
    for offset_seconds in [1, 30, 59]:
        now = datetime(2026, 2, 24, 9, 29, 60 - offset_seconds, tzinfo=tz)
        assert not _is_schedule_due("30 9 * * *", now), (
            f"Schedule should NOT fire {offset_seconds}s before cron time"
        )


def test_schedule_does_not_fire_after_window():
    """Schedule must not fire more than 60 seconds after the cron time."""
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 2, 24, 9, 31, 0, tzinfo=tz)  # 60 seconds after 9:30
    assert not _is_schedule_due("30 9 * * *", now), (
        "Schedule should NOT fire 60 seconds after cron time"
    )


@pytest.mark.asyncio
async def test_schedule_list_empty(signal_server, test_config, mock_llm, running_penny):
    """Test /schedule with no schedules shows empty message."""
    async with running_penny(test_config) as penny:
        # Create user profile so we have timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Send /schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule")

        # Wait for response
        await wait_until(
            lambda: _has_message(signal_server, "You don't have any scheduled tasks yet")
        )


@pytest.mark.asyncio
async def test_schedule_create_requires_timezone(
    signal_server, test_config, mock_llm, running_penny
):
    """Test /schedule creation requires user timezone to be set."""
    async with running_penny(test_config) as _penny:
        # Try to create schedule without user profile
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule daily 9am what's the news?"
        )

        # Should prompt for timezone
        await wait_until(lambda: _has_message(signal_server, "I need to know your timezone first"))
        response = _find_message(signal_server, "I need to know your timezone first")
        assert response is not None
        assert "Send me your location" in response["message"]


@pytest.mark.asyncio
async def test_schedule_create_and_list(signal_server, test_config, mock_llm, running_penny):
    """Test creating a schedule and listing it."""
    schedule_json = (
        '{"timing_description": "daily 9am", '
        '"prompt_text": "what\'s the news?", '
        '"cron_expression": "0 9 * * *"}'
    )

    def handler(request, count):
        return mock_llm._make_text_response(request, schedule_json)

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        # Create user profile with timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Create schedule
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule daily 9am what's the news?"
        )

        # Should confirm creation
        await wait_until(lambda: _has_message(signal_server, "Added daily 9am: what's the news?"))

        # The parse prompt is grounded in the current date, rendered in the user's
        # profile timezone (LA) — never a bare UTC now() — so relative cadences
        # resolve against the right calendar day.  Bracket the render so a minute
        # rollover between snapshots can't flake the exact-stamp assertion.
        before = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            PennyConstants.CURRENT_DATETIME_FORMAT
        )
        parse_prompt = _find_request(mock_llm, "Parse this schedule command")
        after = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            PennyConstants.CURRENT_DATETIME_FORMAT
        )
        assert "Current date and time: " in parse_prompt
        assert any(f"Current date and time: {stamp}" in parse_prompt for stamp in (before, after))

        # List schedules
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule")

        # Should list the schedule
        await wait_until(lambda: _has_message(signal_server, "1. **daily 9am**: what's the news?"))


@pytest.mark.asyncio
async def test_schedule_delete(signal_server, test_config, mock_llm, running_penny):
    """Test deleting a schedule."""
    schedule_json = (
        '{"timing_description": "hourly", '
        '"prompt_text": "sports scores", '
        '"cron_expression": "0 * * * *"}'
    )

    def handler(request, count):
        return mock_llm._make_text_response(request, schedule_json)

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        # Create user profile with timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Create schedule
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule hourly sports scores"
        )

        await wait_until(lambda: _has_message(signal_server, "Added hourly: sports scores"))

        # Delete schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/unschedule 1")

        # Should confirm deletion
        await wait_until(lambda: _has_message(signal_server, "Deleted"))
        response = _find_message(signal_server, "Deleted")
        assert response is not None
        assert "Deleted 'hourly sports scores'" in response["message"]
        assert "No more scheduled tasks" in response["message"]


@pytest.mark.asyncio
async def test_schedule_delete_invalid_index(signal_server, test_config, mock_llm, running_penny):
    """Test deleting with invalid index shows error."""
    async with running_penny(test_config) as penny:
        # Create user profile
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Try to delete non-existent schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/unschedule 99")

        # Should show empty message (no schedules to delete)
        await wait_until(
            lambda: _has_message(signal_server, "You don't have any scheduled tasks yet")
        )


@pytest.mark.asyncio
async def test_schedule_executor_fires_through_chat_agent(
    signal_server, test_config, mock_llm, running_penny
):
    """A due schedule must execute through ChatAgent.handle() — installing tools
    and building the recall-grounded prompt — and deliver a response.

    Regression test: ScheduleExecutor called ``chat_agent.run()`` directly
    instead of going through ``handle()``.  That skipped ``_install_tools``, so a
    scheduled prompt ran with NO tools offered to the model — a "fetch me the
    news" schedule could only emit a browse call the loop stripped as a tool-less
    hallucination, then apologize it had nothing.  We assert the scheduled run
    offers the browse tool to the model, proving it goes through handle()."""

    def handler(request, count):
        return mock_llm._make_text_response(request, "morning! here's the news.")

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        with penny.db.get_session() as session:
            session.add(
                UserInfo(
                    sender=TEST_SENDER,
                    name="Test User",
                    location="Seattle",
                    timezone="America/Los_Angeles",
                    date_of_birth="1990-01-01",
                )
            )
            session.add(
                Schedule(
                    user_id=TEST_SENDER,
                    user_timezone="America/Los_Angeles",
                    cron_expression="* * * * *",
                    prompt_text="fetch the news",
                    timing_description="every minute",
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

        # Trigger the executor directly — the regression we're guarding
        # against is the ChatAgent crash path, not the scheduler's polling
        # timing. Calling ``execute()`` exercises the same path the
        # production ``AlwaysRunSchedule`` eventually triggers, without
        # waiting on the 60s background poll interval.
        await penny.schedule_executor.execute()

        await wait_until(
            lambda: _has_message(signal_server, "morning! here's the news."),
            timeout=5.0,
        )

        # The scheduled prompt must run with tools installed (browse + memory).
        # Find the LLM request for the scheduled prompt and assert browse was
        # offered — a tool-less request is the regression this guards.
        scheduled_request = next(
            r
            for r in mock_llm.requests
            if any("fetch the news" in str(m.get("content", "")) for m in r["messages"])
        )
        offered_tools = {t["function"]["name"] for t in (scheduled_request["tools"] or [])}
        assert BrowseTool.name in offered_tools
