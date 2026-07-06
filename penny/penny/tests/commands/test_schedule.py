"""Integration tests for the schedule tools + the schedule executor mechanism.

The user-facing surface is the ``schedule_create`` / ``schedule_delete`` /
``schedule_list`` tools the chat agent drives from natural language (the
``/schedule`` and ``/unschedule`` commands retired in epic #1445).  These tests
pin the tools' mechanism deterministically (the NL→cron parse is mocked at the
model boundary); the live-model NL-dispatch contracts live in
``tests/eval/test_schedule_dispatch.py``.  The cron-firing + executor tests below
guard the ScheduleExecutor and are unrelated to the surface change.
"""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session, select

from penny.database.models import Schedule, UserInfo
from penny.tests.conftest import TEST_SENDER, wait_until
from penny.tools import Tool
from penny.tools.browse import BrowseTool


def _has_message(server, text: str) -> bool:
    """Check if any outgoing message contains text."""
    return any(text in msg.get("message", "") for msg in server.outgoing_messages)


def _schedule_tool(penny, name: str) -> Tool:
    """The wired schedule tool from the chat surface — proves registration."""
    tool = next((t for t in penny.chat_agent.get_tools() if t.name == name), None)
    assert tool is not None, f"{name} is not registered on the chat surface"
    return tool


def _seed_user(penny, *, timezone: str = "America/Los_Angeles") -> None:
    """Create the test user profile (get_primary_sender reads the first UserInfo)."""
    with penny.db.get_session() as session:
        session.add(
            UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone=timezone,
                date_of_birth="1990-01-01",
            )
        )
        session.commit()


def _schedules(penny) -> list[Schedule]:
    with Session(penny.db.engine) as session:
        return list(session.exec(select(Schedule)))


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
async def test_schedule_create_tool_writes_row(signal_server, test_config, mock_llm, running_penny):
    """schedule_create parses the NL request (mocked) into a Schedule row and mirrors it back."""
    schedule_json = (
        '{"timing_description": "daily 9am", '
        '"prompt_text": "what\'s the news?", '
        '"cron_expression": "0 9 * * *"}'
    )
    mock_llm.set_response_handler(
        lambda request, count: mock_llm._make_text_response(request, schedule_json)
    )

    async with running_penny(test_config) as penny:
        _seed_user(penny)
        tool = _schedule_tool(penny, "schedule_create")

        result = await tool.run(request="every day at 9am what's the news?")

        assert result.success and result.mutated
        assert "0 9 * * *" in result.message  # honest mirror-back of the parsed cadence
        schedules = _schedules(penny)
        assert len(schedules) == 1
        assert schedules[0].cron_expression == "0 9 * * *"
        assert schedules[0].user_timezone == "America/Los_Angeles"
        assert schedules[0].prompt_text == "what's the news?"

        # The parse is grounded in the user's timezone clock, never a bare UTC now().
        parse_request = next(
            r
            for r in mock_llm.requests
            if any(
                "Parse this schedule command" in str(m.get("content", "")) for m in r["messages"]
            )
        )
        assert any(
            "America/Los_Angeles" in str(m.get("content", "")) for m in parse_request["messages"]
        )


@pytest.mark.asyncio
async def test_schedule_create_tool_needs_timezone(
    signal_server, test_config, mock_llm, running_penny
):
    """Without a profile timezone, schedule_create fails actionably and writes nothing."""
    async with running_penny(test_config) as penny:
        _seed_user(penny, timezone="")
        tool = _schedule_tool(penny, "schedule_create")

        result = await tool.run(request="every day at 9am what's the news?")

        assert not result.success
        assert "timezone" in result.message.lower()
        assert _schedules(penny) == []


@pytest.mark.asyncio
async def test_schedule_delete_tool_removes_match(
    signal_server, test_config, mock_llm, running_penny
):
    """schedule_delete removes the matching schedule and names what it removed."""
    async with running_penny(test_config) as penny:
        _seed_user(penny)
        with Session(penny.db.engine) as session:
            session.add(
                Schedule(
                    user_id=TEST_SENDER,
                    user_timezone="America/Los_Angeles",
                    cron_expression="0 * * * *",
                    prompt_text="sports scores",
                    timing_description="hourly",
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()
        tool = _schedule_tool(penny, "schedule_delete")

        result = await tool.run(description="the hourly sports scores")

        assert result.success and result.mutated
        assert "sports scores" in result.message  # mirror-back of what was removed
        assert _schedules(penny) == []


@pytest.mark.asyncio
async def test_schedule_delete_tool_matches_by_meaning(
    signal_server, test_config, mock_llm, running_penny
):
    """With multiple schedules, schedule_delete removes the nearest by embedding — not
    the oldest by index. Pins the argmax match mechanism with controlled embeddings
    (the live-model semantic contract lives in tests/eval/test_schedule_dispatch.py)."""
    async with running_penny(test_config) as penny:
        _seed_user(penny)
        with Session(penny.db.engine) as session:
            session.add(
                Schedule(
                    user_id=TEST_SENDER,
                    user_timezone="America/Los_Angeles",
                    cron_expression="0 7 * * *",
                    prompt_text="weather forecast",
                    timing_description="every morning at 7am",
                    created_at=datetime.now(UTC),
                )
            )
            session.add(
                Schedule(
                    user_id=TEST_SENDER,
                    user_timezone="America/Los_Angeles",
                    cron_expression="0 * * * *",
                    prompt_text="sports scores",
                    timing_description="hourly",
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

        # Steer the match: the query and the weather schedule share a vector, the
        # sports one is orthogonal — so cosine picks weather regardless of insert order.
        def embed_handler(model, text):
            texts = [text] if isinstance(text, str) else text
            return [[1.0, 0.0] if "weather" in t.lower() else [0.0, 1.0] for t in texts]

        mock_llm.set_embed_handler(embed_handler)
        tool = _schedule_tool(penny, "schedule_delete")

        result = await tool.run(description="the weather forecast")

        assert result.success and result.mutated
        remaining = {s.prompt_text for s in _schedules(penny)}
        assert remaining == {"sports scores"}  # the nearest match, not the oldest, was removed


@pytest.mark.asyncio
async def test_schedule_delete_tool_no_schedules(
    signal_server, test_config, mock_llm, running_penny
):
    """schedule_delete with nothing scheduled declines actionably."""
    async with running_penny(test_config) as penny:
        _seed_user(penny)
        tool = _schedule_tool(penny, "schedule_delete")

        result = await tool.run(description="the morning digest")

        assert not result.success
        assert "no scheduled tasks" in result.message.lower()


@pytest.mark.asyncio
async def test_schedule_list_tool(signal_server, test_config, mock_llm, running_penny):
    """schedule_list reports the user's schedules, and an empty state when there are none."""
    async with running_penny(test_config) as penny:
        _seed_user(penny)
        tool = _schedule_tool(penny, "schedule_list")

        empty = await tool.run()
        assert "empty" in empty.message.lower()

        with Session(penny.db.engine) as session:
            session.add(
                Schedule(
                    user_id=TEST_SENDER,
                    user_timezone="America/Los_Angeles",
                    cron_expression="0 9 * * *",
                    prompt_text="what's the news?",
                    timing_description="daily 9am",
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

        listed = await tool.run()
        assert "daily 9am" in listed.message
        assert "what's the news?" in listed.message


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
