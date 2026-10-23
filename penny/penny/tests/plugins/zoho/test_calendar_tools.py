"""Tests for the Zoho Calendar tools."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from penny.llm.call_link import CallLink
from penny.plugins.zoho.calendar_tools import (
    CALENDAR_AVAILABILITY_FAILED,
    CALENDAR_EVENT_CREATE_FAILED,
    CALENDAR_EVENT_UPDATE_FAILED,
    CALENDAR_EVENTS_FAILED,
    CALENDAR_FREE_SLOTS_FAILED,
    CALENDAR_LIST_FAILED,
    CheckAvailabilityTool,
    CreateEventTool,
    FindFreeSlotsTool,
    GetEventsTool,
    ListCalendarsTool,
    UpdateEventTool,
)
from penny.plugins.zoho.models import BusySlot, FreeSlot, ZohoCalendarInfo, ZohoEvent


@pytest.mark.asyncio
async def test_check_availability_renders_typed_busy_slots():
    """CheckAvailabilityTool reads the typed BusySlot fields (start/end), not dicts."""
    client = MagicMock()
    client.check_availability = AsyncMock(
        return_value=[
            BusySlot(
                start=datetime(2026, 7, 22, 10, 0, tzinfo=UTC),
                end=datetime(2026, 7, 22, 11, 0, tzinfo=UTC),
            )
        ]
    )

    result = await CheckAvailabilityTool(client).execute(
        CallLink.new_top(),
        start_date="2026-07-22T09:00:00",
        end_date="2026-07-22T17:00:00",
        attendees=["attendee@example.com"],
    )

    assert result.success
    assert result.message == (
        "The requested time has conflicts:\n\n- Busy: 2026-07-22 10:00 to 11:00"
    )


@pytest.mark.asyncio
async def test_check_availability_reports_free_range_when_no_busy_slots():
    """No busy slots (with attendees) renders the range as available."""
    client = MagicMock()
    client.check_availability = AsyncMock(return_value=[])

    result = await CheckAvailabilityTool(client).execute(
        CallLink.new_top(),
        start_date="2026-07-22T09:00:00",
        end_date="2026-07-22T17:00:00",
        attendees=["attendee@example.com"],
    )

    assert result.success
    assert result.message == "The time slot 2026-07-22 09:00 to 17:00 is **available**."


@pytest.mark.asyncio
async def test_find_free_slots_renders_typed_slot():
    """FindFreeSlotsTool reads the typed FreeSlot fields (start/end), not dicts."""
    client = MagicMock()
    client.find_free_slots = AsyncMock(
        return_value=[
            FreeSlot(
                start=datetime(2026, 7, 22, 14, 0, tzinfo=UTC),
                end=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
            )
        ]
    )

    result = await FindFreeSlotsTool(client).execute(CallLink.new_top(), duration_minutes=30)

    assert result.success
    assert result.message == (
        "Found 1 available slot(s) of 30 minutes:\n\n- 2026-07-22 14:00 to 15:00"
    )


@pytest.mark.asyncio
async def test_find_free_slots_lists_every_slot():
    """Every free slot is rendered — no invented ``[:10]`` truncation of the result."""
    client = MagicMock()
    slots = [
        FreeSlot(
            start=datetime(2026, 7, 22, 9 + i, 0, tzinfo=UTC),
            end=datetime(2026, 7, 22, 9 + i, 30, tzinfo=UTC),
        )
        for i in range(12)
    ]
    client.find_free_slots = AsyncMock(return_value=slots)

    result = await FindFreeSlotsTool(client).execute(CallLink.new_top(), duration_minutes=30)

    assert result.success
    assert "Found 12 available slot(s)" in result.message
    assert result.message.count("- 2026-07-22") == 12
    assert "more slots available" not in result.message


@pytest.mark.asyncio
async def test_create_event_reports_missing_calendar_as_failure():
    """A create against a non-existent named calendar is an honest failure — the model
    must not read the un-created event as a success."""
    client = MagicMock()
    client.get_calendar_by_name = AsyncMock(return_value=None)

    result = await CreateEventTool(client).execute(
        CallLink.new_top(),
        title="Standup",
        start="2026-07-22T10:00:00",
        end="2026-07-22T11:00:00",
        calendar_name="Nonexistent",
    )

    assert result.success is False
    assert "Calendar not found" in result.message


def _calendar() -> ZohoCalendarInfo:
    return ZohoCalendarInfo(caluid="C1", name="Default", is_default=True)


@pytest.mark.asyncio
async def test_update_event_applies_changes_and_renders_summary():
    """A full update (decomposed into resolve-calendar / find-event / load-detail /
    resolve-times / compute-recurrence / build-result) renders the same change summary
    and calls update_event with the resolved values."""
    client = MagicMock()
    client.get_default_calendar = AsyncMock(return_value=_calendar())
    client.get_events = AsyncMock(
        return_value=[
            ZohoEvent(uid="E1", title="Standup", start=datetime(2026, 7, 22, 10, 0, tzinfo=UTC))
        ]
    )
    client.get_event = AsyncMock(
        return_value=ZohoEvent(
            uid="E1",
            title="Standup",
            start=datetime(2026, 7, 22, 10, 0, tzinfo=UTC),
            end=datetime(2026, 7, 22, 11, 0, tzinfo=UTC),
            etag=7,
            timezone="UTC",
        )
    )
    # The mock mirrors what the real endpoint returns: the response it parses
    # (``_event_from_update_response``) echoes the CONFIRMED fields back, which is
    # what the result now renders from (#2768) — never the request's own args.
    client.update_event = AsyncMock(
        return_value=ZohoEvent(
            uid="E1",
            title="Standup",
            start=datetime(2026, 7, 22, 14, 0),
            end=datetime(2026, 7, 22, 15, 0),
            location="Room 2",
        )
    )

    result = await UpdateEventTool(client).execute(
        CallLink.new_top(),
        event_title="Standup",
        new_start="2026-07-22T14:00:00",
        new_end="2026-07-22T15:00:00",
        new_location="Room 2",
    )

    assert result.success
    assert result.mutated is True
    assert result.message == (
        "Event updated successfully (all occurrences):\n\n"
        "**Standup**\n"
        "Time: 2026-07-22 14:00 to 15:00\n"
        "Location: Room 2"
    )
    await_args = client.update_event.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["etag"] == 7
    # _parse_iso_datetime keeps a bare ISO string naive (no 'Z' → no tzinfo).
    assert kwargs["start"] == datetime(2026, 7, 22, 14, 0)
    assert kwargs["end"] == datetime(2026, 7, 22, 15, 0)
    # A non-recurring event keeps the requested edit type.
    assert kwargs["recurrence_edittype"] == "all"
    assert kwargs["recurrenceid"] is None


@pytest.mark.asyncio
async def test_update_recurring_event_switches_edittype_and_sets_recurrenceid():
    """The compute-recurrence step flips 'all'→'following' on a recurring time change and
    derives the recurrenceid from the matched occurrence — while the rendered edit_scope
    still reflects the caller's original arg ('all')."""
    occurrence = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
    client = MagicMock()
    client.get_default_calendar = AsyncMock(return_value=_calendar())
    client.get_events = AsyncMock(
        return_value=[ZohoEvent(uid="E1", title="Weekly sync", start=occurrence)]
    )
    client.get_event = AsyncMock(
        return_value=ZohoEvent(
            uid="E1",
            title="Weekly sync",
            start=occurrence,
            end=datetime(2026, 7, 22, 11, 0, tzinfo=UTC),
            etag=3,
            is_recurring=True,
            rrule="FREQ=WEEKLY",
        )
    )
    client.update_event = AsyncMock(return_value=ZohoEvent(uid="E1", title="Weekly sync"))

    result = await UpdateEventTool(client).execute(
        CallLink.new_top(),
        event_title="Weekly sync",
        new_start="2026-07-29T10:00:00",
        new_end="2026-07-29T11:00:00",
    )

    assert result.success
    await_args = client.update_event.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["recurrence_edittype"] == "following"
    assert kwargs["recurrenceid"] == "20260722T100000Z"
    assert kwargs["rrule"] == "FREQ=WEEKLY"
    assert "(all occurrences)" in result.message


@pytest.mark.asyncio
async def test_update_event_reports_missing_calendar_as_failure():
    """The resolve-calendar step returns the honest failure for an unknown named calendar."""
    client = MagicMock()
    client.get_calendar_by_name = AsyncMock(return_value=None)

    result = await UpdateEventTool(client).execute(
        CallLink.new_top(), event_title="Standup", calendar_name="Nope"
    )

    assert result.success is False
    assert result.message == "Calendar not found: Nope. Check the calendar name and try again."


@pytest.mark.asyncio
async def test_update_event_reports_missing_event_as_failure():
    """The find-event step returns the honest failure when no event matches the title."""
    client = MagicMock()
    client.get_default_calendar = AsyncMock(return_value=_calendar())
    client.get_events = AsyncMock(return_value=[])

    result = await UpdateEventTool(client).execute(CallLink.new_top(), event_title="Ghost")

    assert result.success is False
    assert result.message == (
        "No event found matching 'Ghost' in calendar 'Default'. "
        "Please check the event title and try again."
    )


@pytest.mark.asyncio
async def test_update_event_names_other_matches_when_title_is_ambiguous():
    """More than one event sharing the title picks the soonest-starting match, but
    says so (#2768) — a silent pick read as an unambiguous update when it wasn't."""
    first = datetime(2026, 7, 22, 9, 0, tzinfo=UTC)
    second = datetime(2026, 7, 23, 9, 0, tzinfo=UTC)
    client = MagicMock()
    client.get_default_calendar = AsyncMock(return_value=_calendar())
    client.get_events = AsyncMock(
        return_value=[
            ZohoEvent(uid="E1", title="Standup", start=first),
            ZohoEvent(uid="E2", title="Standup", start=second),
        ]
    )
    client.get_event = AsyncMock(
        return_value=ZohoEvent(
            uid="E1", title="Standup", start=first, end=first, etag=7, timezone="UTC"
        )
    )
    client.update_event = AsyncMock(
        return_value=ZohoEvent(uid="E1", title="Standup", start=first, end=first)
    )

    result = await UpdateEventTool(client).execute(
        CallLink.new_top(), event_title="Standup", new_location="Room 2"
    )

    assert result.success
    assert (
        "Heads up: 2 events matched 'Standup' — updated the one starting "
        "2026-07-22 09:00; the others weren't touched."
    ) in result.message


# ── Calendar tools report their own client failures (#2369) ──────────────────
# Each tool's backend call is wrapped: a raised failure is caught and narrated — what
# happened, with no order attached — rather than falling through to the executor's generic
# "check the arguments" narration, which makes no sense for a backend outage. Mocked at the
# backend boundary (the client each tool holds), the same seam every other test in this file
# mocks at. The catch is narrow — ``httpx.HTTPError`` (a non-2xx response or a transport
# error) plus ``RuntimeError`` (the calendar client's shared ``ZohoOAuthClient`` raises this
# on a failed token refresh) — so a bug in this file's own logic is never mislabeled as a
# calendar outage.


def _http_401() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://mail.zoho.com/api")
    response = httpx.Response(401, request=request)
    return httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)


@pytest.mark.asyncio
async def test_list_calendars_reports_a_raised_listing():
    client = MagicMock()
    client.get_calendars = AsyncMock(side_effect=_http_401())

    result = await ListCalendarsTool(client).execute(CallLink.new_top())

    assert result.success is False
    assert result.message == CALENDAR_LIST_FAILED.format(e=_http_401())


@pytest.mark.asyncio
async def test_get_events_reports_a_raised_fetch():
    client = MagicMock()
    client.get_default_calendar = AsyncMock(
        side_effect=RuntimeError("Zoho OAuth error: invalid_grant")
    )

    result = await GetEventsTool(client).execute(CallLink.new_top())

    assert result.success is False
    assert result.message == CALENDAR_EVENTS_FAILED.format(e="Zoho OAuth error: invalid_grant")


@pytest.mark.asyncio
async def test_check_availability_reports_a_raised_check():
    client = MagicMock()
    client.check_availability = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CheckAvailabilityTool(client).execute(
        CallLink.new_top(),
        start_date="2026-07-22T09:00:00",
        end_date="2026-07-22T17:00:00",
        attendees=["attendee@example.com"],
    )

    assert result.success is False
    assert result.message == CALENDAR_AVAILABILITY_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_event_reports_a_raised_create():
    client = MagicMock()
    client.get_default_calendar = AsyncMock(return_value=_calendar())
    client.create_event = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateEventTool(client).execute(
        CallLink.new_top(),
        title="Standup",
        start="2026-07-22T10:00:00",
        end="2026-07-22T11:00:00",
    )

    assert result.success is False
    assert result.message == CALENDAR_EVENT_CREATE_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_find_free_slots_reports_a_raised_search():
    client = MagicMock()
    client.find_free_slots = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await FindFreeSlotsTool(client).execute(CallLink.new_top(), duration_minutes=30)

    assert result.success is False
    assert result.message == CALENDAR_FREE_SLOTS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_update_event_reports_a_raised_update():
    client = MagicMock()
    client.get_default_calendar = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await UpdateEventTool(client).execute(CallLink.new_top(), event_title="Standup")

    assert result.success is False
    assert result.message == CALENDAR_EVENT_UPDATE_FAILED.format(e="connection refused")
