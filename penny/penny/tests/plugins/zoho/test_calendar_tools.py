"""Tests for the Zoho Calendar tools."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.plugins.zoho.calendar_tools import (
    CheckAvailabilityTool,
    CreateEventTool,
    FindFreeSlotsTool,
)
from penny.plugins.zoho.models import BusySlot, FreeSlot


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

    result = await FindFreeSlotsTool(client).execute(duration_minutes=30)

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

    result = await FindFreeSlotsTool(client).execute(duration_minutes=30)

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
        title="Standup",
        start="2026-07-22T10:00:00",
        end="2026-07-22T11:00:00",
        calendar_name="Nonexistent",
    )

    assert result.success is False
    assert "Calendar not found" in result.message
