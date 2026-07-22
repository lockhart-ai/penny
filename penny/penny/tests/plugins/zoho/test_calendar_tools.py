"""Tests for the Zoho Calendar tools."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.plugins.zoho.calendar_tools import CreateEventTool, FindFreeSlotsTool


@pytest.mark.asyncio
async def test_find_free_slots_lists_every_slot():
    """Every free slot is rendered — no invented ``[:10]`` truncation of the result."""
    client = MagicMock()
    slots = [
        {
            "start": datetime(2026, 7, 22, 9 + i, 0, tzinfo=UTC),
            "end": datetime(2026, 7, 22, 9 + i, 30, tzinfo=UTC),
        }
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
