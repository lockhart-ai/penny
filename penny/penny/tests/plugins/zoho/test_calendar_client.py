"""Tests for the Zoho Calendar API client."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from penny.plugins.zoho.calendar_client import ZohoCalendarClient


@pytest.mark.asyncio
async def test_create_event_converts_local_time_to_utc():
    """A tz-aware start/end is converted to UTC before the ``Z``-suffixed format —
    not stamped with the raw wall-clock fields (which mislabelled local time as
    UTC). Mirrors ``update_event``'s handling."""
    request = httpx.Request("POST", "https://calendar.zoho.com/api/v1/calendars/C1/events")
    with patch("penny.plugins.zoho.calendar_client.httpx.AsyncClient") as mock_client_cls:
        client = ZohoCalendarClient(client_id="id", client_secret="secret", refresh_token="token")
        client._get_headers = AsyncMock(return_value={})
        mock_http = mock_client_cls.return_value
        mock_http.post = AsyncMock(
            return_value=httpx.Response(200, json={"events": [{"uid": "E1"}]}, request=request)
        )

        plus_two = timezone(timedelta(hours=2))
        await client.create_event(
            caluid="C1",
            title="Standup",
            start=datetime(2026, 7, 22, 10, 0, tzinfo=plus_two),
            end=datetime(2026, 7, 22, 11, 0, tzinfo=plus_two),
        )

    eventdata = json.loads(mock_http.post.call_args.kwargs["params"]["eventdata"])
    assert eventdata["dateandtime"]["start"] == "20260722T080000Z"
    assert eventdata["dateandtime"]["end"] == "20260722T090000Z"
