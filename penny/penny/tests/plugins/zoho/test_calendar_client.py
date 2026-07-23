"""Tests for the Zoho Calendar API client — endpoints + the shared OAuth base."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from penny.constants import PennyConstants
from penny.plugins.zoho.base_client import ZohoOAuthClient
from penny.plugins.zoho.calendar_client import ZohoCalendarClient
from penny.plugins.zoho.models import BusySlot, FreeSlot, ZohoSession
from penny.plugins.zoho.projects_client import ZohoProjectsClient


def _response(payload: dict, status_code: int = 200) -> MagicMock:
    """A stand-in for an httpx.Response with sync json()/raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


def _make_calendar_client() -> ZohoCalendarClient:
    """A calendar client whose httpx client is a harmless mock (no real socket)."""
    with patch("penny.plugins.zoho.base_client.httpx.AsyncClient"):
        return ZohoCalendarClient(client_id="id", client_secret="secret", refresh_token="refresh")


def test_both_clients_share_the_oauth_base():
    """Calendar and Projects clients both subclass the shared OAuth base."""
    assert issubclass(ZohoCalendarClient, ZohoOAuthClient)
    assert issubclass(ZohoProjectsClient, ZohoOAuthClient)


def test_client_defaults_to_named_timeout_constant():
    """The hardcoded 30.0 is now the named PennyConstants.ZOHO_CLIENT_TIMEOUT."""
    assert PennyConstants.ZOHO_CLIENT_TIMEOUT == 30.0
    with patch("penny.plugins.zoho.base_client.httpx.AsyncClient") as mock_cls:
        ZohoCalendarClient(client_id="id", client_secret="secret", refresh_token="token")
    mock_cls.assert_called_once_with(timeout=PennyConstants.ZOHO_CLIENT_TIMEOUT)


@pytest.mark.asyncio
async def test_get_headers_refreshes_token_via_shared_base():
    """_get_headers refreshes the token through the base and carries it."""
    client = _make_calendar_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(
        return_value=_response({"access_token": "tok-123", "expires_in": 3600})
    )

    headers = await client._get_headers()

    assert headers == {
        "Authorization": "Zoho-oauthtoken tok-123",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    client._http.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_token_is_cached_until_near_expiry():
    """A valid cached session short-circuits the refresh POST."""
    client = _make_calendar_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(
        return_value=_response({"access_token": "tok-1", "expires_in": 3600})
    )

    first = await client._ensure_access_token()
    second = await client._ensure_access_token()

    assert first == second == "tok-1"
    client._http.post.assert_awaited_once()  # cached — only one refresh


@pytest.mark.asyncio
async def test_oauth_error_raises():
    """An error field in the token response surfaces as a RuntimeError."""
    client = _make_calendar_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_response({"error": "invalid_grant"}))

    with pytest.raises(RuntimeError, match="Zoho OAuth error: invalid_grant"):
        await client._ensure_access_token()


@pytest.mark.asyncio
async def test_close_closes_the_http_client():
    """close() delegates to the base's httpx client aclose()."""
    client = _make_calendar_client()
    client._http = MagicMock()
    client._http.aclose = AsyncMock()

    await client.close()

    client._http.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_availability_returns_typed_busy_slots():
    """check_availability parses freebusy slots into typed BusySlot models."""
    client = _make_calendar_client()
    client._session = ZohoSession(access_token="tok", expires_at=9_999_999_999.0)
    client._http = MagicMock()
    client._http.get = AsyncMock(
        return_value=_response(
            {
                "freebusy": {
                    "busyslots": [
                        {"start": "20260722T100000Z", "end": "20260722T110000Z"},
                    ]
                }
            }
        )
    )

    slots = await client.check_availability(
        start=datetime(2026, 7, 22, 9, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 17, 0, tzinfo=UTC),
        attendees=["attendee@example.com"],
    )

    assert slots == [
        BusySlot(
            start=datetime(2026, 7, 22, 10, 0, tzinfo=UTC),
            end=datetime(2026, 7, 22, 11, 0, tzinfo=UTC),
        )
    ]


@pytest.mark.asyncio
async def test_check_availability_without_attendees_returns_empty():
    """No attendees → the freebusy path returns an empty list (unchanged)."""
    client = _make_calendar_client()
    client._session = ZohoSession(access_token="tok", expires_at=9_999_999_999.0)

    slots = await client.check_availability(
        start=datetime(2026, 7, 22, 9, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 17, 0, tzinfo=UTC),
        attendees=None,
    )

    assert slots == []


@pytest.mark.asyncio
async def test_find_free_slots_returns_typed_free_slots():
    """find_free_slots parses freeslots into typed FreeSlot models; bad rows drop."""
    client = _make_calendar_client()
    client._session = ZohoSession(access_token="tok", expires_at=9_999_999_999.0)
    client._http = MagicMock()
    client._http.get = AsyncMock(
        return_value=_response(
            {
                "freeslots": [
                    {"start": "20260722T140000Z", "end": "20260722T150000Z"},
                    {"start": "bad-value", "end": "20260722T160000Z"},
                ]
            }
        )
    )

    slots = await client.find_free_slots(
        duration_minutes=30,
        start=datetime(2026, 7, 22, 9, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 17, 0, tzinfo=UTC),
    )

    # The unparseable slot is skipped; the good one becomes a typed FreeSlot.
    assert slots == [
        FreeSlot(
            start=datetime(2026, 7, 22, 14, 0, tzinfo=UTC),
            end=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
        )
    ]


@pytest.mark.asyncio
async def test_create_event_converts_local_time_to_utc():
    """A tz-aware start/end is converted to UTC before the ``Z``-suffixed format —
    not stamped with the raw wall-clock fields (which mislabelled local time as
    UTC). Mirrors ``update_event``'s handling. The httpx client is constructed on
    the shared base, so that's where it is patched."""
    request = httpx.Request("POST", "https://calendar.zoho.com/api/v1/calendars/C1/events")
    with patch("penny.plugins.zoho.base_client.httpx.AsyncClient") as mock_client_cls:
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
