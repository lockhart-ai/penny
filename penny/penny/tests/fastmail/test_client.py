"""Tests for JMAP client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.html_utils import strip_html
from penny.plugins.fastmail.client import JmapClient

_JMAP_TIMEOUT = float(RUNTIME_CONFIG_PARAMS["JMAP_REQUEST_TIMEOUT"].default)
_EMAIL_MAX_LENGTH = int(RUNTIME_CONFIG_PARAMS["EMAIL_BODY_MAX_LENGTH"].default)
_EMAIL_SEARCH_LIMIT = int(RUNTIME_CONFIG_PARAMS["EMAIL_SEARCH_LIMIT"].default)

FAKE_TOKEN = "fmu1-test-token"
FAKE_API_URL = "https://api.fastmail.com/jmap/api/"
FAKE_ACCOUNT_ID = "u12345678"

SESSION_RESPONSE = {
    "apiUrl": FAKE_API_URL,
    "primaryAccounts": {
        "urn:ietf:params:jmap:core": FAKE_ACCOUNT_ID,
        "urn:ietf:params:jmap:mail": FAKE_ACCOUNT_ID,
    },
}

QUERY_AND_GET_RESPONSE = {
    "methodResponses": [
        [
            "Email/query",
            {"ids": ["M001", "M002"], "total": 2},
            "0",
        ],
        [
            "Email/get",
            {
                "list": [
                    {
                        "id": "M001",
                        "subject": "Your package shipped",
                        "from": [{"name": "Amazon", "email": "ship@amazon.com"}],
                        "receivedAt": "2026-02-10T14:30:00Z",
                        "preview": "Your order has shipped...",
                    },
                    {
                        "id": "M002",
                        "subject": "Meeting tomorrow",
                        "from": [{"name": "Bob", "email": "bob@example.com"}],
                        "receivedAt": "2026-02-10T10:00:00Z",
                        "preview": "Reminder: team meeting at 10am",
                    },
                ]
            },
            "1",
        ],
    ]
}

READ_EMAIL_RESPONSE = {
    "methodResponses": [
        [
            "Email/get",
            {
                "list": [
                    {
                        "id": "M001",
                        "subject": "Your package shipped",
                        "from": [{"name": "Amazon", "email": "ship@amazon.com"}],
                        "to": [{"name": "User", "email": "user@fastmail.com"}],
                        "receivedAt": "2026-02-10T14:30:00Z",
                        "textBody": [{"partId": "1"}],
                        "htmlBody": [{"partId": "2"}],
                        "bodyValues": {
                            "1": {"value": "Your order #123 has shipped!"},
                            "2": {"value": "<p>Your order <b>#123</b> has shipped!</p>"},
                        },
                    }
                ]
            },
            "0",
        ]
    ]
}

READ_EMAIL_HTML_ONLY_RESPONSE = {
    "methodResponses": [
        [
            "Email/get",
            {
                "list": [
                    {
                        "id": "M003",
                        "subject": "HTML only email",
                        "from": [{"email": "sender@example.com"}],
                        "to": [{"email": "user@fastmail.com"}],
                        "receivedAt": "2026-02-10T12:00:00Z",
                        "textBody": [],
                        "htmlBody": [{"partId": "1"}],
                        "bodyValues": {
                            "1": {"value": "<h1>Hello</h1><p>World</p>"},
                        },
                    }
                ]
            },
            "0",
        ]
    ]
}

READ_EMAIL_NOT_FOUND_RESPONSE = {
    "methodResponses": [
        [
            "Email/get",
            {"list": [], "notFound": ["M999"]},
            "0",
        ]
    ]
}


def _make_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://api.fastmail.com/jmap/api/"),
    )


@pytest.mark.asyncio
async def test_session_fetched_and_cached():
    """Test that the JMAP session is fetched once and cached."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(QUERY_AND_GET_RESPONSE)

            # First call fetches session
            await client.search_emails(text="test")
            assert mock_get.call_count == 1

            # Second call reuses cached session
            await client.search_emails(text="test2")
            assert mock_get.call_count == 1

    await client.close()


@pytest.mark.asyncio
async def test_search_emails_returns_summaries():
    """Test that search_emails parses JMAP response into EmailSummary objects."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(QUERY_AND_GET_RESPONSE)

            results = await client.search_emails(text="package")

    assert len(results) == 2
    assert results[0].id == "M001"
    assert results[0].subject == "Your package shipped"
    assert results[0].from_addresses[0].email == "ship@amazon.com"
    assert results[1].id == "M002"
    await client.close()


@pytest.mark.asyncio
async def test_search_emails_builds_filter():
    """Test that search parameters are passed as JMAP filter properties."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(QUERY_AND_GET_RESPONSE)

            await client.search_emails(
                text="hello",
                from_addr="bob@example.com",
                after="2026-01-01T00:00:00Z",
            )

            # Check the POST body
            call_args = mock_post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json") or call_args[0][1]
            method_calls = body["methodCalls"]
            query_filter = method_calls[0][1]["filter"]
            assert query_filter["text"] == "hello"
            assert query_filter["from"] == "bob@example.com"
            assert query_filter["after"] == "2026-01-01T00:00:00Z"
            assert "subject" not in query_filter
            assert "before" not in query_filter

    await client.close()


@pytest.mark.asyncio
async def test_read_emails_returns_details():
    """Test that read_emails parses full email bodies from bodyValues."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(READ_EMAIL_RESPONSE)

            results = await client.read_emails(["M001"])

    assert len(results) == 1
    assert results[0].id == "M001"
    assert results[0].subject == "Your package shipped"
    assert "Your order #123 has shipped!" in results[0].text_body
    await client.close()


@pytest.mark.asyncio
async def test_read_emails_falls_back_to_html():
    """Test that read_emails strips HTML tags when no text body is available."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(READ_EMAIL_HTML_ONLY_RESPONSE)

            results = await client.read_emails(["M003"])

    assert len(results) == 1
    assert "Hello" in results[0].text_body
    assert "World" in results[0].text_body
    assert "<" not in results[0].text_body
    await client.close()


@pytest.mark.asyncio
async def test_read_emails_not_found():
    """Test that read_emails returns empty list for missing email."""
    client = JmapClient(
        FAKE_TOKEN,
        timeout=_JMAP_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
    )

    with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _make_response(SESSION_RESPONSE)

        with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _make_response(READ_EMAIL_NOT_FOUND_RESPONSE)

            results = await client.read_emails(["M999"])

    assert results == []
    await client.close()


def teststrip_html():
    """Test HTML tag stripping utility."""
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_html("no tags here") == "no tags here"
    assert strip_html("<div><span>nested</span></div>") == "nested"
    assert strip_html("") == ""
