"""Tests for Zoho Mail API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.html_utils import strip_html
from penny.plugins.zoho.mail_client import ZohoClient

_ZOHO_TIMEOUT = float(RUNTIME_CONFIG_PARAMS["JMAP_REQUEST_TIMEOUT"].default)
_EMAIL_MAX_LENGTH = int(RUNTIME_CONFIG_PARAMS["EMAIL_BODY_MAX_LENGTH"].default)
_EMAIL_SEARCH_LIMIT = int(RUNTIME_CONFIG_PARAMS["EMAIL_SEARCH_LIMIT"].default)
_EMAIL_LIST_LIMIT = int(RUNTIME_CONFIG_PARAMS["EMAIL_LIST_LIMIT"].default)

FAKE_CLIENT_ID = "1000.TESTCLIENTID"
FAKE_CLIENT_SECRET = "testsecret123"
FAKE_REFRESH_TOKEN = "1000.testrefreshtoken"
FAKE_ACCESS_TOKEN = "1000.testaccesstoken"
FAKE_ACCOUNT_ID = "123456789"

TOKEN_RESPONSE = {
    "access_token": FAKE_ACCESS_TOKEN,
    "expires_in": 3600,
    "api_domain": "https://www.zohoapis.com",
    "token_type": "Bearer",
}

ACCOUNTS_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": [
        {
            "accountId": FAKE_ACCOUNT_ID,
            "emailAddress": [{"mailId": "test@zohomail.com"}],
            "displayName": "Test User",
        }
    ],
}

SEARCH_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": [
        {
            "messageId": "M001",
            "folderId": "F001",
            "subject": "Your package shipped",
            "fromAddress": "ship@amazon.com",
            "sender": "Amazon",
            "receivedTime": 1707573000000,
            "summary": "Your order has shipped...",
        },
        {
            "messageId": "M002",
            "folderId": "F001",
            "subject": "Meeting tomorrow",
            "fromAddress": "bob@example.com",
            "sender": "Bob",
            "receivedTime": 1707556800000,
            "summary": "Reminder: team meeting at 10am",
        },
    ],
}

# content endpoint response
EMAIL_CONTENT_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": {
        "subject": "Your package shipped",
        "fromAddress": "ship@amazon.com",
        "sender": "Amazon",
        "toAddress": "user@zohomail.com",
        "receivedTime": 1707573000000,
        "content": "Your order #123 has shipped!",
    },
}

EMAIL_CONTENT_HTML_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": {
        "subject": "HTML Email",
        "fromAddress": "sender@example.com",
        "content": "<h1>Hello</h1><p>World</p>",
    },
}


def _make_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://mail.zoho.com/api/"),
    )


@pytest.mark.asyncio
async def test_access_token_refreshed_and_cached():
    """Test that the access token is refreshed and cached."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response({"data": []}),
                _make_response({"data": []}),  # Second search call
            ]

            # First call refreshes token
            await client.search_emails(text="test")
            assert mock_post.call_count == 1

            # Second call reuses cached token (account also cached)
            await client.search_emails(text="test2")
            assert mock_post.call_count == 1

    await client.close()


@pytest.mark.asyncio
async def test_search_emails_returns_summaries():
    """Test that search_emails parses Zoho response into EmailSummary objects."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(SEARCH_RESPONSE),
            ]

            results = await client.search_emails(text="package")

    assert len(results) == 2
    # IDs are folderId:messageId format (URI not available from Zoho search API)
    assert results[0].id == "F001:M001"
    assert results[0].subject == "Your package shipped"
    assert results[0].from_addresses[0].email == "ship@amazon.com"
    assert results[1].id == "F001:M002"
    await client.close()


@pytest.mark.asyncio
async def test_search_emails_builds_search_key():
    """Test that search parameters are passed as Zoho searchKey."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(SEARCH_RESPONSE),
            ]

            await client.search_emails(
                text="hello",
                from_addr="bob@example.com",
                subject="meeting",
            )

            # Check the GET params - Zoho uses entire:, sender:, subject: with :: separator
            call_args = mock_get.call_args_list[-1]
            params = call_args.kwargs.get("params", {})
            search_key = params.get("searchKey", "")
            assert "entire:hello" in search_key
            assert "sender:bob@example.com" in search_key
            assert "subject:meeting" in search_key
            assert "::" in search_key  # Conditions joined with ::

    await client.close()


@pytest.mark.asyncio
async def test_read_emails_returns_details():
    """Test that read_emails parses full email content."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    # Use folderId:messageId format (as returned by search)
    email_id = "F001:M001"

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),  # _ensure_account
                _make_response(EMAIL_CONTENT_RESPONSE),  # content endpoint
            ]

            results = await client.read_emails([email_id])

    assert len(results) == 1
    assert results[0].id == email_id
    assert results[0].subject == "Your package shipped"
    assert "Your order #123 has shipped!" in results[0].text_body
    await client.close()


@pytest.mark.asyncio
async def test_read_emails_strips_html():
    """Test that read_emails strips HTML tags from content."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    email_id = "F001:M001"

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),  # _ensure_account
                _make_response(EMAIL_CONTENT_HTML_RESPONSE),  # content endpoint
            ]

            results = await client.read_emails([email_id])

    assert len(results) == 1
    assert "Hello" in results[0].text_body
    assert "World" in results[0].text_body
    assert "<" not in results[0].text_body
    await client.close()


@pytest.mark.asyncio
async def test_read_emails_empty_ids():
    """Test that read_emails returns empty list for empty ID list."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    results = await client.read_emails([])
    assert results == []
    await client.close()


def teststrip_html():
    """Test HTML tag stripping utility."""
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_html("no tags here") == "no tags here"
    assert strip_html("<div><span>nested</span></div>") == "nested"
    assert strip_html("") == ""


# Folder API response mock
FOLDERS_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": [
        {
            "folderId": "F001",
            "folderName": "Inbox",
            "folderType": "Inbox",
            "path": "/Inbox",
            "isArchived": 0,
        },
        {
            "folderId": "F002",
            "folderName": "Sent",
            "folderType": "Sent",
            "path": "/Sent",
            "isArchived": 0,
        },
        {
            "folderId": "F003",
            "folderName": "Drafts",
            "folderType": "Drafts",
            "path": "/Drafts",
            "isArchived": 0,
        },
    ],
}

LIST_EMAILS_RESPONSE = {
    "status": {"code": 200, "description": "success"},
    "data": [
        {
            "messageId": "M101",
            "folderId": "F001",
            "subject": "Welcome email",
            "fromAddress": "welcome@example.com",
            "sender": "Welcome Team",
            "receivedTime": 1707573000000,
            "summary": "Welcome to our service...",
        },
    ],
}


@pytest.mark.asyncio
async def test_get_folders():
    """Test that get_folders fetches and caches folder list."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
            ]

            folders = await client.get_folders()

    assert len(folders) == 3
    assert folders[0].folder_name == "Inbox"
    assert folders[0].folder_type == "Inbox"
    assert folders[1].folder_name == "Sent"
    assert folders[2].folder_name == "Drafts"
    await client.close()


@pytest.mark.asyncio
async def test_get_folder_by_name():
    """Test finding a folder by name."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
            ]

            folder = await client.get_folder_by_name("sent")  # case-insensitive

    assert folder is not None
    assert folder.folder_name == "Sent"
    assert folder.folder_id == "F002"
    await client.close()


@pytest.mark.asyncio
async def test_list_emails_from_folder():
    """Test listing emails from a specific folder."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
                _make_response(LIST_EMAILS_RESPONSE),
            ]

            results = await client.list_emails(folder_name="Inbox")

    assert len(results) == 1
    assert results[0].subject == "Welcome email"
    assert results[0].id == "F001:M101"
    await client.close()


@pytest.mark.asyncio
async def test_list_emails_uses_constructor_list_limit():
    """The list_limit kwarg threads from /config → constructor → API params."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=42,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
                _make_response(LIST_EMAILS_RESPONSE),
            ]

            await client.list_emails(folder_name="Inbox")

            list_call = mock_get.call_args_list[-1]
            assert list_call.kwargs["params"]["limit"] == 42

    await client.close()


@pytest.mark.asyncio
async def test_search_emails_uses_constructor_search_limit():
    """The search_limit kwarg threads from /config → constructor → API params."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=27,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(LIST_EMAILS_RESPONSE),
            ]

            await client.search_emails(text="welcome")

            search_call = mock_get.call_args_list[-1]
            assert search_call.kwargs["params"]["limit"] == 27

    await client.close()


@pytest.mark.asyncio
async def test_create_folder():
    """Test creating a new email folder."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    create_response = {
        "status": {"code": 200, "description": "success"},
        "data": {
            "folderId": "F003",
            "folderName": "Invoices",
            "folderType": "Custom",
            "path": "/Invoices",
        },
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            _make_response(TOKEN_RESPONSE),
            _make_response(create_response),
        ]

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
            ]

            folder = await client.create_folder("Invoices")

    assert folder is not None
    assert folder.folder_id == "F003"
    assert folder.folder_name == "Invoices"
    await client.close()


@pytest.mark.asyncio
async def test_create_nested_folder_creates_parent_then_child():
    """Test creating a nested folder path with parents created on demand."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    responses = [
        {"folderId": "F100", "folderName": "Clients", "folderType": "Custom", "path": "/Clients"},
        {
            "folderId": "F101",
            "folderName": "Acme",
            "folderType": "Custom",
            "path": "/Clients/Acme",
        },
    ]

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            _make_response(TOKEN_RESPONSE),
            *(_make_response({"status": {"code": 200}, "data": resp}) for resp in responses),
        ]

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
                _make_response(FOLDERS_RESPONSE),
            ]

            folder = await client.create_nested_folder("Clients/Acme")

    assert folder is not None
    assert folder.folder_id == "F101"
    assert folder.folder_name == "Acme"
    await client.close()


@pytest.mark.asyncio
async def test_move_messages_strips_composite_ids():
    """Test moving messages strips composite folder:message IDs."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    move_response = {
        "status": {"code": 200, "description": "success"},
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = _make_response(move_response)

            with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_response(ACCOUNTS_RESPONSE)

                success = await client.move_messages(
                    ["F001:M001", "F001:M002"],
                    "F003",
                )

    assert success is True
    call_args = mock_put.call_args
    assert call_args.kwargs["json"]["messageId"] == ["M001", "M002"]
    assert call_args.kwargs["json"]["destfolderId"] == "F003"
    await client.close()


@pytest.mark.asyncio
async def test_get_labels():
    """Test fetching all labels."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    labels_response = {
        "status": {"code": 200, "description": "success"},
        "data": [
            {"labelId": "L001", "displayName": "Work", "color": "#4285f4"},
            {"labelId": "L002", "displayName": "Personal", "color": "#34a853"},
        ],
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(labels_response),
            ]

            labels = await client.get_labels()

    assert len(labels) == 2
    assert labels[0]["displayName"] == "Work"
    await client.close()


@pytest.mark.asyncio
async def test_get_label_by_name_is_case_insensitive():
    """Test fetching a label by name is case-insensitive."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    labels_response = {
        "status": {"code": 200, "description": "success"},
        "data": [
            {"labelId": "L001", "displayName": "Work", "color": "#4285f4"},
        ],
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
                _make_response(labels_response),
            ]

            label = await client.get_label_by_name("work")

    assert label is not None
    assert label["labelId"] == "L001"
    await client.close()


@pytest.mark.asyncio
async def test_create_label():
    """Test creating a new label."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    create_response = {
        "status": {"code": 200, "description": "success"},
        "data": {"labelId": "L003", "displayName": "Urgent", "color": "#4285f4"},
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            _make_response(TOKEN_RESPONSE),
            _make_response(create_response),
        ]

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                _make_response(ACCOUNTS_RESPONSE),
            ]

            label = await client.create_label("Urgent")

    assert label is not None
    assert label["labelId"] == "L003"
    assert label["displayName"] == "Urgent"
    await client.close()


@pytest.mark.asyncio
async def test_apply_label():
    """Test applying a label to messages."""
    client = ZohoClient(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=_ZOHO_TIMEOUT,
        max_body_length=_EMAIL_MAX_LENGTH,
        search_limit=_EMAIL_SEARCH_LIMIT,
        list_limit=_EMAIL_LIST_LIMIT,
    )

    apply_response = {
        "status": {"code": 200, "description": "success"},
    }

    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _make_response(TOKEN_RESPONSE)

        with patch.object(client._http, "put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = _make_response(apply_response)

            with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_response(ACCOUNTS_RESPONSE)

                success = await client.apply_label(
                    ["F001:M001", "F001:M002"],
                    "L001",
                )

    assert success is True
    call_args = mock_put.call_args
    assert call_args.kwargs["json"]["messageId"] == ["M001", "M002"]
    assert call_args.kwargs["json"]["labelId"] == ["L001"]
    await client.close()
