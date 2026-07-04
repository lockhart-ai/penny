"""Integration tests for /zoho command."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from penny.commands.models import CommandContext
from penny.commands.zoho import ZohoCommand
from penny.config import Config
from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER

FAKE_CLIENT_ID = "1000.TESTCLIENTID"
FAKE_CLIENT_SECRET = "testsecret123"
FAKE_REFRESH_TOKEN = "1000.testrefreshtoken"

SAMPLE_SUMMARIES = [
    EmailSummary(
        id="F001:M001",
        subject="Your package has shipped!",
        from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
        received_at="2026-02-10T14:30:00Z",
        preview="Your order #123-456 has been shipped and is on its way...",
    ),
    EmailSummary(
        id="F001:M002",
        subject="Delivery scheduled for tomorrow",
        from_addresses=[EmailAddress(name="UPS", email="noreply@ups.com")],
        received_at="2026-02-10T10:00:00Z",
        preview="Your package is scheduled for delivery on Feb 11...",
    ),
]

SAMPLE_DETAIL = EmailDetail(
    id="F001:M001",
    subject="Your package has shipped!",
    from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@zohomail.com")],
    received_at="2026-02-10T14:30:00Z",
    text_body=(
        "Your order #123-456 has been shipped!\n\n"
        "Tracking number: 1Z999AA10123456784\n"
        "Estimated delivery: February 12, 2026"
    ),
)


@pytest.fixture
def mock_zoho_client():
    """Create a mock ZohoClient."""
    client = AsyncMock()
    client.search_emails.return_value = SAMPLE_SUMMARIES
    client.read_emails.return_value = [SAMPLE_DETAIL]
    client.close.return_value = None
    return client


@pytest.fixture
def zoho_context():
    """Create a CommandContext for zoho command tests."""
    config = MagicMock(spec=Config)
    config.email_max_steps = 5
    config.tool_timeout = 60.0
    runtime = MagicMock()
    runtime.JMAP_REQUEST_TIMEOUT = 30.0
    runtime.EMAIL_BODY_MAX_LENGTH = 4000
    runtime.EMAIL_SEARCH_LIMIT = 10
    runtime.EMAIL_LIST_LIMIT = 10
    config.runtime = runtime
    db = MagicMock()
    # current_datetime_line() reads the primary user's profile timezone for the dated
    # anchor injected into the email-summarize prompt.
    db.users.get_primary_sender.return_value = TEST_SENDER
    db.users.get_info.return_value = MagicMock(timezone="America/Los_Angeles")
    return CommandContext(
        db=db,
        config=config,
        model_client=MagicMock(),
        embedding_model_client=MagicMock(),
        user=TEST_SENDER,
        channel_type="signal",
        start_time=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_zoho_empty_prompt(zoho_context):
    """Test /zoho with no args returns usage text."""
    cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
    result = await cmd.execute("", zoho_context)

    assert result.text == PennyResponse.ZOHO_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_zoho_whitespace_only_prompt(zoho_context):
    """Test /zoho with whitespace-only args returns usage text."""
    cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
    result = await cmd.execute("   ", zoho_context)

    assert result.text == PennyResponse.ZOHO_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_zoho_search_and_answer(mock_zoho_client, zoho_context):
    """Test /zoho runs the agent loop and returns an answer."""
    mock_response = MagicMock()
    mock_response.answer = "You have 2 packages coming! One from Amazon arriving Feb 12."

    with (
        patch("penny.commands.zoho.ZohoClient", return_value=mock_zoho_client),
        patch("penny.commands.zoho.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_response
        mock_agent_cls.return_value = mock_agent_instance

        cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
        result = await cmd.execute("what packages am I expecting", zoho_context)

    assert "packages" in result.text.lower()
    mock_agent_instance.run.assert_called_once_with(
        "what packages am I expecting", max_steps=zoho_context.config.email_max_steps
    )
    mock_agent_instance.close.assert_called_once()
    mock_zoho_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_zoho_agent_construction(mock_zoho_client, zoho_context):
    """The zoho agent is built with allow_repeat_tools=True and the verbatim
    house-style ZOHO_SYSTEM_PROMPT (numbered call steps for all five tools,
    canonical notation, explicit plain-text terminal answer)."""
    with (
        patch("penny.commands.zoho.ZohoClient", return_value=mock_zoho_client),
        patch("penny.commands.zoho.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
        await cmd.execute("check my email", zoho_context)

    call_kwargs = mock_agent_cls.call_args
    assert call_kwargs.kwargs["allow_repeat_tools"] is True
    assert (
        call_kwargs.kwargs["system_prompt"]
        == """\
You are searching the user's Zoho email to answer their question. Work in order:

1. search_emails(text=<keywords>) — find candidate emails across the mailbox; \
narrow with from_addr=<sender>, subject=<subject text>, after=<ISO date>, or \
before=<ISO date>. To browse a whole folder instead, \
list_emails(folder=<folder name>); call list_folders() first if you are unsure \
which folders exist. Each result carries an id.
2. read_emails(email_ids=[<id>, <id>]) — read the full bodies of the promising \
results, passing ALL relevant ids in ONE call.
3. If the answer is still incomplete, search or list again and \
read_emails(email_ids=[<id>]) the new hits.
4. If the user asked you to reply, draft_email(to=[<address>], subject=<subject>, \
body=<text>) — this saves a draft to their Drafts folder for review; it NEVER \
sends.
5. Answer the user in plain text with the concrete details you found — specific \
dates, names, and amounts — and name the email (sender + subject) each fact \
came from.

ALWAYS ground every claim in an email you actually read — NEVER guess at a date, \
sender, or amount you did not see. Use **bold** for the load-bearing terms \
(dates, names, amounts) and bullet points when summarizing more than one email."""
    )


@pytest.mark.asyncio
async def test_zoho_agent_cleanup_on_error(mock_zoho_client, zoho_context):
    """Test that agent and Zoho client are cleaned up even on error."""
    with (
        patch("penny.commands.zoho.ZohoClient", return_value=mock_zoho_client),
        patch("penny.commands.zoho.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.side_effect = RuntimeError("Ollama down")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
        result = await cmd.execute("check my email", zoho_context)

    assert "Failed to search Zoho email" in result.text
    mock_agent_instance.close.assert_called_once()
    mock_zoho_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_zoho_client_created_with_credentials(zoho_context):
    """Test that ZohoClient is created with the configured credentials."""
    with (
        patch("penny.commands.zoho.ZohoClient") as mock_zoho_cls,
        patch("penny.commands.zoho.Agent") as mock_agent_cls,
    ):
        mock_client = AsyncMock()
        mock_zoho_cls.return_value = mock_client

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = ZohoCommand(FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_REFRESH_TOKEN)
        await cmd.execute("anything", zoho_context)

    mock_zoho_cls.assert_called_once_with(
        FAKE_CLIENT_ID,
        FAKE_CLIENT_SECRET,
        FAKE_REFRESH_TOKEN,
        timeout=30.0,
        max_body_length=4000,
        search_limit=10,
        list_limit=10,
    )
