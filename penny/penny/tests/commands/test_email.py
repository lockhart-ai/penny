"""Integration tests for /email command."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from penny.commands.email import EmailCommand
from penny.commands.models import CommandContext
from penny.config import Config
from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER
from penny.tools.draft_email import DraftEmailTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

FAKE_TOKEN = "fmu1-test-token"

SAMPLE_SUMMARIES = [
    EmailSummary(
        id="M001",
        subject="Your package has shipped!",
        from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
        received_at="2026-02-10T14:30:00Z",
        preview="Your order #123-456 has been shipped and is on its way...",
    ),
    EmailSummary(
        id="M002",
        subject="Delivery scheduled for tomorrow",
        from_addresses=[EmailAddress(name="UPS", email="noreply@ups.com")],
        received_at="2026-02-10T10:00:00Z",
        preview="Your package is scheduled for delivery on Feb 11...",
    ),
]

SAMPLE_DETAIL = EmailDetail(
    id="M001",
    subject="Your package has shipped!",
    from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@fastmail.com")],
    received_at="2026-02-10T14:30:00Z",
    text_body=(
        "Your order #123-456 has been shipped!\n\n"
        "Tracking number: 1Z999AA10123456784\n"
        "Estimated delivery: February 12, 2026"
    ),
)


@pytest.fixture
def mock_jmap_client():
    """Create a mock JmapClient."""
    client = AsyncMock()
    client.search_emails.return_value = SAMPLE_SUMMARIES
    client.read_emails.return_value = [SAMPLE_DETAIL]
    client.close.return_value = None
    return client


@pytest.fixture
def email_context():
    """Create a CommandContext for email command tests."""
    config = MagicMock(spec=Config)
    config.email_max_steps = 5
    config.tool_timeout = 60.0
    runtime = MagicMock()
    runtime.JMAP_REQUEST_TIMEOUT = 30.0
    runtime.EMAIL_BODY_MAX_LENGTH = 4000
    runtime.EMAIL_SEARCH_LIMIT = 10
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
async def test_email_empty_prompt(email_context):
    """Test /email with no args returns usage text."""
    cmd = EmailCommand(FAKE_TOKEN)
    result = await cmd.execute("", email_context)

    assert result.text == PennyResponse.EMAIL_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_email_whitespace_only_prompt(email_context):
    """Test /email with whitespace-only args returns usage text."""
    cmd = EmailCommand(FAKE_TOKEN)
    result = await cmd.execute("   ", email_context)

    assert result.text == PennyResponse.EMAIL_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_email_search_and_answer(mock_jmap_client, email_context):
    """Test /email runs the agent loop and returns an answer."""
    mock_response = MagicMock()
    mock_response.answer = "You have 2 packages coming! One from Amazon arriving Feb 12."

    with (
        patch("penny.commands.email.JmapClient", return_value=mock_jmap_client),
        patch("penny.commands.email.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_response
        mock_agent_cls.return_value = mock_agent_instance

        cmd = EmailCommand(FAKE_TOKEN)
        result = await cmd.execute("what packages am I expecting", email_context)

    assert "packages" in result.text.lower()
    mock_agent_instance.run.assert_called_once_with(
        "what packages am I expecting", max_steps=email_context.config.email_max_steps
    )
    mock_agent_instance.close.assert_called_once()
    mock_jmap_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_email_agent_created_with_repeat_tools(mock_jmap_client, email_context):
    """Test that the email agent is created with allow_repeat_tools=True."""
    with (
        patch("penny.commands.email.JmapClient", return_value=mock_jmap_client),
        patch("penny.commands.email.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = EmailCommand(FAKE_TOKEN)
        await cmd.execute("check my email", email_context)

    # Verify allow_repeat_tools was passed
    call_kwargs = mock_agent_cls.call_args
    assert call_kwargs.kwargs["allow_repeat_tools"] is True


@pytest.mark.asyncio
async def test_email_agent_cleanup_on_error(mock_jmap_client, email_context):
    """Test that agent and JMAP client are cleaned up even on error."""
    with (
        patch("penny.commands.email.JmapClient", return_value=mock_jmap_client),
        patch("penny.commands.email.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.side_effect = RuntimeError("Ollama down")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = EmailCommand(FAKE_TOKEN)
        result = await cmd.execute("check my email", email_context)

    assert "Failed to search email" in result.text
    mock_agent_instance.close.assert_called_once()
    mock_jmap_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_email_jmap_client_created_with_token(email_context):
    """Test that JmapClient is created with the configured token."""
    with (
        patch("penny.commands.email.JmapClient") as mock_jmap_cls,
        patch("penny.commands.email.Agent") as mock_agent_cls,
    ):
        mock_client = AsyncMock()
        mock_jmap_cls.return_value = mock_client

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = EmailCommand(FAKE_TOKEN)
        await cmd.execute("anything", email_context)

    mock_jmap_cls.assert_called_once_with(
        FAKE_TOKEN,
        timeout=30.0,
        max_body_length=4000,
        search_limit=10,
    )


@pytest.mark.asyncio
async def test_read_emails_tool_summarizes_content():
    """Test that ReadEmailsTool runs fetched emails through Ollama summarization."""
    mock_jmap = AsyncMock()
    mock_jmap.read_emails.return_value = [SAMPLE_DETAIL]

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = MagicMock(
        content="Amazon shipped order #123-456, arriving Feb 12."
    )

    tool = ReadEmailsTool(
        mock_jmap,
        mock_llm,
        "what packages am I expecting",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.execute(email_ids=["M001"])

    assert result.message == "Amazon shipped order #123-456, arriving Feb 12."
    mock_jmap.read_emails.assert_called_once_with(["M001"])
    mock_llm.chat.assert_called_once()
    prompt = mock_llm.chat.call_args[0][0][0]["content"]
    assert "what packages am I expecting" in prompt
    # The summarize prompt is grounded in the current date so the model can resolve
    # relative dates in email bodies ("arriving next Tuesday").
    assert "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT" in prompt


@pytest.mark.asyncio
async def test_read_emails_tool_falls_back_on_empty_summary():
    """Test that ReadEmailsTool returns raw content if Ollama returns empty."""
    mock_jmap = AsyncMock()
    mock_jmap.read_emails.return_value = [SAMPLE_DETAIL]

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = MagicMock(content="")

    tool = ReadEmailsTool(
        mock_jmap,
        mock_llm,
        "test query",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.execute(email_ids=["M001"])

    assert "Your order #123-456 has been shipped!" in result.message


@pytest.mark.asyncio
async def test_read_emails_tool_no_ids():
    """An empty ``email_ids`` is rejected by the ``args_model`` at the ``run`` gate
    — ``execute`` is never reached, so the client/LLM are never called, and the
    actionable error names the field and the fix."""
    mock_jmap = AsyncMock()
    mock_llm = AsyncMock()

    tool = ReadEmailsTool(
        mock_jmap,
        mock_llm,
        "test query",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.run(email_ids=[])

    assert result.success is False
    assert "email_ids" in result.message
    mock_jmap.read_emails.assert_not_called()
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_search_emails_tool_rejects_all_empty_search():
    """An all-empty search (no criterion) is rejected by ``SearchEmailsArgs`` at the
    ``run`` gate before ``execute`` runs — the client is never queried, and the
    actionable error names the missing criteria."""
    mock_client = AsyncMock()

    tool = SearchEmailsTool(mock_client)
    result = await tool.run()

    assert result.success is False
    assert "criterion" in result.message
    mock_client.search_emails.assert_not_called()


@pytest.mark.asyncio
async def test_search_emails_tool_runs_with_one_criterion(mock_jmap_client):
    """A single criterion (text) is enough to pass validation and reach ``execute``."""
    tool = SearchEmailsTool(mock_jmap_client)
    result = await tool.run(text="package")

    assert result.success is True
    assert "email(s)" in result.message
    mock_jmap_client.search_emails.assert_called_once()


@pytest.mark.asyncio
async def test_draft_email_tool_rejects_structural_gaps():
    """``DraftEmailArgs`` rejects an empty recipient list and blank subject/body at
    the ``run`` gate — ``execute`` never runs, so the client is never called."""
    mock_client = AsyncMock()
    tool = DraftEmailTool(mock_client)

    empty_to = await tool.run(to=[], subject="Hi", body="Hello there")
    assert empty_to.success is False
    assert "to" in empty_to.message

    blank_subject = await tool.run(to=["a@b.com"], subject="   ", body="Hello there")
    assert blank_subject.success is False
    assert "subject" in blank_subject.message

    blank_body = await tool.run(to=["a@b.com"], subject="Hi", body="")
    assert blank_body.success is False
    assert "body" in blank_body.message

    mock_client.draft_response.assert_not_called()


@pytest.mark.asyncio
async def test_draft_email_tool_saves_well_formed_draft():
    """A draft with a recipient and non-blank subject/body passes validation and is
    saved through the client."""
    mock_client = AsyncMock()
    mock_client.draft_response.return_value = "draft-123"
    tool = DraftEmailTool(mock_client)

    result = await tool.run(to=["a@b.com"], subject="Hi", body="Hello there")

    assert result.success is True
    assert result.mutated is True
    mock_client.draft_response.assert_called_once()
