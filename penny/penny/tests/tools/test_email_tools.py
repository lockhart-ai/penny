"""Unit tests for the email tools (search / read / draft).

These tools power the chat agent's email surface (retired /email + /zoho, epic
#1445).  They validate their args at the ``run`` gate and mock the mailbox
client at the boundary — the backend clients themselves are covered in
``tests/jmap`` and ``tests/zoho``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import NO_EMAILS_FOUND as LIST_NO_EMAILS_FOUND
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.models import ToolResult
from penny.tools.read_emails import NO_EMAILS_TO_READ, ReadEmailsTool
from penny.tools.search_emails import NO_EMAILS_FOUND, SearchEmailsTool

SAMPLE_SUMMARIES = [
    EmailSummary(
        id="M001",
        subject="Your package has shipped!",
        from_addresses=[EmailAddress(name="Shipping Co", email="ship-confirm@example.com")],
        received_at="2026-02-10T14:30:00Z",
        preview="Your order #123-456 has been shipped and is on its way...",
    ),
    EmailSummary(
        id="M002",
        subject="Delivery scheduled for tomorrow",
        from_addresses=[EmailAddress(name="Courier Co", email="noreply@example.org")],
        received_at="2026-02-10T10:00:00Z",
        preview="Your package is scheduled for delivery on Feb 11...",
    ),
]

SAMPLE_DETAIL = EmailDetail(
    id="M001",
    subject="Your package has shipped!",
    from_addresses=[EmailAddress(name="Shipping Co", email="ship-confirm@example.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@example.com")],
    received_at="2026-02-10T14:30:00Z",
    text_body=(
        "Your order #123-456 has been shipped!\n\n"
        "Tracking number: 1Z999AA10123456784\n"
        "Estimated delivery: February 12, 2026"
    ),
)


@pytest.fixture
def mock_email_client():
    """A mailbox client stub satisfying the EmailClient protocol."""
    client = AsyncMock()
    client.search_emails.return_value = SAMPLE_SUMMARIES
    client.read_emails.return_value = [SAMPLE_DETAIL]
    client.close.return_value = None
    return client


@pytest.mark.asyncio
async def test_read_emails_tool_summarizes_content():
    """ReadEmailsTool runs fetched emails through model summarization, grounded
    in the user's question and the current date."""
    mock_client = AsyncMock()
    mock_client.read_emails.return_value = [SAMPLE_DETAIL]

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = MagicMock(
        content="Shipping Co shipped order #123-456, arriving Feb 12."
    )

    tool = ReadEmailsTool(
        mock_client,
        mock_llm,
        "what packages am I expecting",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.execute(email_ids=["M001"])

    assert result.message == "Shipping Co shipped order #123-456, arriving Feb 12."
    mock_client.read_emails.assert_called_once_with(["M001"])
    mock_llm.chat.assert_called_once()
    prompt = mock_llm.chat.call_args[0][0][0]["content"]
    assert "what packages am I expecting" in prompt
    # The summarize prompt is grounded in the current date so the model can resolve
    # relative dates in email bodies ("arriving next Tuesday").
    assert "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT" in prompt


@pytest.mark.asyncio
async def test_read_emails_tool_falls_back_on_empty_summary():
    """ReadEmailsTool returns raw content if the model returns empty."""
    mock_client = AsyncMock()
    mock_client.read_emails.return_value = [SAMPLE_DETAIL]

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = MagicMock(content="")

    tool = ReadEmailsTool(
        mock_client,
        mock_llm,
        "test query",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.execute(email_ids=["M001"])

    assert "Your order #123-456 has been shipped!" in result.message


@pytest.mark.asyncio
async def test_read_emails_tool_no_ids():
    """An empty ``email_ids`` is rejected by the ``args_model`` at the ``run`` gate
    — ``execute`` is never reached, so the client/model are never called, and the
    actionable error names the field and the fix."""
    mock_client = AsyncMock()
    mock_llm = AsyncMock()

    tool = ReadEmailsTool(
        mock_client,
        mock_llm,
        "test query",
        "Current date and time: Friday, May 09, 2025 at 03:00 PM PDT",
    )
    result = await tool.run(email_ids=[])

    assert result.success is False
    assert "email_ids" in result.message
    mock_client.read_emails.assert_not_called()
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
async def test_search_emails_tool_runs_with_one_criterion(mock_email_client):
    """A single criterion (text) is enough to pass validation and reach ``execute``."""
    tool = SearchEmailsTool(mock_email_client)
    result = await tool.run(text="package")

    assert result.success is True
    assert "email(s)" in result.message
    mock_email_client.search_emails.assert_called_once()


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


# ── Result narration (epic #1478) ─────────────────────────────────────────────
# Each tool's ``to_result_narration`` leads its result with a first-person recap
# of the action; the seam adds the ``(<tool> result)`` tag and the body, so these
# assert only the narration STRING (registry-dispatched by ``Tool.format_result``).


def test_search_emails_narration_branches():
    ok = SearchEmailsTool.to_result_narration(
        {"from_addr": "Priya", "text": "solar"}, ToolResult(message="Found 1 email(s):")
    )
    assert ok == 'You searched your email for "solar · from Priya":'

    empty = SearchEmailsTool.to_result_narration(
        {"text": "solar"}, ToolResult(message=NO_EMAILS_FOUND)
    )
    assert empty == "You searched your email but found nothing matching:"

    failed = SearchEmailsTool.to_result_narration(
        {"text": "solar"}, ToolResult(message="boom", success=False)
    )
    assert failed == "You tried to search your email but it didn't work:"


def test_read_emails_narration_branches():
    one = ReadEmailsTool.to_result_narration(
        {"email_ids": ["M001"]}, ToolResult(message="A summary")
    )
    assert one == "You read 1 email:"

    several = ReadEmailsTool.to_result_narration(
        {"email_ids": ["M001", "M002"]}, ToolResult(message="A summary")
    )
    assert several == "You read 2 emails:"

    nothing = ReadEmailsTool.to_result_narration(
        {"email_ids": ["M404"]}, ToolResult(message=NO_EMAILS_TO_READ)
    )
    assert nothing == "You tried to open some email but there was nothing to read:"

    failed = ReadEmailsTool.to_result_narration(
        {"email_ids": ["M001"]}, ToolResult(message="boom", success=False)
    )
    assert failed == "You tried to read your email but it didn't work:"


def test_list_emails_narration_branches():
    ok = ListEmailsTool.to_result_narration(
        {"folder": "Sent"}, ToolResult(message="Found 2 email(s) in Sent:")
    )
    assert ok == "You listed the emails in Sent:"

    default_folder = ListEmailsTool.to_result_narration({}, ToolResult(message="Found 1 email(s):"))
    assert default_folder == "You listed the emails in Inbox:"

    empty = ListEmailsTool.to_result_narration(
        {"folder": "Spam"}, ToolResult(message=LIST_NO_EMAILS_FOUND)
    )
    assert empty == "You looked in Spam but found no emails:"

    failed = ListEmailsTool.to_result_narration(
        {"folder": "Spam"}, ToolResult(message="boom", success=False)
    )
    assert failed == "You tried to list the emails in Spam but it didn't work:"


def test_list_folders_narration_branches():
    ok = ListFoldersTool.to_result_narration({}, ToolResult(message="Found 3 folder(s):"))
    assert ok == "You looked at your email folders:"

    failed = ListFoldersTool.to_result_narration({}, ToolResult(message="boom", success=False))
    assert failed == "You tried to list your email folders but it didn't work:"


def test_draft_email_narration_branches():
    one = DraftEmailTool.to_result_narration(
        {"to": ["sam@example.com"]}, ToolResult(message="Draft saved", mutated=True)
    )
    assert one == "You drafted an email to sam@example.com:"

    several = DraftEmailTool.to_result_narration(
        {"to": ["a@example.com", "b@example.com"]}, ToolResult(message="Draft saved", mutated=True)
    )
    assert several == "You drafted an email to 2 recipients:"

    failed = DraftEmailTool.to_result_narration(
        {"to": ["sam@example.com"]}, ToolResult(message="boom", success=False)
    )
    assert failed == "You tried to draft an email to sam@example.com but it didn't work:"
