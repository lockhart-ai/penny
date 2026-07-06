"""Integration tests for /bug command."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from penny.channels.base import IncomingMessage
from penny.commands.bug import BugCommand
from penny.commands.models import CommandContext
from penny.config import Config
from penny.database import Database
from penny.database.migrate import migrate
from penny.tests.conftest import TEST_SENDER


@pytest.fixture
def bug_db(tmp_path):
    """Create a test database with tables and migrations."""
    db_path = str(tmp_path / "bug_test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


@pytest.fixture
def mock_github_api():
    """Create a mock GitHubAPI that returns a fake issue URL."""
    api = MagicMock()
    api.create_issue.return_value = "https://github.com/lockhart-ai/penny/issues/999"
    return api


@pytest.fixture
def bug_context(bug_db):
    """Create a CommandContext for bug command tests."""
    config = MagicMock(spec=Config)
    ollama = MagicMock()
    return CommandContext(
        db=bug_db,
        config=config,
        model_client=ollama,
        embedding_model_client=ollama,
        user=TEST_SENDER,
        channel_type="signal",
        start_time=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_bug_files_issue(mock_github_api, bug_context):
    """Test /bug creates a GitHub issue and returns the URL."""
    cmd = BugCommand(mock_github_api)
    result = await cmd.execute("the app crashes on startup", bug_context)

    assert "Bug filed!" in result.text
    assert "issues/999" in result.text
    mock_github_api.create_issue.assert_called_once()
    call_kwargs = mock_github_api.create_issue.call_args
    assert call_kwargs.kwargs["labels"] == ["bug"]


@pytest.mark.asyncio
async def test_bug_empty_description(mock_github_api, bug_context):
    """Test /bug with no description shows usage."""
    cmd = BugCommand(mock_github_api)
    result = await cmd.execute("", bug_context)

    assert "Usage: /bug" in result.text
    mock_github_api.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_bug_body_does_not_contain_user_identifier(mock_github_api, bug_context):
    """Test that the issue body does not leak the user's phone number or ID."""
    cmd = BugCommand(mock_github_api)
    await cmd.execute("something is broken", bug_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]

    # Must not contain the sender's phone number
    assert TEST_SENDER not in body
    # Should still have the channel type metadata
    assert "signal" in body


@pytest.mark.asyncio
async def test_bug_title_truncation(mock_github_api, bug_context):
    """Test that long descriptions get truncated to ~60 char titles at word boundary."""
    long_desc = "the application crashes when I try to send a message that is longer than expected"
    cmd = BugCommand(mock_github_api)
    await cmd.execute(long_desc, bug_context)

    call_kwargs = mock_github_api.create_issue.call_args
    title = call_kwargs.kwargs["title"]
    assert title.endswith("...")
    assert len(title) <= 64  # 60 + "..."


@pytest.mark.asyncio
async def test_bug_short_title_not_truncated(mock_github_api, bug_context):
    """Test that short descriptions are used as-is for the title."""
    short_desc = "button is broken"
    cmd = BugCommand(mock_github_api)
    await cmd.execute(short_desc, bug_context)

    call_kwargs = mock_github_api.create_issue.call_args
    title = call_kwargs.kwargs["title"]
    assert title == short_desc


@pytest.mark.asyncio
async def test_bug_api_failure(mock_github_api, bug_context):
    """Test /bug handles GitHub API errors gracefully."""
    mock_github_api.create_issue.side_effect = RuntimeError("API rate limited")
    cmd = BugCommand(mock_github_api)
    result = await cmd.execute("something broke", bug_context)

    assert "Failed to create issue" in result.text
    assert "API rate limited" in result.text


@pytest.mark.asyncio
async def test_bug_with_quoted_message(mock_github_api, bug_context, bug_db):
    """Test /bug with a quote-reply includes quoted message timestamp."""
    # Store an outgoing message that can be found by quote lookup
    bug_db.messages.log_message(
        sender="penny",
        content="Here is a response that will be quoted",
        direction="outgoing",
    )

    # Set up context with a quoted message
    bug_context.message = IncomingMessage(
        sender=TEST_SENDER,
        content="/bug this response was wrong",
        quoted_text="Here is a response that will be quoted",
    )

    cmd = BugCommand(mock_github_api)
    await cmd.execute("this response was wrong", bug_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]
    assert "Refers to message sent at" in body


@pytest.mark.asyncio
async def test_bug_discord_channel_type(mock_github_api, bug_context):
    """Test /bug from Discord shows discord in the footer."""
    bug_context.channel_type = "discord"
    bug_context.user = "123456789"  # Discord user ID

    cmd = BugCommand(mock_github_api)
    await cmd.execute("discord bug", bug_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]
    assert "discord" in body
    # Must not contain Discord user ID
    assert "123456789" not in body
