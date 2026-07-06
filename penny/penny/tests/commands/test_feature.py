"""Integration tests for /feature command."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from penny.channels.base import IncomingMessage
from penny.commands.feature import FeatureCommand
from penny.commands.models import CommandContext
from penny.config import Config
from penny.database import Database
from penny.database.migrate import migrate
from penny.tests.conftest import TEST_SENDER


@pytest.fixture
def feature_db(tmp_path):
    """Create a test database with tables and migrations."""
    db_path = str(tmp_path / "feature_test.db")
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
def feature_context(feature_db):
    """Create a CommandContext for feature command tests."""
    config = MagicMock(spec=Config)
    ollama = MagicMock()
    return CommandContext(
        db=feature_db,
        config=config,
        model_client=ollama,
        embedding_model_client=ollama,
        user=TEST_SENDER,
        channel_type="signal",
        start_time=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_feature_files_issue(mock_github_api, feature_context):
    """Test /feature creates a GitHub issue with requirements label and returns the URL."""
    cmd = FeatureCommand(mock_github_api)
    result = await cmd.execute("add dark mode support", feature_context)

    assert "Feature request filed!" in result.text
    assert "issues/999" in result.text
    mock_github_api.create_issue.assert_called_once()
    call_kwargs = mock_github_api.create_issue.call_args
    assert call_kwargs.kwargs["labels"] == ["requirements"]


@pytest.mark.asyncio
async def test_feature_empty_description(mock_github_api, feature_context):
    """Test /feature with no description shows usage."""
    cmd = FeatureCommand(mock_github_api)
    result = await cmd.execute("", feature_context)

    assert "Usage: /feature" in result.text
    mock_github_api.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_feature_body_does_not_contain_user_identifier(mock_github_api, feature_context):
    """Test that the issue body does not leak the user's phone number or ID."""
    cmd = FeatureCommand(mock_github_api)
    await cmd.execute("add a new dashboard", feature_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]

    # Must not contain the sender's phone number
    assert TEST_SENDER not in body
    # Should still have the channel type metadata
    assert "signal" in body


@pytest.mark.asyncio
async def test_feature_title_truncation(mock_github_api, feature_context):
    """Test that long descriptions get truncated to ~60 char titles at word boundary."""
    long_desc = (
        "add the ability to customize the dashboard layout with drag and drop widgets and themes"
    )
    cmd = FeatureCommand(mock_github_api)
    await cmd.execute(long_desc, feature_context)

    call_kwargs = mock_github_api.create_issue.call_args
    title = call_kwargs.kwargs["title"]
    assert title.endswith("...")
    assert len(title) <= 64  # 60 + "..."


@pytest.mark.asyncio
async def test_feature_short_title_not_truncated(mock_github_api, feature_context):
    """Test that short descriptions are used as-is for the title."""
    short_desc = "add dark mode"
    cmd = FeatureCommand(mock_github_api)
    await cmd.execute(short_desc, feature_context)

    call_kwargs = mock_github_api.create_issue.call_args
    title = call_kwargs.kwargs["title"]
    assert title == short_desc


@pytest.mark.asyncio
async def test_feature_api_failure(mock_github_api, feature_context):
    """Test /feature handles GitHub API errors gracefully."""
    mock_github_api.create_issue.side_effect = RuntimeError("API rate limited")
    cmd = FeatureCommand(mock_github_api)
    result = await cmd.execute("add notifications", feature_context)

    assert "Failed to create issue" in result.text
    assert "API rate limited" in result.text


@pytest.mark.asyncio
async def test_feature_with_quoted_message(mock_github_api, feature_context, feature_db):
    """Test /feature with a quote-reply includes quoted message timestamp."""
    # Store an outgoing message that can be found by quote lookup
    feature_db.messages.log_message(
        sender="penny",
        content="Here is a response that will be quoted",
        direction="outgoing",
    )

    # Set up context with a quoted message
    feature_context.message = IncomingMessage(
        sender=TEST_SENDER,
        content="/feature this could be improved",
        quoted_text="Here is a response that will be quoted",
    )

    cmd = FeatureCommand(mock_github_api)
    await cmd.execute("this could be improved", feature_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]
    assert "Refers to message sent at" in body


@pytest.mark.asyncio
async def test_feature_discord_channel_type(mock_github_api, feature_context):
    """Test /feature from Discord shows discord in the footer."""
    feature_context.channel_type = "discord"
    feature_context.user = "123456789"  # Discord user ID

    cmd = FeatureCommand(mock_github_api)
    await cmd.execute("discord feature request", feature_context)

    call_kwargs = mock_github_api.create_issue.call_args
    body = call_kwargs.kwargs["body"]
    assert "discord" in body
    # Must not contain Discord user ID
    assert "123456789" not in body
