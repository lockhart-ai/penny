"""Integration tests for the general command system (routing, logging, threading)."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_unknown_command(signal_server, test_config, mock_llm, running_penny):
    """Test unknown command shows error."""
    async with running_penny(test_config) as _penny:
        # Send unknown command
        await signal_server.push_message(sender=TEST_SENDER, content="/unknown")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown command: /unknown" in response["message"]
        assert "Use /commands to see available commands" in response["message"]


@pytest.mark.asyncio
async def test_command_threading_blocked(signal_server, test_config, mock_llm, running_penny):
    """Test that thread-replying to a command is blocked."""
    async with running_penny(test_config) as _penny:
        # First, send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "**Available Commands**" in response1["message"]

        # Try to thread-reply to the command
        # Build quote dict matching Signal's Quote model
        quote = {"id": 1, "text": "/commands"}
        await signal_server.push_message(
            sender=TEST_SENDER, content="What does this mean?", quote=quote
        )

        # Should get threading not supported message
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "Commands can't be used in threads" in response2["message"]


@pytest.mark.asyncio
async def test_command_logging(signal_server, test_config, mock_llm, running_penny):
    """Test that commands are logged to the database."""
    async with running_penny(test_config) as penny:
        # Send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/commands profile")
        await signal_server.wait_for_message(timeout=5.0)

        # Check database for command log
        from penny.database.models import CommandLog

        with penny.db.get_session() as session:
            from sqlmodel import select

            logs = list(session.exec(select(CommandLog)).all())
            assert len(logs) == 1
            log = logs[0]
            assert log.command_name == "commands"
            assert log.command_args == "profile"
            assert log.user == TEST_SENDER
            assert log.channel_type == "signal"
            assert "**Command: /profile**" in log.response
            assert log.error is None


@pytest.mark.asyncio
async def test_command_not_logged_to_message_table(
    signal_server, test_config, mock_llm, running_penny
):
    """Test that commands are NOT logged to MessageLog table."""
    async with running_penny(test_config) as penny:
        # Send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")
        await signal_server.wait_for_message(timeout=5.0)

        # Check that no messages were logged
        from penny.database.models import MessageLog

        with penny.db.get_session() as session:
            from sqlmodel import select

            logs = list(session.exec(select(MessageLog)).all())
            assert len(logs) == 0
