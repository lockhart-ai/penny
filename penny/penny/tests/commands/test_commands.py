"""Integration tests for /commands command."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_commands_list(signal_server, test_config, mock_llm, running_penny):
    """Test /commands lists all available commands."""
    async with running_penny(test_config) as _penny:
        # Send /commands
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list registered commands with their descriptions
        assert "**Available Commands**" in response["message"]
        assert "**/commands**" in response["message"]
        assert "**/profile**" in response["message"]
        assert "List all commands" in response["message"]
        assert "View or update your profile" in response["message"]


@pytest.mark.asyncio
async def test_commands_help_specific(signal_server, test_config, mock_llm, running_penny):
    """Test /commands <name> shows help for specific command."""
    async with running_penny(test_config) as _penny:
        # Send /commands profile
        await signal_server.push_message(sender=TEST_SENDER, content="/commands profile")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show profile command help
        assert "**Command: /profile**" in response["message"]
        assert "View your current profile" in response["message"]
        assert "**Usage**:" in response["message"]
        assert "`/profile`" in response["message"]


@pytest.mark.asyncio
async def test_commands_unknown(signal_server, test_config, mock_llm, running_penny):
    """Test /commands <unknown> shows error."""
    async with running_penny(test_config) as _penny:
        # Send /commands unknown
        await signal_server.push_message(sender=TEST_SENDER, content="/commands unknown")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown command: /unknown" in response["message"]
        assert "Use /commands to see available commands" in response["message"]
