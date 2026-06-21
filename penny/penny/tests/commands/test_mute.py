"""Integration tests for /mute and /unmute commands."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_mute_command(signal_server, test_config, mock_llm, running_penny):
    """Test /mute sets mute state and returns acknowledgment."""
    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/mute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Notifications muted" in response["message"]
        assert "Use /unmute" in response["message"]
        assert penny.db.users.is_muted(TEST_SENDER) is True

        # Command responses now go through the channel send chokepoint, so the
        # acknowledgment is logged as an outgoing message and surfaces in the
        # penny-messages facade (previously command output bypassed messagelog).
        outgoing = penny.db.memory("penny-messages").read_all()
        assert any("Notifications muted" in e.content for e in outgoing)


@pytest.mark.asyncio
async def test_unmute_command(signal_server, test_config, mock_llm, running_penny):
    """Test /unmute clears mute state and returns acknowledgment."""
    async with running_penny(test_config) as penny:
        # Mute first
        penny.db.users.set_muted(TEST_SENDER)

        await signal_server.push_message(sender=TEST_SENDER, content="/unmute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Notifications unmuted" in response["message"]
        assert penny.db.users.is_muted(TEST_SENDER) is False


@pytest.mark.asyncio
async def test_mute_already_muted(signal_server, test_config, mock_llm, running_penny):
    """Test /mute when already muted returns 'already muted' message."""
    async with running_penny(test_config) as penny:
        penny.db.users.set_muted(TEST_SENDER)

        await signal_server.push_message(sender=TEST_SENDER, content="/mute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "already muted" in response["message"]


@pytest.mark.asyncio
async def test_unmute_already_unmuted(signal_server, test_config, mock_llm, running_penny):
    """Test /unmute when not muted returns 'aren't muted' message."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/unmute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "aren't muted" in response["message"]
