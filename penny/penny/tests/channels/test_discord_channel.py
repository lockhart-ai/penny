"""Tests for the Discord channel's startup behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import discord
import pytest

from penny.channels.discord.channel import (
    DISCORD_DEVELOPER_PORTAL_URL,
    DISCORD_PRIVILEGED_INTENTS_ERROR,
    DiscordChannel,
)


def _make_channel() -> DiscordChannel:
    """Build a DiscordChannel with stubbed dependencies (listen() uses none of them)."""
    return DiscordChannel(
        token="test-token",
        channel_id="123456789",
        message_agent=Mock(),
        db=Mock(),
    )


class TestPrivilegedIntentsStartup:
    async def test_listen_surfaces_actionable_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing Message Content Intent becomes an actionable ConnectionError
        (logged at ERROR to penny.log), not a raw PrivilegedIntentsRequired traceback."""
        channel = _make_channel()
        channel.client.start = AsyncMock(  # type: ignore[method-assign]
            side_effect=discord.errors.PrivilegedIntentsRequired(None)
        )

        with caplog.at_level("ERROR"), pytest.raises(ConnectionError) as excinfo:
            await channel.listen()

        message = str(excinfo.value)
        assert message == DISCORD_PRIVILEGED_INTENTS_ERROR
        assert "Message Content Intent" in message
        assert DISCORD_DEVELOPER_PORTAL_URL in message
        assert isinstance(excinfo.value.__cause__, discord.errors.PrivilegedIntentsRequired)
        assert DISCORD_PRIVILEGED_INTENTS_ERROR in caplog.text
