"""Tests for configured channel registration."""

from unittest.mock import MagicMock

import pytest

from penny.channels import create_channel_manager
from penny.channels.ios import IosChannel
from penny.channels.signal import SignalChannel
from penny.constants import ChannelType


@pytest.mark.asyncio
async def test_ios_enabled_registers_ios_alongside_signal(make_config):
    """IOS_ENABLED keeps Signal primary while adding the iOS listener."""
    config = make_config(ios_enabled=True)
    manager = create_channel_manager(config, message_agent=MagicMock(), db=MagicMock())
    try:
        assert isinstance(manager.get_channel(ChannelType.SIGNAL), SignalChannel)
        assert isinstance(manager.get_channel(ChannelType.IOS), IosChannel)
        assert manager.default_channel_type == ChannelType.SIGNAL
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_ios_primary_can_run_signal_sidecar(make_config):
    """CHANNEL_TYPE=ios can still run Signal in parallel when Signal is configured."""
    config = make_config(channel_type=ChannelType.IOS)
    manager = create_channel_manager(config, message_agent=MagicMock(), db=MagicMock())
    try:
        assert isinstance(manager.get_channel(ChannelType.IOS), IosChannel)
        assert isinstance(manager.get_channel(ChannelType.SIGNAL), SignalChannel)
        assert manager.default_channel_type == ChannelType.IOS
    finally:
        await manager.close()
