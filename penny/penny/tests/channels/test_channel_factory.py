"""Tests for configured channel registration."""

from unittest.mock import MagicMock

import pytest

from penny.channels import create_channel_manager
from penny.channels.discord import DiscordChannel
from penny.channels.ios import IosChannel
from penny.channels.signal import SignalChannel
from penny.constants import ChannelType
from penny.scheduler.send_queue_drainer import SendQueueDrainer

_USER_NUMBER = "+15559876543"  # distinct from DEFAULT_TEST_CONFIG's signal_number


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
async def test_ios_primary_does_not_register_signal_sidecar(make_config):
    """CHANNEL_TYPE=ios stays iOS-only so prod-ios works without signal-api."""
    config = make_config(channel_type=ChannelType.IOS)
    manager = create_channel_manager(config, message_agent=MagicMock(), db=MagicMock())
    try:
        assert isinstance(manager.get_channel(ChannelType.IOS), IosChannel)
        assert manager.get_channel(ChannelType.SIGNAL) is None
        assert manager.default_channel_type == ChannelType.IOS
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_discord_primary_seeds_default_device(make_config, db):
    """Discord primary seeds a default device (mirrors Signal) so proactive sends
    resolve to Discord structurally via ``is_default`` — not by registration
    order — and can never be captured by a browser addon (#1298)."""
    config = make_config(
        channel_type=ChannelType.DISCORD,
        discord_bot_token="test-token",
        discord_channel_id="1234567890",
    )
    manager = create_channel_manager(config, message_agent=MagicMock(), db=db)
    try:
        assert isinstance(manager.get_channel(ChannelType.DISCORD), DiscordChannel)
        default = db.devices.get_default()
        assert default is not None
        assert default.channel_type == ChannelType.DISCORD
        assert default.identifier == "1234567890"
        assert manager.default_channel_type == ChannelType.DISCORD
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_signal_primary_registers_no_device_on_a_fresh_install(make_config, db):
    """#2765: a fresh install has no profile yet, so startup has no user
    identifier to register — and Penny's own number must never become one."""
    config = make_config(channel_type=ChannelType.SIGNAL)
    manager = create_channel_manager(config, message_agent=MagicMock(), db=db)
    try:
        assert db.devices.get_by_identifier(config.signal_number) is None
        assert db.devices.get_default() is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_signal_primary_registers_the_existing_profiles_number(make_config, db):
    """#2765: when a profile already exists (a restart), startup registers the
    PRIMARY SENDER as the default Signal device — never ``config.signal_number``."""
    db.users.save_info(
        sender=_USER_NUMBER,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    config = make_config(channel_type=ChannelType.SIGNAL)
    manager = create_channel_manager(config, message_agent=MagicMock(), db=db)
    try:
        default = db.devices.get_default()
        assert default is not None
        assert default.identifier == _USER_NUMBER
        assert db.devices.get_by_identifier(config.signal_number) is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_signal_primary_reclaims_a_stale_penny_number_default(make_config, db):
    """#2765: an existing deployment already carries the pre-fix residue — a
    device row keyed on Penny's own number, wrongly marked default. Startup
    must remove it and make the real user's device the default again, so the
    drainer (which reads the default device's identifier) resolves the user,
    never Penny talking to herself."""
    config = make_config(channel_type=ChannelType.SIGNAL)
    db.devices.register(ChannelType.SIGNAL, config.signal_number, "Signal", is_default=True)
    db.users.save_info(
        sender=_USER_NUMBER,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )

    manager = create_channel_manager(config, message_agent=MagicMock(), db=db)
    try:
        assert db.devices.get_by_identifier(config.signal_number) is None
        default = db.devices.get_default()
        assert default is not None
        assert default.identifier == _USER_NUMBER

        drainer = SendQueueDrainer(db=db, config=MagicMock(), chat_agent=MagicMock())
        assert drainer._resolve_recipient() == _USER_NUMBER
        assert drainer._resolve_recipient() != config.signal_number
    finally:
        await manager.close()
