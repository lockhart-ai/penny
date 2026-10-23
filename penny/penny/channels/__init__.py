"""Channel abstraction for communication platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.discord import DiscordChannel
from penny.channels.ios import ApnsClient, ApnsConfig, IosChannel
from penny.channels.manager import ChannelManager
from penny.channels.signal import SignalChannel
from penny.config import Config
from penny.constants import ChannelType

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database


def create_channel_manager(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None = None,
) -> ChannelManager:
    """Create a ChannelManager with all configured channels registered."""
    manager = ChannelManager(
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
    )

    _register_primary_channel(config, message_agent, db, command_registry, manager)

    if config.ios_enabled and config.channel_type != ChannelType.IOS:
        _register_ios_channel(config, message_agent, db, command_registry, manager)

    if config.browser_enabled:
        _register_browser_channel(config, message_agent, db, command_registry, manager)

    return manager


def _register_primary_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
) -> None:
    """Create and register the primary channel (Signal or Discord)."""
    if config.channel_type == ChannelType.DISCORD:
        if not config.discord_bot_token or not config.discord_channel_id:
            raise ValueError("Discord requires DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID")
        channel = DiscordChannel(
            token=config.discord_bot_token,
            channel_id=config.discord_channel_id,
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
        )
        manager.register_channel(ChannelType.DISCORD, channel)
        # Seed a default device for the primary channel (mirrors Signal), so
        # proactive sends resolve to Discord structurally via ``is_default``
        # instead of relying on registration order — and never get captured by a
        # browser addon whose device identifier the profile's sender points at.
        db.devices.register(
            ChannelType.DISCORD,
            config.discord_channel_id,
            "Discord",
            is_default=True,
        )
    elif config.channel_type == ChannelType.SIGNAL:
        if not config.signal_number:
            raise ValueError("Signal requires SIGNAL_NUMBER")
        _register_signal_channel(
            config,
            message_agent,
            db,
            command_registry,
            manager,
            is_default_device=True,
        )
    elif config.channel_type == ChannelType.IOS:
        _register_ios_channel(config, message_agent, db, command_registry, manager)
    else:
        raise ValueError(f"Unknown channel type: {config.channel_type}")


def _register_signal_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
    *,
    is_default_device: bool,
) -> None:
    """Create and register the Signal channel.

    Penny's own number is never a user device (#2765) — it is the channel's
    identity, not an endpoint belonging to the single user. The real Signal
    device is the user's own number: known immediately when a profile already
    exists (a restart), registered here; otherwise left for
    ``SignalChannel._bootstrap_device`` to claim on first contact, since a
    fresh install has no identifier to register yet.
    """
    if not config.signal_number:
        raise ValueError("Signal requires SIGNAL_NUMBER")
    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number,
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
        max_retries=config.llm_max_retries,
        retry_delay=config.llm_retry_delay,
    )
    manager.register_channel(ChannelType.SIGNAL, channel)
    _reclaim_signal_default_device(db, config.signal_number, is_default_device)


def _reclaim_signal_default_device(
    db: Database, penny_number: str, is_default_device: bool
) -> None:
    """Undo the pre-#2765 startup bug's residue, every time Signal starts.

    An existing deployment may still hold a device row keyed on Penny's own
    number, wrongly marked default from every prior startup — left there, the
    send-queue drainer's default-device read would misroute a proactive
    notification to Note-to-Self. Remove that row outright (safe: nothing
    legitimate is ever looked up by Penny's own number), then make the real
    user's Signal device the default when one is already known (an existing
    profile) — a fresh install has no primary sender yet, so there is nothing
    to make default until the bootstrap hook claims one on first contact.
    """
    db.devices.delete_by_identifier(penny_number)
    if not is_default_device:
        return
    primary_sender = db.users.get_primary_sender()
    if primary_sender is None:
        return
    device = db.devices.register(ChannelType.SIGNAL, primary_sender, "Signal", is_default=True)
    if device.id is not None:
        db.devices.set_default(device.id)


def _register_ios_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
) -> None:
    """Create and register the iOS channel."""
    apns_client = _build_apns_client(config)
    channel = IosChannel(
        host=config.ios_host,
        port=config.ios_port,
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
        pairing_token=config.ios_pairing_token,
        apns_client=apns_client,
        is_primary_channel=config.channel_type == ChannelType.IOS,
    )
    manager.register_channel(ChannelType.IOS, channel)


def _build_apns_client(config: Config) -> ApnsClient | None:
    """Build APNs client only when the full credential set is configured."""
    if not (
        config.ios_apns_team_id
        and config.ios_apns_key_id
        and config.ios_apns_key_path
        and config.ios_bundle_id
    ):
        return None
    return ApnsClient(
        ApnsConfig(
            team_id=config.ios_apns_team_id,
            key_id=config.ios_apns_key_id,
            key_path=config.ios_apns_key_path,
            bundle_id=config.ios_bundle_id,
            sandbox=config.ios_apns_sandbox,
            production_team_id=config.ios_apns_production_team_id,
            production_key_id=config.ios_apns_production_key_id,
            production_key_path=config.ios_apns_production_key_path,
            production_bundle_id=config.ios_apns_production_bundle_id,
        )
    )


def _register_browser_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
) -> None:
    """Create and register the browser channel if enabled."""
    from penny.channels.browser import BrowserChannel

    channel = BrowserChannel(
        host=config.browser_host,
        port=config.browser_port,
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
    )
    manager.register_channel(ChannelType.BROWSER, channel)


def create_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None = None,
) -> MessageChannel:
    """Create a single channel (backward compat for tests)."""
    manager = create_channel_manager(config, message_agent, db, command_registry)
    return manager


__all__ = [
    "MessageChannel",
    "IncomingMessage",
    "SignalChannel",
    "DiscordChannel",
    "IosChannel",
    "ChannelManager",
    "create_channel",
    "create_channel_manager",
]
