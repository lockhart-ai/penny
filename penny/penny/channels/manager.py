"""Channel manager — routes messages to/from multiple channels via the device table."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.config import Config
from penny.constants import PermissionResolution
from penny.conversation_machine import ConversationMachine

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import Device, MessageLog
    from penny.llm import LlmClient
    from penny.scheduler import BackgroundScheduler

logger = logging.getLogger(__name__)


class ChannelManager(MessageChannel):
    """Routes messages to/from multiple channels via the device table.

    Incoming: each concrete channel handles its own receive loop, message
    extraction, and reply routing. The manager is not in the incoming path.

    Outgoing (proactive): send_message(recipient) looks up the device table,
    resolves the channel type, and delegates to the correct concrete channel.
    This is what the collector's autonomous sends and startup announcements use.
    """

    def __init__(
        self,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self._channels: dict[str, MessageChannel] = {}
        self._default_channel_type: str | None = None

    # --- Registration ---

    def register_channel(self, channel_type: str, channel: MessageChannel) -> None:
        """Add a concrete channel to the routing table."""
        self._channels[channel_type] = channel
        if self._default_channel_type is None:
            self._default_channel_type = channel_type
        logger.info("Registered channel: %s", channel_type)

    # --- Channel lookup ---

    def _get_default_channel(self) -> MessageChannel:
        """Get the default channel for proactive messages."""
        resolved = self.default_channel_type
        if resolved is None:
            raise RuntimeError("No channels registered")
        return self._channels[resolved]

    def _channel_for_device(self, device: Device | None) -> MessageChannel | None:
        """The registered channel backing a device, or None if absent/unregistered."""
        if device is None:
            return None
        return self._channels.get(device.channel_type)

    def _resolve_channel(self, recipient: str) -> MessageChannel:
        """Resolve the channel for a proactive send — the default device wins.

        Only proactive/autonomous sends flow through the manager (each concrete
        channel handles its own receive→reply loop, which never consults this),
        and they must land on the configured primary channel. So a registered
        default device — seeded at startup for the primary channel (Signal,
        Discord) and set on pairing for iOS — takes precedence over a
        device-identifier match on ``recipient``. Without this, a ``recipient``
        equal to a non-primary device's identifier (e.g. the browser-addon label
        the profile's sender was pinned to during onboarding) captures the send
        and misroutes it to the addon (#1298). The identifier lookup remains only
        as a fallback for a deployment with no default device registered.
        """
        return (
            self._channel_for_device(self._db.devices.get_default())
            or self._channel_for_device(self._db.devices.get_by_identifier(recipient))
            or self._get_default_channel()
        )

    def get_channel(self, channel_type: str) -> MessageChannel | None:
        """Get a specific channel by type."""
        return self._channels.get(channel_type)

    @property
    def default_channel_type(self) -> str | None:
        """Channel type proactive sends resolve to (default device, else first registered).

        The single source of the default-resolution rule: ``_get_default_channel()``
        indexes ``self._channels`` with this, and the startup preflight compares it
        against the configured primary channel — so the check can't drift from what
        actually routes. ``None`` only when no channel is registered.
        """
        default = self._db.devices.get_default()
        if default and default.channel_type in self._channels:
            return default.channel_type
        return self._default_channel_type

    # --- MessageChannel interface ---

    @property
    def sender_id(self) -> str:
        """Sender ID of the default channel."""
        return self._get_default_channel().sender_id

    async def listen(self) -> None:
        """Start all registered channels listening concurrently."""
        tasks = [channel.listen() for channel in self._channels.values()]
        await asyncio.gather(*tasks)

    async def wait_until_ready(self) -> None:
        """Wait until the default outgoing channel can send."""
        await self._get_default_channel().wait_until_ready()

    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        source_name: str | None = None,
        message_log_id: int | None = None,
    ) -> int | None:
        """Route a prepared message to the correct channel via device lookup.

        Logging happens once in the inherited base ``send_message`` /
        ``send_response`` (using the shared db) before this routes the raw send
        to the resolved concrete channel — so manager-routed sends are logged
        exactly once, not double-logged by the concrete channel.
        """
        channel = self._resolve_channel(recipient)
        return await channel._send_raw(
            recipient, message, attachments, quote_message, source_name, message_log_id
        )

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Route a typing indicator to the correct channel."""
        channel = self._resolve_channel(recipient)
        return await channel.send_typing(recipient, typing)

    def prepare_outgoing(self, text: str) -> str:
        """Use the default channel's formatting."""
        return self._get_default_channel().prepare_outgoing(text)

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Not used — each concrete channel extracts its own messages."""
        raise NotImplementedError("ChannelManager does not extract messages directly")

    async def close(self) -> None:
        """Close all registered channels."""
        for channel_type, channel in self._channels.items():
            logger.info("Closing channel: %s", channel_type)
            await channel.close()

    # --- Permission prompt broadcasting ---

    async def broadcast_permission_prompt(
        self,
        request_id: str,
        domain: str,
        url: str,
    ) -> None:
        """Prompt every channel — a channel that fails costs no other its prompt.

        The channels are independent by construction: one prompt goes out to all
        of them and the user answers on whichever device they reach first. So a
        raise from one — a dead addon socket, a Signal API wobble — must not
        abort the loop, which would leave every channel after it with no prompt
        at all and the answer only reachable from a device already broken.
        """
        for channel_type, channel in self._channels.items():
            try:
                await channel.handle_permission_prompt(request_id, domain, url)
            except Exception:
                # Deliberately broad: a channel is an arbitrary transport and its
                # failure modes are its own (httpx, websockets, discord.py, APNs),
                # so there is no type set to enumerate here. Logged with the
                # traceback, never swallowed — louder than the bail it replaces —
                # and CancelledError is a BaseException, so cancelling a broadcast
                # still cancels it. Same shape and reason as the browser channel's
                # connection/sweep guards.
                logger.exception(
                    "Channel %s could not show permission prompt %s", channel_type, request_id
                )

    async def sync_domain_permissions(self) -> None:
        """Notify all channels that domain permissions have changed."""
        for channel in self._channels.values():
            await channel.handle_domain_permissions_changed()

    async def broadcast_permission_dismiss(
        self, request_id: str, resolution: PermissionResolution
    ) -> None:
        """Tell every channel a prompt is resolved, and how it ended.

        Isolated per channel for the reason the prompt is, plus one of its own:
        a channel that never hears the prompt is over keeps whatever it set up
        to watch for an answer — on Signal, a live reaction callback on a
        message nobody is waiting on — so aborting the loop on the first raise
        leaves that state behind on every channel after it.
        """
        for channel_type, channel in self._channels.items():
            try:
                await channel.handle_permission_dismiss(request_id, resolution)
            except Exception:
                # Broad for the reason the prompt's guard is broad — see there.
                logger.exception(
                    "Channel %s could not resolve permission prompt %s", channel_type, request_id
                )

    # --- Delegation to all channels ---

    def set_scheduler(self, scheduler: BackgroundScheduler) -> None:
        """Forward scheduler to all registered channels."""
        super().set_scheduler(scheduler)
        for channel in self._channels.values():
            channel.set_scheduler(scheduler)

    def set_conversation_machine(self, machine: ConversationMachine) -> None:
        """Forward the conversation state machine to all registered channels.

        A receive→reply loop lives on each CONCRETE channel (the manager only
        routes outgoing sends), so the machine has to reach them, not just here."""
        super().set_conversation_machine(machine)
        for channel in self._channels.values():
            channel.set_conversation_machine(machine)

    def set_command_context(
        self,
        config: Config,
        channel_type: str,
        start_time: datetime,
        model_client: LlmClient,
        embedding_model_client: LlmClient,
    ) -> None:
        """Forward command context to all registered channels."""
        super().set_command_context(
            config,
            channel_type,
            start_time,
            model_client,
            embedding_model_client,
        )
        for ch_type, channel in self._channels.items():
            channel.set_command_context(
                config,
                ch_type,
                start_time,
                model_client,
                embedding_model_client,
            )

    async def validate_connectivity(self) -> None:
        """Validate connectivity for all channels."""
        for channel in self._channels.values():
            await channel.validate_connectivity()
