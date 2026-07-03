"""Channel manager — routes messages to/from multiple channels via the device table."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.config import Config

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog
    from penny.llm import LlmClient
    from penny.llm.image_client import OllamaImageClient
    from penny.scheduler import BackgroundScheduler

logger = logging.getLogger(__name__)


class ChannelManager(MessageChannel):
    """Routes messages to/from multiple channels via the device table.

    Incoming: each concrete channel handles its own receive loop, message
    extraction, and reply routing. The manager is not in the incoming path.

    Outgoing (proactive): send_message(recipient) looks up the device table,
    resolves the channel type, and delegates to the correct concrete channel.
    This is what NotifyAgent, ScheduleExecutor, and startup announcements use.
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
        default = self._db.devices.get_default()
        if default:
            channel = self._channels.get(default.channel_type)
            if channel:
                return channel
        if self._default_channel_type:
            return self._channels[self._default_channel_type]
        raise RuntimeError("No channels registered")

    def _resolve_channel(self, recipient: str) -> MessageChannel:
        """Look up the channel for a recipient via the device table."""
        device = self._db.devices.get_by_identifier(recipient)
        if device:
            channel = self._channels.get(device.channel_type)
            if channel:
                return channel
        return self._get_default_channel()

    def get_channel(self, channel_type: str) -> MessageChannel | None:
        """Get a specific channel by type."""
        return self._channels.get(channel_type)

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
    ) -> int | None:
        """Route a prepared message to the correct channel via device lookup.

        Logging happens once in the inherited base ``send_message`` /
        ``send_response`` (using the shared db) before this routes the raw send
        to the resolved concrete channel — so manager-routed sends are logged
        exactly once, not double-logged by the concrete channel.
        """
        channel = self._resolve_channel(recipient)
        return await channel._send_raw(recipient, message, attachments, quote_message)

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
        """Broadcast a permission prompt to all channels."""
        for channel in self._channels.values():
            await channel.handle_permission_prompt(request_id, domain, url)

    async def sync_domain_permissions(self) -> None:
        """Notify all channels that domain permissions have changed."""
        for channel in self._channels.values():
            await channel.handle_domain_permissions_changed()

    async def broadcast_permission_dismiss(self, request_id: str) -> None:
        """Broadcast a permission dismiss to all channels."""
        for channel in self._channels.values():
            await channel.handle_permission_dismiss(request_id)

    # --- Delegation to all channels ---

    def set_scheduler(self, scheduler: BackgroundScheduler) -> None:
        """Forward scheduler to all registered channels."""
        super().set_scheduler(scheduler)
        for channel in self._channels.values():
            channel.set_scheduler(scheduler)

    def set_command_context(
        self,
        config: Config,
        channel_type: str,
        start_time: datetime,
        model_client: LlmClient,
        embedding_model_client: LlmClient | None = None,
        image_model_client: OllamaImageClient | None = None,
    ) -> None:
        """Forward command context to all registered channels."""
        super().set_command_context(
            config,
            channel_type,
            start_time,
            model_client,
            embedding_model_client,
            image_model_client,
        )
        for ch_type, channel in self._channels.items():
            channel.set_command_context(
                config,
                ch_type,
                start_time,
                model_client,
                embedding_model_client,
                image_model_client,
            )

    async def validate_connectivity(self) -> None:
        """Validate connectivity for all channels."""
        for channel in self._channels.values():
            await channel.validate_connectivity()
