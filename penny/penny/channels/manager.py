"""Channel manager — routes messages to/from multiple channels via the device table."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.turns import TurnRegistry
from penny.config import Config
from penny.constants import PermissionResolution

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
        *,
        turn_registry: TurnRegistry,
    ):
        super().__init__(
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
            turn_registry=turn_registry,
        )
        self._channels: dict[str, MessageChannel] = {}
        self._default_channel_type: str | None = None
        # The manager fans Penny's own outgoing messages out itself (#3137):
        # it already holds every registered channel directly, so it is its
        # own ``PeerFanout`` — this is what makes an autonomous send (through
        # ``SendQueueDrainer``, whose channel IS the manager) fan out exactly
        # like a per-turn reply from a concrete channel does.
        self.set_peer_manager(self)

    # --- Registration ---

    def register_channel(self, channel_type: str, channel: MessageChannel) -> None:
        """Add a concrete channel to the routing table.

        Also wires the channel's outgoing fan-out collaborator (#3137) to
        this manager — the one registration dependency every channel needs
        to reach every OTHER device; see ``PeerFanout``.
        """
        self._channels[channel_type] = channel
        channel.set_peer_manager(self)
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

    def _device_for_outgoing(self, recipient: str) -> Device | None:
        """The device a PROACTIVE send to ``recipient`` actually reaches —
        mirrors ``_resolve_channel``'s own default-device-wins rule (#1298),
        so the messagelog row's ``device_id`` stamp and the outgoing fan-out's
        exclusion (#3137) both name the device that really got the primary
        send, not whichever device ``recipient`` happens to name.  Falls back
        to the identifier lookup only when there is no default device at all
        (a fresh install, before #2765's bootstrap claims one)."""
        return self._db.devices.get_default() or self._db.devices.get_by_identifier(recipient)

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
        """Close all registered channels — one failing to close costs no other its own close."""
        await self._fan_out(
            (channel_type, channel.close()) for channel_type, channel in self._channels.items()
        )

    # --- Guarded fan-out (#1981) ---

    async def _fan_out(self, calls: Iterable[tuple[str, Awaitable[None]]]) -> int:
        """Await every ``(label, call)`` independently: one raising is logged
        and never blocks the rest.  The shared shape behind every broadcast on
        this class — permission prompts/dismissals, domain-permission sync,
        channel close, and the outgoing delivery fan-out (#3137) — extracted
        once enough customers existed to justify it (#1981).  Returns how many
        calls completed without raising, so a caller that needs to know
        whether ANYONE got through (``broadcast_permission_prompt``) can tell
        that apart from every channel failing.
        """
        succeeded = 0
        for label, call in calls:
            try:
                await call
                succeeded += 1
            except Exception:
                # Deliberately broad: a channel is an arbitrary transport and its
                # failure modes are its own (httpx, websockets, discord.py, APNs),
                # so there is no type set to enumerate here. Logged with the
                # traceback, never swallowed, and CancelledError is a
                # BaseException, so cancelling a fan-out still cancels it.
                logger.exception("Fan-out to %s failed", label)
        return succeeded

    # --- Permission prompt broadcasting ---

    async def broadcast_permission_prompt(
        self,
        request_id: str,
        domain: str,
        url: str,
    ) -> bool:
        """Prompt every channel — a channel that fails costs no other its prompt.

        The channels are independent by construction: one prompt goes out to all
        of them and the user answers on whichever device they reach first.

        Returns whether at least one channel took the prompt (#1981 part 2):
        when every channel fails, ``PermissionManager._prompt`` can name that
        as the real cause instead of waiting out the full timeout and
        reporting a plain "timed out" that implies someone saw it and never
        answered.
        """
        shown = await self._fan_out(
            (channel_type, channel.handle_permission_prompt(request_id, domain, url))
            for channel_type, channel in self._channels.items()
        )
        return shown > 0

    async def sync_domain_permissions(self) -> None:
        """Notify all channels that domain permissions have changed — one
        channel failing to sync costs no other its own sync."""
        await self._fan_out(
            (channel_type, channel.handle_domain_permissions_changed())
            for channel_type, channel in self._channels.items()
        )

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
        await self._fan_out(
            (channel_type, channel.handle_permission_dismiss(request_id, resolution))
            for channel_type, channel in self._channels.items()
        )

    # --- Outgoing delivery fan-out (#3137) ---

    async def fan_out_outgoing(
        self,
        *,
        exclude_device_id: int | None,
        message_id: int,
        content: str,
        attachments: list[str] | None,
        source_name: str | None,
    ) -> None:
        """Deliver an already-logged outgoing message to every registered
        device OTHER than ``exclude_device_id`` (rule 1) — through each
        device's own channel's own native send primitive, so a Signal fan-out
        is a real new Signal message, an iOS fan-out is a real durable
        ``ios_outbox`` row with its own APNs preview, and so on.  One
        ``messagelog`` row stays logged exactly once; this only adds more
        DELIVERIES of it, each recorded in ``message_delivery``.  A device
        whose channel type isn't registered, or whose send raises, is
        recorded and skipped — never blocking another device's delivery.
        """
        await self._fan_out(
            (
                device.identifier,
                self._deliver_to_device(device, message_id, content, attachments, source_name),
            )
            for device in self._db.devices.get_all()
            if device.id != exclude_device_id
        )

    async def _deliver_to_device(
        self,
        device: Device,
        message_id: int,
        content: str,
        attachments: list[str] | None,
        source_name: str | None,
    ) -> None:
        """One device's own delivery attempt, recorded either way — self-
        contained, so ``_fan_out``'s own guard never has to tell one device's
        expected failure (a channel not registered, a dead transport) apart
        from a genuine bug here. An unregistered channel type and a raising
        send both end the same way: logged, and a ``message_delivery`` row
        naming why, never an unretrieved exception.
        """
        assert device.id is not None
        channel = self._channel_for_device(device)
        if channel is None:
            self._db.messages.record_delivery(
                message_id, device.id, external_id=None, error="channel not registered"
            )
            return
        try:
            external_id = await channel._send_raw(
                device.identifier,
                channel.prepare_outgoing(content),
                attachments,
                None,
                source_name,
                message_id,
            )
        except Exception as exc:
            logger.exception("Outgoing fan-out to device %s failed", device.identifier)
            self._db.messages.record_delivery(
                message_id, device.id, external_id=None, error=str(exc)
            )
            return
        self._db.messages.record_delivery(
            message_id,
            device.id,
            external_id=str(external_id) if external_id else None,
            error=None if external_id else "send returned no external id",
        )

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
