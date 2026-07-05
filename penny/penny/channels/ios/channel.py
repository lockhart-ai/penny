"""iOS channel: foreground WebSocket plus APNs preview notifications."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import websockets
from pydantic import BaseModel, ValidationError
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.ios.apns import ApnsClient, ApnsError
from penny.channels.ios.models import (
    IOS_MSG_TYPE_ACK,
    IOS_MSG_TYPE_HEARTBEAT,
    IOS_MSG_TYPE_MESSAGE,
    IOS_MSG_TYPE_PULL,
    IOS_MSG_TYPE_REGISTER,
    IosAckMessages,
    IosIncomingMessage,
    IosMessages,
    IosMessagesAcked,
    IosOutboxChanged,
    IosOutboxRecord,
    IosPullMessages,
    IosRegister,
    IosRegistered,
    IosStatus,
    IosTyping,
)
from penny.constants import ChannelType, PennyConstants

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import IosOutboxItem, MessageLog

logger = logging.getLogger(__name__)

TEST_PUSH_MESSAGE = "This is a Penny test push."
PUSH_GREETING_TITLE = "Hi from Penny"

# Recognized ``source_name`` values (the ``author`` an outbound message carries)
# for iOS push attribution.  ``chat``/``notifier`` reuse their canonical homes on
# ``PennyConstants``; these are the ones with no home elsewhere.
SOURCE_NAME_STARTUP = "startup"
SOURCE_NAME_SCHEDULE = "schedule"
SOURCE_NAME_TEST_PUSH = "test_push"

# ``source_type`` values stamped on an outbox row and the APNs payload.
SOURCE_TYPE_TEST_PUSH = "test_push"
SOURCE_TYPE_COLLECTOR = "collector"

# ``source_hint`` display labels surfaced to the app.
SOURCE_HINT_DEFAULT = "Penny"
SOURCE_HINT_TEST_PUSH = "Test Push"
SOURCE_HINT_NOTIFIER = "Notifier"
SOURCE_HINT_COLLECTOR_PREFIX = "Collector: "


@dataclass
class IosConnectionInfo:
    """Metadata for a connected iOS websocket."""

    ws: ServerConnection
    device_id: int
    identifier: str


class IosChannel(MessageChannel):
    """Primary iOS messaging channel.

    Outbound delivery is durable: every send writes an ``ios_outbox`` row.  A
    foreground websocket receives an ``outbox_changed`` hint and pulls rows; a
    background/offline app gets an APNs preview notification, then pulls on open.
    """

    def __init__(
        self,
        host: str,
        port: int,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
        pairing_token: str | None = None,
        apns_client: ApnsClient | None = None,
        is_primary_channel: bool = False,
    ) -> None:
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self._host = host
        self._port = port
        self._pairing_token = pairing_token
        self._apns_client = apns_client
        self._is_primary_channel = is_primary_channel
        self._server: Server | None = None
        self._connections: dict[str, IosConnectionInfo] = {}
        self._closed = asyncio.Event()

    @property
    def sender_id(self) -> str:
        """Identifier for outgoing iOS messages."""
        return "penny"

    async def listen(self) -> None:
        """Start the iOS WebSocket server and block forever."""
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=PennyConstants.BROWSER_WS_MAX_FRAME_BYTES,
        )
        logger.info("iOS channel listening on ws://%s:%d", self._host, self._port)
        await self._closed.wait()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle one iOS websocket connection."""
        logger.info("iOS websocket client connected")
        await self._send_ws(ws, IosStatus(connected=True))
        device_identifier: str | None = None
        try:
            async for raw in ws:
                device_identifier = await self._process_raw_message(ws, raw, device_identifier)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("iOS connection handler crashed")
        finally:
            logger.info(
                "iOS websocket client disconnected (device=%s, code=%s, reason=%r)",
                device_identifier or "unregistered",
                ws.close_code,
                ws.close_reason,
            )
            self._cleanup_connection(ws, device_identifier)

    async def _process_raw_message(
        self, ws: ServerConnection, raw: str | bytes, device_identifier: str | None
    ) -> str | None:
        """Parse and dispatch a single websocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_ws(ws, IosStatus(error="invalid_json"))
            return device_identifier

        msg_type = data.get("type", "")
        if msg_type == IOS_MSG_TYPE_REGISTER:
            return await self._handle_register(ws, data)
        if device_identifier is None:
            await self._send_ws(ws, IosStatus(error="register_required"))
            return None
        if msg_type == IOS_MSG_TYPE_MESSAGE:
            await self._handle_chat_message(data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_PULL:
            await self._handle_pull(ws, data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_ACK:
            await self._handle_ack(ws, data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_HEARTBEAT:
            pass  # Keepalive frame — acknowledged by staying connected; no server-side state.
        return device_identifier

    async def _handle_register(self, ws: ServerConnection, data: dict) -> str | None:
        """Register or refresh an iOS device."""
        try:
            msg = IosRegister(**data)
        except ValidationError:
            await self._send_ws(ws, IosStatus(error="invalid_register"))
            return None
        if self._pairing_token and msg.pairing_token != self._pairing_token:
            await self._send_ws(ws, IosStatus(error="invalid_pairing_token"))
            return None

        is_default, pending_count = self._persist_registration(ws, msg)
        await self._send_ws(
            ws,
            IosRegistered(
                device_id=msg.device_id,
                is_default=is_default,
                pending_count=pending_count,
            ),
        )
        return msg.device_id

    def _persist_registration(self, ws: ServerConnection, msg: IosRegister) -> tuple[bool, int]:
        """Upsert the device + iOS registration; claim default only as the primary channel.

        In sidecar mode (``CHANNEL_TYPE`` != ios) the device is registered without touching
        the default flag, so the primary channel (e.g. Signal) keeps proactive-send routing.
        """
        device = self._db.devices.register(
            channel_type=ChannelType.IOS,
            identifier=msg.device_id,
            label=msg.label,
            is_default=self._is_primary_channel,
        )
        if device.id is None:
            return False, 0
        if self._is_primary_channel:
            self._db.devices.set_default(device.id)
        self._db.ios.upsert_registration(
            device=device,
            apns_token=msg.apns_token,
            apns_environment=msg.apns_environment,
            app_version=msg.app_version,
            device_secret=msg.device_secret,
        )
        self._connections[msg.device_id] = IosConnectionInfo(
            ws=ws, device_id=device.id, identifier=msg.device_id
        )
        is_default = self._db.devices.get_default_identifier() == msg.device_id
        return is_default, self._db.ios.pending_count(device.id)

    async def _handle_chat_message(self, data: dict, device_identifier: str) -> None:
        """Forward an iOS user message through the shared channel pipeline."""
        try:
            msg = IosIncomingMessage(**data)
        except ValidationError:
            return
        if _is_test_push_request(msg.content):
            await self._send_test_push(device_identifier)
            return
        await self.handle_message(
            {
                "ios_sender": device_identifier,
                "content": msg.content,
            }
        )

    async def _handle_pull(self, ws: ServerConnection, data: dict, device_identifier: str) -> None:
        """Return unacknowledged outbox messages."""
        try:
            msg = IosPullMessages(**data)
        except ValidationError:
            await self._send_ws(ws, IosStatus(error="invalid_pull"))
            return
        conn = self._connections.get(device_identifier)
        if conn is None:
            await self._send_ws(ws, IosStatus(error="register_required"))
            return
        rows = self._db.ios.pending_for_device(conn.device_id, limit=msg.limit)
        await self._send_ws(ws, IosMessages(messages=[_outbox_record(row) for row in rows]))

    async def _handle_ack(self, ws: ServerConnection, data: dict, device_identifier: str) -> None:
        """Acknowledge displayed/persisted outbox rows."""
        try:
            msg = IosAckMessages(**data)
        except ValidationError:
            await self._send_ws(ws, IosStatus(error="invalid_ack"))
            return
        conn = self._connections.get(device_identifier)
        if conn is None:
            await self._send_ws(ws, IosStatus(error="register_required"))
            return
        count = self._db.ios.mark_acked(conn.device_id, msg.ids)
        await self._send_ws(ws, IosMessagesAcked(count=count))

    def _cleanup_connection(self, ws: ServerConnection, device_identifier: str | None) -> None:
        if not device_identifier:
            return
        conn = self._connections.get(device_identifier)
        if conn is not None and conn.ws is ws:
            self._connections.pop(device_identifier, None)

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract a message from iOS WebSocket data."""
        sender = raw_data.get("ios_sender")
        content = raw_data.get("content", "").strip()
        if not sender or not content:
            return None
        return IncomingMessage(
            sender=sender,
            content=content,
            channel_type=ChannelType.IOS,
            device_identifier=sender,
        )

    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        source_name: str | None = None,
    ) -> int | None:
        """Persist message to the iOS outbox and notify the app."""
        device = self._db.devices.get_by_identifier(recipient)
        if device is None or device.id is None:
            logger.warning("No iOS device for recipient: %s", recipient)
            return None

        item = self._enqueue_ios_outbox_item(
            device_id=device.id,
            message=message,
            attachments=attachments,
            source_name=source_name,
        )
        if item.id is None:
            return None

        conn = self._connections.get(recipient)
        if conn is not None:
            await self._send_outbox_changed(conn)
        else:
            await self._send_push_preview(device.id, item)
        return item.id

    async def _send_test_push(self, recipient: str) -> int | None:
        """Force a diagnostic APNs notification for the registered iOS device."""
        device = self._db.devices.get_by_identifier(recipient)
        if device is None or device.id is None:
            logger.warning("No iOS device for test push recipient: %s", recipient)
            return None

        item = self._enqueue_ios_outbox_item(
            device_id=device.id,
            message=TEST_PUSH_MESSAGE,
            attachments=None,
            source_name=SOURCE_NAME_TEST_PUSH,
        )
        if item.id is None:
            return None

        logger.info("Sending test iOS push notification to %s (outbox_id=%s)", recipient, item.id)
        await self._send_push_preview(device.id, item)
        return item.id

    def _enqueue_ios_outbox_item(
        self,
        *,
        device_id: int,
        message: str,
        attachments: list[str] | None,
        source_name: str | None,
    ) -> IosOutboxItem:
        source_type, source_hint = _source_metadata(source_name)
        push_title, push_summary = _push_preview(message)
        return self._db.ios.enqueue_outbox(
            device_id=device_id,
            content=message,
            attachments=attachments,
            source_type=source_type,
            source_name=source_name,
            source_hint=source_hint,
            push_title=push_title,
            push_summary=push_summary,
        )

    async def _send_outbox_changed(self, conn: IosConnectionInfo) -> None:
        pending = self._db.ios.pending_count(conn.device_id)
        await self._send_ws(conn.ws, IosOutboxChanged(pending_count=pending))

    async def _send_push_preview(self, device_id: int, item: IosOutboxItem) -> None:
        """Send APNs preview when possible; record errors but keep outbox durable."""
        if self._apns_client is None or item.id is None:
            return
        registration = self._db.ios.get_registration(device_id)
        if registration is None or not registration.push_enabled or not registration.apns_token:
            return
        try:
            logger.info(
                "Sending iOS preview notification to APNs "
                "(device_id=%s, outbox_id=%s, source_type=%s, source_name=%s)",
                device_id,
                item.id,
                item.source_type,
                item.source_name,
            )
            await self._apns_client.send_preview(
                device_token=registration.apns_token,
                title=item.push_title,
                body=item.push_summary,
                badge=self._db.ios.pending_count(device_id),
                outbox_id=item.id,
                source_type=item.source_type,
                source_name=item.source_name,
                thread_id=_thread_id(item.source_name),
                environment=registration.apns_environment,
            )
            self._db.ios.mark_push_sent(item.id)
        except ApnsError as error:
            self._db.ios.mark_push_error(item.id, error.reason)
            if error.invalid_token:
                logger.warning(
                    "APNs rejected iOS device token; disabling push "
                    "(device_id=%s, outbox_id=%s, reason=%s)",
                    device_id,
                    item.id,
                    error.reason,
                )
                self._db.ios.disable_push(device_id)
            else:
                logger.warning(
                    "APNs rejected preview notification "
                    "(device_id=%s, outbox_id=%s, status=%s, reason=%s)",
                    device_id,
                    item.id,
                    error.status_code,
                    error.reason,
                )
        except httpx.HTTPError as error:
            self._db.ios.mark_push_error(item.id, str(error))

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send typing state to a connected iOS client."""
        conn = self._connections.get(recipient)
        if conn is None:
            return False
        await self._send_ws(conn.ws, IosTyping(active=typing))
        return True

    async def close(self) -> None:
        """Shut down websocket and APNs resources."""
        self._closed.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._apns_client:
            await self._apns_client.close()
        logger.info("iOS channel closed")

    @staticmethod
    async def _send_ws(ws: ServerConnection, msg: BaseModel) -> None:
        """Send a protocol message, suppressing closed sockets."""
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg.model_dump_json(exclude_none=True))


def _outbox_record(row: IosOutboxItem) -> IosOutboxRecord:
    attachments = json.loads(row.attachments_json) if row.attachments_json else []
    return IosOutboxRecord(
        id=row.id or 0,
        created_at=row.created_at.isoformat(),
        content=row.content,
        attachments=attachments,
        source_type=row.source_type,
        source_name=row.source_name,
        source_hint=row.source_hint,
        push_title=row.push_title,
        push_summary=row.push_summary,
    )


_PASSTHROUGH_SOURCE_NAMES = frozenset(
    {SOURCE_NAME_STARTUP, SOURCE_NAME_SCHEDULE, PennyConstants.CHAT_AGENT_NAME}
)


def _source_metadata(source_name: str | None) -> tuple[str | None, str | None]:
    if not source_name:
        return None, SOURCE_HINT_DEFAULT
    if source_name in _PASSTHROUGH_SOURCE_NAMES:
        return source_name, source_name.title()
    if source_name == SOURCE_NAME_TEST_PUSH:
        return SOURCE_TYPE_TEST_PUSH, SOURCE_HINT_TEST_PUSH
    if source_name == PennyConstants.MEMORY_NOTIFIER_COLLECTION:
        return SOURCE_TYPE_COLLECTOR, SOURCE_HINT_NOTIFIER
    return SOURCE_TYPE_COLLECTOR, f"{SOURCE_HINT_COLLECTOR_PREFIX}{source_name}"


def _push_preview(message: str) -> tuple[str, str]:
    body = _summarize_for_push(message)
    return PUSH_GREETING_TITLE, body


def _summarize_for_push(message: str, limit: int = 140) -> str:
    text = _strip_markup(message)
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    summary = first_sentence or text
    if len(summary) <= limit:
        return summary
    return summary[: limit - 1].rstrip() + "..."


def _strip_markup(message: str) -> str:
    text = re.sub(r"<[^>]+>", " ", message)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`~#>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_test_push_request(message: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
    return normalized in {
        "send me a test push",
        "send a test push",
        "send test push",
        "test push",
        "send me a test notification",
        "send a test notification",
        "send test notification",
        "test notification",
    }


def _thread_id(source_name: str | None) -> str | None:
    if not source_name:
        return None
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", source_name).strip("-")
    return f"penny-{safe}" if safe else None
