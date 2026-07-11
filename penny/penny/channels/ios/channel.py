"""iOS channel: foreground WebSocket plus APNs preview notifications."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import websockets
from pydantic import BaseModel, ValidationError
from sqlmodel import Session, select
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.browser.models import (
    BROWSER_MSG_TYPE_COLLECTION_TRIGGER,
    BROWSER_MSG_TYPE_CONFIG_REQUEST,
    BROWSER_MSG_TYPE_CONFIG_UPDATE,
    BROWSER_MSG_TYPE_CURSOR_CLEAR,
    BROWSER_MSG_TYPE_CURSOR_SET,
    BROWSER_MSG_TYPE_DOMAIN_DELETE,
    BROWSER_MSG_TYPE_DOMAIN_UPDATE,
    BROWSER_MSG_TYPE_ENTRY_CREATE,
    BROWSER_MSG_TYPE_ENTRY_DELETE,
    BROWSER_MSG_TYPE_ENTRY_UPDATE,
    BROWSER_MSG_TYPE_MEMORIES_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_ARCHIVE,
    BROWSER_MSG_TYPE_MEMORY_CREATE,
    BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_PAGE_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_UPDATE,
    BROWSER_MSG_TYPE_PERMISSION_DECISION,
    BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST,
    BROWSER_MSG_TYPE_SCHEDULE_ADD,
    BROWSER_MSG_TYPE_SCHEDULE_DELETE,
    BROWSER_MSG_TYPE_SCHEDULE_UPDATE,
    BROWSER_MSG_TYPE_SCHEDULES_REQUEST,
    BROWSER_RESP_TYPE_CONFIG,
    BROWSER_RESP_TYPE_PROMPT_LOGS,
    BROWSER_RESP_TYPE_SCHEDULES,
    MEMORY_SECTION_COLLECTOR_RUNS,
    BrowserCollectionTrigger,
    BrowserCollectionTriggerResult,
    BrowserConfigUpdate,
    BrowserCursorClear,
    BrowserCursorSet,
    BrowserDomainDelete,
    BrowserDomainPermissionsSync,
    BrowserDomainUpdate,
    BrowserEntryCreate,
    BrowserEntryDelete,
    BrowserEntryUpdate,
    BrowserMemoriesResponse,
    BrowserMemoryArchive,
    BrowserMemoryChanged,
    BrowserMemoryCreate,
    BrowserMemoryDetailRequest,
    BrowserMemoryDetailResponse,
    BrowserMemoryPageRequest,
    BrowserMemoryPageResponse,
    BrowserMemoryUpdate,
    BrowserPermissionDecision,
    BrowserPermissionDismiss,
    BrowserPermissionPrompt,
    BrowserScheduleAdd,
    BrowserScheduleDelete,
    BrowserScheduleUpdate,
    CursorRecord,
    DomainPermissionRecord,
    MemoryEntryRecord,
    MemoryRecord,
    ScheduleRecord,
)
from penny.channels.ios.apns import ApnsClient, ApnsError
from penny.channels.ios.models import (
    IOS_MSG_TYPE_ACK,
    IOS_MSG_TYPE_EMBEDDING_REQUEST,
    IOS_MSG_TYPE_HEARTBEAT,
    IOS_MSG_TYPE_HISTORY,
    IOS_MSG_TYPE_MESSAGE,
    IOS_MSG_TYPE_PULL,
    IOS_MSG_TYPE_REGISTER,
    IosAckMessages,
    IosEmbeddingRequest,
    IosEmbeddingResponse,
    IosHistoryRequest,
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
from penny.channels.permission_manager import PermissionManager
from penny.config_params import RUNTIME_CONFIG_PARAMS, get_params_by_group
from penny.constants import ChannelType, PennyConstants
from penny.database.memory import (
    EntryInput,
    Inclusion,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryTypeError,
    RecallMode,
)
from penny.database.models import RuntimeConfig, Schedule, UserInfo
from penny.datetime_utils import current_datetime_line
from penny.llm.embeddings import serialize_embedding
from penny.prompts import Prompt
from penny.tools.schedule_tools import ScheduleParseResult

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.agents.collector import Collector
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import Device, IosOutboxItem, MessageLog

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
        self._permission_manager: PermissionManager | None = None
        self._collector: Collector | None = None
        db.messages._on_prompt_logged = self._on_prompt_logged
        db.messages._on_run_outcome_set = self._on_run_outcome_set
        db.memories._on_memory_changed = self._on_memory_changed

    @property
    def sender_id(self) -> str:
        """Identifier for outgoing iOS messages."""
        return "penny"

    def set_permission_manager(self, manager: PermissionManager) -> None:
        """Set the permission manager for routing iOS permission decisions."""
        self._permission_manager = manager

    def set_collector(self, collector: Collector) -> None:
        """Wire the collector so iOS can run a collection extractor on demand."""
        self._collector = collector

    def _on_prompt_logged(self, prompt_data: dict) -> None:
        """Broadcast prompt log updates to connected iOS clients."""
        message = json.dumps({"type": "prompt_log_update", "prompt": prompt_data})
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

    def _on_run_outcome_set(self, run_id: str, outcome: str, reason: str) -> None:
        """Broadcast run outcome updates to connected iOS clients."""
        message = json.dumps(
            {"type": "run_outcome_update", "run_id": run_id, "outcome": outcome, "reason": reason}
        )
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

    def _on_memory_changed(self, name: str | None) -> None:
        """Broadcast memory mutation notifications to connected iOS clients."""
        message = BrowserMemoryChanged(name=name).model_dump_json()
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

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
        elif msg_type == IOS_MSG_TYPE_EMBEDDING_REQUEST:
            await self._handle_embedding_request(ws, data)
        elif msg_type == IOS_MSG_TYPE_PULL:
            await self._handle_pull(ws, data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_HISTORY:
            await self._handle_history(ws, data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_ACK:
            await self._handle_ack(ws, data, device_identifier)
        elif msg_type == IOS_MSG_TYPE_HEARTBEAT:
            pass  # Keepalive frame — acknowledged by staying connected; no server-side state.
        elif msg_type == BROWSER_MSG_TYPE_CONFIG_REQUEST:
            await self._handle_config_request(ws)
        elif msg_type == BROWSER_MSG_TYPE_CONFIG_UPDATE:
            await self._handle_config_update(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_SCHEDULES_REQUEST:
            await self._handle_schedules_request(ws)
        elif msg_type == BROWSER_MSG_TYPE_SCHEDULE_ADD:
            await self._handle_schedule_add(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_SCHEDULE_UPDATE:
            await self._handle_schedule_update(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_SCHEDULE_DELETE:
            await self._handle_schedule_delete(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST:
            await self._handle_prompt_logs_request(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORIES_REQUEST:
            await self._handle_memories_request(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST:
            await self._handle_memory_detail_request(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORY_PAGE_REQUEST:
            await self._handle_memory_page_request(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_COLLECTION_TRIGGER:
            await self._handle_collection_trigger(ws, data)
        elif msg_type == BROWSER_MSG_TYPE_CURSOR_SET:
            self._handle_cursor_set(data)
        elif msg_type == BROWSER_MSG_TYPE_CURSOR_CLEAR:
            self._handle_cursor_clear(data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORY_CREATE:
            await self._handle_memory_create(data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORY_UPDATE:
            await self._handle_memory_update(data)
        elif msg_type == BROWSER_MSG_TYPE_MEMORY_ARCHIVE:
            self._handle_memory_archive(data)
        elif msg_type == BROWSER_MSG_TYPE_ENTRY_CREATE:
            self._handle_entry_create(data)
        elif msg_type == BROWSER_MSG_TYPE_ENTRY_UPDATE:
            self._handle_entry_update(data)
        elif msg_type == BROWSER_MSG_TYPE_ENTRY_DELETE:
            self._handle_entry_delete(data)
        elif msg_type == BROWSER_MSG_TYPE_DOMAIN_UPDATE:
            await self._handle_domain_update(data)
        elif msg_type == BROWSER_MSG_TYPE_DOMAIN_DELETE:
            await self._handle_domain_delete(data)
        elif msg_type == BROWSER_MSG_TYPE_PERMISSION_DECISION:
            self._handle_permission_decision(data)
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

    # --- Shared admin surface ---

    async def _handle_config_request(self, ws: ServerConnection) -> None:
        """Return all runtime config params with current values."""
        params = []
        for group, group_params in get_params_by_group():
            for param in group_params:
                current = (
                    getattr(self._config.runtime, param.key) if self._config else param.default
                )
                params.append(
                    {
                        "key": param.key,
                        "value": str(current),
                        "default": str(param.default),
                        "description": param.description,
                        "type": param.type.__name__,
                        "group": group,
                    }
                )
        await self._send_json(ws, {"type": BROWSER_RESP_TYPE_CONFIG, "params": params})

    async def _handle_config_update(self, ws: ServerConnection, data: dict) -> None:
        """Validate and persist a single config param update."""
        try:
            req = BrowserConfigUpdate(**data)
        except ValidationError:
            logger.warning("Invalid config_update: %s", str(data)[:200])
            return
        param = RUNTIME_CONFIG_PARAMS.get(req.key)
        if not param:
            logger.warning("Unknown config key: %s", req.key)
            return
        try:
            validated = param.validator(req.value)
        except ValueError as error:
            logger.warning("Invalid config value %s=%s: %s", req.key, req.value, error)
            return
        with Session(self._db.engine) as session:
            existing = session.get(RuntimeConfig, req.key)
            if existing:
                existing.value = str(validated)
                existing.updated_at = datetime.utcnow()
                session.add(existing)
            else:
                session.add(
                    RuntimeConfig(
                        key=req.key,
                        value=str(validated),
                        description=param.description,
                        updated_at=datetime.utcnow(),
                    )
                )
            session.commit()
        logger.info("Config updated via iOS: %s = %s", req.key, validated)
        await self._handle_config_request(ws)

    async def _handle_schedules_request(self, ws: ServerConnection) -> None:
        """Query all schedules for the primary user and send them."""
        await self._send_schedules(ws)

    async def _handle_schedule_add(self, ws: ServerConnection, data: dict) -> None:
        """Parse a natural language schedule command and create it."""
        try:
            req = BrowserScheduleAdd(**data)
        except ValidationError:
            logger.warning("Invalid schedule_add: %s", str(data)[:200])
            return

        primary = self._db.users.get_primary_sender()
        if not primary or not req.command.strip():
            await self._send_schedules(ws, error="No user profile found")
            return

        with Session(self._db.engine) as session:
            user_info = session.exec(select(UserInfo).where(UserInfo.sender == primary)).first()
            if not user_info or not user_info.timezone:
                await self._send_schedules(ws, error="Set your timezone first via /profile")
                return
            user_timezone = user_info.timezone

        prompt = Prompt.SCHEDULE_PARSE_PROMPT.format(
            today=current_datetime_line(self._db),
            timezone=user_timezone,
            command=req.command.strip(),
        )
        try:
            response = await self._model_client.generate(prompt=prompt, format="json")
            result = ScheduleParseResult.model_validate_json(response.message.content)
        except Exception as error:
            logger.warning("Failed to parse schedule from iOS: %s", error)
            await self._send_schedules(ws, error="Could not parse schedule timing")
            return

        cron_parts = result.cron_expression.split()
        if len(cron_parts) != 5:
            await self._send_schedules(ws, error="Invalid cron expression")
            return

        with Session(self._db.engine) as session:
            session.add(
                Schedule(
                    user_id=primary,
                    user_timezone=user_timezone,
                    cron_expression=result.cron_expression,
                    prompt_text=result.prompt_text,
                    timing_description=result.timing_description,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()
        await self._send_schedules(ws)

    async def _handle_schedule_update(self, ws: ServerConnection, data: dict) -> None:
        """Update a schedule's prompt text."""
        try:
            req = BrowserScheduleUpdate(**data)
        except ValidationError:
            logger.warning("Invalid schedule_update: %s", str(data)[:200])
            return
        with Session(self._db.engine) as session:
            schedule = session.get(Schedule, req.schedule_id)
            if schedule:
                schedule.prompt_text = req.prompt_text
                session.add(schedule)
                session.commit()
        await self._send_schedules(ws)

    async def _handle_schedule_delete(self, ws: ServerConnection, data: dict) -> None:
        """Delete a schedule by ID."""
        try:
            req = BrowserScheduleDelete(**data)
        except ValidationError:
            logger.warning("Invalid schedule_delete: %s", str(data)[:200])
            return
        with Session(self._db.engine) as session:
            schedule = session.get(Schedule, req.schedule_id)
            if schedule:
                session.delete(schedule)
                session.commit()
        await self._send_schedules(ws)

    async def _send_schedules(self, ws: ServerConnection, error: str | None = None) -> None:
        """Send all schedules for the primary user."""
        primary = self._db.users.get_primary_sender()
        schedules: list[ScheduleRecord] = []
        if primary:
            with Session(self._db.engine) as session:
                rows = list(session.exec(select(Schedule).where(Schedule.user_id == primary)))
                schedules = []
                for s in sorted(rows, key=lambda schedule: schedule.created_at):
                    if s.id is None:
                        continue
                    schedules.append(
                        ScheduleRecord(
                            id=s.id,
                            timing_description=s.timing_description,
                            prompt_text=s.prompt_text,
                            cron_expression=s.cron_expression,
                        )
                    )
        await self._send_json(
            ws,
            {
                "type": BROWSER_RESP_TYPE_SCHEDULES,
                "schedules": [s.model_dump() for s in schedules],
                "error": error,
            },
        )

    _PROMPT_LOG_PAGE_SIZE = 50

    async def _handle_prompt_logs_request(self, ws: ServerConnection, data: dict) -> None:
        """Query prompt logs grouped by run_id and send them to iOS."""
        agent_name = data.get("agent_name") or None
        offset = int(data.get("offset", 0))
        query = (data.get("query") or "").strip() or None
        flagged_only = bool(data.get("flagged_only", False))
        runs = self._db.messages.get_prompt_log_runs(
            limit=self._PROMPT_LOG_PAGE_SIZE,
            offset=offset,
            agent_name=agent_name,
            query=query,
            flagged_only=flagged_only,
        )
        await self._send_json(
            ws,
            {
                "type": BROWSER_RESP_TYPE_PROMPT_LOGS,
                "runs": runs,
                "has_more": (not flagged_only) and len(runs) == self._PROMPT_LOG_PAGE_SIZE,
            },
        )

    async def _handle_memories_request(self, ws: ServerConnection, data: dict) -> None:
        """List every memory with metadata and entry counts."""
        memories = self._db.memories.list_all()
        query = (data.get("query") or "").strip()
        if query:
            memories = self._filter_memories(memories, query)
        counts = self._db.memories.entry_counts()
        records = [self._memory_to_record(m, counts.get(m.name, 0)) for m in memories]
        await self._send_ws(ws, BrowserMemoriesResponse(memories=records))

    def _filter_memories(self, memories: list, query: str) -> list:
        """Keep memories matching query by metadata or entry content."""
        needle = query.lower()
        entry_matches = self._db.memories.names_with_entry_match(query)

        def matches(memory) -> bool:
            return (
                needle in memory.name.lower()
                or needle in memory.description.lower()
                or (memory.intent is not None and needle in memory.intent.lower())
                or memory.name in entry_matches
            )

        return [memory for memory in memories if matches(memory)]

    _MEMORY_PAGE_SIZE = 50

    async def _handle_memory_detail_request(self, ws: ServerConnection, data: dict) -> None:
        """Send metadata plus first page of memory entries/activity."""
        try:
            req = BrowserMemoryDetailRequest(**data)
        except ValidationError:
            logger.warning("Invalid memory_detail_request: %s", str(data)[:200])
            return
        memory = self._db.memories.get(req.name)
        if memory is None:
            logger.warning("memory_detail_request for unknown memory: %s", req.name)
            return
        await self._send_ws(ws, self._build_memory_detail(memory, data))

    def _build_memory_detail(self, memory, data: dict) -> BrowserMemoryDetailResponse:
        """Assemble one memory detail payload."""
        counts = self._db.memories.entry_counts()
        query = (data.get("query") or "").strip() or None
        record = self._memory_to_record(memory, counts.get(memory.name, 0))
        entries, entries_has_more = self._entries_page(memory, 0, query)
        runs, runs_has_more = self._collector_runs_page(memory, 0)
        return BrowserMemoryDetailResponse(
            memory=record,
            entries=entries,
            entries_has_more=entries_has_more,
            collector_runs=runs,
            collector_runs_has_more=runs_has_more,
            cursors=self._cursors_for(memory),
        )

    async def _handle_memory_page_request(self, ws: ServerConnection, data: dict) -> None:
        """Send one more page of a memory-detail section."""
        try:
            req = BrowserMemoryPageRequest(**data)
        except ValidationError:
            logger.warning("Invalid memory_page_request: %s", str(data)[:200])
            return
        memory = self._db.memories.get(req.name)
        if memory is None:
            logger.warning("memory_page_request for unknown memory: %s", req.name)
            return
        await self._send_ws(ws, self._memory_page_payload(memory, req, data))

    def _memory_page_payload(
        self, memory, req: BrowserMemoryPageRequest, data: dict
    ) -> BrowserMemoryPageResponse:
        """One page of the requested memory detail section."""
        if req.section == MEMORY_SECTION_COLLECTOR_RUNS:
            runs, has_more = self._collector_runs_page(memory, req.offset)
            return BrowserMemoryPageResponse(
                name=req.name, section=req.section, runs=runs, has_more=has_more
            )
        query = (data.get("query") or "").strip() or None
        entries, has_more = self._entries_page(memory, req.offset, query)
        return BrowserMemoryPageResponse(
            name=req.name, section=req.section, entries=entries, has_more=has_more
        )

    async def _handle_collection_trigger(self, ws: ServerConnection, data: dict) -> None:
        """Run a collection extractor on demand."""
        try:
            req = BrowserCollectionTrigger(**data)
        except ValidationError:
            logger.warning("Invalid collection_trigger: %s", str(data)[:200])
            return
        if self._collector is None:
            success, message = False, "Collector is not available."
        else:
            success, message = await self._collector.run_for(req.name)
        await self._send_ws(
            ws, BrowserCollectionTriggerResult(name=req.name, success=success, message=message)
        )

    def _handle_cursor_set(self, data: dict) -> None:
        """Set a collection cursor over one log."""
        try:
            req = BrowserCursorSet(**data)
            last_read_at = datetime.fromisoformat(req.last_read_at)
        except ValidationError, ValueError:
            logger.warning("Invalid cursor_set: %s", str(data)[:200])
            return
        self._db.cursors.set_position(req.name, req.log_name, last_read_at)
        self._on_memory_changed(req.name)

    def _handle_cursor_clear(self, data: dict) -> None:
        """Clear a collection cursor over one log."""
        try:
            req = BrowserCursorClear(**data)
        except ValidationError:
            logger.warning("Invalid cursor_clear: %s", str(data)[:200])
            return
        self._db.cursors.clear(req.name, req.log_name)
        self._on_memory_changed(req.name)

    def _entries_page(
        self, memory, offset: int, query: str | None = None
    ) -> tuple[list[MemoryEntryRecord], bool]:
        """One newest-first page of a memory's entries."""
        if memory.name == PennyConstants.MEMORY_COLLECTOR_RUNS_LOG:
            run_log = self._db.memories.run_log()
            rows = (
                run_log.newest_entries(self._MEMORY_PAGE_SIZE, offset)
                if run_log is not None
                else []
            )
        else:
            content = self._db.memory(memory.name)
            rows = (
                content.newest_entries(self._MEMORY_PAGE_SIZE, offset, search=query)
                if content is not None
                else []
            )
        records = [self._entry_to_record(row) for row in rows]
        return records, len(records) == self._MEMORY_PAGE_SIZE

    def _collector_runs_page(self, memory, offset: int) -> tuple[list[dict], bool]:
        """One newest-first page of this collection's collector runs."""
        if memory.type != "collection":
            return [], False
        runs = self._db.messages.get_target_runs(memory.name, self._MEMORY_PAGE_SIZE, offset)
        return runs, len(runs) == self._MEMORY_PAGE_SIZE

    def _cursors_for(self, memory) -> list[CursorRecord]:
        """The collection's read positions over logs it reads."""
        if memory.type != "collection":
            return []
        cursors = self._db.cursors.list_for(memory.name)
        return [
            CursorRecord(log_name=log_name, last_read_at=last_read_at.isoformat())
            for log_name, last_read_at in sorted(cursors)
        ]

    @staticmethod
    def _memory_to_record(memory, entry_count: int) -> MemoryRecord:
        return MemoryRecord(
            name=memory.name,
            type=memory.type,
            description=memory.description,
            intent=memory.intent,
            inclusion=memory.inclusion,
            recall=memory.recall,
            published=memory.published,
            archived=memory.archived,
            extraction_prompt=memory.extraction_prompt,
            collector_interval_seconds=memory.collector_interval_seconds,
            last_collected_at=(
                memory.last_collected_at.isoformat() if memory.last_collected_at else None
            ),
            entry_count=entry_count,
        )

    @staticmethod
    def _entry_to_record(entry) -> MemoryEntryRecord:
        return MemoryEntryRecord(
            id=entry.id,
            key=entry.key,
            content=entry.content,
            author=entry.author,
            created_at=entry.created_at.isoformat(),
        )

    _IOS_ENTRY_AUTHOR = "user"

    @staticmethod
    def _parse_routing(
        inclusion: str | None, recall: str | None
    ) -> tuple[Inclusion | None, RecallMode | None] | None:
        """Resolve a supplied memory routing pair."""
        if recall == "off":
            return Inclusion.NEVER, RecallMode.RECENT
        try:
            parsed_inclusion = Inclusion(inclusion) if inclusion is not None else None
            parsed_recall = RecallMode(recall) if recall is not None else None
        except ValueError:
            return None
        return parsed_inclusion, parsed_recall

    async def _handle_memory_create(self, data: dict) -> None:
        """Create a user-authored collection."""
        try:
            req = BrowserMemoryCreate(**data)
        except ValidationError:
            logger.warning("Invalid memory_create: %s", str(data)[:200])
            return
        routing = self._parse_routing(req.inclusion, req.recall)
        if routing is None:
            logger.warning(
                "Invalid inclusion/recall in memory_create: %s/%s", req.inclusion, req.recall
            )
            return
        inclusion, recall = routing
        description_embedding = await self._message_agent.embed_description(req.description)
        try:
            self._db.memories.create_collection(
                req.name,
                req.description,
                inclusion or Inclusion.RELEVANT,
                recall or RecallMode.RELEVANT,
                extraction_prompt=req.extraction_prompt,
                collector_interval_seconds=req.collector_interval_seconds,
                description_embedding=description_embedding,
                intent=req.intent,
                published=req.published,
            )
        except MemoryAlreadyExistsError:
            logger.warning("memory_create with duplicate name: %s", req.name)

    async def _handle_memory_update(self, data: dict) -> None:
        """Edit collection metadata."""
        try:
            req = BrowserMemoryUpdate(**data)
        except ValidationError:
            logger.warning("Invalid memory_update: %s", str(data)[:200])
            return
        routing = self._parse_routing(req.inclusion, req.recall)
        if routing is None:
            logger.warning(
                "Invalid inclusion/recall in memory_update: %s/%s", req.inclusion, req.recall
            )
            return
        inclusion, recall = routing
        description_embedding = (
            await self._message_agent.embed_description(req.description)
            if req.description is not None
            else None
        )
        try:
            self._db.memories.update_collection_metadata(
                req.name,
                description=req.description,
                intent=req.intent,
                inclusion=inclusion,
                recall=recall,
                extraction_prompt=req.extraction_prompt,
                collector_interval_seconds=req.collector_interval_seconds,
                description_embedding=description_embedding,
                published=req.published,
            )
        except (MemoryNotFoundError, MemoryTypeError) as exc:
            logger.warning("memory_update failed for %s: %s", req.name, exc)

    def _handle_memory_archive(self, data: dict) -> None:
        """Archive a memory."""
        try:
            req = BrowserMemoryArchive(**data)
        except ValidationError:
            logger.warning("Invalid memory_archive: %s", str(data)[:200])
            return
        try:
            self._db.memories.archive(req.name)
        except MemoryNotFoundError as exc:
            logger.warning("memory_archive failed for %s: %s", req.name, exc)

    def _handle_entry_create(self, data: dict) -> None:
        """Manually add an entry to a collection."""
        try:
            req = BrowserEntryCreate(**data)
        except ValidationError:
            logger.warning("Invalid entry_create: %s", str(data)[:200])
            return
        memory = self._db.memory(req.memory)
        if memory is None:
            logger.warning("entry_create on missing memory %s", req.memory)
            return
        try:
            memory.write(
                [EntryInput(key=req.key, content=req.content)],
                author=self._IOS_ENTRY_AUTHOR,
            )
        except MemoryTypeError as exc:
            logger.warning("entry_create on non-collection %s: %s", req.memory, exc)

    def _handle_entry_update(self, data: dict) -> None:
        """Replace an existing keyed entry."""
        try:
            req = BrowserEntryUpdate(**data)
        except ValidationError:
            logger.warning("Invalid entry_update: %s", str(data)[:200])
            return
        memory = self._db.memory(req.memory)
        if memory is None:
            logger.warning("entry_update on missing memory %s", req.memory)
            return
        try:
            memory.update(req.key, req.content, author=self._IOS_ENTRY_AUTHOR)
        except MemoryTypeError as exc:
            logger.warning("entry_update on non-collection %s: %s", req.memory, exc)

    def _handle_entry_delete(self, data: dict) -> None:
        """Delete a keyed entry."""
        try:
            req = BrowserEntryDelete(**data)
        except ValidationError:
            logger.warning("Invalid entry_delete: %s", str(data)[:200])
            return
        memory = self._db.memory(req.memory)
        if memory is None:
            logger.warning("entry_delete on missing memory %s", req.memory)
            return
        try:
            memory.delete(req.key)
        except MemoryTypeError as exc:
            logger.warning("entry_delete on non-collection %s: %s", req.memory, exc)

    async def _handle_domain_update(self, data: dict) -> None:
        """Persist a domain permission and sync this connection."""
        try:
            req = BrowserDomainUpdate(**data)
        except ValidationError:
            logger.warning("Invalid domain_update: %s", str(data)[:200])
            return
        if self._permission_manager:
            await self._permission_manager.set_permission(req.domain, req.permission)
        else:
            self._db.domain_permissions.set_permission(req.domain, req.permission)

    async def _handle_domain_delete(self, data: dict) -> None:
        """Delete a domain permission."""
        try:
            req = BrowserDomainDelete(**data)
        except ValidationError:
            logger.warning("Invalid domain_delete: %s", str(data)[:200])
            return
        if self._permission_manager:
            await self._permission_manager.delete_permission(req.domain)
        else:
            self._db.domain_permissions.delete(req.domain)

    def _handle_permission_decision(self, data: dict) -> None:
        """Forward an iOS permission prompt decision to the permission manager."""
        try:
            req = BrowserPermissionDecision(**data)
        except ValidationError:
            logger.warning("Invalid permission_decision: %s", str(data)[:200])
            return
        if self._permission_manager:
            self._permission_manager.handle_decision(req.request_id, req.allowed)

    async def handle_permission_prompt(self, request_id: str, domain: str, url: str) -> None:
        """Send a permission prompt to connected iOS clients."""
        prompt = BrowserPermissionPrompt(request_id=request_id, domain=domain, url=url)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, prompt)

    async def handle_permission_dismiss(self, request_id: str) -> None:
        """Dismiss a permission prompt on connected iOS clients."""
        dismiss = BrowserPermissionDismiss(request_id=request_id)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, dismiss)

    async def handle_domain_permissions_changed(self) -> None:
        """Sync the full domain permissions list to connected iOS clients."""
        await self._sync_domain_permissions()

    async def _sync_domain_permissions(self) -> None:
        """Broadcast the full domain permissions list to iOS clients."""
        rows = self._db.domain_permissions.get_all()
        records = [DomainPermissionRecord(domain=r.domain, permission=r.permission) for r in rows]
        msg = BrowserDomainPermissionsSync(permissions=records)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, msg)

    async def _send_json(self, ws: ServerConnection, payload: dict) -> None:
        """Send a raw JSON-compatible payload."""
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(payload))

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

    async def _handle_embedding_request(self, ws: ServerConnection, data: dict) -> None:
        """Return one query embedding for client-side semantic search."""
        try:
            request = IosEmbeddingRequest(**data)
        except ValidationError:
            await self._send_ws(
                ws,
                IosEmbeddingResponse(
                    request_id=str(data.get("request_id", "")),
                    error="invalid_embedding_request",
                ),
            )
            return

        text = request.text.strip()
        if not text:
            await self._send_ws(
                ws,
                IosEmbeddingResponse(request_id=request.request_id, error="empty_embedding_text"),
            )
            return
        embedding = await self._embed_message(text)
        if embedding is None:
            await self._send_ws(
                ws,
                IosEmbeddingResponse(
                    request_id=request.request_id,
                    error="embedding_unavailable",
                ),
            )
            return
        await self._send_ws(
            ws,
            IosEmbeddingResponse(
                request_id=request.request_id,
                embedding=base64.b64encode(serialize_embedding(embedding)).decode("ascii"),
            ),
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
        messages_by_id = self._db.messages.get_by_ids(
            {row.message_log_id for row in rows if row.message_log_id is not None}
        )
        records = []
        for row in rows:
            message = messages_by_id.get(row.message_log_id) if row.message_log_id else None
            records.append(_outbox_record(row, message.embedding if message else None))
        await self._send_ws(ws, IosMessages(messages=records, mode="outbox"))

    async def _handle_history(
        self, ws: ServerConnection, data: dict, device_identifier: str
    ) -> None:
        """Return one bounded, older-first page from the shared message log."""
        try:
            request = IosHistoryRequest(**data)
            cursor = _decode_history_cursor(request.before, request.channel_types)
        except ValidationError, ValueError:
            await self._send_ws(ws, IosStatus(error="invalid_history_request"))
            return
        if self._connections.get(device_identifier) is None:
            await self._send_ws(ws, IosStatus(error="register_required"))
            return

        if request.count_only:
            await self._send_ws(
                ws,
                IosMessages(
                    messages=[],
                    mode="history_count",
                    total_count=self._db.messages.ios_history_count(
                        channel_types=request.channel_types
                    ),
                    attachments_included=request.include_attachments,
                ),
            )
            return

        rows, has_more = self._db.messages.ios_history_page(
            channel_types=request.channel_types,
            before=cursor,
            limit=request.limit,
        )
        records = [
            _history_record(row, include_attachments=request.include_attachments) for row in rows
        ]
        next_cursor = None
        if has_more and rows:
            oldest = rows[0]
            next_cursor = _encode_history_cursor(
                oldest[0].timestamp, _required_message_id(oldest[0]), request.channel_types
            )
        await self._send_ws(
            ws,
            IosMessages(
                messages=records,
                mode="history",
                next_cursor=next_cursor,
                has_more=has_more,
                attachments_included=request.include_attachments,
            ),
        )

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
        message_log_id: int | None = None,
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
            message_log_id=message_log_id,
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
        message_log_id: int | None = None,
        device_id: int,
        message: str,
        attachments: list[str] | None,
        source_name: str | None,
    ) -> IosOutboxItem:
        source_type, source_hint = _source_metadata(source_name)
        push_title, push_summary = _push_preview(message)
        return self._db.ios.enqueue_outbox(
            message_log_id=message_log_id,
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


def _outbox_record(row: IosOutboxItem, embedding: bytes | None = None) -> IosOutboxRecord:
    return IosOutboxRecord(
        id=_required_outbox_id(row),
        message_id=row.message_log_id,
        outbox_id=row.id,
        created_at=row.created_at.isoformat(),
        content=row.content,
        attachments=_decode_outbox_attachments(row),
        source_type=row.source_type,
        source_name=row.source_name,
        source_hint=row.source_hint,
        push_title=row.push_title,
        push_summary=row.push_summary,
        embedding=_encode_embedding(embedding),
    )


def _history_record(
    row: tuple[MessageLog, Device | None, IosOutboxItem | None],
    *,
    include_attachments: bool = True,
) -> IosOutboxRecord:
    message, device, outbox = row
    if outbox is not None and (outbox.source_hint or outbox.source_name):
        source_type = outbox.source_type
        source_name = outbox.source_name
        _, derived_hint = _source_metadata(outbox.source_name)
        source_hint = outbox.source_hint or derived_hint
    elif message.direction == "incoming" or message.parent_id is not None:
        source_hint = "Chat"
        source_type = None
        source_name = None
    elif message.thought_id is not None:
        source_hint = "Notifier"
        source_type = "collector"
        source_name = None
    else:
        source_hint = "Penny"
        source_type = None
        source_name = None
    return IosOutboxRecord(
        id=_required_message_id(message),
        message_id=message.id,
        outbox_id=outbox.id if outbox is not None else None,
        created_at=message.timestamp.isoformat(),
        content=message.content,
        attachments=(
            _decode_outbox_attachments(outbox) if include_attachments and outbox is not None else []
        ),
        source_type=source_type,
        source_name=source_name,
        source_hint=source_hint,
        push_title="",
        push_summary="",
        direction=message.direction,
        channel_type=device.channel_type if device is not None else None,
        device_label=device.label if device is not None else None,
        device_identifier=device.identifier if device is not None else None,
        parent_id=message.parent_id,
        embedding=_encode_embedding(message.embedding),
    )


def _encode_embedding(embedding: bytes | None) -> str | None:
    """Encode Penny's stored Float32 vector for the iOS wire protocol."""
    return base64.b64encode(embedding).decode("ascii") if embedding is not None else None


def _decode_outbox_attachments(row: IosOutboxItem) -> list[str]:
    """Recover the inline attachment payload retained by an outbox row."""
    if not row.attachments_json:
        return []
    try:
        attachments = json.loads(row.attachments_json)
    except TypeError, ValueError:
        logger.warning("Ignoring malformed iOS outbox attachments (outbox_id=%s)", row.id)
        return []
    if not isinstance(attachments, list):
        return []
    return [attachment for attachment in attachments if isinstance(attachment, str)]


def _encode_history_cursor(
    timestamp: datetime, message_id: int, channel_types: list[str] | None
) -> str:
    payload = {
        "timestamp": timestamp.isoformat(),
        "id": message_id,
        "channels": channel_types if channel_types is not None else [],
    }
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _decode_history_cursor(
    value: str | None, channel_types: list[str] | None
) -> tuple[datetime, int] | None:
    if value is None:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(value.encode()).decode())
        expected_channels = channel_types if channel_types is not None else []
        if payload.get("channels", []) != expected_channels:
            raise ValueError("history cursor scope mismatch")
        timestamp = datetime.fromisoformat(payload["timestamp"])
        return timestamp.replace(tzinfo=None), int(payload["id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid history cursor") from exc


def _required_message_id(message: MessageLog) -> int:
    if message.id is None:
        raise ValueError("history message is missing its database ID")
    return message.id


def _required_outbox_id(row: IosOutboxItem) -> int:
    if row.id is None:
        raise ValueError("iOS outbox row is missing its database ID")
    return row.id


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
