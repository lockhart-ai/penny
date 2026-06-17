"""Browser extension channel — WebSocket server implementing MessageChannel."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import websockets
from pydantic import BaseModel, ValidationError
from sqlmodel import Session, select
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.base import IncomingMessage, MessageChannel, PageContext, ProgressTracker
from penny.channels.browser.models import (
    BROWSER_MSG_TYPE_CAPABILITIES_UPDATE,
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
    BROWSER_MSG_TYPE_HEARTBEAT,
    BROWSER_MSG_TYPE_MEMORIES_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_ARCHIVE,
    BROWSER_MSG_TYPE_MEMORY_CREATE,
    BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_PAGE_REQUEST,
    BROWSER_MSG_TYPE_MEMORY_UPDATE,
    BROWSER_MSG_TYPE_MESSAGE,
    BROWSER_MSG_TYPE_PERMISSION_DECISION,
    BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST,
    BROWSER_MSG_TYPE_REGISTER,
    BROWSER_MSG_TYPE_SCHEDULE_ADD,
    BROWSER_MSG_TYPE_SCHEDULE_DELETE,
    BROWSER_MSG_TYPE_SCHEDULE_UPDATE,
    BROWSER_MSG_TYPE_SCHEDULES_REQUEST,
    BROWSER_MSG_TYPE_TOOL_RESPONSE,
    BROWSER_RESP_TYPE_CONFIG,
    BROWSER_RESP_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE,
    BROWSER_RESP_TYPE_PROMPT_LOGS,
    BROWSER_RESP_TYPE_SCHEDULES,
    BROWSER_RESP_TYPE_STATUS,
    BROWSER_RESP_TYPE_TYPING,
    MEMORY_SECTION_COLLECTOR_RUNS,
    MEMORY_SECTION_ENTRIES,
    BrowserCapabilitiesUpdate,
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
    BrowserIncoming,
    BrowserMemoriesResponse,
    BrowserMemoryArchive,
    BrowserMemoryChanged,
    BrowserMemoryCreate,
    BrowserMemoryDetailRequest,
    BrowserMemoryDetailResponse,
    BrowserMemoryPageRequest,
    BrowserMemoryPageResponse,
    BrowserMemoryUpdate,
    BrowserOutgoing,
    BrowserPermissionDecision,
    BrowserPermissionDismiss,
    BrowserPermissionPrompt,
    BrowserRegister,
    BrowserRunOutcomeUpdate,
    BrowserScheduleAdd,
    BrowserScheduleDelete,
    BrowserScheduleUpdate,
    BrowserToolRequest,
    BrowserToolResponse,
    CursorRecord,
    DomainPermissionRecord,
    MemoryEntryRecord,
    MemoryRecord,
    ScheduleRecord,
)
from penny.channels.permission_manager import PermissionManager
from penny.commands.schedule import ScheduleParseResult
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
from penny.prompts import Prompt
from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.agents.collector import Collector
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


def _attachment_to_src(attachment: str) -> str | None:
    """Convert an attachment string to an <img> src value."""
    if attachment.startswith("http"):
        return attachment
    if attachment.startswith("data:"):
        return attachment
    # Raw base64 — assume PNG (Ollama image generation output)
    if len(attachment) > 100:
        return f"data:image/png;base64,{attachment}"
    return None


@dataclass
class ConnectionInfo:
    """Metadata about a connected browser extension."""

    ws: ServerConnection
    tool_use_enabled: bool = False
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))


class BrowserChannel(MessageChannel):
    """WebSocket server channel for the browser extension sidebar."""

    def __init__(
        self,
        host: str,
        port: int,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self._host = host
        self._port = port
        self._server: Server | None = None
        self._connections: dict[str, ConnectionInfo] = {}
        self._pending_requests: dict[str, asyncio.Future[tuple[str, str | None]]] = {}
        self._permission_manager: PermissionManager | None = None
        self._collector: Collector | None = None
        db.messages._on_prompt_logged = self._on_prompt_logged
        db.messages._on_run_outcome_set = self._on_run_outcome_set
        db.memories._on_memory_changed = self._on_memory_changed

    def _on_prompt_logged(self, prompt_data: dict) -> None:
        """Callback fired after each prompt is logged — broadcast to browsers."""
        message = json.dumps({"type": BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE, "prompt": prompt_data})
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

    def _on_run_outcome_set(self, run_id: str, outcome: str, reason: str) -> None:
        """Callback fired when a run outcome is set — broadcast to browsers."""
        payload = BrowserRunOutcomeUpdate(run_id=run_id, outcome=outcome, reason=reason)
        message = payload.model_dump_json()
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

    def _on_memory_changed(self, name: str | None) -> None:
        """Callback fired after any memory mutation — broadcast to browsers
        so the Memories tab can refresh.  ``name`` is the affected memory
        when the change is scoped to one (writes, archives, metadata edits);
        ``None`` for fan-out events."""
        message = BrowserMemoryChanged(name=name).model_dump_json()
        for conn in self._connections.values():
            asyncio.ensure_future(conn.ws.send(message))

    @property
    def sender_id(self) -> str:
        """Identifier for outgoing browser messages."""
        return "penny"

    def set_permission_manager(self, manager: PermissionManager) -> None:
        """Set the permission manager for routing addon permission decisions."""
        self._permission_manager = manager

    def set_collector(self, collector: Collector) -> None:
        """Wire the collector so the addon can run a collection's extractor
        on demand (the "run extractor" button)."""
        self._collector = collector

    @property
    def has_tool_connection(self) -> bool:
        """Whether any tool-use-enabled browser is connected.

        Deliberately lenient about heartbeat staleness: a connection that has
        gone quiet might be a suspended background script, but it might equally
        be an older addon build that doesn't send the keepalive — so we still let
        the browse attempt happen rather than declaring the browser offline.
        ``_get_tool_connection`` prefers a fresh socket when one exists.
        """
        return any(c.tool_use_enabled for c in self._connections.values())

    def _has_fresh_heartbeat(self, conn: ConnectionInfo) -> bool:
        """True if this connection has heartbeated within the liveness window.

        A suspended background script keeps its TCP socket alive (Firefox pongs
        the server's pings at the network layer) but stops sending app
        heartbeats — so a stale heartbeat marks a socket that is *likely*
        no longer processing tool requests, to be deprioritized in routing.
        """
        age = (datetime.now(UTC) - conn.last_heartbeat).total_seconds()
        return age <= PennyConstants.BROWSER_HEARTBEAT_TIMEOUT_SECONDS

    # --- WebSocket server ---

    async def listen(self) -> None:
        """Start the WebSocket server and block forever.

        ``max_size`` lifts the websockets default frame cap (1 MiB) so an addon
        tool response carrying a page's base64 image data URI doesn't overflow
        the frame — which the library would otherwise reject with a 1009 close,
        tearing down the connection mid-browse.
        """
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=PennyConstants.BROWSER_WS_MAX_FRAME_BYTES,
        )
        logger.info("Browser channel listening on ws://%s:%d", self._host, self._port)
        await asyncio.Future()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single browser extension connection."""
        logger.info("Browser connected")
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_STATUS, connected=True))

        device_label: str | None = None
        try:
            async for raw in ws:
                device_label = await self._process_raw_message(ws, raw, device_label)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            # A handler raising anything other than ConnectionClosed aborts the
            # receive loop and silently drops the socket — log it loudly so a
            # handler bug shows up as the cause of a disconnect, not a mystery.
            logger.exception(
                "Browser connection handler crashed (device=%s) — closing socket",
                device_label or "unregistered",
            )
        finally:
            logger.info(
                "Browser socket closed (device=%s, code=%s, reason=%r)",
                device_label or "unregistered",
                ws.close_code,
                ws.close_reason,
            )
            self._cleanup_connection(ws, device_label)

    def _cleanup_connection(self, ws: ServerConnection, device_label: str | None) -> None:
        """Remove this socket and reject pending requests on disconnect.

        Only evict the registry entry if it still points at *this* socket: when
        an addon reconnects, ``_handle_register`` rewrites
        ``connections[label].ws`` to the new socket before the old socket's
        handler reaches here, so a blind ``pop(label)`` would drop the live
        replacement and leave the addon "connected but unreachable" until the
        next message re-registers it.
        """
        if device_label:
            conn = self._connections.get(device_label)
            if conn is not None and conn.ws is ws:
                self._connections.pop(device_label, None)
            elif conn is not None:
                logger.info(
                    "Browser %s already reconnected on a newer socket; keeping it", device_label
                )
        # Reject any pending tool requests — they were sent on this socket and
        # can't be answered now it's gone; the browse tool retries on whatever
        # connection is live.
        for _request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(ConnectionError("Browser disconnected"))
        logger.info("Browser disconnected: %s", device_label or "unregistered")

    # --- Message dispatch ---

    async def _process_raw_message(
        self, ws: ServerConnection, raw: str | bytes, device_label: str | None
    ) -> str | None:
        """Parse and dispatch a single WebSocket message. Returns updated device_label."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from browser: %s", str(raw)[:200])
            return device_label

        msg_type = data.get("type", "")

        if msg_type == BROWSER_MSG_TYPE_REGISTER:
            label = self._handle_register(ws, data)
            await self._sync_domain_permissions()
            return label

        if msg_type == BROWSER_MSG_TYPE_TOOL_RESPONSE:
            self._handle_tool_response(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_PERMISSION_DECISION:
            msg = BrowserPermissionDecision(**data)
            if self._permission_manager:
                self._permission_manager.handle_decision(msg.request_id, msg.allowed)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MESSAGE:
            return await self._handle_chat_message(ws, data, device_label)

        if msg_type == BROWSER_MSG_TYPE_HEARTBEAT:
            self._handle_heartbeat(device_label)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_CAPABILITIES_UPDATE:
            self._handle_capabilities_update(data, device_label)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_DOMAIN_UPDATE:
            await self._handle_domain_update(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_DOMAIN_DELETE:
            await self._handle_domain_delete(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_CONFIG_REQUEST:
            await self._handle_config_request(ws)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_CONFIG_UPDATE:
            await self._handle_config_update(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_SCHEDULES_REQUEST:
            await self._handle_schedules_request(ws)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_SCHEDULE_ADD:
            await self._handle_schedule_add(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_SCHEDULE_UPDATE:
            await self._handle_schedule_update(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_SCHEDULE_DELETE:
            await self._handle_schedule_delete(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST:
            await self._handle_prompt_logs_request(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORIES_REQUEST:
            await self._handle_memories_request(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST:
            await self._handle_memory_detail_request(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORY_PAGE_REQUEST:
            await self._handle_memory_page_request(ws, data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_COLLECTION_TRIGGER:
            await self._handle_collection_trigger(ws, data)
            return

        if msg_type == BROWSER_MSG_TYPE_CURSOR_SET:
            self._handle_cursor_set(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_CURSOR_CLEAR:
            self._handle_cursor_clear(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORY_CREATE:
            await self._handle_memory_create(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORY_UPDATE:
            await self._handle_memory_update(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MEMORY_ARCHIVE:
            self._handle_memory_archive(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_ENTRY_CREATE:
            self._handle_entry_create(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_ENTRY_UPDATE:
            self._handle_entry_update(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_ENTRY_DELETE:
            self._handle_entry_delete(data)
            return device_label

        return device_label

    def _handle_heartbeat(self, device_label: str | None) -> None:
        """Record connection liveness from the addon's keepalive ping.

        This is a liveness signal, not user activity: it refreshes the
        connection's ``last_heartbeat`` (so ``_get_tool_connection`` can route
        around stale, JS-suspended sockets) but deliberately does NOT reset the
        scheduler's idle timer.  The addon pings every ~15s; resetting idle on
        each would keep the system perpetually "active" and starve the idle-
        gated background collectors.  Only real conversation (a chat message)
        counts as activity.
        """
        if device_label:
            conn = self._connections.get(device_label)
            if conn:
                conn.last_heartbeat = datetime.now(UTC)

    def _handle_capabilities_update(self, data: dict, device_label: str | None) -> None:
        """Update a connection's tool-use capability."""
        update = BrowserCapabilitiesUpdate(**data)
        if device_label:
            conn = self._connections.get(device_label)
            if conn:
                conn.tool_use_enabled = update.tool_use_enabled
                logger.info("Browser %s tool_use_enabled=%s", device_label, update.tool_use_enabled)

    def _handle_register(self, ws: ServerConnection, data: dict) -> str:
        """Register a browser connection by device label."""
        msg = BrowserRegister(**data)
        device_label = msg.sender
        existing = self._connections.get(device_label)
        if existing:
            existing.ws = ws
            # A reconnect re-points the entry at the new socket — refresh
            # liveness so the freshly reconnected addon isn't judged stale by
            # ``_get_tool_connection`` on its old (pre-suspension) timestamp.
            existing.last_heartbeat = datetime.now(UTC)
        else:
            self._connections[device_label] = ConnectionInfo(ws=ws)
        self._auto_register_device(device_label)
        logger.info("Browser registered: %s", device_label)
        return device_label

    # --- Domain permissions ---

    async def _handle_domain_update(self, data: dict) -> None:
        """Route domain update to the permission manager."""
        msg = BrowserDomainUpdate(**data)
        if self._permission_manager:
            await self._permission_manager.set_permission(msg.domain, msg.permission)

    async def _handle_domain_delete(self, data: dict) -> None:
        """Route domain delete to the permission manager."""
        msg = BrowserDomainDelete(**data)
        if self._permission_manager:
            await self._permission_manager.delete_permission(msg.domain)

    async def _sync_domain_permissions(self) -> None:
        """Broadcast the full domain permissions list to all connected addons."""
        rows = self._db.domain_permissions.get_all()
        records = [DomainPermissionRecord(domain=r.domain, permission=r.permission) for r in rows]
        msg = BrowserDomainPermissionsSync(permissions=records)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, msg)

    # --- Permission prompts (called by ChannelManager) ---

    async def handle_permission_prompt(self, request_id: str, domain: str, url: str) -> None:
        """Send a permission prompt to all connected browser addons."""
        prompt = BrowserPermissionPrompt(request_id=request_id, domain=domain, url=url)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, prompt)

    async def handle_permission_dismiss(self, request_id: str) -> None:
        """Dismiss the permission dialog on all connected browser addons."""
        dismiss = BrowserPermissionDismiss(request_id=request_id)
        for conn in self._connections.values():
            await self._send_ws(conn.ws, dismiss)

    async def handle_domain_permissions_changed(self) -> None:
        """Sync the full domain permissions list to all connected addons."""
        await self._sync_domain_permissions()

    def _handle_tool_response(self, data: dict) -> None:
        """Resolve a pending tool request future."""
        try:
            response = BrowserToolResponse(**data)
        except Exception:
            logger.warning("Invalid tool response: %s", str(data)[:200])
            return

        future = self._pending_requests.pop(response.request_id, None)
        if not future or future.done():
            logger.warning("No pending request for id: %s", response.request_id)
            return

        logger.debug(
            "Tool response: result=%d chars, image=%s",
            len(response.result or ""),
            f"{len(response.image)} chars" if response.image else "none",
        )
        if response.error:
            future.set_exception(RuntimeError(response.error))
        else:
            future.set_result((response.result or "", response.image))

    _PROMPT_LOG_PAGE_SIZE = 50

    async def _handle_prompt_logs_request(self, ws: ServerConnection, data: dict) -> None:
        """Query prompt logs grouped by run_id and send them to the browser."""
        agent_name = data.get("agent_name") or None
        offset = int(data.get("offset", 0))
        query = (data.get("query") or "").strip() or None
        runs = self._db.messages.get_prompt_log_runs(
            limit=self._PROMPT_LOG_PAGE_SIZE, offset=offset, agent_name=agent_name, query=query
        )
        response = {
            "type": BROWSER_RESP_TYPE_PROMPT_LOGS,
            "runs": runs,
            "has_more": len(runs) == self._PROMPT_LOG_PAGE_SIZE,
        }
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(response))

    async def _handle_memories_request(self, ws: ServerConnection, data: dict) -> None:
        """List every memory (collections + logs, archived included) with
        metadata + entry counts for the addon's Memories tab list view.  An
        optional ``query`` keeps memories matching by name / description /
        intent OR holding an entry whose key or content contains the text."""
        memories = self._db.memories.list_all()
        query = (data.get("query") or "").strip()
        if query:
            memories = self._filter_memories(memories, query)
        counts = self._db.memories.entry_counts()
        records = [self._memory_to_record(m, counts.get(m.name, 0)) for m in memories]
        payload = BrowserMemoriesResponse(memories=records)
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(payload.model_dump_json())

    def _filter_memories(self, memories: list, query: str) -> list:
        """Keep memories matching ``query`` by metadata or by entry content."""
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
        """Send one memory's metadata + the first page of each section: its
        entries, and — for collections — the matching ``collector-runs``
        entries rendered inline as collector activity.  Both sections page
        independently via ``_handle_memory_page_request`` so opening a memory
        never loads its whole (potentially multi-thousand-row) history."""
        try:
            req = BrowserMemoryDetailRequest(**data)
        except ValidationError:
            logger.warning("Invalid memory_detail_request: %s", str(data)[:200])
            return
        memory = self._db.memories.get(req.name)
        if memory is None:
            logger.warning("memory_detail_request for unknown memory: %s", req.name)
            return
        payload = self._build_memory_detail(memory, data)
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(payload.model_dump_json())

    def _build_memory_detail(self, memory, data: dict) -> BrowserMemoryDetailResponse:
        """Assemble the detail payload: metadata + entry count + the first page
        of each section (entries, collector runs) + the collection's cursors."""
        counts = self._db.memories.entry_counts()
        query = (data.get("query") or "").strip() or None
        record = self._memory_to_record(memory, counts.get(memory.name, 0))
        entries, entries_has_more = self._memory_section_page(
            memory, MEMORY_SECTION_ENTRIES, 0, query
        )
        runs, runs_has_more = self._memory_section_page(memory, MEMORY_SECTION_COLLECTOR_RUNS, 0)
        return BrowserMemoryDetailResponse(
            memory=record,
            entries=entries,
            entries_has_more=entries_has_more,
            collector_runs=runs,
            collector_runs_has_more=runs_has_more,
            cursors=self._cursors_for(memory),
        )

    async def _handle_memory_page_request(self, ws: ServerConnection, data: dict) -> None:
        """Send one more page of a single memory-detail section (entries or
        collector runs), advancing past the rows the addon already holds."""
        try:
            req = BrowserMemoryPageRequest(**data)
        except ValidationError:
            logger.warning("Invalid memory_page_request: %s", str(data)[:200])
            return
        memory = self._db.memories.get(req.name)
        if memory is None:
            logger.warning("memory_page_request for unknown memory: %s", req.name)
            return
        query = (data.get("query") or "").strip() or None
        entries, has_more = self._memory_section_page(memory, req.section, req.offset, query)
        payload = BrowserMemoryPageResponse(
            name=req.name, section=req.section, entries=entries, has_more=has_more
        )
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(payload.model_dump_json())

    async def _handle_collection_trigger(self, ws: ServerConnection, data: dict) -> None:
        """Run a collection's extractor on demand and report the outcome back
        to the requesting addon.  ``run_for`` validates the target and is
        serialized against the background cadence by the collector's cycle
        lock; its ``mark_collected`` fans out a ``memory_changed`` event that
        refreshes the detail view's entries + collector activity."""
        try:
            req = BrowserCollectionTrigger(**data)
        except ValidationError:
            logger.warning("Invalid collection_trigger: %s", str(data)[:200])
            return
        if self._collector is None:
            success, message = False, "Collector is not available."
        else:
            success, message = await self._collector.run_for(req.name)
        result = BrowserCollectionTriggerResult(name=req.name, success=success, message=message)
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(result.model_dump_json())

    def _handle_cursor_set(self, data: dict) -> None:
        """Set a collection's read cursor over one log to a chosen point — a
        user override (``set_position``) that may move backward to re-read."""
        try:
            req = BrowserCursorSet(**data)
            last_read_at = datetime.fromisoformat(req.last_read_at)
        except ValidationError, ValueError:
            logger.warning("Invalid cursor_set: %s", str(data)[:200])
            return
        self._db.cursors.set_position(req.name, req.log_name, last_read_at)
        self._on_memory_changed(req.name)

    def _handle_cursor_clear(self, data: dict) -> None:
        """Clear a collection's read cursor over one log — next cycle reads
        recent entries afresh, not the whole history."""
        try:
            req = BrowserCursorClear(**data)
        except ValidationError:
            logger.warning("Invalid cursor_clear: %s", str(data)[:200])
            return
        self._db.cursors.clear(req.name, req.log_name)
        self._on_memory_changed(req.name)

    def _memory_section_page(
        self, memory, section: str, offset: int, query: str | None = None
    ) -> tuple[list[MemoryEntryRecord], bool]:
        """One newest-first page of a memory-detail section.  ``has_more`` is
        true when the page filled the page size, matching the prompts tab.
        ``query`` filters the *entries* section to matching key/content so the
        detail view mirrors the Memories-list search; collector runs are
        activity, not search results, so they're never filtered."""
        if section == MEMORY_SECTION_COLLECTOR_RUNS:
            rows = self._collector_runs_for(memory, self._MEMORY_PAGE_SIZE, offset)
        elif memory.name == PennyConstants.MEMORY_COLLECTOR_RUNS_LOG:
            # The collector-runs log is itself a facade over promptlog — its
            # "entries" are runs (every collection's), not stored rows.
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

    def _collector_runs_for(self, memory, limit: int, offset: int) -> list:
        """Newest-first ``collector-runs`` for this collection.

        ``collector-runs`` is a read facade over ``promptlog`` (no stored
        entries), so the runs come from the ``RunLog`` scoped to this
        collection's ``run_target``, rendered as records.  Empty for logs
        (collectors only target collections)."""
        if memory.type != "collection":
            return []
        run_log = self._db.memories.run_log(target=memory.name)
        return run_log.newest_entries(limit, offset) if run_log is not None else []

    def _cursors_for(self, memory) -> list[CursorRecord]:
        """The collection's read positions over the logs it reads, oldest log
        name first for stable display.  Empty for logs (which aren't readers)."""
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

    # ── Memory edits (refresh fanned out via _on_memory_changed) ─────────

    # Author tag for entries the user adds manually via the addon —
    # distinguishes addon-authored from collector-authored when reading
    # the entries list.  Matches the convention used elsewhere in the
    # codebase (chat-side ``/like`` writes too land as ``"user"``).
    _ADDON_ENTRY_AUTHOR = "user"

    @staticmethod
    def _parse_routing(
        inclusion: str | None, recall: str | None
    ) -> tuple[Inclusion | None, RecallMode | None] | None:
        """Resolve a browser-supplied (inclusion, recall) pair, or None if invalid.

        Translates the legacy single-flag ``recall='off'`` to the new
        ``inclusion=never`` + ``recall=recent`` split.  Unset values stay
        ``None`` so the update path applies only what changed; the create path
        fills its own defaults for any ``None``.
        """
        if recall == "off":
            return Inclusion.NEVER, RecallMode.RECENT
        try:
            parsed_inclusion = Inclusion(inclusion) if inclusion is not None else None
            parsed_recall = RecallMode(recall) if recall is not None else None
        except ValueError:
            return None
        return parsed_inclusion, parsed_recall

    async def _handle_memory_create(self, data: dict) -> None:
        """Create a new collection from the addon.  Logs are seeded by
        migrations and not user-creatable here."""
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
            )
        except MemoryAlreadyExistsError:
            logger.warning("memory_create with duplicate name: %s", req.name)

    async def _handle_memory_update(self, data: dict) -> None:
        """Edit metadata on an existing collection.  Only fields that are
        not ``None`` are applied, matching ``update_collection_metadata``."""
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
        # Re-embed the routing anchor whenever the description changes.
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
            )
        except (MemoryNotFoundError, MemoryTypeError) as exc:
            logger.warning("memory_update failed for %s: %s", req.name, exc)

    def _handle_memory_archive(self, data: dict) -> None:
        """Soft-delete a memory from the active list."""
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
        """Manually add a single entry to a collection (bypasses the
        collector — useful when the user wants to record something the
        auto-extractor missed).  Dedup still runs; duplicates are silently
        dropped — the addon will see the existing entry on refresh."""
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
                author=self._ADDON_ENTRY_AUTHOR,
            )
        except MemoryTypeError as exc:
            logger.warning("entry_create on non-collection %s: %s", req.memory, exc)

    def _handle_entry_update(self, data: dict) -> None:
        """Replace the content of an existing keyed entry."""
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
            memory.update(req.key, req.content, author=self._ADDON_ENTRY_AUTHOR)
        except MemoryTypeError as exc:
            logger.warning("entry_update on non-collection %s: %s", req.memory, exc)

    def _handle_entry_delete(self, data: dict) -> None:
        """Delete a keyed entry from a collection."""
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
        response = {"type": BROWSER_RESP_TYPE_CONFIG, "params": params}
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(response))

    async def _handle_config_update(self, ws: ServerConnection, data: dict) -> None:
        """Validate and persist a single config param update."""
        try:
            req = BrowserConfigUpdate(**data)
        except Exception:
            logger.warning("Invalid config_update: %s", str(data)[:200])
            return
        param = RUNTIME_CONFIG_PARAMS.get(req.key)
        if not param:
            logger.warning("Unknown config key: %s", req.key)
            return
        try:
            validated = param.validator(req.value)
        except ValueError as e:
            logger.warning("Invalid config value %s=%s: %s", req.key, req.value, e)
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
        logger.info("Config updated via browser: %s = %s", req.key, validated)
        await self._handle_config_request(ws)

    # --- Schedule handlers ---

    async def _handle_schedules_request(self, ws: ServerConnection) -> None:
        """Query all schedules for the primary user and send them."""
        await self._send_schedules(ws)

    async def _handle_schedule_add(self, ws: ServerConnection, data: dict) -> None:
        """Parse a natural language schedule command and create it."""
        try:
            req = BrowserScheduleAdd(**data)
        except Exception:
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
            timezone=user_timezone, command=req.command.strip()
        )
        try:
            response = await self._model_client.generate(prompt=prompt, format="json")
            result = ScheduleParseResult.model_validate_json(response.message.content)
        except Exception as e:
            logger.warning("Failed to parse schedule from browser: %s", e)
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

        logger.info(
            "Schedule added via browser: %s — %s", result.timing_description, result.prompt_text
        )
        await self._send_schedules(ws)

    async def _handle_schedule_update(self, ws: ServerConnection, data: dict) -> None:
        """Update a schedule's prompt text."""
        try:
            req = BrowserScheduleUpdate(**data)
        except Exception:
            logger.warning("Invalid schedule_update: %s", str(data)[:200])
            return

        with Session(self._db.engine) as session:
            schedule = session.get(Schedule, req.schedule_id)
            if schedule:
                schedule.prompt_text = req.prompt_text
                session.add(schedule)
                session.commit()
                logger.info("Schedule %d updated via browser", req.schedule_id)

        await self._send_schedules(ws)

    async def _handle_schedule_delete(self, ws: ServerConnection, data: dict) -> None:
        """Delete a schedule by ID."""
        try:
            req = BrowserScheduleDelete(**data)
        except Exception:
            logger.warning("Invalid schedule_delete: %s", str(data)[:200])
            return

        with Session(self._db.engine) as session:
            schedule = session.get(Schedule, req.schedule_id)
            if schedule:
                session.delete(schedule)
                session.commit()
                logger.info("Schedule %d deleted via browser", req.schedule_id)

        await self._send_schedules(ws)

    async def _send_schedules(self, ws: ServerConnection, error: str | None = None) -> None:
        """Send all schedules for the primary user."""
        primary = self._db.users.get_primary_sender()
        schedules: list[ScheduleRecord] = []
        if primary:
            with Session(self._db.engine) as session:
                rows = list(
                    session.exec(
                        select(Schedule)
                        .where(Schedule.user_id == primary)
                        .order_by(Schedule.created_at)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    )
                )
                schedules = [
                    ScheduleRecord(
                        id=s.id,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                        timing_description=s.timing_description,
                        prompt_text=s.prompt_text,
                        cron_expression=s.cron_expression,
                    )
                    for s in rows
                ]

        response = {
            "type": BROWSER_RESP_TYPE_SCHEDULES,
            "schedules": [s.model_dump() for s in schedules],
            "error": error,
        }
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(response))

    async def _handle_chat_message(
        self, ws: ServerConnection, data: dict, device_label: str | None
    ) -> str | None:
        """Process a chat message from the browser."""
        try:
            msg = BrowserIncoming(**data)
        except Exception:
            logger.warning("Invalid chat message: %s", str(data)[:200])
            return device_label

        if not msg.content.strip():
            return device_label

        device_label = msg.sender or "browser-user"
        existing = self._connections.get(device_label)
        if existing:
            existing.ws = ws
        else:
            self._connections[device_label] = ConnectionInfo(ws=ws)
        self._auto_register_device(device_label)

        envelope: dict = {"browser_sender": device_label, "content": msg.content}
        if msg.page_context and msg.page_context.text:
            envelope["page_context"] = PageContext(
                title=msg.page_context.title,
                url=msg.page_context.url,
                text=msg.page_context.text,
            )
        asyncio.create_task(self.handle_message(envelope))
        return device_label

    # --- Tool requests ---

    async def send_tool_request(
        self,
        tool: str,
        arguments: dict,
    ) -> tuple[str, str | None]:
        """Send a tool request to a connected browser and await the response.

        The per-request timeout is owned by the caller: the browse tool wraps
        each call in ``asyncio.wait_for(BROWSE_REQUEST_TIMEOUT)`` and drives the
        retry/backoff loop.  This transport simply delivers the request and
        awaits its response future, dropping the pending entry on completion
        *or* cancellation (when the caller's timeout fires).  A second timeout
        here would only ever be the longer, losing one — and a response landing
        in the gap between the two would be discarded as "No pending request".
        Returns (result_text, image_url).
        """
        ws = self._get_tool_connection()
        if ws is None:
            raise RuntimeError("No browser with tool-use enabled is connected")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[tuple[str, str | None]] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        request = BrowserToolRequest(
            request_id=request_id,
            tool=tool,
            arguments=arguments,
        )
        logger.debug("Sending browser tool request %s (tool=%s)", request_id, tool)
        await self._send_ws(ws, request)

        try:
            return await future
        finally:
            self._pending_requests.pop(request_id, None)

    def _get_tool_connection(self) -> ServerConnection | None:
        """Get the best browser connection for tool execution.

        Among tool-use-enabled connections, prefer those still heartbeating
        (routing around a suspended socket whose JS has stopped servicing
        requests) and pick the most recent.  If none are fresh, fall back to the
        most-recently-seen connection rather than refusing — a lone quiet socket
        is still worth trying (it may just be an addon without the keepalive).
        """
        tool_conns = [c for c in self._connections.values() if c.tool_use_enabled]
        if not tool_conns:
            return None
        fresh = [c for c in tool_conns if self._has_fresh_heartbeat(c)]
        pool = fresh or tool_conns
        return max(pool, key=lambda c: c.last_heartbeat).ws

    # --- Device registration ---

    def _auto_register_device(self, device_label: str) -> None:
        """Register the browser device if not already known."""
        self._db.devices.register(
            channel_type=ChannelType.BROWSER,
            identifier=device_label,
            label=device_label,
        )

    # --- MessageChannel interface ---

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract a message from browser WebSocket data."""
        sender = raw_data.get("browser_sender", "browser-user")
        content = raw_data.get("content", "").strip()
        if not content:
            return None
        return IncomingMessage(
            sender=sender,
            content=content,
            channel_type=ChannelType.BROWSER,
            device_identifier=sender,
            page_context=raw_data.get("page_context"),
        )

    async def send_message(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """Send a message to a browser client by device label."""
        conn = self._connections.get(recipient)
        if not conn:
            logger.warning("No browser connection for device: %s", recipient)
            return None
        content = self._prepend_images(message, attachments)
        await self._send_ws(
            conn.ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_MESSAGE, content=content)
        )
        return 1

    @staticmethod
    def _prepend_images(message: str, attachments: list[str] | None) -> str:
        """Prepend image attachments as <img> tags before the message HTML."""
        if not attachments:
            return message
        tags: list[str] = []
        for att in attachments:
            src = _attachment_to_src(att)
            if src:
                tags.append(f'<img src="{src}" alt="image"><br>')
        return f"{''.join(tags)}{message}" if tags else message

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator to a browser client."""
        conn = self._connections.get(recipient)
        if not conn:
            return False
        await self._send_ws(conn.ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_TYPING, active=typing))
        return True

    def _make_handle_kwargs(
        self, message: IncomingMessage, progress: ProgressTracker | None = None
    ) -> dict:
        """Pass an on_tool_start callback so tool calls update the typing indicator.

        Builds a cumulative checklist: prior steps show as completed (checkmark),
        current step shows as in-progress (dots). The browser channel renders
        progress directly into the typing indicator HTML rather than going
        through a ``ProgressTracker``, so ``progress`` is intentionally unused
        — ``_begin_progress`` returns ``None`` for this channel.
        """
        recipient = message.sender
        completed: list[str] = []

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            current = [self._format_tool_status(name, args) for name, args in tools]
            lines: list[str] = []
            for item in completed:
                lines.append(f"&#x2713; {item}")
            for item in current:
                lines.append(item)
            await self._send_tool_status(recipient, "<br>".join(lines))
            completed.extend(current)

        return {"on_tool_start": on_tool_start}

    @staticmethod
    def _format_tool_status(tool_name: str, arguments: dict) -> str:
        """Format a human-readable status label for a tool call."""
        return Tool.format_status(tool_name, arguments)

    async def _send_tool_status(self, recipient: str, text: str) -> None:
        """Update the typing indicator with a tool status message."""
        conn = self._connections.get(recipient)
        if not conn:
            return
        await self._send_ws(
            conn.ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_TYPING, active=True, content=text)
        )

    def make_background_tool_callback(
        self,
    ) -> tuple[
        Callable[[list[tuple[str, dict]]], Awaitable[None]],
        Callable[[], Awaitable[None]],
    ]:
        """Create an on_tool_start callback and cleanup for background agents.

        Sends tool status to the addon that would handle tool requests
        (the connection returned by _get_tool_connection).
        Returns (on_tool_start, cleanup) — call cleanup after the run to clear the indicator.
        """
        completed: list[str] = []

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            ws = self._get_tool_connection()
            if not ws:
                return
            current = [self._format_tool_status(name, args) for name, args in tools]
            lines: list[str] = []
            for item in completed:
                lines.append(f"&#x2713; {item}")
            for item in current:
                lines.append(item)
            await self._send_ws(
                ws,
                BrowserOutgoing(
                    type=BROWSER_RESP_TYPE_TYPING, active=True, content="<br>".join(lines)
                ),
            )
            completed.extend(current)

        async def cleanup() -> None:
            if not completed:
                return
            ws = self._get_tool_connection()
            if ws:
                await self._send_ws(
                    ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_TYPING, active=False)
                )

        return on_tool_start, cleanup

    # --- Markdown to HTML formatting ---

    _TABLE_PATTERN = re.compile(
        r"^(\|[^\n]+\|)\n"
        r"(\|[-:\s|]+\|)\n"
        r"((?:\|[^\n]+\|\n?)+)",
        re.MULTILINE,
    )

    def prepare_outgoing(self, text: str) -> str:
        """Convert markdown to HTML for the browser sidebar."""
        text = self._table_to_bullets(text)
        text = html.escape(text)
        text = self._convert_markdown_to_html(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    @classmethod
    def _table_to_bullets(cls, text: str) -> str:
        """Convert markdown tables to bullet points (same as Signal)."""

        def convert_table(match: re.Match[str]) -> str:
            header_line, _, data_block = match.groups()
            headers = [c.strip() for c in header_line.strip("|").split("|")]
            result = []
            for line in data_block.strip().split("\n"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells and cells[0]:
                    title = cells[0].strip("*").strip()
                    result.append(f"**{title}**")
                    result.extend(
                        f"  \u2022 **{h}**: {c}"
                        for h, c in zip(headers[1:], cells[1:], strict=False)
                        if c
                    )
                    result.append("")
            return "\n".join(result)

        return cls._TABLE_PATTERN.sub(convert_table, text)

    @staticmethod
    def _convert_markdown_to_html(text: str) -> str:
        """Convert markdown formatting to HTML tags (text is already escaped)."""
        text = re.sub(r"```([\s\S]*?)```", r"<pre><code>\1</code></pre>", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
        text = re.sub(r"^#{1,6}\s+(.+)$", r"<strong>\1</strong>", text, flags=re.MULTILINE)
        text = re.sub(r"^-{3,}\s*$", "<hr>", text, flags=re.MULTILINE)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', text)
        text = re.sub(r"(https?://[^\s<>&]+)", r'<a href="\1" target="_blank">\1</a>', text)
        text = text.replace("\n", "<br>")
        return text

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Collapse multiple consecutive <br> tags."""
        return re.sub(r"(<br>){3,}", "<br><br>", text)

    # --- Connection management ---

    async def close(self) -> None:
        """Shut down the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Browser channel closed")

    @staticmethod
    async def _send_ws(ws: ServerConnection, msg: BaseModel) -> None:
        """Send a message to a WebSocket connection, suppressing closed errors."""
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg.model_dump_json(exclude_none=True))


# Backward compat alias
BrowserServer = BrowserChannel
