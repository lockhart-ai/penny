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
from sqlmodel import Session
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
    BROWSER_MSG_TYPE_TOOL_RESPONSE,
    BROWSER_RESP_TYPE_CONFIG,
    BROWSER_RESP_TYPE_MEMORY_CHANGED,
    BROWSER_RESP_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE,
    BROWSER_RESP_TYPE_PROMPT_LOGS,
    BROWSER_RESP_TYPE_RUN_OUTCOME,
    BROWSER_RESP_TYPE_STATUS,
    BROWSER_RESP_TYPE_TYPING,
    MEMORY_SECTION_COLLECTOR_RUNS,
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
    BrowserToolRequest,
    BrowserToolResponse,
    CursorRecord,
    DomainPermissionRecord,
    MemoryEntryRecord,
    MemoryRecord,
)
from penny.channels.permission_manager import PermissionManager
from penny.config_params import RUNTIME_CONFIG_PARAMS, get_params_by_group
from penny.constants import ChannelType, PennyConstants, PermissionResolution
from penny.database.memory import (
    EntryInput,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryTypeError,
)
from penny.database.models import RuntimeConfig
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
    """Metadata about a connected browser extension.

    ``registered`` records that the addon's BACKGROUND SCRIPT announced itself on
    this socket — the half of the extension that owns the websocket and services
    tool requests.  A registry entry can exist without it (the sidebar's own chat
    message mints one), and such a socket has never claimed it can answer a tool
    request, so routing ranks it last.
    """

    ws: ServerConnection
    tool_use_enabled: bool = False
    registered: bool = False
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PendingToolRequest:
    """A tool request awaiting its answer, and the socket it was sent on.

    The socket is part of the record because a flush must reject only what THIS
    connection owes.  Rejecting every pending future on any disconnect killed
    in-flight requests belonging to connections that were still perfectly alive.
    """

    future: asyncio.Future[tuple[str, str | None]]
    ws: ServerConnection


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
        self._pending_requests: dict[str, PendingToolRequest] = {}
        self._liveness_task: asyncio.Task | None = None
        self._permission_manager: PermissionManager | None = None
        self._collector: Collector | None = None
        db.messages._on_prompt_logged = self._on_prompt_logged
        db.messages._on_run_outcome_set = self._on_run_outcome_set
        db.memories._on_memory_changed = self._on_memory_changed

    def _on_prompt_logged(self, prompt_data: dict) -> None:
        """Callback fired after each prompt is logged — broadcast to browsers."""
        message = json.dumps({"type": BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE, "prompt": prompt_data})
        self._push(message, BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE)

    def _on_run_outcome_set(self, run_id: str, outcome: str, reason: str) -> None:
        """Callback fired when a run outcome is set — broadcast to browsers."""
        payload = BrowserRunOutcomeUpdate(run_id=run_id, outcome=outcome, reason=reason)
        self._push(payload.model_dump_json(), BROWSER_RESP_TYPE_RUN_OUTCOME)

    def _on_memory_changed(self, name: str | None) -> None:
        """Callback fired after any memory mutation — broadcast to browsers
        so the Memories tab can refresh.  ``name`` is the affected memory
        when the change is scoped to one (writes, archives, metadata edits);
        ``None`` for fan-out events."""
        message = BrowserMemoryChanged(name=name).model_dump_json()
        self._push(message, BROWSER_RESP_TYPE_MEMORY_CHANGED)

    def _push(self, payload: str, update: str) -> None:
        """Schedule a push notification's fan-out to every connected addon.

        These three callbacks are SYNCHRONOUS — the database stores call them from
        ordinary code the moment a row lands — so the fan-out can only be scheduled,
        never awaited.  What is scheduled therefore has to handle its own failures:
        a detached task is the one place an exception has nobody to raise to, and
        ``asyncio.ensure_future(ws.send(...))`` per socket meant an addon window
        closed a moment earlier raised ``ConnectionClosed`` into a task nobody
        awaited — reported later, out of context, as "Task exception was never
        retrieved", and handled nowhere.

        Nothing is scheduled when nobody is connected: a push with no recipient is
        not a reason to require a running event loop (these callbacks also fire on
        the startup backfill path, where there is none).
        """
        if not self._connections:
            return
        asyncio.ensure_future(self._deliver_push(payload, update))

    async def _deliver_push(self, payload: str, update: str) -> None:
        """Deliver one push frame to every connection, and name the ones that are gone.

        Fanned out CONCURRENTLY over a snapshot, each send guarded on its own: a push
        is a notification, so a socket that has closed is one fewer recipient and must
        never cost the remaining windows their update.  The outer guard is broad for
        the reason ``_watch_liveness``'s is — this coroutine runs DETACHED, so anything
        it lets out is the unretrieved-task-exception this whole path exists to stop.
        """
        targets = list(self._connections.items())
        try:
            delivered = await asyncio.gather(
                *(self._deliver_text(conn.ws, payload) for _, conn in targets)
            )
        except Exception:
            logger.exception("Broadcasting a %s update to the browser addons failed", update)
            return
        gone = [label for (label, _), taken in zip(targets, delivered, strict=True) if not taken]
        if gone:
            logger.info(
                "Browser %s closed before the %s push arrived — dropped for %d of %d connection(s)",
                ", ".join(gone),
                update,
                len(gone),
                len(targets),
            )

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
    def has_browser_connection(self) -> bool:
        """Whether any browser addon websocket is connected."""
        return bool(self._connections)

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
        self._liveness_task = asyncio.create_task(self._watch_liveness())
        await asyncio.Future()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single browser extension connection."""
        logger.info("Browser connected")
        await self._send_ws(ws, self._liveness_probe())

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
                device_label or PennyConstants.BROWSER_UNREGISTERED_DEVICE,
            )
        finally:
            logger.info(
                "Browser socket closed (device=%s, code=%s, reason=%r)",
                device_label or PennyConstants.BROWSER_UNREGISTERED_DEVICE,
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
        self._fail_pending_on(ws, device_label)
        logger.info(
            "Browser disconnected: %s", device_label or PennyConstants.BROWSER_UNREGISTERED_DEVICE
        )

    def _fail_pending_on(self, ws: ServerConnection, device_label: str | None) -> None:
        """Reject every tool request still waiting on this socket, and say how many.

        Scoped to ``ws`` on purpose: the flush this replaces rejected EVERY pending
        future, so one addon window closing killed browse requests in flight on a
        connection that was still answering.  The rejection is what re-routes the
        work — the browse tool catches it and retries on whatever connection is
        live, rather than waiting out its own timeout against a socket that can
        never answer.
        """
        device = device_label or PennyConstants.BROWSER_UNREGISTERED_DEVICE
        stranded = [
            request_id for request_id, pending in self._pending_requests.items() if pending.ws is ws
        ]
        for request_id in stranded:
            pending = self._pending_requests.pop(request_id, None)
            if pending is not None and not pending.future.done():
                pending.future.set_exception(ConnectionError(self._stranded_error(device)))
        if stranded:
            logger.info("Failed %d tool request(s) stranded on browser %s", len(stranded), device)

    @staticmethod
    def _stranded_error(device: str) -> str:
        """What a request sent on a socket that turned out to be gone is told."""
        return PennyConstants.BROWSER_STRANDED_REQUEST_ERROR.format(device=device)

    # --- Liveness ---

    @staticmethod
    def _liveness_probe() -> BrowserOutgoing:
        """The frame a quiet connection is probed with.

        Deliberately the connect-time status frame rather than a new message type:
        the addon already answers it by re-registering and re-sending its
        capabilities, so a live-but-quiet socket REPAIRS its registry entry (label,
        capabilities, heartbeat) instead of merely surviving the probe.
        """
        return BrowserOutgoing(type=BROWSER_RESP_TYPE_STATUS, connected=True)

    async def _watch_liveness(self) -> None:
        """Probe quiet connections on a timer, so a dead socket is found before a
        tool request is sent into it.

        Dead-socket detection used to be write-driven: nothing noticed a socket was
        gone until something happened to write to it, so browse requests went into a
        corpse for ~2.4 minutes — four retry rounds, twelve unanswered requests —
        until a write finally failed with code 1006.  A timer puts that on a clock,
        and costs nothing on a healthy addon (which heartbeats every ~15s and is
        therefore never quiet enough to probe).
        """
        while True:
            await asyncio.sleep(PennyConstants.BROWSER_LIVENESS_SWEEP_SECONDS)
            try:
                await self._probe_quiet_connections()
            except Exception:
                # Deliberately broad, and the reason is the loop itself: a sweep
                # that raises would END the task, taking dead-socket detection down
                # for the whole process and silently restoring the write-driven
                # behaviour this exists to replace.  Logged, never swallowed — the
                # same shape (and reason) as ``_handle_connection``'s guard.
                logger.exception("Browser liveness sweep failed — continuing")

    async def _probe_quiet_connections(self) -> None:
        """Probe every connection past its heartbeat window; evict what can't take it.

        Probed CONCURRENTLY and each on its own timeout: the probes are independent,
        and a send awaits flow-control drain, so a half-open peer that has stopped
        reading never raises — it just never returns.  Serially and unbounded, one
        probe of exactly the socket this sweep exists to find would wedge the sweep.
        """
        quiet = [
            (label, conn)
            for label, conn in list(self._connections.items())
            if not self._has_fresh_heartbeat(conn)
        ]
        answered = await asyncio.gather(*(self._probe(conn.ws) for _, conn in quiet))
        for (label, conn), alive in zip(quiet, answered, strict=True):
            if alive:
                continue
            logger.warning(
                "Browser %s stopped heartbeating and its socket is unreachable — closing "
                "it and taking it out of routing",
                label,
            )
            await self._evict_socket(conn.ws)

    async def _probe(self, ws: ServerConnection) -> bool:
        """Whether this socket took the probe within the probe window."""
        try:
            return await asyncio.wait_for(
                self._deliver_ws(ws, self._liveness_probe()),
                timeout=PennyConstants.BROWSER_LIVENESS_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return False

    async def _evict_socket(self, ws: ServerConnection) -> None:
        """Take a dead socket out of routing, fail what it owes, and close it."""
        device_label = self._label_for(ws)
        if device_label is not None:
            self._connections.pop(device_label, None)
        self._fail_pending_on(ws, device_label)
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.close()

    def _label_for(self, ws: ServerConnection) -> str | None:
        """The device label this socket currently holds in the registry."""
        return next(
            (label for label, conn in self._connections.items() if conn.ws is ws),
            None,
        )

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
            return device_label

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
            existing.registered = True
            # A reconnect re-points the entry at the new socket — refresh
            # liveness so the freshly reconnected addon isn't judged stale by
            # ``_get_tool_connection`` on its old (pre-suspension) timestamp.
            existing.last_heartbeat = datetime.now(UTC)
        else:
            self._connections[device_label] = ConnectionInfo(ws=ws, registered=True)
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

    async def handle_permission_dismiss(
        self, request_id: str, resolution: PermissionResolution
    ) -> None:
        """Dismiss the permission dialog on all connected browser addons.

        The addon's prompt is a UI popup, so resolving it means closing the
        popup — there is no message to mark, and nothing for ``resolution`` to
        say here (the answer the user gave is already reflected in the domain
        permissions the addon is synced with).
        """
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

        pending = self._pending_requests.pop(response.request_id, None)
        if pending is None or pending.future.done():
            logger.warning("No pending request for id: %s", response.request_id)
            return

        logger.debug(
            "Tool response: result=%d chars, image=%s",
            len(response.result or ""),
            f"{len(response.image)} chars" if response.image else "none",
        )
        if response.error:
            pending.future.set_exception(RuntimeError(response.error))
        else:
            pending.future.set_result((response.result or "", response.image))

    _PROMPT_LOG_PAGE_SIZE = 50

    async def _handle_prompt_logs_request(self, ws: ServerConnection, data: dict) -> None:
        """Query prompt logs grouped by run_id and send them to the browser."""
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
        response = {
            "type": BROWSER_RESP_TYPE_PROMPT_LOGS,
            "runs": runs,
            # Flagged-only is a single-shot view of the recent-runs window — it
            # returns every flagged run at once, so there's no further page.
            "has_more": (not flagged_only) and len(runs) == self._PROMPT_LOG_PAGE_SIZE,
        }
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(response))

    async def _handle_memories_request(self, ws: ServerConnection, data: dict) -> None:
        """List every memory (collections + logs, archived included) with
        metadata + entry counts for the addon's Memories tab list view.  An
        optional ``query`` keeps memories matching by name / description
        OR holding an entry whose key or content contains the text."""
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
        payload = self._memory_page_payload(memory, req, data)
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(payload.model_dump_json())

    def _memory_page_payload(
        self, memory, req: BrowserMemoryPageRequest, data: dict
    ) -> BrowserMemoryPageResponse:
        """One page of the requested section: collector runs serialize as full
        runs (run → prompts → turns), entries as entry records."""
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
        """Run a collection's extractor on demand and report the outcome back
        to the requesting addon.  ``run_for`` validates the target and is
        serialized against the background cadence by the collector's cycle
        lock; every cycle it runs ends by fanning out a ``memory_changed`` event
        that refreshes the detail view's entries + collector activity — the stamp
        when the cycle spent its occurrence, the collector itself when it kept it
        for a retry (#1935)."""
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
        await self._deliver_trigger_result(ws, result)

    async def _deliver_trigger_result(
        self, ws: ServerConnection, result: BrowserCollectionTriggerResult
    ) -> None:
        """Get an on-demand run's outcome back to the addon, and never silently.

        A cycle takes minutes and the socket that asked for it can die inside that
        window (a suspended background script, a closed sidebar).  The send used to sit
        under ``contextlib.suppress(ConnectionClosed)``, so a dead requester swallowed
        the outcome whole: measured live, the addon's spinner never resolved and an
        18-minute run read as an indefinite hang, with nothing in the log to say why.

        So the outcome is delivered somewhere or SAID somewhere: the requesting socket
        first, then every registered addon connection (the same user, another window),
        and failing that a warning carrying the outcome text itself, so it survives in
        the run record's own log rather than nowhere.
        """
        if await self._deliver_ws(ws, result):
            return
        logger.warning(
            "The socket that asked to run '%s' closed before its result arrived — "
            "falling back to the registered addon connections",
            result.name,
        )
        delivered = await self._broadcast_trigger_result(result)
        if delivered:
            logger.warning(
                "Delivered the '%s' run result to %s instead", result.name, ", ".join(delivered)
            )
            return
        logger.warning(
            "No live addon connection could take the '%s' run result — it is recorded "
            "here only: success=%s, %s",
            result.name,
            result.success,
            result.message,
        )

    async def _broadcast_trigger_result(self, result: BrowserCollectionTriggerResult) -> list[str]:
        """The registered connections that took the result, by device label.

        Every one of them, not the first that answers: an addon may be open in more
        than one window and they all show the same collection's spinner.  Iterated over
        a snapshot because a closed socket is evicted from the registry by its own
        handler while this runs.
        """
        return [
            label
            for label, conn in list(self._connections.items())
            if await self._deliver_ws(conn.ws, result)
        ]

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

    def _entries_page(
        self, memory, offset: int, query: str | None = None
    ) -> tuple[list[MemoryEntryRecord], bool]:
        """One newest-first page of a memory's entries.  ``has_more`` is true
        when the page filled the page size, matching the prompts tab.  ``query``
        filters to matching key/content so the detail view mirrors the
        Memories-list search."""
        if memory.name == PennyConstants.MEMORY_COLLECTOR_RUNS_LOG:
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

    def _collector_runs_page(self, memory, offset: int) -> tuple[list[dict], bool]:
        """One newest-first page of this collection's collector runs as full
        serialized runs (run → prompts → turns), so the Activity tab renders the
        same cards as the prompts tab.  Empty for logs (collectors only target
        collections)."""
        if memory.type != "collection":
            return [], False
        runs = self._db.messages.get_target_runs(memory.name, self._MEMORY_PAGE_SIZE, offset)
        return runs, len(runs) == self._MEMORY_PAGE_SIZE

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
            published=memory.notify,  # wire field `published` ← the `notify` column (#1557)
            archived=memory.archived,
            extraction_prompt=memory.extraction_prompt,
            schedule=memory.schedule,
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
    # codebase (user-directed writes land as ``"user"``).
    _ADDON_ENTRY_AUTHOR = "user"

    async def _handle_memory_create(self, data: dict) -> None:
        """Create a new collection from the addon.  Logs are seeded by
        migrations and not user-creatable here."""
        try:
            req = BrowserMemoryCreate(**data)
        except ValidationError:
            logger.warning("Invalid memory_create: %s", str(data)[:200])
            return
        description_embedding = await self._message_agent.embed_description(req.description)
        try:
            self._db.memories.create_collection(
                req.name,
                req.description,
                extraction_prompt=req.extraction_prompt,
                schedule=req.schedule,
                description_embedding=description_embedding,
                notify=req.published,  # wire field `published` → the `notify` column (#1557)
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
        # Re-embed the meaning anchor whenever the description changes.
        description_embedding = (
            await self._message_agent.embed_description(req.description)
            if req.description is not None
            else None
        )
        try:
            self._db.memories.update_collection_metadata(
                req.name,
                description=req.description,
                extraction_prompt=req.extraction_prompt,
                schedule=req.schedule,
                description_embedding=description_embedding,
                notify=req.published,  # wire field `published` → the `notify` column (#1557)
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
            if existing.ws is not ws:
                # A different socket now carries this label and has not announced
                # itself on its own account — until it registers, routing must not
                # credit it with the previous socket's claim to service tool calls.
                existing.registered = False
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

        A send that FAILS is the one thing not left to the caller's clock: the
        socket is gone, so no answer is coming, and waiting the timeout out only
        delays the retry.  The request is failed immediately and the socket is
        evicted.  Returns (result_text, image_url).
        """
        ws = self._get_tool_connection()
        if ws is None:
            if self._connections:
                raise RuntimeError(
                    "browser is connected, but tool use is disabled — enable Tool use in the "
                    "browser extension to let collectors search and read pages"
                )
            raise RuntimeError("No browser with tool-use enabled is connected")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[tuple[str, str | None]] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = PendingToolRequest(future=future, ws=ws)

        request = BrowserToolRequest(
            request_id=request_id,
            tool=tool,
            arguments=arguments,
        )
        logger.debug("Sending browser tool request %s (tool=%s)", request_id, tool)
        if not await self._deliver_ws(ws, request):
            raise ConnectionError(await self._abandon_send(ws, request_id))

        try:
            return await future
        finally:
            self._pending_requests.pop(request_id, None)

    async def _abandon_send(self, ws: ServerConnection, request_id: str) -> str:
        """Give up on a request whose send failed, and say why it was given up on.

        The pending entry is dropped BEFORE the eviction flush so that flush doesn't
        set an exception nobody will ever retrieve — the caller's raise is this
        request's answer.
        """
        self._pending_requests.pop(request_id, None)
        device = self._label_for(ws) or PennyConstants.BROWSER_UNREGISTERED_DEVICE
        await self._evict_socket(ws)
        return self._stranded_error(device)

    def _get_tool_connection(self) -> ServerConnection | None:
        """Get the best browser connection for tool execution.

        Among tool-use-enabled connections, rank by what each fact says about the
        socket's ability to answer: REGISTERED first, then still heartbeating, then
        most recently seen.  Ranking rather than filtering is what keeps a lone
        quiet socket routable — it is deprioritized whenever a better one exists,
        and never declared offline on its own (it may just be an addon without the
        keepalive).

        Registration outranks the heartbeat because they answer different questions.
        A quiet REGISTERED socket has told us its background script is there and may
        simply be suspended; an unregistered one has never claimed it can service a
        tool request at all, so freshness on it is evidence of nothing.  That
        ordering is only safe because a corpse no longer lingers: the liveness sweep
        closes and evicts an unreachable socket, where before nothing removed one
        and recency was the only proxy available for "still there".
        """
        tool_conns = [c for c in self._connections.values() if c.tool_use_enabled]
        if not tool_conns:
            return None
        return max(tool_conns, key=self._routing_rank).ws

    def _routing_rank(self, conn: ConnectionInfo) -> tuple[bool, bool, datetime]:
        """How good a tool-request target this connection is — best compares highest."""
        return (conn.registered, self._has_fresh_heartbeat(conn), conn.last_heartbeat)

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

    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        source_name: str | None = None,
        message_log_id: int | None = None,
    ) -> int | None:
        """Deliver a prepared message to a browser client by device label.

        Logging happens in the base ``_log_and_send`` chokepoint before this
        is called.
        """
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
        """Shut down the WebSocket server and its liveness sweep."""
        if self._liveness_task is not None:
            self._liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._liveness_task
            self._liveness_task = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Browser channel closed")

    @staticmethod
    async def _send_ws(ws: ServerConnection, msg: BaseModel) -> None:
        """Send a message to a WebSocket connection, suppressing closed errors.

        The fire-and-forget form, for the broadcasts and responses where a closed
        socket is simply one fewer recipient.  Where the send is the ONLY carrier of
        something the user asked for, use ``_deliver_ws`` and act on the answer."""
        await BrowserChannel._deliver_ws(ws, msg)

    @staticmethod
    async def _deliver_ws(ws: ServerConnection, msg: BaseModel) -> bool:
        """Send to one socket; ``False`` when that socket is closed.

        The same send, with the outcome returned instead of discarded — so a caller
        that owes the user an answer can fall back or say so (#1939) rather than
        losing it to a suppressed exception."""
        return await BrowserChannel._deliver_text(ws, msg.model_dump_json(exclude_none=True))

    @staticmethod
    async def _deliver_text(ws: ServerConnection, payload: str) -> bool:
        """Send one already-serialized frame; ``False`` when that socket is closed.

        The single place a send's ``ConnectionClosed`` is caught, so every guarded
        path answers a dead socket the same way.  It takes TEXT rather than a model
        because the push broadcasts serialize their own frames — a prompt-log update
        carries a payload dict the addon renders verbatim, and the two model-shaped
        ones go out without ``exclude_none`` — and routing a notification through a
        guard must not reshape it on the wire.
        """
        try:
            await ws.send(payload)
        except websockets.ConnectionClosed:
            return False
        return True


# Backward compat alias
BrowserServer = BrowserChannel
