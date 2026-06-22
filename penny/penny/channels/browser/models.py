"""Pydantic models for browser extension WebSocket protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from penny.channels.base import PageContext

# Incoming message types (browser → server)
BROWSER_MSG_TYPE_MESSAGE = "message"
BROWSER_MSG_TYPE_TOOL_RESPONSE = "tool_response"
BROWSER_MSG_TYPE_HEARTBEAT = "heartbeat"
BROWSER_MSG_TYPE_CONFIG_REQUEST = "config_request"
BROWSER_MSG_TYPE_CONFIG_UPDATE = "config_update"
BROWSER_MSG_TYPE_REGISTER = "register"
BROWSER_MSG_TYPE_CAPABILITIES_UPDATE = "capabilities_update"
BROWSER_MSG_TYPE_DOMAIN_UPDATE = "domain_update"
BROWSER_MSG_TYPE_DOMAIN_DELETE = "domain_delete"
BROWSER_MSG_TYPE_PERMISSION_REQUEST = "permission_request"
BROWSER_MSG_TYPE_PERMISSION_DECISION = "permission_decision"
BROWSER_MSG_TYPE_SCHEDULES_REQUEST = "schedules_request"
BROWSER_MSG_TYPE_SCHEDULE_ADD = "schedule_add"
BROWSER_MSG_TYPE_SCHEDULE_UPDATE = "schedule_update"
BROWSER_MSG_TYPE_SCHEDULE_DELETE = "schedule_delete"
BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST = "prompt_logs_request"
BROWSER_MSG_TYPE_MEMORIES_REQUEST = "memories_request"
BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST = "memory_detail_request"
BROWSER_MSG_TYPE_MEMORY_PAGE_REQUEST = "memory_page_request"
BROWSER_MSG_TYPE_MEMORY_CREATE = "memory_create"
BROWSER_MSG_TYPE_MEMORY_UPDATE = "memory_update"
BROWSER_MSG_TYPE_MEMORY_ARCHIVE = "memory_archive"
BROWSER_MSG_TYPE_ENTRY_CREATE = "entry_create"
BROWSER_MSG_TYPE_ENTRY_UPDATE = "entry_update"
BROWSER_MSG_TYPE_ENTRY_DELETE = "entry_delete"
BROWSER_MSG_TYPE_COLLECTION_TRIGGER = "collection_trigger"
BROWSER_MSG_TYPE_CURSOR_SET = "cursor_set"
BROWSER_MSG_TYPE_CURSOR_CLEAR = "cursor_clear"

# Outgoing message types (server → browser)
BROWSER_RESP_TYPE_MESSAGE = "message"
BROWSER_RESP_TYPE_TYPING = "typing"
BROWSER_RESP_TYPE_STATUS = "status"
BROWSER_RESP_TYPE_TOOL_REQUEST = "tool_request"
BROWSER_RESP_TYPE_CONFIG = "config_response"
BROWSER_RESP_TYPE_DOMAIN_PERMISSIONS = "domain_permissions_sync"
BROWSER_RESP_TYPE_PERMISSION_PROMPT = "permission_prompt"
BROWSER_RESP_TYPE_PERMISSION_DISMISS = "permission_dismiss"
BROWSER_RESP_TYPE_SCHEDULES = "schedules_response"
BROWSER_RESP_TYPE_PROMPT_LOGS = "prompt_logs_response"
BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE = "prompt_log_update"
BROWSER_RESP_TYPE_RUN_OUTCOME = "run_outcome_update"
BROWSER_RESP_TYPE_MEMORIES = "memories_response"
BROWSER_RESP_TYPE_MEMORY_DETAIL = "memory_detail_response"
BROWSER_RESP_TYPE_MEMORY_PAGE = "memory_page_response"
BROWSER_RESP_TYPE_MEMORY_CHANGED = "memory_changed"
BROWSER_RESP_TYPE_COLLECTION_TRIGGER_RESULT = "collection_trigger_result"

# Sections of a memory's detail view that paginate independently.
MEMORY_SECTION_ENTRIES = "entries"
MEMORY_SECTION_COLLECTOR_RUNS = "collector_runs"


class BrowserIncoming(BaseModel):
    """A chat message received from the browser extension."""

    type: str
    content: str
    sender: str
    page_context: PageContext | None = None


class BrowserToolResponse(BaseModel):
    """A tool execution result from the browser extension."""

    type: str
    request_id: str
    result: str | None = None
    error: str | None = None
    image: str | None = None


class BrowserOutgoing(BaseModel):
    """A message sent to the browser extension."""

    type: str
    content: str | None = None
    active: bool | None = None
    connected: bool | None = None


class BrowserToolRequest(BaseModel):
    """A tool execution request sent to the browser extension."""

    type: str = BROWSER_RESP_TYPE_TOOL_REQUEST
    request_id: str
    tool: str
    arguments: dict


class BrowserConfigUpdate(BaseModel):
    """A request to update a runtime config param."""

    type: str
    key: str
    value: str


class BrowserRegister(BaseModel):
    """Addon registers its device label on connect."""

    type: str
    sender: str


class BrowserCapabilitiesUpdate(BaseModel):
    """Addon declares its tool-use capability."""

    type: str
    tool_use_enabled: bool


class BrowserDomainUpdate(BaseModel):
    """A request to add or update a domain permission."""

    type: str
    domain: str
    permission: str


class BrowserDomainDelete(BaseModel):
    """A request to delete a domain permission."""

    type: str
    domain: str


class DomainPermissionRecord(BaseModel):
    """A single domain permission entry for sync payloads."""

    domain: str
    permission: str


class BrowserDomainPermissionsSync(BaseModel):
    """Full domain permissions list sent to all connected addons."""

    type: str = BROWSER_RESP_TYPE_DOMAIN_PERMISSIONS
    permissions: list[DomainPermissionRecord]


class BrowserPermissionRequest(BaseModel):
    """Addon reports it needs a domain permission decision."""

    type: str
    request_id: str
    domain: str
    url: str


class BrowserPermissionDecision(BaseModel):
    """Addon or Signal user decided on a domain permission."""

    type: str
    request_id: str
    allowed: bool


class BrowserPermissionPrompt(BaseModel):
    """Server asks an addon to show a permission dialog."""

    type: str = BROWSER_RESP_TYPE_PERMISSION_PROMPT
    request_id: str
    domain: str
    url: str


class BrowserPermissionDismiss(BaseModel):
    """Server tells addons to close a pending permission dialog."""

    type: str = BROWSER_RESP_TYPE_PERMISSION_DISMISS
    request_id: str


class BrowserScheduleAdd(BaseModel):
    """A request to add a new schedule via natural language."""

    type: str
    command: str


class BrowserScheduleUpdate(BaseModel):
    """A request to update a schedule's prompt text."""

    type: str
    schedule_id: int
    prompt_text: str


class BrowserScheduleDelete(BaseModel):
    """A request to delete a schedule by ID."""

    type: str
    schedule_id: int


class ScheduleRecord(BaseModel):
    """A single schedule entry for response payloads."""

    id: int
    timing_description: str
    prompt_text: str
    cron_expression: str


class BrowserRunOutcomeUpdate(BaseModel):
    """Push notification: a promptlog run's outcome (outcome/reason) was set.
    ``outcome`` is a RunOutcome value: failed | no_work | worked | cancelled.
    (The run's collection is carried by each prompt's ``run_target``.)"""

    type: str = BROWSER_RESP_TYPE_RUN_OUTCOME
    run_id: str
    outcome: str
    reason: str


class BrowserMemoryDetailRequest(BaseModel):
    """A request to load metadata + the first page of each section for a
    single memory."""

    type: str
    name: str


class BrowserMemoryPageRequest(BaseModel):
    """A request for one more page of a memory-detail section (entries or
    collector runs), advancing past ``offset`` already-shown rows."""

    type: str
    name: str
    section: Literal["entries", "collector_runs"]
    offset: int = 0


class MemoryRecord(BaseModel):
    """One memory's metadata for the addon's Memories tab list view."""

    name: str
    type: str  # "collection" | "log"
    description: str
    intent: str | None  # the user's stated goal at creation (editable only here)
    inclusion: str  # "always" | "relevant" | "never" — stage-1 routing
    recall: str  # "all" | "relevant" | "recent" — stage-2 entry rendering
    published: bool  # pub/sub: when true the notifier delivers new entries — orthogonal to recall
    archived: bool
    extraction_prompt: str | None
    collector_interval_seconds: int | None
    last_collected_at: str | None
    entry_count: int


class CursorRecord(BaseModel):
    """One read-cursor a collection holds over a log it reads — its position
    (``last_read_at``, ISO-8601 UTC) in that log."""

    log_name: str
    last_read_at: str


class MemoryEntryRecord(BaseModel):
    """One memory entry as serialized for the drill-in view."""

    id: int
    key: str | None
    content: str
    author: str
    created_at: str


class BrowserMemoriesResponse(BaseModel):
    """Full list of memories sent to the addon for the Memories tab."""

    type: str = BROWSER_RESP_TYPE_MEMORIES
    memories: list[MemoryRecord]


class BrowserMemoryDetailResponse(BaseModel):
    """One memory's metadata + the first page of entries (newest-first), plus
    the first page of this collection's matching ``collector-runs`` entries
    when the memory is a collection (empty for logs).  The addon renders the
    collector activity inline on the collection's detail page.  Each section
    paginates independently via :class:`BrowserMemoryPageRequest`; the
    ``*_has_more`` flags tell the addon whether to show a "load more" control."""

    type: str = BROWSER_RESP_TYPE_MEMORY_DETAIL
    memory: MemoryRecord
    entries: list[MemoryEntryRecord]
    entries_has_more: bool = False
    collector_runs: list[MemoryEntryRecord] = []
    collector_runs_has_more: bool = False
    cursors: list[CursorRecord] = []  # read positions over the logs this collection reads


class BrowserMemoryPageResponse(BaseModel):
    """One more page of a single memory-detail section, newest-first, in
    response to a :class:`BrowserMemoryPageRequest`."""

    type: str = BROWSER_RESP_TYPE_MEMORY_PAGE
    name: str
    section: Literal["entries", "collector_runs"]
    entries: list[MemoryEntryRecord]
    has_more: bool


class BrowserCollectionTrigger(BaseModel):
    """Run a collection's extractor on demand, off the collector's cadence."""

    type: str
    name: str


class BrowserCollectionTriggerResult(BaseModel):
    """Outcome of an on-demand extractor run, sent back to the addon so the
    button can report success/failure."""

    type: str = BROWSER_RESP_TYPE_COLLECTION_TRIGGER_RESULT
    name: str
    success: bool
    message: str


class BrowserMemoryChanged(BaseModel):
    """Push notification: a memory was mutated.  ``name`` is the affected
    memory, or ``None`` for fan-out events not scoped to one memory."""

    type: str = BROWSER_RESP_TYPE_MEMORY_CHANGED
    name: str | None = None


class BrowserMemoryCreate(BaseModel):
    """Create a new collection from the addon.  Only collections are user-
    creatable; logs are seeded by migrations."""

    type: str
    name: str
    description: str
    intent: str | None = None  # the user's goal for this collection
    inclusion: str | None = None  # "always" | "relevant" | "never" (default relevant)
    recall: str  # "all" | "relevant" | "recent" (legacy "off" → inclusion=never)
    published: bool = False  # notify-on-new: a consumer delivers new entries (default silent)
    extraction_prompt: str | None = None
    collector_interval_seconds: int | None = None


class BrowserMemoryUpdate(BaseModel):
    """Edit metadata on an existing collection.  Only collections are user-
    editable; logs are read-only by design.  ``intent`` is editable here (the
    user owns the spec) even though the agent's ``collection_update`` tool
    cannot touch it."""

    type: str
    name: str
    description: str | None = None
    intent: str | None = None
    inclusion: str | None = None  # "always" | "relevant" | "never"
    recall: str | None = None  # "all" | "relevant" | "recent"
    published: bool | None = None  # flip notify-on-new; None = leave unchanged
    extraction_prompt: str | None = None
    collector_interval_seconds: int | None = None


class BrowserCursorSet(BaseModel):
    """Set a collection's read cursor over one log to a chosen point (a user
    override that may move backward — e.g. re-read from an earlier date)."""

    type: str
    name: str  # the collection (cursor owner)
    log_name: str
    last_read_at: str  # ISO-8601


class BrowserCursorClear(BaseModel):
    """Clear a collection's read cursor over one log — next cycle reads recent
    entries afresh (the first-cycle behavior), not the whole history."""

    type: str
    name: str  # the collection (cursor owner)
    log_name: str


class BrowserMemoryArchive(BaseModel):
    """Archive a memory by name (soft-delete from the active list)."""

    type: str
    name: str


class BrowserEntryCreate(BaseModel):
    """Manually add one entry to a collection.  Bypasses the collector —
    useful when the user wants to record something the auto-extractor
    missed.  Logs are append-only by the system; no manual entry path."""

    type: str
    memory: str
    key: str
    content: str


class BrowserEntryUpdate(BaseModel):
    """Replace the content of a keyed entry in a collection."""

    type: str
    memory: str
    key: str
    content: str


class BrowserEntryDelete(BaseModel):
    """Delete a keyed entry from a collection."""

    type: str
    memory: str
    key: str
