"""SQLModel tables for the Penny database."""

import json
from datetime import UTC, datetime

from sqlalchemy import Index, text
from sqlmodel import Field, SQLModel

from penny.constants import CycleTrigger


class PromptLog(SQLModel, table=True):
    """Log of every prompt sent to Ollama and its response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    model: str
    messages: str  # JSON-serialized list of message dicts
    tools: str | None = None  # JSON-serialized tool definitions
    response: str  # JSON-serialized response dict
    thinking: str | None = None  # Model's thinking/reasoning trace
    duration_ms: int | None = None  # How long the call took
    agent_name: str | None = None  # Which agent produced this call (chat, history, etc.)
    prompt_type: str | None = (
        None  # Which flow within the agent (user_message, free, daily_summary, etc.)
    )
    run_id: str | None = None  # Groups all prompts from one agentic loop invocation
    # Run outcome is set on the last prompt of a collector cycle.  All three
    # are NULL for non-collector agents (chat, schedule executor).
    run_outcome: str | None = None  # RunOutcome: failed|no_work|worked|incomplete|cancelled
    run_reason: str | None = None  # Structural reason (write-gate stop / no-done() close); #1569
    run_target: str | None = None  # Collection name the cycle was bound to
    # Count of failed tool calls in the run (ToolCallRecord.failed), stamped on
    # the last prompt alongside run_outcome.  NULL = not measured (old rows,
    # untagged/non-collector runs); the run-health classifier reads NULL as 0.
    tool_failures: int | None = None
    # The run's TRAILING turns — the conversation the loop appended after this
    # call, which no later call carried (#1778).  A tool result is otherwise
    # durable only as a side effect of riding into the NEXT call's `messages`, so
    # a run that ends immediately after executing a tool loses its terminal
    # outcome.  Same wire shape as `messages` (it is literally what the next
    # call's `messages` would have ended with — the final assistant tool-call turn
    # plus its tool-result turns), so the run renderers read the two as one
    # sequence.  Stamped on the run's last prompt by `set_run_trailing_messages`;
    # NULL for a run that ended on a text reply (every result already
    # round-tripped) and for every row that isn't its run's last.
    trailing_messages: str | None = None
    # How this CALL ended (#2355 part A) — "completed" | "cancelled" | "timed_out" |
    # "raised" (see ``penny.llm.models.CallOutcome``).  NULL on every row logged before
    # this column existed, which is read as "completed" by convention (the old code
    # path only ever logged a call that had succeeded).  ``response`` stays "{}" rather
    # than NULL for a call that never got one — every existing reader already treats a
    # falsy ``response`` as an empty dict, so this needs no new NULL-handling anywhere,
    # and it avoids widening a NOT NULL column (a SQLite table rebuild) for no reader
    # that needs the distinction.
    call_outcome: str | None = None
    # The failed call's own exception class name and message, set only on
    # "timed_out"/"raised" — empty for "completed"/"cancelled" (a cancellation's WHY is
    # the canceller's to record, not this client's).
    call_error_class: str | None = None
    call_error_message: str | None = None
    # The failed call's own ``LlmFault`` value (``penny.llm.models.LlmFault`` — "429",
    # "5xx", "4xx", "no choices", …), set only on "timed_out"/"raised" alongside the two
    # columns above.  The client already classifies a response error by status at the
    # moment it raises (``fault_for_status``) — this is that same classification,
    # persisted, so a reader can tell a provider-side refusal (429/5xx) from Penny's own
    # malformed or oversized request (4xx) without re-parsing the error message (#2526).
    call_error_fault: str | None = None
    # The enclosing chat/collector run this call was made ON BEHALF OF, when it is a
    # micro-context draw made inside that run rather than the run's own top-level call
    # (every micro-context customer sets this; it is None for an ordinary agent-loop
    # call, which already has no "parent" above its own ``run_id``).
    parent_run_id: str | None = None
    # The specific ToolCall.id the draw was made for — only ever set alongside
    # ``parent_run_id`` for a draw made from INSIDE a dispatched tool call (today, only
    # ``browse``'s extraction); the other micro-context customers run directly in agent
    # code, not nested inside a tool, so they set ``parent_run_id`` alone.
    parent_tool_call_id: str | None = None
    # Why this COMPLETED call's output was NOT used, set after the fact at the
    # rejection site (#2702): a ``ConditionKey`` value ("degenerate_output",
    # "xml", "budget_cut", …) for a draw the loop discarded, a validator
    # rejected, or a parse refused.  A reroll/retry attempt's row reads
    # ``call_outcome="completed"`` like a used call — this column is what
    # distinguishes it.  NULL on every used call and every row logged before
    # this column existed; ``discard_detail`` carries the optional detail
    # string.  Additive and never model-facing: only the addon serializers and
    # the live push read these.
    discard_reason: str | None = None
    discard_detail: str | None = None

    def get_trailing_messages(self) -> list[dict]:
        """The trailing turns as message dicts — empty when the run round-tripped
        every result (the overwhelmingly common case, and every pre-#1778 row)."""
        return json.loads(self.trailing_messages) if self.trailing_messages else []

    def get_response(self) -> dict:
        return json.loads(self.response)


class MessageLog(SQLModel, table=True):
    """Log of every user message and agent response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    direction: str = Field(index=True)  # "incoming" or "outgoing"
    sender: str = Field(index=True)
    content: str
    parent_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    signal_timestamp: int | None = Field(default=None)  # Original Signal timestamp (ms since epoch)
    recipient: str | None = Field(default=None, index=True)  # Who the message was sent to
    external_id: str | None = Field(default=None, index=True)  # Platform-specific message ID
    is_reaction: bool = Field(default=False, index=True)  # True if this is a reaction message
    processed: bool = Field(
        default=False
    )  # True if this message has been processed by extraction pipeline
    device_id: int | None = Field(
        default=None, foreign_key="device.id", index=True
    )  # FK to device that sent/received this message
    embedding: bytes | None = None  # Serialized float32 embedding vector
    # Emission provenance (#1568): the mechanism (bound collection) whose
    # autonomous cycle produced this send.  NULL for a direct reply — a chat turn
    # with a live triggering user message names no mechanism.  Stamped by the
    # drainer at delivery time from the queued row (the collection is known at
    # enqueue), so "which mechanism sent this?" is a read, not a diagnosis.
    # A by-name reference to ``memory.name`` (the FK-by-name discipline the memory
    # layer uses), a plain column rather than a DB-level FK — like
    # ``mutation_event.entity_name`` — so it doesn't close a circular foreign-key
    # cycle with ``memory.source_message_id`` (→ ``messagelog.id``).  Served by the
    # partial ``ix_messagelog_emission_time`` index below, not a single-column one
    # (no query filters on mechanism equality; the hot read is the newest-emissions
    # scan).
    mechanism: str | None = Field(default=None)
    # The run whose reply this outgoing row carries (#2630): stamped by the
    # framework at log time on the direct-reply path, so the conversation
    # history's per-turn tool line pairs a run with its reply by identity, not
    # by a timestamp guess.  NULL for an autonomous send (the drainer's, or any
    # ``send_message``) — a send with no run behind it can never claim one.  A
    # by-value reference to ``promptlog.run_id`` (the FK-by-name discipline the
    # memory layer uses, like ``mechanism`` above) — ``run_id`` is not unique on
    # ``promptlog``, so no DB-level FK is possible; the pairing joins on it.
    run_id: str | None = Field(default=None, index=True)

    __table_args__ = (
        Index("ix_messagelog_device_timestamp_id", "device_id", "timestamp", "id"),
        Index("ix_messagelog_sender_timestamp_id", "sender", "timestamp", "id"),
        Index("ix_messagelog_recipient_timestamp_id", "recipient", "timestamp", "id"),
        # Partial index over emission rows only (#1568): ``recent_emissions`` runs on
        # the self-state render hot path (every chat prompt build) filtering
        # ``mechanism IS NOT NULL`` and ordering ``timestamp DESC, id DESC`` over the
        # unbounded messagelog — mechanism-bearing rows are sparse, so without this a
        # timestamp-index walk tests mechanism on many rows before the small LIMIT
        # fills.  A backward scan of this index fills it immediately.
        Index(
            "ix_messagelog_emission_time",
            "timestamp",
            "id",
            sqlite_where=text("mechanism IS NOT NULL"),
        ),
    )


class UserInfo(SQLModel, table=True):
    """Basic user information collected on first interaction."""

    __tablename__ = "userinfo"

    id: int | None = Field(default=None, primary_key=True)
    sender: str = Field(unique=True, index=True)
    name: str
    location: str
    timezone: str  # IANA timezone (e.g., "America/Los_Angeles")
    date_of_birth: str  # YYYY-MM-DD format
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CommandLog(SQLModel, table=True):
    """Log of every command invocation and its response."""

    __tablename__ = "command_logs"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    channel_type: str  # "signal" or "discord"
    command_name: str = Field(index=True)  # e.g., "debug"
    command_args: str  # e.g., "" or "debug" (for /commands debug)
    response: str  # Full response text sent to user
    error: str | None = None  # Error message if command failed


class RuntimeConfig(SQLModel, table=True):
    """User-configurable runtime settings stored in database."""

    __tablename__ = "runtime_config"

    key: str = Field(primary_key=True)
    value: str  # Store as string, parse to correct type on load
    description: str  # Human-readable description with units
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MuteState(SQLModel, table=True):
    """Per-user mute state for notifications.

    Row exists = muted. Delete row = unmuted.
    """

    user: str = Field(primary_key=True)
    muted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainPermission(SQLModel, table=True):
    """A domain access permission for browser tool execution."""

    __tablename__ = "domain_permission"

    id: int | None = Field(default=None, primary_key=True)
    domain: str = Field(unique=True, index=True)
    permission: str  # "allowed" or "blocked"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Device(SQLModel, table=True):
    """A registered device (channel endpoint) for the single user."""

    id: int | None = Field(default=None, primary_key=True)
    channel_type: str = Field(index=True)  # ChannelType enum value
    identifier: str = Field(unique=True, index=True)  # Phone number, discord ID, browser label
    label: str  # Human-readable name (e.g., "Signal", "firefox macbook 16")
    is_default: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IosDeviceRegistration(SQLModel, table=True):
    """iOS-specific state for a registered Penny client device."""

    __tablename__ = "ios_device_registration"

    device_id: int = Field(foreign_key="device.id", primary_key=True)
    apns_token: str | None = Field(default=None, index=True)
    apns_environment: str = Field(default="sandbox")
    app_version: str | None = None
    device_secret_hash: str | None = None
    push_enabled: bool = Field(default=True)
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_updated_at: datetime | None = None


class Knowledge(SQLModel, table=True):
    """Legacy summarized-page table.  No Python readers anymore — kept so
    ``create_tables()`` materialises the schema that migration 0027 reads
    from when backfilling the ``knowledge`` memory collection.

    Per Stage 12 of the migration plan, legacy tables stay in place until
    a separate, post-migration drop.
    """

    id: int | None = Field(default=None, primary_key=True)
    url: str = Field(unique=True, index=True)
    title: str
    summary: str
    embedding: bytes | None = None
    source_prompt_id: int = Field(foreign_key="promptlog.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryRow(SQLModel, table=True):
    """A named memory's persisted metadata — a keyed collection or an
    append-only log.

    The stored row behind a :class:`penny.database.memory.Memory` object: its
    name, shape (``type``), description, and collector cadence.  The
    polymorphic ``Memory`` wraps one of these and adds the read/write behaviour;
    the facades (messages, collector-runs) wrap a marker row but read their
    canonical tables.

    A flag — ``notify`` — is emission-as-property (#1557): when set,
    the collector tells the user about new/changed entries in the same cycle that
    produced them, via the run-time notify suffix appended to its system prompt.
    """

    __tablename__ = "memory"

    name: str = Field(primary_key=True)
    type: str  # MemoryType enum value: "collection" or "log"
    description: str  # Content-reflective summary; the resolve-by-meaning anchor
    # Embedding of ``description`` — the meaning anchor for ``find`` /
    # resolve-by-meaning (#1558), computed once on create/description-edit
    # (NULL until backfilled at startup).
    description_embedding: bytes | None = None
    archived: bool = Field(default=False, index=True)
    # Emission-as-property (#1557, exposed by #1591's ``collection_configure``): when
    # true the collection notifies the user of new/changed entries — assembly
    # appends the run-time notify steps (``Prompt.COLLECTOR_NOTIFY_STEPS``) to the
    # collector's composed prompt, numbered continuously before the injected
    # terminal ``done()``.  Opt-in (default false); the sole emission flag since
    # #1557 retired the ``published`` pub/sub side-channel + the notifier consumer.
    # ``server_default`` so raw-SQL inserts predating the column satisfy NOT NULL.
    notify: bool = Field(default=False, sa_column_kwargs={"server_default": "0"})
    extraction_prompt: str | None = Field(default=None)
    # The collection's schedule: ONE RRULE string (#1857, migration 0104) — the whole
    # schedule grammar, replacing the interval / run_at / on_advance / cron union.  An
    # optional leading ``DTSTART:`` line says when the recurrence starts; without one it
    # starts at ``created_at``.  ``Collector._is_ready`` gates on the rule's next
    # occurrence after ``last_collected_at or created_at``, so the rule IS the cadence —
    # there is no interval column to widen and no auto-throttle to widen it.  NULL for a
    # log or an inert collection (one with no job yet), which never dispatches.
    schedule: str | None = Field(default=None)
    # Provenance + lifecycle (operational registry, #1566) — every mechanism
    # created from a chat request can answer *who* asked for it, *what* run
    # created it, and *when* it ends, by a read.  All nullable: seeded / system
    # / migration-created rows have no chat provenance and no end condition.
    # ``source_message_id`` is the spawning user message; ``created_by_run_id``
    # is the ``promptlog.run_id`` of the creating run; ``expires_at`` is the
    # end condition (consumed by #1562's lifecycle axis).
    source_message_id: int | None = Field(default=None, foreign_key="messagelog.id")
    created_by_run_id: str | None = Field(default=None)
    # The end conditions the archive lifecycle retires a collection on.  Both are
    # settable directly AND lifted out of the schedule's own ``UNTIL=`` / ``COUNT=``
    # parts at parse time (#1857), so a bounded rule and a bounded collection are one
    # thing.  ``expires_at``: the collection archives itself once this passes.
    # ``max_runs``: after this many completed (non-cancelled) cycles the scheduler
    # archives it via a system-actor mutation, so a one-shot (``COUNT=1``) retires
    # itself.  NULL = no end condition / unlimited.
    expires_at: datetime | None = Field(default=None)
    max_runs: int | None = Field(default=None)
    last_collected_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": "1970-01-01 00:00:00"},
    )


class MemoryEntry(SQLModel, table=True):
    """One immutable entry belonging to a memory.

    Collections have keys; log entries carry `key=None`. Content is embedded
    on write so similarity reads are cheap; keys are embedded too when present.
    """

    __tablename__ = "memory_entry"

    id: int | None = Field(default=None, primary_key=True)
    memory_name: str = Field(foreign_key="memory.name", index=True)
    key: str | None = Field(default=None, index=True)
    content: str
    author: str = Field(index=True)
    key_embedding: bytes | None = None
    content_embedding: bytes | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    # Entry-level provenance (#1560): the run that first wrote this entry and the
    # run that last rewrote it (a collection entry is overwritten — a watch's
    # baseline changes every cycle — so "who wrote the current value?" differs
    # from "who created it?").  The run id is the join key into the ledger
    # (``promptlog`` for the run + tool calls); the full write history is those
    # call records, and these two stamps are the read-path anchors that make
    # ``run_list`` one guess-free hop from wherever an entry renders.
    # Threaded as a parameter from the writing run — never ambient state.  NULL
    # for migration-seeded / pre-#1560 entries.
    created_by_run_id: str | None = Field(default=None, index=True)
    last_written_by_run_id: str | None = None


class MutationEvent(SQLModel, table=True):
    """One durable create / update / archive / unarchive of a registry entity —
    the event half of the operational spine's mutation stream (#1560).

    The registry row (``memory``) holds the *current* materialized state (the
    ``archived`` flag is the truth); this table is the audit + provenance trail
    of how it got there — so "when was this archived, and by what?" is answered
    by a read, not re-decided by the model from its own past narration.  It is
    the one ledger table with no other home: an entry write is a ``promptlog``
    tool call and a run is a ``promptlog`` group, but a *system* archive (the
    scheduler's ``max_runs`` / ``expires_at`` retire) runs no model and logs no
    prompt, so without this row it would be invisible.

    Audit, not event sourcing: state stays materialized on ``memory``; this only
    records the transitions.
    """

    __tablename__ = "mutation_event"

    id: int | None = Field(default=None, primary_key=True)
    # The kind + name of the entity mutated (``MutationEntityType`` /
    # ``memory.name`` — a join key into the registry, matching the FK-by-name
    # discipline the rest of the memory layer uses).
    entity_type: str = Field(index=True)  # MutationEntityType value
    entity_name: str = Field(index=True)  # e.g. the collection name
    action: str  # MutationAction value: created | updated | archived | unarchived
    # Who caused it (``MutationActor``) and the run that did (the join key into
    # the ledger; NULL only when no run was in the loop and the actor is system).
    actor: str = Field(index=True)  # MutationActor value
    run_id: str | None = Field(default=None, index=True)
    # JSON-serialized ``MutationDetail`` — what changed (the edited field names),
    # a human cause note (e.g. the system archive's "max_runs reached"), and the
    # options-presented accommodation (present in the shape, populated by the
    # enumerated-decision unions of #1562/#1563 — not forced at call sites now).
    detail: str | None = None
    # Datetime for ordering — never the id (criterion 2 enumerates a mechanism's
    # history in time order).
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)


class AgentCursor(SQLModel, table=True):
    """Per-agent read cursor into a log-shaped memory.

    `last_read_at` is the high-water mark: the agent has consumed every entry
    with `created_at <= last_read_at`. Advanced two-phase by the orchestrator
    (pending during the run, committed on successful completion).
    """

    __tablename__ = "agent_cursor"

    agent_name: str = Field(primary_key=True)
    memory_name: str = Field(primary_key=True, foreign_key="memory.name")
    last_read_at: datetime
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SendQueueItem(SQLModel, table=True):
    """One outbound message awaiting delivery — the durable send queue.

    ``send_message`` enqueues here instead of dropping a message when the
    autonomous-send cooldown hasn't elapsed; a background drain schedule
    delivers the oldest pending row once the cooldown clears, then stamps
    ``sent_at``.  A row has four states, and each marker means exactly one
    thing: **pending** (``sent_at IS NULL AND cancelled_at IS NULL AND
    failed_at IS NULL``), **delivered** (``sent_at`` stamped — the single
    source of truth for "was it sent"), **cancelled** (``cancelled_at``
    stamped while ``sent_at`` stays NULL — the collector was archived before
    the message went out, so it was never sent; #1634), and **failed**
    (``failed_at`` stamped, ``sent_at`` NULL — a composing report whose
    message never became sendable within the retry budget, #2656).
    ``collection`` is the collector that queued it (the bound target name), so
    delivery — and cancellation — is attributable.
    """

    __tablename__ = "send_queue"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str
    collection: str
    # What set the cycle that queued this message running (``CycleTrigger``):
    # the schedule's own cadence, or the user's "run this now" (#1939).  The
    # drainer reads it to pick the delivery lane — a message the user is sitting
    # waiting for is conversational and skips the autonomous-send cooldown, where
    # a cadence message waits it out.  Never NULL: an unstamped row is a cadence
    # row, which is the conservative direction (the bypass must be claimed).
    origin: str = Field(default=CycleTrigger.CADENCE)  # CycleTrigger value
    # The composing report (#2656): the collector cycle's run id whose written
    # entries the report is built from.  NULL for a row that already carries its
    # content (the ordinary queued message); a row with this set and empty
    # content is a report awaiting composition — the drainer builds the
    # JobReport from the cycle's entries, runs the delivery chat turn, and
    # stores what it wrote before delivering.
    report_run_id: str | None = Field(default=None, index=True)
    # How many times composing this row's report has failed, so the retry is
    # bounded and the bounded end is visible (#2656).
    attempts: int = Field(default=0)
    # Stamped when the retry budget is spent without a sendable message — a
    # VISIBLE terminal state: the row is kept, never delivered, and excluded
    # from the pending query structurally.  Distinct from ``cancelled_at``: a
    # failed report is a compose that never produced a message, where a
    # cancelled one was withheld on purpose (an archived collector, a mute).
    # ``sent_at`` stays NULL on both — neither was ever sent.
    failed_at: datetime | None = None
    sent_at: datetime | None = None
    # Stamped when the queuing collector is archived before delivery (#1634):
    # a VISIBLE cancellation (the row is kept as an audit trail, not deleted),
    # excluded from the pending query structurally.  ``sent_at`` stays NULL — a
    # cancelled row was never sent — so the delivered/cancelled distinction is
    # never ambiguous.
    cancelled_at: datetime | None = None

    __table_args__ = (
        # Partial index over the PENDING tail, leading with the lane (#1939).  The
        # drainer asks for the oldest pending row OF ONE ORIGIN while the cooldown is
        # running, and the sparse case is the one that matters — a single on-demand
        # row behind a backlog of cadence ones — so an index that serves only the
        # ORDER BY would walk the whole ordered pending partition before the LIMIT 1
        # filled.  Leading with the equality-filtered column seeks straight to it;
        # the 0061 index still serves the unscoped read.
        Index(
            "ix_send_queue_pending_lane",
            "origin",
            "created_at",
            sqlite_where=text("sent_at IS NULL AND cancelled_at IS NULL AND failed_at IS NULL"),
        ),
    )


class IosOutboxItem(SQLModel, table=True):
    """One message available for an iOS client to pull and acknowledge."""

    __tablename__ = "ios_outbox"

    id: int | None = Field(default=None, primary_key=True)
    message_log_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    device_id: int = Field(foreign_key="device.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    content: str
    attachments_json: str | None = None
    source_type: str | None = Field(default=None, index=True)
    source_name: str | None = Field(default=None, index=True)
    source_hint: str | None = None
    push_title: str
    push_summary: str
    push_sent_at: datetime | None = None
    push_error: str | None = None
    acked_at: datetime | None = Field(default=None, index=True)


class Media(SQLModel, table=True):
    """Binary media (images) captured while browsing or drawn on request.

    The browse tool stores each page's image here with its source URL, page
    title, and an embedding of that metadata.  At channel egress the outgoing
    message text is embedded and ``MediaStore.select_image`` attaches the most
    relevant image (cited URL → domain → jittered nearest), with no model
    involvement.  A ``generate_image`` row is delivered deterministically by id
    to its *own* reply (``send_response(media_ids=...)``) — never fuzzy-matched
    for that reply — but carries an embedding of its description so it joins
    the nearest-image pool for future replies.
    """

    __tablename__ = "media"

    id: int | None = Field(default=None, primary_key=True)
    mime_type: str
    data: bytes
    source_url: str | None = None
    title: str | None = None
    embedding: bytes | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EmailRule(SQLModel, table=True):
    """A persistent email rule for automatic organization.

    Rules are applied during scheduled email checks to automatically
    move, label, or process emails matching specified conditions.
    Works with any email provider plugin (Zoho, Google, etc.).
    """

    __tablename__ = "email_rule"

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)  # Plugin name: "zoho", "google", etc.
    name: str  # Human-readable rule name
    condition: str  # JSON condition: {"from": "aws", "subject_contains": "invoice"}
    action: str  # JSON action: {"move_to": "Accounting/Expenses/AWS", "label": "completed"}
    enabled: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_applied_at: datetime | None = None  # When this rule was last triggered
