"""SQLModel tables for the Penny database."""

import json
from datetime import UTC, datetime

from sqlalchemy import Index, text
from sqlmodel import Field, SQLModel


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

    def get_messages(self) -> list[dict]:
        return json.loads(self.messages)

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
    # Emission-as-property (#1557, exposed by #1591's ``collection_create``): when
    # true the collection notifies the user of new/changed entries — assembly
    # appends the run-time notify steps (``Prompt.COLLECTOR_NOTIFY_STEPS``) to the
    # collector's composed prompt, numbered continuously before the injected
    # terminal ``done()``.  Opt-in (default false); the sole emission flag since
    # #1557 retired the ``published`` pub/sub side-channel + the notifier consumer.
    # ``server_default`` so raw-SQL inserts predating the column satisfy NOT NULL.
    notify: bool = Field(default=False, sa_column_kwargs={"server_default": "0"})
    extraction_prompt: str | None = Field(default=None)
    collector_interval_seconds: int | None = Field(default=None)
    # The user's intended cadence — the value the collector snaps back to when a
    # cycle produces work.  ``collector_interval_seconds`` is the *current*
    # cadence (possibly auto-throttled upward); this is the floor to restore.
    base_interval_seconds: int | None = Field(default=None)
    # Consecutive cycles that produced no work; at COLLECTOR_THROTTLE_AFTER the
    # collector doubles its interval and resets this to 0.  See agents/collector.
    consecutive_idle_runs: int = Field(default=0, sa_column_kwargs={"server_default": "0"})
    # Skill provenance (#1603): the skill this collection was instantiated from
    # (#1591's front door) and the params bound into its render — so "which skill
    # made this, and with what?" is a read off the collection's own row, and a
    # future rebind/re-render has the current bindings as reachable input.
    # ``skill_name`` is a by-name reference to ``skill.name`` (a plain column, not a
    # DB FK — a skill is re-teachable / REPLACE-able, so the reference must survive a
    # re-teach); ``skill_params`` is the bound params as a JSON object.  Both NULL for
    # a hand-authored / seeded / migration collection (no skill origin), so its
    # catalog / metadata render is byte-identical to the pre-provenance shape.  The
    # rendered ``extraction_prompt`` is the snapshot that actually runs — these name
    # the recipe it came from, they don't drive it.
    skill_name: str | None = Field(default=None)
    skill_params: str | None = Field(default=None)
    # Provenance + lifecycle (operational registry, #1566) — every mechanism
    # created from a chat request can answer *who* asked for it, *what* run
    # created it, and *when* it ends, by a read.  All nullable: seeded / system
    # / migration-created rows have no chat provenance and no end condition.
    # ``source_message_id`` is the spawning user message; ``created_by_run_id``
    # is the ``promptlog.run_id`` of the creating run; ``expires_at`` is the
    # end condition (consumed by #1562's lifecycle axis).
    source_message_id: int | None = Field(default=None, foreign_key="messagelog.id")
    created_by_run_id: str | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    # Once-shaped trigger (#1556, store-level only — no model-facing create args
    # yet; #1562 exposes them).  ``run_at``: the collector runs only at/after this
    # UTC time (a delayed / one-shot start), NULL for an ordinary recurring
    # cadence.  ``max_runs``: after this many completed (non-cancelled) cycles the
    # scheduler archives the collection via a system-actor mutation, so a one-shot
    # reminder (``run_at`` + ``max_runs=1``) retires itself.  NULL = unlimited.
    run_at: datetime | None = Field(default=None)
    max_runs: int | None = Field(default=None)
    # On_advance trigger (#1604, model-facing via #1591's ``collection_create``):
    # the declared source LOG whose advance wakes this collection.  When set, the
    # collector gates readiness on the source's high-water mark passing this
    # collection's read cursor (source head > cursor) — the declared-input variant
    # of the inferred cursor gate, reusing the same ``AgentCursor``.  NULL for a
    # recurring (``interval``) or once-shaped (``run_at``) collection — the trigger
    # union is exclusive.  The collection is paced at ``collector_interval_seconds``
    # (the optional ``min_interval`` floor, or the dispatcher tick).
    source_log: str | None = Field(default=None)
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
    # ``read_run_calls`` one guess-free hop from wherever an entry renders.
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


class Skill(SQLModel, table=True):
    """A named, immutable skill — a certified-by-execution script of tool-call
    steps distilled from ONE demonstrated run (#1590, stage ④ of #1562).

    A skill is authored ONLY by the framework — there is no ``skill_create`` tool.
    At the end of every qualifying chat run the run-end extractor
    (``penny.skill_extraction``) snapshots that run's own ledger, copying its
    succeeded, non-``done`` tool calls into ``steps`` (the ``LoggedToolCall`` shape as
    JSON) and factoring each argument by provenance into declared ``parameters`` (JSON)
    — a value from a prior step's result becomes a binding, the scoped-write target a
    retarget-owned constant, every other string leaf a required parameter (#1658/
    #1659), semantically named + described by the run-end naming micro-context
    (#1668).  #1591's ``collection_create`` renders ``steps`` + bound params into the
    collection's numbered TEXT ``extraction_prompt`` at creation.

    **One row per name — no versioning.**  Collections carry the rendered text
    snapshotted at creation, so a re-teach never retroactively changes an
    instantiation; the version pin had no remaining job.  Re-demonstrating a routine
    REPLACES the row (steps/parameters/provenance) by name — or, for a same-shape,
    same-meaning routine, keeping the existing name — so ``name`` is the unique key.

    ``description`` doubles as the resolution anchor (``description_embedding``,
    populated at write) and is the run's triggering message.  ``source_run_id`` is
    the demonstrated run.  ``author`` is the extracting agent.  There is no seed
    library (migration 0084 ships the table empty), so the certified-by-execution
    invariant holds universally — every stored step succeeded in its source run.
    """

    __tablename__ = "skill"

    name: str = Field(primary_key=True)
    steps: str  # JSON-serialized list[SkillStep]
    parameters: str  # JSON-serialized list[SkillParameter]
    intent: str
    description: str
    description_embedding: bytes | None = None
    source_run_id: str | None = Field(default=None, index=True)
    author: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": "1970-01-01 00:00:00"},
    )


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
    ``sent_at``.  A row has three states, and each marker means exactly one
    thing: **pending** (``sent_at IS NULL AND cancelled_at IS NULL``),
    **delivered** (``sent_at`` stamped — the single source of truth for "was
    it sent"), and **cancelled** (``cancelled_at`` stamped while ``sent_at``
    stays NULL — the collector was archived before the message went out, so it
    was never sent; #1634).  ``collection`` is the collector that queued it (the
    bound target name), so delivery — and cancellation — is attributable.
    """

    __tablename__ = "send_queue"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str
    collection: str
    sent_at: datetime | None = None
    # Stamped when the queuing collector is archived before delivery (#1634):
    # a VISIBLE cancellation (the row is kept as an audit trail, not deleted),
    # excluded from the pending query structurally.  ``sent_at`` stays NULL — a
    # cancelled row was never sent — so the delivered/cancelled distinction is
    # never ambiguous.
    cancelled_at: datetime | None = None


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
