"""SQLModel tables for the Penny database."""

import json
from datetime import UTC, datetime

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
    run_success: bool | None = None
    run_reason: str | None = None  # Free-text reason from done(summary=...)
    run_target: str | None = None  # Collection name the cycle was bound to

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
    thought_id: int | None = Field(
        default=None, foreign_key="thought.id", index=True
    )  # FK to thought that triggered this notification
    device_id: int | None = Field(
        default=None, foreign_key="device.id", index=True
    )  # FK to device that sent/received this message
    embedding: bytes | None = None  # Serialized float32 embedding vector


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


class Schedule(SQLModel, table=True):
    """User-created scheduled background tasks."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)  # Signal number or Discord user ID
    user_timezone: str = Field(default="UTC")  # IANA timezone (e.g., "America/Los_Angeles")
    cron_expression: str  # Cron format for recurring execution
    prompt_text: str  # Prompt to execute when schedule fires
    timing_description: str  # Original human description for display (e.g., "daily 9am")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MuteState(SQLModel, table=True):
    """Per-user mute state for notifications.

    Row exists = muted. Delete row = unmuted.
    """

    user: str = Field(primary_key=True)
    muted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Thought(SQLModel, table=True):
    """A persistent inner monologue entry — Penny's stream of consciousness."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)
    content: str
    preference_id: int | None = Field(default=None, foreign_key="preference.id", index=True)
    run_id: str | None = Field(default=None, index=True)  # Links to PromptLog.run_id
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    notified_at: datetime | None = None  # When this thought was shared with the user
    embedding: bytes | None = None  # Serialized float32 content embedding (novelty/sentiment)
    title: str | None = None  # Short topic name for dedup and image search
    title_embedding: bytes | None = None  # Serialized float32 title embedding (dedup)
    image: str | None = None  # Base64 data URI for feed display
    valence: int | None = None  # User reaction: 1 = positive, -1 = negative, None = unreacted


class Preference(SQLModel, table=True):
    """A user preference extracted from conversation sentiment or emoji reactions."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)
    content: str  # The preference topic (e.g., "dark roast coffee", "cold weather")
    valence: str = Field(index=True)  # PreferenceValence enum value
    embedding: bytes | None = None  # Serialized float32 embedding vector
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    last_thought_at: datetime | None = None  # When this preference was last used as a thinking seed
    mention_count: int = Field(default=1)  # Times this topic was mentioned in conversation
    source: str = Field(default="extracted", index=True)  # PreferenceSource enum value


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


class Memory(SQLModel, table=True):
    """A named memory — either a keyed collection or an append-only log.

    Memories are Penny's unified data primitive: user- or system-authored
    containers that agents read from and write to via tools. Each memory
    has a name, a shape (collection vs log), and a recall mode that controls
    whether it appears in the chat agent's ambient context.
    """

    __tablename__ = "memory"

    name: str = Field(primary_key=True)
    type: str  # MemoryType enum value: "collection" or "log"
    description: str  # Human-authored summary shown to the model in the memory registry
    recall: str  # RecallMode enum value: "off" | "recent" | "relevant" | "all"
    archived: bool = Field(default=False, index=True)
    extraction_prompt: str | None = Field(default=None)
    collector_interval_seconds: int | None = Field(default=None)
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


class Media(SQLModel, table=True):
    """Binary media (images, etc.) stored out-of-line from memory entries.

    Entries reference media via `<media:ID>` tokens in their content body; the
    ambient recall assembler resolves tokens to inline image data at prompt
    build time.
    """

    __tablename__ = "media"

    id: int | None = Field(default=None, primary_key=True)
    mime_type: str
    data: bytes
    source_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
