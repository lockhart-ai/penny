"""Pydantic arg models for the memory tool surface.

Each tool validates its kwargs through one of these models as its first line,
per the Pydantic-everywhere rule. Most read tools accept ``k: int | None``
meaning "no cap — return every entry" when omitted; this matches the access
layer's signature.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, model_validator

# Models occasionally substitute Unicode dashes (U+2010–U+2015) for ASCII
# hyphen-minus (U+002D) when emitting memory names — gpt-oss has been
# observed writing ``"board‑games"`` for ``"board-games"``.
# The visual is identical but the string compares unequal, so memory-keyed
# tools (``collection_write``, ``log_read``, etc.) silently failed
# with refusals or "memory not found" errors.  Normalise on the way in so
# the rest of the stack sees a single canonical form.
_UNICODE_DASHES = "‐‑‒–—―−"


def _normalize_dashes(value: object) -> object:
    if not isinstance(value, str):
        return value
    if not any(ch in value for ch in _UNICODE_DASHES):
        return value
    out = value
    for ch in _UNICODE_DASHES:
        out = out.replace(ch, "-")
    return out


def _normalize_dash_list(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [_normalize_dashes(item) for item in value]


MemoryName = Annotated[str, BeforeValidator(_normalize_dashes)]
MemoryNameList = Annotated[list[str], BeforeValidator(_normalize_dash_list)]

# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateArgs(BaseModel):
    """Args for ``collection_create``.

    A collection without an ``extraction_prompt`` is passive (nothing
    fills it) and a collection without ``collector_interval_seconds``
    has no cadence (nothing schedules it).  Both are required at the
    tool surface so every model-created collection gets a working
    collector immediately, instead of silently sitting empty until the
    user notices.

    ``intent`` is also required: capturing what the user asked for at
    creation is part of creating a collection.  It is the spec a quality
    collector later judges the prompt and behavior against, and it has no
    field on ``collection_update`` — once set, it's immutable.
    """

    name: MemoryName
    description: str
    inclusion: str  # "always" | "relevant" | "never" — validated in the store layer
    recall: str  # "all" | "relevant" | "recent" — validated in the store layer
    extraction_prompt: str
    collector_interval_seconds: int
    intent: str


class LogCreateArgs(BaseModel):
    """Args for ``log_create``.

    Logs are append-only streams of events (messages, browse results,
    measurements).  No extraction_prompt — logs are inputs, not curated
    outputs.  No interval — logs don't have a collector.
    """

    name: MemoryName
    description: str
    inclusion: str  # "always" | "relevant" | "never" — validated in the store layer
    recall: str  # "all" | "relevant" | "recent" — validated in the store layer


class MemoryNameArgs(BaseModel):
    """One-field args for ``archive`` / ``unarchive`` / read-all / keys."""

    memory: MemoryName


class CollectionUpdateArgs(BaseModel):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  ``inclusion`` and ``recall`` are validated in the store layer.
    """

    name: MemoryName
    description: str | None = None
    inclusion: str | None = None  # "always" | "relevant" | "never"
    recall: str | None = None  # "all" | "relevant" | "recent"
    extraction_prompt: str | None = None
    collector_interval_seconds: int | None = None


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetArgs(BaseModel):
    """Exact key lookup in a collection."""

    memory: MemoryName
    key: str


class ReadLatestArgs(BaseModel):
    """Newest-first; ``k=None`` returns all."""

    memory: MemoryName
    k: int | None = None


class ReadRandomArgs(BaseModel):
    """Random sample; ``k=None`` returns all."""

    memory: MemoryName
    k: int | None = None


class ReadSimilarArgs(BaseModel):
    """Top-k by content cosine similarity to ``anchor`` (embedded by the tool).

    The similarity floor is fixed (``MEMORY_RELEVANT_ABSOLUTE_FLOOR`` plus
    the adaptive cluster gate) — the model can't override it, since cosine
    thresholds are opaque values it has no grounding to pick.
    """

    memory: MemoryName
    anchor: str
    k: int | None = None


# ── Log-specific reads ──────────────────────────────────────────────────────


class ReadLogArgs(BaseModel):
    """A single ``log_read`` over a log.  The caller names only the log — the
    semantics (cursor batch for collectors, recent window for chat/schedule) and
    all sizes are decided in Python from the caller, never by the model."""

    memory: MemoryName


# ── Collection writes ───────────────────────────────────────────────────────


class CollectionEntrySpec(BaseModel):
    """One entry in a ``collection_write`` batch."""

    key: str
    content: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_stringified_object(cls, value: Any) -> Any:
        """Parse a JSON-stringified dict back into a plain dict.

        Some models wrap array elements in outer quotes, producing a JSON string
        that contains an object literal (e.g. '{"key": "foo", "content": "bar"}')
        instead of a bare object. Detect and unwrap it so field validation proceeds
        normally.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                pass
        return value


class CollectionWriteArgs(BaseModel):
    """Batched write to a collection with dedup applied per entry."""

    memory: MemoryName
    entries: list[CollectionEntrySpec] = Field(min_length=1)


class UpdateEntryArgs(BaseModel):
    """Replace content for an existing key in a collection."""

    memory: MemoryName
    key: str
    content: str


class CollectionMoveArgs(BaseModel):
    """Move an entry between collections by key."""

    key: str
    from_memory: MemoryName
    to_memory: MemoryName


class CollectionMergeArgs(BaseModel):
    """Merge all entries from one collection into another, then archive the source."""

    from_memory: MemoryName
    to_memory: MemoryName


class CollectionDeleteEntryArgs(BaseModel):
    """Delete an entry from a collection by key."""

    memory: MemoryName
    key: str


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendArgs(BaseModel):
    """Append one keyless entry to a log."""

    memory: MemoryName
    content: str


# ── Introspection ───────────────────────────────────────────────────────────


class ExistsArgs(BaseModel):
    """Cross-memory dedup probe used by thinking-class agents before writes."""

    memories: MemoryNameList = Field(min_length=1)
    content: str
    key: str | None = None


class DoneArgs(BaseModel):
    """Cycle terminator — pair the exit with a success flag and a summary.

    ``success`` is true if the cycle accomplished what the prompt asked,
    false on no-op or partial failure.  ``summary`` is a one-sentence
    prose description of what the cycle actually did (entries written,
    messages sent, why no-op).  Both are logged to ``collector-runs`` so
    Penny can audit collector behaviour from chat.
    """

    success: bool
    summary: str
