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


def _blank_to_none(value: object) -> object:
    """Collapse a blank string to ``None`` so an update treats it as "omitted".

    Models routinely emit ``""`` for an optional field they mean to leave
    unchanged — gpt-oss was observed passing ``extraction_prompt=""`` alongside
    a recall change, reasoning "they will not be updated".  But the update layer
    applies any value that ``is not None``, so a blank string would overwrite:
    silently blanking a ``description`` (and re-embedding empty as its stage-1
    routing anchor) or raising on an enum.  None of these fields has a
    meaningful empty value, so a blank means "skip", never "set to empty".
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


# Optional text on an update: a blank string is coerced to ``None`` (omitted)
# rather than written through, so it can never clobber the existing value.
OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]

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
    # true when the user wants to be told about new entries (notify-on-new).
    # Defaults to false (a silent collection) so an omission can't accidentally
    # notify; the tool description + skills drive the model to set it, and the
    # eval suite verifies it does when the user asked to be told.
    published: bool = False


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


class CatalogArgs(BaseModel):
    """No-field args for ``collection_catalog`` — it spans every collection."""


class CollectionUpdateArgs(BaseModel):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  A blank string counts as "not set": the ``OptionalText``
    fields coerce ``""`` to ``None`` so a field the model passes empty (to
    mean "leave it alone") is skipped rather than overwriting the existing
    value.  ``inclusion`` and ``recall`` are validated in the store layer.
    """

    name: MemoryName
    description: OptionalText = None
    inclusion: OptionalText = None  # "always" | "relevant" | "never"
    recall: OptionalText = None  # "all" | "relevant" | "recent"
    extraction_prompt: OptionalText = None
    collector_interval_seconds: int | None = None
    published: bool | None = None  # flip notify-on-new on/off; None = leave unchanged


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


class ReadPublishedLatestArgs(BaseModel):
    """A consumer's fan-in read across every ``published`` collection.

    Returns the ``n`` oldest entries not yet seen by this consumer (across all
    published sources, merged oldest-first), each tagged with its source.  The
    consumer names no source — Python picks from whichever published collections
    have unseen entries, advancing a per-(consumer, source) cursor only for the
    entries actually returned.
    """

    n: int = 1


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
