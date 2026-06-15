"""Shared value types for the memory layer — enums, errors, and input models.

Kept dependency-light (no engine, no Memory classes) so both the polymorphic
``Memory`` objects and the ``MemoryStore`` registry can import from here without
a cycle.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import BaseModel
from similarity.embeddings import normalize_unicode

from penny.config_params import RuntimeParams


class MemoryType(StrEnum):
    COLLECTION = "collection"
    LOG = "log"


class Inclusion(StrEnum):
    """Stage-1 collection-routing flag — does this memory feed recall at all.

    ``always`` participates unconditionally; ``relevant`` participates only
    when the conversation embeds close to the memory's description anchor;
    ``never`` is excluded (the old ``recall=off``).
    """

    ALWAYS = "always"
    RELEVANT = "relevant"
    NEVER = "never"


class RecallMode(StrEnum):
    """Stage-2 entry-rendering flag — which entries of an included memory surface.

    ``recent`` is the newest-first slice; ``all`` is the full set; ``relevant``
    is hybrid-ranked (embedding cosine fused with IDF-lexical) against the
    conversation window, top-N, no floor — the stage-1 gate already decided
    the memory is relevant.
    """

    RECENT = "recent"
    RELEVANT = "relevant"
    ALL = "all"


class MemoryAccessError(Exception):
    """A memory operation refused at the tool boundary — the memory is missing,
    the wrong shape for the op, or a read-only facade.

    ``str(self)`` is the model-readable reason, which the tool layer returns
    verbatim.  Catching this one base handles all three refusals uniformly, so a
    tool doesn't need a per-call type check or a sentinel return value.
    """


class MemoryTypeError(MemoryAccessError):
    """Raised when an operation is called against the wrong memory type."""


class WrongShapeError(MemoryTypeError):
    """Raised by a base ``Memory`` op a subclass doesn't implement for its shape.

    A collection has no cursored ``log_read``; a log has no keyed ``get``.  The
    base defines every op as a no-op that raises this; each shape overrides the
    ones it supports.  The tool layer catches it and returns the readable
    refusal that points the model at the right tool.  Subclasses
    ``MemoryTypeError`` so existing ``except MemoryTypeError`` callers (the addon
    write handlers) and ``pytest.raises(MemoryTypeError)`` keep working.
    """

    def __init__(self, name: str, shape: str, message: str) -> None:
        super().__init__(message)
        self.name = name
        self.shape = shape


class ReadOnlyMemoryError(MemoryAccessError):
    """Raised when a write is attempted against a derived read-only facade.

    ``user-messages`` / ``penny-messages`` / ``collector-runs`` are views over
    ``messagelog`` / ``promptlog`` — they have no rows of their own to append.
    """


class MemoryNotFoundError(MemoryAccessError):
    """Raised when an operation targets a memory that doesn't exist.

    Carries the ``name`` and renders a readable message, so a tool can surface
    ``str(self)`` directly (``db.memory(name)`` returning ``None`` becomes this
    via ``_resolve``).
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"Memory '{name}' not found.")
        self.name = name


class MemoryAlreadyExistsError(Exception):
    """Raised when a collection or log with the given name already exists.

    Like the access errors, it carries the ``name`` and renders a readable
    message, so a tool surfaces ``str(self)`` directly with no format string.
    Kept distinct from ``MemoryAccessError`` (a creation conflict, not an access
    refusal); the ``MemoryTool`` base catches both and returns ``str(exc)``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"Memory '{name}' already exists.")
        self.name = name


class DedupThresholds(BaseModel):
    """Per-signal strict + relaxed thresholds for the memory dedup rule."""

    key_tcr_strict: float
    key_tcr_relaxed: float
    key_sim_strict: float
    key_sim_relaxed: float
    content_sim_strict: float
    content_sim_relaxed: float

    @classmethod
    def from_runtime(cls, runtime: RuntimeParams) -> DedupThresholds:
        """Read the six dedup thresholds from runtime config."""
        return cls(
            key_tcr_strict=runtime.MEMORY_DEDUP_KEY_TCR_STRICT,
            key_tcr_relaxed=runtime.MEMORY_DEDUP_KEY_TCR_RELAXED,
            key_sim_strict=runtime.MEMORY_DEDUP_KEY_SIM_STRICT,
            key_sim_relaxed=runtime.MEMORY_DEDUP_KEY_SIM_RELAXED,
            content_sim_strict=runtime.MEMORY_DEDUP_CONTENT_SIM_STRICT,
            content_sim_relaxed=runtime.MEMORY_DEDUP_CONTENT_SIM_RELAXED,
        )


class EntryInput(BaseModel):
    """Input row for collection_write — key, content, and optional embeddings."""

    key: str
    content: str
    key_embedding: list[float] | None = None
    content_embedding: list[float] | None = None


class LogEntryInput(BaseModel):
    """Input row for log append — keyless content plus optional embedding."""

    content: str
    content_embedding: list[float] | None = None


WriteOutcome = Literal["written", "duplicate", "rejected"]
MoveOutcome = Literal["ok", "not_found", "collision"]
UpdateOutcome = Literal["ok", "not_found"]


class WriteResult(BaseModel):
    key: str
    outcome: WriteOutcome
    entry_id: int | None = None
    # Existing entry's key when ``outcome == "duplicate"`` — surfaces in
    # the rejection message so the model can pivot to ``update_entry``
    # when it has fresher info for the existing row.
    matched_key: str | None = None
    # Human-readable reason when ``outcome == "rejected"``.
    reason: str | None = None


class EntrySide(NamedTuple):
    """One side of a dedup pair: the key plus its key/content embeddings."""

    key: str | None
    key_vec: list[float] | None
    content_vec: list[float] | None


def slug(name: str) -> str:
    """Normalize a memory name: unicode dash variants → ASCII hyphen, lowercase."""
    return normalize_unicode(name).lower()
