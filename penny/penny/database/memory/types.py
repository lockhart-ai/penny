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

# The change-gate outcome enum (#1587) lives in ``penny.constants`` beside its peer
# ``RunOutcome`` (both surface on the run record and want a cycle-free leaf home);
# imported here so ``WriteResult.outcome`` types against it and the memory layer's
# public surface (``penny.database.memory``) re-exports it beside ``WriteResult``.
from penny.constants import WriteGateOutcome


class MemoryType(StrEnum):
    COLLECTION = "collection"
    LOG = "log"


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
        super().__init__(
            f"Memory '{name}' not found. Check the name (it may be misspelled), or find "
            f"it by meaning with find(query=<what it's about>) — it resolves your "
            f"collections, logs, and skills (archived included) and names the exact tool "
            f"for each. Or create it with collection_create(name='{name}') / "
            f"log_create(name='{name}') if it should exist."
        )
        self.name = name


def wrong_shape_message(name: str, actual_type: str) -> str:
    """The single wrong-shape refusal string — names the value, states its actual
    shape, and binds the read tool that shape *does* support.

    One source for every "collection op on a log" / "log op on a collection"
    refusal, so the base ``Memory`` no-ops (``_refuse_collection_op`` /
    ``_refuse_log_op``) and the collection guards (``_require_collection`` /
    ``_require_destination_collection``) all speak the same house wording instead
    of a bare ``memory '<x>' is a <t>, not a collection``.
    """
    if actual_type == MemoryType.LOG:
        return (
            f"Refused: '{name}' is a log, not a collection.  Read a log with "
            f"log_read('{name}') (recent batch / cursored, oldest-first)."
        )
    return (
        f"Refused: '{name}' is a collection, not a log.  Read a collection with "
        f"collection_read_latest('{name}') / collection_get(memory='{name}', key=<key>) / "
        f"collection_read_random('{name}') / "
        f"read_similar(memory='{name}', anchor=<what you're looking for>)."
    )


class MemoryAlreadyExistsError(Exception):
    """Raised when a collection or log with the given name already exists.

    Like the access errors, it carries the ``name`` and renders a readable
    message, so a tool surfaces ``str(self)`` directly with no format string.
    Kept distinct from ``MemoryAccessError`` (a creation conflict, not an access
    refusal); the ``MemoryTool`` base catches both and returns ``str(exc)``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Memory '{name}' already exists. Use it as-is (write to it directly), or "
            f"choose a different name if you meant to create a separate memory."
        )
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


MoveOutcome = Literal["ok", "not_found", "collision"]
UpdateOutcome = Literal["ok", "not_found"]


class WriteResult(BaseModel):
    key: str
    # The change-gate outcome (#1587) — one member of the closed ``WriteGateOutcome``
    # union, computed deterministically by the write path (never a model judgment).
    outcome: WriteGateOutcome
    entry_id: int | None = None
    # The existing entry's key that this write resolved against — its own key for an
    # exact-key hit (``KEY_EXISTS_CHANGED`` / ``KEY_EXISTS_UNCHANGED``), or the
    # colliding key for a ``DUPLICATE``.  Surfaces in the result so the model can
    # pivot to ``update_entry`` when it has fresher info for the existing row.
    matched_key: str | None = None
    # Human-readable reason when ``outcome == WriteGateOutcome.DEGENERATE``.
    reason: str | None = None


class EntrySide(NamedTuple):
    """One side of a dedup pair: the key plus its key/content embeddings."""

    key: str | None
    key_vec: list[float] | None
    content_vec: list[float] | None


class ResolvedKind(StrEnum):
    """The family of an addressable thing ``find`` resolves (#1558, #1640).

    The axis that fixes the thing's *finite action set*: the tool layer maps
    ``kind`` (+ archived state) to the exact tool call that operates on it, so the
    model never derives the type→addressing mapping itself.  ``skill`` is a taught
    skill in the ``skill`` table (the sole skills store, #1624);
    ``collection`` / ``log`` are registry rows; ``entry`` is a single stored
    ``memory_entry`` (a fact inside a collection or log — #1640).  An entry's
    *container* is always a ``collection`` or ``log`` (never an ``entry``/``skill``),
    the axis that fixes how the entry is read back.
    """

    COLLECTION = "collection"
    LOG = "log"
    SKILL = "skill"
    ENTRY = "entry"


class ResolvedMatch(BaseModel):
    """One resolve-by-meaning hit on an *object* — identity fused with its
    affordances (#1558).  The collection/log/skill arm of a :data:`ResolvedHit`;
    the entry arm is :class:`ResolvedEntry`.

    Carries the *exact* identity (``name`` — a collection/log name, or a taught
    skill's name), its ``kind``, whether it's ``archived`` (collections/logs; a
    skill is always live), and a one-line ``label`` (the description) for
    disambiguation.  The tool renders identity + state
    + the deterministic addressing from these fields; ranking (plain cosine) is
    the store's, so no score is carried here.
    """

    name: str
    kind: ResolvedKind
    archived: bool
    label: str


class ResolvedEntry(BaseModel):
    """One resolve-by-meaning hit on a *stored entry* (#1640) — the entry arm of a
    :data:`ResolvedHit` (the object arm is :class:`ResolvedMatch`).

    A fact stored in a collection or log, found by the meaning of its content or
    key rather than by guessing its container: the hit CARRIES the content, so for
    a short fact the find IS the answer.  ``memory_name`` is the containing
    collection/log; ``container_kind`` (``collection`` | ``log``) fixes how the
    entry is read back; ``key`` is the invocation-form lookup key (``None`` for a
    keyless log entry, which is addressed by its ``entry_id`` handle instead);
    ``content`` is the stored value the tool previews.  Ranking (plain cosine over
    the entry's content/key embeddings) is the store's, so no score is carried
    here.
    """

    entry_id: int
    memory_name: str
    container_kind: ResolvedKind
    key: str | None
    content: str


# The two arms of one best-first ``find`` result list (#1640): an object hit
# (collection / log / skill) or a stored-entry hit.  The tool renders each by its
# own house pattern but ranks them in ONE fused list — same embedding space, one
# cosine, no second result shape.
ResolvedHit = ResolvedMatch | ResolvedEntry


def slug(name: str) -> str:
    """Normalize a memory name: unicode dash variants → ASCII hyphen, lowercase."""
    return normalize_unicode(name).lower()


def render_key(key: str) -> str:
    """Render an entry key in **invocation form** — ``key='<key>'`` — for every
    model-facing entry render.

    The displayed form IS the form a key-taking tool accepts, so the model
    copies what it reads straight into a valid ``key=`` argument.  The single
    source of the convention: the entry-list renders and the chat recall
    headers all call this, so the form can't
    partially revert to the old copy-hostile ``[key]`` display (whose brackets
    the model pasted verbatim into key args — the eval contract in
    ``tests/eval/test_key_render.py`` guards the behaviour).
    """
    return f"key='{key}'"


def strip_display_brackets(key: str) -> str:
    """Strip one layer of enclosing display brackets from an entry key.

    Entry lists used to render an entry as ``[key] content`` — the brackets were
    *display framing*, not part of the key — and the model copied that rendered
    form back into a later key argument (``key="[foo]"``).  The render now shows
    keys in invocation form (:func:`render_key`), but the model's ingrained
    bracket habit persists, so the guard stays.  Lookups stay strictly exact:
    this helper never rewrites what a lookup searches for.  It exists so the
    key-taking tools can *detect* the copied display form on a miss and reject
    with a teaching error that names the bare key to reuse.  Strips exactly ONE
    enclosing ``[...]`` layer (``[[k]]`` → ``[k]``); a key with no enclosing
    brackets is returned unchanged.
    """
    if len(key) > 2 and key.startswith("[") and key.endswith("]"):
        return key[1:-1]
    return key
