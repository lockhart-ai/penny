"""The memory layer — polymorphic ``Memory`` objects + the ``MemoryStore`` registry.

Import surface: callers ask ``db.memory(name)`` for a ``Memory`` and call methods
on it (the object refuses ops that don't fit its shape / read-only-ness), and
``db.memories`` for registry operations (create/list/archive, ``exists``,
backfill).  The shapes and facades live in :mod:`objects`; the registry and
dispatch in :mod:`store`; shared value types in :mod:`types`.
"""

from penny.database.memory.objects import (
    Collection,
    Log,
    Memory,
    MessageLogMemory,
    RunHealth,
    RunLog,
    classify_run,
    render_run_calls,
    render_run_record,
    render_tool_call,
)
from penny.database.memory.store import MemoryStore
from penny.database.memory.types import (
    DedupThresholds,
    EntryInput,
    EntrySide,
    Inclusion,
    LogEntryInput,
    MemoryAccessError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    MoveOutcome,
    ReadOnlyMemoryError,
    RecallMode,
    UpdateOutcome,
    WriteOutcome,
    WriteResult,
    WrongShapeError,
    slug,
)
from penny.text_validity import (
    degenerate_reason,
    half_formed_send_reason,
    is_blank,
)

__all__ = [
    "Collection",
    "DedupThresholds",
    "degenerate_reason",
    "half_formed_send_reason",
    "EntryInput",
    "EntrySide",
    "Inclusion",
    "is_blank",
    "Log",
    "LogEntryInput",
    "Memory",
    "MemoryAccessError",
    "MemoryAlreadyExistsError",
    "MemoryNotFoundError",
    "MemoryStore",
    "MemoryType",
    "MemoryTypeError",
    "MessageLogMemory",
    "MoveOutcome",
    "ReadOnlyMemoryError",
    "RecallMode",
    "RunHealth",
    "RunLog",
    "classify_run",
    "render_run_record",
    "render_run_calls",
    "render_tool_call",
    "UpdateOutcome",
    "WriteOutcome",
    "WriteResult",
    "WrongShapeError",
    "slug",
]
