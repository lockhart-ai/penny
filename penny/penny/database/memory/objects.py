"""The polymorphic ``Memory`` objects — one class per shape and backing store.

A *memory* is Penny's data primitive: a named, typed container of entries the
tools read and write.  This module is the single place that says *how each
operation is implemented for each kind of memory* — so the tool layer never
branches on a name or a shape; it gets a ``Memory`` from ``db.memory(name)`` and
calls a method, and the object either does the work or refuses (wrong shape /
read-only) via a base no-op.

    Memory (base)            memory_entry row primitives + shared similarity /
    │                        cursor reads; every shape op is a no-op that
    │                        raises WrongShapeError (overridden where it applies)
    ├─ Collection            keyed: get / keys / read_latest / read_random /
    │                        write / update / move / delete  (log ops refuse)
    └─ Log                   stream: read_batch (cursored) / read_window /
        │                    append  (keyed ops refuse)
        ├─ MessageLogMemory  row primitives → ``messagelog`` (user-/penny-
        │                    messages); append refuses (derived, read-only)
        └─ RunLog            row primitives → ``promptlog`` rendered as run
                             records (collector-runs); append refuses

``Collection`` / ``Log`` ARE the ``memory_entry``-backed ("stored")
implementations — the native backing lives on the base, so it sits as high as
it goes; the facades are the exceptions that override the row primitives to read
their canonical tables.  Nothing about how those tables are stored changes — the
facades are read views.

Similarity reads require pre-computed embeddings passed in by the caller; this
layer stays synchronous (the tool layer owns async embedding).
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, computed_field
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants, RunOutcome
from penny.database.memory import _similarity as sim
from penny.database.memory.types import (
    DedupThresholds,
    EntryInput,
    EntrySide,
    LogEntryInput,
    MemoryType,
    MemoryTypeError,
    MoveOutcome,
    ReadOnlyMemoryError,
    UpdateOutcome,
    WriteResult,
    WrongShapeError,
    slug,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog
from penny.text_validity import degenerate_reason, half_formed_send_reason, is_low_info
from penny.validation.conditions import ConditionKey, run_flag_conditions

logger = logging.getLogger(__name__)


class Memory:
    """Base memory: ``memory_entry`` row access + shape-independent reads.

    Never instantiated directly — ``db.memory(name)`` returns a ``Collection``,
    ``Log``, or a facade.  Holds the metadata ``row`` and exposes it as
    properties; defines every shape op as a no-op that raises ``WrongShapeError``
    so a subclass that doesn't serve that shape refuses with a readable message.
    """

    def __init__(self, row: MemoryRow, engine, *, on_changed=None) -> None:
        self.row = row
        self._engine = engine
        self._on_changed = on_changed

    # ── Metadata passthroughs ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.row.name

    @property
    def type(self) -> str:
        return self.row.type

    @property
    def description(self) -> str:
        return self.row.description

    @property
    def inclusion(self) -> str:
        return self.row.inclusion

    @property
    def recall(self) -> str:
        return self.row.recall

    @property
    def intent(self) -> str | None:
        return self.row.intent

    @property
    def archived(self) -> bool:
        return self.row.archived

    @property
    def description_embedding(self) -> bytes | None:
        return self.row.description_embedding

    @property
    def is_log(self) -> bool:
        return self.row.type == MemoryType.LOG

    def _session(self) -> Session:
        return Session(self._engine)

    def _notify(self, name: str | None = None) -> None:
        """Fire the change callback for ``name`` (default this memory)."""
        if self._on_changed is not None:
            self._on_changed(name if name is not None else self.name)

    # ── Shape-op refusals (overridden by the shape that serves them) ──────────

    def _refuse_collection_op(self) -> None:
        raise WrongShapeError(
            self.name,
            self.type,
            f"Refused: '{self.name}' is a log, not a collection.  Read a log with "
            f"log_read (recent batch / cursored, oldest-first).",
        )

    def _refuse_log_op(self) -> None:
        raise WrongShapeError(
            self.name,
            self.type,
            f"Refused: '{self.name}' is a collection, not a log.  Read a collection with "
            f"collection_read_latest / collection_get / collection_read_random / read_similar.",
        )

    def get(self, key: str) -> list[MemoryEntry]:
        self._refuse_collection_op()
        return []

    def keys(self) -> list[str]:
        self._refuse_collection_op()
        return []

    def read_latest(
        self, k: int | None = None, offset: int = 0, search: str | None = None
    ) -> list[MemoryEntry]:
        self._refuse_collection_op()
        return []

    def read_random(self, k: int | None = None) -> list[MemoryEntry]:
        self._refuse_collection_op()
        return []

    def write(
        self, entries: list[EntryInput], author: str, thresholds: DedupThresholds | None = None
    ) -> list[WriteResult]:
        self._refuse_collection_op()
        return []

    def update(self, key: str, content: str, author: str) -> UpdateOutcome:
        self._refuse_collection_op()
        return "not_found"

    def move(self, key: str, to_name: str, author: str) -> MoveOutcome:
        self._refuse_collection_op()
        return "not_found"

    def delete(self, key: str) -> int:
        self._refuse_collection_op()
        return 0

    def append(self, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        self._refuse_log_op()
        return []

    def read_batch(self, cursor: datetime | None, limit: int) -> list[MemoryEntry]:
        self._refuse_log_op()
        return []

    def read_window(self, window_seconds: int, limit: int | None = None) -> list[MemoryEntry]:
        self._refuse_log_op()
        return []

    # ── Shared reads (every shape, via the row primitives below) ──────────────

    def newest_entries(
        self, k: int | None = None, offset: int = 0, search: str | None = None
    ) -> list[MemoryEntry]:
        """Newest-first entries — the shape-independent internal read used by
        recall (recent mode), the log cursor's first read, and send_message's
        cooldown probe.  The model-facing ``collection_read_latest`` is the
        ``Collection.read_latest`` wrapper over this; logs reach it only through
        ``read_batch``."""
        return self._newest_rows(k, offset, search)

    def read_all(self) -> list[MemoryEntry]:
        return self._all_rows()

    def read_since(self, cursor: datetime, cap: int | None = None) -> list[MemoryEntry]:
        return self._rows_since(cursor, cap)

    def read_recent(self, window_seconds: int, cap: int | None = None) -> list[MemoryEntry]:
        cutoff = datetime.fromtimestamp(datetime.now(UTC).timestamp() - window_seconds, tz=UTC)
        return self._rows_since(cutoff, cap)

    def read_similar(
        self, anchor: list[float], k: int | None = None, floor: float = 0.0
    ) -> list[MemoryEntry]:
        """Entries similar to ``anchor`` by cosine, with the adaptive
        cluster-strength cutoff suppressing flat noise plateaus.  Entries
        without a content embedding are skipped; ``k=None`` returns all that
        clear the cutoff."""
        scored = [
            (row, row.content_embedding)
            for row in self._embedded_rows()
            if row.id is not None and row.content_embedding is not None
        ]
        if not scored:
            return []
        valid = [row for row, _ in scored]
        scores = sim.score_against_anchors([blob for _, blob in scored], [anchor])
        order = list(np.argsort(-scores))
        cutoff = sim.adaptive_cutoff([float(scores[i]) for i in order], floor)
        if cutoff is None:
            return []
        ordered = [valid[i] for i in order if float(scores[i]) >= cutoff]
        return ordered if k is None else ordered[:k]

    def read_similar_hybrid(
        self,
        conversation_anchors: list[list[float]],
        query_text: str,
        k: int | None = None,
        exclude_contents: set[str] | None = None,
    ) -> list[MemoryEntry]:
        """Stage-2 hybrid ranking: embedding cosine fused with IDF-lexical
        coverage (RRF), top-``k``, no relevance floor — stage-1 inclusion
        already decided this memory is relevant.

        ``exclude_contents`` drops corpus rows matching the anchors (channel
        ingress writes the current message/history into log memories, so an
        anchor would otherwise self-match at cosine ≈ 1.0 and dominate).
        Log-shaped memories also drop low-information rows; collections keep
        their deliberately short keyed entries."""
        if not conversation_anchors:
            return []
        rows = self._embedded_rows()
        if exclude_contents:
            rows = [row for row in rows if row.content not in exclude_contents]
        if self.is_log:
            rows = [row for row in rows if not is_low_info(row.content)]
        valid = [
            (row, row.content_embedding, row.id)
            for row in rows
            if row.id is not None and row.content_embedding is not None
        ]
        if not valid:
            return []
        ranked_ids = sim.hybrid_rank_ids(
            [blob for _, blob, _ in valid],
            [row.content for row, _, _ in valid],
            [entry_id for _, _, entry_id in valid],
            conversation_anchors,
            query_text,
        )
        by_id = {entry_id: row for row, _, entry_id in valid}
        ranked = [by_id[entry_id] for entry_id in ranked_ids]
        return ranked if k is None else ranked[:k]

    def expand_with_temporal_neighbors(
        self, hits: list[MemoryEntry], window_minutes: int, per_hit_cap: int | None = None
    ) -> list[MemoryEntry]:
        """Collections have no conversational timeline — return hits unchanged.
        Overridden by ``Log`` to pull in surrounding entries of each hit."""
        return hits

    # ── Row primitives (memory_entry; facades override) ───────────────────────

    def _all_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .where(MemoryEntry.memory_name == self.name)
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == self.name)
                .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
            )
            if search:
                like = f"%{search}%"
                query = query.where(
                    MemoryEntry.content.like(like)  # ty: ignore[unresolved-attribute]
                    | MemoryEntry.key.like(like)  # ty: ignore[unresolved-attribute]
                )
            if offset:
                query = query.offset(offset)
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def _rows_since(self, cursor: datetime, cap: int | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == self.name, MemoryEntry.created_at > cursor)
                .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
            )
            if cap is not None:
                query = query.limit(cap)
            return list(session.exec(query).all())

    def _embedded_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == self.name,
                        MemoryEntry.content_embedding.is_not(None),  # type: ignore[union-attr]
                    )
                ).all()
            )

    def _rows_in_window(self, start: datetime, end: datetime) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .where(
                        MemoryEntry.memory_name == self.name,
                        MemoryEntry.created_at >= start,
                        MemoryEntry.created_at <= end,
                    )
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )


class Collection(Memory):
    """A keyed collection — similarity-deduped writes, exact-key lookup.

    Backed by ``memory_entry`` (the base's row primitives).  Implements the
    keyed/write surface; the log ops stay refused via the base no-ops.
    """

    def __init__(self, row: MemoryRow, engine, *, runtime: RuntimeParams, on_changed=None) -> None:
        super().__init__(row, engine, on_changed=on_changed)
        self._runtime = runtime

    def read_latest(
        self, k: int | None = None, offset: int = 0, search: str | None = None
    ) -> list[MemoryEntry]:
        """The model-facing newest-first collection read (``collection_read_latest``).
        Logs don't expose this — the base refuses it so a log read can't bypass
        its cursor."""
        return self.newest_entries(k, offset, search)

    def read_random(self, k: int | None = None) -> list[MemoryEntry]:
        rows = self._all_rows()
        if k is None or len(rows) <= k:
            return rows
        return random.sample(rows, k)

    def get(self, key: str) -> list[MemoryEntry]:
        with self._session() as session:
            return self._rows_by_key(session, self.name, key)

    def keys(self) -> list[str]:
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MemoryEntry.key)
                    .where(
                        MemoryEntry.memory_name == self.name,
                        MemoryEntry.key.is_not(None),  # type: ignore[union-attr]
                    )
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for key in rows:
            if key is None or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def write(
        self, entries: list[EntryInput], author: str, thresholds: DedupThresholds | None = None
    ) -> list[WriteResult]:
        """Write entries with per-entry similarity dedup.  One ``WriteResult``
        per input; dedup runs against the existing corpus using the configured
        (or default) thresholds."""
        thresholds = thresholds or DedupThresholds.from_runtime(self._runtime)
        existing = self._entries_with_vectors()
        results: list[WriteResult] = []
        with self._session() as session:
            for entry in entries:
                results.append(self._write_one(session, entry, author, existing, thresholds))
            session.commit()
        if any(result.outcome == "written" for result in results):
            self._notify()
        return results

    def update(self, key: str, content: str, author: str) -> UpdateOutcome:
        """Replace the content of every entry with ``key`` (dedup keeps it to one)."""
        with self._session() as session:
            rows = self._rows_by_key(session, self.name, key)
            if not rows:
                return "not_found"
            for row in rows:
                row.content = content
                row.author = author
                session.add(row)
            session.commit()
        self._notify()
        return "ok"

    def move(self, key: str, to_name: str, author: str) -> MoveOutcome:
        """Move every entry with ``key`` into another collection.  ``collision``
        when the destination already has that key."""
        to_name = slug(to_name)
        self._require_destination_collection(to_name)
        with self._session() as session:
            src_rows = self._rows_by_key(session, self.name, key)
            if not src_rows:
                return "not_found"
            if self._rows_by_key(session, to_name, key):
                return "collision"
            for row in src_rows:
                row.memory_name = to_name
                row.author = author
                session.add(row)
            session.commit()
        self._notify()
        self._notify(to_name)
        return "ok"

    def delete(self, key: str) -> int:
        with self._session() as session:
            rows = self._rows_by_key(session, self.name, key)
            for row in rows:
                session.delete(row)
            session.commit()
        if rows:
            self._notify()
        return len(rows)

    def _require_destination_collection(self, to_name: str) -> None:
        with self._session() as session:
            row = session.get(MemoryRow, to_name)
        if row is None:
            raise MemoryTypeError(f"memory '{to_name}' does not exist")
        if row.type != MemoryType.COLLECTION:
            raise MemoryTypeError(f"memory '{to_name}' is a {row.type}, not a collection")

    @staticmethod
    def _rows_by_key(session: Session, name: str, key: str) -> list[MemoryEntry]:
        return list(
            session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == name, MemoryEntry.key == key)
            ).all()
        )

    def _entries_with_vectors(self) -> list[EntrySide]:
        """Every entry as an ``EntrySide`` (key + key/content vectors) for dedup."""
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == self.name)).all()
            )
        return [
            EntrySide(
                row.key,
                sim.maybe_deserialize(row.key_embedding),
                sim.maybe_deserialize(row.content_embedding),
            )
            for row in rows
        ]

    def _write_one(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        existing: list[EntrySide],
        thresholds: DedupThresholds,
    ) -> WriteResult:
        rejection_reason = degenerate_reason(entry.content)
        if rejection_reason is not None:
            logger.debug("Rejected degenerate collection entry %r: %s", entry.key, rejection_reason)
            return WriteResult(key=entry.key, outcome="rejected", reason=rejection_reason)
        candidate = EntrySide(entry.key, entry.key_embedding, entry.content_embedding)
        matched = sim.is_duplicate(candidate, existing, thresholds)
        if matched is not None:
            return WriteResult(key=entry.key, outcome="duplicate", matched_key=matched.key)
        row = MemoryEntry(
            memory_name=self.name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=sim.maybe_serialize(entry.key_embedding),
            content_embedding=sim.maybe_serialize(entry.content_embedding),
            created_at=datetime.now(UTC),
        )
        session.add(row)
        session.flush()
        existing.append(candidate)
        return WriteResult(key=entry.key, outcome="written", entry_id=row.id)


class Log(Memory):
    """An append-only stream — keyless entries in time order.

    Backed by ``memory_entry`` (the base's row primitives).  Implements the
    cursored/window read surface and ``append``; keyed ops stay refused.  The
    cursor *read* logic lives here, uniform across every log backing — the
    reader's pending/commit lifecycle stays in ``LogReadTool``.
    """

    def append(self, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        created: list[MemoryEntry] = []
        with self._session() as session:
            for entry in entries:
                row = MemoryEntry(
                    memory_name=self.name,
                    key=None,
                    content=entry.content,
                    author=author,
                    key_embedding=None,
                    content_embedding=sim.maybe_serialize(entry.content_embedding),
                    created_at=datetime.now(UTC),
                )
                session.add(row)
                created.append(row)
            session.commit()
            for row in created:
                session.refresh(row)
        if created:
            self._notify()
        return created

    def read_batch(self, cursor: datetime | None, limit: int) -> list[MemoryEntry]:
        """The next bounded batch since ``cursor`` (oldest-first), or the
        most-recent ``limit`` on a first read (no cursor).  Uniform for every
        log backing; the reader tracks/commits the cursor itself."""
        if cursor is None:
            return list(reversed(self.newest_entries(k=limit)))
        return self.read_since(cursor, limit)

    def read_window(self, window_seconds: int, limit: int | None = None) -> list[MemoryEntry]:
        """Recent entries within a short look-back window, oldest-first."""
        return self.read_recent(window_seconds, limit)

    def expand_with_temporal_neighbors(
        self, hits: list[MemoryEntry], window_minutes: int, per_hit_cap: int | None = None
    ) -> list[MemoryEntry]:
        """Augment each hit with entries within ±``window_minutes`` of it — so a
        single keyword match pulls in its surrounding conversation rather than a
        line stripped of context.  Union, deduped by id, chronological.

        ``per_hit_cap`` bounds each hit's window to its ``cap`` nearest-in-time
        entries (the hit included).  Without it the expansion is unbounded — a
        dense burst around a hit drags every entry in it into the prompt."""
        if not hits:
            return []
        delta = timedelta(minutes=window_minutes)
        seen_ids: set[int] = set()
        expanded: list[MemoryEntry] = []
        for hit in hits:
            window = self._rows_in_window(hit.created_at - delta, hit.created_at + delta)
            if per_hit_cap is not None and len(window) > per_hit_cap:
                window = self._nearest_in_time(window, hit.created_at, per_hit_cap)
            for row in window:
                if row.id is not None and row.id not in seen_ids:
                    seen_ids.add(row.id)
                    expanded.append(row)
        expanded.sort(key=lambda row: row.created_at)
        return expanded

    @staticmethod
    def _nearest_in_time(rows: list[MemoryEntry], pivot: datetime, cap: int) -> list[MemoryEntry]:
        """The ``cap`` entries closest in time to ``pivot`` (the hit, distance 0,
        is always among them)."""
        return sorted(rows, key=lambda row: abs(row.created_at - pivot))[:cap]


class MessageLogMemory(Log):
    """Read facade over ``messagelog`` for ``user-messages`` / ``penny-messages``.

    A message has two conversational authors — the user (incoming) or Penny
    (outgoing), which IS the direction; the internal agent that produced a Penny
    message isn't a conversational author, so it isn't recorded here.  Reactions
    are excluded (not conversation).  ``content_embedding`` is the messagelog
    embedding (same serialized-bytes format), written at ingress/egress and
    backfilled at startup.  Read-only — the channel owns the canonical writes.
    """

    def __init__(self, row: MemoryRow, engine, *, direction: str, on_changed=None) -> None:
        super().__init__(row, engine, on_changed=on_changed)
        self._direction = direction
        self._author = (
            PennyConstants.MessageAuthor.USER
            if direction == PennyConstants.MessageDirection.INCOMING
            else PennyConstants.MessageAuthor.PENNY
        )

    def append(self, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        raise ReadOnlyMemoryError(
            f"'{self.name}' is a read view over the conversation log — messages "
            "are recorded by the channel, not appended here."
        )

    def _to_entry(self, row: MessageLog) -> MemoryEntry:
        return MemoryEntry(
            id=row.id,
            memory_name=self.name,
            key=None,
            content=row.content,
            author=self._author,
            key_embedding=None,
            content_embedding=row.embedding,
            created_at=row.timestamp,
        )

    def _select(self):
        return select(MessageLog).where(
            MessageLog.direction == self._direction,
            MessageLog.is_reaction.is_(False),  # ty: ignore[unresolved-attribute]
        )

    def _all_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            rows = session.exec(self._select().order_by(MessageLog.timestamp.asc())).all()
        return [self._to_entry(row) for row in rows]

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = self._select().order_by(MessageLog.timestamp.desc())
            if search:
                query = query.where(MessageLog.content.like(f"%{search}%"))  # ty: ignore[union-attr]
            if offset:
                query = query.offset(offset)
            if k is not None:
                query = query.limit(k)
            rows = session.exec(query).all()
        return [self._to_entry(row) for row in rows]

    def _rows_since(self, cursor: datetime, cap: int | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                self._select()
                .where(MessageLog.timestamp > cursor)
                .order_by(MessageLog.timestamp.asc())
            )
            if cap is not None:
                query = query.limit(cap)
            rows = session.exec(query).all()
        return [self._to_entry(row) for row in rows]

    def _embedded_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            rows = session.exec(
                self._select().where(MessageLog.embedding.is_not(None))  # ty: ignore[union-attr]
            ).all()
        return [self._to_entry(row) for row in rows]

    def _rows_in_window(self, start: datetime, end: datetime) -> list[MemoryEntry]:
        with self._session() as session:
            rows = session.exec(
                self._select()
                .where(MessageLog.timestamp >= start, MessageLog.timestamp <= end)
                .order_by(MessageLog.timestamp.asc())
            ).all()
        return [self._to_entry(row) for row in rows]


def render_tool_call(name: str, args: object) -> str:
    """Compact, grokkable render of one tool call (the salient args only).

    The single shared format for the ``collector-runs`` run trace, so the model
    reads what a cycle did the same way everywhere.  Reads render under their real
    tool name (``collection_read_latest('x')`` / ``log_read('y')``) so a
    wrong-shape or unknown-tool call is identifiable.  Content is never truncated.
    """
    fields = cast("dict[str, Any]", args if isinstance(args, dict) else {})
    if name == "collection_write":
        return f"write({fields.get('memory', '?')}, {_write_contents(fields)!r})"
    if name == "update_entry":
        return f"update({fields.get('memory', '?')}, {fields.get('key', '?')!r})"
    if name == "send_message":
        return f"send({fields.get('content', '')!r})"
    if name == "browse":
        return f"browse({fields.get('queries', list(fields.values()))!r})"
    rendered = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    return f"{name}({rendered or repr(args)})"


def _parse_tool_args(function: dict) -> object:
    """Deserialize a stored tool call's ``arguments`` (a JSON string) to a dict.

    Falls back to the raw string when the model emitted malformed JSON, so the
    trace still shows what it tried rather than dropping the call.
    """
    raw = function.get("arguments")
    if not isinstance(raw, str) or not raw:
        return raw or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _write_contents(fields: dict) -> str:
    """The entry contents of a ``collection_write`` call, joined for a run record."""
    entries = fields.get("entries") or []
    return "; ".join(str(e.get("content", "")) for e in entries if isinstance(e, dict))


def _run_outcome(prompts: list[PromptLog]) -> tuple[str | None, str | None, str | None]:
    """Outcome/reason/target from the last prompt carrying them."""
    for prompt in reversed(prompts):
        if prompt.run_outcome is not None or prompt.run_reason:
            return prompt.run_outcome, prompt.run_reason, prompt.run_target
    return None, None, prompts[0].run_target if prompts else None


def _run_tool_calls(prompts: list[PromptLog]) -> list[tuple[str, object]]:
    """Ordered (name, arguments) for every tool call across the run's prompts."""
    calls: list[tuple[str, object]] = []
    for prompt in prompts:
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                calls.append((function.get("name") or "?", _parse_tool_args(function)))
    return calls


def _run_tool_failures(prompts: list[PromptLog]) -> int:
    """The persisted failed-tool-call count, off the run's outcome-bearing row.

    Stamped by ``set_run_outcome`` on the last prompt; NULL on old/untagged rows
    reads as zero (not measured)."""
    for prompt in reversed(prompts):
        if prompt.run_outcome is not None:
            return prompt.tool_failures or 0
    return 0


def _is_degenerate_send(content: str) -> bool:
    """True if a sent message has no real content the user should have received.

    The flag side of the shared :func:`half_formed_send_reason` rule — the same
    test the ``send_message`` tool gates on before delivery."""
    return half_formed_send_reason(content) is not None


class RunHealth(BaseModel):
    """Structural failure signals for one collector run, derived from its
    ``promptlog`` rows.  One classifier feeds both Penny's own self-review (the
    run record her ``quality`` collector reads) and the addon's prompts tab
    (badges + the "flagged only" filter) — what we use to judge whether a run
    regressed is exactly what Penny sees of it.

    All signals are deterministic and read from stored data — no model judgment:
    ``bailed`` (did no real work — reached ``done()`` / made no tool call without
    any read/write/browse first), ``incomplete`` (hit the step ceiling without a
    closing ``done()``), ``tool_failures`` (count of failed tool calls in the
    run), ``degenerate_send`` (a message went out with no real content)."""

    bailed: bool = False
    incomplete: bool = False
    tool_failures: int = 0
    degenerate_send: bool = False

    @computed_field
    @property
    def flags(self) -> list[str]:
        """Stable flag keys for badges / filtering, in render order."""
        out: list[str] = []
        if self.bailed:
            out.append(ConditionKey.NO_WORK_DONE.value)
        if self.incomplete:
            out.append(ConditionKey.INCOMPLETE.value)
        if self.tool_failures:
            out.append(ConditionKey.TOOL_FAILURES.value)
        if self.degenerate_send:
            out.append(ConditionKey.HALF_FORMED_SEND.value)
        return out

    @computed_field
    @property
    def regressive(self) -> bool:
        """Whether this run is a candidate to review (any flag set)."""
        return bool(self.flags)


def classify_run(prompts: list[PromptLog]) -> RunHealth:
    """The shared run-health determination over a run's ``promptlog`` rows."""
    if not prompts:
        return RunHealth()
    outcome, _reason, _target = _run_outcome(prompts)
    calls = _run_tool_calls(prompts)
    non_done = [name for name, _ in calls if name != "done"]
    bailed = outcome in (RunOutcome.NO_WORK.value, RunOutcome.FAILED.value) and not non_done
    degenerate = any(
        name == "send_message" and _is_degenerate_send(_send_content(args)) for name, args in calls
    )
    return RunHealth(
        bailed=bailed,
        incomplete=outcome == RunOutcome.INCOMPLETE.value,
        tool_failures=_run_tool_failures(prompts),
        degenerate_send=degenerate,
    )


def _send_content(args: object) -> str:
    """The ``content`` of a ``send_message`` call's parsed arguments."""
    fields = cast("dict[str, Any]", args if isinstance(args, dict) else {})
    content = fields.get("content")
    return str(content) if content is not None else ""


# Which ``RunHealth`` field gates each run-flag condition, in canonical render
# order — the marker/detail text itself lives once in the shared catalog
# (``penny.validation.conditions``), so the render never spells the ``⚠`` lines
# inline and the seeded ``quality`` prompt + the addon TS mirror stay in lockstep.
def _flag_is_set(health: RunHealth, key: ConditionKey) -> bool:
    return {
        ConditionKey.NO_WORK_DONE: health.bailed,
        ConditionKey.INCOMPLETE: health.incomplete,
        ConditionKey.TOOL_FAILURES: bool(health.tool_failures),
        ConditionKey.HALF_FORMED_SEND: health.degenerate_send,
    }[key]


def _flag_line(health: RunHealth, condition_key: ConditionKey, marker: str, detail: str) -> str:
    """One ⚠ line — ``{marker} — {detail}``, with the failure count inserted
    after the marker for TOOL FAILURES (``⚠ TOOL FAILURES (n) — …``)."""
    if condition_key == ConditionKey.TOOL_FAILURES:
        return f"{marker} ({health.tool_failures}) — {detail}"
    return f"{marker} — {detail}"


def _health_lines(health: RunHealth) -> list[str]:
    """The ⚠ explanation lines for a run record, one per set flag, in order.

    The verbose form Penny's ``quality`` collector reads; the addon derives its
    compact badges from the same ``RunHealth.flags``.  Marker + detail come from
    the shared catalog so the run record and the quality prompt name one text."""
    return [
        _flag_line(health, entry.key, entry.marker, entry.detail)
        for entry in run_flag_conditions()
        if entry.marker is not None and entry.detail is not None and _flag_is_set(health, entry.key)
    ]


def render_run_record(prompts: list[PromptLog]) -> str:
    """One run as ``[target] summary`` + its health flags + tool-call trace.

    The single representation shared by Penny's self-review (the ``collector-runs``
    record her ``quality`` collector reads) and the addon's prompts tab.
    ``classify_run`` determines the run's health; ``_health_lines`` renders the ⚠
    flags (the addon derives its compact badges from the same ``RunHealth``).  The
    trace shows:

    - **bailed** — the single meagre call (or ``(no tool calls)``): the run jumped
      to ``done()`` / acted not at all, so there's nothing but the bail to see;
    - **worked / incomplete / tool-failure / half-formed-send** — the full ordered
      non-``done()`` trace, so the work (or the failing/degenerate call) can be
      judged in context;
    - **everything else** (a quiet cycle that DID read, a failed/cancelled run that
      DID call real tools with no new flag) — header-only, no trace to tempt an
      over-correction.

    Content is never truncated."""
    if not prompts:
        return "[?] (no data)"
    health = classify_run(prompts)
    outcome, reason, target = _run_outcome(prompts)
    header = f"[{target or '?'}] {reason or outcome or ''}".rstrip()
    lines = [header, *_health_lines(health)]
    calls = _run_tool_calls(prompts)
    non_done = [(name, args) for name, args in calls if name != "done"]
    if health.bailed:
        lines.append("(no tool calls)" if not calls else render_tool_call(*calls[0]))
    elif outcome == RunOutcome.WORKED.value or health.regressive:
        lines.extend(render_tool_call(name, args) for name, args in non_done)
    return "\n".join(lines)


class RunLog(Log):
    """Read facade over ``promptlog`` for the ``collector-runs`` log.

    ``collector-runs`` stores nothing of its own: every collector cycle already
    lives in ``promptlog`` (``run_id`` / ``run_target`` / ``run_outcome`` and its
    tool calls).  ``set_run_outcome`` stamps ``run_outcome`` on exactly one row
    per run (its last prompt), so the completion rows ARE the run index — one row
    per run, its timestamp the completion time, served by the
    ``ix_promptlog_completed_runs`` partial index (a bounded ``ORDER BY ... LIMIT``,
    not a ``GROUP BY`` scan).  Each run renders to a model-readable record: a
    ``[target] summary`` header plus, for a run that did something, its compact
    tool-call trace.  Read-only; has no embeddings (so ``read_similar`` is empty
    and it never enters relevant recall — ``collector-runs`` is inclusion=never).

    ``target`` scopes to one collection's runs (the addon's per-collection
    panel); ``None`` is every collector run (the log itself).
    """

    def __init__(
        self, row: MemoryRow, engine, *, target: str | None = None, on_changed=None
    ) -> None:
        super().__init__(row, engine, on_changed=on_changed)
        self._target = target

    def append(self, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        raise ReadOnlyMemoryError(
            f"'{self.name}' is a read view over collector run history (promptlog) — "
            "runs are recorded by the dispatcher, not appended here."
        )

    def read_window(self, window_seconds: int, limit: int | None = None) -> list[MemoryEntry]:
        # Runs are bounded per read like the cursored batch — a chat "what ran
        # recently" shouldn't dump an unbounded window of run records.
        return self.read_recent(window_seconds, limit or PennyConstants.LOG_READ_LIMIT)

    def _completion_clauses(self) -> list:
        clauses: list = [PromptLog.run_outcome.isnot(None)]  # ty: ignore[unresolved-attribute]
        if self._target is not None:
            clauses.append(PromptLog.run_target == self._target)
        else:
            clauses.append(PromptLog.run_target.isnot(None))  # ty: ignore[unresolved-attribute]
        return clauses

    def _all_rows(self) -> list[MemoryEntry]:
        return self._records(newest_first=False)

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        # ``search`` doesn't apply — runs are activity, not searchable entries.
        return self._records(newest_first=True, limit=k, offset=offset)

    def _rows_since(self, cursor: datetime, cap: int | None) -> list[MemoryEntry]:
        return self._records(newest_first=False, cursor=cursor, limit=cap)

    def _embedded_rows(self) -> list[MemoryEntry]:
        return []

    def _rows_in_window(self, start: datetime, end: datetime) -> list[MemoryEntry]:
        return self._records(newest_first=False, window=(start, end))

    def _records(
        self,
        *,
        newest_first: bool,
        cursor: datetime | None = None,
        window: tuple[datetime, datetime] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """Completion rows → rendered run records as ``MemoryEntry`` (content =
        the record, created_at = completion time)."""
        with self._session() as session:
            rows = self._completion_rows(session, newest_first, cursor, window, limit, offset)
            if not rows:
                return []
            grouped = self._group_prompts(session, [run_id for run_id, _, _ in rows])
        return [self._to_record(run_id, ts, last_id, grouped) for run_id, ts, last_id in rows]

    def _completion_rows(
        self,
        session: Session,
        newest_first: bool,
        cursor: datetime | None,
        window: tuple[datetime, datetime] | None,
        limit: int | None,
        offset: int,
    ) -> list:
        """The ``(run_id, completion_time, last_prompt_id)`` run-index rows for
        this scope — one per completed run, served by the partial index."""
        query = select(PromptLog.run_id, PromptLog.timestamp, PromptLog.id).where(
            *self._completion_clauses()
        )
        if cursor is not None:
            query = query.where(PromptLog.timestamp > cursor)
        if window is not None:
            query = query.where(PromptLog.timestamp >= window[0], PromptLog.timestamp <= window[1])
        order = PromptLog.timestamp.desc() if newest_first else PromptLog.timestamp.asc()
        query = query.order_by(order)  # type: ignore[union-attr]
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return [row for row in session.exec(query).all() if row[0] is not None]

    def _to_record(
        self, run_id: str, timestamp: datetime, last_id: int, grouped: dict[str, list[PromptLog]]
    ) -> MemoryEntry:
        """One run-index row rendered as a ``MemoryEntry`` (content = the record)."""
        return MemoryEntry(
            id=last_id,
            memory_name=self.name,
            key=None,
            content=render_run_record(grouped.get(run_id) or []),
            author=PennyConstants.MessageAuthor.COLLECTOR,
            created_at=timestamp,
        )

    @staticmethod
    def _group_prompts(session: Session, run_ids: list[str]) -> dict[str, list[PromptLog]]:
        rows = session.exec(
            select(PromptLog)
            .where(PromptLog.run_id.in_(run_ids))  # ty: ignore[unresolved-attribute]
            .order_by(PromptLog.timestamp.asc())
        ).all()
        grouped: dict[str, list[PromptLog]] = {}
        for prompt in rows:
            if prompt.run_id is not None:
                grouped.setdefault(prompt.run_id, []).append(prompt)
        return grouped


__all__ = [
    "Collection",
    "Log",
    "Memory",
    "render_tool_call",
    "render_run_record",
    "classify_run",
    "RunHealth",
    "MessageLogMemory",
    "RunLog",
]
