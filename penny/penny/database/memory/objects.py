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
import re
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field, computed_field
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import (
    WRITE_GATE_MUTATING_OUTCOMES,
    PennyConstants,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database.memory import _similarity as sim
from penny.database.memory.types import (
    DedupThresholds,
    EntryInput,
    EntrySide,
    LogEntryInput,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    MoveOutcome,
    ReadOnlyMemoryError,
    UpdateOutcome,
    WriteResult,
    WrongShapeError,
    slug,
    wrong_shape_message,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog
from penny.database.mutation_store import EnumeratedDecision
from penny.text_validity import degenerate_reason, half_formed_send_reason
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
        raise WrongShapeError(self.name, self.type, wrong_shape_message(self.name, self.type))

    def _refuse_log_op(self) -> None:
        raise WrongShapeError(self.name, self.type, wrong_shape_message(self.name, self.type))

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
        self,
        entries: list[EntryInput],
        author: str,
        thresholds: DedupThresholds | None = None,
        run_id: str | None = None,
    ) -> list[WriteResult]:
        self._refuse_collection_op()
        return []

    def update(
        self, key: str, content: str, author: str, run_id: str | None = None
    ) -> UpdateOutcome:
        self._refuse_collection_op()
        return "not_found"

    def move(self, key: str, to_name: str, author: str) -> MoveOutcome:
        self._refuse_collection_op()
        return "not_found"

    def delete(self, key: str) -> int:
        self._refuse_collection_op()
        return 0

    def append(
        self, entries: list[LogEntryInput], author: str, run_id: str | None = None
    ) -> list[MemoryEntry]:
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
        the log cursor's first read and send_message's cooldown probe.  The
        model-facing ``collection_read_latest`` is the ``Collection.read_latest``
        wrapper over this; logs reach it only through ``read_batch``."""
        return self._newest_rows(k, offset, search)

    def read_all(self) -> list[MemoryEntry]:
        return self._all_rows()

    def entry_by_id(self, entry_id: int) -> MemoryEntry | None:
        """The stored entry with this id iff it belongs to this memory.

        The by-handle read behind a browse micro-context's fetch handle
        (``<memory>#<id>``): the full page content stays in browse-results while
        only the typed extracted value + handle return to the main loop, so the
        handle must resolve back to the whole entry.  Scoped to this memory's own
        ``memory_entry`` rows — a facade over another table has none, so it
        honestly returns ``None`` rather than a foreign row."""
        with self._session() as session:
            row = session.get(MemoryEntry, entry_id)
        return row if row is not None and row.memory_name == self.name else None

    def read_since(self, cursor: datetime, cap: int | None = None) -> list[MemoryEntry]:
        return self._rows_since(cursor, cap)

    def read_recent(self, window_seconds: int, cap: int | None = None) -> list[MemoryEntry]:
        cutoff = datetime.fromtimestamp(datetime.now(UTC).timestamp() - window_seconds, tz=UTC)
        return self._rows_since(cutoff, cap)

    def read_similar(
        self, anchor: list[float], k: int | None = None, floor: float = 0.0
    ) -> list[MemoryEntry]:
        """Entries ranked by plain cosine similarity to ``anchor``, best-first.

        Backs the explicit ``read_similar`` search tool, so it returns the
        nearest neighbours and lets the model judge them — no centrality-magnet
        penalty or cluster-strength gate (those are ambient-recall policies that
        decide whether to inject *unprompted*; applying them to an explicit
        search collapsed a populated but homogeneous collection like ``skills``
        to "No entries" and removed the model's fuzzy-recovery path, #1565).
        Entries without a content embedding are skipped.  ``floor`` (default 0.0,
        i.e. drop only anti-correlated entries) filters by cosine; ``k=None``
        returns every survivor.  An empty result reflects the corpus and floor —
        not an ambient "nothing relevant enough" suppression."""
        scored = [
            (row, row.content_embedding)
            for row in self._embedded_rows()
            if row.id is not None and row.content_embedding is not None
        ]
        if not scored:
            return []
        valid = [row for row, _ in scored]
        scores = sim.cosine_scores([blob for _, blob in scored], anchor)
        order = list(np.argsort(-scores))
        ranked = [valid[i] for i in order if float(scores[i]) >= floor]
        return ranked if k is None else ranked[:k]

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


def _content_unchanged(stored: str, incoming: str) -> bool:
    """The change-gate's deterministic value comparison (#1587): same value → never
    news.  Whitespace-trimmed exact equality — the watched field is one value per
    key, so an identical re-observation is byte-identical bar surrounding
    whitespace, and a differing one is genuine news."""
    return stored.strip() == incoming.strip()


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
        self,
        entries: list[EntryInput],
        author: str,
        thresholds: DedupThresholds | None = None,
        run_id: str | None = None,
    ) -> list[WriteResult]:
        """Write entries with per-entry similarity dedup.  One ``WriteResult``
        per input; dedup runs against the existing corpus using the configured
        (or default) thresholds.

        ``run_id`` (the writing run, threaded as a parameter — no ambient state)
        stamps ``created_by_run_id`` and ``last_written_by_run_id`` on each new
        row, so an entry cites the run that produced it (#1560).  A ``KEY_EXISTS_CHANGED``
        entry auto-refreshes its stored baseline in place through the same update
        path (#1633) — so it too advances ``last_written_by_run_id`` and counts as a
        change for the notify below."""
        thresholds = thresholds or DedupThresholds.from_runtime(self._runtime)
        existing = self._entries_with_vectors()
        results: list[WriteResult] = []
        with self._session() as session:
            for entry in entries:
                results.append(
                    self._write_one(session, entry, author, existing, thresholds, run_id)
                )
            session.commit()
        if any(result.outcome in WRITE_GATE_MUTATING_OUTCOMES for result in results):
            self._notify()
        return results

    def update(
        self, key: str, content: str, author: str, run_id: str | None = None
    ) -> UpdateOutcome:
        """Replace the content of every entry with ``key`` (dedup keeps it to one).

        A rewrite, so ``last_written_by_run_id`` advances to ``run_id`` while
        ``created_by_run_id`` is left untouched — the read-path anchors that keep
        "who wrote the current value?" distinct from "who created it?" (#1560)."""
        with self._session() as session:
            rows = self._rows_by_key(session, self.name, key)
            if not rows:
                return "not_found"
            self._apply_content(session, rows, content, author, run_id)
            session.commit()
        self._notify()
        return "ok"

    @staticmethod
    def _apply_content(
        session: Session,
        rows: list[MemoryEntry],
        content: str,
        author: str,
        run_id: str | None,
    ) -> None:
        """Rewrite each row's content in place, stamping the writing run — the one
        mutation shared by ``update`` (the model's correction path) and the
        change-gate's ``KEY_EXISTS_CHANGED`` auto-refresh (#1633), so a baseline
        advances identically whichever path reached it.  ``created_by_run_id`` is
        left untouched (this is a rewrite, not a creation)."""
        for row in rows:
            row.content = content
            row.author = author
            row.last_written_by_run_id = run_id
            session.add(row)

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
            raise MemoryNotFoundError(to_name)
        if row.type != MemoryType.COLLECTION:
            raise MemoryTypeError(wrong_shape_message(to_name, row.type))

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
        run_id: str | None = None,
    ) -> WriteResult:
        """Classify one write into the closed ``WriteGateOutcome`` union — the
        change-gate at the write chokepoint (#1587).  The comparison is
        deterministic and total; nothing here is a model judgment.  The gate decides
        the non-write outcomes; only a genuinely new key is persisted."""
        gated = self._gate_outcome(session, entry, author, existing, thresholds, run_id)
        if gated is not None:
            return gated
        return self._insert_new_entry(session, entry, author, existing, run_id)

    def _gate_outcome(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        existing: list[EntrySide],
        thresholds: DedupThresholds,
        run_id: str | None,
    ) -> WriteResult | None:
        """The change-gate itself: the non-NEW_KEY outcome for an entry that is NOT
        newly inserted — DEGENERATE, KEY_EXISTS_CHANGED/UNCHANGED (an exact-key hit
        compared by value: same value → never news, different value → news, decided
        deterministically, not by an embedding threshold), or DUPLICATE — or ``None``
        when it's a genuinely new key to persist.

        SINGLE-FIELD DISCIPLINE: the compared content must be one value per key;
        cramming a volatile sibling (a countdown, a timestamp) into the same body
        would trip a false CHANGED every cycle — the volatile-sibling false-alert
        this gate exists to kill (#1562)."""
        rejection_reason = degenerate_reason(entry.content)
        if rejection_reason is not None:
            logger.debug("Rejected degenerate collection entry %r: %s", entry.key, rejection_reason)
            return WriteResult(
                key=entry.key, outcome=WriteGateOutcome.DEGENERATE, reason=rejection_reason
            )
        stored = self._rows_by_key(session, self.name, entry.key)
        if stored:
            return self._exact_key_outcome(session, stored, entry, author, run_id)
        candidate = EntrySide(entry.key, entry.key_embedding, entry.content_embedding)
        matched = sim.is_duplicate(candidate, existing, thresholds)
        if matched is not None:
            return WriteResult(
                key=entry.key, outcome=WriteGateOutcome.DUPLICATE, matched_key=matched.key
            )
        return None

    def _exact_key_outcome(
        self,
        session: Session,
        stored: list[MemoryEntry],
        entry: EntryInput,
        author: str,
        run_id: str | None,
    ) -> WriteResult:
        """The exact-key hit — compared by value, not embedding.  Identical value →
        KEY_EXISTS_UNCHANGED (the watch's "no change" signal, STOP-worthy).  Different
        value → KEY_EXISTS_CHANGED: the gate **auto-refreshes the stored baseline
        itself** in place through the shared update path (#1633), so no dangling
        ``update_entry`` is left for the model to run.  A degenerate new value never
        reaches here — it classified DEGENERATE upstream — so the refresh only ever
        stores screened content."""
        if _content_unchanged(stored[0].content, entry.content):
            return WriteResult(
                key=entry.key, outcome=WriteGateOutcome.KEY_EXISTS_UNCHANGED, matched_key=entry.key
            )
        self._apply_content(session, stored, entry.content, author, run_id)
        return WriteResult(
            key=entry.key, outcome=WriteGateOutcome.KEY_EXISTS_CHANGED, matched_key=entry.key
        )

    def _insert_new_entry(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        existing: list[EntrySide],
        run_id: str | None,
    ) -> WriteResult:
        """Persist a genuinely new key (NEW_KEY) and record it for in-batch dedup so
        a later entry in the same batch dedups against it."""
        row = MemoryEntry(
            memory_name=self.name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=sim.maybe_serialize(entry.key_embedding),
            content_embedding=sim.maybe_serialize(entry.content_embedding),
            created_at=datetime.now(UTC),
            created_by_run_id=run_id,
            last_written_by_run_id=run_id,
        )
        session.add(row)
        session.flush()
        existing.append(EntrySide(entry.key, entry.key_embedding, entry.content_embedding))
        return WriteResult(key=entry.key, outcome=WriteGateOutcome.NEW_KEY, entry_id=row.id)


class Log(Memory):
    """An append-only stream — keyless entries in time order.

    Backed by ``memory_entry`` (the base's row primitives).  Implements the
    cursored/window read surface and ``append``; keyed ops stay refused.  The
    cursor *read* logic lives here, uniform across every log backing — the
    reader's pending/commit lifecycle stays in ``LogReadTool``.
    """

    def append(
        self, entries: list[LogEntryInput], author: str, run_id: str | None = None
    ) -> list[MemoryEntry]:
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
                    # Log entries are immutable (append-only), so the creating run
                    # is also the last writer (#1560).
                    created_by_run_id=run_id,
                    last_written_by_run_id=run_id,
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


class MessageLogMemory(Log):
    """Read facade over ``messagelog`` for ``user-messages`` / ``penny-messages``.

    A message has two conversational authors — the user (incoming) or Penny
    (outgoing), which IS the direction; the internal agent that produced a Penny
    message isn't a conversational author, so it isn't recorded here.  Reactions
    are excluded (not conversation).  ``content_embedding`` is the messagelog
    embedding (same serialized-bytes format), written at ingress/egress and
    backfilled at startup.  Read-only — the channel owns the canonical writes.

    **Emission provenance renders inline (#1568)**: an outgoing row stamped with
    the ``mechanism`` that sent it (an autonomous collector send, carried onto the
    row by the drainer) renders ``(sent by <mechanism>) <content>`` — so pulling a
    message by relevance or recency IS resolving its source, on every read path
    (they all synthesize entries through ``_to_entry``).  A NULL-mechanism row (a
    direct reply; every incoming row) renders the bare content, byte-identical to
    the pre-provenance render — the unmarked case stays the quiet default.
    """

    # The inline provenance marker, rendered between the timestamp stamp (added by
    # the tool layer's ``_format_entries``) and the message content:
    # ``3. [2026-07-02 09:14 UTC] (sent by price-watch) Heads up: …``.
    SENT_BY_MARKER = "(sent by {mechanism}) "

    def __init__(self, row: MemoryRow, engine, *, direction: str, on_changed=None) -> None:
        super().__init__(row, engine, on_changed=on_changed)
        self._direction = direction
        self._author = (
            PennyConstants.MessageAuthor.USER
            if direction == PennyConstants.MessageDirection.INCOMING
            else PennyConstants.MessageAuthor.PENNY
        )

    def append(
        self, entries: list[LogEntryInput], author: str, run_id: str | None = None
    ) -> list[MemoryEntry]:
        raise ReadOnlyMemoryError(
            f"'{self.name}' is a read view over the conversation log — messages "
            "are recorded by the channel, not appended here."
        )

    def _to_entry(self, row: MessageLog) -> MemoryEntry:
        return MemoryEntry(
            id=row.id,
            memory_name=self.name,
            key=None,
            content=self._render_content(row),
            author=self._author,
            key_embedding=None,
            content_embedding=row.embedding,
            created_at=row.timestamp,
        )

    def _render_content(self, row: MessageLog) -> str:
        """The entry content with inline emission provenance (#1568): a
        mechanism-stamped send leads with the ``(sent by <mechanism>)`` marker; a
        NULL-mechanism row is the bare content, byte-identical to the
        pre-provenance render."""
        if row.mechanism is None:
            return row.content
        return f"{self.SENT_BY_MARKER.format(mechanism=row.mechanism)}{row.content}"

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


def render_tool_call(name: str, args: object) -> str:
    """Compact, grokkable render of one tool call (the salient args only).

    The single shared format for the ``collector-runs`` run trace, so the model
    reads what a cycle did the same way everywhere.  Every call renders under its
    **real** tool name in the canonical ``tool(args)`` shape — the same dialect a
    prompt writes a call in (``docs/prompt-writing-guide.md`` → "The canonical call
    notation", point 6), so what the model reads of its past matches how it's told
    to act, and a wrong-shape or hallucinated sibling name is identifiable against
    the real one.  kwargs form (``memory=…, key=…``) when a call carries more than
    one argument; a single obvious argument stays positional.  Kept compact (the
    salient args only, content never truncated) — the aliases these replaced saved
    a few tokens per line; the real names cost them back, and nothing else is added.
    """
    fields = cast("dict[str, Any]", args if isinstance(args, dict) else {})
    if name == "collection_write":
        return (
            f"collection_write(memory={fields.get('memory', '?')!r}, "
            f"entries={_write_contents(fields)!r})"
        )
    if name == "update_entry":
        return f"update_entry(memory={fields.get('memory', '?')!r}, key={fields.get('key', '?')!r})"
    if name == "send_message":
        return f"send_message({fields.get('content', '')!r})"
    if name == "browse":
        queries = fields.get("queries", list(fields.values()))
        extract = fields.get("extract")
        if extract:
            return f"browse(queries={queries!r}, extract={extract!r})"
        return f"browse({queries!r})"
    # A dict (incl. the empty dict of an argless call, e.g. ``done()``, #1569)
    # renders its fields; a non-dict / malformed args string shows its repr so the
    # trace still names what the model tried.
    if isinstance(args, dict):
        rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
        return f"{name}({rendered})"
    return f"{name}({args!r})"


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


def _decode_arguments(raw: object) -> dict[str, Any]:
    """A stored/wire tool call's ``arguments`` (a JSON string) as a dict.

    Non-string or unparseable arguments yield ``{}`` — the canonical shape is
    always a dict of fields, and the verbatim string is preserved in the log
    itself (this is a read projection, never the storage)."""
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


class LoggedToolCall(BaseModel):
    """The canonical, round-trippable shape of one logged tool call (#1560).

    A logged call and the outgoing wire call are the SAME structure: the model's
    response is stored verbatim in ``promptlog.response`` (``raw.model_dump()``),
    where a tool call is ``{"function": {"name": …, "arguments": "<json string>"}}``
    — the exact envelope the loop re-emits via ``LlmMessage.to_input_message`` and
    the executor parses via ``LlmClient._parse_tool_call``.  So logs contain calls;
    they never paraphrase them, and ``replay(logged_call) == original_call`` holds
    structurally.  This model formalizes that invariant: ``from_function`` reads a
    stored (or wire) ``function`` dict, ``to_wire`` re-emits the identical envelope,
    and ``from_function(x.to_wire()) == x`` is the identity.

    It is the one datatype that later doubles as a script / skill step (promotion is
    a copy, not a parse — #1471) and carries the options-presented ``decision``
    accommodation for the enumerated-decision unions of #1562/#1563 (a field on the
    shape now; populated at call sites there, not forced here)."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision: EnumeratedDecision | None = None

    @classmethod
    def from_function(cls, function: dict) -> LoggedToolCall:
        """Parse a stored/wire ``function`` dict into the canonical shape."""
        return cls(
            name=function.get("name") or "",
            arguments=_decode_arguments(function.get("arguments")),
        )

    def to_wire(self) -> dict[str, str]:
        """Re-emit the OpenAI wire ``function`` envelope — arguments re-serialized
        exactly as ``LlmMessage.to_input_message`` does, so a round-trip is the
        identity and the loop would re-send this call unchanged."""
        return {"name": self.name, "arguments": json.dumps(self.arguments)}

    def render(self) -> str:
        """The canonical one-line ``name(args)`` projection — call syntax, not
        narration.  ``read_run_calls`` renders a run as this canonical projection
        (the prose is composed by the model from the in-context trace); the
        first-person ``to_result_narration`` is a separate projection."""
        return render_tool_call(self.name, self.arguments)


def _run_logged_steps(prompts: list[PromptLog]) -> list[tuple[str | None, LoggedToolCall]]:
    """Every tool call across a run's prompts as ``(call_id, LoggedToolCall)``, in
    order — the canonical projection paired with the id that keys its result."""
    steps: list[tuple[str | None, LoggedToolCall]] = []
    for prompt in prompts:
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                steps.append((tool_call.get("id"), LoggedToolCall.from_function(function)))
    return steps


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


def _ended_via_write_gate_stop(prompts: list[PromptLog]) -> bool:
    """True when a write-gate STOP ended the run (#1587): the run reached a clean
    outcome (``no_work`` / ``worked``) but recorded NO ``done()`` call.

    Structural signal, no new column: in the normal path a ``no_work`` / ``worked``
    close always follows a successful ``done()`` (that's how the outcome is
    determined), so a clean outcome with no ``done()`` in the trace can only be a
    tool STOP — the collector exited at the chokepoint before ``done()``.  Historical
    rows never satisfy it (they all closed via ``done()``), so it never misfires."""
    outcome, reason, _ = _run_outcome(prompts)
    if not reason or outcome not in (RunOutcome.NO_WORK.value, RunOutcome.WORKED.value):
        return False
    calls = _run_tool_calls(prompts)
    return bool(calls) and not any(name == "done" for name, _ in calls)


def _run_tool_failures(prompts: list[PromptLog]) -> int:
    """The persisted failed-tool-call count, off the run's outcome-bearing row.

    Stamped by ``set_run_outcome`` on the last prompt; NULL on old/untagged rows
    reads as zero (not measured)."""
    for prompt in reversed(prompts):
        if prompt.run_outcome is not None:
            return prompt.tool_failures or 0
    return 0


_WRITE_TOOLS = frozenset(
    {"collection_write", "update_entry", "collection_delete_entry", "log_append"}
)


def _run_io_tally(prompts: list[PromptLog]) -> tuple[int, int, int, int, int]:
    """Structural I/O counts for a run: ``(browses_ok, browses_failed, reads,
    writes, sends)``.

    Pure description of what the run *did* — no judgment of why or what to do about
    it.  Two kinds of read are kept apart because they behave differently:

    - **browses** — external web reads, counted at sub-result granularity from the
      ``## browse`` / ``## browse search`` / ``## browse error`` section headers in
      the tool-result messages.  A *partial* browse failure returns ``success=True``
      (the model works from whatever succeeded), so those failures are visible *only*
      in the result text, never in ``tool_failures`` — which is why they're counted
      from the rendered headers here.  Only an *all-queries-failed* browse returns
      ``success=False`` (and so does add to ``tool_failures``); the header count still
      captures it, so ``no_writes`` is unaffected either way.
    - **reads** — internal collection/log reads (``log_read``,
      ``collection_read_*``, ``read_similar``, ``collection_catalog``, …): every
      tool call that isn't a browse, write, send,
      or ``done()``.  These don't meaningfully fail, so they're a plain call count.

    Writes and sends count the model's own tool *calls* (each appears once across
    the per-step responses), so ``writes == 0`` is the hard fact that nothing was
    ever written — whatever a ``done()`` summary claims.

    Browse results are read off the final prompt's ``messages`` (the full
    accumulated conversation, so every result bar the last step's — which is almost
    always ``done()`` — appears there exactly once); the call counts come from the
    responses, so they're complete regardless."""
    calls = _run_tool_calls(prompts)
    writes = sum(1 for name, _ in calls if name in _WRITE_TOOLS)
    sends = sum(1 for name, _ in calls if name == "send_message")
    reads = sum(
        1
        for name, _ in calls
        if name not in _WRITE_TOOLS and name not in {"browse", "send_message", "done"}
    )
    browses_ok = browses_failed = 0
    messages = json.loads(prompts[-1].messages) if prompts and prompts[-1].messages else []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        browses_ok += content.count(PennyConstants.BROWSE_PAGE_HEADER)
        browses_ok += content.count(PennyConstants.BROWSE_SEARCH_HEADER)
        browses_failed += content.count(PennyConstants.BROWSE_ERROR_HEADER)
    return browses_ok, browses_failed, reads, writes, sends


def _io_tally_line(prompts: list[PromptLog]) -> str | None:
    """The descriptive counts line for a run record, or ``None`` for a read-only
    quiet cycle (nothing notable to tally).

    Shown whenever the run browsed, wrote, or sent — the cases where the tally is
    informative.  ``writes`` is always part of the line (so ``writes: 0`` is a
    plain structural fact — the record is generated from the ledger, #1569);
    browses, reads, and sends appear only when nonzero, so the line stays as short
    as the run was."""
    browses_ok, browses_failed, reads, writes, sends = _run_io_tally(prompts)
    if not (browses_ok or browses_failed or writes or sends):
        return None
    segments: list[str] = []
    if browses_ok or browses_failed:
        segments.append(f"browses: {browses_ok} ok, {browses_failed} failed")
    if reads:
        segments.append(f"reads: {reads}")
    segments.append(f"writes: {writes}")
    if sends:
        segments.append(f"sends: {sends}")
    return " · ".join(segments)


def _is_degenerate_send(content: str) -> bool:
    """True if a sent message has no real content the user should have received.

    The flag side of the shared :func:`half_formed_send_reason` rule — the same
    test the ``send_message`` tool gates on before delivery."""
    return half_formed_send_reason(content) is not None


class RunHealth(BaseModel):
    """Structural failure signals for one collector run, derived from its
    ``promptlog`` rows.  One classifier feeds both Penny's ambient view of her own
    runs (the run record the self-state header and the ``collector-runs`` read
    facade render) and the addon's prompts tab (badges + the "flagged only"
    filter) — what we use to judge whether a run regressed is exactly what Penny
    sees of it.

    All signals are deterministic and read from stored data — no model judgment:
    ``bailed`` (did no real work — recorded a terminating ``done()`` without any
    read/write/browse first), ``no_writes`` (a browse read failed *and* the run
    made no write — the two structural facts only; what they mean is the model's to
    reason about), ``incomplete`` (hit the step ceiling without a closing ``done()``
    — including a run that recorded no tool call at all, having spun on rejected
    premature-``done()``s until the ceiling), ``tool_failures`` (count of failed
    tool calls in the run — browse adds to this only when *every* query failed, not
    on a partial failure, see ``_run_io_tally``), ``degenerate_send`` (a message
    went out with no real content)."""

    bailed: bool = False
    no_writes: bool = False
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
        if self.no_writes:
            out.append(ConditionKey.NO_WRITES.value)
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
    # A bail is a *deliberate* early close: the model recorded a terminating
    # done() with no real read/write/browse first.  A run that recorded NO tool
    # call at all never decided to stop — it spun on rejected premature-done()s
    # (or empty/text steps) until the step ceiling, which is capacity, not drift.
    # Surface that as INCOMPLETE (which quality ignores), not NO WORK DONE (which
    # churned a healthy collector whose other cycles worked fine).
    bailed = (
        bool(calls)
        and not non_done
        and outcome
        in (
            RunOutcome.NO_WORK.value,
            RunOutcome.FAILED.value,
        )
    )
    exhausted_no_call = not calls and outcome == RunOutcome.FAILED.value
    degenerate = any(
        name == "send_message" and _is_degenerate_send(_send_content(args)) for name, args in calls
    )
    _browses_ok, browses_failed, _reads, writes, _sends = _run_io_tally(prompts)
    return RunHealth(
        bailed=bailed,
        no_writes=browses_failed > 0 and writes == 0,
        incomplete=outcome == RunOutcome.INCOMPLETE.value or exhausted_no_call,
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
        ConditionKey.NO_WRITES: health.no_writes,
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

    The verbose form the run record shows (in the self-state header and the
    ``collector-runs`` read facade); the addon derives its compact badges from the
    same ``RunHealth.flags``.  Marker + detail come from the shared catalog so the
    run record and the addon's TS type name one text."""
    return [
        _flag_line(health, entry.key, entry.marker, entry.detail)
        for entry in run_flag_conditions()
        if entry.marker is not None and entry.detail is not None and _flag_is_set(health, entry.key)
    ]


def render_run_record(prompts: list[PromptLog]) -> str:
    """One run: a ``[target] <outcome>`` line, any ⚠ health-flag lines, then the
    run's tool calls, one per line.

    GENERATED from the run's canonical ledger rows (#1569) — the header shows the
    run's structural outcome (its ``RunOutcome`` enum, or the write-gate stop
    reason), never a model-authored ``done()`` summary (``done()`` is argless).
    Carries NO timestamp — each consumer supplies its own (the model via the
    ``log_read`` entry stamp, the addon via the run's ``created_at`` field), so
    embedding one here just duplicated it.  Kept deliberately flat: a format
    bake-off (``format_bakeoff.py``) found markdown headers / numbered tool lists
    gave gpt-oss no reading-comprehension gain over plain text on these records
    (numbering slightly hurt), so the simplest rendering wins.  ``classify_run``
    determines health; ``_health_lines`` renders the ⚠ flags.  A descriptive
    ``_io_tally_line`` (reads ok/failed · writes · sends) sits under the outcome
    when the run did any browse/write/send — the bare structural facts of what the
    run actually did.

    The trace shows:

    - **bailed** — the single meagre call (or ``(no tool calls)``): the run jumped
      to ``done()`` / acted not at all, so there's nothing but the bail to see;
    - **worked / incomplete / tool-failure / half-formed-send** — the full ordered
      non-``done()`` trace, so the work (or the failing/degenerate call) can be
      judged in context;
    - **everything else** (a quiet cycle that DID read, a failed/cancelled run that
      DID call real tools with no new flag) — outcome-line only, no trace to tempt
      an over-correction.

    Content is never truncated."""
    if not prompts:
        return "(no data)"
    health = classify_run(prompts)
    outcome, reason, target = _run_outcome(prompts)
    lines = [f"[{target or '?'}] {reason or outcome or ''}".rstrip()]
    tally = _io_tally_line(prompts)
    if tally:
        lines.append(tally)
    lines.extend(_health_lines(health))
    calls = _run_tool_calls(prompts)
    non_done = [(name, args) for name, args in calls if name != "done"]
    if health.bailed:
        lines.append("(no tool calls)" if not calls else render_tool_call(*calls[0]))
    elif (
        outcome == RunOutcome.WORKED.value
        or health.regressive
        or _ended_via_write_gate_stop(prompts)
    ):
        # A STOP-ended run (#1587) shows its trace too — the stop point (the write
        # call) alongside the stop reason on the header, so "why did it stop?" is a
        # read.  Its reason is already the header line; no ⚠ flag (a clean stop).
        lines.extend(render_tool_call(name, args) for name, args in non_done)
    return "\n".join(lines)


# ── The tool-call-sequence lens ───────────────────────────────────────────────
# A second projection of the same run primitive (grouped promptlog rows), distinct
# from the collector-audit ``render_run_record`` above.  This one drops the health
# signals and shows a run purely as *what it did*: origin → the ordered tool calls
# → conclusion.  It's orthogonal to the target — a chat turn (origin = the user's
# message) or a collector cycle (origin = its bound target) both render through it.

_ROLE_USER = "user"


def _opening_user_message(prompts: list[PromptLog]) -> str:
    """The user's message that opened a run, or ``""`` when there is none.

    The message is the last ``user`` turn of the run's first prompt with the
    injected Live-context block (fused into that turn) stripped back off, so a
    reader sees what the user actually said — not the recall/time scaffolding.
    A collector/background run opens with an empty prompt (pure injected context,
    no separator), so this is ``""`` — the structural signal it had no user intent.
    """
    if not prompts or not prompts[0].messages:
        return ""
    messages = json.loads(prompts[0].messages)
    user_turns = [m.get("content") or "" for m in messages if m.get("role") == _ROLE_USER]
    if not user_turns:
        return ""
    raw = user_turns[-1]
    if PennyConstants.SECTION_SEPARATOR not in raw:
        return ""
    return raw.split(PennyConstants.SECTION_SEPARATOR, 1)[1].strip()


def _final_assistant_text(prompts: list[PromptLog]) -> str:
    """The run's closing plain-text assistant reply (no tool calls) — Penny's reply
    in a chat turn; absent in a collector run (which ends with ``done()``)."""
    for prompt in reversed(prompts):
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            if content and not message.get("tool_calls"):
                return content
    return ""


def _run_origin(prompts: list[PromptLog]) -> str:
    """What prompted the run: ``user: <message>`` (turn-driven) or ``[target]``
    (collector) — resolved by structure, a run has one or the other."""
    message = _opening_user_message(prompts)
    if message:
        return f"user: {message}"
    _, _, target = _run_outcome(prompts)
    return f"[{target}]" if target else ""


def _run_conclusion(prompts: list[PromptLog]) -> str:
    """How the run ended, STRUCTURALLY (#1569): ``penny: <reply>`` (chat — the
    user-facing reply stays model-authored), ``done: <outcome>`` (a collector that
    closed via the argless ``done()`` — its structural outcome enum, never a
    model-authored summary), ``stopped: <reason>`` (a write-gate STOP at the
    chokepoint, #1587), or ``ended: <reason>`` (a collector that never closed
    cleanly — its structural no-``done()`` reason).  Chat vs. collector is read off
    ``run_target`` (collectors stamp it, chat doesn't)."""
    outcome, reason, target = _run_outcome(prompts)
    if _ended_via_write_gate_stop(prompts):
        return f"stopped: {reason}"
    if target is not None:  # a collector run
        if any(name == "done" for name, _ in _run_tool_calls(prompts)):
            return f"done: {outcome}" if outcome else ""
        detail = reason or outcome
        return f"ended: {detail}" if detail else ""
    reply = _final_assistant_text(prompts)
    return f"penny: {reply}" if reply else ""


def _run_header(prompts: list[PromptLog]) -> str:
    """``run <run_id>`` — the run's own addressable identifier, so every rendered
    run is an anchor surface (a reader can name the run it's looking at, not guess
    it), per the compositional n≤1 invariant (#1560)."""
    run_id = prompts[0].run_id if prompts else None
    return f"run {run_id}" if run_id else ""


# ``generate_image``'s result names the stored media row's id via a shared prefix
# constant (so format + parse can't drift) — the egress/media trace reads it back so
# a reader can name (and re-fetch) exactly what was attached to the reply.
_GENERATED_MEDIA_RE = re.compile(re.escape(PennyConstants.GENERATED_IMAGE_RESULT_PREFIX) + r"(\d+)")

# A step's result renders as ONE compact line (bulk results — page content — stay
# whole in the ledger and are rendered by reference here, per the canonical-call
# policy): first line only, capped so a big result can't blow up the trace.
_RESULT_PREVIEW_CHARS = 120


def _generated_media_ids(prompts: list[PromptLog]) -> list[int]:
    """Ids of the media a run generated and delivered with its reply (#1560).

    A drawn image is delivered deterministically to its own reply (#1564); its
    ``generate_image`` tool result records the stored media id, which lives in the
    run's accumulated tool-result messages.  Read it back so the egress trace names
    an addressable id, closing the delivery-introspection gap that let the model
    confabulate a delivery it couldn't inspect."""
    if not prompts or not prompts[-1].messages:
        return []
    messages = json.loads(prompts[-1].messages)
    found: list[int] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        found.extend(
            int(m.group(1)) for m in _GENERATED_MEDIA_RE.finditer(message.get("content") or "")
        )
    return found


def _egress_media_lines(prompts: list[PromptLog]) -> list[str]:
    """The ``attached: image #<id>`` line(s) — what actually went out to the user
    alongside the reply, by addressable id.  Empty when the run attached nothing."""
    return [f"    attached: image #{media_id}" for media_id in _generated_media_ids(prompts)]


def _tool_results_by_id(prompts: list[PromptLog]) -> dict[str, str]:
    """``{tool_call_id: result content}`` from the run's accumulated tool turns.

    Read off the last prompt's messages (the full conversation, where every step's
    result sits exactly once), keyed by ``tool_call_id`` so each call renders with
    its own outcome — the ``their outcomes`` half of chat-run introspection."""
    if not prompts or not prompts[-1].messages:
        return {}
    messages = json.loads(prompts[-1].messages)
    return {
        message["tool_call_id"]: message.get("content") or ""
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }


def _tool_successes_by_id(prompts: list[PromptLog]) -> dict[str, bool]:
    """``{tool_call_id: success}`` — the STRUCTURAL per-call execution stamp (#1600).

    Read off the run's accumulated tool turns (the last prompt's ``messages``, where
    every step's result sits exactly once), keyed by ``tool_call_id`` so each call
    pairs with its own outcome.  The framework wrote each stamp at execution time
    from the tool's ``ToolResult.success`` (``PennyConstants.TOOL_RESULT_SUCCESS_KEY``),
    beside the framed result prose — so "did this call succeed?" is a boolean read,
    not a narration parse.  A tool turn with no stamp (a run logged before #1600) is
    simply absent from the map — the caller treats an absent stamp as uncertified."""
    if not prompts or not prompts[-1].messages:
        return {}
    messages = json.loads(prompts[-1].messages)
    successes: dict[str, bool] = {}
    for message in messages:
        call_id = message.get("tool_call_id")
        if (
            message.get("role") == "tool"
            and call_id
            and PennyConstants.TOOL_RESULT_SUCCESS_KEY in message
        ):
            successes[call_id] = bool(message[PennyConstants.TOOL_RESULT_SUCCESS_KEY])
    return successes


def _compact_result(content: str | None) -> str:
    """One compact line for a step's result — first non-blank line, capped.  Bulk
    results (page content) stay whole in the ledger; this render is by reference."""
    if not content:
        return ""
    line = next((part.strip() for part in content.splitlines() if part.strip()), "")
    return (
        line if len(line) <= _RESULT_PREVIEW_CHARS else f"{line[:_RESULT_PREVIEW_CHARS].rstrip()}…"
    )


def _step_line(index: int, call: LoggedToolCall, result: str | None) -> str:
    """One step of the canonical projection: ``step <N>: <call> => <result>``.

    ``index`` is the step's coordinate in the run's FULL tool-call sequence — an
    absolute ordinal, so a filtered view shows gaps rather than renumbering (a
    persisted coordinate future skill selection can address as ``(run_id, range)``,
    #1471).  The result renders compact/by-reference."""
    rendered = f"    step {index}: {call.render()}"
    preview = _compact_result(result)
    return f"{rendered} => {preview}" if preview else rendered


def render_run_calls(prompts: list[PromptLog]) -> str:
    """One run as its tool-call SEQUENCE: ``run id → origin → the numbered steps →
    conclusion → egress``.

    The sequence lens — what a run *did*, with no health/regression signals (that's
    ``render_run_record``'s job).  Each step renders as the **canonical projection**
    — call syntax (``LoggedToolCall.render``) plus a compact one-line result, ids
    named with their type (``run <id>``, ``step <N>``, ``image #<id>``) so each feeds
    exactly one addressing tool — not first-person narration (the model composes prose
    from this in-context trace).  A run that produced tool calls is a candidate
    workflow (the sequence IS the skill).

    Step numbers are the run's FULL tool-call ordinals (``done`` consumes an index
    but isn't shown), so they are stable coordinates: a view that omits steps shows
    gaps, never renumbers — future skill selection can name ``(run_id, steps 2–5)``
    unambiguously (#1471).  Leads with the run's own id and closes with the egress/
    media trace — what was attached/delivered alongside the reply — so a reader can
    inspect its own turn's delivery (tools + outcomes + what went out) instead of
    confabulating it, every referenced neighbour named, never guessed (#1560)."""
    if not prompts:
        return "(no data)"
    results = _tool_results_by_id(prompts)
    lines = [_run_header(prompts), _run_origin(prompts)]
    for index, (call_id, call) in enumerate(_run_logged_steps(prompts), start=1):
        if call.name == "done":
            continue  # consumes the coordinate (gap, never renumber), not shown
        lines.append(_step_line(index, call, results.get(call_id) if call_id else None))
    conclusion = _run_conclusion(prompts)
    if conclusion:
        lines.append(conclusion)
    lines.extend(_egress_media_lines(prompts))
    return "\n".join(line for line in lines if line)


class RunProjectionStep(BaseModel):
    """One tool call of a run, with its absolute ordinal — the selection
    coordinate ``skill_create`` addresses (#1590).  ``ordinal`` is the run's FULL
    tool-call position (``done`` consumes one, matching ``render_run_calls``); a
    filtered view shows gaps, never renumbers.

    ``success`` is the STRUCTURAL per-call execution stamp (#1600), hydrated from
    the tool-result message's ``tool_success`` bit: ``True`` (the call succeeded),
    ``False`` (it failed), or ``None`` (no stamp — a run logged before #1600).
    ``skill_create``'s certified-by-execution gate reads this boolean, not the
    framed result prose."""

    ordinal: int
    call_id: str | None
    call: LoggedToolCall
    success: bool | None = None


class RunProjection(BaseModel):
    """A run decomposed for skill authoring — its triggering user message, every
    tool call with its absolute ordinal + STRUCTURAL success stamp, and each call's
    framed result text (#1590/#1600).  The structural decomposition: each step
    carries its own ``success`` bit (read from the ledger's per-call stamp, #1600),
    so the certified-by-execution check reads a boolean off the step instead of
    parsing the result frame.  ``results`` (the framed result text) stays — the
    distillation reads it to infer each argument's provenance."""

    origin_message: str
    steps: list[RunProjectionStep]
    results: dict[str, str]


def project_run(prompts: list[PromptLog]) -> RunProjection:
    """Project a run into ``(origin_message, ordinaled steps, results-by-call-id)``
    — the raw material ``skill_create`` selects a contiguous range from (#1590).

    The step ordinals are the run's FULL tool-call ordinals (composed from the same
    ``_run_logged_steps`` ``render_run_calls`` numbers against), so a range the
    model reads there names the same steps here.  Results are the accumulated
    tool-turn contents keyed by call id (the ``_tool_results_by_id`` view)."""
    if not prompts:
        return RunProjection(origin_message="", steps=[], results={})
    successes = _tool_successes_by_id(prompts)
    steps = [
        RunProjectionStep(
            ordinal=index,
            call_id=call_id,
            call=call,
            success=successes.get(call_id) if call_id else None,
        )
        for index, (call_id, call) in enumerate(_run_logged_steps(prompts), start=1)
    ]
    return RunProjection(
        origin_message=_opening_user_message(prompts),
        steps=steps,
        results=_tool_results_by_id(prompts),
    )


class RunLog(Log):
    """Read facade over ``promptlog`` for the ``collector-runs`` log.

    ``collector-runs`` stores nothing of its own: every collector cycle already
    lives in ``promptlog`` (``run_id`` / ``run_target`` / ``run_outcome`` and its
    tool calls).  ``set_run_outcome`` stamps ``run_outcome`` on exactly one row
    per run (its last prompt), so the completion rows ARE the run index — one row
    per run, its timestamp the completion time, served by the
    ``ix_promptlog_completed_runs`` partial index (a bounded ``ORDER BY ... LIMIT``,
    not a ``GROUP BY`` scan).  Each run renders to a model-readable record: a
    ``[target] <outcome>`` header plus, for a run that did something, its compact
    tool-call trace.  Read-only; has no embeddings (so ``read_similar`` is empty).

    Scoped to every collector run (the log itself).  A single collection's runs
    are served as full runs by ``MessageStore.get_target_runs`` (the addon's
    Activity tab renders them as the prompts tab's run → prompts → turns cards).
    """

    def append(
        self, entries: list[LogEntryInput], author: str, run_id: str | None = None
    ) -> list[MemoryEntry]:
        raise ReadOnlyMemoryError(
            f"'{self.name}' is a read view over collector run history (promptlog) — "
            "runs are recorded by the dispatcher, not appended here."
        )

    def read_window(self, window_seconds: int, limit: int | None = None) -> list[MemoryEntry]:
        # Runs are bounded per read like the cursored batch — a chat "what ran
        # recently" shouldn't dump an unbounded window of run records.
        return self.read_recent(window_seconds, limit or PennyConstants.LOG_READ_LIMIT)

    def _completion_clauses(self) -> list:
        return [
            PromptLog.run_outcome.isnot(None),  # ty: ignore[unresolved-attribute]
            PromptLog.run_target.isnot(None),  # ty: ignore[unresolved-attribute]
        ]

    def _all_rows(self) -> list[MemoryEntry]:
        return self._records(newest_first=False)

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        # ``search`` doesn't apply — runs are activity, not searchable entries.
        return self._records(newest_first=True, limit=k, offset=offset)

    def _rows_since(self, cursor: datetime, cap: int | None) -> list[MemoryEntry]:
        return self._records(newest_first=False, cursor=cursor, limit=cap)

    def _embedded_rows(self) -> list[MemoryEntry]:
        return []

    def _records(
        self,
        *,
        newest_first: bool,
        cursor: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """Completion rows → rendered run records as ``MemoryEntry`` (content =
        the record, created_at = completion time)."""
        with self._session() as session:
            rows = self._completion_rows(session, newest_first, cursor, limit, offset)
            if not rows:
                return []
            grouped = self._group_prompts(session, [run_id for run_id, _, _ in rows])
        return [self._to_record(run_id, ts, last_id, grouped) for run_id, ts, last_id in rows]

    def _completion_rows(
        self,
        session: Session,
        newest_first: bool,
        cursor: datetime | None,
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
    "render_run_calls",
    "classify_run",
    "RunHealth",
    "LoggedToolCall",
    "MessageLogMemory",
    "RunLog",
    "RunProjection",
    "RunProjectionStep",
    "project_run",
]
