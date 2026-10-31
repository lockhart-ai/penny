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
    ├─ Collection            keyed: get / keys / entry_count / read_latest /
    │                        read_random / write / update / move / delete
    │                        (log ops refuse)
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
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field, computed_field
from similarity.identity import KeyRule, pair_rule_verdict
from sqlalchemy import func
from sqlmodel import Session, col, select

from penny.clock import Clock
from penny.config_params import RuntimeParams
from penny.constants import (
    WRITE_GATE_MUTATING_OUTCOMES,
    WRITE_GATE_STOP_REASONS,
    DispatchReason,
    PennyConstants,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database.memory import _similarity as sim
from penny.database.memory.identity import (
    EntryDecision,
    EntryGiven,
    plan_entries,
    rule_only_decisions,
)
from penny.database.memory.journal import CycleJournal
from penny.database.memory.types import (
    EntryInput,
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
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog, SendQueueItem
from penny.database.mutation_store import EnumeratedDecision
from penny.datetime_utils import format_duration_plain_words, format_user_clock_time
from penny.llm.models import strip_harmony_control_tokens
from penny.text_validity import LEGACY_TOOL_RENAMES, degenerate_reason, half_formed_send_reason
from penny.validation.conditions import ConditionKey, run_flag_conditions

logger = logging.getLogger(__name__)


class Memory:
    """Base memory: ``memory_entry`` row access + shape-independent reads.

    Never instantiated directly — ``db.memory(name)`` returns a ``Collection``,
    ``Log``, or a facade.  Holds the metadata ``row`` and exposes it as
    properties; defines every shape op as a no-op that raises ``WrongShapeError``
    so a subclass that doesn't serve that shape refuses with a readable message.
    """

    def __init__(
        self, row: MemoryRow, engine, *, on_changed=None, clock: Clock | None = None
    ) -> None:
        self.row = row
        self._engine = engine
        self._on_changed = on_changed
        # The clock entries are stamped from and a recency window is counted back from
        # — the store's own, handed down by the factory that builds this object.
        self._clock = clock if clock is not None else Clock()

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

    def entry_count(self) -> int:
        self._refuse_collection_op()
        return 0

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
        run_id: str | None = None,
        *,
        decisions: list[EntryDecision],
    ) -> list[WriteResult]:
        self._refuse_collection_op()
        return []

    def identity_givens(self) -> list[EntryGiven]:
        """Every stored entry as the identity ladder sees it — collection-only
        (a log's entries are keyless streams, not identity candidates)."""
        self._refuse_collection_op()
        return []

    def write_rule_decided(
        self, entries: list[EntryInput], author: str, run_id: str | None = None
    ) -> list[WriteResult]:
        """The manual write path — rules-only identity, no decision model in the
        loop (collection-only; see ``Collection.write_rule_decided``)."""
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
        model-facing ``collection_list`` is the ``Collection.read_latest``
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
        cutoff = self._clock.now() - timedelta(seconds=window_seconds)
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

    def embedded_rows(self) -> list[MemoryEntry]:
        """Every entry in this memory carrying a content embedding, unranked —
        the public hook onto the same rows ``read_similar`` ranks (``_embedded_rows``,
        shape-specific: ``memory_entry`` for a collection/log, ``messagelog`` for a
        message facade).  ``MemoryStore`` reads this to fuse the two message logs into
        ``find``'s own ranking (#2289), so a message candidate and a ``read_similar``
        hit are the identical row and vector — no second embedding, no second query
        shape."""
        return self._embedded_rows()

    # ── Row primitives (memory_entry; facades override) ───────────────────────

    def _all_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .where(MemoryEntry.memory_name == self.name)
                    .order_by(col(MemoryEntry.created_at).asc())
                ).all()
            )

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == self.name)
                .order_by(col(MemoryEntry.created_at).desc())
            )
            if search:
                like = f"%{search}%"
                query = query.where(
                    col(MemoryEntry.content).like(like) | col(MemoryEntry.key).like(like)
                )
            if offset:
                query = query.offset(offset)
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def _rows_since(
        self, cursor: datetime, cap: int | None, until: datetime | None = None
    ) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == self.name, MemoryEntry.created_at > cursor)
                .order_by(col(MemoryEntry.created_at).asc())
            )
            if until is not None:
                query = query.where(MemoryEntry.created_at < until)
            if cap is not None:
                query = query.limit(cap)
            return list(session.exec(query).all())

    def read_range(
        self, start: datetime, until: datetime, cap: int | None = None
    ) -> list[MemoryEntry]:
        """Entries within ``[start, until)``, oldest first — the conversation
        map's day read (#2752): one local day's entries, bounded both ways so
        the read never walks the whole log to reach an old day."""
        return self._rows_since(start, cap, until=until)

    def _embedded_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == self.name,
                        col(MemoryEntry.content_embedding).is_not(None),
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

    def __init__(
        self,
        row: MemoryRow,
        engine,
        *,
        runtime: RuntimeParams,
        on_changed=None,
        journal: CycleJournal | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(row, engine, on_changed=on_changed, clock=clock)
        self._runtime = runtime
        # The bound collector cycle's UNDO journal (#1936), and ``None`` for every
        # ordinary caller — chat included, whose turn IS its conclusion.  With one, each
        # mutation first records what the key it touches held, so an unhealthy end can
        # put the collection back.  The writes themselves are unchanged.  Collection-only
        # because a journal is keyed by ENTRY KEY, which a log has none of.
        self._journal = journal

    def _journal_prior(self, key: str, rows: list[MemoryEntry]) -> None:
        """Record what ``key`` held before the bound cycle first touched it (#1936).

        Called from each mutating path with the rows that path has ALREADY read to do
        its work, so the record costs no extra query — and it is keyed to the MUTATION
        rather than to a tool, which is what makes ``collection_write``,
        ``collection_remove`` and the change-gate's own
        ``KEY_EXISTS_CHANGED`` baseline refresh (#1633) all recorded identically.  That
        last one is the write nobody makes explicitly, and precisely the one an aborted
        cycle used to leave behind.
        """
        if self._journal is not None:
            self._journal.capture(key, rows)

    def read_latest(
        self, k: int | None = None, offset: int = 0, search: str | None = None
    ) -> list[MemoryEntry]:
        """The model-facing newest-first collection read (``collection_list``).
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

    def entry_count(self) -> int:
        """How many entries this collection holds — a COUNT in SQL, never the length
        of a read.

        What a caller rendering a BOUNDED window needs to say honestly how much it
        left out (the collector's holdings block, #1914): reading the whole partition
        just to subtract is the materialize-then-paginate shape, and this collection
        is the partition the write path already scans on every write.  Collection-only
        like every other keyed read — a log's entries are counted through its own
        cursored reads."""
        with self._session() as session:
            return session.exec(
                select(func.count(MemoryEntry.id)).where(  # ty: ignore[invalid-argument-type]
                    MemoryEntry.memory_name == self.name
                )
            ).one()

    def keys(self) -> list[str]:
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MemoryEntry.key)
                    .where(
                        MemoryEntry.memory_name == self.name,
                        col(MemoryEntry.key).is_not(None),
                    )
                    .order_by(col(MemoryEntry.created_at).asc())
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
        run_id: str | None = None,
        *,
        decisions: list[EntryDecision],
    ) -> list[WriteResult]:
        """Write entries through the change-gate, applying the identity ladder's
        decisions — one ``WriteResult`` per input.

        ``decisions`` (from ``penny.database.memory.identity``, one per entry, in
        order) are a REQUIRED parameter: the caller ran the ladder — rules,
        shortlist, decision model or its named degraded mode — and this write
        only applies what was decided.  ``run_id`` (the writing run, threaded as
        a parameter — no ambient state) stamps ``created_by_run_id`` and
        ``last_written_by_run_id`` on each new row, so an entry cites the run
        that produced it (#1560).  A ``KEY_EXISTS_CHANGED`` entry — an exact-key
        change or a ladder merge — auto-refreshes the stored baseline in place
        through the same update path (#1633) — so it too advances
        ``last_written_by_run_id`` and counts as a change for the notify below.
        """
        if len(decisions) != len(entries):
            raise ValueError(
                f"one identity decision per entry required: {len(decisions)} for {len(entries)}"
            )
        results: list[WriteResult] = []
        written: list[EntryGiven] = []
        with self._session() as session:
            for entry, decision in zip(entries, decisions, strict=True):
                results.append(self._write_one(session, entry, author, decision, written, run_id))
            session.commit()
        if any(result.outcome in WRITE_GATE_MUTATING_OUTCOMES for result in results):
            self._notify()
        return results

    def identity_givens(self) -> list[EntryGiven]:
        """Every stored entry as the identity ladder sees it, oldest first so the
        rules' first match is deterministic across runs."""
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MemoryEntry)
                    .where(MemoryEntry.memory_name == self.name)
                    .order_by(col(MemoryEntry.created_at).asc())
                ).all()
            )
        return [
            EntryGiven(
                row.key,
                row.content,
                sim.maybe_deserialize(row.key_embedding),
                sim.maybe_deserialize(row.content_embedding),
            )
            for row in rows
        ]

    def write_rule_decided(
        self, entries: list[EntryInput], author: str, run_id: str | None = None
    ) -> list[WriteResult]:
        """The manual write path (the addon and iOS entry writes): identity by
        the deterministic rules only, no decision model in the loop.  What the
        rules cannot settle is written as new — a human is driving the UI and
        will see both entries on refresh."""
        givens = self.identity_givens()
        plans = plan_entries(entries, givens)
        decisions = rule_only_decisions(plans, entries, givens)
        return self.write(entries, author, run_id=run_id, decisions=decisions)

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
            self._journal_prior(key, rows)
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
            self._journal_prior(key, rows)
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

    def _write_one(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        decision: EntryDecision,
        written: list[EntryGiven],
        run_id: str | None,
    ) -> WriteResult:
        """Classify one write into the closed ``WriteGateOutcome`` union — the
        change-gate at the write chokepoint (#1587).  Degenerate content and the
        exact-key path are the gate's own deterministic comparisons; everything
        else APPLIES the identity decision the caller's ladder produced
        (``penny.database.memory.identity``) — merge into the matched entry,
        report the value unchanged, or insert — with the in-batch rules guard
        catching a later entry that collides with one this batch just wrote.

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
        if decision.action == "merge":
            return self._apply_merge(session, entry, author, decision, written, run_id)
        if decision.action == "unchanged":
            return self._apply_unchanged(session, entry, author, decision, run_id)
        return self._apply_new(session, entry, author, decision, written, run_id)

    def _apply_merge(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        decision: EntryDecision,
        written: list[EntryGiven],
        run_id: str | None,
    ) -> WriteResult:
        """The merge: the ladder matched this write to an existing entry holding a
        DIFFERENT value — replace that value in place through the shared update
        path (#1633), the prior journaled so a wrong match is reversible.  A
        merge target that vanished between the ladder's read and this write
        (a concurrent write) degrades honestly to a fresh insert, logged."""
        matched = decision.matched_key
        assert matched is not None  # a merge decision always names its target
        rows = self._rows_by_key(session, self.name, matched)
        if not rows:
            logger.warning(
                "collection write: merge target '%s' vanished; writing '%s' as new",
                matched,
                entry.key,
            )
            return self._apply_new(session, entry, author, decision, written, run_id)
        self._journal_prior(matched, rows)
        prior = rows[0].content
        self._apply_content(session, rows, entry.content, author, run_id)
        return WriteResult(
            key=entry.key,
            outcome=WriteGateOutcome.KEY_EXISTS_CHANGED,
            matched_key=matched,
            prior_value=prior,
        )

    def _apply_unchanged(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        decision: EntryDecision,
        run_id: str | None,
    ) -> WriteResult:
        """The no-news outcome: the ladder matched an entry that already says what
        this write says.  Under the write's own key that is ``KEY_EXISTS_UNCHANGED``
        (the watch's signal); under a different key, ``DUPLICATE_UNCHANGED``
        (#1919).  A match that vanished between the ladder's read and this write
        degrades honestly to a fresh insert, logged."""
        matched = decision.matched_key
        assert matched is not None  # an unchanged decision always names its match
        rows = self._rows_by_key(session, self.name, matched)
        if not rows:
            logger.warning(
                "collection write: unchanged target '%s' vanished; writing '%s' as new",
                matched,
                entry.key,
            )
            return self._apply_new(session, entry, author, decision, [], run_id)
        outcome = (
            WriteGateOutcome.KEY_EXISTS_UNCHANGED
            if matched == entry.key
            else WriteGateOutcome.DUPLICATE_UNCHANGED
        )
        return WriteResult(key=entry.key, outcome=outcome, matched_key=matched)

    def _apply_new(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        decision: EntryDecision,
        written: list[EntryGiven],
        run_id: str | None,
    ) -> WriteResult:
        """Write-through: the ladder settled this entry as new (or was unsure —
        the write proceeds and the reply asks honestly).  The in-batch rules
        guard re-runs the deterministic rules against what THIS batch already
        wrote, so a look-alike arriving later in the same batch still merges or
        reads as unchanged — the decisions were planned against the pre-batch
        store."""
        for prior in written:
            prior_key = prior.key or ""
            rule = pair_rule_verdict(entry.key, prior_key, entry.content, prior.content)
            if rule is KeyRule.DIFFERENT or rule is None:
                continue
            rows = self._rows_by_key(session, self.name, prior_key)
            if not rows:
                continue
            if _content_unchanged(rows[0].content, entry.content):
                outcome = (
                    WriteGateOutcome.KEY_EXISTS_UNCHANGED
                    if prior_key == entry.key
                    else WriteGateOutcome.DUPLICATE_UNCHANGED
                )
                return WriteResult(key=entry.key, outcome=outcome, matched_key=prior_key)
            self._journal_prior(prior_key, rows)
            prior = rows[0].content
            self._apply_content(session, rows, entry.content, author, run_id)
            return WriteResult(
                key=entry.key,
                outcome=WriteGateOutcome.KEY_EXISTS_CHANGED,
                matched_key=prior_key,
                prior_value=prior,
            )
        result = self._insert_new_entry(session, entry, author, run_id, decision)
        written.append(
            EntryGiven(entry.key, entry.content, entry.key_embedding, entry.content_embedding)
        )
        return result

    def _exact_key_outcome(
        self,
        session: Session,
        stored: list[MemoryEntry],
        entry: EntryInput,
        author: str,
        run_id: str | None,
    ) -> WriteResult:
        """The exact-key hit — compared by value, not embedding.  Identical value →
        KEY_EXISTS_UNCHANGED (the watch's "no change" signal).  Different
        value → KEY_EXISTS_CHANGED: the gate **auto-refreshes the stored baseline
        itself** in place through the shared update path (#1633), so the model runs no
        further call — the reply shows the prior and the one call that restores it.
        A degenerate new value never
        reaches here — it classified DEGENERATE upstream — so the refresh only ever
        stores screened content."""
        if _content_unchanged(stored[0].content, entry.content):
            return WriteResult(
                key=entry.key, outcome=WriteGateOutcome.KEY_EXISTS_UNCHANGED, matched_key=entry.key
            )
        self._journal_prior(entry.key, stored)
        prior = stored[0].content
        self._apply_content(session, stored, entry.content, author, run_id)
        return WriteResult(
            key=entry.key,
            outcome=WriteGateOutcome.KEY_EXISTS_CHANGED,
            matched_key=entry.key,
            prior_value=prior,
        )

    def _insert_new_entry(
        self,
        session: Session,
        entry: EntryInput,
        author: str,
        run_id: str | None,
        decision: EntryDecision,
    ) -> WriteResult:
        """Persist a genuinely new key (NEW_KEY) — the ladder settled it as new,
        or was unsure and the write proceeds (the reply carries the honest
        question via ``unsure_candidate``).

        Its prior state is the EMPTY one — the gate only reaches here when the key had
        no rows — so an unhealthy end deletes what this write inserted (#1936)."""
        self._journal_prior(entry.key, [])
        row = MemoryEntry(
            memory_name=self.name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=sim.maybe_serialize(entry.key_embedding),
            content_embedding=sim.maybe_serialize(entry.content_embedding),
            created_at=self._clock.now(),
            created_by_run_id=run_id,
            last_written_by_run_id=run_id,
        )
        session.add(row)
        session.flush()
        return WriteResult(
            key=entry.key,
            outcome=WriteGateOutcome.NEW_KEY,
            entry_id=row.id,
            unsure_candidate=decision.unsure_candidate,
        )
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
                    created_at=self._clock.now(),
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
    # the tool layer's ``format_entries``) and the message content:
    # ``3. [2026-07-02 09:14 UTC] (sent by price-watch) Heads up: …``.
    SENT_BY_MARKER = "(sent by {mechanism}) "

    def __init__(
        self,
        row: MemoryRow,
        engine,
        *,
        direction: str,
        on_changed=None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(row, engine, on_changed=on_changed, clock=clock)
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
            col(MessageLog.is_reaction).is_(False),
        )

    def _all_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            rows = session.exec(self._select().order_by(col(MessageLog.timestamp).asc())).all()
        return [self._to_entry(row) for row in rows]

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        with self._session() as session:
            query = self._select().order_by(col(MessageLog.timestamp).desc())
            if search:
                query = query.where(col(MessageLog.content).like(f"%{search}%"))
            if offset:
                query = query.offset(offset)
            if k is not None:
                query = query.limit(k)
            rows = session.exec(query).all()
        return [self._to_entry(row) for row in rows]

    def _rows_since(
        self, cursor: datetime, cap: int | None, until: datetime | None = None
    ) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                self._select()
                .where(MessageLog.timestamp > cursor)
                .order_by(col(MessageLog.timestamp).asc())
            )
            if until is not None:
                query = query.where(MessageLog.timestamp < until)
            if cap is not None:
                query = query.limit(cap)
            rows = session.exec(query).all()
        return [self._to_entry(row) for row in rows]

    def _embedded_rows(self) -> list[MemoryEntry]:
        with self._session() as session:
            rows = session.exec(self._select().where(col(MessageLog.embedding).is_not(None))).all()
        return [self._to_entry(row) for row in rows]


def _collection_field(fields: dict) -> Any:
    """A logged call's collection name, read however it was logged.

    The six entry tools name this argument ``collection`` today; a call logged
    before the rename stored it as ``memory``.  The stored row is never
    rewritten, so a reader pulls a collection's name out of a logged call's
    arguments through this one accessor rather than re-deriving the fallback at
    each site."""
    return fields.get("collection", fields.get("memory", "?"))


def _log_field(fields: dict) -> Any:
    """A logged call's log name, read however it was logged.

    ``log_read`` / ``log_append`` name this argument ``log`` today; a call logged
    before the rename stored it as ``memory``.  The stored row is never
    rewritten, so a reader pulls a log's name out of a logged call's arguments
    through this one accessor rather than re-deriving the fallback at each
    site."""
    return fields.get("log", fields.get("memory", "?"))


def render_tool_call(name: str, args: object) -> str:
    """Compact, grokkable render of one tool call (the salient args only).

    The single shared format for the ``collector-runs`` run trace, so the model
    reads what a cycle did the same way everywhere.  Every call renders under its
    **real** tool name in the canonical ``tool(args)`` shape — the same dialect a
    prompt writes a call in (``docs/prompt-writing-guide.md`` → "The canonical call
    notation", point 6), so what the model reads of its past matches how it's told
    to act, and a wrong-shape or hallucinated sibling name is identifiable against
    the real one.  kwargs form (``collection=…, key=…``) when a call carries more
    than one argument; a single obvious argument stays positional.  Kept compact
    (the salient args only, content never truncated) — the aliases these replaced
    saved a few tokens per line; the real names cost them back, and nothing else
    is added.
    """
    fields = cast("dict[str, Any]", args if isinstance(args, dict) else {})
    if name == PennyConstants.COLLECTION_WRITE_TOOL_NAME:
        return (
            f"collection_write(collection={_collection_field(fields)!r}, "
            f"entries={_write_contents(fields)!r})"
        )
    if name in (PennyConstants.LOG_READ_TOOL_NAME, PennyConstants.LOG_APPEND_TOOL_NAME):
        return f"{name}(log={_log_field(fields)!r})"
    if name == PennyConstants.SEND_MESSAGE_TOOL_NAME:
        return f"send_message({fields.get('content', '')!r})"
    if name == PennyConstants.BROWSE_TOOL_NAME:
        # The items shape (#2718) renders each item's query and its looking-for.
        # A row recorded before the items shape renders in the CURRENT dialect
        # too — the old queries+extract call IS one item per query with
        # looking_for=extract, and rendering it in the dead dialect taught the
        # reading model to emit the dead shape: measured on #2718's scoped run,
        # 90 of 455 browse calls arrived in the old shape, each an arg-gate
        # rejection plus a recovery round trip.  A bare queries row (older than
        # the required extract) keeps its plain render — it has no looking-for
        # to translate.
        if "items" in fields:
            rendered = ", ".join(
                f"({item.get('query', '')!r} looking for {item.get('looking_for', '')!r})"
                for item in fields.get("items", [])
                if isinstance(item, dict)
            )
            return f"browse(items=[{rendered}])"
        queries = fields.get("queries", list(fields.values()))
        extract = fields.get("extract")
        if extract:
            rendered = ", ".join(f"({q!r} looking for {extract!r})" for q in queries)
            return f"browse(items=[{rendered}])"
        return f"browse({queries!r})"
    # A dict (incl. the empty dict of an argless call, e.g. ``done()``, #1569)
    # renders its fields; a non-dict / malformed args string shows its repr so the
    # trace still names what the model tried.  ``reasoning`` is filtered here, not
    # at the source: a row logged before #2813 dropped the framework-injected
    # per-call rationale still carries the raw key verbatim in ``promptlog.response``
    # (that capture is never rewritten), and a model may still send a stray one for
    # a while — this run trace is model-facing and no longer shows it either way.
    if isinstance(args, dict):
        rendered = ", ".join(
            f"{key}={value!r}"
            for key, value in args.items()
            if key != PennyConstants.STRAY_REASONING_ARG_KEY
        )
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


def _resolve_logged_tool_name(name: str) -> str:
    """The dispatchable identifier for a logged/wire tool NAME (#2307): strip a
    leaked Harmony control token, then resolve a retired name to the one it was
    renamed to (:data:`LEGACY_TOOL_RENAMES`) — so every reader that compares a
    logged call's tool against a current surface or a name set (``_WRITE_TOOLS``
    and its kin) compares CURRENT names only, whether the row was logged before
    or after the rename.  The stored row is never rewritten; only the read
    projection resolves it.  Shared by ``LoggedToolCall.tool`` and
    ``run_tool_calls`` so the two readers cannot drift onto two answers for the
    same logged name."""
    stripped = strip_harmony_control_tokens(name)
    return LEGACY_TOOL_RENAMES.get(stripped, stripped)


class LoggedToolCall(BaseModel):
    """The canonical, round-trippable shape of one logged tool call (#1560).

    A logged call and the outgoing wire call are the SAME structure: the model's
    response is stored verbatim in ``promptlog.response`` (``raw.model_dump()``),
    where a tool call is ``{"function": {"name": …, "arguments": "<json string>"}}``
    — the exact envelope the loop re-emits via ``LlmMessage.to_log_dict`` and
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

    @property
    def tool(self) -> str:
        """WHICH TOOL this call names — the dispatchable identifier, the thing to
        compare against a registry, a surface, or a sentinel (#2013).

        ``name`` is verbatim by design (the log never paraphrases the call), and a
        backend that leaks Harmony control tokens makes that verbatim name unusable
        as an identifier: ``collection_write<|channel|>commentary`` dispatched fine
        on the LIVE object — ``LlmToolCallFunction`` strips at the read-off boundary
        — but every consumer re-reading the LOG compared the leaked string and got
        no match, which silently dropped the write step from a distilled routine on
        3 of 18 measured samples.  So the repair lives here rather than in storage:
        the record keeps the leak (it is a real diagnostic signal), and anything
        asking *which tool was this* reads ``tool``.  Uses the production strip, so
        the live object and the log agree on the answer by construction.  Since
        #2307 it also resolves a retired name through ``LEGACY_TOOL_RENAMES``
        (:func:`_resolve_logged_tool_name`), so a call logged under an old name
        (``collection_read_latest``, ``collection_delete_entry``, …) compares equal
        to its current one wherever ``.tool`` is read."""
        return _resolve_logged_tool_name(self.name)

    @classmethod
    def from_function(cls, function: dict) -> LoggedToolCall:
        """Parse a stored/wire ``function`` dict into the canonical shape."""
        return cls(
            name=function.get("name") or "",
            arguments=_decode_arguments(function.get("arguments")),
        )

    def to_wire(self) -> dict[str, str]:
        """Re-emit the OpenAI wire ``function`` envelope — arguments re-serialized
        exactly as ``LlmMessage.to_log_dict`` does, so a round-trip is the
        identity and the loop would re-send this call unchanged."""
        return {"name": self.name, "arguments": json.dumps(self.arguments)}

    def render(self) -> str:
        """The canonical one-line ``tool(args)`` projection — call syntax, not
        narration.  ``run_list`` renders a run as this canonical projection
        (the prose is composed by the model from the in-context trace); the
        first-person ``to_result_narration`` is a separate projection.

        Renders the ``tool``, not the verbatim ``name``: this is an ANCHOR the model
        reads and then addresses (n≤1 reachability, #1560), so a leaked-token name
        here would render an identifier that resolves nowhere.  The verbatim name
        stays in ``promptlog.response``, which the addon's prompt view renders raw."""
        return render_tool_call(self.tool, self.arguments)


def _run_logged_steps(
    prompts: list[PromptLog],
) -> list[tuple[str | None, LoggedToolCall, str | None]]:
    """Every tool call across a run's prompts as ``(call_id, LoggedToolCall,
    thinking)``, in order — the canonical projection paired with the id that keys
    its result and the owning model call's raw thinking (``None`` when that call
    made or logged none).  Several calls from the SAME completion (a parallel
    draw) share that completion's one thinking string — there is one decision per
    model call, not per call it emits."""
    steps: list[tuple[str | None, LoggedToolCall, str | None]] = []
    for prompt in prompts:
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                steps.append(
                    (
                        tool_call.get("id"),
                        LoggedToolCall.from_function(function),
                        prompt.thinking,
                    )
                )
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


def _run_dispatch(prompts: list[PromptLog]) -> tuple[datetime | None, str | None, int | None]:
    """Due occurrence / dispatch reason / attempt (#2862) from the last prompt
    carrying them — the SAME stamp-location convention ``_run_outcome`` reads,
    since both are set once, on the run's last prompt row."""
    for prompt in reversed(prompts):
        if prompt.run_dispatch_reason is not None:
            return prompt.run_due_at, prompt.run_dispatch_reason, prompt.run_dispatch_attempt
    return None, None, None


# ``ON_TIME``/``ON_DEMAND`` are dead branches from ``dispatch_line``'s own call
# (both return early above it) — live only for the self-state jobs-list tail
# (``dispatch_reason_clause`` called directly, #2862), which has no header to
# short-circuit into.
_DISPATCH_TAILS: dict[DispatchReason, str] = {
    DispatchReason.ON_TIME: "on time",
    DispatchReason.CATCH_UP: "catching up a missed run",
    DispatchReason.ON_DEMAND: "on demand",
}


def dispatch_reason_clause(reason: DispatchReason, due: str, attempt: int | None) -> str:
    """The clause after the ``·`` — why this cycle started when it did, never the
    attempt it ended on: a retry names which prior attempt it follows (preempted
    by foreground activity, or failed) and its own attempt number; a catch-up
    across a restart names the due time Penny was offline at (#2862)."""
    if reason is DispatchReason.CATCH_UP_OFFLINE:
        return f"catching up a missed run: Penny was offline at {due}"
    if reason is DispatchReason.RETRY_PREEMPTED:
        return f"retrying after an interrupted attempt (attempt {attempt})"
    if reason is DispatchReason.RETRY_FAILED:
        return f"retrying after a failed attempt (attempt {attempt})"
    return _DISPATCH_TAILS.get(reason, "")


def dispatch_line(prompts: list[PromptLog], timezone_name: str | None, schedule_words: str) -> str:
    """The run's own dispatch-reason line (#2862): ``started <time> for the
    <cadence> <due time> run · <why>`` — or, for an on-demand run, just
    ``started <time> on demand``.  Empty for a chat run and for a row logged
    before this stamp existed.

    ``schedule_words`` is the collection's CURRENT cadence in plain words (e.g.
    "daily"), fetched by the caller off its live ``schedule`` — empty when the
    collection no longer carries one (archived/deleted/retired since this run),
    in which case the clause drops the word rather than inventing one."""
    if not prompts:
        return ""
    due_at, reason_value, attempt = _run_dispatch(prompts)
    if reason_value is None:
        return ""
    reason = DispatchReason(reason_value)
    started = format_user_clock_time(prompts[0].timestamp, timezone_name)
    if reason is DispatchReason.ON_DEMAND:
        return f"started {started} on demand"
    assert due_at is not None  # every non-on-demand reason stamps its occurrence
    due = format_user_clock_time(due_at, timezone_name)
    words = f"{schedule_words} " if schedule_words else ""
    header = f"started {started} for the {words}{due} run"
    tail = dispatch_reason_clause(reason, due, attempt)
    return f"{header} · {tail}" if tail else header


def run_tool_calls(prompts: list[PromptLog]) -> list[tuple[str, object]]:
    """Ordered (tool, arguments) for every tool call across the run's prompts.

    The name is the DISPATCHABLE identifier, not the verbatim one — every caller
    below tallies, classifies or renders by tool, and a leaked Harmony control token
    on the logged name matches none of them (#2013), and nor would a name a later
    rename retired (#2307) — both repaired by the same ``_resolve_logged_tool_name``
    ``LoggedToolCall.tool`` reads, so the two readers cannot disagree.  ``?`` stands
    in when the call named no tool at all, so a nameless call still occupies its
    place in the run."""
    calls: list[tuple[str, object]] = []
    for prompt in prompts:
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                tool = _resolve_logged_tool_name(function.get("name") or "")
                calls.append((tool or "?", _parse_tool_args(function)))
    return calls


def _ended_via_write_gate_stop(prompts: list[PromptLog]) -> bool:
    """True when a write-gate STOP ended the run (#1587): the run reached a clean
    outcome (``no_work`` / ``worked``) carrying one of the DECLARED stop reasons.

    Read against the declared table rather than inferred from the absence of a
    ``done()`` call (#1911).  That inference held while ``done()`` was how every clean
    cycle closed, so "clean and no done" could only be a STOP; a cycle now closes
    CLEANLY when its program is covered, with no ``done()`` either, and the old read
    would have called every completed run a write-gate stop.  The stamped reason is
    the fact that actually distinguishes them, and it was always there."""
    outcome, reason, _ = _run_outcome(prompts)
    if outcome not in (RunOutcome.NO_WORK.value, RunOutcome.WORKED.value):
        return False
    return reason in set(WRITE_GATE_STOP_REASONS.values())


def _run_tool_failures(prompts: list[PromptLog]) -> int:
    """The persisted failed-tool-call count, off the run's outcome-bearing row.

    Stamped by ``set_run_outcome`` on the last prompt; NULL on old/untagged rows
    reads as zero (not measured)."""
    for prompt in reversed(prompts):
        if prompt.run_outcome is not None:
            return prompt.tool_failures or 0
    return 0


_WRITE_TOOLS = frozenset(
    {
        PennyConstants.COLLECTION_WRITE_TOOL_NAME,
        PennyConstants.COLLECTION_REMOVE_TOOL_NAME,
        PennyConstants.LOG_APPEND_TOOL_NAME,
    }
)

_ROLE_TOOL = "tool"


def _run_tool_turns(prompts: list[PromptLog]) -> list[dict]:
    """Every tool-result turn of a run, in order — the COMPLETE set (#1778).

    Two halves, read as one sequence.  The run's last prompt's ``messages`` holds the
    full accumulated conversation, where every result that rode into a later call sits
    exactly once; its ``trailing_messages`` holds the tail no call carried — the
    terminal call's result, which a run ending immediately after executing a tool (a
    write-gate STOP, ``max_steps`` on a tool step, a reroll abort, an exception) would
    otherwise have written nowhere.  Both are the same wire shape, so this is a
    concatenation, not a merge; a run that round-tripped everything has an empty tail
    and reads exactly as it always did."""
    if not prompts:
        return []
    last = prompts[-1]
    turns = json.loads(last.messages) if last.messages else []
    return [
        message
        for message in [*turns, *last.get_trailing_messages()]
        if message.get("role") == _ROLE_TOOL
    ]


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
      ``collection_list``, ``read_similar``, ``memory_get``, …): every
      tool call that isn't a browse, write, send,
      or ``done()``.  These don't meaningfully fail, so they're a plain call count.

    Writes and sends count the model's own tool *calls* (each appears once across
    the per-step responses), so ``writes == 0`` is the hard fact that nothing was
    ever written — whatever a ``done()`` summary claims.

    Browse results are read off the run's COMPLETE tool turns (``_run_tool_turns`` —
    the final prompt's accumulated conversation plus the trailing tail no call carried,
    #1778), so every result appears exactly once including a terminal one; the call
    counts come from the responses, so they're complete regardless."""
    calls = run_tool_calls(prompts)
    writes = sum(1 for name, _ in calls if name in _WRITE_TOOLS)
    sends = sum(1 for name, _ in calls if name == PennyConstants.SEND_MESSAGE_TOOL_NAME)
    reads = sum(
        1
        for name, _ in calls
        if name not in _WRITE_TOOLS
        and name
        not in {
            PennyConstants.BROWSE_TOOL_NAME,
            PennyConstants.SEND_MESSAGE_TOOL_NAME,
            PennyConstants.DONE_TOOL_NAME,
        }
    )
    browses_ok = browses_failed = 0
    for message in _run_tool_turns(prompts):
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


# ── Per-run operational facts (#2843) ─────────────────────────────────────────
# One line per run — calls/output tokens/time, discards, tool errors (by
# tool), and whether a message went out and carried a link — read off existing
# columns only, never inferred from wording.  Shared by the ``run-health``
# facade (every run, cursor/day via ``log_read``) and ``render_run_calls``
# (the ``run_list``/``run_get`` view, already source-filterable).

# A bare ``http(s)://`` URL anywhere in text — the shape ``link_handles``
# registers a handle for.  Duplicated here rather than imported across the
# tools/database layering boundary (neither this module nor ``message_store``
# imports from ``tools/`` today; migration 0036 duplicates this same shape for
# the same reason).  Only ever tested against the FINAL, already-expanded text
# a run actually delivered — never a handle placeholder.
_LINK_RE = re.compile(r"https?://\S+")


def _run_calls_and_cost(prompts: list[PromptLog]) -> tuple[int, int, int]:
    """``(calls, output_tok, duration_ms)`` — one model-loop step per prompt
    row; ``completion_tokens``/``duration_ms`` summed across them, NULL read
    as 0 (the existing convention — a runner that never reported usage is
    never faked one, it's simply absent from the total)."""
    calls = len(prompts)
    output_tok = sum(p.completion_tokens or 0 for p in prompts)
    duration_ms = sum(p.duration_ms or 0 for p in prompts)
    return calls, output_tok, duration_ms


def _tally_clause(label: str, counts: dict[str, int]) -> str:
    """``"<label>: a×2, b×1"`` in first-seen order, or ``"<label>: none"`` —
    the shared shape for both the discard and the tool-error tally."""
    if not counts:
        return f"{label}: none"
    parts = ", ".join(f"{name}×{n}" for name, n in counts.items())
    return f"{label}: {parts}"


def _run_discard_tally(prompts: list[PromptLog]) -> str:
    """Every discarded draw in the run, grouped by its structural
    ``discard_reason`` (a ``ConditionKey`` string) — across ALL of the run's
    rows, not just the last, since more than one draw can be discarded before
    one lands.  Never a fixed reason list: a reason nothing here enumerates
    still tallies by its own stored string (the coming ``thinking_loop``
    reason needs no code change)."""
    counts: dict[str, int] = {}
    for prompt in prompts:
        if prompt.discard_reason:
            counts[prompt.discard_reason] = counts.get(prompt.discard_reason, 0) + 1
    return _tally_clause("discarded", counts)


def _run_tool_error_tally(prompts: list[PromptLog]) -> str:
    """Failed tool calls, BY TOOL — read off the structural per-call
    ``tool_success`` stamp (#1600) on the run's complete tool-result turns
    (``_run_tool_turns``), matched back to the calling tool via
    ``_run_logged_steps``'s ``(call_id, tool)`` pairing.  Never a text-content
    guess: a result logged before the stamp existed carries no key and reads
    as succeeded — the same convention ``_run_tool_failures`` already uses."""
    tool_by_call_id = {
        call_id: call.tool for call_id, call, _thinking in _run_logged_steps(prompts) if call_id
    }
    counts: dict[str, int] = {}
    for message in _run_tool_turns(prompts):
        if message.get(PennyConstants.TOOL_RESULT_SUCCESS_KEY) is False:
            tool = tool_by_call_id.get(message.get("tool_call_id") or "", "?")
            counts[tool] = counts.get(tool, 0) + 1
    return _tally_clause("tool errors", counts)


def run_delivered_text(session: Session, run_id: str) -> str | None:
    """The text actually delivered to the user BECAUSE of this run, or
    ``None`` when nothing went out — read off the two structural links a send
    can carry, never reconstructed from a tool call's arguments (there is no
    ``send_message`` tool any more, #1911, so there is nothing to reconstruct
    from): a chat run's own direct reply is ``messagelog.run_id`` (#2630); a
    collector run's notifying close hands off through the send queue
    (``SendQueueItem.report_run_id``, #2656) — the DELIVERED row's own stored
    content, read only once ``sent_at`` is stamped (a cancelled or failed
    compose never reached the user).  At most one of the two ever matches a
    given run — a run is a chat turn or a collector cycle, never both."""
    direct = session.exec(select(MessageLog.content).where(MessageLog.run_id == run_id)).first()
    if direct is not None:
        return direct
    return session.exec(
        select(SendQueueItem.content).where(
            SendQueueItem.report_run_id == run_id,
            col(SendQueueItem.sent_at).isnot(None),
        )
    ).first()


def _sent_and_links_clause(delivered_text: str | None) -> str:
    """``"sent 1 message, links: 1"`` / ``"sent 0 messages"`` — a run is a chat
    turn or a collector cycle, each capable of at most one delivered message
    (``run_delivered_text``), so this is always 0 or 1.  ``links`` tests the
    STORED, already-expanded delivered text for a URL — never a handle
    placeholder, and never reconstructed from what a tool call argued for."""
    if delivered_text is None:
        return "sent 0 messages"
    links = 1 if _LINK_RE.search(delivered_text) else 0
    return f"sent 1 message, links: {links}"


def run_facts_line(prompts: list[PromptLog], delivered_text: str | None) -> str:
    """The shared per-run facts clause: calls, output tokens, time, discards,
    tool errors (by tool), and what was sent — appended after a run's header
    (the ``run-health`` facade's one-liner, ``render_run_health``) or its
    conclusion (``render_run_calls``).  Facts only: every field reads an
    existing column, never inferred from wording."""
    calls, output_tok, duration_ms = _run_calls_and_cost(prompts)
    return " · ".join(
        [
            f"{calls} calls",
            f"{output_tok} output tok",
            format_duration_plain_words(duration_ms),
            _run_discard_tally(prompts),
            _run_tool_error_tally(prompts),
            _sent_and_links_clause(delivered_text),
        ]
    )


def render_run_health(
    prompts: list[PromptLog],
    delivered_text: str | None,
    *,
    timezone_name: str | None = None,
    schedule_words: str = "",
) -> str:
    """One run as its dispatch line (#2862, when it carries one) then a single
    operational-facts line: ``run <id> [<source>] [<outcome>]`` then the
    shared facts clause (#2843) — the ``run-health`` facade's record.
    ``source`` is ``"chat"`` or the collector's own name, the same vocabulary
    ``run_list``'s ``source`` argument takes; ``outcome`` is the stamped
    structural result when one exists (a collector always stamps one; an
    ordinary, successful chat turn stamps none) and is omitted — never
    guessed — when it doesn't."""
    if not prompts:
        return "(no data)"
    outcome, reason, target = _run_outcome(prompts)
    header = f"{_run_header(prompts)} [{target or PennyConstants.CHAT_AGENT_NAME}]"
    result = reason or outcome
    if result:
        header = f"{header} {result}"
    facts_line = f"{header} · {run_facts_line(prompts, delivered_text)}"
    dispatch = dispatch_line(prompts, timezone_name, schedule_words)
    return f"{dispatch}\n{facts_line}" if dispatch else facts_line


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
    calls = run_tool_calls(prompts)
    non_done = [name for name, _ in calls if name != PennyConstants.DONE_TOOL_NAME]
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
        name == PennyConstants.SEND_MESSAGE_TOOL_NAME and _is_degenerate_send(_send_content(args))
        for name, args in calls
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
    calls = run_tool_calls(prompts)
    non_done = [(name, args) for name, args in calls if name != PennyConstants.DONE_TOOL_NAME]
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

    The message is the last ``user`` turn of the run's first prompt.  When that
    turn carries an injected Live-context block fused in as
    ``<context>{SEP}<message>``, the block is stripped back off so a reader sees
    what the user actually said — not the recall/time scaffolding.  A real chat
    row carries the user's message as the BARE utterance (no separator prefix),
    so when the separator is absent the whole turn IS the message (#1661 — the
    prior split-only code returned ``""`` here, killing the trigger/description
    derivation for skills).  A collector/background run opens with an empty prompt
    (no user content at all), so this stays ``""`` — the structural signal it had
    no user intent.
    """
    if not prompts or not prompts[0].messages:
        return ""
    messages = json.loads(prompts[0].messages)
    user_turns = [m.get("content") or "" for m in messages if m.get("role") == _ROLE_USER]
    if not user_turns:
        return ""
    raw = user_turns[-1]
    if PennyConstants.SECTION_SEPARATOR not in raw:
        return raw.strip()
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
    """How the run ended, STRUCTURALLY (#1569/#1916): ``penny: <reply>`` (chat — the
    user-facing reply stays model-authored), ``stopped: <reason>`` (a write-gate STOP
    at the chokepoint, #1587), ``done: <outcome>`` (a collector that closed with the
    argless ``done()`` — the ordinary end), or ``ended: <reason>`` (a collector that
    never closed at all — its structural no-close reason).  Chat vs. collector is read
    off ``run_target`` (collectors stamp it, chat doesn't).

    A clean close renders its OUTCOME, never its stamped reason: the reason on a legacy
    row is a model-authored summary, and #1569's whole point is that a conclusion is
    generated from the ledger rather than claimed.  What a notifying cycle stamps —
    what telling the user came to — reaches the model through the collector's own
    run-history block (``recent_run_outcomes``), which is where a cycle reads what its
    previous invocations did."""
    outcome, reason, target = _run_outcome(prompts)
    if _ended_via_write_gate_stop(prompts):
        return f"stopped: {reason}"
    if target is not None:  # a collector run
        if any(name == PennyConstants.DONE_TOOL_NAME for name, _ in run_tool_calls(prompts)):
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


def _generated_media_ids(prompts: list[PromptLog]) -> list[int]:
    """Ids of the media a run generated and delivered with its reply (#1560).

    A drawn image is delivered deterministically to its own reply (#1564); its
    ``generate_image`` tool result records the stored media id, which lives in the
    run's tool-result turns.  Read it back so the egress trace names an addressable
    id, closing the delivery-introspection gap that let the model confabulate a
    delivery it couldn't inspect.  Reads the COMPLETE turns (#1778), so a run whose
    last act was the draw still names what it attached."""
    found: list[int] = []
    for message in _run_tool_turns(prompts):
        found.extend(
            int(m.group(1)) for m in _GENERATED_MEDIA_RE.finditer(message.get("content") or "")
        )
    return found


def _egress_media_lines(prompts: list[PromptLog]) -> list[str]:
    """The ``attached: image #<id>`` line(s) — what actually went out to the user
    alongside the reply, by addressable id.  Empty when the run attached nothing."""
    return [f"    attached: image #{media_id}" for media_id in _generated_media_ids(prompts)]


def _tool_results_by_id(prompts: list[PromptLog]) -> dict[str, str]:
    """``{tool_call_id: result content}`` from the run's COMPLETE tool turns.

    Keyed by ``tool_call_id`` so each call renders with its own outcome — the
    ``their outcomes`` half of chat-run introspection.  Reads ``_run_tool_turns``
    (accumulated conversation + the trailing tail, #1778), so a run that ended
    right after a tool call still pairs that call with its result instead of
    rendering a call whose outcome is indistinguishable from nothing happening."""
    return {
        message["tool_call_id"]: message.get("content") or ""
        for message in _run_tool_turns(prompts)
        if message.get("tool_call_id")
    }


def _compact_result(content: str | None) -> str:
    """One compact line for a step's result — its first non-blank line, whole — no
    cap (#2642).  Bulk results (page content) stay whole in the ledger; this render
    is by reference."""
    if not content:
        return ""
    return next((part.strip() for part in content.splitlines() if part.strip()), "")


def _step_line(index: int, call: LoggedToolCall, result: str | None) -> str:
    """One step of the canonical projection: ``step <N>: <call> => <result>``.

    ``index`` is the step's coordinate in the run's FULL tool-call sequence — an
    absolute ordinal, so a filtered view shows gaps rather than renumbering (a
    persisted coordinate future skill selection can address as ``(run_id, range)``,
    #1471).  The result renders compact/by-reference."""
    rendered = f"    step {index}: {call.render()}"
    preview = _compact_result(result)
    return f"{rendered} => {preview}" if preview else rendered


# A step's thinking is bounded by default (#2841) — a run can carry many steps, and
# showing every one's reasoning in full would bloat the very surface the model is
# trying to scan.  The excerpt is the CLOSING part, where a decision gets made, not
# the opening (where a model restates what it's about to do) — capped to the last
# ~600 characters, then advanced to the next sentence boundary inside that tail so
# the excerpt opens on a whole sentence rather than a severed clause.
_THINKING_EXCERPT_CHARS = 600
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+")


def _thinking_excerpt(thinking: str | None) -> str | None:
    """The closing ~600 characters of a step's thinking, cut at a sentence
    boundary — ``None`` for a step that made or logged no thinking (renders as
    today, #2841)."""
    text = (thinking or "").strip()
    if not text:
        return None
    if len(text) <= _THINKING_EXCERPT_CHARS:
        return text
    tail = text[-_THINKING_EXCERPT_CHARS:]
    match = _SENTENCE_BOUNDARY_RE.search(tail)
    return tail[match.end() :] if match else tail


def _step_thinking_lines(index: int, thinking: str | None, *, full: bool) -> list[str]:
    """The ``    step <N> thinking: <text>`` line preceding a step's call — the
    excerpted closing part of its thinking by default, or the WHOLE thing
    verbatim when ``full`` (the one step a ``run_get(step=N)`` ask names, #2841).
    No line at all for a step that made no thinking — identical to the
    pre-#2841 render."""
    text = (thinking or "").strip() if full else _thinking_excerpt(thinking)
    return [f"    step {index} thinking: {text}"] if text else []


# The honest refusal when a ``step`` the caller named matches no rendered step of
# THIS run (#2841) — appended to the full record rather than replacing it, so a
# bad step number still returns everything the render would have shown anyway.
_UNMATCHED_STEP_LINE = "{run_line} has no step {step}; its steps are 1–{max_step}"
_NO_STEPS_LINE = "{run_line} has no steps"


def _unmatched_step_line(run_line: str, step: int, rendered_indices: list[int]) -> str:
    """One honest line naming why ``step`` named nothing — the valid range when
    this run rendered any steps, or that it rendered none at all."""
    if not rendered_indices:
        return _NO_STEPS_LINE.format(run_line=run_line)
    return _UNMATCHED_STEP_LINE.format(run_line=run_line, step=step, max_step=max(rendered_indices))


def _draw_outcome_clause(draw: PromptLog) -> str:
    """How ONE child draw ended, for its nested ``model call:`` line (#2355 part D):
    ``completed`` reads as plain ``completed``, every other outcome adds its duration
    and (when raised/timed out) the exception's own class — the structural facts
    ``LlmClient.chat`` stamped on the row, never a narration."""
    outcome = draw.call_outcome or "completed"
    duration = ""
    if draw.duration_ms is not None:
        plain_words = format_duration_plain_words(draw.duration_ms)
        duration = f" after {draw.duration_ms}ms ({plain_words})"
    error = f" ({draw.call_error_class})" if draw.call_error_class else ""
    return f"{outcome}{duration}{error}"


def _draw_line(draw: PromptLog) -> str:
    """One child draw's line — the model call a micro-context made on behalf of this
    run, by its own agent attribution and how it ended.  Indented one step further
    than the step line it nests under (or, for a draw tied to no particular tool call
    — the five micro-context customers that aren't invoked from inside a dispatched
    tool call — appended after every step instead)."""
    return f"        model call: {draw.agent_name or '?'} — {_draw_outcome_clause(draw)}"


def _draws_by_tool_call(
    child_draws: list[PromptLog],
) -> tuple[dict[str, list[PromptLog]], list[PromptLog]]:
    """Child draws split into ``(by their parent_tool_call_id, the rest)`` — the rest
    being drawn on behalf of the run as a whole (the five non-tool-nested
    micro-context customers), which have no step to nest under."""
    by_call_id: dict[str, list[PromptLog]] = {}
    unanchored: list[PromptLog] = []
    for draw in child_draws:
        if draw.parent_tool_call_id:
            by_call_id.setdefault(draw.parent_tool_call_id, []).append(draw)
        else:
            unanchored.append(draw)
    return by_call_id, unanchored


def render_run_calls(
    prompts: list[PromptLog],
    child_draws: list[PromptLog] | None = None,
    *,
    with_thinking: bool,
    step: int | None = None,
    delivered_text: str | None,
    timezone_name: str | None = None,
    schedule_words: str = "",
) -> str:
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
    unambiguously (#1471).  Leads with the run's own id, the egress/media trace
    names what was attached/delivered alongside the reply — so a reader can inspect
    its own turn's delivery (tools + outcomes + what went out) instead of
    confabulating it, every referenced neighbour named, never guessed (#1560) — and
    the facts clause (below) closes it, with the unmatched-step note (also below)
    trailing that when both apply.

    ``with_thinking`` is REQUIRED, no default (#2841): a step's thinking is a
    one-caller feature.  ``run_get`` passes ``True`` — a single named run, where
    seeing WHY each step happened is the point.  ``run_list`` passes ``False`` — a
    batch of runs, where every step of every run showing an excerpt would bloat the
    read; a caller states its own need rather than inheriting a shared default.
    When ``True``, each step whose owning model call logged thinking gets a
    ``step <N> thinking: <excerpt>`` line before its call line — the closing ~600
    characters, cut at a sentence boundary (``_thinking_excerpt``); a step with no
    thinking renders exactly as it did before this existed.  ``step`` (only
    meaningful with ``with_thinking=True``) names the ONE step whose thinking
    should render WHOLE rather than excerpted — the ``run_get(run='run <id>',
    step=N)`` ask — and a ``step`` that names nothing this run rendered gets one
    honest trailing line (``_unmatched_step_line``) AFTER the facts clause, rather
    than a silent no-op: the record still comes back in full, plus why the ask
    didn't change it.

    ``child_draws`` (#2355 part D) are the model calls a micro-context made ON
    BEHALF of this run — a browse-extract draw, say — linked by ``parent_run_id``/
    ``parent_tool_call_id`` rather than sharing this run's own ``run_id`` (a
    micro-context mints its own, for its own ledger attribution).  ``None`` (the
    default, used by every caller that hasn't fetched them) renders exactly as
    before this existed; an empty list and ``None`` are identical.

    ``delivered_text`` (#2843) appends ONE trailing facts line — calls, output
    tokens, time, discards, tool errors by tool, and what was sent — the same
    clause the ``run-health`` facade renders (``run_facts_line``), computed
    from ``run_delivered_text``.  A separate, appended piece on purpose: this
    is a different lens (facts, not the call sequence) added by a different
    change, so it never touches the step/conclusion rendering above it.

    ``timezone_name``/``schedule_words`` (#2862) feed one more leading line —
    why this cycle started when it did (``dispatch_line``) — empty, and so
    rendering nothing, for a chat run or a row logged before that stamp
    existed."""
    if not prompts:
        return "(no data)"
    results = _tool_results_by_id(prompts)
    by_call_id, unanchored = _draws_by_tool_call(child_draws or [])
    steps = _run_logged_steps(prompts)
    run_line = _run_header(prompts)
    lines = [run_line, _run_origin(prompts)]
    dispatch = dispatch_line(prompts, timezone_name, schedule_words)
    if dispatch:
        lines.insert(0, dispatch)
    rendered_indices: list[int] = []
    for index, (call_id, call, thinking) in enumerate(steps, start=1):
        if call.tool == PennyConstants.DONE_TOOL_NAME:
            continue  # consumes the coordinate (gap, never renumber), not shown
        rendered_indices.append(index)
        if with_thinking:
            lines.extend(_step_thinking_lines(index, thinking, full=(step == index)))
        lines.append(_step_line(index, call, results.get(call_id) if call_id else None))
        if call_id:
            lines.extend(_draw_line(draw) for draw in by_call_id.get(call_id, []))
    lines.extend(_draw_line(draw) for draw in unanchored)
    conclusion = _run_conclusion(prompts)
    if conclusion:
        lines.append(conclusion)
    lines.extend(_egress_media_lines(prompts))
    lines.append(run_facts_line(prompts, delivered_text))
    if with_thinking and step is not None and step not in rendered_indices:
        lines.append(_unmatched_step_line(run_line, step, rendered_indices))
    return "\n".join(line for line in lines if line)


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
            col(PromptLog.run_outcome).isnot(None),
            col(PromptLog.run_target).isnot(None),
        ]

    def _all_rows(self) -> list[MemoryEntry]:
        return self._records(newest_first=False)

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        # ``search`` doesn't apply — runs are activity, not searchable entries.
        return self._records(newest_first=True, limit=k, offset=offset)

    def _rows_since(
        self, cursor: datetime, cap: int | None, until: datetime | None = None
    ) -> list[MemoryEntry]:
        return self._records(newest_first=False, cursor=cursor, limit=cap, until=until)

    def _embedded_rows(self) -> list[MemoryEntry]:
        return []

    def _records(
        self,
        *,
        newest_first: bool,
        cursor: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
        until: datetime | None = None,
    ) -> list[MemoryEntry]:
        """Completion rows → rendered run records as ``MemoryEntry`` (content =
        the record, created_at = completion time)."""
        with self._session() as session:
            rows = self._completion_rows(session, newest_first, cursor, limit, offset, until)
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
        until: datetime | None = None,
    ) -> list:
        """The ``(run_id, completion_time, last_prompt_id)`` run-index rows for
        this scope — one per completed run, served by the partial index."""
        query = select(PromptLog.run_id, PromptLog.timestamp, PromptLog.id).where(
            *self._completion_clauses()
        )
        if cursor is not None:
            query = query.where(PromptLog.timestamp > cursor)
        if until is not None:
            query = query.where(PromptLog.timestamp < until)
        order = col(PromptLog.timestamp).desc() if newest_first else col(PromptLog.timestamp).asc()
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
        return _group_prompts_by_run_id(session, run_ids)


def _group_prompts_by_run_id(session: Session, run_ids: list[str]) -> dict[str, list[PromptLog]]:
    """``{run_id: its prompts, ascending}`` — the raw-row grouping both
    ``RunLog`` (collector completions only) and ``RunHealthLog`` (every run,
    #2843) build their rendered records from."""
    rows = session.exec(
        select(PromptLog)
        .where(col(PromptLog.run_id).in_(run_ids))
        .order_by(col(PromptLog.timestamp).asc())
    ).all()
    grouped: dict[str, list[PromptLog]] = {}
    for prompt in rows:
        if prompt.run_id is not None:
            grouped.setdefault(prompt.run_id, []).append(prompt)
    return grouped


class RunHealthLog(Log):
    """Read facade over ``promptlog`` for the ``run-health`` log (#2843).

    Unlike ``collector-runs`` (``RunLog``, scoped to collector completions by
    design — the seeded ``quality`` prompt reads it expecting ONLY collector
    records), this covers EVERY run: an ordinary, successful chat turn stamps
    no completion row at all (``RunLog``'s own filter would miss it, #2355
    part D), so the run index here is simply every distinct ``run_id`` under
    ``promptlog``, newest completion (``MAX(timestamp)``) first — the same
    grouping shape ``MessageStore.run_call_groups`` uses for ``run_list``.
    Each run renders as ONE line (``render_run_health``): its header, then
    the shared facts clause (calls/tokens/time, discards, tool errors by
    tool, what was sent) — never the full call-sequence trace (that's
    ``run_list``/``run_get``'s job, which carries the same facts via
    ``render_run_calls``).  Read-only; has no embeddings (so ``read_similar``
    is empty)."""

    def append(
        self, entries: list[LogEntryInput], author: str, run_id: str | None = None
    ) -> list[MemoryEntry]:
        raise ReadOnlyMemoryError(
            f"'{self.name}' is a read view over run history (promptlog) — "
            "runs are recorded by the framework, not appended here."
        )

    def read_window(self, window_seconds: int, limit: int | None = None) -> list[MemoryEntry]:
        # Bounded per read like the cursored batch, same as ``RunLog``.
        return self.read_recent(window_seconds, limit or PennyConstants.LOG_READ_LIMIT)

    def _all_rows(self) -> list[MemoryEntry]:
        return self._records(newest_first=False)

    def _newest_rows(self, k: int | None, offset: int, search: str | None) -> list[MemoryEntry]:
        # ``search`` doesn't apply — runs are activity, not searchable entries.
        return self._records(newest_first=True, limit=k, offset=offset)

    def _rows_since(
        self, cursor: datetime, cap: int | None, until: datetime | None = None
    ) -> list[MemoryEntry]:
        return self._records(newest_first=False, cursor=cursor, limit=cap, until=until)

    def _embedded_rows(self) -> list[MemoryEntry]:
        return []

    def _records(
        self,
        *,
        newest_first: bool,
        cursor: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
        until: datetime | None = None,
    ) -> list[MemoryEntry]:
        """Every run in scope → its rendered facts line as a ``MemoryEntry``
        (content = the line, created_at = the run's own completion time)."""
        with self._session() as session:
            rows = self._run_index_rows(session, newest_first, cursor, limit, offset, until)
            if not rows:
                return []
            grouped = _group_prompts_by_run_id(session, [run_id for run_id, _, _ in rows])
            return [
                self._to_record(run_id, ts, last_id, grouped, session)
                for run_id, ts, last_id in rows
            ]

    @staticmethod
    def _run_index_rows(
        session: Session,
        newest_first: bool,
        cursor: datetime | None,
        limit: int | None,
        offset: int,
        until: datetime | None,
    ) -> list:
        """The ``(run_id, finished, last_prompt_id)`` index rows — one per
        EVERY run, chat and every collector alike, no completion stamp
        required (mirrors ``MessageStore.run_call_groups``'s own grouping)."""
        finished = func.max(PromptLog.timestamp)
        last_id = func.max(PromptLog.id)
        query = (
            select(PromptLog.run_id, finished.label("finished"), last_id.label("last_id"))
            .where(col(PromptLog.run_id).isnot(None))
            .group_by(PromptLog.run_id)
        )
        if cursor is not None:
            query = query.having(finished > cursor)
        if until is not None:
            query = query.having(finished < until)
        order = finished.desc() if newest_first else finished.asc()
        query = query.order_by(order)
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return [row for row in session.exec(query).all() if row[0] is not None]

    def _to_record(
        self,
        run_id: str,
        timestamp: datetime,
        last_id: int,
        grouped: dict[str, list[PromptLog]],
        session: Session,
    ) -> MemoryEntry:
        """One run → its facts-line ``MemoryEntry``."""
        prompts = grouped.get(run_id) or []
        delivered_text = run_delivered_text(session, run_id)
        return MemoryEntry(
            id=last_id,
            memory_name=self.name,
            key=None,
            content=render_run_health(prompts, delivered_text),
            author=PennyConstants.MessageAuthor.COLLECTOR,
            created_at=timestamp,
        )


__all__ = [
    "Collection",
    "Log",
    "Memory",
    "render_tool_call",
    "render_run_record",
    "render_run_calls",
    "render_run_health",
    "run_facts_line",
    "run_delivered_text",
    "classify_run",
    "RunHealth",
    "LoggedToolCall",
    "MessageLogMemory",
    "RunLog",
    "RunHealthLog",
    "dispatch_line",
    "dispatch_reason_clause",
]
