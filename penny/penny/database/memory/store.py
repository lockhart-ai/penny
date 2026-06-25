"""``MemoryStore`` — the memory registry, metadata CRUD, and the dispatch factory.

The store owns everything that is *about the set of memories* rather than the
contents of one: create/list/archive, metadata edits, the cross-memory dedup
probe (``exists``), the startup embedding backfill, and the bulk entry counts
for the addon list.

It is also the single dispatch the rest of the system asks for a memory:
``memory(name)`` returns the right ``Memory`` object (``Collection`` / ``Log`` /
``MessageLogMemory`` / ``RunLog``) so callers operate polymorphically and never
branch on a name or shape themselves.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import func
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database.memory import _similarity as sim
from penny.database.memory.objects import Collection, Log, Memory, MessageLogMemory, RunLog
from penny.database.memory.types import (
    DedupThresholds,
    EntrySide,
    Inclusion,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    RecallMode,
    slug,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog

logger = logging.getLogger(__name__)

# The two message logs are read facades over ``messagelog``, keyed by direction;
# ``collector-runs`` is a facade over ``promptlog``.  The factory resolves these
# names to their facade classes; everything else is a stored collection or log.
_MESSAGE_LOG_DIRECTIONS = {
    PennyConstants.MEMORY_USER_MESSAGES_LOG: PennyConstants.MessageDirection.INCOMING,
    PennyConstants.MEMORY_PENNY_MESSAGES_LOG: PennyConstants.MessageDirection.OUTGOING,
}


class MemoryStore:
    """Registry + factory for memories.

    Summary of the public surface:
        * dispatch: memory, active_memories, run_log
        * metadata: create_collection, create_log, get, list_all, archive,
          unarchive, update_collection_metadata, mark_collected, set_cadence
        * inventory: entry_counts, names_with_entry_match
        * dedup probe: exists
        * embedding backfill: get_entries_without_embeddings,
          get_memories_without_description_embedding, set_description_embedding,
          set_entry_embeddings
    """

    def __init__(self, engine, runtime: RuntimeParams | None = None):
        self.engine = engine
        # /config-tunable dedup thresholds; tests get vanilla defaults.
        self._runtime = runtime if runtime is not None else RuntimeParams()
        # Fired after any mutation so observers (the browser channel) can refresh.
        # The factory injects it into each Memory object it builds.
        self._on_memory_changed: Callable[[str | None], None] | None = None

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def memory(self, name: str) -> Memory | None:
        """Return the ``Memory`` object for ``name`` — the single dispatch.

        ``None`` when the memory doesn't exist (callers surface 'not found').
        """
        row = self.get(name)
        return self._build(row) if row is not None else None

    def active_memories(self) -> list[Memory]:
        """Memory objects for every non-archived, routable memory (recall)."""
        return [
            self._build(row)
            for row in self.list_all()
            if not row.archived and row.inclusion != Inclusion.NEVER
        ]

    def run_log(self) -> RunLog | None:
        """The ``collector-runs`` facade over every collector run.  ``None`` if
        the marker row is somehow absent.  (Per-collection run views go through
        ``messages.get_target_runs`` — full runs for the addon's Activity tab.)"""
        row = self.get(PennyConstants.MEMORY_COLLECTOR_RUNS_LOG)
        if row is None:
            return None
        return RunLog(row, self.engine, on_changed=self._on_memory_changed)

    def _build(self, row: MemoryRow) -> Memory:
        """Construct the right ``Memory`` subclass for an already-loaded row."""
        if row.name in _MESSAGE_LOG_DIRECTIONS:
            return MessageLogMemory(
                row,
                self.engine,
                direction=_MESSAGE_LOG_DIRECTIONS[row.name],
                on_changed=self._on_memory_changed,
            )
        if row.name == PennyConstants.MEMORY_COLLECTOR_RUNS_LOG:
            return RunLog(row, self.engine, on_changed=self._on_memory_changed)
        if row.type == MemoryType.COLLECTION:
            return Collection(
                row, self.engine, runtime=self._runtime, on_changed=self._on_memory_changed
            )
        return Log(row, self.engine, on_changed=self._on_memory_changed)

    def _default_thresholds(self) -> DedupThresholds:
        return DedupThresholds.from_runtime(self._runtime)

    def _session(self) -> Session:
        return Session(self.engine)

    # ── Metadata ────────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        description: str,
        inclusion: Inclusion,
        recall: RecallMode,
        archived: bool = False,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        published: bool = False,
    ) -> MemoryRow:
        return self._create_memory(
            name,
            MemoryType.COLLECTION,
            description,
            inclusion,
            recall,
            archived,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=collector_interval_seconds,
            description_embedding=description_embedding,
            intent=intent,
            published=published,
        )

    def create_log(
        self,
        name: str,
        description: str,
        inclusion: Inclusion,
        recall: RecallMode,
        archived: bool = False,
        description_embedding: list[float] | None = None,
    ) -> MemoryRow:
        # Logs are inputs, not curated outputs — no extraction_prompt by design.
        return self._create_memory(
            name,
            MemoryType.LOG,
            description,
            inclusion,
            recall,
            archived,
            description_embedding=description_embedding,
        )

    def _create_memory(
        self,
        name: str,
        type_: MemoryType,
        description: str,
        inclusion: Inclusion,
        recall: RecallMode,
        archived: bool,
        *,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        published: bool = False,
    ) -> MemoryRow:
        name = slug(name)
        if self.get(name) is not None:
            raise MemoryAlreadyExistsError(name)
        with self._session() as session:
            memory = MemoryRow(
                name=name,
                type=type_.value,
                description=description,
                inclusion=inclusion.value,
                recall=recall.value,
                description_embedding=sim.maybe_serialize(description_embedding),
                archived=archived,
                published=published,
                extraction_prompt=extraction_prompt,
                collector_interval_seconds=collector_interval_seconds,
                # The create cadence is the user's intended cadence — the
                # snap-back target for auto-throttle.
                base_interval_seconds=collector_interval_seconds,
                intent=intent,
                created_at=datetime.now(UTC),
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.debug("Created %s memory %s", type_.value, name)
        self._notify_changed(name)
        return memory

    def get(self, name: str) -> MemoryRow | None:
        with self._session() as session:
            return session.get(MemoryRow, slug(name))

    def list_all(self) -> list[MemoryRow]:
        with self._session() as session:
            return list(session.exec(select(MemoryRow).order_by(MemoryRow.name)).all())

    def entry_counts(self) -> dict[str, int]:
        """Return ``{memory_name: entry_count}`` for every memory in one pass.

        Powers the addon's Memories tab counts without N+1 round-trips.  The
        message logs and collector-runs are facades, so they're counted from
        their canonical tables (``messagelog`` / ``promptlog``), matching what
        a read of them returns.  Memories with zero entries are absent — callers
        default to 0.
        """
        with self._session() as session:
            rows = session.exec(
                select(MemoryEntry.memory_name, func.count(MemoryEntry.id)).group_by(  # ty: ignore[invalid-argument-type]
                    MemoryEntry.memory_name
                )  # ty: ignore[invalid-argument-type]
            ).all()
            counts = dict(rows)
            by_direction = dict(
                session.exec(
                    select(MessageLog.direction, func.count(MessageLog.id))  # ty: ignore[invalid-argument-type]
                    .where(MessageLog.is_reaction.is_(False))  # ty: ignore[unresolved-attribute]
                    .group_by(MessageLog.direction)
                ).all()
            )
            run_count = session.exec(
                select(func.count(PromptLog.id)).where(  # ty: ignore[invalid-argument-type]
                    PromptLog.run_outcome.isnot(None),  # ty: ignore[unresolved-attribute]
                    PromptLog.run_target.isnot(None),  # ty: ignore[unresolved-attribute]
                )
            ).one()
        for log_name, direction in _MESSAGE_LOG_DIRECTIONS.items():
            counts[log_name] = by_direction.get(direction, 0)
        counts[PennyConstants.MEMORY_COLLECTOR_RUNS_LOG] = run_count or 0
        return counts

    def names_with_entry_match(self, search: str) -> set[str]:
        """Names of memories holding an entry whose ``key`` or ``content``
        contains ``search`` — powers the addon's "search entries too" filter
        (a substring LIKE; on-demand, so not indexed)."""
        like = f"%{search}%"
        with self._session() as session:
            rows = session.exec(
                select(MemoryEntry.memory_name)
                .where(
                    MemoryEntry.content.like(like)  # ty: ignore[unresolved-attribute]
                    | MemoryEntry.key.like(like)  # ty: ignore[unresolved-attribute]
                )
                .distinct()
            ).all()
            return set(rows)

    def _notify_changed(self, name: str | None) -> None:
        if self._on_memory_changed is not None:
            self._on_memory_changed(name)

    def archive(self, name: str) -> None:
        self._set_archived(name, True)

    def unarchive(self, name: str) -> None:
        self._set_archived(name, False)

    def _set_archived(self, name: str, archived: bool) -> None:
        name = slug(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.archived = archived
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
        self._notify_changed(name)

    def update_collection_metadata(
        self,
        name: str,
        *,
        description: str | None = None,
        inclusion: Inclusion | None = None,
        recall: RecallMode | None = None,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        published: bool | None = None,
    ) -> MemoryRow:
        """Update fields on an existing collection.  Only set fields are applied.

        When ``description`` changes the caller passes the freshly computed
        ``description_embedding`` alongside it so the stage-1 anchor stays in
        sync.  ``intent`` is editable here (the user-authored update path) even
        though it is NOT a field on the ``collection_update`` tool: the user owns
        the spec, the agent cannot rewrite it.
        """
        name = slug(name)
        self._require_collection(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            if description is not None:
                memory.description = description
            if description_embedding is not None:
                memory.description_embedding = sim.maybe_serialize(description_embedding)
            if inclusion is not None:
                memory.inclusion = inclusion.value
            if recall is not None:
                memory.recall = recall.value
            if published is not None:
                memory.published = published
            if extraction_prompt is not None:
                memory.extraction_prompt = extraction_prompt
            if collector_interval_seconds is not None:
                # Editing the interval declares a new intended cadence: current
                # and snap-back base both move, and any throttle backoff clears.
                memory.collector_interval_seconds = collector_interval_seconds
                memory.base_interval_seconds = collector_interval_seconds
                memory.consecutive_idle_runs = 0
            if intent is not None:
                memory.intent = intent
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
            session.refresh(memory)
        self._notify_changed(name)
        return memory

    def mark_collected(self, name: str) -> None:
        """Stamp ``last_collected_at = now`` after a dispatcher cycle (whether it
        did work or exited via ``done()`` — what matters is the check happened)."""
        name = slug(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.last_collected_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
        self._notify_changed(name)

    def set_cadence(self, name: str, interval_seconds: int, consecutive_idle_runs: int) -> None:
        """Persist a collection's (possibly auto-throttled) current interval and
        idle-run counter.  No-op if the memory is gone."""
        with self._session() as session:
            memory = session.get(MemoryRow, slug(name))
            if memory is None:
                return
            memory.collector_interval_seconds = interval_seconds
            memory.consecutive_idle_runs = consecutive_idle_runs
            session.add(memory)
            session.commit()

    # ── Embedding backfill ──────────────────────────────────────────────────

    def get_entries_without_embeddings(self, limit: int) -> list[MemoryEntry]:
        """Entries missing a content embedding that are worth embedding.

        Scoped to non-archived memories whose ``inclusion`` is not ``never`` —
        an ``inclusion=never`` memory never surfaces via recall or
        ``read_similar``, so embedding its entries is pure waste.  Newest first,
        so the most recall-relevant rows embed first when the backfill batches.
        """
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .join(MemoryRow)  # FK memory_entry.memory_name → memory.name
                    .where(
                        MemoryEntry.content_embedding == None,  # noqa: E711
                        MemoryRow.archived == False,  # noqa: E712
                        MemoryRow.inclusion != Inclusion.NEVER.value,
                    )
                    .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
                    .limit(limit)
                ).all()
            )

    def get_memories_without_description_embedding(self, limit: int) -> list[MemoryRow]:
        """Active, routable memories whose stage-1 description anchor is unset.

        Scoped to non-archived memories with ``inclusion != never`` — a
        ``never`` memory is never routed, so its anchor is never consulted.
        """
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryRow)
                    .where(
                        MemoryRow.description_embedding == None,  # noqa: E711
                        MemoryRow.archived == False,  # noqa: E712
                        MemoryRow.inclusion != Inclusion.NEVER.value,
                    )
                    .limit(limit)
                ).all()
            )

    def set_description_embedding(self, name: str, embedding: list[float]) -> None:
        """Persist the stage-1 description anchor on a memory (backfill path)."""
        name = slug(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                return
            memory.description_embedding = sim.maybe_serialize(embedding)
            session.add(memory)
            session.commit()

    def set_entry_embeddings(
        self,
        entry_id: int,
        *,
        key_embedding: list[float] | None,
        content_embedding: list[float] | None,
    ) -> None:
        """Persist computed embeddings on an existing entry (backfill path)."""
        with self._session() as session:
            entry = session.get(MemoryEntry, entry_id)
            if entry is None:
                return
            if key_embedding is not None:
                entry.key_embedding = sim.maybe_serialize(key_embedding)
            if content_embedding is not None:
                entry.content_embedding = sim.maybe_serialize(content_embedding)
            session.add(entry)
            session.commit()

    # ── Dedup probe ───────────────────────────────────────────────────────────

    def exists(
        self,
        names: list[str],
        key: str | None,
        key_embedding: list[float] | None,
        content_embedding: list[float] | None,
        thresholds: DedupThresholds | None = None,
    ) -> bool:
        """Whether an equivalent entry already exists in any of the named memories.

        Runs the same similarity-based dedup as ``Collection.write``, plus an
        exact key-match shortcut when a key is supplied.  True on the first hit.
        """
        names = [slug(n) for n in names]
        thresholds = thresholds or self._default_thresholds()
        candidate = EntrySide(key, key_embedding, content_embedding)
        for name in names:
            if key is not None and self._rows_by_key(name, key):
                return True
            if sim.is_duplicate(candidate, self._entries_with_vectors(name), thresholds):
                return True
        return False

    def _rows_by_key(self, name: str, key: str) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == name, MemoryEntry.key == key
                    )
                ).all()
            )

    def _entries_with_vectors(self, name: str) -> list[EntrySide]:
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == name)).all()
            )
        return [
            EntrySide(
                row.key,
                sim.maybe_deserialize(row.key_embedding),
                sim.maybe_deserialize(row.content_embedding),
            )
            for row in rows
        ]

    def _require_collection(self, name: str) -> None:
        memory = self.get(name)
        if memory is None:
            raise MemoryNotFoundError(name)
        if memory.type != MemoryType.COLLECTION:
            raise MemoryTypeError(f"memory '{name}' is a {memory.type}, not a collection")
