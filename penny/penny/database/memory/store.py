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

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel
from sqlalchemy import and_, func, or_
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.database.memory import _similarity as sim
from penny.database.memory.objects import Collection, Log, Memory, MessageLogMemory, RunLog
from penny.database.memory.types import (
    DedupThresholds,
    EntrySide,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    ResolvedKind,
    ResolvedMatch,
    slug,
    wrong_shape_message,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog, Skill
from penny.database.mutation_store import MutationDetail, MutationStore

logger = logging.getLogger(__name__)

# The two message logs are read facades over ``messagelog``, keyed by direction;
# ``collector-runs`` is a facade over ``promptlog``.  The factory resolves these
# names to their facade classes; everything else is a stored collection or log.
_MESSAGE_LOG_DIRECTIONS = {
    PennyConstants.MEMORY_USER_MESSAGES_LOG: PennyConstants.MessageDirection.INCOMING,
    PennyConstants.MEMORY_PENNY_MESSAGES_LOG: PennyConstants.MessageDirection.OUTGOING,
}


class _MetadataUpdate(BaseModel):
    """The optional fields of a collection metadata update — only the set ones are
    applied.  ``apply_to`` mutates the row in place and returns the names of the
    fields that actually changed (the mutation event's ``changed_fields``, #1560),
    keeping ``update_collection_metadata`` short."""

    description: str | None = None
    description_embedding: list[float] | None = None
    notify: bool | None = None
    extraction_prompt: str | None = None
    collector_interval_seconds: int | None = None
    intent: str | None = None
    # Trigger union (#1629): the apply-time job axis.  ``replace_trigger`` swaps the
    # WHOLE trigger atomically from these four values (cadence + the once-shaped
    # run_at/max_runs overlay + the on_advance source_log), so switching forms clears
    # the members the new form doesn't use.  ``expires_at`` is the end condition — an
    # independent optional (set when provided; no clear path).  When ``replace_trigger``
    # is false the legacy per-field ``collector_interval_seconds`` poke applies instead
    # (the iOS / browser memory UIs, which edit only the interval).
    run_at: datetime | None = None
    max_runs: int | None = None
    source_log: str | None = None
    expires_at: datetime | None = None
    replace_trigger: bool = False
    # Skill provenance re-stamp (#1620): on a re-render both move together — the
    # instantiating skill and the params bound into its render.  Applied as one unit
    # (``skill_name`` set ⇒ set both), so a re-render always re-stamps the pair the
    # collection's catalog / metadata render reads.
    skill_name: str | None = None
    skill_params: dict[str, str] | None = None

    def apply_to(self, memory: MemoryRow) -> list[str]:
        changed: list[str] = []
        if self.description is not None:
            memory.description = self.description
            memory.description_embedding = sim.maybe_serialize(self.description_embedding)
            changed.append("description")
        if self.notify is not None:
            memory.notify = self.notify
            changed.append("notify")
        if self.extraction_prompt is not None:
            memory.extraction_prompt = self.extraction_prompt
            changed.append("extraction_prompt")
        changed.extend(self._apply_trigger(memory))
        if self.intent is not None:
            memory.intent = self.intent
            changed.append("intent")
        if self.skill_name is not None:
            # A re-render re-homes the collection on a skill: stamp the origin skill
            # and its bound params (JSON, as at creation), so provenance stays a read
            # off the row.  ``skill_params`` is serialized even when empty — a
            # hole-less skill binds nothing but is still a skill instantiation.
            memory.skill_name = self.skill_name
            memory.skill_params = (
                json.dumps(self.skill_params) if self.skill_params is not None else None
            )
            changed.append("skill")
        return changed

    def _apply_trigger(self, memory: MemoryRow) -> list[str]:
        """The trigger axis (#1629).  ``replace_trigger`` swaps the whole trigger
        (cadence + run_at/max_runs + source_log) atomically, so a form switch clears
        the unused members; otherwise the legacy per-field cadence poke applies.
        ``expires_at`` is set when provided.  Editing the interval declares a new
        intended cadence, so ``base_interval_seconds`` moves with it and any throttle
        backoff clears."""
        changed: list[str] = []
        if self.replace_trigger:
            memory.collector_interval_seconds = self.collector_interval_seconds
            memory.base_interval_seconds = self.collector_interval_seconds
            memory.consecutive_idle_runs = 0
            memory.run_at = self.run_at
            memory.max_runs = self.max_runs
            memory.source_log = self.source_log
            changed.append("trigger")
        elif self.collector_interval_seconds is not None:
            memory.collector_interval_seconds = self.collector_interval_seconds
            memory.base_interval_seconds = self.collector_interval_seconds
            memory.consecutive_idle_runs = 0
            changed.append("collector_interval_seconds")
        if self.expires_at is not None:
            memory.expires_at = self.expires_at
            changed.append("expires_at")
        return changed


class MemoryStore:
    """Registry + factory for memories.

    Summary of the public surface:
        * dispatch: memory, run_log
        * metadata: create_collection, create_log, get, list_all, archive,
          unarchive, update_collection_metadata, link_source_message,
          mark_collected, set_cadence
        * inventory: entry_counts, names_with_entry_match
        * resolve by meaning: resolve_objects
        * dedup probe: exists
        * embedding backfill: get_entries_without_embeddings,
          get_memories_without_description_embedding, set_description_embedding,
          set_entry_embeddings
    """

    def __init__(
        self,
        engine,
        runtime: RuntimeParams | None = None,
        mutations: MutationStore | None = None,
    ):
        self.engine = engine
        # /config-tunable dedup thresholds; tests get vanilla defaults.
        self._runtime = runtime if runtime is not None else RuntimeParams()
        # The registry-mutation ledger (#1560).  Create/update/archive/unarchive
        # of a collection each write a durable event here, at this Python
        # chokepoint — so no caller (chat tool, scheduler) can mutate a mechanism
        # without the provenance being recorded.  Defaults to one over the shared
        # engine when the facade doesn't inject it (isolated tests), so the
        # recording is never silently skipped.
        self._mutations = mutations if mutations is not None else MutationStore(engine)
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
        archived: bool = False,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        notify: bool = False,
        created_by_run_id: str | None = None,
        expires_at: datetime | None = None,
        run_at: datetime | None = None,
        max_runs: int | None = None,
        skill_name: str | None = None,
        skill_params: dict[str, str] | None = None,
        source_log: str | None = None,
    ) -> MemoryRow:
        return self._create_memory(
            name,
            MemoryType.COLLECTION,
            description,
            archived,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=collector_interval_seconds,
            description_embedding=description_embedding,
            intent=intent,
            notify=notify,
            created_by_run_id=created_by_run_id,
            expires_at=expires_at,
            run_at=run_at,
            max_runs=max_runs,
            skill_name=skill_name,
            skill_params=skill_params,
            source_log=source_log,
        )

    def create_log(
        self,
        name: str,
        description: str,
        archived: bool = False,
        description_embedding: list[float] | None = None,
    ) -> MemoryRow:
        # Logs are inputs, not curated outputs — no extraction_prompt by design.
        return self._create_memory(
            name,
            MemoryType.LOG,
            description,
            archived,
            description_embedding=description_embedding,
        )

    def _create_memory(
        self,
        name: str,
        type_: MemoryType,
        description: str,
        archived: bool,
        *,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        notify: bool = False,
        created_by_run_id: str | None = None,
        expires_at: datetime | None = None,
        run_at: datetime | None = None,
        max_runs: int | None = None,
        skill_name: str | None = None,
        skill_params: dict[str, str] | None = None,
        source_log: str | None = None,
    ) -> MemoryRow:
        name = slug(name)
        if self.get(name) is not None:
            raise MemoryAlreadyExistsError(name)
        with self._session() as session:
            memory = MemoryRow(
                name=name,
                type=type_.value,
                description=description,
                description_embedding=sim.maybe_serialize(description_embedding),
                archived=archived,
                notify=notify,
                extraction_prompt=extraction_prompt,
                collector_interval_seconds=collector_interval_seconds,
                # The create cadence is the user's intended cadence — the
                # snap-back target for auto-throttle.
                base_interval_seconds=collector_interval_seconds,
                intent=intent,
                # Provenance + lifecycle (#1566): the creating run and the end
                # condition.  Both None for seeded / system rows; the spawning
                # message is linked post-run via ``link_source_message``.
                created_by_run_id=created_by_run_id,
                expires_at=expires_at,
                # Once-shaped trigger (#1556): delayed/one-shot start + run quota.
                run_at=run_at,
                max_runs=max_runs,
                # Skill provenance (#1603): the instantiating skill + the params
                # bound into its render, serialized here at the store boundary (the
                # same place the description embedding is serialized).  None for a
                # hand-authored / seeded row — no skill origin.
                skill_name=skill_name,
                skill_params=json.dumps(skill_params) if skill_params is not None else None,
                # On_advance trigger (#1604): the declared source log whose advance
                # wakes this collection.  NULL for the interval / once forms.
                source_log=source_log,
                created_at=datetime.now(UTC),
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.debug("Created %s memory %s", type_.value, name)
        # A collection is a registry mechanism — its creation is a mutation event
        # (#1560).  Logs aren't mechanisms (the reference map's mutation stream is
        # collections/skills), so they don't get one; nor do seeded/migration
        # creates, which bypass this store entirely (correct — they carry no
        # creating run).  The actor is always a chat run here (collectors can't
        # create, #1556), so ``created_by_run_id`` is the cause.
        if type_ == MemoryType.COLLECTION:
            self._mutations.record(
                entity_type=MutationEntityType.COLLECTION,
                entity_name=name,
                action=MutationAction.CREATED,
                actor=MutationActor.USER_RUN,
                run_id=created_by_run_id,
            )
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

    def archive(
        self,
        name: str,
        *,
        actor: MutationActor = MutationActor.USER_RUN,
        run_id: str | None = None,
        note: str | None = None,
    ) -> None:
        """Archive a collection and record the mutation (#1560).

        ``actor`` / ``run_id`` / ``note`` carry the provenance: a chat archive is
        ``USER_RUN`` + its run id; the scheduler's ``max_runs`` / ``expires_at``
        retire is ``SYSTEM`` + the collector run + a policy ``note`` (its cause,
        which no run prompt otherwise records)."""
        self._set_archived(name, True, MutationAction.ARCHIVED, actor, run_id, note)

    def unarchive(
        self,
        name: str,
        *,
        actor: MutationActor = MutationActor.USER_RUN,
        run_id: str | None = None,
        note: str | None = None,
    ) -> None:
        self._set_archived(name, False, MutationAction.UNARCHIVED, actor, run_id, note)

    def _set_archived(
        self,
        name: str,
        archived: bool,
        action: MutationAction,
        actor: MutationActor,
        run_id: str | None,
        note: str | None,
    ) -> None:
        name = slug(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.archived = archived
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
        self._mutations.record(
            entity_type=MutationEntityType.COLLECTION,
            entity_name=name,
            action=action,
            actor=actor,
            run_id=run_id,
            detail=MutationDetail(note=note) if note else None,
        )
        self._notify_changed(name)

    def update_collection_metadata(
        self,
        name: str,
        *,
        description: str | None = None,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
        description_embedding: list[float] | None = None,
        intent: str | None = None,
        notify: bool | None = None,
        skill_name: str | None = None,
        skill_params: dict[str, str] | None = None,
        run_at: datetime | None = None,
        max_runs: int | None = None,
        source_log: str | None = None,
        expires_at: datetime | None = None,
        replace_trigger: bool = False,
        run_id: str | None = None,
    ) -> MemoryRow:
        """Update fields on an existing collection.  Only set fields are applied.

        When ``description`` changes the caller passes the freshly computed
        ``description_embedding`` alongside it so the resolve-by-meaning anchor
        stays in sync — the anchor moves *with* the text: a changed description
        always replaces the embedding with the passed value, and if that value is
        ``None`` (a transient embed failure at the caller) the anchor is cleared
        to ``NULL`` rather than left pointing at the old, now-mismatched text.  A
        ``NULL`` anchor is what the startup description backfill re-heals; a stale
        non-``NULL`` one it could never detect.  ``intent`` is editable here (the
        user-authored update path) even though it is NOT a field on the
        ``collection_update`` tool: the user owns the spec, the agent cannot
        rewrite it.  ``skill_name`` / ``skill_params`` re-stamp the collection's
        skill provenance on a re-render (#1620) — the caller renders the new
        ``extraction_prompt`` from the skill's current steps and passes it alongside
        the pair, so the recorded origin always matches the rendered snapshot.
        """
        name = slug(name)
        self._require_collection(name)
        fields = _MetadataUpdate(
            description=description,
            description_embedding=description_embedding,
            notify=notify,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=collector_interval_seconds,
            intent=intent,
            run_at=run_at,
            max_runs=max_runs,
            source_log=source_log,
            expires_at=expires_at,
            replace_trigger=replace_trigger,
            skill_name=skill_name,
            skill_params=skill_params,
        )
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            changed = fields.apply_to(memory)
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
            session.refresh(memory)
        # Record the config change as a durable event (#1560).  Only when a field
        # actually moved — a no-op update (every field None) isn't a mutation.
        # The new values live verbatim in the run's promptlog tool call; the event
        # names which fields changed so the history reads without re-fetching them.
        if changed:
            self._mutations.record(
                entity_type=MutationEntityType.COLLECTION,
                entity_name=name,
                action=MutationAction.UPDATED,
                actor=MutationActor.USER_RUN,
                run_id=run_id,
                detail=MutationDetail(changed_fields=changed),
            )
        self._notify_changed(name)
        return memory

    def link_source_message(self, run_id: str, source_message_id: int) -> None:
        """Stamp the spawning message on every mechanism a chat run created (#1566).

        ``collection_create`` records ``created_by_run_id`` at creation, but the
        triggering message's id isn't known until the run returns (the channel
        logs it afterward).  The
        channel then calls this to link the two structurally — matching by the
        unique per-turn ``run_id``, and only where a source isn't already set —
        so the provenance is a read, not a reconstruction.  A no-op when the run
        created nothing.
        """
        with self._session() as session:
            rows = session.exec(
                select(MemoryRow).where(
                    MemoryRow.created_by_run_id == run_id,
                    MemoryRow.source_message_id.is_(None),  # ty: ignore[unresolved-attribute]
                )
            ).all()
            changed_names = [memory.name for memory in rows]
            for memory in rows:
                memory.source_message_id = source_message_id
                session.add(memory)
            session.commit()
        for name in changed_names:
            self._notify_changed(name)

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
        """Entries missing either similarity vector that are worth embedding.

        An entry qualifies when its content vector is unset OR — being keyed — its
        key vector is unset: a keyed entry whose content embedded but whose key
        did not (a transient key-embed miss, or migration-seeded content that
        carried a content vector but no key vector) would otherwise slip the
        backfill forever, since selecting on ``content_embedding IS NULL`` alone
        never reaches it.  A keyless log entry has a legitimately-null key vector,
        so it only qualifies on a missing content vector.

        Scoped to non-archived memories — an archived memory never surfaces via
        ``read_similar`` or resolve-by-meaning, so embedding its entries is pure
        waste.  Newest first, so the most relevant rows embed first when the
        backfill batches.
        """
        missing_vector = or_(
            MemoryEntry.content_embedding.is_(None),  # ty: ignore[unresolved-attribute]
            and_(
                MemoryEntry.key.is_not(None),  # ty: ignore[unresolved-attribute]
                MemoryEntry.key_embedding.is_(None),  # ty: ignore[unresolved-attribute]
            ),
        )
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .join(MemoryRow)  # FK memory_entry.memory_name → memory.name
                    .where(
                        missing_vector,
                        MemoryRow.archived == False,  # noqa: E712
                    )
                    .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
                    .limit(limit)
                ).all()
            )

    def get_memories_without_description_embedding(self, limit: int) -> list[MemoryRow]:
        """Active memories whose description anchor (resolve-by-meaning, #1558) is
        unset.  Scoped to non-archived memories — an archived memory's anchor is
        never consulted."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryRow)
                    .where(
                        MemoryRow.description_embedding == None,  # noqa: E711
                        MemoryRow.archived == False,  # noqa: E712
                    )
                    .limit(limit)
                ).all()
            )

    def set_description_embedding(self, name: str, embedding: list[float]) -> None:
        """Persist the description anchor on a memory (backfill path)."""
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

    # ── Resolve by meaning ────────────────────────────────────────────────────

    def resolve_objects(
        self, anchor: list[float], kind: ResolvedKind | None, limit: int
    ) -> list[ResolvedMatch]:
        """Rank Penny's own addressable objects by meaning, best-first (#1558).

        Plain-cosine nearest-neighbour search (the explicit-search path, #1565 —
        no ambient centrality/cluster gating) of ``anchor`` against every registry
        row's description anchor (collections + logs, **archived included**) and
        every taught skill's description anchor (the ``skill`` table, the sole
        skills store — #1624).  ``kind`` narrows to one family;
        ``None`` spans all three.  Only positively-correlated candidates survive
        (cosine > 0 — an orthogonal/anti-correlated object isn't a match, so a
        wholly-unrelated query returns an honest empty), capped at ``limit``.  The
        tool layer turns each match into identity + state + the deterministic
        addressing; ambiguity (several hits) is returned, never silently resolved.
        """
        candidates = self._resolution_candidates(kind)
        if not candidates:
            return []
        scores = sim.cosine_scores([blob for _, blob in candidates], anchor)
        order = list(np.argsort(-scores))
        ranked = [candidates[i][0] for i in order if float(scores[i]) > 0.0]
        return ranked[:limit]

    def _resolution_candidates(
        self, kind: ResolvedKind | None
    ) -> list[tuple[ResolvedMatch, bytes]]:
        """Every (match, embedding-blob) pair eligible for ``resolve_objects`` —
        registry rows carrying a description anchor and, when ``kind`` allows,
        taught skills carrying a description anchor.  A row/skill without its
        vector is silently absent (the backfill fills it) — never surfaced
        unscored."""
        candidates: list[tuple[ResolvedMatch, bytes]] = []
        if kind is not ResolvedKind.SKILL:
            for row in self.list_all():
                if row.description_embedding is None:
                    continue
                row_kind = (
                    ResolvedKind.COLLECTION
                    if row.type == MemoryType.COLLECTION
                    else ResolvedKind.LOG
                )
                if kind is not None and row_kind is not kind:
                    continue
                match = ResolvedMatch(
                    name=row.name, kind=row_kind, archived=row.archived, label=row.description
                )
                candidates.append((match, row.description_embedding))
        if kind in (None, ResolvedKind.SKILL):
            candidates.extend(self._skill_candidates())
        return candidates

    def _skill_candidates(self) -> list[tuple[ResolvedMatch, bytes]]:
        """Taught skills as resolvable objects — the ``skill`` table (the sole
        skills store, #1624), each anchored by its description vector (populated
        at teach time).  Only skills carrying the vector are eligible."""
        with self._session() as session:
            rows = session.exec(
                select(Skill).where(
                    Skill.description_embedding.is_not(None),  # ty: ignore[unresolved-attribute]
                )
            ).all()
        return [
            (
                ResolvedMatch(
                    name=row.name,
                    kind=ResolvedKind.SKILL,
                    archived=False,
                    label=row.description,
                ),
                row.description_embedding,
            )
            for row in rows
            if row.description_embedding is not None
        ]

    # ── Idempotency at birth (#1567) ──────────────────────────────────────────

    def find_duplicate_collection(
        self,
        name: str,
        description_embedding: list[float] | None,
        thresholds: DedupThresholds | None = None,
    ) -> MemoryRow | None:
        """The first EXISTING collection a proposed one is a semantic near-duplicate
        of (#1567), or ``None`` when the target is genuinely distinct.

        Compared name-vs-name (token containment) and description-vs-description
        (content cosine — the intent/purpose anchor, since a skill-instantiated
        collection's ``description`` is its intent) through the SAME three-signal
        dedup rule the entry write uses (``sim.is_duplicate``) with the SAME runtime
        thresholds — never a hand-rolled similarity rule.  Active rows are checked
        before archived ones (a tombstone), so a live "already watching this" wins
        over a retired one.  Framework collections (``SYSTEM_COLLECTIONS``) are
        excluded — Penny's own machinery, not a mechanism a user re-creates — and a
        merely-related topic clears the thresholds, so distinct targets are
        unimpeded.
        """
        thresholds = thresholds or self._default_thresholds()
        candidate = EntrySide(slug(name), None, description_embedding)
        active, archived = self._duplicate_candidates()
        for row in (*active, *archived):
            if sim.is_duplicate(candidate, [self._collection_side(row)], thresholds):
                return row
        return None

    def _duplicate_candidates(self) -> tuple[list[MemoryRow], list[MemoryRow]]:
        """User collections eligible for the idempotency check, partitioned active
        vs. archived — framework system collections and logs excluded."""
        active: list[MemoryRow] = []
        archived: list[MemoryRow] = []
        for row in self.list_all():
            if row.type != MemoryType.COLLECTION.value:
                continue
            if row.name in PennyConstants.SYSTEM_COLLECTIONS:
                continue
            (archived if row.archived else active).append(row)
        return active, archived

    @staticmethod
    def _collection_side(row: MemoryRow) -> EntrySide:
        """One existing collection as a dedup side: its name (token-containment
        signal) and its description anchor (content-cosine signal)."""
        return EntrySide(row.name, None, sim.maybe_deserialize(row.description_embedding))

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

        Every name must resolve to a real memory: an unknown name raises
        ``MemoryNotFoundError`` naming the offending value, rather than reading
        as an empty (and therefore always-``False``) memory — a misspelled probe
        must not misreport ``no`` and green-light the write it was checking for.
        """
        names = [slug(n) for n in names]
        for name in names:
            if self.get(name) is None:
                raise MemoryNotFoundError(name)
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
            raise MemoryTypeError(wrong_shape_message(name, memory.type))
