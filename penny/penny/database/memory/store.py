"""``MemoryStore`` — the memory registry, metadata CRUD, and the dispatch factory.

The store owns everything that is *about the set of memories* rather than the
contents of one: create/list/archive, metadata edits, the cross-memory dedup
probe (``exists``, retired with the three-signal rule — the identity ladder owns
entry identity now), the startup embedding backfill, and the bulk entry counts
for the addon list.

It is also the single dispatch the rest of the system asks for a memory:
``memory(name)`` returns the right ``Memory`` object (``Collection`` / ``Log`` /
``MessageLogMemory`` / ``RunLog``) so callers operate polymorphically and never
branch on a name or shape themselves.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable, Hashable, Sequence
from datetime import datetime
from typing import Any, cast

import numpy as np
from pydantic import BaseModel
from similarity.dedup import JobSide, is_same_job
from sqlalchemy import ColumnElement, and_, func, or_, text
from sqlmodel import Session, col, select

from penny.clock import Clock
from penny.config_params import RuntimeParams
from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.database.memory import _similarity as sim
from penny.database.memory.journal import CycleJournal
from penny.database.memory.objects import Collection, Log, Memory, MessageLogMemory, RunLog
from penny.database.memory.types import (
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    ResolvedEntry,
    ResolvedHit,
    ResolvedKind,
    ResolvedMatch,
    ResolvedMessage,
    RunWrite,
    slug,
    wrong_shape_message,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, PromptLog
from penny.database.mutation_store import (
    FieldPrior,
    MutationDetail,
    MutationStore,
    cancelled_sends_note,
)
from penny.database.send_queue_store import SendQueueStore
from penny.datetime_utils import format_log_timestamp

logger = logging.getLogger(__name__)

# The two message logs are read facades over ``messagelog``, keyed by direction;
# ``collector-runs`` is a facade over ``promptlog``.  The factory resolves these
# names to their facade classes; everything else is a stored collection or log.
_MESSAGE_LOG_DIRECTIONS = {
    PennyConstants.MEMORY_USER_MESSAGES_LOG: PennyConstants.MessageDirection.INCOMING,
    PennyConstants.MEMORY_PENNY_MESSAGES_LOG: PennyConstants.MessageDirection.OUTGOING,
}
# The reverse of the map above — a keyword-matched ``messagelog`` row carries its
# own ``direction``, and the hybrid find's keyword leg (#2774) needs the facade
# name that direction reads through, the same name the embedding leg's
# ``_message_candidates`` already reports.
_MESSAGE_LOG_BY_DIRECTION = {direction: name for name, direction in _MESSAGE_LOG_DIRECTIONS.items()}


def _fts_match_query(text_: str) -> str | None:
    """``text_`` as an FTS5 ``MATCH`` query for find's keyword leg (#2774): every
    word token, quoted and prefix-matched, OR-joined — ``"cats, Mochi, or Biscuit"``
    becomes ``"cats"* OR "mochi"* OR "or"* OR "biscuit"*`` (a stray `OR`/`OR`-shaped
    word costs nothing; it just adds one more term nothing need match).

    ``re.findall(r"\\w+", ...)`` keeps only word characters, so no FTS5 syntax
    character (``"``, ``:``, ``-``, parentheses) ever reaches the query string;
    QUOTING each token additionally defeats a bare word that happens to be an
    FTS5 keyword (``AND``/``OR``/``NOT``/``NEAR``) — a quoted string is always a
    literal term, never an operator.  OR (not implicit AND) because the issue's
    own evidence is a query naming several names at once and ANY of them
    matching should count (#2774) — never AND, which would require every name in
    one query to appear in the same row.  ``None`` when the query tokenizes to
    nothing (no word characters at all), so the caller skips the keyword leg
    entirely rather than issuing an empty-term MATCH."""
    tokens = re.findall(r"\w+", text_.lower())
    if not tokens:
        return None
    return " OR ".join(f'"{token}"*' for token in tokens)


def _count[T](column: T) -> ColumnElement[int]:
    """``func.count(col(...))``, typed once.

    ``ty`` resolves a bare SQLModel attribute (``Model.field``) to the
    field's plain Python type rather than a column expression, so
    ``func.count(Model.field)`` fails ``invalid-argument-type`` — the same
    reason filter expressions wrap through ``col()`` (house rule, root
    ``CLAUDE.md``). This wraps it once so a counting ``select`` never needs
    a suppression at its own call site.
    """
    return func.count(col(column))


class _MetadataUpdate(BaseModel):
    """The optional fields of a collection metadata update — only the set ones are
    applied.  ``apply_to`` mutates the row in place and returns the names of the
    fields that actually changed (the mutation event's ``changed_fields``, #1560),
    keeping ``update_collection_metadata`` short."""

    description: str | None = None
    description_embedding: list[float] | None = None
    notify: bool | None = None
    extraction_prompt: str | None = None
    # The schedule axis (#1857): the apply-time job schedule.  ``replace_schedule`` swaps
    # the whole schedule atomically from the rule plus the ``COUNT=`` quota lifted out of
    # it, so a new rule never leaves the previous rule's quota behind.  ``expires_at`` is
    # the end condition — an independent optional, set when provided, either stated
    # directly or lifted from the rule's ``UNTIL=``.
    schedule: str | None = None
    max_runs: int | None = None
    expires_at: datetime | None = None
    # Setting an end condition and REMOVING one are two different asks, and one optional
    # cannot carry both: ``None`` means "not stated" everywhere else on this model, so an
    # explicitly unbounded job needs its own signal (#1944).  Without it a sentinel expiry
    # normalised at the tool boundary could never come off a row that already held one —
    # exactly the stickiness the normalisation exists to end, since the invented value was
    # copied back on every later call.  Ignored when ``replace_schedule`` is set: a new
    # rule already carries both end conditions and overwrites them whole.
    clear_expiry: bool = False
    replace_schedule: bool = False

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
        changed.extend(self._apply_schedule(memory))
        return changed

    def _apply_schedule(self, memory: MemoryRow) -> list[str]:
        """The schedule axis (#1857).  ``replace_schedule`` swaps the rule AND both end
        conditions lifted out of it — the run quota and the expiry — atomically, so a new
        rule never inherits the previous rule's ``COUNT=`` or ``UNTIL=``; the caller has
        already folded any ``expires_at`` argument in, so what lands is what this call
        stated and nothing older.  Without a new rule there is nothing to inherit, so a
        bare ``expires_at`` edit sets the expiry alone and a plain ``schedule`` poke (the
        iOS / browser memory UIs) sets the rule alone — or, on ``clear_expiry``, takes the
        end condition off entirely (#1944), reported as a change only when there was one
        to remove."""
        changed: list[str] = []
        if self.replace_schedule:
            memory.schedule = self.schedule
            memory.max_runs = self.max_runs
            memory.expires_at = self.expires_at
            return ["schedule", "expires_at"] if self.expires_at is not None else ["schedule"]
        if self.schedule is not None:
            memory.schedule = self.schedule
            changed.append("schedule")
        if self.expires_at is not None:
            memory.expires_at = self.expires_at
            changed.append("expires_at")
        elif self.clear_expiry and memory.expires_at is not None:
            memory.expires_at = None
            changed.append("expires_at")
        return changed


# The label ``_MetadataUpdate.apply_to`` reports a field changed under, mapped to how
# that field reads OFF THE ROW right now (#1946).  A mutation event says which fields
# moved; snapshotting these before ``apply_to`` runs is what lets it also say what they
# moved FROM — the half the row itself cannot keep, since the edit overwrites it.
#
# The label set is this module's own vocabulary (``apply_to`` mints it a few lines
# above), so the table is closed by construction rather than by convention, and a test
# pins that every label ``apply_to`` can emit has a reader here.  A label that somehow
# arrives without one is LOGGED and left unrecorded — an unrecorded prior renders as no
# clause at all, which is the honest shape; an invented one would be the exact defect
# this exists to remove.
#
# Values are rendered as STRINGS here because a prior is read back beside the field's
# current value on a model-facing surface, and ``None`` means the field held nothing.
_FIELD_PRIORS: dict[str, Callable[[MemoryRow], str | None]] = {
    "description": lambda row: row.description,
    "notify": lambda row: str(row.notify),
    "extraction_prompt": lambda row: row.extraction_prompt,
    "schedule": lambda row: row.schedule,
    "expires_at": lambda row: (
        format_log_timestamp(row.expires_at) if row.expires_at is not None else None
    ),
}

# The archive flag is not an ``apply_to`` label — archiving is its own action rather
# than a field edit — so ``_set_archived`` names it here and records its own prior the
# same way, and every mutation this store writes over an existing row says what that
# row held before it.
_ARCHIVED_FIELD = "archived"


def snapshot_fields(memory: MemoryRow) -> dict[str, str | None]:
    """Every prior-readable field as it stands RIGHT NOW, the archive flag included, in the
    exact form a recorded prior carries it.

    Taken before an update is applied, inside the transaction that applies it, so the
    recorded before-values are a copy of the row rather than a reconstruction of it (#1946).
    PUBLIC because a second reader compares a prior against the value the row holds NOW —
    what a run's edits came to, as distinct from what its calls stated — and that comparison
    is only sound when both sides went through one reader."""
    fields = {name: read(memory) for name, read in _FIELD_PRIORS.items()}
    fields[_ARCHIVED_FIELD] = str(memory.archived)
    return fields


def _priors_from(before: dict[str, str | None], changed: list[str]) -> list[FieldPrior]:
    """The snapshot narrowed to the fields that actually moved, in the order the update
    reported them."""
    priors: list[FieldPrior] = []
    for name in changed:
        if name not in before:
            logger.warning("Changed field %r has no prior reader — its before-value is lost", name)
            continue
        priors.append(FieldPrior(field=name, value=before[name]))
    return priors


class MemoryStore:
    """Registry + factory for memories.

    Summary of the public surface:
        * dispatch: memory, run_log
        * the bound cycle's undo journal (#1936): begin_cycle, end_cycle
        * metadata: create_collection, create_log, get, list_all, archive,
          unarchive, update_collection_metadata, link_source_message,
          mark_collected
        * inventory: entry_counts, names_with_entry_match
        * resolve by meaning: resolve_objects
        * embedding backfill: get_entries_without_embeddings,
          get_memories_without_description_embedding, set_description_embedding,
          set_entry_embeddings
    """

    def __init__(
        self,
        engine,
        runtime: RuntimeParams | None = None,
        mutations: MutationStore | None = None,
        send_queue: SendQueueStore | None = None,
        clock: Clock | None = None,
    ):
        self.engine = engine
        # The clock every row this store writes is stamped from, and the one it hands
        # each ``Memory`` object it builds (``penny/clock.py``).
        self._clock = clock if clock is not None else Clock()
        # Runtime params handed to each ``Memory`` object this store builds.
        self._runtime = runtime if runtime is not None else RuntimeParams()
        # The registry-mutation ledger (#1560).  Create/update/archive/unarchive
        # of a collection each write a durable event here, at this Python
        # chokepoint — so no caller (chat tool, scheduler) can mutate a mechanism
        # without the provenance being recorded.  Defaults to one over the shared
        # engine when the facade doesn't inject it (isolated tests), so the
        # recording is never silently skipped.
        self._mutations = mutations if mutations is not None else MutationStore(engine, self._clock)
        # Archiving a collection cancels its still-pending queued sends (#1634):
        # ``_set_archived`` is the chokepoint every archive path funnels through
        # (user, system max_runs/expiry), so cancelling here means teardown is
        # silent through the send queue for all of them.  Defaults like the
        # ledger so isolated tests still get the cancellation.
        self._send_queue = (
            send_queue if send_queue is not None else SendQueueStore(engine, self._clock)
        )
        # Fired after any mutation so observers (the browser extension, iOS) can
        # refresh.  A list, not a slot (#2441): every connected channel subscribes
        # its own listener, so each gets the refresh instead of only the one built
        # last silently taking over the other's.  ``_notify_changed`` is the single
        # callable the factory injects into each Memory object it builds, and it
        # fans out to every subscriber.
        self._memory_changed_listeners: list[Callable[[str | None], None]] = []
        # The bound collector cycle's undo journal (#1936) — ``None`` outside a cycle,
        # which is when there is nothing to undo.  It lives here rather than being
        # passed down because the mutations it records happen at this layer's own
        # chokepoint, several call frames below anything the collector holds.
        self._journal: CycleJournal | None = None

    def subscribe_memory_changed(self, listener: Callable[[str | None], None]) -> None:
        """Add a listener notified after any memory mutation."""
        self._memory_changed_listeners.append(listener)

    # ── The bound cycle's undo journal (#1936) ────────────────────────────────

    def begin_cycle(self, name: str | None) -> None:
        """Record ``name``'s entry mutations from here until the cycle settles.

        ``None`` records nothing, which is the state every caller but a bound collector
        cycle is in — chat writes ride straight past.  A fresh journal replaces any
        previous one, so a cycle that never settled leaves nothing to replay onto the
        next: dropping is the conservative direction, being exactly the behaviour that
        stood before this existed.
        """
        self._journal = CycleJournal(name) if name is not None else None

    def end_cycle(self, *, revert: bool) -> bool:
        """Settle the bound cycle's journal, returning whether its writes SURVIVED.

        A healthy end drops the journal — the writes were already durable, so there is
        nothing to do.  An unhealthy one replays it in ONE short transaction (prior
        absent → delete the row; else restore the fields), so the retry runs against the
        genuine pre-run state and the change-gate never reads a value recorded by a run
        nobody heard from.  Reverting is a real change, so it announces itself like any
        other (``memory_changed``) — the surfaces showing the collection are showing the
        dead cycle's writes.
        """
        journal = self._journal
        self._journal = None
        if journal is None or not revert:
            return True
        if journal.touched:
            with self._session() as session:
                keys = journal.revert(session)
                session.commit()
            logger.info(
                "Reverted %d key(s) of '%s' — the cycle did not end",
                keys,
                journal.collection,
            )
            self._notify_changed(journal.collection)
        return False

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
        return RunLog(row, self.engine, on_changed=self._notify_changed, clock=self._clock)

    def _build(self, row: MemoryRow) -> Memory:
        """Construct the right ``Memory`` subclass for an already-loaded row."""
        if row.name in _MESSAGE_LOG_DIRECTIONS:
            return MessageLogMemory(
                row,
                self.engine,
                direction=_MESSAGE_LOG_DIRECTIONS[row.name],
                on_changed=self._notify_changed,
                clock=self._clock,
            )
        if row.name == PennyConstants.MEMORY_COLLECTOR_RUNS_LOG:
            return RunLog(row, self.engine, on_changed=self._notify_changed, clock=self._clock)
        if row.type == MemoryType.COLLECTION:
            return Collection(
                row,
                self.engine,
                runtime=self._runtime,
                on_changed=self._notify_changed,
                journal=self._journal_for(row.name),
                clock=self._clock,
            )
        return Log(row, self.engine, on_changed=self._notify_changed, clock=self._clock)

    def _journal_for(self, name: str) -> CycleJournal | None:
        """The bound cycle's journal, iff ``name`` is the collection it is bound to.

        The ONE place that rule is expressed: a cycle's writes are scoped to its bound
        target (``Collector._memory_scope``), so nothing else it reaches can be undone
        by its revert.
        """
        if self._journal is not None and self._journal.collection == name:
            return self._journal
        return None

    def _session(self) -> Session:
        return Session(self.engine)

    # ── Metadata ────────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        description: str,
        archived: bool = False,
        extraction_prompt: str | None = None,
        schedule: str | None = None,
        description_embedding: list[float] | None = None,
        notify: bool = False,
        created_by_run_id: str | None = None,
        expires_at: datetime | None = None,
        max_runs: int | None = None,
    ) -> MemoryRow:
        return self._create_memory(
            name,
            MemoryType.COLLECTION,
            description,
            archived,
            extraction_prompt=extraction_prompt,
            schedule=schedule,
            description_embedding=description_embedding,
            notify=notify,
            created_by_run_id=created_by_run_id,
            expires_at=expires_at,
            max_runs=max_runs,
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
        schedule: str | None = None,
        description_embedding: list[float] | None = None,
        notify: bool = False,
        created_by_run_id: str | None = None,
        expires_at: datetime | None = None,
        max_runs: int | None = None,
    ) -> MemoryRow:
        name = slug(name)
        if self.get(name) is not None:
            raise MemoryAlreadyExistsError(name)
        now = self._clock.now()
        with self._session() as session:
            memory = MemoryRow(
                name=name,
                type=type_.value,
                description=description,
                description_embedding=sim.maybe_serialize(description_embedding),
                archived=archived,
                notify=notify,
                extraction_prompt=extraction_prompt,
                # The collection's whole schedule: one RRULE (#1857).  NULL for a log
                # or an inert collection, which never dispatches.
                schedule=schedule,
                # Provenance + lifecycle (#1566): the creating run and the end
                # conditions.  All None for seeded / system rows; the spawning
                # message is linked post-run via ``link_source_message``.  The two
                # end conditions are also what the rule's UNTIL= / COUNT= lift into.
                created_by_run_id=created_by_run_id,
                expires_at=expires_at,
                max_runs=max_runs,
                created_at=now,
                updated_at=now,
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
                select(MemoryEntry.memory_name, _count(MemoryEntry.id)).group_by(
                    MemoryEntry.memory_name
                )
            ).all()
            counts = dict(rows)
            by_direction = dict(
                session.exec(
                    select(MessageLog.direction, _count(MessageLog.id))
                    .where(col(MessageLog.is_reaction).is_(False))
                    .group_by(MessageLog.direction)
                ).all()
            )
            run_count = session.exec(
                select(_count(PromptLog.id)).where(
                    col(PromptLog.run_outcome).isnot(None),
                    col(PromptLog.run_target).isnot(None),
                )
            ).one()
        for log_name, direction in _MESSAGE_LOG_DIRECTIONS.items():
            counts[log_name] = by_direction.get(direction, 0)
        counts[PennyConstants.MEMORY_COLLECTOR_RUNS_LOG] = run_count or 0
        return counts

    def writes_by_run(self, run_ids: list[str]) -> dict[str, list[RunWrite]]:
        """The keyed entries each run wrote the CURRENT value of, grouped by run
        id (#1641) — the ``last_written_by_run_id`` join the self-state activity
        block renders as a run's writes clause.

        Scoped to the given run ids (the runs the block shows) in one query, so
        the header stays a bounded read.  Keyed entries only (``key IS NOT
        NULL``): a log append carries no key and is browse scratch, not a
        registry write.  Ordered by ``(memory_name, created_at, id)`` — the name
        clusters each collection's writes into one render group; ``created_at``
        lists a group's keys oldest-first; ``id`` is only the same-instant
        tiebreak (datetime for ordering, ids for joins — house rule)."""
        if not run_ids:
            return {}
        grouped: dict[str, list[RunWrite]] = {}
        with self._session() as session:
            rows = session.exec(
                select(
                    MemoryEntry.last_written_by_run_id,
                    MemoryEntry.memory_name,
                    MemoryEntry.key,
                )
                .where(
                    col(MemoryEntry.last_written_by_run_id).in_(run_ids),
                    col(MemoryEntry.key).isnot(None),
                )
                .order_by(
                    col(MemoryEntry.memory_name).asc(),
                    col(MemoryEntry.created_at).asc(),
                    col(MemoryEntry.id).asc(),
                )
            ).all()
        for run_id, memory_name, key in rows:
            if run_id is None or key is None:
                continue
            grouped.setdefault(run_id, []).append(RunWrite(memory_name=memory_name, key=key))
        return grouped

    def entries_written_by_run(
        self, run_id: str, *, exclude_system_logs: bool
    ) -> list[MemoryEntry]:
        """Every entry whose CURRENT value this run wrote, oldest first (#1911) — what
        a finished cycle actually produced, read off the ledger stamp rather than
        inferred from which tools it called.

        ``writes_by_run``'s sibling, and deliberately not a widening of it: that one
        answers "which keys did these runs touch?" for the self-state header's one-line
        writes clause, so it projects ``(memory, key)`` across MANY runs and drops
        keyless rows.  This one is one run's whole rows — content included, log appends
        included — because the notify document renders what was written and the
        similarity anchor is built from it.  Reading the stamp rather than a
        ``collection_write`` call's arguments is what keeps it tool-agnostic: a routine
        writes through whatever tools it was taught with, including a plugin's (#1783).

        ``exclude_system_logs`` is required, no default (#2803): every caller states its
        own choice, so the type checker flags one that doesn't.  ``True`` drops rows in
        ``PennyConstants.SYSTEM_LOGS`` — the memories Python writes as a side effect of
        the run rather than one of the job's own write tools (``browse-results``, today;
        any future system log the same way).  A job's report wants only what the job
        itself kept, never a page the browse tool logged in passing; the self-state
        writes clause and the chat turn's own "what landed" record (``False``) want
        every entry.  The mark is the memory's own kind — which logs ``SYSTEM_LOGS``
        already names as framework-managed and the ``log_append`` gate already refuses
        model writes to — never a fresh name list grown for this call.
        """
        with self._session() as session:
            query = select(MemoryEntry).where(MemoryEntry.last_written_by_run_id == run_id)
            if exclude_system_logs:
                query = query.where(
                    col(MemoryEntry.memory_name).notin_(list(PennyConstants.SYSTEM_LOGS))
                )
            return list(
                session.exec(
                    query.order_by(col(MemoryEntry.created_at).asc(), col(MemoryEntry.id).asc())
                ).all()
            )

    def pages_read_by_run(self, run_ids: list[str]) -> dict[str, int]:
        """How many pages each run READ — the ``browse-results`` entries created
        under the run (#2752), one entry per fetched page, whichever tool fetched
        it.  The count rides the stamp browse writes at the store chokepoint
        (``_store_one_page`` → ``append(run_id=…)``), never the tool names a run's
        calls carry (§2.6) — the conversation map renders this as a run's ``read N``
        clause, and a run that read nothing contributes no clause.

        Scoped to the given run ids in one grouped query, so the map stays a
        bounded read like ``writes_by_run``.
        """
        if not run_ids:
            return {}
        with self._session() as session:
            rows = session.exec(
                select(MemoryEntry.created_by_run_id, func.count())
                .where(
                    MemoryEntry.memory_name == PennyConstants.MEMORY_BROWSE_RESULTS_LOG,
                    col(MemoryEntry.created_by_run_id).in_(run_ids),
                )
                .group_by(MemoryEntry.created_by_run_id)
            ).all()
        return {run_id: count for run_id, count in rows if run_id is not None}

    def names_with_entry_match(self, search: str) -> set[str]:
        """Names of memories holding an entry whose ``key`` or ``content``
        contains ``search`` — powers the addon's "search entries too" filter
        (a substring LIKE; on-demand, so not indexed)."""
        like = f"%{search}%"
        with self._session() as session:
            rows = session.exec(
                select(MemoryEntry.memory_name)
                .where(col(MemoryEntry.content).like(like) | col(MemoryEntry.key).like(like))
                .distinct()
            ).all()
            return set(rows)

    def _notify_changed(self, name: str | None) -> None:
        """Fan out to every subscriber; one raising must not stop the rest.

        This is the single callable the factory hands each ``Memory`` object as
        ``on_changed`` (#2441) — the object calls one thing, this is what turns
        that one call into every connected channel's own listener."""
        for listener in tuple(self._memory_changed_listeners):
            try:
                listener(name)
            except Exception:
                logger.exception("A memory-changed listener failed")

    def notify_changed(self, name: str) -> None:
        """Raise the memory-changed refresh signal without writing anything.

        On the collector path ``mark_collected`` used to be the ONLY source of this
        signal — it is what wakes the addon surfaces' detail view — so a cycle that
        withholds the stamp to keep its occurrence (#1935) has to raise it itself.
        Otherwise a "run this now" click whose cycle died on its model call leaves the
        panel unrefreshed, and any entries the cycle managed to write before dying stay
        invisible until something else touches the row.
        """
        self._notify_changed(slug(name))

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
            was_archived = snapshot_fields(memory)[_ARCHIVED_FIELD]
            memory.archived = archived
            memory.updated_at = self._clock.now()
            session.add(memory)
            session.commit()
        # Teardown means silence through the queue (#1634): archiving cancels this
        # collection's still-pending queued sends so the drainer never delivers a
        # message from a mechanism that no longer exists.  Unarchive does NOT
        # resurrect them — they belong to the dead epoch — so this runs on archive
        # only.  The count folds into the mutation note so the cancellation is
        # visible wherever the archive renders.
        cancelled = self._send_queue.cancel_pending(name) if archived else 0
        note = self._archive_note(note, cancelled)
        self._mutations.record(
            entity_type=MutationEntityType.COLLECTION,
            entity_name=name,
            action=action,
            actor=actor,
            run_id=run_id,
            # The flip's own before-value (#1946), beside whatever cause the note
            # carries.  ``changed_fields`` stays empty — archiving is named by its
            # ACTION, not by a field edit — so every render of this event is
            # byte-identical to what it was.
            detail=MutationDetail(
                note=note,
                priors=[FieldPrior(field=_ARCHIVED_FIELD, value=was_archived)],
            ),
        )
        self._notify_changed(name)

    @staticmethod
    def _archive_note(note: str | None, cancelled: int) -> str | None:
        """Fold the cancelled-pending-sends count into the archive event's note
        (#1634), after any policy cause a system archive already carries — so a
        ``max_runs`` retire that also cancelled sends reads
        "<policy reason>; cancelled N pending sends"."""
        if cancelled <= 0:
            return note
        clause = cancelled_sends_note(cancelled)
        return f"{note}; {clause}" if note else clause

    def update_collection_metadata(
        self,
        name: str,
        *,
        description: str | None = None,
        extraction_prompt: str | None = None,
        schedule: str | None = None,
        description_embedding: list[float] | None = None,
        notify: bool | None = None,
        max_runs: int | None = None,
        expires_at: datetime | None = None,
        clear_expiry: bool = False,
        replace_schedule: bool = False,
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
        non-``NULL`` one it could never detect.
        ``clear_expiry`` is how a caller says the job is UNBOUNDED, which a bare
        ``expires_at=None`` cannot say on a model where ``None`` means "not stated" (#1944).
        """
        name = slug(name)
        self._require_collection(name)
        fields = _MetadataUpdate(
            description=description,
            description_embedding=description_embedding,
            notify=notify,
            extraction_prompt=extraction_prompt,
            schedule=schedule,
            max_runs=max_runs,
            expires_at=expires_at,
            clear_expiry=clear_expiry,
            replace_schedule=replace_schedule,
        )
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            # Read the row BEFORE the edit lands, in the same transaction (#1946): once
            # ``apply_to`` runs, what these fields used to hold exists nowhere.
            before = snapshot_fields(memory)
            changed = fields.apply_to(memory)
            memory.updated_at = self._clock.now()
            session.add(memory)
            session.commit()
            session.refresh(memory)
        # Record the config change as a durable event (#1560).  Only when a field
        # actually moved — a no-op update (every field None) isn't a mutation.
        # The new values live verbatim in the run's promptlog tool call; the event
        # names which fields changed so the history reads without re-fetching them —
        # and carries what each one held BEFORE (#1946), which is the one fact neither
        # the row nor the call can still answer once the edit has landed.
        if changed:
            self._mutations.record(
                entity_type=MutationEntityType.COLLECTION,
                entity_name=name,
                action=MutationAction.UPDATED,
                actor=MutationActor.USER_RUN,
                run_id=run_id,
                detail=MutationDetail(changed_fields=changed, priors=_priors_from(before, changed)),
            )
        self._notify_changed(name)
        return memory

    def link_source_message(self, run_id: str, source_message_id: int) -> None:
        """Stamp the spawning message on every mechanism a chat run created (#1566).

        ``collection_configure`` records ``created_by_run_id`` at creation, but the
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
                    col(MemoryRow.source_message_id).is_(None),
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
        """Stamp ``last_collected_at = now`` — the collection has spent the schedule
        occurrence it was dispatched for, whether the cycle did work or closed empty.

        Withheld by the dispatcher from a cycle that did NOT spend one (#1935): a
        preempted or aborted cycle that changed nothing leaves the occurrence due, so
        the schedule is not advanced past a fire that never happened.
        """
        name = slug(name)
        with self._session() as session:
            memory = session.get(MemoryRow, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.last_collected_at = self._clock.now()
            session.add(memory)
            session.commit()
        self._notify_changed(name)

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
            col(MemoryEntry.content_embedding).is_(None),
            and_(
                col(MemoryEntry.key).is_not(None),
                col(MemoryEntry.key_embedding).is_(None),
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
                    .order_by(col(MemoryEntry.created_at).desc())
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

    # ── Resolve by meaning + by word (hybrid find, #2774) ──────────────────────

    def resolve_objects(self, anchor: list[float], query: str, limit: int) -> list[ResolvedHit]:
        """Rank Penny's own addressable things, best-first (#1558, #1640, #2774).

        TWO independently-ranked legs fused by reciprocal rank (``_fuse_rrf``):
        an EMBEDDING leg — plain-cosine nearest-neighbour over every registry row's
        description anchor (collections + logs, **archived included**) AND every
        stored entry's content/key anchors across the non-archived collections +
        real logs (#1640), positively-correlated candidates only (cosine > 0) — and
        a KEYWORD leg over stored entries' key/content via ``memory_entry_fts``
        (#2774; object descriptions are NOT indexed for keywords, so a collection/log
        hit only ever carries an embedding-leg score).  The search spans every
        family in one fused list, no up-front narrowing (#1643).

        Fusing by RANK (not raw score) means a candidate the keyword leg finds but
        whose embedding fell at or below the cosine>0 gate (so it's simply absent
        from the embedding leg) still surfaces on the keyword leg's own strength —
        the literal-name-beats-unremarkable-embedding case #2774 exists for.  An
        entry contributes up to two embedding facets (content + key); the embedding
        leg keeps the FIRST (best-scoring) facet of each entry (max-of-facets, same
        disjunction the write-dedup uses).  Capped at ``limit`` after a relative
        floor drops hits far below the section's own best fused score
        (``_apply_relative_floor``) — an unrelated query honestly returns fewer
        hits, or none, rather than padding out with filler.  The tool layer turns
        each hit into identity + the deterministic addressing; ambiguity (several
        hits) is returned, never silently resolved.
        """
        embedding_ranked = cast(
            "list[ResolvedHit]",
            self._rank_candidates(
                self._resolution_candidates(), anchor, PennyConstants.FIND_RRF_POOL
            ),
        )
        keyword_ranked = self._keyword_candidates_entries(query)
        fused = self._fuse_rrf(embedding_ranked, keyword_ranked, lambda hit: hit.fusion_identity)
        return self._apply_relative_floor(fused, limit)

    def resolve_messages(
        self, anchor: list[float], query: str, limit: int
    ) -> list[ResolvedMessage]:
        """``find``'s share of the conversation logs (#2289, #2774) — ranked and
        budgeted ENTIRELY WITHIN its own section: messages never compete with
        ``resolve_objects``'s store hits for a shared slot (the #2774 fix — a
        generic stored fact used to be able to fill every slot and crowd every
        message out).

        Same two-leg RRF fusion as ``resolve_objects``: an embedding leg (plain
        cosine over every embedded row of ``user-messages``/``penny-messages``,
        read through ``Memory.embedded_rows()`` — the identical rows/vectors
        ``read_similar`` ranks, never a second embed) and a keyword leg over
        ``messagelog_fts``.  Capped at ``limit`` after the same relative floor.
        """
        embedding_ranked = cast(
            "list[ResolvedMessage]",
            self._rank_candidates(self._message_candidates(), anchor, PennyConstants.FIND_RRF_POOL),
        )
        keyword_ranked = self._keyword_candidates_messages(query)
        fused = self._fuse_rrf(embedding_ranked, keyword_ranked, lambda message: message.id)
        return self._apply_relative_floor(fused, limit)

    def _message_candidates(self) -> list[tuple[ResolvedMessage, bytes]]:
        """Every embedded row in the two conversation logs as an embedding-leg
        candidate (#2289) — read through ``Memory.embedded_rows()`` on each log's own
        ``MessageLogMemory`` facade, so a message candidate here and a
        ``read_similar``/notify-time nearest-message hit are the identical row and
        vector.  A log absent from the registry (a bare test database) contributes
        nothing rather than raising."""
        candidates: list[tuple[ResolvedMessage, bytes]] = []
        for name in (
            PennyConstants.MEMORY_USER_MESSAGES_LOG,
            PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
        ):
            log = self.memory(name)
            if log is None:
                continue
            for row in log.embedded_rows():
                if row.content_embedding is None or row.id is None:
                    continue
                hit = ResolvedMessage(
                    id=row.id, memory_name=name, content=row.content, created_at=row.created_at
                )
                candidates.append((hit, row.content_embedding))
        return candidates

    def _fts_table_exists(self, name: str) -> bool:
        """Whether FTS virtual table ``name`` (migration 0127) exists on this
        connection (#2774). The keyword leg checks this before every MATCH query
        rather than assuming it, so a schema that predates (or deliberately skips,
        like the test suite's bare current-models schema — ``schema_template.py``)
        migration 0127 degrades to pure-embedding ranking instead of a raw SQL
        error on a missing table — the same honest-fallback shape the degraded
        decision-model/image/browser modes use elsewhere."""
        with self._session() as session:
            row = session.execute(
                text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": name},
            ).first()
        return row is not None

    def _fts_match_rows(self, table: str, query: str, sql: Any, extra_params: dict) -> list | None:
        """The raw rows one keyword-leg MATCH query returns, or ``None`` when
        there is nothing to run (#2774): an empty/word-less ``query``, or a
        schema without ``table`` yet (``_fts_table_exists``) — so each
        candidate builder's own early-return ("the keyword leg contributes
        nothing, fusion falls back to the embedding leg alone") stays one
        line.  ``extra_params`` carries whatever ``sql`` needs beyond ``:q``/
        ``:pool`` (e.g. the message leg's direction filter)."""
        match = _fts_match_query(query)
        if match is None or not self._fts_table_exists(table):
            return None
        params = {"q": match, "pool": PennyConstants.FIND_RRF_POOL, **extra_params}
        with self._session() as session:
            return list(session.execute(sql, params).all())

    def _keyword_candidates_entries(self, query: str) -> list[tuple[ResolvedEntry, int]]:
        """The keyword leg's ranked ``ResolvedEntry`` candidates (#2774) — every
        ``memory_entry`` whose key/content matches ``query`` via ``memory_entry_fts``
        (migration 0127), best ``bm25`` first, capped at ``FIND_RRF_POOL``, filtered
        to the same live (non-archived) containers the embedding leg's
        ``_entry_candidates`` filters to."""
        sql = text(
            "SELECT e.id, e.memory_name, e.key, e.content FROM memory_entry_fts f "
            "JOIN memory_entry e ON e.id = f.rowid "
            "WHERE memory_entry_fts MATCH :q ORDER BY bm25(memory_entry_fts) LIMIT :pool"
        )
        rows = self._fts_match_rows("memory_entry_fts", query, sql, {})
        if rows is None:
            return []
        kinds = {
            row.name: (
                ResolvedKind.COLLECTION if row.type == MemoryType.COLLECTION else ResolvedKind.LOG
            )
            for row in self.list_all()
            if not row.archived
        }
        return self._entry_hits_from_rows(rows, kinds)

    @staticmethod
    def _entry_hits_from_rows(
        rows: list, kinds: dict[str, ResolvedKind]
    ) -> list[tuple[ResolvedEntry, int]]:
        """One ``(ResolvedEntry, rank)`` pair per matched row, skipping a row
        whose containing collection/log is archived or gone."""
        candidates: list[tuple[ResolvedEntry, int]] = []
        for row in rows:
            if row.memory_name not in kinds:
                continue
            hit = ResolvedEntry(
                entry_id=row.id,
                memory_name=row.memory_name,
                container_kind=kinds[row.memory_name],
                key=row.key,
                content=row.content,
            )
            candidates.append((hit, len(candidates) + 1))
        return candidates

    def _keyword_candidates_messages(self, query: str) -> list[tuple[ResolvedMessage, int]]:
        """The keyword leg's ranked ``ResolvedMessage`` candidates (#2774) — every
        non-reaction conversation message matching ``query`` via ``messagelog_fts``
        (migration 0127), best ``bm25`` first, capped at ``FIND_RRF_POOL``."""
        sql = text(
            "SELECT m.id, m.direction, m.content, m.mechanism, m.timestamp "
            "FROM messagelog_fts f JOIN messagelog m ON m.id = f.rowid "
            "WHERE messagelog_fts MATCH :q AND m.is_reaction = 0 "
            "AND m.direction IN (:incoming, :outgoing) "
            "ORDER BY bm25(messagelog_fts) LIMIT :pool"
        )
        extra = {
            "incoming": PennyConstants.MessageDirection.INCOMING,
            "outgoing": PennyConstants.MessageDirection.OUTGOING,
        }
        rows = self._fts_match_rows("messagelog_fts", query, sql, extra)
        if rows is None:
            return []
        return self._message_hits_from_rows(rows)

    @staticmethod
    def _message_hits_from_rows(rows: list) -> list[tuple[ResolvedMessage, int]]:
        """One ``(ResolvedMessage, rank)`` pair per matched row, rendered
        through the SAME emission-provenance marker
        (``MessageLogMemory.SENT_BY_MARKER``) the embedding leg's
        ``_render_content`` applies, so a keyword-only hit reads identically to
        one the embedding leg would have rendered."""
        candidates: list[tuple[ResolvedMessage, int]] = []
        for row in rows:
            content = row.content
            if row.mechanism is not None:
                content = (
                    f"{MessageLogMemory.SENT_BY_MARKER.format(mechanism=row.mechanism)}{content}"
                )
            hit = ResolvedMessage(
                id=row.id,
                memory_name=_MESSAGE_LOG_BY_DIRECTION[row.direction],
                content=content,
                created_at=row.timestamp,
            )
            candidates.append((hit, len(candidates) + 1))
        return candidates

    @staticmethod
    def _fuse_rrf[T](
        embedding_ranked: Sequence[T],
        keyword_ranked: Sequence[tuple[T, int]],
        identity: Callable[[T], Hashable],
    ) -> list[tuple[T, float]]:
        """Reciprocal rank fusion (#2774) over two independently-ranked lists of the
        SAME item type: ``score = Σ_leg 1 / (FIND_RRF_K + rank)``, 1-indexed rank, a
        leg an item is absent from contributing 0.  Fusing by RANK (never the raw
        cosine/bm25 score, which live on unrelated scales) is what lets a
        keyword-only hit outrank an embedding-only one on the strength of its own
        leg, with no cross-scale rescaling.  ``identity`` says which two candidates
        — one from each leg — are "the same row" (an entry's id, an object's name, a
        message's id); the first occurrence (whichever leg reaches it first) is kept
        as the representative item.  Returns every fused item, best-first — the
        caller applies its own budget/floor."""
        scores: dict[Hashable, float] = {}
        items: dict[Hashable, T] = {}
        for rank, item in enumerate(embedding_ranked, start=1):
            key = identity(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (PennyConstants.FIND_RRF_K + rank)
            items.setdefault(key, item)
        for item, rank in keyword_ranked:
            key = identity(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (PennyConstants.FIND_RRF_K + rank)
            items.setdefault(key, item)
        return sorted(
            ((items[key], score) for key, score in scores.items()), key=lambda pair: -pair[1]
        )

    @staticmethod
    def _apply_relative_floor[T](scored: list[tuple[T, float]], limit: int) -> list[T]:
        """The relative floor (#2774): within one fused, best-first ``scored`` list,
        drop every item scoring below ``FIND_RELATIVE_FLOOR_RATIO`` of the list's OWN
        best score, then cap at ``limit``.  Replaces the old absolute cosine>0 gate
        as the thing that keeps an unrelated query from padding out to the budget
        with the same few generic entries — a weak section (nothing really matches)
        now honestly returns fewer hits, or none, instead of exactly ``limit``."""
        if not scored:
            return []
        floor = scored[0][1] * PennyConstants.FIND_RELATIVE_FLOOR_RATIO
        return [item for item, score in scored if score >= floor][:limit]

    @staticmethod
    def _rank_candidates(
        candidates: Sequence[tuple[ResolvedHit | ResolvedMessage, bytes]],
        anchor: list[float],
        limit: int,
    ) -> list[ResolvedHit | ResolvedMessage]:
        """The best-first walk shared by ``resolve_objects`` and ``resolve_messages``
        (#2289) — plain cosine, positively-correlated only, capped at ``limit``,
        deduping a ``ResolvedEntry``'s two facets (content/key) to its best-scoring
        one (the "either signal" disjunction the write-dedup uses). A
        ``ResolvedMessage`` carries exactly one facet, so it never needs the dedup.
        ``Sequence`` (covariant), not ``list`` (invariant, #2289): ``resolve_objects``
        passes a narrower ``list[tuple[ResolvedHit, bytes]]`` straight through."""
        if not candidates:
            return []
        scores = sim.cosine_scores([blob for _, blob in candidates], anchor)
        ranked: list[ResolvedHit | ResolvedMessage] = []
        seen_entries: set[int] = set()
        for index in np.argsort(-scores):
            if float(scores[index]) <= 0.0:
                break  # ``argsort(-scores)`` is score-descending — the rest are ≤ 0 too.
            hit = candidates[index][0]
            if isinstance(hit, ResolvedEntry):
                if hit.entry_id in seen_entries:
                    continue  # a lower-scoring facet of an entry already ranked.
                seen_entries.add(hit.entry_id)
            ranked.append(hit)
            if len(ranked) >= limit:
                break
        return ranked

    def nearest_memory_name(self, name: str, embedding: list[float] | None = None) -> str | None:
        """The closest EXISTING (active) memory name to a missed ``name`` — the
        did-you-mean candidate a not-found error leads with (#1674), or ``None``
        when nothing is close (the error then stays exactly as before).

        Two legs, string first — it caught the motivating ``aurora-deone`` →
        ``aurora-deck-2`` typo and needs no embedding: a stdlib
        ``difflib`` close-match over the active names, gated on the conservative
        ``DID_YOU_MEAN_STRING_CUTOFF``.  When the string leg finds nothing AND an
        ``embedding`` of the missed name is supplied (only the embedding-capable
        tools compute it — most misses hit a plain read that passes ``None``), the
        MEANING leg ranks the active rows' description anchors by plain cosine and
        takes the best clearing the house ``MEMORY_DEDUP_CONTENT_SIM_RELAXED``
        threshold — catching a semantically-close-but-differently-worded name.
        Reuses existing numbers: the string cutoff is the one named constant (no
        house string-distance constant existed); the meaning gate is the dedup
        threshold.  The candidate excludes ``name`` itself and every archived row.
        """
        active = [
            row for row in self.list_all() if not row.archived and slug(row.name) != slug(name)
        ]
        typo = self._nearest_name_by_string(name, active)
        if typo is not None:
            return typo
        if embedding is not None:
            return self._nearest_name_by_meaning(embedding, active)
        return None

    @staticmethod
    def _nearest_name_by_string(name: str, active: list[MemoryRow]) -> str | None:
        """The string (typo) leg of :meth:`nearest_memory_name`: the closest active
        name by ``difflib`` ratio over slugs, or ``None`` below the cutoff.  Matches
        on the normalized slug (so casing / dash variants don't hide a typo) but
        returns the row's DISPLAY name — what the model copies back."""
        by_slug = {slug(row.name): row.name for row in active}
        matches = difflib.get_close_matches(
            slug(name),
            list(by_slug),
            n=1,
            cutoff=PennyConstants.DID_YOU_MEAN_STRING_CUTOFF,
        )
        return by_slug[matches[0]] if matches else None

    def _nearest_name_by_meaning(
        self, embedding: list[float], active: list[MemoryRow]
    ) -> str | None:
        """The meaning leg of :meth:`nearest_memory_name`: the active memory whose
        description anchor is closest (plain cosine) to the missed name's
        ``embedding``, when it clears ``MEMORY_DEDUP_CONTENT_SIM_RELAXED`` — else
        ``None``.  Rows without a description anchor are skipped (the backfill fills
        them); reuses the same ``cosine_scores`` pass ``resolve_objects`` uses."""
        candidates = [
            (row.name, row.description_embedding)
            for row in active
            if row.description_embedding is not None
        ]
        if not candidates:
            return None
        scores = sim.cosine_scores([blob for _, blob in candidates], embedding)
        best = int(np.argmax(scores))
        threshold = RuntimeParams().MEMORY_DEDUP_CONTENT_SIM_RELAXED
        if float(scores[best]) >= threshold:
            return candidates[best][0]
        return None

    def _resolution_candidates(self) -> list[tuple[ResolvedHit, bytes]]:
        """Every (hit, embedding-blob) pair eligible for ``resolve_objects`` —
        registry rows carrying a description anchor, and stored entries carrying a
        content/key anchor.  A row/entry without its vector is silently absent (the
        backfill fills it) — never surfaced unscored.  Every family contributes; the
        search never narrows up front (#1643)."""
        candidates: list[tuple[ResolvedHit, bytes]] = []
        for row in self.list_all():
            if row.description_embedding is None:
                continue
            row_kind = (
                ResolvedKind.COLLECTION if row.type == MemoryType.COLLECTION else ResolvedKind.LOG
            )
            match = ResolvedMatch(
                name=row.name, kind=row_kind, archived=row.archived, label=row.description
            )
            candidates.append((match, row.description_embedding))
        candidates.extend(self._entry_candidates())
        return candidates

    def _entry_candidates(self) -> list[tuple[ResolvedEntry, bytes]]:
        """Every (entry-hit, facet-embedding) pair eligible for resolution (#1640).

        Each ``memory_entry`` in a NON-archived collection or real log contributes
        up to two facet candidates — its ``content_embedding`` and, when keyed, its
        ``key_embedding`` — so a query paraphrasing either the stored value or its
        key surfaces the entry (``resolve_objects`` dedups the two facets by entry
        id, keeping the better-scoring — max-of-facets).  Facades over
        ``messagelog`` / ``promptlog`` have no ``memory_entry`` rows, so they never
        appear; archived memories are excluded via the containing-row filter."""
        pairs: list[tuple[ResolvedEntry, bytes]] = []
        for entry, container_kind in self._embedded_entry_rows():
            hit = ResolvedEntry(
                entry_id=entry.id,  # ty: ignore[invalid-argument-type]
                memory_name=entry.memory_name,
                container_kind=container_kind,
                key=entry.key,
                content=entry.content,
            )
            if entry.content_embedding is not None:
                pairs.append((hit, entry.content_embedding))
            if entry.key_embedding is not None:
                pairs.append((hit, entry.key_embedding))
        return pairs

    def _embedded_entry_rows(self) -> list[tuple[MemoryEntry, ResolvedKind]]:
        """Every embedded entry in a non-archived collection/log, paired with its
        container's kind (``collection`` | ``log``).  One pass over ``memory_entry``
        — a single-user deployment, so no index gymnastics — filtered against the
        registry's live (non-archived) rows built from ``list_all`` (which also
        supplies each container's shape); an entry whose memory is archived or is a
        facade marker with no real rows never matches."""
        kinds = {
            row.name: (
                ResolvedKind.COLLECTION if row.type == MemoryType.COLLECTION else ResolvedKind.LOG
            )
            for row in self.list_all()
            if not row.archived
        }
        with self._session() as session:
            entries = session.exec(
                select(MemoryEntry).where(
                    or_(
                        col(MemoryEntry.content_embedding).is_not(None),
                        col(MemoryEntry.key_embedding).is_not(None),
                    )
                )
            ).all()
        return [
            (entry, kinds[entry.memory_name]) for entry in entries if entry.memory_name in kinds
        ]

    # ── Idempotency at birth (#1567) ──────────────────────────────────────────

    def find_duplicate_collection(self, name: str) -> MemoryRow | None:
        """The first EXISTING collection a proposed one is the same JOB as (#1567,
        #1775), or ``None`` when the target is genuinely distinct.

        Decided by the shared job-identity rule (``is_same_job``): strict containment
        of the normalised names — never a hand-rolled similarity rule, and with no
        threshold.  **Embeddings deliberately do not participate**: measured against
        the labelled corpus they are anti-correlated on the hard cases (an embedding
        says "same kind of thing", and every trap is two collections of the same kind
        about different specifics), so the ``description`` anchor stays what ``find``
        resolves on and says nothing about duplication.

        Active rows are checked before archived ones (a tombstone), so a live "already
        watching this" wins over a retired one.  Nothing is excluded any more (#1911):
        with no pre-seeded collections left, every row in the registry is one the user
        built, so there is no framework machinery to skip past.
        """
        candidate = JobSide(name=slug(name))
        active, archived = self._duplicate_candidates()
        for row in (*active, *archived):
            if is_same_job(candidate, self._collection_side(row)):
                return row
        return None

    def _duplicate_candidates(self) -> tuple[list[MemoryRow], list[MemoryRow]]:
        """Collections eligible for the idempotency check, partitioned active
        vs. archived — logs excluded, and nothing else (#1911: every collection is a
        user collection now)."""
        active: list[MemoryRow] = []
        archived: list[MemoryRow] = []
        for row in self.list_all():
            if row.type != MemoryType.COLLECTION.value:
                continue
            (archived if row.archived else active).append(row)
        return active, archived

    @staticmethod
    def _collection_side(row: MemoryRow) -> JobSide:
        """One existing collection as a job-identity side: what it is called."""
        return JobSide(name=row.name)

    # ── Dedup probe ───────────────────────────────────────────────────────────

    def _rows_by_key(self, name: str, key: str) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == name, MemoryEntry.key == key
                    )
                ).all()
            )

    def _require_collection(self, name: str) -> None:
        memory = self.get(name)
        if memory is None:
            raise MemoryNotFoundError(name)
        if memory.type != MemoryType.COLLECTION:
            raise MemoryTypeError(wrong_shape_message(name, memory.type))
