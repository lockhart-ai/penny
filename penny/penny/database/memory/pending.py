"""Two-phase entry writes — a run's ``memory_entry`` mutations, held until it ends
healthily (#1936).

A collector cycle that browsed, WROTE the changed value, and then died before its
conclusion left that value behind for the next cycle to trip over: the retry re-observed
the same page, wrote the identical value, and the change-gate read
``KEY_EXISTS_UNCHANGED`` → STOP → the user was never told about a change that really
happened.  The run consumed the change and delivered nothing.

The fix is the ``AgentCursor`` lifecycle applied to entry writes: a cycle's mutations
are PENDING while it runs and land only at its healthy conclusion, so an aborted cycle
is thrown away whole and the retry runs against the true pre-run state.  The buffer is
what "pending" is made of:

- it stages the three shapes an entry mutation takes — an INSERT of a new row, a
  REWRITE of an existing one's value, and a REMOVAL — so a routine's write, the
  change-gate's own ``KEY_EXISTS_CHANGED`` baseline refresh (#1633), an
  ``update_entry``, a ``collection_delete_entry`` and a ``log_append`` all stage
  identically.  Nothing here knows a tool name;
- it is READ THROUGH (:meth:`merge`), so the run sees its own pending writes — its
  dedup reads them, its re-reads return them, and the notify document, which is
  assembled BEFORE the commit, renders them;
- it lands in ONE short transaction (:meth:`commit`).  Never a run-long one: SQLite has
  a single writer and a cycle can run for minutes, so holding the write lock across it
  would block every chat turn.

What it deliberately does NOT hold is the LEDGER — the ``promptlog`` rows, the mutation
events, the send queue.  Those are the forensic record of what happened, including what
a dead cycle did, so they are written live and an aborted run stays fully diagnosable.

**Declared limits**, both about an entry with no id yet:

- ``Memory.entry_by_id`` cannot resolve a staged row, and neither can ``find``
  (``MemoryStore._embedded_entry_rows`` runs its own cross-registry query, and
  ``ResolvedEntry`` is addressed BY id).  A staged entry has no id until its cycle's
  conclusion lands it, so there is nothing for either to name — the honest answer being
  that the entry does not exist yet for anything that addresses entries that way.  The
  keyed and ranked reads a routine actually works through (``get`` · ``read_latest`` ·
  ``read_all`` · ``read_similar`` · ``keys`` · ``entry_count``) all read through.
- ``WriteResult.entry_id`` is ``None`` on a staged insert, for the same reason.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from sqlmodel import Session

from penny.database.models import MemoryEntry

logger = logging.getLogger(__name__)

# What a rewrite carries over onto the live row at commit: the fields an entry mutation
# is allowed to change.  ``created_at`` / ``created_by_run_id`` are deliberately absent
# — a rewrite is not a creation — and ``memory_name`` is present because a move is a
# rewrite of where the entry lives.
_REWRITABLE_FIELDS = (
    "memory_name",
    "key",
    "content",
    "author",
    "key_embedding",
    "content_embedding",
    "last_written_by_run_id",
)

_VANISHED = (
    "Staged entry %d for '%s' is gone from the store — it was removed while the cycle "
    "was running, so this cycle's change to it is dropped rather than re-created."
)


def detached(row: MemoryEntry) -> MemoryEntry:
    """A free-standing copy of ``row`` — same field values, no session behind it.

    Every row a staged read hands back is one of these, so a caller that mutates what it
    read (which is exactly how ``update_entry`` and the baseline refresh are written)
    cannot dirty a live session and land the change early.
    """
    return MemoryEntry(**row.model_dump())


def sort_key(row: MemoryEntry) -> datetime:
    """``row``'s creation time, always UTC-aware.

    Staged rows are minted with an aware ``datetime.now(UTC)`` and stored rows come back
    from SQLite naive, so merging the two means ordering a mixed list — which raises
    rather than mis-sorting if the two forms meet untouched.
    """
    created = row.created_at
    return created if created.tzinfo is not None else created.replace(tzinfo=UTC)


class PendingEntryWrites:
    """One run's staged ``memory_entry`` mutations: read through, then committed or
    discarded as a unit.

    Held by the run's staged store view (``MemoryStore.staged``), never by the store the
    rest of the system uses — so a cycle's staging is invisible to a chat turn happening
    at the same moment, and chat's own writes are untouched.
    """

    def __init__(self) -> None:
        # New rows, in the order they were staged; ``id`` is None until they land.
        self._inserts: list[MemoryEntry] = []
        # Detached copies of stored rows carrying this run's value for them, by row id.
        self._rewrites: dict[int, MemoryEntry] = {}
        # Stored rows this run deleted, by row id (the row is kept for its name).
        self._removals: dict[int, MemoryEntry] = {}
        # Which memory each stored row belonged to when this run first read it, by row
        # id.  Noted at the overlay, because that is the only place the PRE-RUN location
        # is still readable: a mutation that moves an entry hands the buffer a row
        # already carrying its destination, so asking the row at staging time would say
        # the entry had always lived where it has just been put.
        self._origins: dict[int, str] = {}

    def __bool__(self) -> bool:
        return bool(self._inserts or self._rewrites or self._removals)

    # ── Staging ──────────────────────────────────────────────────────────────

    def insert(self, row: MemoryEntry) -> None:
        """Stage a new row.  It has no id until it lands, which is the honest answer to
        "what id is it?" — nothing can address it yet."""
        self._inserts.append(row)

    def rewrite(self, row: MemoryEntry) -> None:
        """Stage ``row``'s current field values as this run's value for it.

        A row the run already staged as an INSERT carries no id, and the caller mutated
        that very instance, so it is already staged — recording it again would file the
        same entry twice.
        """
        if row.id is None:
            return
        self._rewrites[row.id] = row

    def remove(self, row: MemoryEntry) -> None:
        """Stage a delete.  A row this run staged and then deleted never existed as far
        as the store is concerned, so it is dropped from the buffer rather than queued
        as a removal of something that was never written."""
        if row.id is None:
            self._inserts = [staged for staged in self._inserts if staged is not row]
            return
        self._rewrites.pop(row.id, None)
        self._removals[row.id] = row

    # ── Reading through ──────────────────────────────────────────────────────

    def merge(self, memory_name: str, rows: list[MemoryEntry]) -> list[MemoryEntry]:
        """``rows`` as this run sees them — a superset of the answer, unordered.

        The caller re-applies its own predicate and ordering, because only the caller
        knows them; what this settles is WHICH entries exist for it to apply them to.
        """
        return self._overlay(rows, lambda row: row.memory_name == memory_name)

    def written_by_run(self, run_id: str, rows: list[MemoryEntry]) -> list[MemoryEntry]:
        """What ``run_id`` wrote the current value of, oldest first — the notify
        document's read, answered across every memory rather than one.

        The whole reason it has to read through: the document is assembled at the
        cycle's conclusion but BEFORE the commit, so what the cycle wrote is still
        staged and the store alone would say it wrote nothing.
        """
        merged = self._overlay(rows, lambda row: row.last_written_by_run_id == run_id)
        return sorted(merged, key=sort_key)

    def _overlay(
        self, rows: list[MemoryEntry], keep: Callable[[MemoryEntry], bool]
    ) -> list[MemoryEntry]:
        """``rows`` with this run's staging folded in, filtered by ``keep``.

        Removals drop out and rewrites take their staged value.  Staged inserts join,
        and so does a rewrite the query's own window never returned — a row that gained
        an embedding, or moved, or was rewritten by this run into ``keep``'s answer, is
        invisible to the predicate the query itself ran.
        """
        merged: list[MemoryEntry] = []
        seen: set[int] = set()
        for row in rows:
            staged = None
            if row.id is not None:
                # The query reads the store, so this row still says where it lived
                # before the run touched it — the one moment that is readable.
                self._origins.setdefault(row.id, row.memory_name)
                if row.id in self._removals:
                    continue
                seen.add(row.id)
                staged = self._rewrites.get(row.id)
            entry = staged if staged is not None else detached(row)
            if keep(entry):
                merged.append(entry)
        merged.extend(row for row in self._inserts if keep(row))
        merged.extend(
            row for row_id, row in self._rewrites.items() if row_id not in seen and keep(row)
        )
        return merged

    def _transfers(self) -> list[tuple[str | None, str | None]]:
        """Every staged mutation as (where the entry was, where it is now).

        The one shape all three stagings share, so what a mutation DOES to a memory's
        contents is asked once instead of per kind — an insert arrives from nowhere, a
        removal goes nowhere, and a rewrite that changed nothing but a value arrives
        where it already was.  A relocation is then simply a transfer with two
        different ends, and nothing has to know that ``move`` exists.
        """
        return [
            *((None, row.memory_name) for row in self._inserts),
            *(
                (self._origins.get(row_id), row.memory_name)
                for row_id, row in self._rewrites.items()
            ),
            *(
                (self._origins.get(row_id, row.memory_name), None)
                for row_id, row in self._removals.items()
            ),
        ]

    def staged_count(self, memory_name: str) -> int:
        """How many rows of ``memory_name`` this run has touched — the headroom a
        BOUNDED read has to over-fetch by so its window provably contains the answer
        once the buffer is merged in.  Counts both ends of a transfer, since either can
        change what that memory's window holds."""
        return sum(1 for pair in self._transfers() if memory_name in pair)

    def net_change(self, memory_name: str) -> int:
        """How much this run's staging changes ``memory_name``'s entry count — what a
        COUNT in SQL is short by: what arrived, less what left."""
        transfers = self._transfers()
        arrived = sum(1 for _, destination in transfers if destination == memory_name)
        left = sum(1 for origin, _ in transfers if origin == memory_name)
        return arrived - left

    # ── Landing (or not) ─────────────────────────────────────────────────────

    def touched(self) -> set[str]:
        """Every memory this run staged a change to — who to announce the landing to.

        BOTH ends of every transfer: an entry that moved changed two memories, and the
        one it left has as much to refresh as the one it arrived in.
        """
        return {name for pair in self._transfers() for name in pair if name is not None}

    def commit(self, engine) -> set[str]:
        """Land everything staged, in ONE short transaction, and return the memories
        that changed.

        Short deliberately: the cycle that staged these ran for minutes and SQLite takes
        one writer at a time, so the alternative — a transaction open for the length of
        the run — would stall every chat turn behind a background poll.

        A row that vanished under the cycle (chat deleted it mid-run) is logged and
        skipped: re-creating it would resurrect something the user removed, and doing so
        silently is the failure this whole mechanism exists to stop.
        """
        touched = self.touched()
        if not self:
            return touched
        with Session(engine) as session:
            for row in self._inserts:
                session.add(MemoryEntry(**row.model_dump(exclude={"id"})))
            for row_id, staged in self._rewrites.items():
                self._land_rewrite(session, row_id, staged)
            for row_id, staged in self._removals.items():
                self._land_removal(session, row_id, staged)
            session.commit()
        self.discard()
        return touched

    @staticmethod
    def _land_rewrite(session: Session, row_id: int, staged: MemoryEntry) -> None:
        live = session.get(MemoryEntry, row_id)
        if live is None:
            logger.warning(_VANISHED, row_id, staged.memory_name)
            return
        for field in _REWRITABLE_FIELDS:
            setattr(live, field, getattr(staged, field))
        session.add(live)

    @staticmethod
    def _land_removal(session: Session, row_id: int, staged: MemoryEntry) -> None:
        live = session.get(MemoryEntry, row_id)
        if live is None:
            logger.warning(_VANISHED, row_id, staged.memory_name)
            return
        session.delete(live)

    def discard(self) -> None:
        """Throw the staged mutations away — the unhealthy end, and the whole point:
        the next cycle runs against the state this one started from."""
        self._inserts = []
        self._rewrites = {}
        self._removals = {}
        self._origins = {}


class EntryWriter(ABC):
    """WHERE one entry mutation goes — the two-phase seam, declared as two writers
    rather than as a flag every mutation method checks.

    A ``Memory`` states what it is doing (insert this row, rewrite these, remove that)
    and the writer decides whether it lands now or waits for the run to end, so the
    write path reads the same either way and no mutation can accidentally take the
    other branch.
    """

    @abstractmethod
    def insert(self, session: Session, row: MemoryEntry) -> None:
        """Add a new row."""

    @abstractmethod
    def rewrite(self, session: Session, row: MemoryEntry) -> None:
        """Record ``row``'s already-mutated field values as its new value."""

    @abstractmethod
    def remove(self, session: Session, row: MemoryEntry) -> None:
        """Delete an existing row."""

    @abstractmethod
    def settle(self, session: Session, rows: Sequence[MemoryEntry] = ()) -> None:
        """Close the batch — landing it, and reloading ``rows`` so the ids the store
        generated are readable once the session is gone."""


class LiveEntryWriter(EntryWriter):
    """The ordinary writer: every mutation lands as it happens.

    Chat's writer and every non-run caller's.  A chat turn is not must-act and its turn
    IS its conclusion, so there is nothing for it to hold back — its writes are
    byte-identical to what they were before two-phase existed.
    """

    def insert(self, session: Session, row: MemoryEntry) -> None:
        session.add(row)
        session.flush()

    def rewrite(self, session: Session, row: MemoryEntry) -> None:
        session.add(row)

    def remove(self, session: Session, row: MemoryEntry) -> None:
        session.delete(row)

    def settle(self, session: Session, rows: Sequence[MemoryEntry] = ()) -> None:
        session.commit()
        for row in rows:
            session.refresh(row)


class StagedEntryWriter(EntryWriter):
    """The two-phase writer: every mutation is recorded in the run's buffer and nothing
    reaches the store until the run ends healthily.

    ``settle`` deliberately does nothing — the batch is not this call's to land, and the
    session it was handed never had anything added to it.
    """

    def __init__(self, pending: PendingEntryWrites) -> None:
        self._pending = pending

    def insert(self, session: Session, row: MemoryEntry) -> None:
        self._pending.insert(row)

    def rewrite(self, session: Session, row: MemoryEntry) -> None:
        self._pending.rewrite(row)

    def remove(self, session: Session, row: MemoryEntry) -> None:
        self._pending.remove(row)

    def settle(self, session: Session, rows: Sequence[MemoryEntry] = ()) -> None:
        """Nothing lands here; the cycle's conclusion is what settles the batch."""
