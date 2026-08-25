"""One collector cycle's UNDO journal — what it would take to put the collection back
the way the cycle found it (#1936).

A cycle that browsed, WROTE the changed value and then died before its close consumed
the change and delivered nothing: the retry re-observed the same page, re-wrote the
identical value, the change-gate read ``KEY_EXISTS_UNCHANGED`` → write-gate STOP → the
user was never told about a change that really happened.  Code-owner ruling: a partial
run must not persist state later cycles have to reason about.

The writes themselves are unchanged — live, through the existing chokepoint.  The only
addition is this record of what each touched key held BEFORE the cycle first touched it,
kept in memory for the span of one cycle.  A healthy end drops it; an unhealthy one
replays it in one short transaction, and the retry meets the world the dead cycle did.

**FIRST TOUCH PER KEY WINS**, which is what makes N touches revert to the true pre-run
state rather than to whatever the cycle's second-last write left.

Three consequences the code owner WAIVED, recorded here rather than defended against:

* A concurrent reader mid-cycle sees the cycle's in-progress writes.  Accepted.
* A process that CRASHES mid-cycle leaves the partial state behind — the journal is in
  memory, so it dies with the process.  Accepted.
* A chat turn writing the SAME collection mid-cycle can be clobbered by a revert: the
  journal is bound to a collection, not to a run, so it captures whatever touches that
  collection while the cycle is bound and puts every captured key back.  Rare (a chat
  turn writing the very collection a cycle is running, in the seconds it runs), and the
  alternative — threading run identity through every mutation — is machinery this
  design exists to avoid.

One mutation is out of reach rather than waived: an entry MOVED to another collection
leaves the journal's scope with it, so a move is not undone.  No collector cycle can
make one — the lifecycle tier that reaches it is absent from the cycle's tool surface
(#1556) — and reaching it would mean journalling a second collection the cycle is not
bound to.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select

from penny.database.models import MemoryEntry


class EntryPrior(BaseModel):
    """One stored entry exactly as it stood before the cycle touched its key.

    Its FIELDS are the declaration of what is restorable, named exactly as the
    ``memory_entry`` columns they come from, so the capture, the restore and the re-mint
    all read the one list instead of three that can drift: the value, who wrote it, both
    embeddings, both run-id stamps (#1560 — the change-gate's own baseline refresh
    advances ``last_written_by_run_id``, and restoring the content while leaving the
    stamp would leave the row citing a run that changed nothing), and ``created_at``
    (the ordering column every recency read uses).  ``id`` is what a revert matches on,
    so a restored row keeps its identity rather than being re-minted under a new one.

    ``memory_name`` and ``key`` are deliberately absent: they are the journal's own key,
    and an entry that changed either is a different entry.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    content: str
    author: str
    key_embedding: bytes | None
    content_embedding: bytes | None
    created_at: datetime
    created_by_run_id: str | None
    last_written_by_run_id: str | None

    def restore(self, row: MemoryEntry) -> None:
        """Put a row that is still there back to this state."""
        for field, value in self.model_dump().items():
            setattr(row, field, value)

    def recreate(self, memory_name: str, key: str) -> MemoryEntry:
        """Re-mint a row the cycle DELETED, under its original id."""
        return MemoryEntry(memory_name=memory_name, key=key, **self.model_dump())


def entry_prior(row: MemoryEntry) -> EntryPrior:
    """One stored row's restorable state.

    A row read from the database always carries an id, and that id is what the revert
    matches on — so one without is a programming error rather than a case to absorb.
    """
    if row.id is None:
        raise ValueError(f"Cannot journal an unstored entry of '{row.memory_name}'")
    return EntryPrior.model_validate(row, from_attributes=True)


class CycleJournal:
    """The pre-run state of every key one cycle has touched in ONE collection.

    Summary of the surface: ``capture`` records a key's prior state (first touch wins),
    ``touched`` says whether there is anything to undo, and ``revert`` puts every
    captured key back — the whole class, in that order.
    """

    def __init__(self, collection: str) -> None:
        self.collection = collection
        self._priors: dict[str, tuple[EntryPrior, ...]] = {}

    def capture(self, key: str, rows: list[MemoryEntry]) -> None:
        """Record what ``key`` held before this cycle first touched it.

        An empty ``rows`` is the honest record of a key that did not exist — reverting
        it means deleting whatever now stands there.  A key already captured is left
        alone, so the journal always describes the state the CYCLE started from.
        """
        if key in self._priors:
            return
        self._priors[key] = tuple(entry_prior(row) for row in rows)

    @property
    def touched(self) -> bool:
        """Did this cycle mutate anything at all?"""
        return bool(self._priors)

    def revert(self, session: Session) -> int:
        """Put every captured key back, returning how many keys were undone.

        TWO PASSES, because ``memory_entry.id`` is a plain SQLite rowid: deleting a row
        frees its id for the very next insert, so a cycle that deleted one key and then
        added another can leave the added row sitting on the exact id a removed one has
        to come back under.  Every row the cycle added is therefore gone — flushed —
        before a single re-mint is queued.

        Still ONE short transaction, never a run-long one: SQLite has a single writer and
        a cycle runs for minutes, so holding the write lock open for the cycle would
        block chat behind it.  The caller commits.
        """
        removed = {
            key: self._restore_survivors(session, key, priors)
            for key, priors in self._priors.items()
        }
        session.flush()
        for key, priors in removed.items():
            for prior in priors:
                session.add(prior.recreate(self.collection, key))
        return len(self._priors)

    def _restore_survivors(
        self, session: Session, key: str, priors: tuple[EntryPrior, ...]
    ) -> tuple[EntryPrior, ...]:
        """Restore every row of ``key`` still standing, delete every row the cycle added,
        and hand back the priors that have to be re-minted.

        Matched by id, so a key the cycle rewrote in place is restored where it is rather
        than deleted and re-inserted, and a row it added under a fresh id goes."""
        remaining = {prior.id: prior for prior in priors}
        stored = session.exec(
            select(MemoryEntry).where(
                MemoryEntry.memory_name == self.collection, MemoryEntry.key == key
            )
        ).all()
        for row in stored:
            prior = remaining.pop(row.id, None) if row.id is not None else None
            if prior is None:
                session.delete(row)
            else:
                prior.restore(row)
                session.add(row)
        return tuple(remaining.values())
