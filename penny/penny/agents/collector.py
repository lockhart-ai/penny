"""Collector — single dispatcher agent for per-collection extraction.

One ``Collector`` instance runs in the background.  Each cycle it picks
the most-overdue ready collection from ``memory`` (where
``extraction_prompt IS NOT NULL`` and the collection's RRULE ``schedule``
has an occurrence at or before now that its last run didn't already
cover), binds itself to that target, runs the agent loop with the target's
extraction prompt as instructions and a tool surface scoped to writes
against that collection only, then stamps ``last_collected_at = now``.

The stamp CONSUMES the occurrence, so it is withheld from a cycle that did
not spend one (#1935): a cycle preempted by a foreground message, or one
whose model call died, having changed nothing, leaves the occurrence due and
is re-attempted on the next tick — bounded, so a collection that fails that
way every time still stops re-attempting and waits for its next occurrence.

The schedule runs as stated (#1857): there is no interval to widen and no
auto-throttle to widen it — rate protection lives in the send cooldown,
where it belongs.

Readiness has a second gate beyond the schedule: a *log-driven* collection
(one that reads a log via ``log_read``, leaving a read cursor) is skipped
without entering the model whenever every one of its live input logs is
caught up — ``head <= last_read_at``.  The cursors a collection already
holds are its declared inputs, so no spec is needed; a cursor whose log the
prompt no longer names is pruned so it can't keep gating.  Generative /
collection-driven collections (no log cursor) run on the schedule alone.

Dispatcher pattern (vs. one stateful agent per collection):
  - No agent registry to keep in sync with the DB; reading the DB each
    cycle IS the source of truth.
  - Hot-add for free — chat creates a new collection mid-session, the
    next dispatcher tick picks it up.
  - Per-collection cadence respected naturally via the readiness check.
  - Log read cursors partition per collection: ``get_tools`` keys the
    memory tools on the bound collection name (``_memory_scope()``), not
    the constant ``"collector"`` identity.  Keying on the identity would
    collapse every collection that reads the same log (e.g. the many that
    read ``user-messages``) onto one shared cursor — whichever ran first
    would consume the new entries and starve the rest.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from penny.agents.base import BackgroundAgent
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.agents.self_state import standing_rules_section
from penny.config import Config
from penny.constants import (
    CYCLE_END_RETRY_REASONS,
    HEALTHY_CYCLE_ENDS,
    WRITE_GATE_STOP_REASONS,
    CycleEnd,
    CycleTrigger,
    MutationActor,
    PennyConstants,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory.types import MemoryNotFoundError
from penny.database.models import MemoryEntry, MemoryRow
from penny.datetime_utils import format_log_timestamp, stored_as_utc, user_timezone_name
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.notification import (
    NOTIFICATION_NOTES,
    NOTIFIED_CYCLE_ENDS,
    CollectorNotifier,
    NotificationOutcome,
)
from penny.prompts import Prompt
from penny.text_validity import check_extraction_prompt
from penny.tools.collection_instantiation import next_occurrence
from penny.tools.memory_tools import CursorReadTool, format_entries

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# The composed prompt's holdings block (#1914) — WHAT THE COLLECTION HOLDS at the moment
# the cycle starts, rendered beside its recent runs.  A cycle can only act on what it is
# presented, and until now nothing on the surface said what was already stored: a routine
# re-observing the same thing each cycle had no way to see the key it wrote under last
# time, so it invented a fresh one and the write gate read NEW_KEY where the value was
# in fact unchanged (measured: 15 of 25 quiet cycles).  Presentation only — nothing here
# constrains what a routine may be or which keys it may use; a genuinely many-keyed
# routine (a dated digest) reads its own past keys and carries on.
#
# The entries render through the SAME ``format_entries`` the read tools use, so a key
# read ambiently is in the invocation form a ``key=`` argument takes — copied, never
# guessed (n≤1).
_HOLDINGS_HEAD = "## What this collection holds (newest first)"
_HOLDINGS_LEAD = "The entries it holds right now, and the key each one is stored under."
_HOLDINGS_EMPTY = "It holds nothing yet."
_HOLDINGS_MORE = "{count} older {noun} not shown."
_HOLDINGS_GONE = (
    "Collection '%s' vanished between binding the cycle and composing its prompt — the "
    "cycle runs without the block stating what it holds."
)

# The "what you last told the user" line (#2385) — what THIS mechanism last sent, so the
# close's ``notify`` choice reads against the user's actual state of knowledge rather
# than a recollection.  Anchored to the same emission provenance the self-state render
# reads (#1568): the outgoing message the drainer stamped with this collection.
_LAST_TOLD_HEAD = "## What you last told the user"
_LAST_TOLD_NEVER = "You have never sent them anything about this collection."
_LAST_TOLD_LINE = 'The last thing you told the user: "{snippet}" ({when})'


class Collector(BackgroundAgent):
    """Single dispatcher agent — picks the most-overdue ready collection per cycle."""

    name = PennyConstants.COLLECTOR_AGENT_NAME

    # Runtime rules every collector cycle gets, appended to whatever
    # extraction_prompt the chat agent (or migration) wrote on the
    # ``memory`` row.  These are *behaviour* invariants — not authoring
    # guidance — so they're attached structurally rather than relied on
    # the prompt-writer to include.  Penny dropped the provenance line
    # in the first prague-highlights prompt she wrote even though the
    # chat-facing guide called for it; structural enforcement is the
    # fix.  Class-scoped so subclasses (none yet) could override if a
    # different runtime contract emerged.
    #
    # Shrunk to what still applies (#1911).  Two rules went with the framework taking
    # their job: "end every cycle with done()" (the cycle ends when the program's own
    # calls are covered — a read, not a move the model makes) and the QUIET-cycle rule
    # (which existed to stop the model forcing a write when the tail demanded one).
    # Nothing here mentions telling the user: that is not part of a cycle any more.
    #
    # Every rule here names a tool on the fixed naive-baseline surface (#2455 — the
    # chat tools minus the lifecycle tier), so the head always renders; a rule naming
    # a tool outside that surface would be an instruction that cannot be followed —
    # the same n≤1 reachability bug a result message naming an absent sibling would
    # be, and answered the same way.
    _RUNTIME_RULES_HEAD = "## Runtime rules (always apply)"
    _RUNTIME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "- Single batched `collection_write(entries=[...])` per cycle — not one call "
            "per entry.",
            ("collection_write",),
        ),
        (
            "- To correct an entry: `collection_write` it under the key it is stored "
            "under — the store keeps one value per key and replaces the stored content "
            "in place. If the entry should be removed instead: "
            "`collection_remove(key=<key>)`.",
            ("collection_write", "collection_remove"),
        ),
        (
            "- Cite only what you actually browsed this cycle.  Never invent a URL to "
            'populate a "Source:" field — if no real source was fetched, omit the field.',
            ("browse",),
        ),
        (
            "- Don't dedup manually — the store rejects duplicates on write automatically.",
            ("collection_write",),
        ),
    )

    def __init__(
        self,
        model_client: LlmClient,
        db: Database,
        config: Config,
        *,
        embedding_model_client: LlmClient,
    ) -> None:
        super().__init__(
            model_client=model_client,
            db=db,
            config=config,
            embedding_model_client=embedding_model_client,
        )
        # Set per-cycle inside ``_execute_cycle``.  The scheduler runs cycles
        # one at a time, but on-demand triggers (chat's extraction-prompt test
        # tool, the addon's "run extractor" button) call ``run_for`` off the
        # scheduler's cadence.  ``_cycle_lock`` serializes every cycle so
        # ``_current_target`` is never clobbered by an overlapping run.
        self._current_target: MemoryRow | None = None
        self._cycle_lock = asyncio.Lock()
        self._notifier = CollectorNotifier(db, embedding_model_client, model_client)
        # How many times each collection's CURRENT occurrence has been re-attempted
        # (#1935) — the bound on ``_settle_occurrence``'s deferral, cleared the moment
        # a cycle settles.  In memory rather than on the row because the exit it
        # counts leaves nothing durable to count FROM: a cycle cancelled seconds in
        # writes no ``promptlog`` row, so the ledger cannot see it happened, and the
        # count exists only to bound one burst — a restart is itself a fresh start.
        # Keyed by (collection, trigger) since #1939: the scheduler's cadence and the
        # user's "run this now" are two different things attempting the same fire, and
        # a user's clicks must not leave the schedule's own attempt with nothing.
        self._retry_attempts: dict[tuple[str, CycleTrigger], int] = {}

    async def execute(self) -> bool:
        self._retire_expired()
        target = self._next_ready_collection()
        if target is None:
            return False
        success, _ = await self._execute_cycle(target, trigger=CycleTrigger.CADENCE)
        return success

    def _retire_expired(self) -> None:
        """Pre-dispatch sweep: system-archive every collection whose ``expires_at``
        passed while it wasn't running (e.g. Penny was down past the expiry, so no
        cycle's post-cycle check ever fired, #1562).  Keeps ``_is_ready`` a pure
        predicate — readiness only *skips* an expired collection; this pass turns
        that skip into a visible tombstone rather than silent inertia.  The
        post-cycle ``_archive_if_expired`` handles an expiry that passes mid-cycle;
        this handles one that passed while nothing dispatched it.  ``run_id=None``
        — a while-down retire has no run to attribute."""
        for memory in self.db.memories.list_all():
            self._archive_if_expired(memory, run_id=None)

    async def run_for(self, collection_name: str) -> tuple[bool, str]:
        """Run one extraction cycle for the named collection, bypassing readiness checks.

        The USER's on-demand trigger — the addon surfaces' "run this now" control
        (browser + iOS).  Deliberately not reachable from the model: a chat tool that
        ran a cycle on request let a setup turn immediately execute the job it had
        just scheduled, so the turn both scheduled and ran, which is not what the
        user asked for and is not how the job will behave afterwards.  Returns
        ``(success, message)`` where ``message`` is either an error description
        or the cycle's ``done()`` summary prefixed with "Collector cycle complete.".

        The cycle runs under the ON_DEMAND trigger (#1939), which is what makes the
        difference between this and the cadence path READ rather than re-decided: its
        notification skips the autonomous-send cooldown (someone is waiting for it) and
        its attempts draw on their own retry budget rather than the schedule's.
        """
        collection = self.db.memories.get(collection_name)
        if collection is None:
            return False, str(MemoryNotFoundError(collection_name))
        if collection.archived:
            return (
                False,
                f"Collection '{collection_name}' is archived — restore it first with "
                f"collection_unarchive('{collection_name}'), or test a different collection.",
            )
        if collection.extraction_prompt is None:
            return (
                False,
                f"Collection '{collection_name}' has no extraction_prompt — "
                f"set one with collection_configure before testing.",
            )
        if error := check_extraction_prompt(collection.extraction_prompt):
            return False, error
        return await self._execute_cycle(collection, trigger=CycleTrigger.ON_DEMAND)

    async def _execute_cycle(
        self, collection: MemoryRow, *, trigger: CycleTrigger
    ) -> tuple[bool, str]:
        """Run one full agent cycle bound to ``collection`` with audit cleanup.

        A top-level run starter (#2458): mints the top ``CallLink`` around the
        WHOLE cycle — the loop AND the notifier — because the notifier's draw
        happens after ``_run_cycle`` returns, outside ``Agent.run``'s own
        (narrower) link.  Everything this cycle calls links back to it, passed
        explicitly — no ambient state anywhere.  Owns the ``run_id`` so cleanup
        has the correct UUID even if ``_run_cycle`` raises before any prompts
        are logged, and so neighbouring cycles can't smear into each other's
        promptlog rows.

        The locked run-and-settle work is ``_run_locked_cycle``'s; this is just
        the mint, the delegation, and the report back to the caller.
        """
        run_id = uuid.uuid4().hex
        call_link = CallLink.top(run_id)
        success, response, outcome, reason = await self._run_locked_cycle(
            collection, trigger, run_id, call_link
        )
        return success, self._cycle_message(outcome, reason, response)

    async def _run_locked_cycle(
        self,
        collection: MemoryRow,
        trigger: CycleTrigger,
        run_id: str,
        call_link: CallLink,
    ) -> tuple[bool, ControllerResponse | None, RunOutcome, str]:
        """Run the bound target under the cycle lock, then settle it — the whole
        try/except/finally a cancellation has to see through cleanly.

        Between the loop and the audit stamp sits the one thing a finished cycle can
        still do (#1911): when its program ran to completion and the collection says
        notify, the framework enters the notify micro-context and queues what it
        writes.  It runs INSIDE the lock and BEFORE the outcome is determined, so the
        run record can state which of the four terminal shapes this cycle had —
        aborted, stopped, completed-quiet, completed-notified.

        ``trigger`` says what set this cycle running — the schedule, or the user — and
        is carried, never inferred: the notification reads it to pick its delivery lane
        and the retry budget is keyed on it (#1939).
        """
        success = False
        response: ControllerResponse | None = None
        reads: tuple[CursorReadTool, ...] = ()
        cancelled = False
        notified: NotificationOutcome | None = None
        # Determined ONCE, inside the lock, while the cycle's program is still bound —
        # the outcome is read against that program, so computing it again after the
        # release would read an unbound one and report every cycle as unreadable.
        outcome, reason = RunOutcome.FAILED, ""
        async with self._cycle_lock:
            try:
                # These three steps stay inline and in this order, never folded into
                # one sub-call: if `_notify_if_due` raises (a mid-notify cancellation
                # is real — foreground activity can preempt here same as anywhere
                # else), `success`/`response`/`reads` must already be bound from the
                # prior statements, so the settlement below sees the real run result
                # instead of silently falling back to its defaults.
                self._bind(collection)
                result = await self._run_cycle(run_id, call_link)
                success = result.success
                response = result.response
                reads = result.reads
                notified = await self._notify_if_due(
                    collection, run_id, call_link, response, trigger
                )
            except asyncio.CancelledError:
                # Foreground activity preempted the cycle — tag clearly rather
                # than letting it look like a model crash, then re-raise.
                cancelled = True
                raise
            finally:
                outcome, reason = self._settle_cycle(
                    collection, run_id, trigger, cancelled, response, reads, notified
                )
                self._bind(None)
        return success, response, outcome, reason

    def _settle_cycle(
        self,
        collection: MemoryRow,
        run_id: str,
        trigger: CycleTrigger,
        cancelled: bool,
        response: ControllerResponse | None,
        reads: tuple[CursorReadTool, ...],
        notified: NotificationOutcome | None,
    ) -> tuple[RunOutcome, str]:
        """How the cycle ended, and everything that follows from it (#1936) —
        determined ONCE and read by everything that has to know whether it was real.

        How the cycle ENDED is determined once (``_cycle_end``) and read three
        times: it decides whether the cycle's entry writes SURVIVE, how far its
        read cursors advance, and whether its schedule occurrence is spent
        (``_settle_occurrence``, #1935).  One classification, because they are one
        question — a cycle that never reached its conclusion is thrown out whole and
        retried from the state it started in.  Runs while the program is still
        bound, since the deterministic arm is read off it; the caller unbinds after.
        """
        end = self._cycle_end(cancelled, response, notified)
        landed = self._settle_writes(collection.name, end)
        self.settle_reads(reads, landed)
        deferred = self._settle_occurrence(collection.name, trigger, end, response)
        if cancelled:
            self._tag_promptlog_run_cancelled(run_id)
            return RunOutcome.FAILED, ""
        # One determination of this cycle's outcome, used for the audit log and
        # the promptlog tag alike.
        outcome, reason = self._cycle_result(response, notified, landed)
        self._tag_promptlog_run(run_id, outcome, reason, self._tool_failures(response))
        if not deferred:
            self._retire_if_ended(collection, run_id)
        return outcome, reason

    def _cycle_message(
        self, outcome: RunOutcome, reason: str, response: ControllerResponse | None
    ) -> str:
        """The on-demand test message: STRUCTURAL (#1569) — the run's outcome (or its
        write-gate stop reason) plus the actual tool trace — never a model-authored
        summary, which no longer exists.  Reports the SAME determination the audit
        log was stamped with, rather than a second one.
        """
        detail = reason or outcome.value
        message = f"Collector cycle complete: {detail}"
        tool_trace = self._format_tool_trace(response)
        if tool_trace:
            return f"{message}\n\n{tool_trace}"
        return message

    def _bind(self, collection: MemoryRow | None) -> None:
        """Bind (or release) the cycle's target.

        Opens the cycle's UNDO JOURNAL on the target (#1936) — from here until the
        cycle settles, every entry mutation of that collection first records what the
        key it touches held — and releasing the target closes it."""
        self._current_target = collection
        self.db.memories.begin_cycle(collection.name if collection is not None else None)

    async def _notify_if_due(
        self,
        collection: MemoryRow,
        run_id: str,
        call_link: CallLink,
        response: ControllerResponse,
        trigger: CycleTrigger,
    ) -> NotificationOutcome | None:
        """Tell the user about this cycle, when the cycle itself said there was news
        (#1911, the choice moved onto the close in #2385).

        The trigger is exactly three structural facts: the collection carries
        ``notify``, no step raised a STOP, and the cycle closed with a SUCCESSFUL
        ``done(notify=true)`` — the required argument is the cycle's own recorded
        decision, made against what the collection holds and what the user was last
        told (both rendered on the prompt).  A close whose ``notify`` is false, like a
        collection with notifications off, sends nothing: the decision was made and
        recorded, and the run record shows it.

        The STOP clause stays as a guard for whatever a later STOP set names: a
        STOP means the cycle deliberately ended early, whatever else it had got
        through, and is never a notify.  No write-gate outcome stops a cycle today
        (#2385 emptied the stop-worthy set) — a watch's unchanged re-observation runs
        on to its close, where the quiet cycle is recorded as ``notify=false`` rather
        than being pre-empted by the gate.

        **The #2417 fourth fact — a declared write step that mutated nothing this run
        reads as a STOP — is REMOVED on this branch (#2455 naive baseline)**: it read
        the bound PROGRAM's own calls (``_program_writes``), which no longer exists —
        a job's cycle follows a free-text prompt, not a parsed program, so there is no
        declared-write-step fact left to read.

        ``trigger`` travels through to the queued row: WHETHER to tell the user is the
        facts above, WHEN it reaches them is which lane the message is in (#1939).

        ``None`` when the cycle is not one that notifies, which is the ordinary case.
        """
        if not collection.notify or self._stop_reason(response) is not None:
            return None
        if self._done_notify(response) is not True:
            return None
        return await self._notifier.notify(
            collection, run_id, response.tool_calls, trigger, call_link=call_link
        )

    @staticmethod
    def _done_notify(response: ControllerResponse | None) -> bool | None:
        """The ``notify`` flag off the cycle's successful ``done()`` record, or
        ``None`` when the cycle never closed.

        Read off the ledger's own records like ``_has_done_call``: the first
        non-failed done call — the one that ended the loop — carries the args the
        framework acts on; a later one is unreachable.  The REQUIRED argument is
        validated by :class:`~penny.tools.memory_args.DoneArgs`, so a successful
        record always carries a real boolean; a failed one is not a close and is
        skipped, letting the model's retry supply it.
        """
        if response is None:
            return None
        for record in response.tool_calls:
            if record.tool == PennyConstants.DONE_TOOL_NAME and not record.failed:
                return record.arguments.get("notify")
        return None

    def _retire_if_ended(self, collection: MemoryRow, run_id: str) -> None:
        """Post-cycle retirement — at most one archive.

        A bounded schedule retires a collection after its allotted runs (a ``COUNT=1``
        one-shot archives itself, #1857); failing that, the ``expires_at`` end condition
        retires one whose expiry passed mid-life (#1562).  Both run after the outcome is
        tagged, so this cycle is counted.  ``run_id`` is this cycle's run — recorded as
        the system archive's cause in the mutation ledger (#1560).

        A cancelled cycle never reaches here, and a cycle that KEPT its occurrence is
        skipped for the same reason (#1935): neither spent the fire it was dispatched
        for, so neither may spend a run of a bounded schedule.  Without the skip a
        ``COUNT=1`` one-shot that aborted on a transport wobble would archive itself
        while its occurrence sat due, and an archived collection is never dispatched
        again — the retry could not fire at all.
        """
        if not self._archive_if_run_limit_reached(collection, run_id):
            self._archive_if_expired(collection, run_id)

    # ── How the cycle ended, and what follows from it (#1936) ─────────────

    def _cycle_end(
        self,
        cancelled: bool,
        response: ControllerResponse | None,
        notified: NotificationOutcome | None,
    ) -> CycleEnd:
        """Which of the enumerated ends this cycle had — determined ONCE and read by
        everything that has to know whether the cycle was real (#1936).

        The order mirrors ``_cycle_result``'s, because they are reading the same run:
        a STOP is a deliberate early close whatever else the cycle got through, then the
        ``done()`` close (whose end depends on what telling the user came to), then the
        aborted and unfinished shapes.  A cycle that never produced a response at all —
        cancelled, or an exception out of the loop — never reached any of them.
        """
        if cancelled:
            return CycleEnd.CANCELLED
        if response is None:
            return CycleEnd.UNFINISHED
        if self._stop_reason(response) is not None:
            return CycleEnd.STOPPED
        if self._has_done_call(response):
            return CycleEnd.CLOSED_QUIET if notified is None else NOTIFIED_CYCLE_ENDS[notified]
        if response.abort is not None:
            return CycleEnd.ABORTED
        return CycleEnd.UNFINISHED

    def _settle_writes(self, name: str, end: CycleEnd) -> bool:
        """Keep this cycle's entry writes, or put the collection back — and say which.

        A healthy end drops the undo journal; anything else replays it, so the retry
        runs against the state the cycle started from and the change-gate never reads a
        value that was recorded by a run nobody heard from.  The returned fact is what
        the rest of the settlement means by "this cycle was real": how far its read
        cursors advance, whether it produced work, and whether it spent its occurrence.

        The revert is a real SQLite write and the writer is single, so lock contention
        with a chat turn can raise — and this call sits in the cycle's ``finally``, where
        an escaping exception would skip the occurrence settlement and leave the
        collection due with no attempt counted against it, re-picked on every tick.  A
        failed revert is therefore reported loudly and the cycle's writes are left
        standing, which is exactly the behaviour that stood before this existed.
        """
        try:
            return self.db.memories.end_cycle(revert=end not in HEALTHY_CYCLE_ENDS)
        except SQLAlchemyError:
            logger.exception(
                "Could not revert '%s' cycle's writes — they are left in place and the "
                "collection is retried like any other unhealthy end",
                name,
            )
            return True

    # ── Settling the schedule occurrence (#1935) ──────────────────────────

    def _settle_occurrence(
        self,
        name: str,
        trigger: CycleTrigger,
        end: CycleEnd,
        response: ControllerResponse | None,
    ) -> bool:
        """Consume this cycle's schedule occurrence, or leave it DUE for a retry.

        The stamp is what makes a schedule advance — readiness reads the occurrence
        strictly after ``last_collected_at`` — so stamping a cycle that did nothing
        SKIPS that fire entirely.  Measured live: a chat message cancelled a daily
        job's cycle seconds in, before any model call had come back, and the day's
        occurrence was burned.  A cycle that did not reach an end therefore stays due,
        and the next tick picks it up again (it is still the most overdue).

        The bound is the stamp's ORIGINAL job, kept whole: a collection failing this
        way every time would otherwise re-attempt on every tick forever.  After
        ``COLLECTOR_RETRY_ATTEMPTS`` the occurrence is consumed whatever happened and
        the collection waits for its next one.

        Returns whether the occurrence was KEPT, which the post-cycle retirement reads:
        a cycle that did not spend its fire must not spend a run of a bounded schedule
        either (``_retire_if_ended``).
        """
        reason = self._retry_reason(end, response)
        if reason is not None and self._claim_retry(name, trigger):
            logger.info(
                "Leaving '%s' due rather than consuming its occurrence — %s, and the "
                "cycle changed nothing (%s attempt %d of %d)",
                name,
                reason,
                trigger.value,
                self._retry_attempts[(name, trigger)],
                PennyConstants.COLLECTOR_RETRY_ATTEMPTS,
            )
            # The stamp is this path's only refresh signal, so a cycle that withholds
            # it raises the change itself — otherwise an on-demand "run this now" that
            # died leaves the addon's panel showing nothing at all.
            self.db.memories.notify_changed(name)
            return True
        self._release_retries(name)
        self.db.memories.mark_collected(name)
        return False

    def _claim_retry(self, name: str, trigger: CycleTrigger) -> bool:
        """Take one of this TRIGGER's remaining attempts at ``name``'s occurrence.

        ``False`` once the bound is spent, which is what ends the burst.

        The budgets are PER TRIGGER (#1939).  The scheduler's cadence and the user's
        "run this now" are two different things attempting the same fire, and one
        shared count let three failed clicks leave the day's own scheduled attempt with
        nothing: the next foreground preemption consumed the occurrence on its first
        try, and the job silently skipped its day — the very failure #1935 exists to
        prevent, reached through the door the user was pressing.  Each budget still
        bounds one burst of the same size; there are simply two askers.
        """
        key = (name, trigger)
        attempts = self._retry_attempts.get(key, 0)
        if attempts >= PennyConstants.COLLECTOR_RETRY_ATTEMPTS:
            return False
        self._retry_attempts[key] = attempts + 1
        return True

    def _release_retries(self, name: str) -> None:
        """Drop EVERY trigger's count for ``name`` — the occurrence they bounded is
        spent, so the next fire starts both askers fresh.

        Cleared wholesale rather than per trigger because what a count bounds is
        attempts at ONE occurrence: once any cycle consumes it, there is no longer an
        occurrence for the other trigger's count to be about.
        """
        for key in [key for key in self._retry_attempts if key[0] == name]:
            del self._retry_attempts[key]

    def _retry_reason(self, end: CycleEnd, response: ControllerResponse | None) -> str | None:
        """Why this exit did not spend its occurrence, or ``None`` when it did.

        Two reads, in the order that makes each decisive:

        - **The cycle ENDED** — one of the enumerated healthy ends (#1936): a write-gate
          STOP, or a ``done()`` close that either notified, deliberately declined to, or
          had nothing to notify about.  It did the job the occurrence was for, and its
          writes stand.  Everything else stopped short, and since a short cycle's writes
          are reverted there is nothing left of it to protect — which is exactly why the
          two mechanisms need no special case between them.
        What is left is the unhealthy end's own line, off the declared table — except an
        abort, which names its own terminal cause when the run carries one (#1909).
        """
        if end in HEALTHY_CYCLE_ENDS:
            return None
        if end is CycleEnd.ABORTED and response is not None and response.abort is not None:
            return response.abort.render()
        return CYCLE_END_RETRY_REASONS[end]

    @staticmethod
    def _format_tool_trace(response: ControllerResponse | None) -> str:
        """Numbered list of the cycle's tool calls, each argument rendered whole —
        no cap (#2642)."""
        if not response or not response.tool_calls:
            return ""
        lines = []
        for i, record in enumerate(response.tool_calls, 1):
            args = ", ".join(f"{k}={v}" for k, v in record.arguments.items())
            lines.append(f"{i}. {record.tool}({args})")
        return "\n".join(lines)

    @staticmethod
    def _produced_work(
        response: ControllerResponse | None, notified: NotificationOutcome | None
    ) -> bool:
        """Did this cycle change a collection or reach the user?

        TWO sources, because since #1911 a cycle has two ways of doing something and
        only one of them is a tool call.  The per-call ``ToolCallRecord.mutated`` flag —
        set from each tool's own structured ``ToolResult`` (a row actually written, an
        entry moved/deleted) — is what the MODEL did.  A *successful no-op* (a
        duplicate-rejected write, an update/delete/move on a missing key) carries
        ``mutated=False``, so it correctly reads as idle.

        The other is the notification the FRAMEWORK queued after the cycle (#1914).
        Telling the user is the whole point of a collection that notifies, and no tool
        call carries it any more — so a cycle that ran its routine and told the user
        recorded ``no_work``, the outcome saying nothing happened on the cycle that did
        the one thing it exists for.  Only ``QUEUED`` counts: the two failure outcomes
        put nothing in front of the user, exactly like a muted send's ``mutated=False``.

        This is what splits the run's ``worked`` / ``no_work`` outcome.
        """
        if notified is NotificationOutcome.QUEUED:
            return True
        if response is None:
            return False
        return any(record.mutated for record in response.tool_calls)

    def should_stop_loop(self, records: list[ToolCallRecord]) -> bool:
        """A collector cycle ends on a successful ``done()`` OR a write-gate STOP.

        The base terminator is ``done()`` (#1569, restored #1916); a collector
        additionally honors a STOP carried by a tool result's ``stop_reason`` — a
        deliberate close at the write chokepoint, so no trailing ``done()`` is
        required.  Since #2385 no write-gate outcome ever stamps one — an unchanged
        write no longer ends the cycle; the model closes with ``done(notify=…)`` and
        the quiet decision is its own — so this hook reads only the terminator today,
        and stays so a legacy-stamped row (or a future member) still renders honestly.
        STOP is honored only here (must-act cadence); the chat loop uses the base and
        never stops on a write outcome.

        What this replaced is the coverage read (#1911): the calls the program makes
        were settled before the cycle began, so a cycle that departed from them for a
        good reason had no close available to it at all.  ``records`` stays the run's
        whole ordered history — a terminator called at any point closes the run."""
        return super().should_stop_loop(records) or any(
            record.stop_reason is not None for record in records
        )

    def _cycle_result(
        self,
        response: ControllerResponse | None,
        notified: NotificationOutcome | None,
        landed: bool,
    ) -> tuple[RunOutcome, str]:
        """The cycle's outcome + a STRUCTURAL reason — the single determination read
        by the audit log and the promptlog tag (#1569/#1911).

        Derived from the run's tool calls and the program it was given, never a
        model-authored judgment.  The four terminal shapes each read differently:

        - **stopped** — a write-gate STOP (#1587) closed the cycle at the chokepoint;
          the reason is the declared stop reason, the outcome ``worked``/``no_work`` by
          whether durable state changed.  A stopped cycle never notifies.
        - **done** — the model closed the cycle with the argless sentinel (#1569,
          restored #1916).  The outcome is ``worked``/``no_work`` the same way, and the
          reason carries what telling the user came to when the collection notifies —
          empty otherwise, so the run record's header falls back to the outcome enum.
          A queued notification is also WORK (#1914): it is the one thing a cycle does
          that no tool call records.
        - **aborted** — no close at all: a ``failed`` bail naming what ended the run
          (#1909's abort causes are kept whole).  It is unconditionally ``failed`` since
          #1936: a cycle that never reached a close had its writes reverted, so
          ``incomplete`` — "did real work but never closed" — describes a state a
          collector cycle can no longer be in.

        (``cancelled`` is handled separately — a preempted cycle never reaches here.)

        ``landed`` is whether the cycle's entry writes survived (#1936).  It is what a
        DONE-closed cycle whose notification could not be drawn reads as: that cycle
        closed, so it takes the clean branch, but its writes were reverted with the rest
        of the run — so it reports ``no_work`` rather than claiming a change that no
        longer exists.
        """
        produced = landed and self._produced_work(response, notified)
        clean = RunOutcome.WORKED if produced else RunOutcome.NO_WORK
        stop = self._stop_reason(response)
        if stop is not None:
            return clean, WRITE_GATE_STOP_REASONS[stop]
        if self._has_done_call(response):
            return clean, self._closed_reason(notified)
        return RunOutcome.FAILED, self._unfinished_reason(response)

    @staticmethod
    def _has_done_call(response: ControllerResponse | None) -> bool:
        """Did the cycle close with a successful ``done()``?  Read off the ledger's own
        records, so a done whose args failed validation is not a close."""
        if response is None:
            return False
        return any(
            record.tool == PennyConstants.DONE_TOOL_NAME and not record.failed
            for record in response.tool_calls
        )

    @staticmethod
    def _closed_reason(notified: NotificationOutcome | None) -> str:
        """What a cleanly-closed cycle's record says: what telling the user came to
        when the collection notifies, and nothing at all otherwise — a bare close needs
        no reason, and the run record's header falls back to the outcome enum."""
        if notified is None:
            return ""
        return NOTIFICATION_NOTES[notified]

    @staticmethod
    def _stop_reason(response: ControllerResponse | None) -> WriteGateOutcome | None:
        """The write-gate STOP outcome that ended this cycle, or ``None`` (#1587).

        Reads the structural ``ToolCallRecord.stop_reason`` (set from the tool's
        ``ToolResult.stop``) — the last stop-carrying call, since a STOP is the
        cycle's final action."""
        if response is None:
            return None
        for record in reversed(response.tool_calls):
            if record.stop_reason is not None:
                return record.stop_reason
        return None

    def _archive_if_run_limit_reached(self, collection: MemoryRow, run_id: str) -> bool:
        """Archive a ``max_runs``-bounded collection once it has run its quota.

        ``max_runs`` is the schedule's own ``COUNT=`` lifted onto the row (#1857):
        after that many completed (non-cancelled) cycles the collection has done
        its job — a ``COUNT=1`` one-shot retires itself, and any bounded
        collection stops re-firing.  Archival (not deletion) via the ordinary path
        keeps the row as a visible tombstone in the archived-inclusive catalog
        (#1566); the actor is the scheduler, not the user.  ``None`` = unlimited,
        the ordinary recurring case.  The run count is read from the ledger
        (completed ``promptlog`` runs for this target), never re-decided by the
        model.

        The archive is recorded as a durable mutation event with ``actor=system``
        (no model in the loop) and a policy ``note`` naming its cause — the run
        limit — so "when was this archived, and by what?" is answerable by a read
        even though no run prompt records this system action (#1560).
        """
        if collection.max_runs is None:
            return False
        completed = self.db.messages.count_completed_runs(collection.name)
        if completed < collection.max_runs:
            return False
        note = f"reached run limit ({completed} of {collection.max_runs} completed runs)"
        logger.info("Archiving '%s': %s", collection.name, note)
        self.db.memories.archive(
            collection.name, actor=MutationActor.SYSTEM, run_id=run_id, note=note
        )
        return True

    def _archive_if_expired(self, collection: MemoryRow, run_id: str | None) -> bool:
        """Archive a collection whose ``expires_at`` end condition has passed (#1562).

        Mirrors ``_archive_if_run_limit_reached`` exactly: the same system-actor
        archive path (tombstone in the archived-inclusive catalog #1566, a durable
        mutation event whose ``note`` names the cause), the same "read, never
        re-decided by the model" discipline — the clock, not a judgment, ends the
        watch.  ``None`` ``expires_at`` = no end condition, the ordinary case; an
        already-archived row is left alone (the sweep passes every row).

        ``run_id`` is the cycle that was active when the expiry was noticed
        (post-cycle mid-life retire) or ``None`` when the sweep retires one that
        expired while nothing dispatched it (Penny was down past the expiry — no
        run to attribute).  Returns whether it archived.
        """
        if collection.expires_at is None or collection.archived:
            return False
        expiry = stored_as_utc(collection.expires_at)
        if self.db.clock.now() < expiry:
            return False
        note = f"reached expiry ({expiry.isoformat()})"
        logger.info("Archiving '%s': %s", collection.name, note)
        self.db.memories.archive(
            collection.name, actor=MutationActor.SYSTEM, run_id=run_id, note=note
        )
        return True

    # ── Per-cycle audit (on the promptlog run itself) ─────────────────────

    def _tag_promptlog_run(
        self, run_id: str, outcome: RunOutcome, reason: str, tool_failures: int
    ) -> None:
        """Stamp the cycle outcome + its STRUCTURAL reason onto the matching
        promptlog run (#1569 — ``reason`` is a write-gate stop reason or the
        no-``done()`` close reason, empty for a clean ``done()`` close; never a
        model summary).

        Drives the outcome badge in the addon's prompts tab plus ``tool_failures``
        (the count of failed tool calls), which the run-health classifier reads to
        flag a tool-failure spiral.  (The run's collection is already on every
        prompt via the write-time ``run_target`` stamp.)  ``run_id`` is the
        caller's UUID for this cycle; ``set_run_outcome`` is a no-op if no
        promptlog rows exist for it (the cycle raised before the loop ever logged
        a prompt).
        """
        self.db.messages.set_run_outcome(run_id, outcome.value, reason, tool_failures)

    @staticmethod
    def _tool_failures(response: ControllerResponse | None) -> int:
        """How many tool calls in this cycle returned a failure.

        Reads the authoritative per-call ``ToolCallRecord.failed`` flag (set from
        each tool's structured ``ToolResult.success``) — the same records
        ``_produced_work`` scans for ``mutated``.  Persisted so the classifier
        never has to guess a failure from framed tool-result text.
        """
        if response is None:
            return 0
        return sum(1 for record in response.tool_calls if record.failed)

    def _tag_promptlog_run_cancelled(self, run_id: str) -> None:
        """Stamp a cycle that was cut off by foreground activity.

        Cancellation isn't a failure of the cycle's logic — it's the scheduler
        making room for a user message — so it gets its own ``cancelled``
        outcome rather than ``failed``, keeping it out of the addon's
        failure-rate budget (and the throttle ignores it).
        """
        self.db.messages.set_run_outcome(
            run_id,
            RunOutcome.CANCELLED.value,
            "cancelled by foreground activity",
        )

    def _unfinished_reason(self, response: ControllerResponse | None) -> str:
        """The structural reason a cycle ended without finishing its program —
        distinguish a run the model call KILLED from actually hitting the step cap from
        the model trailing off (all failures, but for different reasons).

        A run that aborted on a model call carries the abort's own facts (#1909): which
        step died, the tool the last successful step ran, and the error's class and
        message.  That case used to fall through to the generic no-``done()`` line, so
        the whole class — a failed call writes no ``promptlog`` row — was diagnosable
        only by exclusion.  ``ran_out_of_steps`` is set structurally (#2370) only on
        the real cap, never read off ``answer``'s text — a background cycle never
        narrates one (``BackgroundAgent._draw_narration`` skips the draw), but the
        field is the same one a narrating agent sets too; anything else is an early
        give-up without reporting an outcome.

        """
        if response is None:
            return "no response from cycle"
        if response.abort is not None:
            return response.abort.render()
        if response.ran_out_of_steps:
            return "max steps exceeded — the program was left unfinished"
        return "cycle ended with the program unfinished"

    # ── Per-cycle prompt + tool scope ─────────────────────────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """System prompt for the bound target — re-fetched each cycle.

        Reading from the DB instead of caching means a chat-side
        ``collection_configure`` call that changes ``extraction_prompt`` is
        picked up on the very next collector cycle, no restart needed.

        The two trailing blocks are the cycle's own STATE, read the same way: what the
        collection holds now (#1914) and what this collector's recent runs did (#1569).
        Holdings first — the entries are what the prompt acts ON, so they sit next to
        it, and the run outcomes close the prompt as they always have.

        The standing rules render between the job's own instructions and the state
        blocks (#2633): the same shared section the chat self-state header shows, so a
        rule about how she speaks or what she uses holds in every cycle without a
        lookup.
        """
        target = self._require_target()
        fresh = self.db.memories.get(target.name) or target
        return (
            self._compose_prompt(fresh)
            + "\n\n"
            + standing_rules_section(self.db)
            + self._holdings_section(fresh.name)
            + self._run_history_section(fresh.name)
        )

    def _holdings_section(self, target_name: str) -> str:
        """A trailing block of what the bound collection HOLDS right now — its entries,
        newest first, each with the key it is stored under (#1914).

        Presentation, not policy: the block states what is there and nothing about what
        to do with it, so a routine that re-observes one thing sees the key it wrote
        under last time and a routine that files a fresh entry each run sees the ones it
        already filed.  Neither is prescribed; both are now reading their own state
        instead of guessing at it.

        The window is bounded in the QUERY and the remainder is a COUNT, so a deep
        collection is never materialized to render twenty lines of it.  A collection
        that vanished between binding and composing has nothing to state: the block is
        absent and the prompt is byte-identical to what it was, said out loud rather
        than swallowed.
        """
        memory = self.db.memory(target_name)
        if memory is None:
            logger.warning(_HOLDINGS_GONE, target_name)
            return ""
        limit = PennyConstants.COLLECTOR_HOLDINGS_LIMIT
        body = self._holdings_body(memory.read_latest(limit), memory.entry_count())
        return f"\n\n{_HOLDINGS_HEAD}\n{body}"

    @staticmethod
    def _holdings_body(newest: list[MemoryEntry], total: int) -> str:
        """The holdings block's body: the plain empty line, or the newest entries plus
        an honest count of the ones the budget left out.

        The overflow is a stated number rather than a silent cut — a collection deep
        enough to overflow is one whose older entries the cycle can still reason about
        the existence of."""
        if not newest:
            return _HOLDINGS_EMPTY
        shown = f"{_HOLDINGS_LEAD}\n{format_entries(newest)}"
        rest = total - len(newest)
        if rest <= 0:
            return shown
        noun = "entry" if rest == 1 else "entries"
        return f"{shown}\n{_HOLDINGS_MORE.format(count=rest, noun=noun)}"

    def _run_history_section(self, target_name: str) -> str:
        """A trailing block of this collector's own recent run outcomes (newest
        first) so each cycle knows what its prior invocations did, and — since
        #2385 — what it last TOLD the user through this mechanism.

        The runs block is empty when disabled (``COLLECTOR_RUN_HISTORY`` = 0) or
        there's no history yet; the last-told line renders always, because the
        close's ``notify`` choice needs it in both directions — with a prior send to
        compare against, and the plain "never" when there is none, which is its own
        state rather than an absence.  Each run line is the run's STRUCTURAL outcome
        — its outcome enum, or the write-gate stop reason — generated from the ledger
        (#1569), never a model-authored ``done()`` summary (there is none).  Framed
        as reference, not instruction: it tells the collector what it did, to avoid
        repeating work, without feeding its own past prose back into the next cycle.
        """
        limit = int(self.config.runtime.COLLECTOR_RUN_HISTORY)
        outcomes = self.db.messages.recent_run_outcomes(target_name, limit)
        blocks = []
        if outcomes:
            lines = "\n".join(
                f"{index}. [{format_log_timestamp(when)}] {outcome}"
                for index, (when, outcome) in enumerate(outcomes, start=1)
            )
            blocks.append(
                "## Your recent runs (newest first)\n"
                "What your previous cycles did, and when — context to avoid repeating "
                "work or re-sending, not an instruction to repeat.\n"
                f"{lines}"
            )
        blocks.append(self._last_told_line(target_name))
        return "\n\n" + "\n\n".join(blocks)

    def _last_told_line(self, target_name: str) -> str:
        """What this mechanism last sent the user, or the plain never-sent line
        (#2385).

        Read off the emission provenance the drainer stamps (#1568) — the newest
        outgoing message whose mechanism is this collection — so the answer is a
        read, not a judgment.  Whitespace is collapsed so a multi-line body reads
        as one line, and the whole thing renders — no cap (#2642)."""
        send = self.db.messages.latest_mechanism_send(target_name)
        if send is None:
            return f"{_LAST_TOLD_HEAD}\n{_LAST_TOLD_NEVER}"
        snippet = " ".join(send.content.split())
        when = format_log_timestamp(send.timestamp)
        line = _LAST_TOLD_LINE.format(snippet=snippet, when=when)
        return f"{_LAST_TOLD_HEAD}\n{line}"

    @classmethod
    def _compose_prompt(cls, target: MemoryRow) -> str:
        """Frame the stored ``extraction_prompt`` with target identity + the
        close-with-``done()`` instruction + runtime rules (#2455 naive baseline; was
        #1557/#1907).

        The job's own instructions are the free text the model wrote when it set the
        job up — stored verbatim, run verbatim, with no rendering seam in between.
        There is no routine/values block any more: there is no skill, so there is
        nothing to name beyond the prompt itself.

        The runtime-rules tail is appended structurally — not relayed through
        Penny when she authors the prompt.  This guarantees the rules apply on
        every cycle regardless of how the prompt was written.  The chat-facing
        ``collection_configure`` description only carries authoring-shape guidance;
        the runtime invariants live here.
        """
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n"
            "\n"
            f"{target.extraction_prompt}\n"
            f"\n{Prompt.COLLECTOR_DONE_INSTRUCTION}\n\n"
            f"{cls._runtime_rules()}"
        ).rstrip("\n")

    @classmethod
    def _runtime_rules(cls) -> str:
        """The runtime rules every cycle's surface can carry out.

        A rule renders only when every tool it names is present on the fixed
        naive-baseline surface (the chat tools minus the lifecycle tier, #2455) —
        which is every rule declared here, so the head always renders."""
        lines = [line for line, _named in cls._RUNTIME_RULES]
        if not lines:
            return ""
        return "\n".join([cls._RUNTIME_RULES_HEAD, "", *lines])

    # The cycle's tool surface is the inherited ``BackgroundAgent.get_tools()``
    # default (#2455 naive baseline; was #1911's program-scoped surface, now gone
    # with the parsed program): the chat tool set minus the lifecycle tier
    # (``_include_lifecycle_tools`` below, False for a collector) plus ``done``.
    # A job's cycle gets exactly this — never widened to email, plugin, or
    # notification-mute tools, and never narrowed to a parsed program's own calls,
    # since a free-text prompt has no calls to parse.

    def _memory_scope(self) -> str:
        """Pin entry mutations to the bound target collection."""
        return self._require_target().name

    def _include_lifecycle_tools(self) -> bool:
        """A cadence-fired collector run never reshapes the registry (#1556).

        Overrides the ``Agent`` default: the create / update / merge / archive /
        unarchive / log_create tier is absent from a collector's surface, so a
        background poll cannot create, reconfigure, merge, or archive a mechanism
        — the mid-poll config mutation and create-instead-of-delete slips are
        structurally impossible, not just discouraged.
        """
        return False

    def _require_target(self) -> MemoryRow:
        if self._current_target is None:
            raise RuntimeError(
                "Collector tool surface accessed outside an execute() cycle "
                "— self._current_target is None"
            )
        return self._current_target

    # ── Dispatcher selection ──────────────────────────────────────────────

    def _next_ready_collection(self) -> MemoryRow | None:
        """Pick the most-overdue ready collection, or None if all caught up."""
        now = self.db.clock.now()
        ready = [m for m in self.db.memories.list_all() if self._is_ready(m, now)]
        if not ready:
            return None
        return min(ready, key=self._overdue_sort_key)

    def _is_ready(self, memory: MemoryRow, now: datetime) -> bool:
        if memory.archived or memory.extraction_prompt is None:
            return False
        if check_extraction_prompt(memory.extraction_prompt) is not None:
            logger.warning(
                "Skipping collection '%s': extraction_prompt too short (%d chars, minimum 25) "
                "— update it via collection_configure to enable collection",
                memory.name,
                len(memory.extraction_prompt),
            )
            return False
        schedule = memory.schedule
        if schedule is None:
            logger.warning(
                "Skipping collection '%s': no schedule set — set one via collection_configure "
                "to enable collection",
                memory.name,
            )
            return False
        # End condition (#1562): once ``expires_at`` has passed, the watch is
        # over — it never starts another cycle.  A PURE skip here keeps
        # readiness side-effect-free; the dispatcher's ``_retire_expired`` sweep
        # turns the skip into a visible system archive (the codebase separates
        # readiness from archival).
        if memory.expires_at is not None and now >= stored_as_utc(memory.expires_at):
            return False
        if not self._schedule_due(memory, schedule, now):
            return False
        # The schedule says it's time.  Now the cursor gate: a log-driven
        # collection caught up on every live input is skipped without entering the
        # model — the watermark, not the clock, says there's work.
        return self._input_pending(memory) is not False

    def _schedule_due(self, memory: MemoryRow, schedule: str, now: datetime) -> bool:
        """Whether the collection's RRULE has come round again (#1857).

        Ready iff ``now`` has reached the occurrence the collection is WAITING ON.  On
        the first run that is the rule's very FIRST occurrence — a rule with no
        ``DTSTART`` starts at ``created_at``, so a fresh collection is due right away
        exactly as a fresh collector always has been, and a rule whose own ``DTSTART``
        is already past is due right away too (it is a schedule that came round while
        nothing was watching, not one to skip).  After that it is the next occurrence
        strictly after the last run.  A rule with no occurrence left (a spent ``COUNT=``,
        a passed ``UNTIL=``) is never ready again; the post-cycle quota archive retires
        it.

        WHERE each occurrence falls on the clock is ``next_occurrence``'s (#1932): a
        rule states an hour and never a zone, so it is read on the USER'S clock — the
        hour a schedule states is the hour they said.  The profile zone is fetched here
        and passed to the pure function, which never reaches into the database.

        An unreadable rule is a stored value the parse gate already refused, so it can
        only mean a hand-edited row: it is logged and skipped rather than crashing the
        dispatcher for every other collection (visible degradation over a silent stall).
        """
        try:
            next_fire = next_occurrence(
                schedule,
                memory.created_at,
                user_timezone_name(self.db),
                after=memory.last_collected_at,
            )
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Skipping collection '%s': schedule %r isn't a readable rule (%s) — "
                "set a valid one via collection_configure",
                memory.name,
                schedule,
                exc,
            )
            return False
        return next_fire is not None and now >= next_fire

    # ── Cursor gate (skip-when-no-new-input) ──────────────────────────────

    def _input_pending(self, memory: MemoryRow) -> bool | None:
        """Pre-model gate signal, read from the collection's own read cursors.

        ``True`` — at least one live input log has entries past its cursor: run.
        ``False`` — every live cursor is caught up: skip, don't enter the model.
        ``None`` — no live cursor at all: a generative or collection-driven
        collection (browses, picks from another collection) with no log to gate
        on; not gate-eligible, so its schedule alone decides.

        The cursors a collection already holds *are* its declared inputs — no
        separate spec.  ``commit_pending`` advances a cursor to the newest entry
        actually consumed, so ``head > last_read_at`` means unread input exists.
        This INFERRED gate is untouched by the schedule collapse (#1857): the
        declared-input ``on advance of`` form is gone, this one stays.
        """
        live = self._live_cursors(memory)
        if not live:
            return None
        return any(self._log_has_new(log_name, position) for log_name, position in live)

    def _live_cursors(self, memory: MemoryRow) -> list[tuple[str, datetime]]:
        """The collection's cursors for logs it *still* reads, with positions.

        A cursor whose log is no longer named in the current ``extraction_prompt``
        was left behind by a since-dropped read (e.g. a migration that removed a
        ``log_read``); it would lie about what the collection consumes, so it's
        pruned here — an exact identifier match, deterministic, self-healing.
        """
        live: list[tuple[str, datetime]] = []
        for log_name, position in self.db.cursors.list_for(memory.name):
            if memory.extraction_prompt is not None and log_name in memory.extraction_prompt:
                live.append((log_name, position))
            else:
                self.db.cursors.clear(memory.name, log_name)
        return live

    def _log_has_new(self, log_name: str, last_read_at: datetime) -> bool:
        """Is there ≥1 entry in ``log_name`` past ``last_read_at``?  Uses the same
        batched read the collector itself would — uniform across every log
        backing (the ``messagelog`` / ``promptlog`` facades and real logs)."""
        log = self.db.memory(log_name)
        return bool(log and log.read_batch(last_read_at, 1))

    @staticmethod
    def _overdue_sort_key(memory: MemoryRow) -> datetime:
        # Earliest last_collected_at runs first; never-collected sorts to the front.
        return (
            stored_as_utc(memory.last_collected_at)
            if memory.last_collected_at
            else datetime.min.replace(tzinfo=UTC)
        )
