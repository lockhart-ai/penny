"""Collector — single dispatcher agent for per-collection extraction.

One ``Collector`` instance runs in the background.  Each cycle it picks
the most-overdue ready collection from ``memory`` (where
``extraction_prompt IS NOT NULL`` and
``now - last_collected_at >= collector_interval_seconds``), binds itself
to that target, runs the agent loop with the target's extraction prompt
as instructions and a tool surface scoped to writes against that
collection only, then stamps ``last_collected_at = now``.

Readiness has a second gate beyond the interval: a *log-driven* collection
(one that reads a log via ``log_read``, leaving a read cursor) is skipped
without entering the model whenever every one of its live input logs is
caught up — ``head <= last_read_at``.  The cursors a collection already
holds are its declared inputs, so no spec is needed; a cursor whose log the
prompt no longer names is pruned so it can't keep gating.  This replaces the
auto-throttle for these collections: instead of widening the interval after
idle cycles (which stalls catch-up when the log starts moving again), the
gate runs the collection exactly when — and only when — its inputs advance.
Generative / collection-driven collections (no log cursor) keep the
interval + auto-throttle fallback.

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
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from penny.agents.base import BackgroundAgent
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.config import Config
from penny.constants import (
    WRITE_GATE_STOP_REASONS,
    MutationActor,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory.types import MemoryNotFoundError
from penny.database.models import MemoryRow
from penny.datetime_utils import format_log_timestamp
from penny.llm.client import LlmClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.text_validity import check_extraction_prompt
from penny.tools.memory_tools import DoneTool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Tools whose successful use means a cycle produced work — it changed a
# collection or reached out to the user.  Reads and ``done()`` don't count; a
# run of only those is "idle" and feeds the auto-throttle counter.
class Collector(BackgroundAgent):
    """Single dispatcher agent — picks the most-overdue ready collection per cycle."""

    name = "collector"

    # Runtime rules every collector cycle gets, appended to whatever
    # extraction_prompt the chat agent (or migration) wrote on the
    # ``memory`` row.  These are *behaviour* invariants — not authoring
    # guidance — so they're attached structurally rather than relied on
    # the prompt-writer to include.  Penny dropped the provenance line
    # in the first prague-highlights prompt she wrote even though the
    # chat-facing guide called for it; structural enforcement is the
    # fix.  Class-scoped so subclasses (none yet) could override if a
    # different runtime contract emerged.
    _RUNTIME_RULES = (
        "## Runtime rules (always apply)\n"
        "\n"
        "- Single batched `collection_write(entries=[...])` per cycle — not one call per entry.\n"
        "- End every cycle with `done()` — it takes NO arguments.  It just marks the cycle "
        "finished; the run record is generated automatically from the tool calls you actually "
        "made, so there is nothing to summarise or report.\n"
        "- If nothing new matched, this is a QUIET cycle: do NOT force a `collection_write` "
        "just to have one — read your sources, then call `done()`.  Quiet cycles are normal "
        "and expected.\n"
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, `update_entry(key=<key>, content=<corrected "
        "content>)` or `collection_delete_entry(key=<key>)` rather than appending alongside.\n"
        "- Cite only what you actually browsed this cycle.  Never invent a URL to populate a "
        '"Source:" field — if no real source was fetched, omit the field.\n'
        "- Don't dedup manually — the store rejects duplicates on write automatically."
    )

    def __init__(
        self,
        model_client: LlmClient,
        db: Database,
        config: Config,
        *,
        embedding_model_client: LlmClient,
        vision_model_client: LlmClient | None = None,
    ) -> None:
        super().__init__(
            model_client=model_client,
            db=db,
            config=config,
            embedding_model_client=embedding_model_client,
            vision_model_client=vision_model_client,
        )
        # Set per-cycle inside ``_execute_cycle``.  The scheduler runs cycles
        # one at a time, but on-demand triggers (chat's extraction-prompt test
        # tool, the addon's "run extractor" button) call ``run_for`` off the
        # scheduler's cadence.  ``_cycle_lock`` serializes every cycle so
        # ``_current_target`` is never clobbered by an overlapping run.
        self._current_target: MemoryRow | None = None
        self._cycle_lock = asyncio.Lock()

    async def execute(self) -> bool:
        target = self._next_ready_collection()
        if target is None:
            return False
        success, _ = await self._execute_cycle(target)
        return success

    async def run_for(self, collection_name: str) -> tuple[bool, str]:
        """Run one extraction cycle for the named collection, bypassing readiness checks.

        Used by the chat agent's TestExtractionPromptTool to trigger on-demand
        cycles while authoring or refining an extraction_prompt.  Returns
        ``(success, message)`` where ``message`` is either an error description
        or the cycle's ``done()`` summary prefixed with "Collector cycle complete.".
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
                f"set one with collection_update before testing.",
            )
        if error := check_extraction_prompt(collection.extraction_prompt):
            return False, error
        return await self._execute_cycle(collection)

    async def _execute_cycle(self, collection: MemoryRow) -> tuple[bool, str]:
        """Run one full agent cycle bound to ``collection`` with audit cleanup.

        Owns the ``run_id`` so cleanup has the correct UUID even if
        ``_run_cycle`` raises before any prompts are logged, and so
        neighbouring cycles can't smear into each other's promptlog rows.
        """
        run_id = uuid.uuid4().hex
        success = False
        response: ControllerResponse | None = None
        cancelled = False
        async with self._cycle_lock:
            try:
                self._current_target = collection
                result = await self._run_cycle(run_id)
                success = result.success
                response = result.response
            except asyncio.CancelledError:
                # Foreground activity preempted the cycle — tag clearly rather
                # than letting it look like a model crash, then re-raise.
                cancelled = True
                raise
            finally:
                # Stamp regardless of success — cadence is driven by the check
                # happening, not by success.  A persistently-failing collection
                # would otherwise be re-attempted on every tick.
                self.db.memories.mark_collected(collection.name)
                if cancelled:
                    self._tag_promptlog_run_cancelled(run_id)
                else:
                    # One determination of this cycle's outcome, used for the
                    # audit log, the promptlog tag, and the throttle alike.
                    outcome, reason = self._cycle_result(response)
                    self._tag_promptlog_run(run_id, outcome, reason, self._tool_failures(response))
                    self._apply_throttle(collection, outcome)
                    # Once-shaped trigger: retire the collection after it has run
                    # its allotted number of times (a one-shot reminder archives
                    # itself).  Runs after the outcome is tagged so this cycle is
                    # counted; a cancelled cycle never reaches here, so it doesn't
                    # burn a run.  ``run_id`` is this cycle's run — recorded as the
                    # system archive's cause in the mutation ledger (#1560).
                    self._archive_if_run_limit_reached(collection, run_id)
                self._current_target = None
        # The on-demand test message is STRUCTURAL (#1569): the run's outcome (or
        # its write-gate stop reason) plus the actual tool trace — never a
        # model-authored ``done()`` summary, which no longer exists.
        outcome, reason = self._cycle_result(response)
        detail = reason or outcome.value
        message = f"Collector cycle complete: {detail}"
        tool_trace = self._format_tool_trace(response)
        if tool_trace:
            message = f"{message}\n\n{tool_trace}"
        return success, message

    @staticmethod
    def _format_tool_trace(response: ControllerResponse | None) -> str:
        """Numbered list of tool calls from the cycle, with long args truncated."""
        if not response or not response.tool_calls:
            return ""
        lines = []
        for i, record in enumerate(response.tool_calls, 1):
            args = ", ".join(
                f"{k}={Collector._truncate_arg(v)}" for k, v in record.arguments.items()
            )
            lines.append(f"{i}. {record.tool}({args})")
        return "\n".join(lines)

    @staticmethod
    def _truncate_arg(value: object) -> str:
        """Stringify a tool argument value, truncating to 50 chars."""
        rendered = str(value)
        return rendered if len(rendered) <= 50 else rendered[:47] + "..."

    @staticmethod
    def _produced_work(response: ControllerResponse | None) -> bool:
        """Did this cycle change a collection or message the user?

        Reads the per-call ``ToolCallRecord.mutated`` flag — set from each tool's
        own structured ``ToolResult`` (a row actually written, an entry
        moved/deleted, a message actually sent).  A *successful no-op* (a
        duplicate-rejected write, an update/delete/move on a missing key, a
        muted/cooled-down send) carries ``mutated=False``, so it correctly reads
        as idle — unlike the old "a write tool didn't error" heuristic, which
        counted duplicate-rejected writes as work and starved the throttle.
        """
        if response is None:
            return False
        return any(record.mutated for record in response.tool_calls)

    def should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """A collector cycle ends on a successful ``done()`` OR a write-gate STOP.

        The base terminator is ``done()``; a collector additionally honors a STOP
        carried by a tool result (``collection_write`` → ``KEY_EXISTS_UNCHANGED`` on
        a scoped write, #1587) — a deliberate close at the write chokepoint, so no
        trailing ``done()`` is required (a ``done()`` after a STOP would just be a
        no-op the loop never reaches).  STOP is honored only here (must-act cadence);
        the chat loop uses the base and never stops on a write outcome."""
        return super().should_stop_loop(step_records) or any(
            record.stop_reason is not None for record in step_records
        )

    @classmethod
    def _cycle_result(cls, response: ControllerResponse | None) -> tuple[RunOutcome, str]:
        """The cycle's outcome + a STRUCTURAL reason — the single determination read
        by the audit log, the promptlog tag, and the throttle (#1569).

        Derived from the run's tool calls alone, never a model-authored judgment:
        ``done()`` is an argless sentinel, so there is no ``success``/``summary`` to
        read.  A write-gate STOP (#1587) closes the cycle at the chokepoint with no
        ``done()`` — its stamped reason is the declared stop reason, and the outcome
        is ``worked``/``no_work`` by whether any durable state changed.  A clean
        ``done()`` close is ``worked``/``no_work`` the same way (an empty reason —
        the run record's header falls back to the outcome enum).  Without a
        ``done()`` the run never closed cleanly: durable state changed →
        ``incomplete``, nothing changed → a ``failed`` bail, both with a structural
        no-``done()`` reason.  (``cancelled`` is handled separately — a preempted
        cycle never reaches here.)
        """
        produced = cls._produced_work(response)
        stop = cls._stop_reason(response)
        if stop is not None:
            outcome = RunOutcome.WORKED if produced else RunOutcome.NO_WORK
            return outcome, WRITE_GATE_STOP_REASONS[stop]
        if cls._has_done_call(response):
            return (RunOutcome.WORKED if produced else RunOutcome.NO_WORK), ""
        reason = cls._no_done_reason(response)
        return (RunOutcome.INCOMPLETE if produced else RunOutcome.FAILED), reason

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

    def _apply_throttle(self, collection: MemoryRow, outcome: RunOutcome) -> None:
        """Auto-tune the collection's interval from this cycle's outcome.

        Throttle is now the fallback for collections the cursor gate can't reach
        — generative / collection-driven ones with no live log cursor.  A
        log-driven collection is exempt: the gate skips its idle ticks before
        they run, so it never idles its way into a wider interval (which would
        just re-create the catch-up lag the gate exists to remove).

        A productive cycle (``worked`` or ``incomplete`` — both changed durable
        state) snaps the interval back to the user's set cadence
        (``base_interval_seconds``) and clears the idle counter.  After
        ``COLLECTOR_THROTTLE_AFTER`` consecutive non-productive cycles the
        interval doubles (capped at ``COLLECTOR_MAX_INTERVAL``) and the counter
        resets.  ``COLLECTOR_THROTTLE_AFTER = 0`` disables it.

        Both intervals are guaranteed non-NULL here — only a ready collection
        runs a cycle, and ``_is_ready`` skips any collector collection without a
        ``collector_interval_seconds``.  The ``None`` guard is defensive.
        """
        threshold = int(self.config.runtime.COLLECTOR_THROTTLE_AFTER)
        base = collection.base_interval_seconds
        current = collection.collector_interval_seconds
        if threshold <= 0 or base is None or current is None:
            return
        if outcome in (RunOutcome.WORKED, RunOutcome.INCOMPLETE):
            interval, idle = base, 0
        elif self._live_cursors(collection):
            # Log-driven collection: the cursor gate already skips its idle
            # ticks, so it never accrues idle runs to throttle on — and widening
            # its interval would re-introduce the very catch-up lag the gate
            # removes (new log entries waiting out a stretched floor).  Pinned at
            # base; the watermark, not a timer, decides when it runs.
            return
        else:
            idle = collection.consecutive_idle_runs + 1
            if idle >= threshold:
                ceiling = int(self.config.runtime.COLLECTOR_MAX_INTERVAL)
                interval, idle = min(current * 2, ceiling), 0
            else:
                interval = current
        if interval != current or idle != collection.consecutive_idle_runs:
            self.db.memories.set_cadence(collection.name, interval, idle)

    def _archive_if_run_limit_reached(self, collection: MemoryRow, run_id: str) -> None:
        """Archive a ``max_runs``-bounded collection once it has run its quota.

        The once-shaped trigger (#1556): after ``max_runs`` completed (non-
        cancelled) cycles the collection has done its job — a one-shot reminder
        (``run_at`` + ``max_runs=1``) retires itself, and any bounded collection
        stops re-firing.  Archival (not deletion) via the ordinary archive path
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
            return
        completed = self.db.messages.count_completed_runs(collection.name)
        if completed < collection.max_runs:
            return
        note = f"reached run limit ({completed} of {collection.max_runs} completed runs)"
        logger.info("Archiving '%s': %s", collection.name, note)
        self.db.memories.archive(
            collection.name, actor=MutationActor.SYSTEM, run_id=run_id, note=note
        )

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

    @staticmethod
    def _has_done_call(response: ControllerResponse | None) -> bool:
        """True when the cycle closed via the argless ``done()`` sentinel (#1569) —
        a structural read of the tool trace, not a model judgment."""
        if response is None:
            return False
        return any(record.tool == DoneTool.name for record in response.tool_calls)

    @staticmethod
    def _no_done_reason(response: ControllerResponse | None) -> str:
        """The structural reason a cycle ended without a ``done()`` — distinguish
        actually hitting the step cap from the model trailing off with a text answer
        (both are failures, but only one is "max steps").  The loop returns the
        ``AGENT_MAX_STEPS`` sentinel only on the real cap; anything else is an early
        give-up without reporting an outcome."""
        if response is None:
            return "no response from cycle"
        if response.answer == PennyResponse.AGENT_MAX_STEPS:
            return "max steps exceeded — no done() call"
        return "cycle ended without a done() call"

    # ── Per-cycle prompt + tool scope ─────────────────────────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """System prompt for the bound target — re-fetched each cycle.

        Reading from the DB instead of caching means a chat-side
        ``collection_update`` call that changes ``extraction_prompt`` is
        picked up on the very next collector cycle, no restart needed.
        """
        target = self._require_target()
        fresh = self.db.memories.get(target.name) or target
        return self._compose_prompt(fresh) + self._run_history_section(fresh.name)

    def _run_history_section(self, target_name: str) -> str:
        """A trailing block of this collector's own recent run outcomes (newest
        first) so each cycle knows what its prior invocations did.

        Empty when disabled (``COLLECTOR_RUN_HISTORY`` = 0) or there's no history
        yet.  Each line is the run's STRUCTURAL outcome — its outcome enum, or the
        write-gate stop reason — generated from the ledger (#1569), never a
        model-authored ``done()`` summary (there is none).  Framed as reference,
        not instruction: it tells the collector what it did, to avoid repeating
        work, without feeding its own past prose back into the next cycle.
        """
        limit = int(self.config.runtime.COLLECTOR_RUN_HISTORY)
        outcomes = self.db.messages.recent_run_outcomes(target_name, limit)
        if not outcomes:
            return ""
        lines = "\n".join(
            f"{index}. [{format_log_timestamp(when)}] {outcome}"
            for index, (when, outcome) in enumerate(outcomes, start=1)
        )
        return (
            "\n\n## Your recent runs (newest first)\n"
            "What your previous cycles did, and when — context to avoid repeating "
            "work or re-sending, not an instruction to repeat.\n"
            f"{lines}"
        )

    @classmethod
    def _compose_prompt(cls, target: MemoryRow) -> str:
        """Frame the extraction_prompt with target identity + the assembly-owned
        step tail + runtime rules — one continuous numbered program (#1557).

        The runtime-rules tail is appended structurally — not relayed through
        Penny when she authors the extraction_prompt.  This guarantees the
        rules apply on every cycle regardless of how the prompt was written
        (or whether Penny remembered to include them).  The chat-facing
        ``collection_create`` description only carries authoring-shape
        guidance; the runtime invariants live here.

        The stored prompt is steps ``1..A`` with NO ``done()`` — a skill render
        cannot produce one (the chat ledger has no ``done`` tool; a chat turn ends
        in text), and migration 0087 stripped the legacy seeds' trailing done
        steps.  Assembly appends the tail (:meth:`_injected_steps`): the notify
        steps when the collection notifies, then the terminal ``done()`` — always,
        exactly once, numbered continuing from ``A``.  A write-gate STOP on a
        no-change cycle ends the run at the chokepoint before the later steps, so
        no-news never notifies — structurally.  Uniform for skill-backed and
        legacy hand-authored collections; nothing here is ever written into the
        stored ``extraction_prompt``.
        """
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{target.extraction_prompt}\n"
            f"{cls._injected_steps(target)}\n\n"
            f"{cls._RUNTIME_RULES}"
        )

    @classmethod
    def _injected_steps(cls, target: MemoryRow) -> str:
        """The assembly-owned step tail: notify steps (``notify=true`` only), then
        the terminal ``done()`` — numbered continuing from the stored prompt's
        highest step, so the whole prompt reads as one program (#1557)."""
        base = cls._max_step_number(target.extraction_prompt or "")
        steps: list[str] = list(Prompt.COLLECTOR_NOTIFY_STEPS) if target.notify else []
        steps.append(Prompt.COLLECTOR_DONE_STEP)
        return "\n".join(f"{base + n}. {step}" for n, step in enumerate(steps, start=1))

    @staticmethod
    def _max_step_number(prompt: str) -> int:
        """``A`` — the highest leading step number in the stored prompt (a
        ``^\\d+.`` scan), 0 for an unnumbered prose prompt so injected steps
        start at 1."""
        numbers = re.findall(r"^(\d+)\.", prompt, re.MULTILINE)
        return max((int(number) for number in numbers), default=0)

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
        now = datetime.now(UTC)
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
                "— update it via collection_update to enable collection",
                memory.name,
                len(memory.extraction_prompt),
            )
            return False
        if memory.collector_interval_seconds is None:
            logger.warning(
                "Skipping collection '%s': no collector_interval_seconds set — "
                "set a cadence via collection_update to enable collection",
                memory.name,
            )
            return False
        # Once-shaped trigger (#1556): a collection with a ``run_at`` doesn't fire
        # until that UTC time — a delayed / one-shot start.  NULL for the ordinary
        # recurring cadence.  ``max_runs`` retires it after firing (handled in the
        # cycle-completion path), so the interval never re-triggers a one-shot.
        if memory.run_at is not None and now < _aware(memory.run_at):
            return False
        if memory.last_collected_at is not None:
            elapsed = (now - _aware(memory.last_collected_at)).total_seconds()
            if elapsed < memory.collector_interval_seconds:
                return False  # within its cadence floor
        # Interval floor cleared (or never run).  Now the cursor gate: a
        # log-driven collection caught up on every live input is skipped without
        # entering the model — the watermark, not the clock, says there's work.
        return self._input_pending(memory) is not False

    # ── Cursor gate (skip-when-no-new-input) ──────────────────────────────

    def _input_pending(self, memory: MemoryRow) -> bool | None:
        """Pre-model gate signal, read from the collection's own read cursors.

        ``True`` — at least one live input log has entries past its cursor: run.
        ``False`` — every live cursor is caught up: skip, don't enter the model.
        ``None`` — no live cursor at all: a generative or collection-driven
        collection (browses, picks from another collection) with no log to gate
        on; not gate-eligible, so it runs on its plain interval.

        The cursors a collection already holds *are* its declared inputs — no
        separate spec.  ``commit_pending`` advances a cursor to the newest entry
        actually consumed, so ``head > last_read_at`` means unread input exists.
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
            _aware(memory.last_collected_at)
            if memory.last_collected_at
            else datetime.min.replace(tzinfo=UTC)
        )


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; assume UTC and attach tzinfo."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
