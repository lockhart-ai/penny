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
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from penny.agents.base import BackgroundAgent
from penny.agents.models import ControllerResponse
from penny.config import Config
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.models import MemoryRow
from penny.datetime_utils import format_log_timestamp
from penny.llm.client import LlmClient
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
        "- Single batched ``collection_write`` per cycle — not one call per entry.\n"
        "- Always end the cycle with ``done(success=<bool>, summary=<one-sentence prose>)``. "
        "``success`` is true only if the cycle did what the prompt asked, false on no-op or "
        "failure.  ``summary`` must state what *actually* happened this cycle — never claim "
        "entries were written unless a ``collection_write`` succeeded.  If your sources could "
        "not be read (every browse failed), say so and set ``success=false`` (e.g. "
        '``done(success=false, summary="could not read any source this cycle; 0 entries '
        'written")``).  If the sources read fine but nothing new matched, call '
        '``done(success=true, summary="no new matches this cycle")`` — quiet cycles are '
        "normal.\n"
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, ``update_entry`` or ``collection_delete_entry`` "
        "rather than appending alongside.\n"
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
        embedding_model_client: LlmClient | None = None,
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
            return False, f"Collection '{collection_name}' not found."
        if collection.archived:
            return False, f"Collection '{collection_name}' is archived."
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
                    outcome, summary = self._cycle_result(response)
                    self._tag_promptlog_run(run_id, outcome, summary, self._tool_failures(response))
                    self._apply_throttle(collection, outcome)
                self._current_target = None
        _, summary = self._extract_done_args(response)
        tool_trace = self._format_tool_trace(response)
        message = f"Collector cycle complete. {summary}"
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

    @classmethod
    def _cycle_result(cls, response: ControllerResponse | None) -> tuple[RunOutcome, str]:
        """The cycle's outcome + its summary — the single determination read by
        the audit log, the promptlog tag, and the throttle.

        A successful ``done()`` is ``worked`` or ``no_work`` by whether a state-
        changing tool actually fired (``_produced_work``).  Without a successful
        ``done()`` the split is by work too: a cycle that still changed durable
        state (the model hit max steps / trailed off after writing) is
        ``incomplete`` — real work landed, it just never closed cleanly — while
        one that changed nothing is a true ``failed`` bail.  (``cancelled`` is
        handled separately — a preempted cycle never reaches here.)
        """
        success, summary = cls._extract_done_args(response)
        produced = cls._produced_work(response)
        if success:
            return (RunOutcome.WORKED if produced else RunOutcome.NO_WORK), summary
        if produced:
            return RunOutcome.INCOMPLETE, summary
        return RunOutcome.FAILED, summary

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
        if self._is_consumer(collection):
            # A consumer is gate-controlled like a log-driven collection: it only
            # wakes when a published source has unseen entries, so it never idles
            # its way into a wider interval.  Pinned at base.
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

    # ── Per-cycle audit (on the promptlog run itself) ─────────────────────

    def _tag_promptlog_run(
        self, run_id: str, outcome: RunOutcome, summary: str, tool_failures: int
    ) -> None:
        """Stamp the cycle outcome onto the matching promptlog run.

        Drives the outcome badge in the addon's prompts tab — the same
        ``(outcome, summary)`` the audit log gets — plus ``tool_failures`` (the
        count of failed tool calls), which the run-health classifier reads to
        flag a tool-failure spiral.  (The run's collection is already on every
        prompt via the write-time ``run_target`` stamp.)  ``run_id`` is the
        caller's UUID for this cycle; ``set_run_outcome`` is a no-op if no
        promptlog rows exist for it (the cycle raised before the loop ever logged
        a prompt).
        """
        self.db.messages.set_run_outcome(run_id, outcome.value, summary, tool_failures)

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
    def _extract_done_args(response: ControllerResponse | None) -> tuple[bool, str]:
        if response is None:
            return (False, "no response from cycle")
        for record in reversed(response.tool_calls):
            if record.tool == DoneTool.name:
                return (
                    bool(record.arguments.get("success", False)),
                    str(record.arguments.get("summary", "")),
                )
        # No done() — distinguish actually hitting the step cap from the model
        # trailing off with a text answer (both are failures, but only one is
        # "max steps").  The loop returns the AGENT_MAX_STEPS sentinel only on the
        # real cap; anything else is an early give-up without reporting an outcome.
        if response.answer == PennyResponse.AGENT_MAX_STEPS:
            return (False, "max steps exceeded — no done() call")
        return (False, "cycle ended without a done() call")

    # ── Per-cycle prompt + tool scope ─────────────────────────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """Static system prompt for the bound target — re-fetched each cycle.

        Reading from the DB instead of caching means a chat-side
        ``collection_update`` call that changes ``extraction_prompt`` is
        picked up on the very next collector cycle, no restart needed.

        The shared runtime rules lead (identical across every collector, so
        the prefix stays warm in the KV cache), then the per-collection body.
        The volatile run history rides in the Live-context turn via
        ``_build_injected_context`` rather than here.
        """
        target = self._require_target()
        fresh = self.db.memories.get(target.name) or target
        return self._compose_prompt(fresh)

    async def _build_injected_context(self, user: str | None, content: str | None) -> str:
        """Volatile collector context for the Live-context turn: recent run history."""
        target = self._require_target()
        return self._run_history_section(target.name)

    def _run_history_section(self, target_name: str) -> str:
        """A trailing block of this collector's own recent ``done`` summaries
        (newest first) so each cycle knows what its prior invocations did.

        Empty when disabled (``COLLECTOR_RUN_HISTORY`` = 0) or there's no history
        yet.  Framed as reference, not instruction — the summaries are the model's
        own past outputs, so the heading tells it to use them to avoid repeating
        work, not to replicate them (guards against fixation on its own output).
        """
        limit = int(self.config.runtime.COLLECTOR_RUN_HISTORY)
        summaries = self.db.messages.recent_run_summaries(target_name, limit)
        if not summaries:
            return ""
        lines = "\n".join(
            f"{index}. [{format_log_timestamp(when)}] {summary}"
            for index, (when, summary) in enumerate(summaries, start=1)
        )
        return (
            "## Your recent runs (newest first)\n"
            "What your previous cycles did, and when — context to avoid repeating "
            "work or re-sending, not an instruction to repeat.\n"
            f"{lines}"
        )

    @classmethod
    def _compose_prompt(cls, target: MemoryRow) -> str:
        """Frame the user-authored extraction_prompt with runtime rules + target identity.

        The runtime rules lead — prepended structurally, not relayed through
        Penny when she authors the extraction_prompt — so they apply on every
        cycle regardless of how the prompt was written (or whether Penny
        remembered to include them).  Leading with them (identical across every
        collector) also keeps the shared prefix warm in the KV cache.  The
        chat-facing ``collection_create`` description only carries
        authoring-shape guidance; the runtime invariants live here.
        """
        return (
            f"{cls._RUNTIME_RULES}\n\n"
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{target.extraction_prompt}"
        )

    def _memory_scope(self) -> str:
        """Pin entry mutations to the bound target collection."""
        return self._require_target().name

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

        A *consumer* (its prompt calls ``read_published_latest``) is the pub/sub
        analogue: its inputs aren't prompt-named logs but every ``published``
        collection.  It's always gate-eligible — it wakes only when some
        published source has entries past this consumer's cursor, never on a
        blind interval.
        """
        if self._is_consumer(memory):
            return self._published_pending(memory)
        live = self._live_cursors(memory)
        if not live:
            return None
        return any(self._log_has_new(log_name, position) for log_name, position in live)

    @staticmethod
    def _is_consumer(memory: MemoryRow) -> bool:
        """A collection whose prompt drains the published stream via
        ``read_published_latest`` — identified by the call, exactly as a
        log-reader is identified by naming a log in ``log_read``."""
        return (
            memory.extraction_prompt is not None
            and "read_published_latest" in memory.extraction_prompt
        )

    def _published_pending(self, memory: MemoryRow) -> bool:
        """Does any published collection have an entry this consumer hasn't seen?

        The pub/sub gate: the consumer's inputs are every ``published`` (non-
        archived) collection other than itself, each read against this consumer's
        own cursor.  Returns ``False`` (skip) when nothing publishes yet or every
        source is caught up — a consumer never enters the model with nothing to do.
        """
        return any(
            self._source_has_new(memory.name, source.name)
            for source in self.db.memories.list_all()
            if source.published and not source.archived and source.name != memory.name
        )

    def _source_has_new(self, consumer_name: str, source_name: str) -> bool:
        """Is there ≥1 entry in published ``source_name`` past ``consumer_name``'s
        cursor?  A source this consumer has never read starts at the cold-start
        window, matching ``read_published_latest`` so the gate and the read agree."""
        cursor = self.db.cursors.get(consumer_name, source_name)
        if cursor is None:
            cursor = datetime.now(UTC) - timedelta(
                seconds=PennyConstants.PUBLISHED_COLDSTART_LOOKBACK_SECONDS
            )
        source = self.db.memory(source_name)
        return bool(source and source.read_since(cursor, 1))

    def _live_cursors(self, memory: MemoryRow) -> list[tuple[str, datetime]]:
        """The collection's cursors for logs it *still* reads, with positions.

        A cursor whose log is no longer named in the current ``extraction_prompt``
        was left behind by a since-dropped read (e.g. a migration that removed a
        ``log_read``); it would lie about what the collection consumes, so it's
        pruned here — an exact identifier match, deterministic, self-healing.

        A consumer is excluded: its cursors are ``(consumer, published-source)``
        pairs whose source names are deliberately absent from its prompt, so the
        log-name pruning heuristic would wrongly wipe them.  Consumers gate via
        :meth:`_published_pending`, never here.
        """
        if self._is_consumer(memory):
            return []
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
