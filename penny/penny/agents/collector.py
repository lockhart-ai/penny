"""Collector — single dispatcher agent for per-collection extraction.

One ``Collector`` instance runs in the background.  Each cycle it picks
the most-overdue ready collection from ``memory`` (where
``extraction_prompt IS NOT NULL`` and
``now - last_collected_at >= collector_interval_seconds``), binds itself
to that target, runs the agent loop with the target's extraction prompt
as instructions and a tool surface scoped to writes against that
collection only, then stamps ``last_collected_at = now``.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from penny.agents.base import BackgroundAgent
from penny.agents.models import ControllerResponse
from penny.config import Config
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory_store import LogEntryInput
from penny.database.models import Memory
from penny.llm.client import LlmClient
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import (
    CollectionDeleteEntryTool,
    CollectionMoveTool,
    CollectionWriteTool,
    DoneTool,
    LogAppendTool,
    LogGetTool,
    UpdateEntryTool,
    check_extraction_prompt,
)
from penny.tools.prompt_test import PromptTestTool
from penny.tools.send_message import SendMessageTool

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel

logger = logging.getLogger(__name__)

# Tools whose successful use means a cycle produced work — it changed a
# collection or reached out to the user.  Reads and ``done()`` don't count; a
# run of only those is "idle" and feeds the auto-throttle counter.
_WORK_TOOLS = frozenset(
    {
        CollectionWriteTool.name,
        UpdateEntryTool.name,
        CollectionDeleteEntryTool.name,
        CollectionMoveTool.name,
        LogAppendTool.name,
        SendMessageTool.name,
    }
)


class Collector(BackgroundAgent):
    """Single dispatcher agent — picks the most-overdue ready collection per cycle."""

    name = "collector"

    # Per-outcome marker prefixed to the ``collector-runs`` entry — readable to
    # the user in the addon and to Penny when she reads the log.  The outcome
    # word itself is included too, so the state is unambiguous either way.
    _OUTCOME_MARKER = {
        RunOutcome.WORKED: "✅",
        RunOutcome.NO_WORK: "💤",
        RunOutcome.FAILED: "❌",
        RunOutcome.CANCELLED: "⏸️",
    }

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
        "- ``send_message`` (when the prompt above asks for notify-on-new) is gated on a "
        "successful write: only call it after ``collection_write`` returns without "
        "duplicate-rejection.\n"
        "- Always end the cycle with ``done(success=<bool>, summary=<one-sentence prose>)``. "
        "``success`` is true if the cycle did what the prompt asked, false on no-op or failure. "
        "``summary`` describes what actually happened (entries written, messages sent, why no-op). "
        'If nothing matches the prompt, call ``done(success=true, summary="no new matches this '
        'cycle")`` — quiet cycles are normal.\n'
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
        self._current_target: Memory | None = None
        self._cycle_lock = asyncio.Lock()

    def get_tools(self) -> list[Tool]:
        """Standard collector surface, plus the quality cycle's review tools.

        The self-correcting ``quality`` collector is the only cycle that
        inspects another run's full trace (``log_get``) and revises another
        collection's extraction_prompt (``prompt_test``), so it's the only one
        that needs them.  Gating per bound target keeps both out of every other
        collector's surface (and out of the dry-run's own surface, which
        bypasses this override — no nested prompt_test/log_get).
        """
        tools = super().get_tools()
        target = self._current_target
        if target is not None and target.name == PennyConstants.MEMORY_QUALITY_COLLECTION:
            tools.append(LogGetTool(self.db))
            tools.append(PromptTestTool(self))
        return tools

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

    async def dry_run(self, collection_name: str, candidate_prompt: str) -> str:
        """Simulate one cycle for ``collection_name`` with ``candidate_prompt``,
        reporting what it WOULD do without applying anything.

        Backs the ``prompt_test`` tool: the self-correcting collector vets a
        candidate extraction_prompt before committing it.  Runs on a throwaway
        ``_DryRunCollector`` (its own loop/target state, so it never collides
        with this live cycle) where reads are real-but-non-consuming and
        writes/sends/browse are captured rather than executed.
        """
        collection = self.db.memories.get(collection_name)
        if collection is None:
            return f"Collection '{collection_name}' not found."
        if error := check_extraction_prompt(candidate_prompt):
            return f"Candidate prompt rejected: {error}"
        runner = _DryRunCollector(
            model_client=self._model_client,
            db=self.db,
            config=self.config,
            embedding_model_client=self._embedding_model_client,
            channel=self._channel,
            candidate_prompt=candidate_prompt,
        )
        captures = await runner.simulate(collection)
        return _summarize_dry_run(collection_name, captures)

    async def _execute_cycle(self, collection: Memory) -> tuple[bool, str]:
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
                    self._log_run(collection, outcome, summary, run_id)
                    self._tag_promptlog_run(run_id, outcome, summary)
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

        True when a state-changing tool (write / update / delete / move /
        log_append / send_message) succeeded.  A run of only reads + ``done()``
        is idle — it found nothing to do.
        """
        if response is None:
            return False
        return any(
            not record.failed and record.tool in _WORK_TOOLS for record in response.tool_calls
        )

    @classmethod
    def _cycle_result(cls, response: ControllerResponse | None) -> tuple[RunOutcome, str]:
        """The cycle's outcome + its summary — the single determination read by
        the audit log, the promptlog tag, and the throttle.

        ``done(success=False)`` / max-steps / a crashed cycle → ``failed``.  A
        clean completion is ``worked`` or ``no_work`` by whether a state-
        changing tool actually fired (``_produced_work``).  (``cancelled`` is
        handled separately — a preempted cycle never reaches here.)
        """
        success, summary = cls._extract_done_args(response)
        if not success:
            return RunOutcome.FAILED, summary
        if cls._produced_work(response):
            return RunOutcome.WORKED, summary
        return RunOutcome.NO_WORK, summary

    def _apply_throttle(self, collection: Memory, outcome: RunOutcome) -> None:
        """Auto-tune the collection's interval from this cycle's outcome.

        A ``worked`` cycle snaps the interval back to the user's set cadence
        (``base_interval_seconds``) and clears the idle counter.  After
        ``COLLECTOR_THROTTLE_AFTER`` consecutive non-``worked`` cycles the
        interval doubles (capped at ``COLLECTOR_MAX_INTERVAL``) and the counter
        resets.  ``COLLECTOR_THROTTLE_AFTER = 0`` disables it.
        """
        threshold = int(self.config.runtime.COLLECTOR_THROTTLE_AFTER)
        base = collection.base_interval_seconds
        current = collection.collector_interval_seconds
        if threshold <= 0 or base is None or current is None:
            return
        if outcome == RunOutcome.WORKED:
            interval, idle = base, 0
        else:
            idle = collection.consecutive_idle_runs + 1
            if idle >= threshold:
                ceiling = int(self.config.runtime.COLLECTOR_MAX_INTERVAL)
                interval, idle = min(current * 2, ceiling), 0
            else:
                interval = current
        if interval != current or idle != collection.consecutive_idle_runs:
            self.db.memories.set_cadence(collection.name, interval, idle)

    # ── Per-cycle audit log ───────────────────────────────────────────────

    def _log_run(self, target: Memory, outcome: RunOutcome, summary: str, run_id: str) -> None:
        """Append one entry to ``collector-runs`` describing this cycle.

        The entry leads with the outcome (marker + word) so the state is
        legible both to the user in the addon and to Penny when she reads the
        log — e.g. ``[espresso-gear] 💤 no_work — nothing new``.  It is keyed by
        this cycle's ``run_id`` so the quality collector can expand a suspicious
        summary into the run's full trace via ``log_get``.
        """
        marker = self._OUTCOME_MARKER[outcome]
        self.db.memories.append(
            PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
            [
                LogEntryInput(
                    content=f"[{target.name}] {marker} {outcome.value} — {summary}",
                    key=run_id,
                )
            ],
            author=self.name,
        )

    def _tag_promptlog_run(self, run_id: str, outcome: RunOutcome, summary: str) -> None:
        """Stamp the cycle outcome onto the matching promptlog run.

        Drives the outcome badge in the addon's prompts tab — the same
        ``(outcome, summary)`` the audit log gets.  (The run's collection is
        already on every prompt via the write-time ``run_target`` stamp.)
        ``run_id`` is the caller's UUID for this cycle; ``set_run_outcome`` is a
        no-op if no promptlog rows exist for it (the cycle raised before the loop
        ever logged a prompt).
        """
        self.db.messages.set_run_outcome(run_id, outcome.value, summary)

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
        return (False, "max steps exceeded — no done() call")

    # ── Per-cycle prompt + tool scope ─────────────────────────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """System prompt for the bound target — re-fetched each cycle.

        Reading from the DB instead of caching means a chat-side
        ``collection_update`` call that changes ``extraction_prompt`` is
        picked up on the very next collector cycle, no restart needed.
        """
        target = self._require_target()
        fresh = self.db.memories.get(target.name) or target
        return self._compose_prompt(fresh)

    @classmethod
    def _compose_prompt(cls, target: Memory) -> str:
        """Frame the user-authored extraction_prompt with target identity + runtime rules.

        The runtime-rules tail is appended structurally — not relayed through
        Penny when she authors the extraction_prompt.  This guarantees the
        rules apply on every cycle regardless of how the prompt was written
        (or whether Penny remembered to include them).  The chat-facing
        ``collection_create`` description only carries authoring-shape
        guidance; the runtime invariants live here.
        """
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{target.extraction_prompt}\n\n"
            f"{cls._RUNTIME_RULES}"
        )

    def _memory_scope(self) -> str:
        """Pin entry mutations to the bound target collection."""
        return self._require_target().name

    def _require_target(self) -> Memory:
        if self._current_target is None:
            raise RuntimeError(
                "Collector tool surface accessed outside an execute() cycle "
                "— self._current_target is None"
            )
        return self._current_target

    # ── Dispatcher selection ──────────────────────────────────────────────

    def _next_ready_collection(self) -> Memory | None:
        """Pick the most-overdue ready collection, or None if all caught up."""
        now = datetime.now(UTC)
        ready = [m for m in self.db.memories.list_all() if self._is_ready(m, now)]
        if not ready:
            return None
        return min(ready, key=self._overdue_sort_key)

    @staticmethod
    def _is_ready(memory: Memory, now: datetime) -> bool:
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
        if memory.last_collected_at is None:
            return True  # Never run — always ready
        interval = memory.collector_interval_seconds or PennyConstants.COLLECTOR_DEFAULT_INTERVAL
        elapsed = (now - _aware(memory.last_collected_at)).total_seconds()
        return elapsed >= interval

    @staticmethod
    def _overdue_sort_key(memory: Memory) -> datetime:
        # Earliest last_collected_at runs first; never-collected sorts to the front.
        return (
            _aware(memory.last_collected_at)
            if memory.last_collected_at
            else datetime.min.replace(tzinfo=UTC)
        )


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; assume UTC and attach tzinfo."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# ── Dry-run sandbox (backs the prompt_test tool) ──────────────────────────────


@dataclass
class _CapturedCall:
    """A side-effecting tool call a dry-run cycle made — recorded, not executed."""

    tool: str
    arguments: dict


class _CapturingTool(Tool):
    """Stands in for a side-effecting (or networked) tool during a dry run.

    Presents the wrapped tool's real name/description/parameters so the model
    sees the identical surface, but ``execute`` records the call and returns a
    canned result instead of mutating the DB, messaging the user, or hitting the
    network.  No class-level ``name`` so it isn't entered in the Tool registry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        captures: list[_CapturedCall],
        canned: str,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._captures = captures
        self._canned = canned

    async def execute(self, **kwargs: object) -> str:
        self._captures.append(_CapturedCall(self.name, dict(kwargs)))
        return self._canned


class _DryRunCollector(Collector):
    """A throwaway collector that simulates one cycle with a candidate prompt.

    Lives only for the duration of a ``prompt_test`` call.  It binds the suspect
    collection and runs the real agent loop, but: the system prompt uses the
    *candidate* extraction_prompt; the write/send/browse tools are captured (no
    DB mutation, no outbound message, no network); and the log-read cursor is
    never committed, so reads are real but non-consuming.  Its own loop/target
    state means it never collides with the live cycle that triggered it.
    """

    def __init__(
        self,
        model_client: LlmClient,
        db: Database,
        config: Config,
        *,
        embedding_model_client: LlmClient | None,
        channel: MessageChannel | None,
        candidate_prompt: str,
    ) -> None:
        super().__init__(
            model_client=model_client,
            db=db,
            config=config,
            embedding_model_client=embedding_model_client,
        )
        self._candidate_prompt = candidate_prompt
        self._captures: list[_CapturedCall] = []
        # send_message only enters the surface when a channel is bound; bind the
        # live one so the model can "call" it — the capturing wrapper means it's
        # never actually used.
        if channel is not None:
            self.set_channel(channel)

    def _should_commit_cursor(self, success: bool) -> bool:
        return False  # a simulation never consumes the reads it makes

    async def _build_system_prompt(self, user: str | None) -> str:
        target = self._require_target()
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{self._candidate_prompt}\n\n"
            f"{self._RUNTIME_RULES}"
        )

    def get_tools(self) -> list[Tool]:
        # BackgroundAgent surface (memory + browse + done + send_message),
        # skipping Collector's _extra_tools, then sandboxed.
        return [self._sandbox(tool) for tool in super(Collector, self).get_tools()]

    def _sandbox(self, tool: Tool) -> Tool:
        if tool.name in _WORK_TOOLS:
            return _CapturingTool(
                tool.name,
                tool.description,
                tool.parameters,
                self._captures,
                canned=f"(dry run) recorded {tool.name} — not applied",
            )
        if tool.name == BrowseTool.name:
            return _CapturingTool(
                tool.name,
                tool.description,
                tool.parameters,
                self._captures,
                canned="(dry run) browse simulated — assume a few relevant items were found",
            )
        return tool

    async def simulate(self, collection: Memory) -> list[_CapturedCall]:
        self._current_target = collection
        try:
            await self._run_cycle(uuid.uuid4().hex)
        finally:
            self._current_target = None
        return self._captures


def _summarize_dry_run(collection_name: str, captures: list[_CapturedCall]) -> str:
    """Render the captured calls into a result the model reasons about."""
    sends = [call for call in captures if call.tool == SendMessageTool.name]
    writes = [
        call for call in captures if call.tool in _WORK_TOOLS and call.tool != SendMessageTool.name
    ]
    lines = [
        f"Dry run of `{collection_name}` with the candidate prompt — what it WOULD do "
        "this cycle (nothing was applied):",
        f"- messages it would send to the user: {len(sends)}",
    ]
    lines.extend(f"    · {str(call.arguments.get('content', ''))[:160]}" for call in sends)
    lines.append(f"- collection writes/edits/deletes it would make: {len(writes)}")
    return "\n".join(lines)
