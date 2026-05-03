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
  - The agent_cursor table still partitions per (agent_name, memory_name),
    so log read cursors stay correctly partitioned per collection even
    though one agent identity (``"collector"``) drives all of them.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from penny.agents.base import BackgroundAgent
from penny.agents.models import ControllerResponse
from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import LogEntryInput
from penny.database.models import Memory
from penny.llm.client import LlmClient
from penny.tools.memory_tools import DoneTool, check_extraction_prompt

logger = logging.getLogger(__name__)


class Collector(BackgroundAgent):
    """Single dispatcher agent — picks the most-overdue ready collection per cycle."""

    name = "collector"

    # Per-cycle audit-log markers — printable success/failure in the
    # ``collector-runs`` entry.  Class-scoped: tied to this agent's
    # behaviour, not used elsewhere in the project.
    _RUN_SUCCESS_MARKER = "✅"
    _RUN_FAILURE_MARKER = "❌"

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
        # Set per-cycle inside ``execute``.  The cycle is single-threaded
        # by the scheduler, so transient instance state is safe.
        self._current_target: Memory | None = None

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
                self._tag_promptlog_run_cancelled(collection, run_id)
            else:
                self._log_run(collection, response)
                self._tag_promptlog_run(collection, run_id, response)
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

    # ── Per-cycle audit log ───────────────────────────────────────────────

    def _log_run(self, target: Memory, response: ControllerResponse | None) -> None:
        """Append one entry to ``collector-runs`` describing this cycle.

        Reads ``done()``'s ``success`` and ``summary`` args from the last
        recorded tool call.  When the cycle hit max_steps without ever
        calling done, both are synthetic (``success=False`` + a sentinel
        summary) so the log still has a row per cycle.
        """
        success, summary = self._extract_done_args(response)
        marker = self._RUN_SUCCESS_MARKER if success else self._RUN_FAILURE_MARKER
        self.db.memories.append(
            PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
            [LogEntryInput(content=f"[{target.name}] {marker} {summary}")],
            author=self.name,
        )

    def _tag_promptlog_run(
        self, target: Memory, run_id: str, response: ControllerResponse | None
    ) -> None:
        """Stamp the cycle outcome onto the matching promptlog run.

        Drives the green/red tag in the addon's prompts tab — same
        ``(success, summary)`` the audit log gets, plus the target name
        so the addon can attribute the run to a collection.  ``run_id``
        is the caller's UUID for this cycle; ``set_run_outcome`` is a
        no-op if no promptlog rows exist for it (the cycle raised before
        the loop ever logged a prompt).
        """
        success, summary = self._extract_done_args(response)
        self.db.messages.set_run_outcome(run_id, success, summary, target.name)

    def _tag_promptlog_run_cancelled(self, target: Memory, run_id: str) -> None:
        """Stamp a cycle that was cut off by foreground activity.

        Cancellation isn't a failure of the cycle's logic — it's the
        scheduler making room for a user message — so the tag uses
        ``success=True`` with a clear reason.  Keeps these out of the
        addon's failure-rate budget.
        """
        self.db.messages.set_run_outcome(
            run_id,
            True,
            "cancelled by foreground activity",
            target.name,
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
