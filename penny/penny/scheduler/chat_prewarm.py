"""ChatPrewarmTask — primes the next chat turn's prompt-cache prefix while idle (#3078)."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from penny.agents.chat import ChatAgent
    from penny.config_params import RuntimeParams
    from penny.llm.models import LlmMessage, UsageTiming

logger = logging.getLogger(__name__)


class CycleGate(Protocol):
    """The one capability this task needs from the Collector — duck-typed
    (``Collector.is_cycle_running``), never importing ``Collector`` itself:
    the same pattern as ``tools/collection_sandbox_run.py``'s ``SandboxRunner``,
    so this module never imports ``penny.agents.collector``."""

    def is_cycle_running(self) -> bool: ...


class PrewarmModelClient(Protocol):
    """The two ``LlmClient`` members this task actually reads/calls, duck-typed
    the same way — never the concrete class, so a test double only has to
    satisfy this shape (``raw_template_enabled``'s real form is a read-only
    property; a Protocol attribute matches either a property or a plain
    field)."""

    raw_template_enabled: bool

    async def prewarm(
        self, messages: list[LlmMessage], tools: list[dict] | None
    ) -> UsageTiming: ...

    def render_prewarm_prompt(
        self, messages: list[LlmMessage], tools: list[dict] | None
    ) -> str: ...


class ChatPrewarmTask:
    """Sends the next chat turn's rendered prefix to the model server once,
    while idle, so the server's prefix cache is warm before the real turn
    arrives (#3078).

    A ``ScheduledTask`` (``name`` + ``async execute() -> bool``), wired in
    behind the collector's own schedule, at the LOWEST priority, by a
    ``PeriodicSchedule`` whose own idle threshold is
    ``CHAT_PREWARM_IDLE_SECONDS`` — the scheduler's existing per-schedule
    idle gate (#2779), the same mechanism ``IDLE_SECONDS``/
    ``COLLECTOR_IDLE_SECONDS`` already use, so a foreground turn starting
    mid-warm cancels it through the scheduler's own existing
    ``notify_foreground_start`` (no new cancellation plumbing).

    Never fires twice for the same prefix: the dedup key is (the rendered
    prefix's own hash, the newest promptlog row's own timestamp across
    every agent). A collector cycle within the session leaves the chat
    prefix's TEXT unchanged but can evict it from the server's own cache,
    so the newest-call timestamp being part of the key is what makes the
    cycle's own completion trigger a fresh warm even though nothing about
    the text moved; a chat turn changes the key too, which is what warms
    the history form of the turn that just finished.
    """

    name = "chat_prewarm"

    def __init__(
        self,
        chat_agent: ChatAgent,
        model_client: PrewarmModelClient,
        cycle_gate: CycleGate,
        runtime: RuntimeParams,
    ) -> None:
        self._chat_agent = chat_agent
        self._model_client = model_client
        self._cycle_gate = cycle_gate
        self._runtime = runtime
        self._last_warmed_key: tuple[str, datetime | None] | None = None

    async def execute(self) -> bool:
        """Warm the next turn's prefix if it's due — never while a collector
        cycle is running, never on a transport with no vendored chat
        template, never with no primary user configured yet (a fresh
        install), and never twice for the same (prefix, newest-call) key.

        A pure read (#3125): the session boundary this renders against is
        whatever ``ChatSessionRotationTask`` (or a real turn's own inline
        write) has already recorded — this task never rotates it itself."""
        if self._cycle_gate.is_cycle_running():
            return False
        if not self._model_client.raw_template_enabled:
            return False
        sender = self._chat_agent.db.users.get_primary_sender()
        if sender is None:
            return False
        newest_call_at = self._newest_call_timestamp()
        messages, tools = await self._chat_agent.build_prewarm_prefix(sender)
        rendered_prompt = self._model_client.render_prewarm_prompt(messages, tools)
        key = (_prefix_hash(rendered_prompt), newest_call_at)
        if key == self._last_warmed_key:
            return False
        start = time.monotonic()
        usage = await self._model_client.prewarm(messages, tools)
        duration_ms = int((time.monotonic() - start) * 1000)
        self._last_warmed_key = key
        logger.info(
            "Chat prefix pre-warmed: hash=%s prompt_tokens=%s cached_tokens=%s duration_ms=%d",
            key[0],
            usage.prompt_tokens,
            usage.cached_tokens,
            duration_ms,
        )
        return True

    def _newest_call_timestamp(self) -> datetime | None:
        """The newest promptlog row's own timestamp, across every agent
        (#3078) — the dedup key's own freshness signal: a job run or a chat
        turn both move it, even when neither changes the chat prefix's own
        text, because either can have evicted it from the server's cache."""
        newest = self._chat_agent.db.messages.recent_prompts(limit=1)
        return newest[0].timestamp if newest else None


def _prefix_hash(rendered_prompt: str) -> str:
    """A short, deterministic hash of the rendered prefix (#3078) — over the
    EXACT raw-template prompt text the model server receives
    (``LlmClient.render_prewarm_prompt``, #3133), never the pre-wire
    message/tool shapes. Those shapes can carry bytes the render never
    prints — e.g. a replayed tool call's freshly-minted id
    (``new_history_replay_call_id``, re-minted on every history rebuild) —
    which made the OLD hash (over ``to_log_dict()`` + the tool dicts) differ
    on every tick even though the actual rendered prefix, and so the
    server's cache, never changed: the warm re-fired roughly every tick
    instead of once per prefix."""
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()[:16]
