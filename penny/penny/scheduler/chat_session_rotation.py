"""ChatSessionRotationTask — records the chat session's idle rotation while idle (#3125)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.scheduler.chat_prewarm import CycleGate

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
    from penny.database.database import Database

logger = logging.getLogger(__name__)


class ChatSessionRotationTask:
    """Writes the chat session's stored boundary once the user has gone
    idle long enough (#3125) — ahead of the chat pre-warm in priority, so
    the pre-warm (a pure read) always warms the ROTATED prefix rather than
    a stale one.

    A ``ScheduledTask`` (``name`` + ``async execute() -> bool``), wired in
    BEFORE ``ChatPrewarmTask`` by a ``PeriodicSchedule`` whose own
    ``idle_threshold`` is ``lambda: 0.0`` — this task's own idleness check
    (``MessageStore.record_idle_rotation``) reads stored message timestamps
    against the live clock, never the scheduler's process-local
    ``idle_seconds`` (which resets on every restart and would under-report
    how long the user was actually idle before one), so a restart inside
    the window still rotates correctly. ``PeriodicSchedule``'s own
    ``interval`` throttles how often this cheap check runs.

    Because ``BackgroundScheduler.run()`` only advances to the next
    schedule in its priority list when the current one returns ``False``,
    a rotation here (``True``) defers the pre-warm to the NEXT tick — "rotate,
    then warm" falls out of the scheduler's existing priority semantics,
    with no new ordering plumbing.
    """

    name = "chat_session_rotation"

    def __init__(
        self,
        db: Database,
        cycle_gate: CycleGate,
        runtime: RuntimeParams,
    ) -> None:
        self._db = db
        self._cycle_gate = cycle_gate
        self._runtime = runtime

    async def execute(self) -> bool:
        """Record the idle rotation if it's due — never while a collector
        cycle is running (a chat turn is already excluded: the scheduler
        skips every schedule while a foreground turn is in flight), and
        never with no primary user configured yet (a fresh install)."""
        if self._cycle_gate.is_cycle_running():
            return False
        sender = self._db.users.get_primary_sender()
        if sender is None:
            return False
        idle_seconds = int(self._runtime.CHAT_SESSION_IDLE_SECONDS)
        rotated = self._db.messages.record_idle_rotation(sender, idle_seconds)
        if rotated:
            logger.info("Chat session rotated for %s (idle %ds)", sender, idle_seconds)
        return rotated
