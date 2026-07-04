"""SendQueueDrainer — delivers queued outbound messages on the send cooldown.

``send_message`` enqueues into ``db.send_queue`` rather than dropping a message
when the autonomous-send cooldown hasn't elapsed.  This deterministic task is
the other half: each tick it pops the oldest pending message and delivers it,
but only once the flat-interval cooldown has cleared — so no message is lost,
just delayed.

It's a plain ``ScheduledTask`` (not an LLM agent): no model calls, all Python.
Wired as an idle-gated ``PeriodicSchedule`` so a queued message never interrupts
an active conversation — it goes out during the next idle window.

The cooldown logic is identical to the gate ``send_message`` used to apply
inline: bypassed when the user has spoken since Penny's last message (the queued
send is then conversational, not autonomous), otherwise the message waits
``SEND_COOLDOWN_SECONDS`` since Penny's previous outgoing message of any kind
(chat reply included — the cooldown is per-Penny, not per-collection).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from penny.constants import PennyConstants

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel
    from penny.config import Config
    from penny.database import Database

logger = logging.getLogger(__name__)


class SendQueueDrainer:
    """Deliver the oldest queued message once the send cooldown has elapsed."""

    name = "send_queue_drain"

    def __init__(self, db: Database, config: Config) -> None:
        self._db = db
        self._config = config
        self._channel: MessageChannel | None = None

    def set_channel(self, channel: MessageChannel) -> None:
        """Bind the channel the drained messages are delivered through."""
        self._channel = channel

    async def execute(self) -> bool:
        """Deliver one queued message if the cooldown allows; return whether one went out.

        Returns False (no work) when the queue is empty, no channel/recipient is
        wired, or the cooldown hasn't elapsed — the scheduler then moves on to
        the next schedule this tick.
        """
        if self._channel is None:
            return False
        item = self._db.send_queue.next_pending()
        if item is None:
            return False
        if not self._cooldown_elapsed():
            return False
        recipient = self._db.devices.get_default_identifier()
        if recipient is None:
            return False
        await self._channel.send_response(
            recipient=recipient,
            content=item.content,
            parent_id=None,
            author=item.collection,
            quote_message=None,
        )
        if item.id is not None:
            self._db.send_queue.mark_sent(item.id)
        logger.info("send_queue drained: %s → %s", item.collection, recipient)
        return True

    # ── Cooldown (moved verbatim from SendMessageTool) ────────────────────

    def _cooldown_elapsed(self) -> bool:
        """Flat-interval cooldown between *autonomous* sends.

        When ``count == 0`` the user has spoken since Penny last sent, so the
        queued message is conversational, not autonomous — deliver immediately.
        Otherwise wait ``SEND_COOLDOWN_SECONDS`` since Penny's previous outgoing
        message (any author — a chat reply holds off a queued autonomous ping).
        """
        count = self._count_sends_since_user_message()
        if count == 0:
            return True
        latest = self._latest_send_time()
        if latest is None:
            return True
        elapsed = (_naive_utc_now() - _to_naive(latest)).total_seconds()
        return elapsed >= self._config.runtime.SEND_COOLDOWN_SECONDS

    def _latest_send_time(self) -> datetime | None:
        """Created-at of Penny's most recent outgoing message."""
        log = self._db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        entries = log.newest_entries(k=1) if log is not None else []
        return entries[0].created_at if entries else None

    def _count_sends_since_user_message(self) -> int:
        """Number of Penny's outgoing messages newer than the latest user message.

        Bounded read: ``read_since(cutoff)`` pushes the ``timestamp > cutoff``
        filter into SQL, so this touches only the post-cutoff tail.  With no user
        message yet, any prior Penny send counts — a single bounded existence
        probe."""
        log = self._db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        if log is None:
            return 0
        cutoff = self._latest_user_message_time()
        if cutoff is None:
            return len(log.newest_entries(k=1))
        return len(log.read_since(cutoff))

    def _latest_user_message_time(self) -> datetime | None:
        """Created-at of the most recent ``user-messages`` entry."""
        log = self._db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
        entries = log.newest_entries(k=1) if log is not None else []
        return entries[0].created_at if entries else None


def _naive_utc_now() -> datetime:
    """Naive UTC ``now`` to compare against ``MemoryEntry.created_at``,
    which round-trips through SQLite as a tz-naive value."""
    return datetime.now(UTC).replace(tzinfo=None)


def _to_naive(value: datetime) -> datetime:
    """Strip tzinfo if present so naive/aware mixes don't crash arithmetic."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
