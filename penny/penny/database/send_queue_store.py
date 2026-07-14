"""Send queue store — durable outbound message queue drained on a cooldown.

``send_message`` enqueues here rather than dropping a message when the
autonomous-send cooldown hasn't elapsed; the background drain schedule pops the
oldest pending row once the cooldown clears.  ``sent_at IS NULL`` marks a row
pending; stamping ``sent_at`` marks it delivered (kept, not deleted, so the
queue doubles as a delivery audit trail).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import SendQueueItem

logger = logging.getLogger(__name__)


class SendQueueStore:
    """Enqueue outbound messages and drain them oldest-first."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def enqueue(self, content: str, collection: str) -> int:
        """Append a pending message and return its assigned id."""
        with self._session() as session:
            row = SendQueueItem(
                content=content,
                collection=collection,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("send_queue row was inserted but has no id")
            logger.info("Queued message %d from %s (%d chars)", row.id, collection, len(content))
            return row.id

    def next_pending(self) -> SendQueueItem | None:
        """The oldest message still awaiting delivery, or None if the queue is empty."""
        with self._session() as session:
            return session.exec(
                select(SendQueueItem)
                .where(SendQueueItem.sent_at.is_(None))  # ty: ignore[unresolved-attribute]
                .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def pending_items(self) -> list[SendQueueItem]:
        """Every message still awaiting delivery, oldest-first.

        ``next_pending`` returns just the head the drainer pops; this returns the
        whole pending tail — used to observe what a single cycle enqueued (the
        eval harness reads sends here, since a collector cycle enqueues but never
        runs the drainer that would deliver to the channel)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(SendQueueItem)
                    .where(SendQueueItem.sent_at.is_(None))  # ty: ignore[unresolved-attribute]
                    .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                )
            )

    def mark_sent(self, item_id: int) -> None:
        """Stamp a row delivered so the drain never re-sends it."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None:
                return
            row.sent_at = datetime.now(UTC)
            session.add(row)
            session.commit()
            logger.debug("Marked send_queue %d delivered", item_id)
