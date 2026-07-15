"""Send queue store — durable outbound message queue drained on a cooldown.

``send_message`` enqueues here rather than dropping a message when the
autonomous-send cooldown hasn't elapsed; the background drain schedule pops the
oldest pending row once the cooldown clears.  A row is **pending** while
``sent_at IS NULL AND cancelled_at IS NULL``; stamping ``sent_at`` marks it
delivered and stamping ``cancelled_at`` marks it cancelled (the queuing collector
was archived before delivery, #1634) — both are kept, not deleted, so the queue
doubles as a delivery/cancellation audit trail.  ``sent_at`` stays the single
source of truth for "was it sent": a cancelled row leaves it NULL because it never
was.
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
        """The oldest message still awaiting delivery, or None if the queue is empty.

        A cancelled row (``cancelled_at`` stamped) is excluded structurally — in
        the WHERE, never a Python filter downstream — so the drainer can never
        deliver a message whose collector was archived (#1634)."""
        with self._session() as session:
            return session.exec(
                select(SendQueueItem)
                .where(
                    SendQueueItem.sent_at.is_(None),  # ty: ignore[unresolved-attribute]
                    SendQueueItem.cancelled_at.is_(None),  # ty: ignore[unresolved-attribute]
                )
                .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def pending_items(self) -> list[SendQueueItem]:
        """Every message still awaiting delivery, oldest-first.

        ``next_pending`` returns just the head the drainer pops; this returns the
        whole pending tail — used to observe what a single cycle enqueued (the
        eval harness reads sends here, since a collector cycle enqueues but never
        runs the drainer that would deliver to the channel).  Cancelled rows are
        excluded structurally, exactly as in ``next_pending`` (#1634)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(SendQueueItem)
                    .where(
                        SendQueueItem.sent_at.is_(None),  # ty: ignore[unresolved-attribute]
                        SendQueueItem.cancelled_at.is_(None),  # ty: ignore[unresolved-attribute]
                    )
                    .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                )
            )

    def cancel_pending(self, collection: str) -> int:
        """Cancel this collection's still-pending queued sends; return the count.

        Called at the archive chokepoint (``MemoryStore._set_archived``) so a
        teardown is silent through the queue — a message queued during the send
        cooldown, then archived, never goes out (#1634).  Cancellation is
        VISIBLE, not deletion: each pending row is stamped ``cancelled_at`` (kept
        as an audit trail), while ``sent_at`` stays NULL — a cancelled row was
        never sent.  Already-delivered rows (``sent_at`` set) and already-cancelled
        rows are untouched, so the call is idempotent."""
        now = datetime.now(UTC)
        with self._session() as session:
            rows = list(
                session.exec(
                    select(SendQueueItem).where(
                        SendQueueItem.collection == collection,
                        SendQueueItem.sent_at.is_(None),  # ty: ignore[unresolved-attribute]
                        SendQueueItem.cancelled_at.is_(None),  # ty: ignore[unresolved-attribute]
                    )
                )
            )
            for row in rows:
                row.cancelled_at = now
                session.add(row)
            session.commit()
        if rows:
            logger.info("Cancelled %d pending send(s) from archived %s", len(rows), collection)
        return len(rows)

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
