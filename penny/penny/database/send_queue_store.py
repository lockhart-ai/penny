"""Send queue store — durable outbound message queue drained on a cooldown.

``send_message`` enqueues here rather than dropping a message when the
autonomous-send cooldown hasn't elapsed; the background drain schedule pops the
oldest pending row once the cooldown clears.  A row is **pending** while
``sent_at IS NULL AND cancelled_at IS NULL AND failed_at IS NULL``; stamping
``sent_at`` marks it delivered, ``cancelled_at`` marks it cancelled (the queuing
collector was archived before delivery, #1634) and ``failed_at`` marks it failed
(a composing report whose message never became sendable within the retry budget,
#2656) — all are kept, not deleted, so the queue doubles as a
delivery/cancellation audit trail.  ``sent_at`` stays the single source of truth
for "was it sent": a cancelled or failed row leaves it NULL because it never was.

A **composing report** (#2656) is a row enqueued with ``report_run_id`` set and
empty content: the notifying cycle's hand-off, naming WHERE the message comes
from rather than carrying it.  The drainer builds the report from that run's
written entries, runs the delivery chat turn, and stores what it wrote before
delivering — the model never sees the queue, and the row never reaches a user
without content.
"""

from __future__ import annotations

import logging

from sqlmodel import Session, col, select

from penny.clock import Clock
from penny.constants import CycleTrigger
from penny.database.models import SendQueueItem

logger = logging.getLogger(__name__)


class SendQueueStore:
    """Enqueue outbound messages and drain them oldest-first."""

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()

    def _session(self) -> Session:
        return Session(self.engine)

    def enqueue(
        self, content: str, collection: str, origin: CycleTrigger = CycleTrigger.CADENCE
    ) -> int:
        """Append a pending message and return its assigned id.

        ``origin`` is the lane the drainer delivers it in (#1939) — what set the
        queuing cycle running.  It DEFAULTS to the cadence lane, the one that waits
        out the autonomous-send cooldown, so a caller that says nothing gets the
        conservative behaviour and skipping the cooldown has to be claimed."""
        with self._session() as session:
            row = SendQueueItem(
                content=content,
                collection=collection,
                origin=origin,
                created_at=self._clock.now(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("send_queue row was inserted but has no id")
            logger.info("Queued message %d from %s (%d chars)", row.id, collection, len(content))
            return row.id

    def enqueue_report(self, collection: str, origin: CycleTrigger, report_run_id: str) -> int:
        """Queue a composing report and return its assigned id.

        The notifying cycle's hand-off (#2656): the row carries WHAT SET THE CYCLE
        RUNNING (``origin``, the delivery lane) and WHERE THE MESSAGE COMES FROM
        (``report_run_id``, the cycle's run id), and no content yet — the message
        is written later, by the delivery chat turn the drainer runs.  A cycle
        whose notification could not be enqueued because the user is muted or
        nobody is registered never reaches here: the decline is decided before
        the row exists."""
        with self._session() as session:
            row = SendQueueItem(
                content="",
                collection=collection,
                origin=origin,
                report_run_id=report_run_id,
                created_at=self._clock.now(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("send_queue row was inserted but has no id")
            logger.info(
                "Queued report %d from %s (cycle run %s)", row.id, collection, report_run_id
            )
            return row.id

    def enqueue_composed(
        self, content: str, collection: str, origin: CycleTrigger, report_run_id: str
    ) -> int:
        """Queue an ALREADY-COMPOSED message that still names the run that wrote it,
        and return its assigned id.

        For a caller that runs its own delivery turn up front (#2846's startup
        changelog announcement) rather than handing the drainer a bare run id to
        compose from later (``enqueue_report``).  The row lands in the exact shape
        ``store_composed`` leaves a composing job report in — content AND
        ``report_run_id`` both set — so the drainer's own structural read
        (``_is_report``) sees it as an ordinary row (content is no longer empty)
        while still carrying the run id the delivered message is stamped with.
        Every argument is required: a caller with no real run behind it uses
        ``enqueue`` instead, deliberately, rather than passing a hollow id here."""
        with self._session() as session:
            row = SendQueueItem(
                content=content,
                collection=collection,
                origin=origin,
                report_run_id=report_run_id,
                created_at=self._clock.now(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("send_queue row was inserted but has no id")
            logger.info(
                "Queued composed message %d from %s (run %s, %d chars)",
                row.id,
                collection,
                report_run_id,
                len(content),
            )
            return row.id

    def store_composed(self, item_id: int, content: str) -> None:
        """Store the message the delivery chat turn wrote for this report row."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None:
                return
            row.content = content
            session.add(row)
            session.commit()
            logger.info("Report %d composed (%d chars)", item_id, len(content))

    def bump_compose_attempts(self, item_id: int) -> int:
        """Count one failed compose; return the row's new attempt count."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None:
                return 0
            row.attempts = (row.attempts or 0) + 1
            session.add(row)
            session.commit()
            return row.attempts

    def mark_failed(self, item_id: int) -> None:
        """Stamp a row failed — the retry budget spent with nothing sendable.

        VISIBLE, not deletion: the row stays as the audit trail, ``sent_at``
        stays NULL (it was never sent), and the pending queries exclude it
        structurally so the drainer can never pick it up again."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None or row.failed_at is not None:
                return
            row.failed_at = self._clock.now()
            session.add(row)
            session.commit()
            logger.error("Report %d failed to compose within its retry budget", item_id)

    def next_pending(self, origin: CycleTrigger | None = None) -> SendQueueItem | None:
        """The oldest message still awaiting delivery, or None if the queue is empty.

        A cancelled row (``cancelled_at`` stamped) and a failed one
        (``failed_at`` stamped) are excluded structurally — in the WHERE, never
        a Python filter downstream — so the drainer can never deliver a message
        whose collector was archived (#1634) or re-pick a report whose retry
        budget is spent (#2656).

        ``origin`` narrows the queue to one delivery lane (#1939), and it is asked
        in the WHERE for the same reason: reading the head and then discarding it
        for being in the wrong lane would stall every message behind it, which is
        exactly the case this exists for — a user waiting on their own run while an
        older cadence message sits out the cooldown.  ``None`` = every lane."""
        with self._session() as session:
            query = select(SendQueueItem).where(
                col(SendQueueItem.sent_at).is_(None),
                col(SendQueueItem.cancelled_at).is_(None),
                col(SendQueueItem.failed_at).is_(None),
            )
            if origin is not None:
                query = query.where(SendQueueItem.origin == origin)
            return session.exec(
                query.order_by(col(SendQueueItem.created_at).asc()).limit(1)
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
                        col(SendQueueItem.sent_at).is_(None),
                        col(SendQueueItem.cancelled_at).is_(None),
                        col(SendQueueItem.failed_at).is_(None),
                    )
                    .order_by(col(SendQueueItem.created_at).asc())
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
        now = self._clock.now()
        with self._session() as session:
            rows = list(
                session.exec(
                    select(SendQueueItem).where(
                        SendQueueItem.collection == collection,
                        col(SendQueueItem.sent_at).is_(None),
                        col(SendQueueItem.cancelled_at).is_(None),
                        col(SendQueueItem.failed_at).is_(None),
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

    def mark_cancelled(self, item_id: int) -> None:
        """Stamp one row cancelled — withheld on purpose, never to be delivered.

        The single-row form of ``cancel_pending``: a composing report the send gate
        declined at compose time (a refusal, a mute taken since the enqueue) is a
        DECISION, not a failure (#2656) — the row is kept as the audit trail,
        ``sent_at`` stays NULL, and the pending queries exclude it structurally."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None or row.cancelled_at is not None:
                return
            row.cancelled_at = self._clock.now()
            session.add(row)
            session.commit()
            logger.info("Cancelled send_queue row %d — withheld before delivery", item_id)

    def mark_sent(self, item_id: int) -> None:
        """Stamp a row delivered so the drain never re-sends it."""
        with self._session() as session:
            row = session.get(SendQueueItem, item_id)
            if row is None:
                return
            row.sent_at = self._clock.now()
            session.add(row)
            session.commit()
            logger.debug("Marked send_queue %d delivered", item_id)
