"""Send queue store — durable outbound message queue drained on a cooldown.

``send_message`` enqueues here rather than dropping a message when the
autonomous-send cooldown hasn't elapsed; the background drain schedule pops the
oldest pending row once the cooldown clears.  ``sent_at IS NULL`` marks a row
pending; stamping ``sent_at`` marks it delivered (kept, not deleted, so the
queue doubles as a delivery audit trail).

**Novelty-keyed suppression (#1568)**: every real emission carries a
``novelty_key`` (computed in Python from the cycle's write-gate outcomes).
Before enqueueing, ``send_message`` compares it against the mechanism's most
recent non-suppressed emission (``latest_novelty_key``); an identical key means
the mechanism is re-sending unchanged news, so the row is durably recorded via
``record_suppressed`` (``suppressed_reason`` set) instead of enqueued.  A
suppressed row is never delivered and is excluded from the pending drain.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import MemoryEntry, SendQueueItem

logger = logging.getLogger(__name__)

# The novelty-key body for a cycle that produced no new/changed entries — a
# collector that ran the notify steps but wrote nothing new (a dedup-rejected
# write, or a pure re-observation the STOP gate didn't catch).  A fixed, readable
# marker rather than a digest, so two consecutive no-news emissions collapse to
# the same key and the second is suppressed.  Can't collide with a real digest
# (a 16-hex string).
NOVELTY_NO_CHANGE = "no-change"

# How many hex chars of the sha256 digest name the novelty of a cycle's
# new/changed entries — enough to make an accidental collision between two
# genuinely-different emissions astronomically unlikely, short enough to render
# in the honest receipt.
_NOVELTY_DIGEST_CHARS = 16


def derive_novelty_key(collection: str, news_entries: list[MemoryEntry]) -> str:
    """The novelty identity of a mechanism's emission, computed in Python (#1568).

    ``news_entries`` are the entries this run created or rewrote (the run's
    NEW_KEY / KEY_EXISTS_CHANGED writes, read structurally by their run-id stamp)
    — never model-authored, never parsed from prose.  The key is
    ``<collection>:<digest>`` over the sorted ``key=content`` pairs of those
    entries, so re-reporting the same values (even under a fresh entry key)
    collapses to the same key and a changed value produces a new one.  A cycle
    that wrote nothing new (a dedup-rejected re-observation the STOP gate didn't
    catch) has no news entries, so it keys to ``<collection>:no-change`` — two
    such emissions in a row collapse, and the second is suppressed."""
    if not news_entries:
        return f"{collection}:{NOVELTY_NO_CHANGE}"
    basis = "\n".join(sorted(f"{entry.key}={entry.content}" for entry in news_entries))
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_NOVELTY_DIGEST_CHARS]
    return f"{collection}:{digest}"


class SendQueueStore:
    """Enqueue outbound messages, drain them oldest-first, record suppressions."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def enqueue(self, content: str, collection: str, novelty_key: str) -> int:
        """Append a pending message with its novelty key and return its id."""
        with self._session() as session:
            row = SendQueueItem(
                content=content,
                collection=collection,
                novelty_key=novelty_key,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("send_queue row was inserted but has no id")
            logger.info("Queued message %d from %s (%d chars)", row.id, collection, len(content))
            return row.id

    def record_suppressed(
        self, content: str, collection: str, novelty_key: str, reason: str
    ) -> int:
        """Durably record an emission the novelty gate held back, and return its id.

        The row carries the same ``collection`` / ``novelty_key`` / ``content`` as
        a real emission but a non-NULL ``suppressed_reason`` — so it is never
        delivered (excluded from the pending drain) yet stays a datetime-ordered
        record of "this mechanism tried to re-send unchanged news, and we didn't"
        (#1568)."""
        with self._session() as session:
            row = SendQueueItem(
                content=content,
                collection=collection,
                novelty_key=novelty_key,
                suppressed_reason=reason,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("suppressed send_queue row was inserted but has no id")
            logger.info("Suppressed emission %d from %s: %s", row.id, collection, reason)
            return row.id

    def latest_novelty_key(self, collection: str) -> str | None:
        """The novelty key of this mechanism's most recent REAL emission — the one
        the next send is compared against (#1568).

        Reads the newest non-suppressed row for ``collection`` (pending OR
        delivered — both are real emissions that will/did reach the user), so a
        second cycle whose message is still queued behind the drainer is suppressed
        just as one that already delivered.  Suppressed rows are excluded (they
        never reached the user, so they aren't the baseline).  ``None`` when the
        mechanism has never emitted."""
        with self._session() as session:
            return session.exec(
                select(SendQueueItem.novelty_key)
                .where(
                    SendQueueItem.collection == collection,
                    SendQueueItem.suppressed_reason.is_(None),  # ty: ignore[unresolved-attribute]
                )
                .order_by(SendQueueItem.created_at.desc())  # ty: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def next_pending(self) -> SendQueueItem | None:
        """The oldest message still awaiting delivery, or None if the queue is empty.

        A suppressed row is not pending — it was recorded, never queued — so it is
        excluded here alongside already-delivered rows."""
        with self._session() as session:
            return session.exec(
                select(SendQueueItem)
                .where(
                    SendQueueItem.sent_at.is_(None),  # ty: ignore[unresolved-attribute]
                    SendQueueItem.suppressed_reason.is_(None),  # ty: ignore[unresolved-attribute]
                )
                .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def pending_items(self) -> list[SendQueueItem]:
        """Every message still awaiting delivery, oldest-first (suppressed excluded).

        ``next_pending`` returns just the head the drainer pops; this returns the
        whole pending tail — used to observe what a single cycle enqueued (the
        eval harness reads sends here, since a collector cycle enqueues but never
        runs the drainer that would deliver to the channel)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(SendQueueItem)
                    .where(
                        SendQueueItem.sent_at.is_(None),  # ty: ignore[unresolved-attribute]
                        SendQueueItem.suppressed_reason.is_(None),  # ty: ignore[unresolved-attribute]
                    )
                    .order_by(SendQueueItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                )
            )

    def suppressed_items(self, collection: str | None = None) -> list[SendQueueItem]:
        """Emissions the novelty gate held back, newest first (the suppression
        ledger).  Optionally scoped to one ``collection``."""
        with self._session() as session:
            query = select(SendQueueItem).where(
                SendQueueItem.suppressed_reason.isnot(None)  # ty: ignore[unresolved-attribute]
            )
            if collection is not None:
                query = query.where(SendQueueItem.collection == collection)
            return list(
                session.exec(
                    query.order_by(SendQueueItem.created_at.desc())  # ty: ignore[unresolved-attribute]
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
