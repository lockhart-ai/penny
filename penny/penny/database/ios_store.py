"""iOS channel persistence: device registration and durable outbox."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import Device, IosDeviceRegistration, IosOutboxItem

logger = logging.getLogger(__name__)


class IosStore:
    """Manage iOS-specific device state and message delivery state."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def upsert_registration(
        self,
        *,
        device: Device,
        apns_token: str | None,
        apns_environment: str,
        app_version: str | None,
        device_secret: str | None = None,
    ) -> IosDeviceRegistration:
        """Create or update APNs/client metadata for a registered iOS device."""
        if device.id is None:
            raise ValueError("device must be persisted before iOS registration")

        now = datetime.now(UTC)
        with self._session() as session:
            row = session.get(IosDeviceRegistration, device.id)
            if row is None:
                row = IosDeviceRegistration(device_id=device.id)

            if apns_token != row.apns_token:
                row.token_updated_at = now
            row.apns_token = apns_token
            row.apns_environment = apns_environment
            row.app_version = app_version
            row.last_seen_at = now
            row.push_enabled = bool(apns_token)
            if device_secret:
                row.device_secret_hash = _hash_secret(device_secret)

            session.add(row)
            session.commit()
            session.refresh(row)
            logger.info("Updated iOS registration for device_id=%s", device.id)
            return row

    def get_registration(self, device_id: int) -> IosDeviceRegistration | None:
        """Return iOS registration state for a device id."""
        with self._session() as session:
            return session.get(IosDeviceRegistration, device_id)

    def enqueue_outbox(
        self,
        *,
        device_id: int,
        content: str,
        attachments: list[str] | None,
        source_type: str | None,
        source_name: str | None,
        source_hint: str | None,
        push_title: str,
        push_summary: str,
    ) -> IosOutboxItem:
        """Append a message to the durable iOS outbox."""
        with self._session() as session:
            row = IosOutboxItem(
                device_id=device_id,
                content=content,
                attachments_json=json.dumps(attachments) if attachments else None,
                source_type=source_type,
                source_name=source_name,
                source_hint=source_hint,
                push_title=push_title,
                push_summary=push_summary,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            logger.info("Queued iOS outbox item %s for device_id=%s", row.id, device_id)
            return row

    def pending_for_device(self, device_id: int, limit: int = 50) -> list[IosOutboxItem]:
        """Return unacknowledged messages for an iOS device, oldest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(IosOutboxItem)
                    .where(
                        IosOutboxItem.device_id == device_id,
                        IosOutboxItem.acked_at.is_(None),  # ty: ignore[unresolved-attribute]
                    )
                    .order_by(IosOutboxItem.created_at.asc())  # ty: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def pending_count(self, device_id: int) -> int:
        """Count unacknowledged outbox rows for a device."""
        return len(self.pending_for_device(device_id, limit=10_000))

    def mark_acked(self, device_id: int, item_ids: list[int]) -> int:
        """Acknowledge messages for a device. Returns the number updated."""
        if not item_ids:
            return 0
        now = datetime.now(UTC)
        count = 0
        with self._session() as session:
            rows = session.exec(
                select(IosOutboxItem).where(
                    IosOutboxItem.device_id == device_id,
                    IosOutboxItem.id.in_(item_ids),  # ty: ignore[unresolved-attribute]
                    IosOutboxItem.acked_at.is_(None),  # ty: ignore[unresolved-attribute]
                )
            ).all()
            for row in rows:
                row.acked_at = now
                session.add(row)
                count += 1
            session.commit()
        return count

    def mark_push_sent(self, item_id: int) -> None:
        """Record that APNs accepted a push for an outbox row."""
        with self._session() as session:
            row = session.get(IosOutboxItem, item_id)
            if row is None:
                return
            row.push_sent_at = datetime.now(UTC)
            row.push_error = None
            session.add(row)
            session.commit()

    def mark_push_error(self, item_id: int, error: str) -> None:
        """Record the latest APNs send error for an outbox row."""
        with self._session() as session:
            row = session.get(IosOutboxItem, item_id)
            if row is None:
                return
            row.push_error = error[:500]
            session.add(row)
            session.commit()

    def disable_push(self, device_id: int) -> None:
        """Disable push after APNs reports a token is invalid."""
        with self._session() as session:
            row = session.get(IosDeviceRegistration, device_id)
            if row is None:
                return
            row.push_enabled = False
            session.add(row)
            session.commit()


def _hash_secret(secret: str) -> str:
    """Hash a device secret before storage."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()
