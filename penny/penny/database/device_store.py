"""Device store — registration and lookup for channel endpoints."""

import logging

from sqlmodel import Session, select

from penny.database.models import Device

logger = logging.getLogger(__name__)


class DeviceStore:
    """Manages Device records — one user, many devices."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def get_by_identifier(self, identifier: str) -> Device | None:
        """Look up a device by its unique identifier."""
        with self._session() as session:
            return session.exec(select(Device).where(Device.identifier == identifier)).first()

    def get_by_id(self, device_id: int) -> Device | None:
        """Look up a device by its primary key."""
        with self._session() as session:
            return session.get(Device, device_id)

    def get_default(self) -> Device | None:
        """Get the default device for proactive notifications.

        Ordered so the resolution is deterministic even if several rows carry
        ``is_default`` (registration keeps a single default, but ordering makes
        the read total-order safe regardless): the earliest-registered wins.
        """
        with self._session() as session:
            return session.exec(
                select(Device)
                .where(Device.is_default == True)  # noqa: E712
                .order_by(Device.created_at.asc(), Device.id.asc())
            ).first()

    def get_default_identifier(self) -> str | None:
        """Return the default device identifier for proactive sends."""
        device = self.get_default()
        return device.identifier if device else None

    def get_all(self) -> list[Device]:
        """Get all registered devices."""
        with self._session() as session:
            return list(session.exec(select(Device)).all())

    def register(
        self,
        channel_type: str,
        identifier: str,
        label: str,
        is_default: bool = False,
    ) -> Device:
        """Register a device (upsert — returns existing if identifier matches)."""
        existing = self.get_by_identifier(identifier)
        if existing:
            return existing

        with self._session() as session:
            if is_default:
                self._clear_defaults(session)
            device = Device(
                channel_type=channel_type,
                identifier=identifier,
                label=label,
                is_default=is_default,
            )
            session.add(device)
            session.commit()
            session.refresh(device)
            logger.info("Registered device: %s (%s, %s)", label, channel_type, identifier)
            return device

    def _clear_defaults(self, session: Session) -> None:
        """Clear ``is_default`` on every device — keeps a single default row."""
        for device in session.exec(
            select(Device).where(Device.is_default == True)  # noqa: E712
        ).all():
            device.is_default = False
            session.add(device)

    def set_default(self, device_id: int) -> None:
        """Set a device as the default, clearing the flag on all others."""
        with self._session() as session:
            for device in session.exec(select(Device)).all():
                device.is_default = device.id == device_id
                session.add(device)
            session.commit()
