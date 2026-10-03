"""Domain permission store — server-side domain allowlist for browser tools."""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from penny.clock import Clock
from penny.database.models import DomainPermission

logger = logging.getLogger(__name__)


class DomainPermissionStore:
    """Manages domain access permissions shared across all browser addons."""

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()

    def _session(self) -> Session:
        return Session(self.engine)

    def get_all(self) -> list[DomainPermission]:
        """Get all domain permissions."""
        with self._session() as session:
            return list(session.exec(select(DomainPermission)).all())

    def check_domain(self, domain: str) -> str | None:
        """Check permission for a domain, including parent domain matching.

        Returns "allowed", "blocked", or None (unknown).
        """
        with self._session() as session:
            # Exact match
            row = session.exec(
                select(DomainPermission).where(DomainPermission.domain == domain)
            ).first()
            if row:
                return row.permission

            # Parent domain match (e.g., "www.example.com" matches "example.com")
            parts = domain.split(".")
            for i in range(1, len(parts) - 1):
                parent = ".".join(parts[i:])
                row = session.exec(
                    select(DomainPermission).where(DomainPermission.domain == parent)
                ).first()
                if row:
                    return row.permission

        return None

    def set_permission(self, domain: str, permission: str) -> DomainPermission:
        """Set or update a domain permission (upsert)."""
        with self._session() as session:
            existing = session.exec(
                select(DomainPermission).where(DomainPermission.domain == domain)
            ).first()
            if existing:
                existing.permission = permission
                existing.updated_at = self._clock.now()
                session.add(existing)
                session.commit()
                session.refresh(existing)
                logger.info("Updated domain permission: %s → %s", domain, permission)
                return existing

            now = self._clock.now()
            row = DomainPermission(
                domain=domain, permission=permission, created_at=now, updated_at=now
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            logger.info("Added domain permission: %s → %s", domain, permission)
            return row

    def delete(self, domain: str) -> None:
        """Delete a domain permission."""
        with self._session() as session:
            row = session.exec(
                select(DomainPermission).where(DomainPermission.domain == domain)
            ).first()
            if row:
                session.delete(row)
                session.commit()
                logger.info("Deleted domain permission: %s", domain)
