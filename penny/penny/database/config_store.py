"""Runtime config store — persistence for the ``runtime_config`` table.

Domain store for ``/config`` and the iOS/browser config-update handlers: get the
stored override for a key, and upsert it. ``clock`` is required (no ``None``
default) — a value every caller already has via ``Database``, so there is no
ambient "current clock" to fall back on.
"""

from __future__ import annotations

import logging

from sqlmodel import Session

from penny.clock import Clock
from penny.database.models import RuntimeConfig

logger = logging.getLogger(__name__)


class ConfigStore:
    """Get and upsert runtime-configurable parameter overrides."""

    def __init__(self, engine, clock: Clock):
        self.engine = engine
        self._clock = clock

    def _session(self) -> Session:
        return Session(self.engine)

    def get(self, key: str) -> RuntimeConfig | None:
        """The stored override row for ``key``, or ``None`` when unset."""
        with self._session() as session:
            return session.get(RuntimeConfig, key)

    def set(self, key: str, value: str, description: str) -> RuntimeConfig:
        """Create or update the stored override for ``key``.

        Matches the upsert this replaces: an existing row keeps the description
        it was created with and only its value/``updated_at`` change; a new row
        records the description it's created with.
        """
        with self._session() as session:
            existing = session.get(RuntimeConfig, key)
            if existing:
                existing.value = value
                existing.updated_at = self._clock.now()
                session.add(existing)
                session.commit()
                session.refresh(existing)
                logger.debug("Updated runtime config %s=%s", key, value)
                return existing
            new_config = RuntimeConfig(
                key=key,
                value=value,
                description=description,
                updated_at=self._clock.now(),
            )
            session.add(new_config)
            session.commit()
            session.refresh(new_config)
            logger.debug("Created runtime config %s=%s", key, value)
            return new_config
