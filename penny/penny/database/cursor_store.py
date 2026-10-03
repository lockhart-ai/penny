"""Cursor store — per-agent read progress through log-shaped memories.

The DB layer handles committed advances only. Pending/rollback lives in the
orchestration layer: a run records the batch's max timestamp in-memory, then
calls `advance_committed` on successful completion.

SQLite strips tzinfo on roundtrip, so cursors are stored naive and always
returned as UTC-aware. Callers can pass either form.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.clock import Clock
from penny.database.models import AgentCursor

logger = logging.getLogger(__name__)


class CursorStore:
    """Read-cursor persistence for agent consumption of log-shaped memories."""

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()

    def _session(self) -> Session:
        return Session(self.engine)

    def get(self, agent_name: str, memory_name: str) -> datetime | None:
        with self._session() as session:
            row = session.exec(
                select(AgentCursor).where(
                    AgentCursor.agent_name == agent_name,
                    AgentCursor.memory_name == memory_name,
                )
            ).first()
            return _to_utc(row.last_read_at) if row else None

    def advance_committed(self, agent_name: str, memory_name: str, last_read_at: datetime) -> None:
        """Upsert the cursor to `last_read_at`. Monotonic: never moves backward."""
        incoming = _to_utc(last_read_at)
        with self._session() as session:
            row = session.exec(
                select(AgentCursor).where(
                    AgentCursor.agent_name == agent_name,
                    AgentCursor.memory_name == memory_name,
                )
            ).first()
            now = self._clock.now()
            if row is None:
                session.add(
                    AgentCursor(
                        agent_name=agent_name,
                        memory_name=memory_name,
                        last_read_at=incoming,
                        updated_at=now,
                    )
                )
            elif incoming > _to_utc(row.last_read_at):
                row.last_read_at = incoming
                row.updated_at = now
                session.add(row)
            session.commit()

    def list_for(self, agent_name: str) -> list[tuple[str, datetime]]:
        """Every cursor this owner holds, as ``(memory_name, last_read_at)``.

        For a collector the owner is the bound collection name, so this is the
        set of logs that collection has read, with its position in each.
        """
        with self._session() as session:
            rows = session.exec(
                select(AgentCursor).where(AgentCursor.agent_name == agent_name)
            ).all()
            return [(row.memory_name, _to_utc(row.last_read_at)) for row in rows]

    def set_position(self, agent_name: str, memory_name: str, last_read_at: datetime) -> None:
        """Unconditionally set the cursor — a user override that, unlike
        ``advance_committed``, MAY move backward (re-read from an earlier point)."""
        incoming = _to_utc(last_read_at)
        with self._session() as session:
            row = session.exec(
                select(AgentCursor).where(
                    AgentCursor.agent_name == agent_name,
                    AgentCursor.memory_name == memory_name,
                )
            ).first()
            now = self._clock.now()
            if row is None:
                row = AgentCursor(
                    agent_name=agent_name, memory_name=memory_name, last_read_at=incoming
                )
            row.last_read_at = incoming
            row.updated_at = now
            session.add(row)
            session.commit()

    def clear(self, agent_name: str, memory_name: str) -> None:
        """Delete the cursor row — the next cycle then behaves as a first cycle
        (reads the most recent entries, not the whole history)."""
        with self._session() as session:
            row = session.exec(
                select(AgentCursor).where(
                    AgentCursor.agent_name == agent_name,
                    AgentCursor.memory_name == memory_name,
                )
            ).first()
            if row is not None:
                session.delete(row)
                session.commit()


def _to_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime; normalize aware datetimes to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
