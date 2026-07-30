"""User store — user info, sender queries, and mute state."""

import logging
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, col, select

from penny.constants import PennyConstants
from penny.database.models import MessageLog, MuteState, UserInfo

logger = logging.getLogger(__name__)


class UserStore:
    """Manages UserInfo and MuteState records, and sender-related queries."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def get_info(self, sender: str) -> UserInfo | None:
        """Get the basic user info for a user."""
        with self._session() as session:
            return session.exec(select(UserInfo).where(UserInfo.sender == sender)).first()

    def get_primary_sender(self) -> str | None:
        """Get the single user's primary sender identity (Penny is single-user)."""
        with self._session() as session:
            info = session.exec(select(UserInfo).limit(1)).first()
            return info.sender if info else None

    def save_info(
        self,
        sender: str,
        name: str,
        location: str,
        timezone: str,
        date_of_birth: str,
    ) -> None:
        """Create or update user info."""
        try:
            with self._session() as session:
                existing = session.exec(select(UserInfo).where(UserInfo.sender == sender)).first()
                if existing:
                    existing.name = name
                    existing.location = location
                    existing.timezone = timezone
                    existing.date_of_birth = date_of_birth
                    existing.updated_at = datetime.now(UTC)
                    session.add(existing)
                else:
                    session.add(
                        UserInfo(
                            sender=sender,
                            name=name,
                            location=location,
                            timezone=timezone,
                            date_of_birth=date_of_birth,
                        )
                    )
                session.commit()
                logger.debug("Saved user info for %s", sender)
        except Exception as e:
            logger.error("Failed to save user info: %s", e)

    def get_all_senders(self) -> list[str]:
        """Get all unique senders who have sent messages."""
        with self._session() as session:
            senders = session.exec(
                select(MessageLog.sender)
                .where(MessageLog.direction == PennyConstants.MessageDirection.INCOMING)
                .distinct()
            ).all()
            return list(senders)

    def find_sender_for_timestamp(self, timestamp: datetime) -> str | None:
        """Find the sender of the most recent incoming message near a timestamp."""
        buffer = timedelta(minutes=5)
        with self._session() as session:
            return session.exec(
                select(MessageLog.sender)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.timestamp <= timestamp + buffer,
                )
                .order_by(col(MessageLog.timestamp).desc())
                .limit(1)
            ).first()

    # --- Mute state ---

    def is_muted(self, user: str) -> bool:
        """Check if a user has muted notifications."""
        with self._session() as session:
            return session.get(MuteState, user) is not None

    def set_muted(self, user: str) -> None:
        """Mute proactive notifications for a user."""
        with self._session() as session:
            if not session.get(MuteState, user):
                session.add(MuteState(user=user))
                session.commit()

    def set_unmuted(self, user: str) -> None:
        """Unmute proactive notifications for a user."""
        with self._session() as session:
            existing = session.get(MuteState, user)
            if existing:
                session.delete(existing)
                session.commit()
