"""Database facade — composes domain-specific stores."""

import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from penny.config_params import RuntimeParams
from penny.database.cursor_store import CursorStore
from penny.database.device_store import DeviceStore
from penny.database.domain_permission_store import DomainPermissionStore
from penny.database.ios_store import IosStore
from penny.database.media_store import MediaStore
from penny.database.memory import Memory, MemoryStore
from penny.database.message_store import MessageStore
from penny.database.preference_store import PreferenceStore
from penny.database.send_queue_store import SendQueueStore
from penny.database.thought_store import ThoughtStore
from penny.database.user_store import UserStore

logger = logging.getLogger(__name__)


class Database:
    """Database facade — provides access to domain-specific stores.

    Stores:
        cursors: Per-agent read cursors into log-shaped memories
        devices: Device registration and lookup
        domain_permissions: Domain access permissions for browser tools
        media: Binary media referenced by memory entries via <media:ID> tokens
        memories: Unified collection + log access (task/memory framework)
        messages: Message/prompt/command logging, threading, queries
        preferences: User preference CRUD and dedup
        send_queue: Durable outbound message queue, drained on the send cooldown
        thoughts: Inner monologue persistence (append-only thought log)
        users: UserInfo, sender queries, mute state
    """

    def __init__(self, db_path: str, runtime: RuntimeParams | None = None):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")

        self.cursors = CursorStore(self.engine)
        self.devices = DeviceStore(self.engine)
        self.domain_permissions = DomainPermissionStore(self.engine)
        self.ios = IosStore(self.engine)
        self.media = MediaStore(self.engine)
        self.memories = MemoryStore(self.engine, runtime=runtime)
        self.messages = MessageStore(self.engine)
        self.preferences = PreferenceStore(self.engine)
        self.send_queue = SendQueueStore(self.engine)
        self.thoughts = ThoughtStore(self.engine)
        self.users = UserStore(self.engine)

        logger.info("Database initialized: %s", db_path)

    def memory(self, name: str) -> Memory | None:
        """The single memory dispatch — return the ``Memory`` object for ``name``.

        Callers operate polymorphically (``db.memory(name).read_latest(...)`` etc.)
        and never branch on the memory's name or shape themselves; the object
        refuses ops that don't fit its shape or read-only-ness.  ``None`` when the
        memory doesn't exist.
        """
        return self.memories.memory(name)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get a database session (for direct use by schedule/config modules)."""
        return Session(self.engine)
