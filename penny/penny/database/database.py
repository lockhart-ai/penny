"""Database facade — composes domain-specific stores."""

import logging
import time
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
from penny.database.mutation_store import MutationStore
from penny.database.send_queue_store import SendQueueStore
from penny.database.skill_store import SkillStore
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
        mutations: Registry-mutation event ledger (create/update/archive provenance)
        send_queue: Durable outbound message queue, drained on the send cooldown
        skills: Versionless skill registry (certified-by-execution tool-call scripts)
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
        # The registry-mutation ledger (#1560) and the send queue (#1634) are both
        # constructed before ``memories`` so the memory store can record
        # create/update/archive events through the ledger AND cancel a collection's
        # pending queued sends when it's archived (the chokepoint that makes
        # teardown silent through the queue).
        self.mutations = MutationStore(self.engine)
        self.send_queue = SendQueueStore(self.engine)
        self.memories = MemoryStore(
            self.engine,
            runtime=runtime,
            mutations=self.mutations,
            send_queue=self.send_queue,
        )
        self.messages = MessageStore(self.engine)
        self.skills = SkillStore(self.engine)
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

    def analyze(self) -> None:
        """Refresh SQLite query-planner statistics for the current schema."""
        started = time.perf_counter()
        try:
            with self.engine.begin() as connection:
                connection.exec_driver_sql("ANALYZE")
        except Exception:
            logger.exception("Database query failed: ANALYZE")
            return
        elapsed_ms = int((time.perf_counter() - started) * 1_000)
        logger.info("Database query completed: ANALYZE (elapsed_ms=%d)", elapsed_ms)

    def get_session(self) -> Session:
        """Get a database session (for direct use by config modules)."""
        return Session(self.engine)
