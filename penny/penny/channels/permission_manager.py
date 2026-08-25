"""Permission manager — cross-channel domain permission coordination.

Owns the permission queue, futures, and timeout logic. Talks to the
ChannelManager for broadcasting prompts and dismissals. Each channel
handles its own prompt UX and cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from penny.config_params import DOMAIN_MODE_ALLOW_ALL
from penny.constants import DomainPermissionValue, PennyConstants, PermissionResolution

if TYPE_CHECKING:
    from penny.channels.manager import ChannelManager
    from penny.config import Config
    from penny.database import Database

logger = logging.getLogger(__name__)


class PermissionManager:
    """Coordinates domain permission prompts across all channels.

    Queue-based: only one prompt at a time. Callers enqueue and await.
    The worker pops, prompts, waits for a response or timeout, stores
    the result, and moves on.
    """

    def __init__(self, db: Database, channel_manager: ChannelManager, config: Config):
        self._db = db
        self._channel_manager = channel_manager
        self._config = config
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._queue: asyncio.Queue[tuple[str, str, asyncio.Future[bool | None]]] = asyncio.Queue()
        self._worker_started = False

    # --- Public API ---

    async def check_domain(self, url: str) -> None:
        """Check domain permission, prompting all devices if unknown.

        Raises RuntimeError if blocked, denied, or timed out.
        """
        parsed = urlparse(url)
        domain = parsed.hostname
        if not domain:
            return

        permission = self._db.domain_permissions.check_domain(domain)
        if permission == DomainPermissionValue.ALLOWED:
            return
        if permission == DomainPermissionValue.BLOCKED:
            raise RuntimeError(f"Domain {domain} is blocked by user")

        # Unknown domain — check mode
        if str(self._config.runtime.DOMAIN_PERMISSION_MODE) == DOMAIN_MODE_ALLOW_ALL:
            self._db.domain_permissions.set_permission(domain, DomainPermissionValue.ALLOWED)
            await self._channel_manager.sync_domain_permissions()
            return

        result_future: asyncio.Future[bool | None] = asyncio.get_event_loop().create_future()
        await self._queue.put((domain, url, result_future))
        self._ensure_worker()

        allowed = await result_future
        if allowed is None:
            raise RuntimeError(f"Permission prompt timed out for {domain}")
        if not allowed:
            raise RuntimeError(f"User denied access to {domain}")

    def handle_decision(self, request_id: str, allowed: bool) -> None:
        """Handle a permission decision from any channel."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(allowed)

    async def set_permission(self, domain: str, permission: str) -> None:
        """Set a domain permission and sync to all channels."""
        self._db.domain_permissions.set_permission(domain, permission)
        await self._channel_manager.sync_domain_permissions()

    async def delete_permission(self, domain: str) -> None:
        """Delete a domain permission and sync to all channels."""
        self._db.domain_permissions.delete(domain)
        await self._channel_manager.sync_domain_permissions()

    # --- Worker ---

    def _ensure_worker(self) -> None:
        """Start the queue worker if not already running."""
        if not self._worker_started:
            self._worker_started = True
            asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        """Process permission requests one at a time."""
        while True:
            domain, url, result_future = await self._queue.get()

            # Re-check — a prior prompt may have resolved this domain
            permission = self._db.domain_permissions.check_domain(domain)
            if permission == DomainPermissionValue.ALLOWED:
                result_future.set_result(True)
                continue
            if permission == DomainPermissionValue.BLOCKED:
                result_future.set_result(False)
                continue

            allowed = await self._prompt(domain, url)
            if allowed is not None:
                perm = DomainPermissionValue.ALLOWED if allowed else DomainPermissionValue.BLOCKED
                self._db.domain_permissions.set_permission(domain, perm)
                await self._channel_manager.sync_domain_permissions()

            result_future.set_result(allowed)

    # --- Prompt orchestration ---

    async def _prompt(self, domain: str, url: str) -> bool | None:
        """Broadcast a prompt to all channels. First response wins."""
        request_id = str(uuid.uuid4())
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        await self._channel_manager.broadcast_permission_prompt(request_id, domain, url)

        try:
            return await asyncio.wait_for(future, timeout=PennyConstants.PERMISSION_PROMPT_TIMEOUT)
        except TimeoutError:
            return None
        finally:
            self._pending.pop(request_id, None)
            resolution = self._resolution(future)
            if not future.done():
                future.cancel()
            await self._channel_manager.broadcast_permission_dismiss(request_id, resolution)

    @staticmethod
    def _resolution(future: asyncio.Future[bool]) -> PermissionResolution:
        """How the prompt ended, read off the future before it is cancelled.

        An answer landed on the future from whichever channel the user reached;
        anything else — the wait timing out, the awaiting task being cancelled,
        a failure set on the future — left the prompt unanswered, which is the
        same thing to a channel acknowledging it. Each clause guards the next:
        this runs in a ``finally``, so raising here would replace the caller's
        own error AND skip the dismiss, leaving every channel's prompt hanging.
        """
        answered = future.done() and not future.cancelled() and future.exception() is None
        return PermissionResolution.from_decision(future.result() if answered else None)
