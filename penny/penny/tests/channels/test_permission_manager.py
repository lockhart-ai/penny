"""Tests for PermissionManager — domain permission coordination."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.channels.permission_manager import PermissionManager
from penny.tests.conftest import wait_until


async def _resolve_pending(mgr: PermissionManager, decision: bool) -> None:
    """Wait until a permission request is pending, then resolve it."""
    await wait_until(lambda: any(not f.done() for f in mgr._pending.values()), timeout=2.0)
    for future in mgr._pending.values():
        if not future.done():
            future.set_result(decision)
            return


def _make_manager(db, domain_mode="restrict"):
    """Create a PermissionManager with a mock ChannelManager and config."""
    channel_manager = MagicMock()
    channel_manager.broadcast_permission_prompt = AsyncMock()
    channel_manager.broadcast_permission_dismiss = AsyncMock()
    channel_manager.sync_domain_permissions = AsyncMock()
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.DOMAIN_PERMISSION_MODE = domain_mode
    return PermissionManager(db=db, channel_manager=channel_manager, config=config), channel_manager


class TestDomainCheck:
    """Permission checks for known domains (no prompting needed)."""

    @pytest.mark.asyncio
    async def test_allowed_domain_passes(self, db):
        """No error raised when domain is allowed."""
        db.domain_permissions.set_permission("example.com", "allowed")
        mgr, _cm = _make_manager(db)

        await mgr.check_domain("https://example.com/page")

    @pytest.mark.asyncio
    async def test_blocked_domain_raises(self, db):
        """RuntimeError raised when domain is blocked."""
        db.domain_permissions.set_permission("blocked.com", "blocked")
        mgr, _cm = _make_manager(db)

        with pytest.raises(RuntimeError, match="blocked by user"):
            await mgr.check_domain("https://blocked.com/")

    @pytest.mark.asyncio
    async def test_parent_domain_matches(self, db):
        """Allowing example.com also allows www.example.com."""
        db.domain_permissions.set_permission("example.com", "allowed")
        mgr, _cm = _make_manager(db)

        await mgr.check_domain("https://www.example.com/page")

    @pytest.mark.asyncio
    async def test_domain_extracted_with_port_and_path(self, db):
        """Domain is extracted correctly from URLs with ports and query strings."""
        db.domain_permissions.set_permission("example.com", "blocked")
        mgr, _cm = _make_manager(db)

        with pytest.raises(RuntimeError, match="blocked"):
            await mgr.check_domain("https://example.com:8080/path?q=1")


class TestPromptFlow:
    """Unknown domain triggers prompt, stores result on response."""

    @pytest.mark.asyncio
    async def test_unknown_domain_broadcasts_prompt(self, db):
        """Unknown domain broadcasts prompt via channel manager."""
        mgr, cm = _make_manager(db)

        asyncio.create_task(_resolve_pending(mgr, True))
        await mgr.check_domain("https://newsite.com/")

        cm.broadcast_permission_prompt.assert_called_once()
        call_args = cm.broadcast_permission_prompt.call_args
        assert call_args[0][1] == "newsite.com"
        assert db.domain_permissions.check_domain("newsite.com") == "allowed"

    @pytest.mark.asyncio
    async def test_denial_stores_blocked(self, db):
        """User denial stores the domain as blocked."""
        mgr, _cm = _make_manager(db)

        asyncio.create_task(_resolve_pending(mgr, False))
        with pytest.raises(RuntimeError, match="denied"):
            await mgr.check_domain("https://denied.com/")

        assert db.domain_permissions.check_domain("denied.com") == "blocked"

    @pytest.mark.asyncio
    async def test_handle_decision_resolves_future(self, db):
        """handle_decision from any channel resolves the pending future."""
        mgr, cm = _make_manager(db)

        async def decide_via_handle_decision() -> None:
            await wait_until(
                lambda: cm.broadcast_permission_prompt.call_args is not None, timeout=2.0
            )
            req_id = cm.broadcast_permission_prompt.call_args[0][0]
            mgr.handle_decision(req_id, True)

        asyncio.create_task(decide_via_handle_decision())
        await mgr.check_domain("https://decided.com/")

        assert db.domain_permissions.check_domain("decided.com") == "allowed"


class TestTimeout:
    """Permission prompt timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_does_not_store_domain(self, db):
        """Timeout does NOT store the domain — it stays unknown."""
        from unittest.mock import patch

        mgr, cm = _make_manager(db)

        with (
            patch.object(mgr, "_prompt", return_value=None),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await mgr.check_domain("https://timeout.com/")

        assert db.domain_permissions.check_domain("timeout.com") is None

    @pytest.mark.asyncio
    async def test_timeout_broadcasts_dismiss(self, db):
        """Timeout broadcasts dismiss to all channels."""
        from unittest.mock import patch

        mgr, cm = _make_manager(db)

        with (
            patch.object(mgr, "_prompt", return_value=None),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await mgr.check_domain("https://dismiss.com/")

        # _prompt was mocked so dismiss (which is inside _prompt) was not called
        cm.broadcast_permission_dismiss.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_timeout_sends_dismiss(self, db):
        """A real timeout (no response) broadcasts dismiss."""
        from unittest.mock import patch

        from penny.constants import PennyConstants

        mgr, cm = _make_manager(db)

        # Patch timeout to 0.1s
        with (
            patch.object(PennyConstants, "PERMISSION_PROMPT_TIMEOUT", 0.1),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await mgr.check_domain("https://realtimeout.com/")

        cm.broadcast_permission_prompt.assert_called_once()
        cm.broadcast_permission_dismiss.assert_called_once()


class TestSerialization:
    """Only one prompt at a time via the queue."""

    @pytest.mark.asyncio
    async def test_concurrent_same_domain_one_prompt(self, db):
        """Two requests for the same domain produce only one prompt."""
        mgr, cm = _make_manager(db)

        asyncio.create_task(_resolve_pending(mgr, True))

        await asyncio.gather(
            mgr.check_domain("https://dedup.com/page1"),
            mgr.check_domain("https://dedup.com/page2"),
        )

        # Only one prompt — the second sees it already allowed after the worker re-checks
        assert cm.broadcast_permission_prompt.call_count == 1

    @pytest.mark.asyncio
    async def test_different_domains_queued_sequentially(self, db):
        """Two different unknown domains are prompted one at a time."""

        mgr, cm = _make_manager(db)

        async def approve_each() -> None:
            for _ in range(2):
                await _resolve_pending(mgr, True)

        asyncio.create_task(approve_each())

        await asyncio.gather(
            mgr.check_domain("https://site-a.com/"),
            mgr.check_domain("https://site-b.com/"),
        )

        # Both prompted, but sequentially (queue)
        assert cm.broadcast_permission_prompt.call_count == 2
        assert db.domain_permissions.check_domain("site-a.com") == "allowed"
        assert db.domain_permissions.check_domain("site-b.com") == "allowed"


class TestAllowAllMode:
    """allow_all mode auto-approves unknown domains without prompting."""

    @pytest.mark.asyncio
    async def test_unknown_domain_auto_allowed(self, db):
        """Unknown domain is auto-approved and stored without prompting."""
        mgr, cm = _make_manager(db, domain_mode="allow_all")

        await mgr.check_domain("https://auto-allowed.com/page")

        assert db.domain_permissions.check_domain("auto-allowed.com") == "allowed"
        cm.broadcast_permission_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_domain_still_enforced(self, db):
        """Explicitly blocked domains are still rejected in allow_all mode."""
        db.domain_permissions.set_permission("blocked.com", "blocked")
        mgr, _cm = _make_manager(db, domain_mode="allow_all")

        with pytest.raises(RuntimeError, match="blocked by user"):
            await mgr.check_domain("https://blocked.com/")

    @pytest.mark.asyncio
    async def test_allowed_domain_still_passes(self, db):
        """Already-allowed domains pass without re-storing."""
        db.domain_permissions.set_permission("known.com", "allowed")
        mgr, cm = _make_manager(db, domain_mode="allow_all")

        await mgr.check_domain("https://known.com/page")

        cm.broadcast_permission_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_unknown_domains_all_auto_allowed(self, db):
        """Multiple unknown domains are all auto-approved."""
        mgr, cm = _make_manager(db, domain_mode="allow_all")

        await mgr.check_domain("https://site-a.com/")
        await mgr.check_domain("https://site-b.com/")
        await mgr.check_domain("https://site-c.com/")

        assert db.domain_permissions.check_domain("site-a.com") == "allowed"
        assert db.domain_permissions.check_domain("site-b.com") == "allowed"
        assert db.domain_permissions.check_domain("site-c.com") == "allowed"
        cm.broadcast_permission_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_all_syncs_to_channels(self, db):
        """Auto-approved domains trigger a domain permissions sync."""
        mgr, cm = _make_manager(db, domain_mode="allow_all")

        await mgr.check_domain("https://synced.com/")

        cm.sync_domain_permissions.assert_called_once()


class TestDomainCRUD:
    """Domain CRUD operations store and sync."""

    @pytest.mark.asyncio
    async def test_set_permission_stores_and_syncs(self, db):
        """set_permission stores in DB and syncs to channels."""
        mgr, cm = _make_manager(db)

        await mgr.set_permission("example.com", "allowed")

        assert db.domain_permissions.check_domain("example.com") == "allowed"
        cm.sync_domain_permissions.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_permission_removes_and_syncs(self, db):
        """delete_permission removes from DB and syncs to channels."""
        db.domain_permissions.set_permission("example.com", "allowed")
        mgr, cm = _make_manager(db)

        await mgr.delete_permission("example.com")

        assert db.domain_permissions.check_domain("example.com") is None
        cm.sync_domain_permissions.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompted_approval_syncs(self, db):
        """Prompted domain approval syncs to channels."""
        mgr, cm = _make_manager(db)

        asyncio.create_task(_resolve_pending(mgr, True))
        await mgr.check_domain("https://prompted.com/")

        assert db.domain_permissions.check_domain("prompted.com") == "allowed"
        cm.sync_domain_permissions.assert_called()
