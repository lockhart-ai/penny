"""Tests for ChannelManager routing and delegation."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from penny.channels.manager import ChannelManager
from penny.constants import ChannelType
from penny.database import Database
from penny.database.migrate import migrate


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


def _make_mock_channel(channel_type: str) -> MagicMock:
    """Create a mock MessageChannel with async methods."""
    channel = MagicMock()
    # The manager routes the raw delivery (_send_raw); base send_message/
    # send_response on the manager do the logging once before routing.
    channel._send_raw = AsyncMock(return_value=123)
    channel.send_typing = AsyncMock(return_value=True)
    channel.listen = AsyncMock()
    channel.close = AsyncMock()
    channel.set_scheduler = MagicMock()
    channel.set_command_context = MagicMock()
    channel.prepare_outgoing = MagicMock(side_effect=lambda t: t)
    type(channel).sender_id = PropertyMock(return_value=f"mock-{channel_type}")
    return channel


class TestChannelManagerRouting:
    """Route outgoing messages to the correct channel via device lookup."""

    @pytest.mark.asyncio
    async def test_routes_to_signal_device(self, tmp_path):
        db = _make_db(tmp_path)
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db)

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_message("+15551234567", "hello")
        signal_ch._send_raw.assert_called_once()
        browser_ch._send_raw.assert_not_called()

        # The manager logs the outgoing message exactly once (not double-logged
        # by the concrete channel) and it surfaces in the penny-messages facade.
        outgoing = db.memory("penny-messages").read_all()
        assert len(outgoing) == 1
        assert outgoing[0].content == "hello"

    @pytest.mark.asyncio
    async def test_routes_to_browser_device(self, tmp_path):
        db = _make_db(tmp_path)
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db)

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_message("firefox-laptop", "hello from browser")
        browser_ch._send_raw.assert_called_once()
        signal_ch._send_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_recipient_falls_back_to_default(self, tmp_path):
        db = _make_db(tmp_path)
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db)

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)

        await manager.send_message("unknown-device", "hello")
        signal_ch._send_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_typing_routes_to_correct_channel(self, tmp_path):
        db = _make_db(tmp_path)
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db)

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_typing("firefox-laptop", True)
        browser_ch.send_typing.assert_called_once_with("firefox-laptop", True)
        signal_ch.send_typing.assert_not_called()


class TestChannelManagerDelegation:
    """Forward lifecycle calls to all registered channels."""

    def test_set_scheduler_forwards_to_all(self, tmp_path):
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        mock_scheduler = MagicMock()
        manager.set_scheduler(mock_scheduler)

        signal_ch.set_scheduler.assert_called_once_with(mock_scheduler)
        browser_ch.set_scheduler.assert_called_once_with(mock_scheduler)

    @pytest.mark.asyncio
    async def test_close_closes_all(self, tmp_path):
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        await manager.close()

        signal_ch.close.assert_called_once()
        browser_ch.close.assert_called_once()

    def test_get_channel_returns_registered(self, tmp_path):
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        assert manager.get_channel(ChannelType.SIGNAL) is signal_ch
        assert manager.get_channel(ChannelType.BROWSER) is None

    def test_sender_id_from_default_channel(self, tmp_path):
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        assert manager.sender_id == "mock-signal"

    def test_prepare_outgoing_uses_default(self, tmp_path):
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        manager.prepare_outgoing("hello")
        signal_ch.prepare_outgoing.assert_called_once_with("hello")


class TestUserSenderResolution:
    """_resolve_user_sender returns primary sender for any device identifier."""

    def test_resolves_browser_device_to_primary_sender(self, tmp_path):
        """A browser device identifier resolves to the primary user's sender."""
        db = _make_db(tmp_path)
        db.users.save_info(
            sender="+15551234567",
            name="Test",
            location="NYC",
            timezone="America/New_York",
            date_of_birth="1990-01-01",
        )
        manager = ChannelManager(message_agent=MagicMock(), db=db)
        assert manager._resolve_user_sender("firefox-macbook") == "+15551234567"

    def test_resolves_signal_device_to_itself(self, tmp_path):
        """The primary sender resolves to itself."""
        db = _make_db(tmp_path)
        db.users.save_info(
            sender="+15551234567",
            name="Test",
            location="NYC",
            timezone="America/New_York",
            date_of_birth="1990-01-01",
        )
        manager = ChannelManager(message_agent=MagicMock(), db=db)
        assert manager._resolve_user_sender("+15551234567") == "+15551234567"

    def test_falls_back_to_device_when_no_profile(self, tmp_path):
        """Without a profile, returns the device identifier as-is."""
        db = _make_db(tmp_path)
        manager = ChannelManager(message_agent=MagicMock(), db=db)
        assert manager._resolve_user_sender("firefox-macbook") == "firefox-macbook"
