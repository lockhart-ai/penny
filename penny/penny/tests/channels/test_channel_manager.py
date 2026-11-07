"""Tests for ChannelManager routing and delegation."""

import logging
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from sqlmodel import select

from penny.channels.discord import DiscordChannel
from penny.channels.manager import ChannelManager
from penny.channels.signal import SignalChannel
from penny.channels.turns import TurnRegistry
from penny.constants import ChannelType, PennyConstants, PermissionResolution
from penny.database.models import MessageDelivery


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
    channel.handle_permission_prompt = AsyncMock()
    channel.handle_permission_dismiss = AsyncMock()
    type(channel).sender_id = PropertyMock(return_value=f"mock-{channel_type}")
    return channel


class TestChannelManagerRouting:
    """Route outgoing messages to the correct channel via device lookup."""

    @pytest.mark.asyncio
    async def test_routes_to_signal_device(self, db):
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_message("+15551234567", "hello")
        signal_ch._send_raw.assert_called_once()
        # Penny's own messages fan out to every OTHER registered device too
        # (#3137, rule 1) — the browser gets this one delivered, once.
        browser_ch._send_raw.assert_called_once()
        # The Signal-strict case (#1298): the phone-number recipient is passed
        # through unchanged to _send_raw so Signal's REST send has a real number,
        # not a browser label.
        assert signal_ch._send_raw.call_args.args[0] == "+15551234567"

        # The manager logs the outgoing message exactly once (not double-logged
        # by the concrete channel, and not doubled by the fan-out either) and it
        # surfaces in the penny-messages facade.
        outgoing = db.memory("penny-messages").read_all()
        assert len(outgoing) == 1
        assert outgoing[0].content == "hello"

    @pytest.mark.asyncio
    async def test_browser_device_never_captures_proactive_send(self, db):
        """#1298: even addressed to the browser device's own identifier, a
        proactive send's PRIMARY delivery resolves to the default channel —
        the browser's own identifier never captures it.  The browser still
        gets its own copy through the outgoing fan-out (#3137, rule 1), same
        as any other registered device; what #1298 guards is which channel
        the PRIMARY delivery picks, not whether the browser ever hears about
        the message at all."""
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_message("firefox-laptop", "proactive ping")
        signal_ch._send_raw.assert_called_once()  # the PRIMARY send — signal_ch resolved
        browser_ch._send_raw.assert_called_once()  # the fan-out, not a capture

    @pytest.mark.asyncio
    async def test_proactive_prefers_default_over_browser_identifier(self, db):
        """Regression for #1298: when the onboarding-derived primary sender is a
        browser device label and Discord is the configured primary (its default
        device), a proactive send's PRIMARY delivery lands on Discord, not the
        addon that owns that identifier.  The browser still gets its own copy
        through the outgoing fan-out (#3137, rule 1), same as any other
        registered device."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        discord_ch = _make_mock_channel("discord")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.DISCORD, discord_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.DISCORD, "1234567890", "Discord", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        # userinfo.sender was pinned to the browser label during addon onboarding.
        await manager.send_message("firefox-laptop", "proactive ping")
        discord_ch._send_raw.assert_called_once()
        browser_ch._send_raw.assert_called_once()  # the fan-out, not a capture

    @pytest.mark.asyncio
    async def test_unknown_recipient_falls_back_to_default(self, db):
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)

        await manager.send_message("unknown-device", "hello")
        signal_ch._send_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_typing_routes_to_default_channel(self, db):
        """Typing shares the proactive ``_resolve_channel`` path, so it too
        resolves to the default device's channel rather than the addon whose
        identifier the recipient matches (#1298)."""
        mock_agent = MagicMock()
        manager = ChannelManager(message_agent=mock_agent, db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        await manager.send_typing("firefox-laptop", True)
        signal_ch.send_typing.assert_called_once_with("firefox-laptop", True)
        browser_ch.send_typing.assert_not_called()


class TestChannelManagerDelegation:
    """Forward lifecycle calls to all registered channels."""

    def test_set_scheduler_forwards_to_all(self, db):
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        mock_scheduler = MagicMock()
        manager.set_scheduler(mock_scheduler)

        signal_ch.set_scheduler.assert_called_once_with(mock_scheduler)
        browser_ch.set_scheduler.assert_called_once_with(mock_scheduler)

    @pytest.mark.asyncio
    async def test_close_closes_all(self, db):
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)

        await manager.close()

        signal_ch.close.assert_called_once()
        browser_ch.close.assert_called_once()

    def test_get_channel_returns_registered(self, db):
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        assert manager.get_channel(ChannelType.SIGNAL) is signal_ch
        assert manager.get_channel(ChannelType.BROWSER) is None

    def test_sender_id_from_default_channel(self, db):
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        assert manager.sender_id == "mock-signal"

    def test_prepare_outgoing_uses_default(self, db):
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())

        signal_ch = _make_mock_channel("signal")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)

        manager.prepare_outgoing("hello")
        signal_ch.prepare_outgoing.assert_called_once_with("hello")


class TestPermissionBroadcast:
    """One prompt goes to every channel, so no channel's failure may cost another.

    A permission prompt is shown on every device and answered on whichever one
    the user reaches first, which makes the channels independent by
    construction: awaiting them in a bare loop meant the first raise — a dead
    addon socket, a Signal API wobble — took every channel after it down with
    it, and the answer was then only reachable from the device that had already
    failed.
    """

    @staticmethod
    def _manager_with_two_channels(db) -> tuple[ChannelManager, MagicMock, MagicMock]:
        """A manager holding two channels, in registration order."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        signal_ch = _make_mock_channel("signal")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)
        return manager, signal_ch, browser_ch

    @pytest.mark.asyncio
    async def test_dismiss_reaches_the_siblings_of_a_failing_channel(self, db, caplog):
        """A channel that can't be told the prompt is over doesn't silence the rest.

        The dismiss is what retires each channel's prompt — and on Signal, the
        reaction callback watching its prompt message — so a channel that never
        hears it keeps waiting for an answer to a question already settled.
        """
        manager, failing, sibling = self._manager_with_two_channels(db)
        failing.handle_permission_dismiss.side_effect = RuntimeError("the transport is unreachable")

        with caplog.at_level(logging.ERROR, logger="penny.channels.manager"):
            await manager.broadcast_permission_dismiss("request-1", PermissionResolution.ALLOWED)

        sibling.handle_permission_dismiss.assert_awaited_once_with(
            "request-1", PermissionResolution.ALLOWED
        )
        assert "Fan-out to signal failed" in caplog.text, (
            "the shared guarded fan-out (#1981) names the failing channel"
        )
        assert "the transport is unreachable" in caplog.text, (
            "with the failure itself — logged, never swallowed"
        )

    @pytest.mark.asyncio
    async def test_prompt_reaches_the_siblings_of_a_failing_channel(self, db, caplog):
        """A channel that can't show the prompt doesn't stop the others showing it."""
        manager, failing, sibling = self._manager_with_two_channels(db)
        failing.handle_permission_prompt.side_effect = RuntimeError("the transport is unreachable")

        with caplog.at_level(logging.ERROR, logger="penny.channels.manager"):
            shown = await manager.broadcast_permission_prompt(
                "request-1", "example.test", "https://example.test/page"
            )

        sibling.handle_permission_prompt.assert_awaited_once_with(
            "request-1", "example.test", "https://example.test/page"
        )
        assert shown, "the sibling still showed it, so at least one channel got through"
        assert "Fan-out to signal failed" in caplog.text, (
            "the shared guarded fan-out (#1981) names the failing channel"
        )
        assert "the transport is unreachable" in caplog.text, (
            "with the failure itself — logged, never swallowed"
        )

    @pytest.mark.asyncio
    async def test_prompt_reports_unshown_when_every_channel_fails(self, db, caplog):
        """#1981 part 2: when EVERY channel fails, the broadcast reports it,
        so a caller can name the real cause instead of waiting out a timeout
        implying someone saw the prompt and never answered."""
        manager, failing, other_failing = self._manager_with_two_channels(db)
        failing.handle_permission_prompt.side_effect = RuntimeError("unreachable")
        other_failing.handle_permission_prompt.side_effect = RuntimeError("unreachable")

        shown = await manager.broadcast_permission_prompt(
            "request-1", "example.test", "https://example.test/page"
        )

        assert shown is False


class TestUserSenderResolution:
    """_resolve_user_sender returns primary sender for any device identifier."""

    def test_resolves_browser_device_to_primary_sender(self, db):
        """A browser device identifier resolves to the primary user's sender."""
        db.users.save_info(
            sender="+15551234567",
            name="Test",
            location="NYC",
            timezone="America/New_York",
            date_of_birth="1990-01-01",
        )
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        assert manager._resolve_user_sender("firefox-macbook") == "+15551234567"

    def test_resolves_signal_device_to_itself(self, db):
        """The primary sender resolves to itself."""
        db.users.save_info(
            sender="+15551234567",
            name="Test",
            location="NYC",
            timezone="America/New_York",
            date_of_birth="1990-01-01",
        )
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        assert manager._resolve_user_sender("+15551234567") == "+15551234567"

    def test_falls_back_to_device_when_no_profile(self, db):
        """Without a profile, returns the device identifier as-is."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        assert manager._resolve_user_sender("firefox-macbook") == "firefox-macbook"


class TestOutgoingFanOut:
    """Penny's own messages reach every registered device (#3137, rule 1) —
    not just the one the primary send resolved to — and the attempt is
    recorded per device so "who got it" is a read, never a guess."""

    @pytest.mark.asyncio
    async def test_reaches_every_other_device(self, db):
        """A reply triggered on one channel still fans out to a second,
        unrelated channel registered for a different device."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        signal_ch = _make_mock_channel("signal")
        ios_ch = _make_mock_channel("ios")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.IOS, ios_ch)

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        ios_device = db.devices.register(ChannelType.IOS, "iphone-1", "iPhone")

        await manager.send_message("+15551234567", "hello everywhere")

        signal_ch._send_raw.assert_called_once()
        ios_ch._send_raw.assert_called_once()
        assert ios_ch._send_raw.call_args.args[0] == ios_device.identifier

    @pytest.mark.asyncio
    async def test_one_failing_device_never_blocks_another(self, db, caplog):
        """A fan-out send that raises on one device still reaches the rest
        (#1981's guard, reused for the new fan-out)."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        signal_ch = _make_mock_channel("signal")
        ios_ch = _make_mock_channel("ios")
        browser_ch = _make_mock_channel("browser")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.IOS, ios_ch)
        manager.register_channel(ChannelType.BROWSER, browser_ch)
        ios_ch._send_raw.side_effect = RuntimeError("socket is dead")

        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        db.devices.register(ChannelType.IOS, "iphone-1", "iPhone")
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox")

        with caplog.at_level(logging.ERROR, logger="penny.channels.manager"):
            await manager.send_message("+15551234567", "hello everywhere")

        signal_ch._send_raw.assert_called_once()
        browser_ch._send_raw.assert_called_once()
        assert "socket is dead" in caplog.text

    @pytest.mark.asyncio
    async def test_delivery_recorded_per_device(self, db):
        """``message_delivery`` names which devices got the message, including
        the primary one — a read, never a reconstruction from scattered logs."""
        manager = ChannelManager(message_agent=MagicMock(), db=db, turn_registry=TurnRegistry())
        signal_ch = _make_mock_channel("signal")
        ios_ch = _make_mock_channel("ios")
        manager.register_channel(ChannelType.SIGNAL, signal_ch)
        manager.register_channel(ChannelType.IOS, ios_ch)
        ios_ch._send_raw.side_effect = RuntimeError("socket is dead")

        signal_device = db.devices.register(
            ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True
        )
        ios_device = db.devices.register(ChannelType.IOS, "iphone-1", "iPhone")

        await manager.send_message("+15551234567", "hello everywhere")

        message = db.memory("penny-messages").read_all()[0]
        with db.get_session() as session:
            rows = {
                row.device_id: row
                for row in session.exec(
                    select(MessageDelivery).where(MessageDelivery.message_id == message.id)
                ).all()
            }
        assert rows[signal_device.id].error is None
        assert rows[ios_device.id].error == "socket is dead"

    @pytest.mark.asyncio
    async def test_unregistered_peer_manager_skips_fan_out_loudly(self, db, caplog):
        """A channel that was never registered (no ``ChannelManager`` ever
        wired it) gets no fan-out — but it is logged loudly, never a silent
        skip (standing rule: no ``None`` that quietly means "not linked")."""
        channel = SignalChannel(
            api_url="http://localhost:8080",
            phone_number="+15551234567",
            message_agent=MagicMock(),
            db=db,
            turn_registry=TurnRegistry(),
        )
        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)

        with caplog.at_level(logging.ERROR, logger="penny.channels.base"):
            await channel._fan_out_delivery(1, None, None, "hello", None, None)

        assert "No peer manager wired on SignalChannel" in caplog.text


class TestIncomingNeverReachesSignal:
    """Rule 3 is structural: Signal and Discord never subscribe to a just-
    logged incoming row, so a user message from iOS or the browser cannot
    reach Signal, whatever happens elsewhere in the fan-out."""

    def test_signal_and_discord_never_subscribe(self, db):
        """Constructing Signal/Discord channels never adds a listener to the
        incoming-logged hook — only the socket channels (iOS, browser) do."""
        before = len(db.messages._incoming_logged_listeners)
        SignalChannel(
            api_url="http://localhost:8080",
            phone_number="+15551234567",
            message_agent=MagicMock(),
            db=db,
            turn_registry=TurnRegistry(),
        )
        DiscordChannel(
            token="test-token",
            channel_id="123456789",
            message_agent=MagicMock(),
            db=db,
            turn_registry=TurnRegistry(),
        )
        assert len(db.messages._incoming_logged_listeners) == before

    @pytest.mark.asyncio
    async def test_signal_channel_send_raw_never_called_by_an_incoming_log(self, db):
        """Logging an incoming row never calls Signal's own send primitive —
        the one concrete way a user message could leak onto Signal."""
        channel = SignalChannel(
            api_url="http://localhost:8080",
            phone_number="+15551234567",
            message_agent=MagicMock(),
            db=db,
            turn_registry=TurnRegistry(),
        )
        with patch.object(channel, "_send_raw", new=AsyncMock()) as send_raw:
            db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                "some-ios-device",
                "a message typed on iOS",
            )
            send_raw.assert_not_called()
