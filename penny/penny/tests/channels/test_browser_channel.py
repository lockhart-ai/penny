"""Tests for BrowserChannel message extraction and device registration."""

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.channels.base import IncomingMessage
from penny.channels.browser.channel import BrowserChannel, ConnectionInfo
from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.memory_store import EntryInput, LogEntryInput, RecallMode
from penny.database.migrate import migrate
from penny.database.models import PromptLog, RuntimeConfig
from penny.tests.conftest import wait_until
from penny.tools.browse import BrowseTool
from penny.tools.models import SearchResult
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


class TestBrowserChannelExtract:
    """extract_message produces IncomingMessage with correct fields."""

    def test_extracts_message_with_channel_type(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        raw = {"browser_sender": "firefox-macbook", "content": "hello penny"}
        msg = channel.extract_message(raw)

        assert msg is not None
        assert msg.sender == "firefox-macbook"
        assert msg.content == "hello penny"
        assert msg.channel_type == ChannelType.BROWSER
        assert msg.device_identifier == "firefox-macbook"

    def test_extracts_default_sender(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        raw = {"content": "hello"}
        msg = channel.extract_message(raw)

        assert msg is not None
        assert msg.sender == "browser-user"
        assert msg.device_identifier == "browser-user"

    def test_returns_none_for_empty_content(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        assert channel.extract_message({"content": ""}) is None
        assert channel.extract_message({"content": "   "}) is None


class TestBrowserAutoRegistration:
    """_auto_register_device creates device entries in the database."""

    def test_registers_new_device(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._auto_register_device("firefox-macbook-16")

        device = db.devices.get_by_identifier("firefox-macbook-16")
        assert device is not None
        assert device.channel_type == ChannelType.BROWSER
        assert device.label == "firefox-macbook-16"

    def test_register_is_idempotent(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._auto_register_device("firefox-macbook-16")
        channel._auto_register_device("firefox-macbook-16")

        all_devices = db.devices.get_all()
        browser_devices = [d for d in all_devices if d.identifier == "firefox-macbook-16"]
        assert len(browser_devices) == 1


class TestBrowserPrepareOutgoing:
    """prepare_outgoing converts markdown to HTML."""

    def _channel(self, tmp_path):
        db = _make_db(tmp_path)
        return BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

    def test_bold(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("**hello**")
        assert "<strong>hello</strong>" in result

    def test_italic(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("*hello*")
        assert "<em>hello</em>" in result

    def test_strikethrough(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("~~deleted~~")
        assert "<s>deleted</s>" in result

    def test_inline_code(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("use `pip install`")
        assert "<code>pip install</code>" in result

    def test_fenced_code_block(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("```\nprint('hi')\n```")
        assert "<pre><code>" in result
        assert "print" in result

    def test_heading_becomes_strong(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("## Section Title")
        assert "<strong>Section Title</strong>" in result

    def test_markdown_link(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("[click](https://example.com)")
        assert '<a href="https://example.com"' in result
        assert "click</a>" in result

    def test_bare_url(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("visit https://example.com today")
        assert '<a href="https://example.com"' in result

    def test_html_escaped(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("use <script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_table_to_bullets(self, tmp_path):
        table = "| Model | Price |\n|-------|-------|\n| Foo   | $100  |\n| Bar   | $200  |"
        result = self._channel(tmp_path).prepare_outgoing(table)
        assert "<strong>Foo</strong>" in result
        assert "$100" in result
        assert "<strong>Bar</strong>" in result
        assert "|" not in result

    def test_newlines_to_br(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("line one\nline two")
        assert "<br>" in result

    def test_collapses_excessive_breaks(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("a\n\n\n\n\nb")
        assert "<br><br><br>" not in result


class TestBrowserImageHandling:
    """_prepend_images puts images before the message content."""

    def test_prepends_image_url(self):
        result = BrowserChannel._prepend_images("hello", ["https://example.com/img.jpg"])
        assert result.startswith('<img src="https://example.com/img.jpg"')
        assert result.endswith("hello")

    def test_prepends_data_uri(self):
        result = BrowserChannel._prepend_images("hello", ["data:image/png;base64,abc123"])
        assert '<img src="data:image/png;base64,abc123"' in result
        assert result.endswith("hello")

    def test_prepends_raw_base64_as_data_uri(self):
        """Raw base64 from /draw gets wrapped in a data:image/png URI."""
        raw_b64 = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
        result = BrowserChannel._prepend_images("hello", [raw_b64])
        assert '<img src="data:image/png;base64,' in result
        assert result.endswith("hello")

    def test_skips_short_non_url(self):
        result = BrowserChannel._prepend_images("hello", ["short"])
        assert result == "hello"

    def test_no_attachments(self):
        assert BrowserChannel._prepend_images("hello", None) == "hello"
        assert BrowserChannel._prepend_images("hello", []) == "hello"

    def test_multiple_images(self):
        urls = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
        result = BrowserChannel._prepend_images("text", urls)
        assert result.count("<img") == 2
        assert result.endswith("text")


class TestBrowseTool:
    """BrowseTool passes through pre-sanitized content from the channel."""

    @staticmethod
    def _make_tool(request_fn, permission_manager=None):
        """Create a BrowseTool wired to a mock browse provider."""
        perm = permission_manager or MagicMock(check_domain=AsyncMock())
        tool = BrowseTool(max_calls=3)
        tool.set_browse_provider(lambda: (request_fn, perm))
        return tool

    @pytest.mark.asyncio
    async def test_returns_channel_content_as_search_result(self):
        """Tool returns a SearchResult with the channel content."""
        request_fn = AsyncMock(
            return_value=("Title: Example\nURL: https://example.com\n\nPage content.", None)
        )
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, SearchResult)
        assert "Page content." in result.text
        request_fn.assert_called_once_with("browse_url", {"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_image_url_from_response(self):
        """Tool passes through image URL from the tool response tuple."""
        request_fn = AsyncMock(
            return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", "https://ex.com/og.jpg")
        )
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, SearchResult)
        assert result.image_base64 == "https://ex.com/og.jpg"

    @pytest.mark.asyncio
    async def test_cleans_kagi_cruft_from_content(self):
        """Tool strips Kagi proxy images, empty links, and image grid sections."""
        kagi_content = (
            "### Best guitar amps 2026\n"
            "\n"
            "![Favicon of  guitarworld.com](https://p.kagi.com/proxy/favicons?c=abc)\n"
            "\n"
            "[](https://www.guitarworld.com/best-amps)\n"
            "\n"
            "[guitarworld.com/best-amps](https://www.guitarworld.com/best-amps)\n"
            "\n"
            "Jan 27, 2026 The Catalyst 200 is based on 6 amp models.\n"
            "\n"
            "[Images](https://kagi.com/images?q=guitar+amps)\n"
            "\n"
            "[![](https://p.kagi.com/proxy/OIP.abc)](https://p.kagi.com/proxy/img.jpg)\n"
            "\n"
            "[www.example.com](https://www.example.com/page)\n"
            "\n"
            "1920 x 1080\n"
            "\n"
            " Loading source... Made with [Openverse](https://openverse.org/)\n"
            "\n"
            "### Second result title\n"
            "\n"
            "[example.org/page](https://example.org/page)\n"
            "\n"
            "A useful snippet about the second result."
        )
        request_fn = AsyncMock(return_value=(kagi_content, None))
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://kagi.com/search?q=test"])

        assert isinstance(result, SearchResult)
        # Signal preserved
        assert "### Best guitar amps 2026" in result.text
        assert "Jan 27, 2026 The Catalyst 200" in result.text
        assert "[guitarworld.com/best-amps]" in result.text
        assert "### Second result title" in result.text
        assert "A useful snippet" in result.text
        # Cruft removed
        assert "![Favicon" not in result.text
        assert "p.kagi.com/proxy" not in result.text
        assert "[](https://" not in result.text
        assert "1920 x 1080" not in result.text
        assert "Openverse" not in result.text
        assert "[www.example.com]" not in result.text

    @pytest.mark.asyncio
    async def test_no_image_returns_none(self):
        """SearchResult.image_base64 is None when response has no image."""
        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, SearchResult)
        assert result.image_base64 is None

    @pytest.mark.asyncio
    async def test_browser_runtime_error_becomes_error_section(self, monkeypatch):
        """A RuntimeError from request_fn (structured browser failure) is surfaced
        under the dedicated error header, not the success header."""

        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        request_fn = AsyncMock(side_effect=RuntimeError("extraction failed after 10 retries"))
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, SearchResult)
        assert PennyConstants.BROWSE_ERROR_HEADER + "https://example.com" in result.text
        assert "extraction failed" in result.text
        assert PennyConstants.BROWSE_PAGE_HEADER + "https://example.com" not in result.text

    @pytest.mark.asyncio
    async def test_checks_permission_before_browsing(self):
        """Tool calls permission_manager.check_domain before requesting the page."""
        mock_perm = MagicMock()
        mock_perm.check_domain = AsyncMock()
        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        tool = self._make_tool(request_fn, permission_manager=mock_perm)
        await tool.execute(queries=["https://example.com"])

        mock_perm.check_domain.assert_called_once_with("https://example.com")
        request_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_permission_denied_reports_error_in_result(self, monkeypatch):
        """Permission denial appears under the browse error header, request_fn not called."""

        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        mock_perm = MagicMock()
        mock_perm.check_domain = AsyncMock(side_effect=RuntimeError("blocked"))
        request_fn = AsyncMock()
        tool = self._make_tool(request_fn, permission_manager=mock_perm)

        result = await tool.execute(queries=["https://blocked.com"])

        assert PennyConstants.BROWSE_ERROR_HEADER + "https://blocked.com" in result.text
        assert "blocked" in result.text
        request_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_hanging_request_fn_times_out_per_attempt(self, monkeypatch):
        """A hung request_fn should timeout per-attempt and surface as an error section."""
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 1)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        monkeypatch.setattr(PennyConstants, "BROWSE_REQUEST_TIMEOUT", 0.05)

        async def hanging_request_fn(method: str, params: dict):
            await asyncio.sleep(1000)
            return ("", None)

        tool = self._make_tool(hanging_request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, SearchResult)
        assert PennyConstants.BROWSE_ERROR_HEADER + "https://example.com" in result.text


class TestBrowseToolImagePassthrough:
    """BrowseTool passes the first browse image through to the combined result."""

    @pytest.mark.asyncio
    async def test_image_from_browse_propagates(self):
        """Image from a browse sub-call appears on the combined SearchResult."""
        request_fn = AsyncMock(
            return_value=(
                "Title: Ex\nURL: https://ex.com\nImage: https://ex.com/img.jpg\n\nContent.",
                "https://ex.com/img.jpg",
            )
        )
        mock_perm = MagicMock(check_domain=AsyncMock())
        tool = BrowseTool(max_calls=3)
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        result = await tool.execute(queries=["https://ex.com"])
        assert isinstance(result, SearchResult)
        assert result.image_base64 == "https://ex.com/img.jpg"

    @pytest.mark.asyncio
    async def test_no_image_when_browse_has_none(self):
        """Combined SearchResult has no image when browse returns none."""
        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        mock_perm = MagicMock(check_domain=AsyncMock())
        tool = BrowseTool(max_calls=3)
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        result = await tool.execute(queries=["https://ex.com"])
        assert isinstance(result, SearchResult)
        assert result.image_base64 is None


class _MockWs:
    """Minimal mock WebSocket that captures sent JSON messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))


class TestBrowserConfigHandlers:
    """config_request and config_update handlers send and persist correctly."""

    def _channel(self, tmp_path) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        # Give channel a real RuntimeParams so DB lookups work after updates
        config = MagicMock()
        config.runtime = RuntimeParams(db=db)
        channel._config = config
        return channel, db

    @pytest.mark.asyncio
    async def test_config_request_returns_all_params(self, tmp_path):
        """config_request sends a config_response containing every registered param."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_request(ws)  # ty: ignore[invalid-argument-type]

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "config_response"
        keys = {p["key"] for p in resp["params"]}
        assert keys == set(RUNTIME_CONFIG_PARAMS.keys())

    @pytest.mark.asyncio
    async def test_config_request_param_shape(self, tmp_path):
        """Each param includes key, value, default, description, type, and group."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_request(ws)  # ty: ignore[invalid-argument-type]

        param = next(p for p in ws.sent[0]["params"] if p["key"] == "IDLE_SECONDS")
        assert param["value"] == "60.0"
        assert param["default"] == "60.0"
        assert param["type"] == "float"
        assert "silence" in param["description"].lower()
        assert param["group"] == "Background"

    @pytest.mark.asyncio
    async def test_config_update_persists_value(self, tmp_path):
        """config_update writes the validated value to the runtime_config table."""
        channel, db = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MAX_STEPS", "value": "12"},
        )

        with Session(db.engine) as session:
            row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == "MAX_STEPS")
            ).first()
        assert row is not None
        assert row.value == "12"

    @pytest.mark.asyncio
    async def test_config_update_returns_updated_config_response(self, tmp_path):
        """config_update sends back a config_response reflecting the new value."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MAX_STEPS", "value": "15"},
        )

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "config_response"
        param = next(p for p in resp["params"] if p["key"] == "MAX_STEPS")
        assert param["value"] == "15"

    @pytest.mark.asyncio
    async def test_config_update_unknown_key_is_noop(self, tmp_path):
        """Unknown config key sends nothing and writes nothing to the DB."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "NOT_A_REAL_KEY", "value": "42"},
        )

        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_config_update_invalid_value_is_noop(self, tmp_path):
        """Value that fails validation sends nothing and writes nothing to the DB."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MAX_STEPS", "value": "-5"},
        )

        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_config_request_dispatched_via_process_raw_message(self, tmp_path):
        """config_request type is dispatched through _process_raw_message."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "config_request"}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "config_response"

    @pytest.mark.asyncio
    async def test_config_update_dispatched_via_process_raw_message(self, tmp_path):
        """config_update type is dispatched through _process_raw_message."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "config_update", "key": "MAX_STEPS", "value": "10"}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "config_response"


class TestBrowserHeartbeat:
    """Heartbeat resets the scheduler idle timer without touching schedule intervals."""

    @pytest.mark.asyncio
    async def test_heartbeat_calls_notify_activity(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        scheduler = MagicMock()
        channel.set_scheduler(scheduler)

        ws = _MockWs()
        await channel._process_raw_message(ws, '{"type": "heartbeat"}', None)  # ty: ignore[invalid-argument-type]

        scheduler.notify_activity.assert_called_once()
        scheduler.notify_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_without_scheduler_is_noop(self, tmp_path):
        """No scheduler set — heartbeat is silently ignored."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        # Should not raise
        await channel._process_raw_message(ws, '{"type": "heartbeat"}', None)  # ty: ignore[invalid-argument-type]


class TestBrowserRegister:
    """Register message populates _connections so tool requests can be routed."""

    @pytest.mark.asyncio
    async def test_register_populates_connections(self, tmp_path):
        """After register, _connections has the device."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        assert len(channel._connections) == 0

        ws = _MockWs()
        label = await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )

        assert label == "firefox-macbook"
        assert "firefox-macbook" in channel._connections
        assert channel._connections["firefox-macbook"].ws is ws

    @pytest.mark.asyncio
    async def test_register_creates_device_in_db(self, tmp_path):
        """Register auto-registers the device in the database."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )

        device = db.devices.get_by_identifier("firefox-macbook")
        assert device is not None
        assert device.label == "firefox-macbook"

    @pytest.mark.asyncio
    async def test_tool_request_works_after_register_without_chat(self, tmp_path):
        """Tool requests succeed after register + capabilities even if no chat message was sent."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "capabilities_update", "tool_use_enabled": True}),
            "firefox-macbook",
        )

        # Pre-allow the domain so the permission check passes
        db.domain_permissions.set_permission("example.com", "allowed")

        # Simulate a tool response arriving after we send the request
        async def fake_tool_response() -> None:
            await wait_until(
                lambda: any(not f.done() for f in channel._pending_requests.values()),
                timeout=2.0,
            )
            for future in channel._pending_requests.values():
                if not future.done():
                    future.set_result(("page content here", None))
                    return

        asyncio.create_task(fake_tool_response())
        result = await channel.send_tool_request("browse_url", {"url": "https://example.com"})
        assert result == ("page content here", None)


class TestCapabilitiesAndToolRouting:
    """Tool-use toggle and smart routing based on capabilities."""

    async def _register(self, channel, label, ws=None):
        """Register a browser connection by device label."""
        ws = ws or _MockWs()
        await channel._process_raw_message(
            ws,
            json.dumps({"type": "register", "sender": label}),
            None,
        )
        return ws

    async def _set_capabilities(self, channel, label, ws, tool_use_enabled):
        """Send a capabilities_update for a registered connection."""
        await channel._process_raw_message(
            ws,
            json.dumps({"type": "capabilities_update", "tool_use_enabled": tool_use_enabled}),
            label,
        )

    @pytest.mark.asyncio
    async def test_capabilities_update_sets_tool_use(self, tmp_path):
        """capabilities_update toggles tool_use_enabled on the connection."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = await self._register(channel, "firefox-1")
        assert not channel._connections["firefox-1"].tool_use_enabled

        await self._set_capabilities(channel, "firefox-1", ws, True)
        assert channel._connections["firefox-1"].tool_use_enabled

        await self._set_capabilities(channel, "firefox-1", ws, False)
        assert not channel._connections["firefox-1"].tool_use_enabled

    @pytest.mark.asyncio
    async def test_has_tool_connection_requires_tool_use_enabled(self, tmp_path):
        """has_tool_connection is False when connections exist but none have tool_use enabled."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = await self._register(channel, "firefox-1")
        assert not channel.has_tool_connection

        await self._set_capabilities(channel, "firefox-1", ws, True)
        assert channel.has_tool_connection

    @pytest.mark.asyncio
    async def test_get_tool_connection_picks_enabled_addon(self, tmp_path):
        """Smart routing picks the tool-enabled connection, not the first one."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        await self._register(channel, "firefox-personal")
        ws_penny = await self._register(channel, "firefox-penny")

        # Only enable tool use on the second one
        await self._set_capabilities(channel, "firefox-penny", ws_penny, True)

        routed = channel._get_tool_connection()
        assert routed is ws_penny

    @pytest.mark.asyncio
    async def test_get_tool_connection_none_when_all_disabled(self, tmp_path):
        """Returns None when no connections have tool_use enabled."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        await self._register(channel, "firefox-1")
        await self._register(channel, "firefox-2")

        assert channel._get_tool_connection() is None


class TestBrowserPermissionDelegation:
    """BrowserChannel delegates permission checks to PermissionManager."""

    async def _setup_channel(self, tmp_path):
        """Create a channel with a registered, tool-enabled connection and permission manager."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-penny"}),
            None,
        )
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "capabilities_update", "tool_use_enabled": True}),
            "firefox-penny",
        )
        return channel, db, ws

    @pytest.mark.asyncio
    async def test_permission_decision_routes_to_manager(self, tmp_path):
        """permission_decision message routes to the permission manager."""
        channel, db, ws = await self._setup_channel(tmp_path)
        mock_perm_mgr = MagicMock()
        channel.set_permission_manager(mock_perm_mgr)

        await channel._process_raw_message(
            ws,
            json.dumps({"type": "permission_decision", "request_id": "test-123", "allowed": True}),
            "firefox-penny",
        )

        mock_perm_mgr.handle_decision.assert_called_once_with("test-123", True)

    @pytest.mark.asyncio
    async def test_handle_permission_prompt_sends_to_all_addons(self, tmp_path):
        """handle_permission_prompt sends prompt to all connected addons."""
        channel, db, ws1 = await self._setup_channel(tmp_path)

        ws2 = _MockWs()
        await channel._process_raw_message(
            ws2,
            json.dumps({"type": "register", "sender": "firefox-personal"}),
            None,
        )

        await channel.handle_permission_prompt("req-1", "example.com", "https://example.com/")

        for ws in [ws1, ws2]:
            prompts = [m for m in ws.sent if m.get("type") == "permission_prompt"]
            assert len(prompts) == 1
            assert prompts[0]["domain"] == "example.com"

    @pytest.mark.asyncio
    async def test_handle_permission_dismiss_sends_to_all_addons(self, tmp_path):
        """handle_permission_dismiss sends dismiss to all connected addons."""
        channel, db, ws1 = await self._setup_channel(tmp_path)

        ws2 = _MockWs()
        await channel._process_raw_message(
            ws2,
            json.dumps({"type": "register", "sender": "firefox-personal"}),
            None,
        )

        await channel.handle_permission_dismiss("req-1")

        for ws in [ws1, ws2]:
            dismissals = [m for m in ws.sent if m.get("type") == "permission_dismiss"]
            assert len(dismissals) == 1


class TestFormatToolStatus:
    """_format_tool_status produces human-readable labels for each tool."""

    def test_browse_with_url_query(self):
        result = BrowserChannel._format_tool_status(
            BrowseTool.name, {"queries": ["https://example.com"]}
        )
        assert result == "Reading example.com"

    def test_browse_with_text_query(self):
        result = BrowserChannel._format_tool_status(
            BrowseTool.name, {"queries": ["best guitar amps"]}
        )
        assert result == 'Searching "best guitar amps"'

    def test_browse_without_queries(self):
        result = BrowserChannel._format_tool_status(BrowseTool.name, {})
        assert result == "Looking up..."

    def test_search_emails(self):
        result = BrowserChannel._format_tool_status(SearchEmailsTool.name, {"text": "invoice"})
        assert result == "Searching emails"

    def test_read_emails(self):
        result = BrowserChannel._format_tool_status(ReadEmailsTool.name, {"email_ids": ["123"]})
        assert result == "Reading emails"

    def test_unknown_tool(self):
        result = BrowserChannel._format_tool_status("my_custom_tool", {})
        assert result == "Using my_custom_tool"


class TestMakeHandleKwargs:
    """_make_handle_kwargs returns a callback that sends tool status to the browser."""

    @pytest.mark.asyncio
    async def test_returns_on_tool_start_key(self, tmp_path):
        """_make_handle_kwargs always returns a dict with an on_tool_start callable."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        message = IncomingMessage(sender="browser-user", content="hello")
        kwargs = channel._make_handle_kwargs(message)

        assert "on_tool_start" in kwargs
        assert callable(kwargs["on_tool_start"])

    @pytest.mark.asyncio
    async def test_callback_sends_tool_status(self, tmp_path):
        """Callback calls _send_tool_status with the sender and formatted text."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._send_tool_status = AsyncMock()  # ty: ignore[invalid-assignment]

        message = IncomingMessage(sender="firefox-macbook", content="hello")
        kwargs = channel._make_handle_kwargs(message)
        await kwargs["on_tool_start"]([("browse", {"queries": ["test query"]})])

        channel._send_tool_status.assert_called_once()
        recipient, text = channel._send_tool_status.call_args.args
        assert recipient == "firefox-macbook"
        assert "test query" in text

    @pytest.mark.asyncio
    async def test_send_tool_status_sends_typing_with_content(self, tmp_path):
        """_send_tool_status sends a typing message with the status text as content."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        ws = _MockWs()
        cast(dict, channel._connections)["browser-user"] = ConnectionInfo(ws=ws)  # ty: ignore[invalid-argument-type]

        await channel._send_tool_status("browser-user", "Searching for stuff")

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "typing"
        assert msg["active"] is True
        assert msg["content"] == "Searching for stuff"


class TestBrowserScheduleHandlers:
    """Schedule request/add/update/delete handlers for browser extension."""

    USER = "testuser"

    def _channel(self, tmp_path, monkeypatch) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        monkeypatch.setattr(db.users, "get_primary_sender", lambda: self.USER)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        return channel, db

    def _add_schedule(self, db, timing="daily 9am", prompt="check the news", cron="0 9 * * *"):
        with Session(db.engine) as session:
            from penny.database.models import Schedule

            sched = Schedule(
                user_id=self.USER,
                user_timezone="America/New_York",
                cron_expression=cron,
                prompt_text=prompt,
                timing_description=timing,
            )
            session.add(sched)
            session.commit()
            session.refresh(sched)
            return sched

    def _add_user_info(self, db):
        with Session(db.engine) as session:
            from penny.database.models import UserInfo

            session.add(
                UserInfo(
                    sender=self.USER,
                    name="Test User",
                    location="New York",
                    timezone="America/New_York",
                    date_of_birth="1990-01-01",
                )
            )
            session.commit()

    @pytest.mark.asyncio
    async def test_schedules_request_empty(self, tmp_path, monkeypatch):
        """Request with no schedules sends an empty list."""
        channel, _ = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._handle_schedules_request(ws)  # ty: ignore[invalid-argument-type]

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "schedules_response"
        assert resp["schedules"] == []
        assert resp["error"] is None

    @pytest.mark.asyncio
    async def test_schedules_request_returns_existing(self, tmp_path, monkeypatch):
        """Request returns all schedules for the user with correct fields."""
        channel, db = self._channel(tmp_path, monkeypatch)
        self._add_schedule(db, timing="daily 9am", prompt="check the news", cron="0 9 * * *")
        self._add_schedule(db, timing="every monday", prompt="meal ideas", cron="0 9 * * 1")

        ws = _MockWs()
        await channel._handle_schedules_request(ws)  # ty: ignore[invalid-argument-type]

        resp = ws.sent[0]
        assert len(resp["schedules"]) == 2
        first = resp["schedules"][0]
        assert first["timing_description"] == "daily 9am"
        assert first["prompt_text"] == "check the news"
        assert first["cron_expression"] == "0 9 * * *"
        assert "id" in first

    @pytest.mark.asyncio
    async def test_schedule_delete_removes_and_returns_list(self, tmp_path, monkeypatch):
        """schedule_delete removes the entry and returns the remaining list."""
        channel, db = self._channel(tmp_path, monkeypatch)
        sched1 = self._add_schedule(db, prompt="check the news")
        self._add_schedule(db, timing="every monday", prompt="meal ideas", cron="0 9 * * 1")

        ws = _MockWs()
        await channel._handle_schedule_delete(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_delete", "schedule_id": sched1.id},
        )

        resp = ws.sent[0]
        prompts = [s["prompt_text"] for s in resp["schedules"]]
        assert "check the news" not in prompts
        assert "meal ideas" in prompts

    @pytest.mark.asyncio
    async def test_schedule_delete_unknown_id_still_returns_list(self, tmp_path, monkeypatch):
        """Deleting a nonexistent ID still returns the current schedule list."""
        channel, db = self._channel(tmp_path, monkeypatch)
        self._add_schedule(db, prompt="check the news")

        ws = _MockWs()
        await channel._handle_schedule_delete(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_delete", "schedule_id": 9999},
        )

        resp = ws.sent[0]
        assert len(resp["schedules"]) == 1

    @pytest.mark.asyncio
    async def test_schedule_update_changes_prompt_text(self, tmp_path, monkeypatch):
        """schedule_update persists the new prompt text and returns the updated list."""
        channel, db = self._channel(tmp_path, monkeypatch)
        sched = self._add_schedule(db, prompt="check the news")

        ws = _MockWs()
        await channel._handle_schedule_update(
            ws,  # ty: ignore[invalid-argument-type]
            {
                "type": "schedule_update",
                "schedule_id": sched.id,
                "prompt_text": "check sports scores",
            },
        )

        resp = ws.sent[0]
        assert resp["schedules"][0]["prompt_text"] == "check sports scores"

        # Verify persisted in DB
        with Session(db.engine) as session:
            from penny.database.models import Schedule

            updated = session.get(Schedule, sched.id)
            assert updated is not None
            assert updated.prompt_text == "check sports scores"

    @pytest.mark.asyncio
    async def test_schedule_add_creates_via_llm_parsing(self, tmp_path, monkeypatch):
        """schedule_add parses the command via LLM and creates a schedule."""
        channel, db = self._channel(tmp_path, monkeypatch)
        self._add_user_info(db)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps(
            {
                "timing_description": "daily 9am",
                "prompt_text": "check the news",
                "cron_expression": "0 9 * * *",
            }
        )
        mock_client.generate.return_value = mock_response
        channel._model_client = mock_client

        ws = _MockWs()
        await channel._handle_schedule_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_add", "command": "daily 9am check the news"},
        )

        resp = ws.sent[0]
        assert resp["error"] is None
        assert len(resp["schedules"]) == 1
        assert resp["schedules"][0]["timing_description"] == "daily 9am"
        assert resp["schedules"][0]["prompt_text"] == "check the news"
        assert resp["schedules"][0]["cron_expression"] == "0 9 * * *"

    @pytest.mark.asyncio
    async def test_schedule_add_without_timezone_returns_error(self, tmp_path, monkeypatch):
        """schedule_add without user timezone returns an error."""
        channel, _ = self._channel(tmp_path, monkeypatch)

        ws = _MockWs()
        await channel._handle_schedule_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_add", "command": "daily 9am check the news"},
        )

        resp = ws.sent[0]
        assert resp["error"] is not None
        assert "timezone" in resp["error"].lower() or "profile" in resp["error"].lower()

    @pytest.mark.asyncio
    async def test_schedule_add_llm_failure_returns_error(self, tmp_path, monkeypatch):
        """schedule_add returns an error when LLM parsing fails."""
        channel, db = self._channel(tmp_path, monkeypatch)
        self._add_user_info(db)

        mock_client = AsyncMock()
        mock_client.generate.side_effect = RuntimeError("ollama down")
        channel._model_client = mock_client

        ws = _MockWs()
        await channel._handle_schedule_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_add", "command": "daily 9am check the news"},
        )

        resp = ws.sent[0]
        assert resp["error"] is not None
        assert resp["schedules"] == []

    @pytest.mark.asyncio
    async def test_schedule_add_invalid_cron_returns_error(self, tmp_path, monkeypatch):
        """schedule_add returns an error when LLM produces an invalid cron expression."""
        channel, db = self._channel(tmp_path, monkeypatch)
        self._add_user_info(db)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps(
            {
                "timing_description": "daily 9am",
                "prompt_text": "check the news",
                "cron_expression": "0 9 * *",
            }
        )
        mock_client.generate.return_value = mock_response
        channel._model_client = mock_client

        ws = _MockWs()
        await channel._handle_schedule_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "schedule_add", "command": "daily 9am check the news"},
        )

        resp = ws.sent[0]
        assert resp["error"] is not None
        assert resp["schedules"] == []

    @pytest.mark.asyncio
    async def test_schedules_request_dispatched_via_process_raw_message(
        self, tmp_path, monkeypatch
    ):
        """schedules_request type is dispatched through _process_raw_message."""
        channel, _ = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "schedules_request"}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "schedules_response"

    @pytest.mark.asyncio
    async def test_schedule_delete_dispatched_via_process_raw_message(self, tmp_path, monkeypatch):
        """schedule_delete type is dispatched through _process_raw_message."""
        channel, db = self._channel(tmp_path, monkeypatch)
        sched = self._add_schedule(db, prompt="check the news")

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "schedule_delete", "schedule_id": sched.id}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "schedules_response"
        assert ws.sent[0]["schedules"] == []


class TestBrowserPromptLogHandlers:
    """Prompt log request handlers: filtering, pagination, outcome tracking."""

    def _channel(self, tmp_path) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        return channel, db

    def _log_prompt(self, db: Database, agent_name: str, run_id: str) -> None:
        db.messages.log_prompt(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response={"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            agent_name=agent_name,
            run_id=run_id,
            duration_ms=100,
        )

    async def _request_prompt_logs(self, channel: BrowserChannel, data: dict | None = None) -> dict:
        ws = _MockWs()
        payload = {"type": "prompt_logs_request", **(data or {})}
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps(payload),
            None,
        )
        return ws.sent[0]

    @pytest.mark.asyncio
    async def test_prompt_logs_request_returns_runs(self, tmp_path):
        """Basic prompt logs request returns grouped runs."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "chat", "run1")
        self._log_prompt(db, "chat", "run1")
        self._log_prompt(db, "inner_monologue", "run2")

        response = await self._request_prompt_logs(channel)
        assert response["type"] == "prompt_logs_response"
        assert len(response["runs"]) == 2
        assert response["has_more"] is False
        assert set(response["agent_names"]) == {"chat", "inner_monologue"}

    @pytest.mark.asyncio
    async def test_prompt_logs_filter_by_agent(self, tmp_path):
        """Filtering by agent_name returns only matching runs."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "chat", "run1")
        self._log_prompt(db, "inner_monologue", "run2")

        response = await self._request_prompt_logs(channel, {"agent_name": "chat"})
        assert len(response["runs"]) == 1
        assert response["runs"][0]["agent_name"] == "chat"
        # agent_names still lists all available types
        assert set(response["agent_names"]) == {"chat", "inner_monologue"}

    @pytest.mark.asyncio
    async def test_prompt_logs_pagination(self, tmp_path):
        """Offset skips earlier runs."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "chat", "run1")
        self._log_prompt(db, "chat", "run2")
        self._log_prompt(db, "chat", "run3")

        response = await self._request_prompt_logs(channel, {"offset": 1})
        assert len(response["runs"]) == 2

    @pytest.mark.asyncio
    async def test_prompt_logs_include_run_outcome(self, tmp_path):
        """Run outcome (success / reason / target) is included in the response when set."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "collector", "run1")
        db.messages.set_run_outcome("run1", True, "wrote 2 new spots", "prague-trip")

        response = await self._request_prompt_logs(channel)
        run = response["runs"][0]
        assert run["run_success"] is True
        assert run["run_reason"] == "wrote 2 new spots"
        assert run["run_target"] == "prague-trip"

    @pytest.mark.asyncio
    async def test_prompt_logs_include_token_counts(self, tmp_path):
        """Token counts are extracted from usage in response."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "chat", "run1")

        response = await self._request_prompt_logs(channel)
        run = response["runs"][0]
        assert run["total_input_tokens"] == 10
        assert run["total_output_tokens"] == 5
        assert run["prompts"][0]["input_tokens"] == 10

    def test_set_run_outcome(self, tmp_path):
        """set_run_outcome stamps success / reason / target on the last prompt log for a run."""
        _, db = self._channel(tmp_path)
        self._log_prompt(db, "collector", "run1")
        self._log_prompt(db, "collector", "run1")
        db.messages.set_run_outcome("run1", False, "duplicate of 'test'", "likes")

        with Session(db.engine) as session:
            logs = session.exec(
                select(PromptLog).where(PromptLog.run_id == "run1").order_by(PromptLog.timestamp)  # ty: ignore[invalid-argument-type]
            ).all()

        assert logs[0].run_success is None
        assert logs[0].run_reason is None
        assert logs[0].run_target is None
        assert logs[1].run_success is False
        assert logs[1].run_reason == "duplicate of 'test'"
        assert logs[1].run_target == "likes"

    def test_on_run_outcome_set_callback(self, tmp_path):
        """_on_run_outcome_set callback fires with structured outcome when set."""
        _, db = self._channel(tmp_path)
        received: list[tuple[str, bool, str, str | None]] = []
        db.messages._on_run_outcome_set = lambda run_id, success, reason, target: received.append(
            (run_id, success, reason, target)
        )

        self._log_prompt(db, "collector", "run1")
        db.messages.set_run_outcome("run1", True, "wrote 2 new spots", "prague-trip")

        assert len(received) == 1
        assert received[0] == ("run1", True, "wrote 2 new spots", "prague-trip")

    def test_on_prompt_logged_callback(self, tmp_path):
        """_on_prompt_logged callback fires with prompt data for prompts with run_id."""
        _, db = self._channel(tmp_path)
        received: list[dict] = []
        db.messages._on_prompt_logged = lambda data: received.append(data)

        # Prompt without run_id — no callback
        db.messages.log_prompt(
            model="test",
            messages=[],
            response={},
            agent_name="chat",
        )
        assert len(received) == 0

        # Prompt with run_id — callback fires
        db.messages.log_prompt(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            response={"usage": {"prompt_tokens": 50, "completion_tokens": 20}},
            agent_name="chat",
            run_id="run1",
            duration_ms=200,
        )
        assert len(received) == 1
        assert received[0]["run_id"] == "run1"
        assert received[0]["input_tokens"] == 50
        assert received[0]["output_tokens"] == 20


class TestBrowserMemoryHandlers:
    """memories_request / memory_detail_request / memory_changed handlers."""

    def _channel(self, tmp_path) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        return channel, db

    def _seed_memories(self, db) -> None:

        # ``collector-runs`` already exists from migration 0034 — just append.
        db.memories.create_collection(
            "prague-trip",
            "Prague spots",
            RecallMode.RELEVANT,
            extraction_prompt="extract spots",
            collector_interval_seconds=300,
        )
        db.memories.write(
            "prague-trip",
            [EntryInput(key="petrin-hill", content="Hill with a tower")],
            author="user",
        )
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="[prague-trip] ✅ wrote 2 new spots")],
            author="collector",
        )

    @pytest.mark.asyncio
    async def test_memories_request_returns_collections_and_logs(self, tmp_path):
        """The list view receives every memory (collections + logs) with metadata
        and entry counts in one response."""
        channel, db = self._channel(tmp_path)
        self._seed_memories(db)

        ws = _MockWs()
        await channel._handle_memories_request(ws)  # ty: ignore[invalid-argument-type]

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "memories_response"
        by_name = {m["name"]: m for m in resp["memories"]}
        assert "prague-trip" in by_name and "collector-runs" in by_name
        assert by_name["prague-trip"]["type"] == "collection"
        assert by_name["prague-trip"]["entry_count"] == 1
        assert by_name["prague-trip"]["extraction_prompt"] == "extract spots"
        assert by_name["prague-trip"]["collector_interval_seconds"] == 300
        assert by_name["collector-runs"]["type"] == "log"
        assert by_name["collector-runs"]["entry_count"] == 1
        assert by_name["collector-runs"]["extraction_prompt"] is None

    @pytest.mark.asyncio
    async def test_memory_detail_request_returns_entries_newest_first(self, tmp_path):
        """The drill-in view returns metadata + entries (newest first, capped)."""

        channel, db = self._channel(tmp_path)
        # ``collector-runs`` is created by migration 0034 — append only.
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="first")],
            author="collector",
        )
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="second")],
            author="collector",
        )

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "collector-runs"},
        )

        resp = ws.sent[0]
        assert resp["type"] == "memory_detail_response"
        assert resp["memory"]["name"] == "collector-runs"
        assert resp["memory"]["entry_count"] == 2
        assert [e["content"] for e in resp["entries"]] == ["second", "first"]
        assert all(e["author"] == "collector" for e in resp["entries"])
        # Logs don't get collector_runs filtering — the field exists but is empty.
        assert resp["collector_runs"] == []

    @pytest.mark.asyncio
    async def test_memory_detail_request_includes_matching_collector_runs(self, tmp_path):
        """A collection's drill-in view includes the ``collector-runs``
        entries scoped to that target — newest-first, prefix-filtered."""
        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "prague-trip", "spots", RecallMode.RELEVANT, extraction_prompt="x"
        )
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="[other-target] ✅ unrelated cycle")],
            author="collector",
        )
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="[prague-trip] ✅ wrote 2 spots")],
            author="collector",
        )
        db.memories.append(
            "collector-runs",
            [LogEntryInput(content="[prague-trip] ❌ no source URL found")],
            author="collector",
        )

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "prague-trip"},
        )

        resp = ws.sent[0]
        runs = resp["collector_runs"]
        assert len(runs) == 2
        # Newest first.
        assert runs[0]["content"] == "[prague-trip] ❌ no source URL found"
        assert runs[1]["content"] == "[prague-trip] ✅ wrote 2 spots"
        # Other-target run is excluded.
        assert all("other-target" not in r["content"] for r in runs)

    @pytest.mark.asyncio
    async def test_memory_detail_request_unknown_memory_silently_drops(self, tmp_path):
        """An unknown memory name produces no response (logged warning)."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "not-a-real-memory"},
        )
        assert ws.sent == []

    def test_memory_changed_callback_fires_on_collection_write(self, tmp_path):
        """A collection write triggers the change callback so the addon can refresh."""

        _, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "spots", RecallMode.OFF)

        received: list[str | None] = []
        db.memories._on_memory_changed = lambda name: received.append(name)

        db.memories.write(
            "prague-trip",
            [EntryInput(key="petrin-hill", content="Hill")],
            author="user",
        )

        assert "prague-trip" in received

    def test_memory_changed_callback_fires_on_log_append(self, tmp_path):
        """Log appends fire the callback too — the audit log lives behind it."""

        _, db = self._channel(tmp_path)
        # ``collector-runs`` is created by migration 0034.
        received: list[str | None] = []
        db.memories._on_memory_changed = lambda name: received.append(name)

        db.memories.append("collector-runs", [LogEntryInput(content="cycle x")], author="collector")

        assert "collector-runs" in received

    # ── Edit handlers ────────────────────────────────────────────────────

    def test_memory_create_persists_collection(self, tmp_path):
        """memory_create writes a new collection to the store with the
        addon-supplied recall + extraction_prompt."""
        channel, db = self._channel(tmp_path)
        channel._handle_memory_create(
            {
                "type": "memory_create",
                "name": "prague-trip",
                "description": "Prague spots",
                "recall": "relevant",
                "extraction_prompt": "extract spots",
                "collector_interval_seconds": 600,
            }
        )
        memory = db.memories.get("prague-trip")
        assert memory is not None
        assert memory.type == "collection"
        assert memory.recall == "relevant"
        assert memory.extraction_prompt == "extract spots"
        assert memory.collector_interval_seconds == 600

    def test_memory_create_silently_drops_duplicate_name(self, tmp_path):
        """A duplicate name is logged and dropped — no crash."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "first", RecallMode.OFF)
        channel._handle_memory_create(
            {
                "type": "memory_create",
                "name": "prague-trip",
                "description": "second",
                "recall": "off",
            }
        )
        memory = db.memories.get("prague-trip")
        assert memory is not None
        assert memory.description == "first"  # original wins, no overwrite

    def test_memory_update_changes_only_supplied_fields(self, tmp_path):
        """Fields the addon doesn't send stay untouched."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "prague-trip",
            "old description",
            RecallMode.OFF,
            extraction_prompt="old prompt",
            collector_interval_seconds=300,
        )
        channel._handle_memory_update(
            {
                "type": "memory_update",
                "name": "prague-trip",
                "description": "new description",
                "recall": None,
                "extraction_prompt": None,
                "collector_interval_seconds": None,
            }
        )
        memory = db.memories.get("prague-trip")
        assert memory is not None
        assert memory.description == "new description"
        assert memory.recall == "off"
        assert memory.extraction_prompt == "old prompt"
        assert memory.collector_interval_seconds == 300

    def test_memory_archive_marks_archived(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "x", RecallMode.OFF)
        channel._handle_memory_archive({"type": "memory_archive", "name": "prague-trip"})
        memory = db.memories.get("prague-trip")
        assert memory is not None
        assert memory.archived is True

    def test_entry_create_writes_with_user_author(self, tmp_path):
        """Manual entries land with author=``user`` — distinguishes addon-
        authored from collector-authored when reading the entries list."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "x", RecallMode.OFF)
        channel._handle_entry_create(
            {
                "type": "entry_create",
                "memory": "prague-trip",
                "key": "petrin-hill",
                "content": "Hill with a tower",
            }
        )
        entries = db.memories.read_latest("prague-trip")
        assert len(entries) == 1
        assert entries[0].key == "petrin-hill"
        assert entries[0].content == "Hill with a tower"
        assert entries[0].author == "user"

    def test_entry_update_replaces_content(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "x", RecallMode.OFF)
        db.memories.write(
            "prague-trip",
            [EntryInput(key="petrin-hill", content="old")],
            author="user",
        )
        channel._handle_entry_update(
            {
                "type": "entry_update",
                "memory": "prague-trip",
                "key": "petrin-hill",
                "content": "Hill with the lookout tower",
            }
        )
        entries = db.memories.get_entry("prague-trip", "petrin-hill")
        assert entries[0].content == "Hill with the lookout tower"

    def test_entry_delete_removes_row(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("prague-trip", "x", RecallMode.OFF)
        db.memories.write(
            "prague-trip",
            [EntryInput(key="petrin-hill", content="Hill")],
            author="user",
        )
        channel._handle_entry_delete(
            {"type": "entry_delete", "memory": "prague-trip", "key": "petrin-hill"}
        )
        assert db.memories.get_entry("prague-trip", "petrin-hill") == []
