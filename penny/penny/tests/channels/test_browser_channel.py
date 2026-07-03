"""Tests for BrowserChannel message extraction and device registration."""

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.channels.base import IncomingMessage
from penny.channels.browser.channel import BrowserChannel, ConnectionInfo
from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, LogEntryInput, RecallMode
from penny.database.migrate import migrate
from penny.database.models import Media, PromptLog, RuntimeConfig
from penny.tests.conftest import wait_until
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.browse import BrowseTool
from penny.tools.models import ToolResult
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


def _data_uri(raw: bytes, mime: str = "image/jpeg") -> str:
    """Wrap raw bytes as the base64 data URI the extension returns for images."""
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _all_media(db: Database) -> list[Media]:
    with Session(db.engine) as session:
        return list(session.exec(select(Media)).all())


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
        """Tool returns a ToolResult with the channel content."""
        request_fn = AsyncMock(
            return_value=("Title: Example\nURL: https://example.com\n\nPage content.", None)
        )
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, ToolResult)
        assert "Page content." in result.message
        request_fn.assert_called_once_with("browse_url", {"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_stores_browsed_image_as_media(self, tmp_path):
        """A page image is stored in the media table with title, URL, and embedding.

        The image no longer rides along on the ToolResult — it's captured
        side-channel for the egress matcher.
        """
        db = _make_db(tmp_path)
        raw = b"\xff\xd8\xff jpeg payload"
        request_fn = AsyncMock(
            return_value=("Title: Tasty Recipe\nURL: https://ex.com/r\n\nContent.", _data_uri(raw))
        )
        tool = BrowseTool(
            max_calls=3, db=db, embedding_client=cast(Any, MockLlmClient()), author="penny"
        )
        tool.set_browse_provider(lambda: (request_fn, MagicMock(check_domain=AsyncMock())))
        result = await tool.execute(queries=["https://ex.com/r"])

        assert isinstance(result, ToolResult)
        assert not hasattr(result, "image_base64")
        rows = _all_media(db)
        assert len(rows) == 1
        assert rows[0].data == raw
        assert rows[0].mime_type == "image/jpeg"
        assert rows[0].source_url == "https://ex.com/r"
        assert rows[0].title == "Tasty Recipe"
        assert rows[0].embedding is not None

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

        assert isinstance(result, ToolResult)
        # Signal preserved
        assert "### Best guitar amps 2026" in result.message
        assert "Jan 27, 2026 The Catalyst 200" in result.message
        assert "[guitarworld.com/best-amps]" in result.message
        assert "### Second result title" in result.message
        assert "A useful snippet" in result.message
        # Cruft removed
        assert "![Favicon" not in result.message
        assert "p.kagi.com/proxy" not in result.message
        assert "[](https://" not in result.message
        assert "1920 x 1080" not in result.message
        assert "Openverse" not in result.message
        assert "[www.example.com]" not in result.message

    @pytest.mark.asyncio
    async def test_no_image_stores_no_media(self, tmp_path):
        """A page with no image stores nothing in the media table."""
        db = _make_db(tmp_path)
        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        tool = BrowseTool(
            max_calls=3, db=db, embedding_client=cast(Any, MockLlmClient()), author="penny"
        )
        tool.set_browse_provider(lambda: (request_fn, MagicMock(check_domain=AsyncMock())))
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, ToolResult)
        assert _all_media(db) == []

    @pytest.mark.asyncio
    async def test_browser_runtime_error_becomes_error_section(self, monkeypatch):
        """A RuntimeError from request_fn (structured browser failure) is surfaced
        under the dedicated error header, not the success header."""

        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        request_fn = AsyncMock(side_effect=RuntimeError("extraction failed after 10 retries"))
        tool = self._make_tool(request_fn)
        result = await tool.execute(queries=["https://example.com"])

        assert isinstance(result, ToolResult)
        assert PennyConstants.BROWSE_ERROR_HEADER + "https://example.com" in result.message
        assert "extraction failed" in result.message
        assert PennyConstants.BROWSE_PAGE_HEADER + "https://example.com" not in result.message

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

        assert PennyConstants.BROWSE_ERROR_HEADER + "https://blocked.com" in result.message
        assert "blocked" in result.message
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

        assert isinstance(result, ToolResult)
        assert PennyConstants.BROWSE_ERROR_HEADER + "https://example.com" in result.message


class TestBrowseToolMediaCapture:
    """BrowseTool stores every page's image side-channel, not just the first."""

    @pytest.mark.asyncio
    async def test_each_page_image_stored(self, tmp_path):
        """Reading two pages in one call stores two distinct media rows."""
        db = _make_db(tmp_path)
        pages = {
            "https://a.com": ("Title: A\nURL: https://a.com\n\nAlpha.", _data_uri(b"aaa")),
            "https://b.com": ("Title: B\nURL: https://b.com\n\nBeta.", _data_uri(b"bbb")),
        }

        async def request_fn(method: str, params: dict):
            return pages[params["url"]]

        tool = BrowseTool(
            max_calls=3, db=db, embedding_client=cast(Any, MockLlmClient()), author="penny"
        )
        tool.set_browse_provider(lambda: (request_fn, MagicMock(check_domain=AsyncMock())))

        result = await tool.execute(queries=["https://a.com", "https://b.com"])
        assert isinstance(result, ToolResult)
        rows = sorted(_all_media(db), key=lambda r: r.source_url or "")
        assert [r.source_url for r in rows] == ["https://a.com", "https://b.com"]
        assert [r.data for r in rows] == [b"aaa", b"bbb"]
        assert [r.title for r in rows] == ["A", "B"]


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
    async def test_heartbeat_refreshes_liveness_without_resetting_idle(self, tmp_path):
        """The keepalive heartbeat refreshes the connection's liveness timestamp
        but must NOT reset the idle timer — a ~15s ping that counted as activity
        would keep the system perpetually busy and starve the idle-gated
        collectors."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        scheduler = MagicMock()
        channel.set_scheduler(scheduler)

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-1"}),
            None,
        )
        channel._connections["firefox-1"].last_heartbeat = datetime.now(UTC) - timedelta(minutes=5)
        await channel._process_raw_message(ws, '{"type": "heartbeat"}', "firefox-1")  # ty: ignore[invalid-argument-type]

        scheduler.notify_activity.assert_not_called()
        scheduler.notify_message.assert_not_called()
        age = (datetime.now(UTC) - channel._connections["firefox-1"].last_heartbeat).total_seconds()
        assert age < 5

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
    async def test_stale_socket_cleanup_keeps_reconnected_connection(self, tmp_path):
        """A reconnect rewrites the registry to the new socket; the old socket's
        delayed cleanup must not evict that live replacement (the flapping-
        reconnect race that left the addon 'connected but unreachable')."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws_old = _MockWs()
        await channel._process_raw_message(
            ws_old,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )
        ws_new = _MockWs()
        await channel._process_raw_message(
            ws_new,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            "firefox-macbook",
        )
        assert channel._connections["firefox-macbook"].ws is ws_new

        # Old socket's handler reaches its finally after the reconnect — it must
        # leave the live ws_new entry intact.
        channel._cleanup_connection(ws_old, "firefox-macbook")  # ty: ignore[invalid-argument-type]
        assert "firefox-macbook" in channel._connections
        assert channel._connections["firefox-macbook"].ws is ws_new

        # The current socket closing for real does evict the entry.
        channel._cleanup_connection(ws_new, "firefox-macbook")  # ty: ignore[invalid-argument-type]
        assert "firefox-macbook" not in channel._connections

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

        assert not channel.has_browser_connection

        ws = await self._register(channel, "firefox-1")
        assert channel.has_browser_connection
        assert not channel.has_tool_connection

        await self._set_capabilities(channel, "firefox-1", ws, True)
        assert channel.has_tool_connection

    @pytest.mark.asyncio
    async def test_send_tool_request_explains_connected_but_tool_use_disabled(self, tmp_path):
        """A connected browser without tool-use reports the actionable cause."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        await self._register(channel, "firefox-1")

        with pytest.raises(RuntimeError, match="tool use is disabled"):
            await channel.send_tool_request("browse_url", {"url": "https://example.com"})

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

    @pytest.mark.asyncio
    async def test_stale_lone_connection_still_routed_as_fallback(self, tmp_path):
        """A lone tool connection that has gone quiet is still used — it may be
        an older addon without the keepalive rather than a dead one, so staleness
        only *deprioritizes* a socket when a fresher alternative exists; it never
        declares the sole browser offline."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = await self._register(channel, "firefox-1")
        await self._set_capabilities(channel, "firefox-1", ws, True)

        # Age the heartbeat past the liveness window — still the only connection.
        channel._connections["firefox-1"].last_heartbeat = datetime.now(UTC) - timedelta(
            seconds=PennyConstants.BROWSER_HEARTBEAT_TIMEOUT_SECONDS + 5
        )
        assert channel.has_tool_connection
        assert channel._get_tool_connection() is ws

    @pytest.mark.asyncio
    async def test_get_tool_connection_prefers_live_over_stale(self, tmp_path):
        """With one stale and one live tool connection, routing picks the live
        one rather than the stale (most-recent-heartbeat) socket."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws_stale = await self._register(channel, "firefox-stale")
        await self._set_capabilities(channel, "firefox-stale", ws_stale, True)
        ws_live = await self._register(channel, "firefox-live")
        await self._set_capabilities(channel, "firefox-live", ws_live, True)

        channel._connections["firefox-stale"].last_heartbeat = datetime.now(UTC) - timedelta(
            seconds=PennyConstants.BROWSER_HEARTBEAT_TIMEOUT_SECONDS + 5
        )
        assert channel._get_tool_connection() is ws_live


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

    def _log_prompt(
        self, db: Database, agent_name: str, run_id: str, run_target: str | None = None
    ) -> None:
        db.messages.log_prompt(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response={"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            agent_name=agent_name,
            run_id=run_id,
            run_target=run_target,
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
        # The agent-filter chips are a static frontend list — the response
        # carries only runs, not a scanned set of agent names.
        assert "agent_names" not in response

    @pytest.mark.asyncio
    async def test_prompt_logs_filter_by_agent(self, tmp_path):
        """Filtering by agent_name returns only matching runs."""
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "chat", "run1")
        self._log_prompt(db, "inner_monologue", "run2")

        response = await self._request_prompt_logs(channel, {"agent_name": "chat"})
        assert len(response["runs"]) == 1
        assert response["runs"][0]["agent_name"] == "chat"

    @pytest.mark.asyncio
    async def test_prompt_logs_query_matches_output_not_input(self, tmp_path):
        """A text query matches a run's response/thinking (what it produced),
        not its input messages (shared scaffolding that would match noise)."""
        channel, db = self._channel(tmp_path)
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            response={"choices": [{"message": {"content": "here is info on basketball"}}]},
            agent_name="chat",
            run_id="hoops",
            duration_ms=1,
        )
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            response={"choices": [{"message": {"content": "here is info on gardening"}}]},
            agent_name="chat",
            run_id="plants",
            duration_ms=1,
        )
        # "reuters" only in the INPUT — must NOT match (this is the noise we cut).
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": "what did reuters say"}],
            response={"choices": [{"message": {"content": "no idea"}}]},
            agent_name="chat",
            run_id="noise",
            duration_ms=1,
        )

        response = await self._request_prompt_logs(channel, {"query": "basketball"})
        assert {r["run_id"] for r in response["runs"]} == {"hoops"}

        response = await self._request_prompt_logs(channel, {"query": "reuters"})
        assert response["runs"] == []

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
    async def test_prompt_logs_has_more_paginates(self, tmp_path):
        """A full first page sets ``has_more``; the trailing page clears it.
        Exercises the SQL run-level pagination — only one page of runs is ever
        materialized, regardless of total history size."""
        channel, db = self._channel(tmp_path)
        channel._PROMPT_LOG_PAGE_SIZE = 2  # shrink so 3 runs span 2 pages
        for run_id in ("run1", "run2", "run3"):
            self._log_prompt(db, "chat", run_id)

        first = await self._request_prompt_logs(channel)
        assert len(first["runs"]) == 2
        assert first["has_more"] is True

        second = await self._request_prompt_logs(channel, {"offset": 2})
        assert len(second["runs"]) == 1
        assert second["has_more"] is False

    @pytest.mark.asyncio
    async def test_prompt_logs_include_run_outcome(self, tmp_path):
        """Run outcome (outcome / reason / target) is included in the response when set.

        run_target is read straight off the prompt rows (every prompt carries it
        at write time), independent of the outcome — so an outcome-less run
        (single prompt, in-progress, or never tagged) still reports its bound
        collection rather than dropping it and rendering the bare agent identity.
        """
        channel, db = self._channel(tmp_path)
        self._log_prompt(db, "collector", "run1", run_target="board-games")
        db.messages.set_run_outcome("run1", "worked", "wrote 2 new games")
        # A second run that never got an outcome tagged (e.g. still running).
        self._log_prompt(db, "collector", "run2", run_target="knowledge")

        response = await self._request_prompt_logs(channel)
        runs = {run["run_id"]: run for run in response["runs"]}
        assert runs["run1"]["run_outcome"] == "worked"
        assert runs["run1"]["run_reason"] == "wrote 2 new games"
        assert runs["run1"]["run_target"] == "board-games"
        # Outcome-less run keeps its collection name (the bug: it was dropped).
        assert runs["run2"]["run_outcome"] is None
        assert runs["run2"]["run_target"] == "knowledge"
        # Each run carries the shared run-health (badges + filter) and the concise
        # record (the SAME representation Penny's quality collector reads).
        assert runs["run1"]["health"]["regressive"] is False
        assert runs["run1"]["health"]["flags"] == []
        assert runs["run1"]["record"].startswith("[board-games] wrote 2 new games")

    @pytest.mark.asyncio
    async def test_prompt_logs_flagged_only(self, tmp_path):
        """flagged_only returns only regressive runs — and carries their flags.

        A half-formed send on an otherwise-worked run (the real notifier shape)
        is regressive; a clean worked run is not."""
        channel, db = self._channel(tmp_path)
        # Healthy worked run.
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            response={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c0",
                                    "type": "function",
                                    "function": {
                                        "name": "send_message",
                                        "arguments": '{"content": "A new title dropped."}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            agent_name="collector",
            run_id="good",
            run_target="games",
            duration_ms=1,
        )
        db.messages.set_run_outcome("good", "worked", "delivered a notification")
        # Worked run that ALSO sent a half-formed message.
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            response={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c0",
                                    "type": "function",
                                    "function": {
                                        "name": "send_message",
                                        "arguments": '{"content": "Hi there! ......???"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            agent_name="collector",
            run_id="bad",
            run_target="games",
            duration_ms=1,
        )
        db.messages.set_run_outcome("bad", "worked", "delivered a notification")

        flagged = await self._request_prompt_logs(channel, {"flagged_only": True})
        assert {r["run_id"] for r in flagged["runs"]} == {"bad"}
        assert flagged["runs"][0]["health"]["degenerate_send"] is True
        assert "half_formed_send" in flagged["runs"][0]["health"]["flags"]

        # Without the filter, both runs come back.
        both = await self._request_prompt_logs(channel)
        assert {r["run_id"] for r in both["runs"]} == {"good", "bad"}

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
        """set_run_outcome stamps outcome / reason on the last prompt log for a run.

        (run_target is stamped on every prompt at write time, not here — see
        test_log_prompt_stamps_run_target.)"""
        _, db = self._channel(tmp_path)
        self._log_prompt(db, "collector", "run1")
        self._log_prompt(db, "collector", "run1")
        db.messages.set_run_outcome("run1", "failed", "duplicate of 'test'", tool_failures=3)

        with Session(db.engine) as session:
            logs = session.exec(
                select(PromptLog).where(PromptLog.run_id == "run1").order_by(PromptLog.timestamp)  # ty: ignore[invalid-argument-type]
            ).all()

        assert logs[0].run_outcome is None
        assert logs[0].run_reason is None
        assert logs[0].tool_failures is None
        assert logs[1].run_outcome == "failed"
        assert logs[1].run_reason == "duplicate of 'test'"
        # Failed-tool count is stamped on the same last row as the outcome.
        assert logs[1].tool_failures == 3

    def test_log_prompt_stamps_run_target(self, tmp_path):
        """run_target is written on the prompt row at log time (every prompt),
        so a run is attributable to its collection from its first prompt."""
        _, db = self._channel(tmp_path)
        self._log_prompt(db, "collector", "run1", run_target="board-games")

        with Session(db.engine) as session:
            log = session.exec(
                select(PromptLog).where(PromptLog.run_id == "run1")  # ty: ignore[invalid-argument-type]
            ).first()

        assert log is not None
        assert log.run_target == "board-games"

    def test_on_run_outcome_set_callback(self, tmp_path):
        """_on_run_outcome_set callback fires with the structured outcome when set."""
        _, db = self._channel(tmp_path)
        received: list[tuple[str, str, str]] = []
        db.messages._on_run_outcome_set = lambda run_id, outcome, reason: received.append(
            (run_id, outcome, reason)
        )

        self._log_prompt(db, "collector", "run1")
        db.messages.set_run_outcome("run1", "worked", "wrote 2 new games")

        assert len(received) == 1
        assert received[0] == ("run1", "worked", "wrote 2 new games")

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


def _seed_collector_run(db, run_target: str, run_id: str, outcome: str, reason: str) -> None:
    """Seed one completed promptlog run — ``collector-runs`` is a read facade
    over ``promptlog``, so the addon's run views come from there.  A worked run
    renders ``[target] reason`` + its tool trace; a non-worked run that did real
    work (or a quiet cycle) renders the header only; a non-worked run that
    recorded no tool call at all (it spun until the step ceiling) renders the
    header + a ``⚠ INCOMPLETE`` flag — capacity, not a deliberate bail."""
    db.messages.log_prompt(
        model="m",
        messages=[],
        response={"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]},
        agent_name="collector",
        run_id=run_id,
        run_target=run_target,
    )
    db.messages.set_run_outcome(run_id, outcome, reason)


class TestBrowserMemoryHandlers:
    """memories_request / memory_detail_request / memory_changed handlers."""

    def _channel(self, tmp_path) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        # The create/edit handlers re-embed the description via the agent;
        # no embedding model in tests → an awaitable that returns None.
        agent = MagicMock()
        agent.embed_description = AsyncMock(return_value=None)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=agent, db=db)
        return channel, db

    def _seed_memories(self, db) -> None:
        db.memories.create_collection(
            "board-games",
            "board games",
            Inclusion.RELEVANT,
            RecallMode.RELEVANT,
            extraction_prompt="extract games",
            collector_interval_seconds=300,
        )
        db.memory("board-games").write(
            [EntryInput(key="catan", content="Gateway strategy game")],
            author="user",
        )
        # collector-runs is a facade over promptlog — seed a run, not an entry.
        _seed_collector_run(db, "board-games", "seed-run", "worked", "wrote 2 new games")

    @pytest.mark.asyncio
    async def test_memories_request_returns_collections_and_logs(self, tmp_path):
        """The list view receives every memory (collections + logs) with metadata
        and entry counts in one response."""
        channel, db = self._channel(tmp_path)
        self._seed_memories(db)

        ws = _MockWs()
        await channel._handle_memories_request(ws, {"type": "memories_request"})  # ty: ignore[invalid-argument-type]

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "memories_response"
        by_name = {m["name"]: m for m in resp["memories"]}
        assert "board-games" in by_name and "collector-runs" in by_name
        assert by_name["board-games"]["type"] == "collection"
        assert by_name["board-games"]["entry_count"] == 1
        assert by_name["board-games"]["extraction_prompt"] == "extract games"
        assert by_name["board-games"]["collector_interval_seconds"] == 300
        assert by_name["collector-runs"]["type"] == "log"
        assert by_name["collector-runs"]["entry_count"] == 1
        assert by_name["collector-runs"]["extraction_prompt"] is None

    @pytest.mark.asyncio
    async def test_memories_request_query_matches_name_or_entry_content(self, tmp_path):
        """The query keeps memories matching by metadata OR by entry content —
        so searching an item inside a collection surfaces that collection."""
        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "board-games", "tabletop strategy", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.create_collection(
            "espresso-gear", "coffee equipment", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memory("espresso-gear").write(
            [EntryInput(key="grinder", content="Niche Zero burr grinder")],
            author="test",
        )

        ws = _MockWs()
        await channel._handle_memories_request(ws, {"query": "board"})  # ty: ignore[invalid-argument-type]
        names = {m["name"] for m in ws.sent[0]["memories"]}
        assert "board-games" in names  # matched by name
        assert "espresso-gear" not in names  # no "board" anywhere

        # "niche" appears only in an espresso-gear entry, not its metadata.
        ws = _MockWs()
        await channel._handle_memories_request(ws, {"query": "niche"})  # ty: ignore[invalid-argument-type]
        names = {m["name"] for m in ws.sent[0]["memories"]}
        assert "espresso-gear" in names
        assert "board-games" not in names

    @pytest.mark.asyncio
    async def test_memory_detail_query_filters_entries(self, tmp_path):
        """Drilling into a collection while a search is active filters its
        entries section to matches — but metadata keeps the full count."""
        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "espresso-gear", "coffee", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memory("espresso-gear").write(
            [
                EntryInput(key="grinder", content="Niche Zero burr grinder"),
                EntryInput(key="machine", content="Gaggia Classic espresso machine"),
            ],
            author="test",
        )

        ws = _MockWs()
        data = {"type": "memory_detail_request", "name": "espresso-gear", "query": "niche"}
        await channel._handle_memory_detail_request(ws, data)  # ty: ignore[invalid-argument-type]
        resp = ws.sent[0]
        assert {e["key"] for e in resp["entries"]} == {"grinder"}
        assert resp["memory"]["entry_count"] == 2  # full count, not the filtered subset

    @pytest.mark.asyncio
    async def test_memory_detail_request_returns_entries_newest_first(self, tmp_path):
        """The drill-in view returns metadata + entries (newest first, capped)."""

        channel, db = self._channel(tmp_path)
        # collector-runs is a facade over promptlog — its entries + count are runs.
        _seed_collector_run(db, "knowledge", "r0", "worked", "first")
        _seed_collector_run(db, "knowledge", "r1", "worked", "second")

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "collector-runs"},
        )

        resp = ws.sent[0]
        assert resp["type"] == "memory_detail_response"
        assert resp["memory"]["name"] == "collector-runs"
        assert resp["memory"]["entry_count"] == 2
        assert [e["content"] for e in resp["entries"]] == [
            "[knowledge] second",
            "[knowledge] first",
        ]
        # The run's timestamp lives in the entry's created_at field, not the record text.
        assert all(e["created_at"] for e in resp["entries"])
        assert all(e["author"] == "collector" for e in resp["entries"])
        # Both sections fit in one page → no "load more".
        assert resp["entries_has_more"] is False
        # Logs don't get the per-collection collector_runs panel — that's empty.
        assert resp["collector_runs"] == []
        assert resp["collector_runs_has_more"] is False

    @pytest.mark.asyncio
    async def test_memory_detail_request_includes_matching_collector_runs(self, tmp_path):
        """A collection's drill-in view includes its collector runs scoped to
        that target — newest-first, as full runs (the Activity tab renders them
        as the prompts tab's run → prompts → turns cards)."""
        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "board-games", "games", Inclusion.RELEVANT, RecallMode.RELEVANT, extraction_prompt="x"
        )
        _seed_collector_run(db, "other-target", "r0", "worked", "unrelated cycle")
        _seed_collector_run(db, "board-games", "r1", "worked", "wrote 2 games")
        _seed_collector_run(db, "board-games", "r2", "failed", "no source URL found")

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "board-games"},
        )

        resp = ws.sent[0]
        runs = resp["collector_runs"]
        assert len(runs) == 2
        # Full serialized runs (run_id, prompts, health, record), newest first.
        assert all(r["run_target"] == "board-games" for r in runs)
        assert all(r["prompts"] for r in runs)
        # The shared run record carries the [target] summary + run-health flags.
        # r2 failed with no tool calls — it spun until the step ceiling, which is
        # capacity (⚠ INCOMPLETE, ignored by quality), not a deliberate NO WORK
        # DONE bail (which requires a recorded done() with no real work first).
        assert "[board-games] no source URL found" in runs[0]["record"]
        assert "⚠ INCOMPLETE" in runs[0]["record"]
        assert runs[1]["record"] == "[board-games] wrote 2 games"
        assert runs[1]["run_reason"] == "wrote 2 games"
        # Both target runs fit in one page.
        assert resp["collector_runs_has_more"] is False

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

    @pytest.mark.asyncio
    async def test_memory_detail_entries_paginate(self, tmp_path):
        """A log with more entries than one page returns the first page plus
        ``entries_has_more``; ``memory_page_request`` walks the rest
        newest-first without ever loading the whole history at once."""
        channel, db = self._channel(tmp_path)
        channel._MEMORY_PAGE_SIZE = 2  # shrink the page so 3 entries span 2 pages
        # collector-runs is a facade over promptlog — its "entries" are runs.
        for index, marker in enumerate(("first", "second", "third")):
            _seed_collector_run(db, "knowledge", f"r{index}", "worked", marker)

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "collector-runs"},
        )
        first = ws.sent[0]
        assert [e["content"] for e in first["entries"]] == [
            "[knowledge] third",
            "[knowledge] second",
        ]
        assert first["entries_has_more"] is True

        ws = _MockWs()
        await channel._handle_memory_page_request(
            ws,  # ty: ignore[invalid-argument-type]
            {
                "type": "memory_page_request",
                "name": "collector-runs",
                "section": "entries",
                "offset": 2,
            },
        )
        page = ws.sent[0]
        assert page["type"] == "memory_page_response"
        assert page["section"] == "entries"
        assert [e["content"] for e in page["entries"]] == ["[knowledge] first"]
        assert page["has_more"] is False

    @pytest.mark.asyncio
    async def test_memory_page_request_collector_runs_paginate(self, tmp_path):
        """The Activity section paginates independently of the entries section,
        scoped to the collection's target, carried in the ``runs`` field."""
        channel, db = self._channel(tmp_path)
        channel._MEMORY_PAGE_SIZE = 2
        db.memories.create_collection(
            "board-games", "games", Inclusion.RELEVANT, RecallMode.RELEVANT, extraction_prompt="x"
        )
        for index, marker in enumerate(("cycle-1", "cycle-2", "cycle-3")):
            _seed_collector_run(db, "board-games", f"r{index}", "worked", marker)

        ws = _MockWs()
        await channel._handle_memory_detail_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_detail_request", "name": "board-games"},
        )
        first = ws.sent[0]
        assert [r["record"] for r in first["collector_runs"]] == [
            "[board-games] cycle-3",
            "[board-games] cycle-2",
        ]
        assert first["collector_runs_has_more"] is True

        ws = _MockWs()
        await channel._handle_memory_page_request(
            ws,  # ty: ignore[invalid-argument-type]
            {
                "type": "memory_page_request",
                "name": "board-games",
                "section": "collector_runs",
                "offset": 2,
            },
        )
        page = ws.sent[0]
        assert page["section"] == "collector_runs"
        assert [r["record"] for r in page["runs"]] == ["[board-games] cycle-1"]
        assert page["has_more"] is False

    @pytest.mark.asyncio
    async def test_memory_page_request_unknown_memory_silently_drops(self, tmp_path):
        """A page request for an unknown memory produces no response."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_memory_page_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "memory_page_request", "name": "nope", "section": "entries", "offset": 0},
        )
        assert ws.sent == []

    def test_memory_changed_callback_fires_on_collection_write(self, tmp_path):
        """A collection write triggers the change callback so the addon can refresh."""

        _, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "games", Inclusion.NEVER, RecallMode.RECENT)

        received: list[str | None] = []
        db.memories._on_memory_changed = lambda name: received.append(name)

        db.memory("board-games").write(
            [EntryInput(key="catan", content="A classic")],
            author="user",
        )

        assert "board-games" in received

    def test_memory_changed_callback_fires_on_log_append(self, tmp_path):
        """Log appends fire the callback too — a writable (non-facade) log."""

        _, db = self._channel(tmp_path)
        db.memories.create_log("notes", "scratch log", Inclusion.ALWAYS, RecallMode.RECENT)
        received: list[str | None] = []
        db.memories._on_memory_changed = lambda name: received.append(name)

        db.memory("notes").append([LogEntryInput(content="cycle x")], author="user")

        assert "notes" in received

    # ── On-demand extractor trigger ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_collection_trigger_runs_extractor(self, tmp_path):
        """The trigger delegates to the collector's on-demand ``run_for`` and
        reports the outcome back to the requesting addon."""
        channel, db = self._channel(tmp_path)
        collector = MagicMock()
        collector.run_for = AsyncMock(
            return_value=(True, "Collector cycle complete. wrote 2 games")
        )
        channel.set_collector(collector)

        ws = _MockWs()
        await channel._handle_collection_trigger(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "collection_trigger", "name": "board-games"},
        )

        collector.run_for.assert_awaited_once_with("board-games")
        resp = ws.sent[0]
        assert resp["type"] == "collection_trigger_result"
        assert resp["name"] == "board-games"
        assert resp["success"] is True
        assert "wrote 2 games" in resp["message"]

    @pytest.mark.asyncio
    async def test_collection_trigger_without_collector_reports_failure(self, tmp_path):
        """With no collector wired, the trigger fails loudly rather than silently."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_collection_trigger(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "collection_trigger", "name": "board-games"},
        )
        resp = ws.sent[0]
        assert resp["success"] is False
        assert resp["message"] == "Collector is not available."

    # ── Edit handlers ────────────────────────────────────────────────────

    def test_memory_create_persists_collection(self, tmp_path):
        """memory_create writes a new collection to the store with the
        addon-supplied recall + extraction_prompt."""
        channel, db = self._channel(tmp_path)
        asyncio.run(
            channel._handle_memory_create(
                {
                    "type": "memory_create",
                    "name": "board-games",
                    "description": "board games",
                    "inclusion": "relevant",
                    "recall": "relevant",
                    "published": True,
                    "extraction_prompt": "extract games",
                    "collector_interval_seconds": 600,
                }
            )
        )
        memory = db.memories.get("board-games")
        assert memory is not None
        assert memory.type == "collection"
        assert memory.inclusion == "relevant"
        assert memory.recall == "relevant"
        assert memory.published is True  # pub/sub flag persisted from the addon
        assert memory.extraction_prompt == "extract games"
        assert memory.collector_interval_seconds == 600

    def test_memory_create_silently_drops_duplicate_name(self, tmp_path):
        """A duplicate name is logged and dropped — no crash."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "first", Inclusion.NEVER, RecallMode.RECENT)
        asyncio.run(
            channel._handle_memory_create(
                {
                    "type": "memory_create",
                    "name": "board-games",
                    "description": "second",
                    "recall": "off",
                }
            )
        )
        memory = db.memories.get("board-games")
        assert memory is not None
        assert memory.description == "first"  # original wins, no overwrite

    def test_memory_update_changes_only_supplied_fields(self, tmp_path):
        """Fields the addon doesn't send stay untouched."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection(
            "board-games",
            "old description",
            Inclusion.NEVER,
            RecallMode.RECENT,
            extraction_prompt="old prompt",
            collector_interval_seconds=300,
            intent="track heavy euro strategy games",
        )
        asyncio.run(
            channel._handle_memory_update(
                {
                    "type": "memory_update",
                    "name": "board-games",
                    "description": "new description",
                    # intent is user-editable via this path (unlike the agent's
                    # collection_update tool, which has no intent field).
                    "intent": "track only co-op games now",
                    "recall": None,
                    "published": True,  # flip notify-on-new; other unsupplied fields stay put
                    "extraction_prompt": None,
                    "collector_interval_seconds": None,
                }
            )
        )
        memory = db.memories.get("board-games")
        assert memory is not None
        assert memory.description == "new description"
        assert memory.intent == "track only co-op games now"
        assert memory.recall == "recent"
        assert memory.published is True  # supplied flag applied
        assert memory.extraction_prompt == "old prompt"
        assert memory.collector_interval_seconds == 300

    def test_memory_archive_marks_archived(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "x", Inclusion.NEVER, RecallMode.RECENT)
        channel._handle_memory_archive({"type": "memory_archive", "name": "board-games"})
        memory = db.memories.get("board-games")
        assert memory is not None
        assert memory.archived is True

    def test_cursor_set_then_clear(self, tmp_path):
        """The addon can move a collection's read cursor over a log to a chosen
        point (a backward-capable user override) and clear it back to unset."""
        channel, db = self._channel(tmp_path)
        db.memories.create_collection("journal", "x", Inclusion.NEVER, RecallMode.RECENT)
        target = datetime(2026, 3, 1, tzinfo=UTC)
        channel._handle_cursor_set(
            {
                "type": "cursor_set",
                "name": "journal",
                "log_name": "user-messages",
                "last_read_at": target.isoformat(),
            }
        )
        assert db.cursors.get("journal", "user-messages") == target
        # list_for powers the detail view's cursor rows.
        assert db.cursors.list_for("journal") == [("user-messages", target)]

        channel._handle_cursor_clear(
            {"type": "cursor_clear", "name": "journal", "log_name": "user-messages"}
        )
        assert db.cursors.get("journal", "user-messages") is None

    def test_entry_create_writes_with_user_author(self, tmp_path):
        """Manual entries land with author=``user`` — distinguishes addon-
        authored from collector-authored when reading the entries list."""

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "x", Inclusion.NEVER, RecallMode.RECENT)
        channel._handle_entry_create(
            {
                "type": "entry_create",
                "memory": "board-games",
                "key": "catan",
                "content": "Gateway strategy game",
            }
        )
        entries = db.memory("board-games").read_latest()
        assert len(entries) == 1
        assert entries[0].key == "catan"
        assert entries[0].content == "Gateway strategy game"
        assert entries[0].author == "user"

    def test_entry_update_replaces_content(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "x", Inclusion.NEVER, RecallMode.RECENT)
        db.memory("board-games").write(
            [EntryInput(key="catan", content="old")],
            author="user",
        )
        channel._handle_entry_update(
            {
                "type": "entry_update",
                "memory": "board-games",
                "key": "catan",
                "content": "Gateway strategy game, updated",
            }
        )
        entries = db.memory("board-games").get("catan")
        assert entries[0].content == "Gateway strategy game, updated"

    def test_entry_delete_removes_row(self, tmp_path):

        channel, db = self._channel(tmp_path)
        db.memories.create_collection("board-games", "x", Inclusion.NEVER, RecallMode.RECENT)
        db.memory("board-games").write(
            [EntryInput(key="catan", content="A classic")],
            author="user",
        )
        channel._handle_entry_delete(
            {"type": "entry_delete", "memory": "board-games", "key": "catan"}
        )
        assert db.memory("board-games").get("catan") == []
