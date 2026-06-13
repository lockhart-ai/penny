"""Pytest fixtures for Penny tests."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.config import Config
from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams
from penny.penny import Penny

# Re-export LLM mock fixture so it can be used directly in tests
from penny.tests.mocks.llm_patches import mock_llm  # noqa: F401
from penny.tests.mocks.signal_server import MockSignalServer

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# Standard test sender phone number
TEST_SENDER = "+15559876543"

# Default config values for tests (background tasks disabled)
DEFAULT_TEST_CONFIG = {
    "channel_type": "signal",
    "signal_number": "+15551234567",
    "discord_bot_token": None,
    "discord_channel_id": None,
    "llm_api_url": "http://localhost:11434",
    "llm_model": "test-model",
    "log_level": "DEBUG",
    "tool_timeout": 60.0,
    # Fast scheduler ticks for tests
    "scheduler_tick_interval": 0.05,
    # Fast retries for tests
    "llm_max_retries": 1,
    "llm_retry_delay": 0.1,
}

# Default runtime param overrides for tests (disable background tasks)
DEFAULT_TEST_RUNTIME_OVERRIDES: dict[str, int | float] = {
    "IDLE_SECONDS": 99999.0,
    # Bump every background-agent interval past any test timeout so the
    # scheduler never fires them mid-test.
    "COLLECTOR_TICK_INTERVAL": 99999.0,
}


async def wait_until(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.05,
) -> None:
    """
    Poll a condition until it becomes true, or raise TimeoutError.

    Replaces arbitrary ``asyncio.sleep(N)`` calls in tests with deterministic,
    condition-based waiting that returns as soon as the expected state is reached.

    Args:
        condition: Synchronous callable that returns True when ready.
        timeout: Maximum seconds to wait before raising TimeoutError.
        interval: Seconds between polls.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"Condition not met within {timeout}s")


@pytest.fixture
async def signal_server():
    """Start a mock Signal server and yield it."""
    server = MockSignalServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def make_config(signal_server, test_db) -> Callable[..., Config]:
    """
    Factory fixture for creating test configs with custom overrides.

    Usage:
        config = make_config()  # defaults
        config = make_config(summarize_idle_seconds=0.5)  # with override
    """

    def _make_config(**overrides: Any) -> Config:
        # Separate runtime param overrides from Config kwargs
        runtime_overrides = dict(DEFAULT_TEST_RUNTIME_OVERRIDES)
        config_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if key.upper() in RUNTIME_CONFIG_PARAMS:
                runtime_overrides[key.upper()] = value
            else:
                config_overrides[key] = value

        config_kwargs: dict[str, Any] = {
            **DEFAULT_TEST_CONFIG,
            "signal_api_url": f"http://localhost:{signal_server.port}",
            "db_path": test_db,
            "runtime": RuntimeParams(env_overrides=runtime_overrides),
            **config_overrides,
        }
        return Config(**cast(Any, config_kwargs))

    return _make_config


@pytest.fixture
def test_config(make_config) -> Config:
    """
    Create a test Config pointing to mock servers.

    Background schedules are disabled by setting high idle times.
    For custom configs, use make_config fixture instead.
    """
    return make_config()


@pytest.fixture
def test_user_info(test_config):
    """
    Create a test user profile to bypass profile prompting.

    This sets up a UserInfo record for TEST_SENDER so tests don't get
    intercepted by profile collection prompts. The DB is initialized (tables
    created, then migrations run) before creating the user.
    """
    from penny.database import Database
    from penny.database.migrate import migrate

    # Create database and tables first
    db = Database(test_config.db_path)
    db.create_tables()

    # Then run migrations
    migrate(test_config.db_path)

    # Now create the test user
    db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )

    # Register the test sender as a Signal device
    from penny.constants import ChannelType

    db.devices.register(ChannelType.SIGNAL, TEST_SENDER, "Test Signal", is_default=True)

    return db


@pytest.fixture
def running_penny(signal_server) -> Callable[[Config], AbstractAsyncContextManager[Penny]]:
    """
    Async context manager fixture for running Penny with proper cleanup.

    Usage:
        async with running_penny(config) as penny:
            # penny is running and ready
            await signal_server.push_message(...)
    """

    @asynccontextmanager
    async def _running_penny(config: Config) -> AsyncIterator[Penny]:
        penny = Penny(config)
        penny_task = asyncio.create_task(penny.run())
        try:
            # Wait for WebSocket connection to establish
            await wait_until(lambda: len(signal_server._websockets) > 0)

            # Mock browse provider on all agents so tool calls don't hit
            # real retry/sleep loops when no browser extension is connected
            def mock_browse():
                return (
                    AsyncMock(return_value=("Mock search results", "data:image/png;base64,mock")),
                    MagicMock(check_domain=AsyncMock()),
                )

            penny.chat_agent._browse_provider = mock_browse
            penny.collector._browse_provider = mock_browse

            yield penny
        finally:
            penny_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await penny_task
            await penny.shutdown()

    return _running_penny


@pytest.fixture
def setup_llm_flow(mock_llm):  # noqa: F811
    """
    Factory fixture to configure mock_llm for a standard message + background task flow.

    Sets up a multi-phase handler:
    1. First call: tool call (search) with given query
    2. Second call: message response
    3. Third call onwards: background task response

    Usage:
        setup_llm_flow(
            message_response="here's the weather!",
            background_response="background task response (optional)",
        )
    """

    def _setup(
        message_response: str,
        background_response: str = "",
        search_query: str = "test query",
    ) -> None:
        request_count = [0]

        def multi_phase_handler(request: dict, count: int) -> dict:
            request_count[0] += 1
            if request_count[0] == 1:
                return mock_llm._make_tool_call_response(
                    request, "browse", {"queries": [search_query]}
                )
            elif request_count[0] == 2:
                return mock_llm._make_text_response(request, message_response)
            else:
                return mock_llm._make_text_response(request, background_response)

        mock_llm.set_response_handler(multi_phase_handler)

    return _setup
