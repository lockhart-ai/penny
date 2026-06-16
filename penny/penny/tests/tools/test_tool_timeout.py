"""Tests for tool execution timeout configuration."""

import asyncio

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.llm import LlmClient
from penny.tools import ToolCall, ToolExecutor, ToolRegistry
from penny.tools.base import Tool


class SlowTool(Tool):
    """Test tool that sleeps for a configurable duration."""

    name = "slow_tool"
    description = "A tool that takes a long time"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, sleep_duration: float):
        self.sleep_duration = sleep_duration

    async def execute(self, **kwargs):
        """Sleep for the configured duration."""
        await asyncio.sleep(self.sleep_duration)
        return "completed"


class LongTimeoutTool(Tool):
    """Test tool with a per-tool timeout override."""

    name = "long_timeout_tool"
    description = "A tool with a generous per-tool timeout"
    parameters = {"type": "object", "properties": {}}
    timeout = 10.0

    async def execute(self, **kwargs):
        await asyncio.sleep(0.05)
        return "done"


class TestToolTimeout:
    """Test tool execution timeout behavior."""

    @pytest.mark.asyncio
    async def test_tool_timeout_enforced(self):
        """Tool execution should timeout after configured duration."""
        registry = ToolRegistry()
        slow_tool = SlowTool(sleep_duration=1.0)
        registry.register(slow_tool)

        # Set timeout to 0.1 seconds (tool takes 1s, well past the timeout)
        executor = ToolExecutor(registry, timeout=0.1)

        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.success is False
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tool_completes_within_timeout(self):
        """Tool execution should succeed if it completes within timeout."""
        registry = ToolRegistry()
        fast_tool = SlowTool(sleep_duration=0.1)
        registry.register(fast_tool)

        # Set timeout to 2 seconds
        executor = ToolExecutor(registry, timeout=2.0)

        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.success is True
        assert result.message == "completed"

    @pytest.mark.asyncio
    async def test_agent_uses_configured_timeout(self, test_db):
        """Agent should use tool_timeout parameter when creating ToolExecutor."""
        db = Database(test_db)
        db.create_tables()

        config = Config(
            channel_type="signal",
            signal_number="+15551234567",
            signal_api_url="http://localhost:8080",
            discord_bot_token=None,
            discord_channel_id=None,
            llm_api_url="http://localhost:11434",
            llm_model="test-model",
            log_level="DEBUG",
            db_path=test_db,
        )
        # Create agent with custom timeout
        client = LlmClient(
            api_url="http://localhost:11434",
            model="test-model",
            db=db,
            max_retries=1,
            retry_delay=0.1,
        )
        config.tool_timeout = 90.0
        agent = Agent(
            system_prompt="test",
            model_client=client,
            tools=[],
            db=db,
            config=config,
        )

        # Check that the ToolExecutor was initialized with the correct timeout
        assert agent._tool_executor.timeout == 90.0

        await agent.close()

    @pytest.mark.asyncio
    async def test_per_tool_timeout_overrides_global(self):
        """Tool.timeout takes precedence over the executor's global timeout."""
        registry = ToolRegistry()
        tool = LongTimeoutTool()
        registry.register(tool)

        # Global timeout is much shorter than the tool's own timeout.
        executor = ToolExecutor(registry, timeout=0.01)

        tool_call = ToolCall(tool="long_timeout_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.success is True
        assert result.message == "done"

    @pytest.mark.asyncio
    async def test_per_tool_timeout_respected_when_exceeded(self):
        """A per-tool timeout that is exceeded still produces a timeout error."""
        registry = ToolRegistry()
        slow = SlowTool(sleep_duration=1.0)
        slow.timeout = 0.05  # type: ignore[assignment]
        registry.register(slow)

        executor = ToolExecutor(registry, timeout=60.0)

        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.success is False
        assert "timed out" in result.message.lower()
        assert "0.05s" in result.message
