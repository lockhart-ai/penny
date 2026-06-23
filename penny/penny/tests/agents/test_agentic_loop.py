"""Tests for agentic loop changes: reasoning, last step, and after_step hook."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from penny.agents.base import Agent, BackgroundAgent
from penny.config import Config
from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm import LlmClient
from penny.llm.models import LlmConnectionError, LlmTimeoutError, LlmToolParseError
from penny.responses import PennyResponse
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool, _trim_search_result
from penny.tools.memory_tools import DoneTool
from penny.tools.models import ToolResult


class StubSearchTool(Tool):
    """Minimal stub tool for agentic loop testing."""

    name = "search"
    description = "Search for information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }

    async def execute(self, **kwargs):
        return ToolResult(message="Mock search results for testing")


def _make_agent(test_db, mock_llm, *, max_steps=3, runtime_overrides=None):
    """Create a minimal Agent for loop testing.

    Returns (agent, db, max_steps) — max_steps must be passed to agent.run().
    Pass runtime_overrides={key: value} to override runtime config params for the test.
    """
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
        runtime=RuntimeParams(db=db, env_overrides=runtime_overrides or {}),
    )
    stub_tool = StubSearchTool()
    client = LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.1,
    )
    agent = Agent(
        system_prompt="test",
        model_client=client,
        tools=[stub_tool],
        db=db,
        config=config,
    )
    # These tests exercise the "strip tools on final step → force text"
    # path that powers chat agent's final-answer reply mechanism.
    # Subagents (notify, thinking, etc.) keep tools on the final step
    # because they exit via a terminator tool call (done / send_message).
    agent._keep_tools_on_final_step = False
    return agent, db, max_steps


class TestReasoningStripped:
    """Test that reasoning is popped from tool arguments and stored on the record."""

    @pytest.mark.asyncio
    async def test_reasoning_captured_on_tool_call_record(self, test_db, mock_llm):
        """Reasoning from tool call args is stored on ToolCallRecord."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request,
                    "search",
                    {"query": "weather", "reasoning": "User asked about weather"},
                )
            return mock_llm._make_text_response(request, "here's the weather!")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what's the weather?", max_steps=max_steps)
        assert response.answer == "here's the weather!"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning == "User asked about weather"
        # reasoning should NOT be in the arguments dict
        assert "reasoning" not in response.tool_calls[0].arguments

        await agent.close()

    @pytest.mark.asyncio
    async def test_reasoning_none_when_not_provided(self, test_db, mock_llm):
        """ToolCallRecord.reasoning is None when model doesn't provide it."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "weather"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning is None

        await agent.close()


class TestLastStepToolRemoval:
    """Test that on the final step, tools are removed so the model must produce text."""

    @pytest.mark.asyncio
    async def test_final_step_has_no_tools(self, test_db, mock_llm):
        """On the last step, the model is called without tools."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                # Step 1: model makes a tool call
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            # Step 2 (final): model must produce text — verify no tools sent
            return mock_llm._make_text_response(request, "final answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "final answer"

        # Step 1 should have tools, step 2 should not
        assert mock_llm.requests[0]["tools"] is not None
        assert len(mock_llm.requests[0]["tools"]) > 0
        assert mock_llm.requests[1]["tools"] is None

        await agent.close()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_call_gets_nudged_and_recovers(self, test_db, mock_llm):
        """If model hallucinates tool calls on final step, it gets nudged and can recover."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                # Final step: model hallucinates a tool call despite no tools offered
                return mock_llm._make_tool_call_response(request, "search", {"query": "more"})
            # Nudge retry: model produces text
            return mock_llm._make_text_response(request, "here is the answer after nudge")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test query", max_steps=max_steps)
        assert response.answer == "here is the answer after nudge"

        # Hallucinated tool calls should have been stripped — retry called without tools
        assert mock_llm.requests[2]["tools"] is None

        # The nudge message should include the forceful prefix and original question
        nudge_messages = mock_llm.requests[2]["messages"]
        last_user_message = [m for m in nudge_messages if m["role"] == "user"][-1]
        assert "STOP" in last_user_message["content"]
        assert "Tools are no longer available" in last_user_message["content"]
        assert "test query" in last_user_message["content"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_call_fallback_when_nudge_fails(self, test_db, mock_llm):
        """If model keeps hallucinating after nudge, falls back gracefully."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            # All subsequent calls: model keeps hallucinating tool calls
            return mock_llm._make_tool_call_response(request, "search", {"query": "more"})

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        # Fallback when nudge cannot recover
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_call_with_text_uses_text(self, test_db, mock_llm):
        """If model returns both text and tool calls on final step, text is used."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            # Final step: model returns text AND a hallucinated tool call
            resp = mock_llm._make_tool_call_response(request, "search", {"query": "more"})
            resp.message.content = "here is the answer"
            return resp

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "here is the answer"

        await agent.close()


class TestRepeatCallGuard:
    """Test that repeat tool calls are blocked by args, not just name."""

    @pytest.mark.asyncio
    async def test_same_tool_different_args_allowed(self, test_db, mock_llm):
        """Calling the same tool with different arguments is allowed."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)
        # Mock tool executor so tool calls don't fail (this test checks dedup, not tools)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="search result"))

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "first topic"}
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "second topic"}
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"
        # Both searches should have executed
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].arguments["query"] == "first topic"
        assert response.tool_calls[1].arguments["query"] == "second topic"

        await agent.close()

    @pytest.mark.asyncio
    async def test_same_tool_same_args_blocked(self, test_db, mock_llm):
        """Calling the same tool with identical arguments is blocked."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count <= 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "same query"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"
        # Only first call should have executed
        assert len(response.tool_calls) == 1

        await agent.close()


class TestModelErrorHandling:
    """`_invoke_model` swallows LlmError → returns AGENT_MODEL_ERROR; other exceptions propagate."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_model_error(self, test_db, mock_llm):
        """Connection/response errors from the LLM result in AGENT_MODEL_ERROR, not a crash."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            raise LlmConnectionError("backend down")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR

        await agent.close()


class TestToolParseErrorRetry:
    """500 'error parsing tool call' recovers via a format nudge, not a fatal abort."""

    @pytest.mark.asyncio
    async def test_tool_parse_error_retries_with_format_nudge(self, test_db, mock_llm):
        """When the server returns a tool-parse 500, agent injects format nudge and retries."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                raise LlmToolParseError("error parsing tool call: raw='We need to produce...'")
            return mock_llm._make_tool_call_response(request, "search", {"query": "test"})

        mock_llm.set_response_handler(handler)

        await agent.run("test prompt", max_steps=max_steps)

        # Second call (after nudge) should include the format reminder
        assert len(mock_llm.requests) >= 2
        nudge_messages = mock_llm.requests[1]["messages"]
        last_user = next(m for m in reversed(nudge_messages) if m["role"] == "user")
        assert "tool call" in last_user["content"].lower()
        assert "plain text" in last_user["content"].lower()

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_parse_error_recovers_and_completes(self, test_db, mock_llm):
        """Cycle completes normally after a tool-parse error and retry."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                raise LlmToolParseError("error parsing tool call: raw='Let me reason first...'")
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "vitamins"})
            return mock_llm._make_text_response(request, "Here is the info about vitamins!")

        mock_llm.set_response_handler(handler)

        response = await agent.run("tell me about vitamins", max_steps=max_steps)
        assert response.answer == "Here is the info about vitamins!"
        assert len(mock_llm.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_parse_error_only_retried_once(self, test_db, mock_llm):
        """Tool-parse error retry only fires once — second parse error aborts the loop."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            raise LlmToolParseError("error parsing tool call: raw='plain text again...'")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        # First call + one retry = 2 total calls
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_timeout_error_returns_agent_model_error(self, test_db, mock_llm, caplog):
        """LLM timeouts also return AGENT_MODEL_ERROR and are logged at WARNING not ERROR."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            raise LlmTimeoutError("Request timed out.")

        mock_llm.set_response_handler(handler)

        with caplog.at_level(logging.WARNING, logger="penny.agents.base"):
            response = await agent.run("test prompt", max_steps=max_steps)

        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("timed out" in m.lower() for m in warning_msgs)
        assert not any("timed out" in m.lower() for m in error_msgs)

        await agent.close()

    @pytest.mark.asyncio
    async def test_non_llm_exception_propagates(self, test_db, mock_llm):
        """Programmer bugs in the LLM call path must surface, not be swallowed."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            raise RuntimeError("unexpected programmer bug")

        mock_llm.set_response_handler(handler)

        with pytest.raises(RuntimeError, match="unexpected programmer bug"):
            await agent.run("test prompt", max_steps=max_steps)

        await agent.close()


class TestEmptyContentFallback:
    """Test that an empty model response falls back to AGENT_EMPTY_RESPONSE."""

    @pytest.mark.asyncio
    async def test_empty_response_returns_agent_empty_response(self, test_db, mock_llm):
        """When the model returns empty content, AGENT_EMPTY_RESPONSE is returned."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_response_after_tool_call(self, test_db, mock_llm):
        """FALLBACK_RESPONSE is returned when model returns empty after preceding tool calls."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()


class TestThinkTagStripping:
    """Test that <think>...</think> blocks are stripped from final responses."""

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_content(self, test_db, mock_llm):
        """<think>...</think> blocks in content are removed before sending to user."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "<think>Internal reasoning here.</think>\nHere is the real answer."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("test", max_steps=max_steps)
        assert "<think>" not in response.answer
        assert "Internal reasoning here." not in response.answer
        assert response.answer == "Here is the real answer."

        await agent.close()

    @pytest.mark.asyncio
    async def test_think_tags_moved_to_thinking_field(self, test_db, mock_llm):
        """Content inside <think> blocks is captured in the thinking field."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "<think>Step-by-step plan.</think>\nFinal response."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("test", max_steps=max_steps)
        assert response.thinking == "Step-by-step plan."
        assert response.answer == "Final response."

        await agent.close()

    @pytest.mark.asyncio
    async def test_response_without_think_tags_unchanged(self, test_db, mock_llm):
        """Responses that contain no <think> tags are returned as-is."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        mock_llm.set_response_handler(
            lambda req, count: mock_llm._make_text_response(req, "Normal answer.")
        )

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "Normal answer."
        assert response.thinking is None

        await agent.close()


class TestAfterStepHook:
    """Test the after_step hook fires after tool calls."""

    @pytest.mark.asyncio
    async def testafter_step_called_with_step_records(self, test_db, mock_llm):
        """after_step receives only the records from the current step."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        # Mock tool executor so tool calls don't fail (this test checks after_step hook)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="search result"))

        captured_step_records = []

        async def captureafter_step(step_records, messages, conversation=None):
            captured_step_records.append(list(step_records))

        agent.after_step = captureafter_step

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "first", "reasoning": "step 1 reason"}
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "second", "reasoning": "step 2 reason"}
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"

        # Two steps with tool calls → two after_step calls
        assert len(captured_step_records) == 2
        assert len(captured_step_records[0]) == 1
        assert captured_step_records[0][0].reasoning == "step 1 reason"
        assert len(captured_step_records[1]) == 1
        assert captured_step_records[1][0].reasoning == "step 2 reason"

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_result_text_no_duplicates_across_steps(self, test_db, mock_llm):
        """Each step's tool result should appear exactly once in _tool_result_text."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)
        agent._tool_executor.execute = AsyncMock(
            side_effect=[
                ToolResult(message="result_A"),
                ToolResult(message="result_B"),
                ToolResult(message="result_C"),
            ]
        )

        def handler(request, count):
            if count <= 3:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"query_{count}"}
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        await agent.run("test", max_steps=max_steps)

        # 3 tool calls → exactly 3 entries, no duplicates from re-scanning history.
        # Each is wrapped by Tool.format_result so the model reads it as the
        # response to its own call (the body is unchanged).
        assert len(agent._tool_result_text) == 3
        assert agent._tool_result_text == [
            "Result of your `search` call:\nresult_A",
            "Result of your `search` call:\nresult_B",
            "Result of your `search` call:\nresult_C",
        ]

        await agent.close()


class TestEmptyContentRetry:
    """Test that empty content responses trigger a retry with a follow-up prompt."""

    @pytest.mark.asyncio
    async def test_empty_content_on_nonfinal_step_retries_with_followup(self, test_db, mock_llm):
        """When model returns empty (or garbage) content mid-loop, agent retries with follow-up.

        Garbage shapes — bare separators, lone punctuation, single emoji — must be
        treated the same as a literally empty string. Otherwise a model that emits
        `\n\n---` after a tool call will silently overwrite a real prior answer.
        """
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                # Garbage response: just a markdown separator. No "real words".
                return mock_llm._make_text_response(request, "\n\n---")
            # After follow-up injection, model returns actual text
            return mock_llm._make_text_response(request, "here's the answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "here's the answer"
        # Three model calls: tool call, empty response, final answer
        assert len(mock_llm.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_content_on_final_step_retries_and_succeeds(self, test_db, mock_llm):
        """When model returns empty content on the final step, agent retries once and succeeds."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        def handler(request, count):
            if count == 1:
                # Final step returns empty content
                return mock_llm._make_text_response(request, "")
            # Retry (extra step) returns real content
            return mock_llm._make_text_response(request, "here's the answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "here's the answer"
        # Two model calls: empty final step + retry
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_content_twice_returns_fallback(self, test_db, mock_llm):
        """When model returns empty content on both the final step and retry, returns fallback."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        def handler(request, count):
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE
        # Two model calls: empty final step + one retry that also returns empty
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_content_after_tool_calls_returns_fallback_response(
        self, test_db, mock_llm
    ):
        """After tool calls, double empty returns FALLBACK_RESPONSE not AGENT_EMPTY_RESPONSE.

        Reproduces the production scenario where the model makes several searches then
        fails to synthesize on the final step (preceding_tool_calls > 0).  The distinction
        matters: FALLBACK_RESPONSE signals "searched but couldn't synthesise" while
        AGENT_EMPTY_RESPONSE signals "never tried to answer".
        """
        # 4 steps: 3 tool calls + 1 final step where model must synthesise but fails
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)

        def handler(request, count):
            if count <= 3:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"query {count}"}
                )
            # Final step and its retry both return empty — model can't synthesise
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == PennyResponse.FALLBACK_RESPONSE
        # 3 tool calls + empty final step + empty retry = 5 model calls
        assert len(mock_llm.requests) == 5

        await agent.close()


class TestToolCallCap:
    """Test that the tool-call cap forces an early final step before context saturation."""

    @pytest.mark.asyncio
    async def test_batched_tool_calls_cap_forces_early_final_step(self, test_db, mock_llm):
        """When batched tool calls accumulate to steps-1, the final step is forced early.

        Regression guard for the observed bug: preceding_tool_calls=11 with MAX_STEPS=8.
        Each agentic loop step can produce multiple tool call records (parallel calls),
        so the step count alone does not bound the total tool call context. The cap
        ensures the model gets a final step before accumulating more than steps-1 records.
        """
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        def handler(request, count):
            # Steps 1 and 2: 2 parallel tool calls each → 4 total records.
            # Cap = max_steps - 1 = 3, so after 4 records the next step is final.
            if count in (1, 2):
                return mock_llm._make_parallel_tool_calls_response(
                    request,
                    [
                        ("search", {"query": f"query {count}a"}),
                        ("search", {"query": f"query {count}b"}),
                    ],
                )
            # Step 3 is forced final (tools stripped) — model produces answer.
            return mock_llm._make_text_response(request, "here is the answer")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "here is the answer"
        assert len(response.tool_calls) == 4

        # Third model call must have no tools (early forced final step from cap).
        assert mock_llm.requests[2]["tools"] is None
        # Only 3 model calls — cap fired one step before max_steps would have.
        assert len(mock_llm.requests) == 3

        await agent.close()


class TestParallelToolCalls:
    """Test that multiple tool calls in a single turn are dispatched in parallel."""

    @pytest.mark.asyncio
    async def test_two_tool_calls_produce_separate_tool_messages(self, test_db, mock_llm):
        """Two tool calls returned in one response each get their own tool message."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(
            side_effect=lambda tool_call: ToolResult(
                message=f"result for {tool_call.arguments.get('query', '')}"
            )
        )

        def handler(request, count):
            if count == 1:
                return mock_llm._make_parallel_tool_calls_response(
                    request,
                    [("search", {"query": "topic A"}), ("search", {"query": "topic B"})],
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test", max_steps=max_steps)

        assert response.answer == "done"
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].arguments["query"] == "topic A"
        assert response.tool_calls[1].arguments["query"] == "topic B"

        # The second Ollama call should include two separate role=tool messages, not one merged blob
        second_call_messages = mock_llm.requests[1]["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        assert "topic A" in tool_messages[0]["content"]
        assert "topic B" in tool_messages[1]["content"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_large_browse_tool_results_not_truncated(self, test_db, mock_llm):
        """Two large tool results from BrowseTool both survive into the model context."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        page_a = "A" * 15000  # 15k chars — realistic extracted web page
        page_b = "B" * 15000

        sep = PennyConstants.SECTION_SEPARATOR
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message=f"## page A\n{page_a}{sep}## page B\n{page_b}")
        )

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "browse", {"queries": ["https://a.com", "https://b.com"]}
                )
            # Verify both pages present in the tool message
            messages = request["messages"]
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            content = tool_messages[0]["content"]
            assert "A" * 1000 in content, "Page A content was truncated"
            assert "B" * 1000 in content, "Page B content was truncated"
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"
        await agent.close()

    @pytest.mark.asyncio
    async def test_text_queries_route_to_search_url_when_browser_connected(self, test_db, mock_llm):
        """When a browser is connected, text queries become search URLs via BrowseTool."""
        browsed_urls: dict[str, str] = {}

        async def fake_request(command, params):
            url = params["url"]
            browsed_urls[url] = f"Results for {url}"
            return (browsed_urls[url], None)

        request_fn = AsyncMock(side_effect=fake_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(max_calls=5)
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(queries=["best pizza toronto"])

        assert len(browsed_urls) == 1
        search_url = list(browsed_urls.keys())[0]
        assert search_url.startswith("https://duckduckgo.com/?q=")
        assert "best%20pizza%20toronto" in search_url

    @pytest.mark.asyncio
    async def test_text_queries_fail_without_browser(self, test_db, mock_llm, monkeypatch):
        """Without a browser, queries surface a structured browse error section."""
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        tool = BrowseTool(max_calls=5)

        result = await tool.execute(queries=["best pizza toronto"])

        assert PennyConstants.BROWSE_ERROR_HEADER in result.message
        assert "no browser is connected" in result.message
        assert PennyConstants.BROWSE_PAGE_HEADER not in result.message

    @pytest.mark.asyncio
    async def test_urls_always_route_to_browse(self, test_db, mock_llm):
        """URLs always go to BrowseTool regardless of browser connection."""
        browsed_urls: list[str] = []

        async def fake_request(command, params):
            browsed_urls.append(params["url"])
            return (f"Page content from {params['url']}", None)

        request_fn = AsyncMock(side_effect=fake_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(max_calls=5)
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(queries=["https://example.com/page", "https://other.com"])

        assert len(browsed_urls) == 2
        assert "https://example.com/page" in browsed_urls
        assert "https://other.com" in browsed_urls

    @pytest.mark.asyncio
    async def test_url_timeout_returns_error_section(self, monkeypatch):
        """When request_fn raises TimeoutError, execute() returns an error section.

        This is the regression test for the 'Tool execution timeout: browse' bug:
        BrowseTool.timeout must exceed BROWSE_REQUEST_TIMEOUT so the inner per-URL
        timeout fires first and is captured by asyncio.gather(return_exceptions=True),
        allowing execute() to return a graceful error rather than the whole tool
        timing out at the executor level.
        """
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)

        async def timed_out_request(command, params):
            raise TimeoutError("Browser tool 'browse_url' timed out after 60.0s")

        request_fn = AsyncMock(side_effect=timed_out_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(max_calls=5)
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        result = await tool.execute(queries=["https://slow.example.com"])

        assert isinstance(result, ToolResult)
        assert PennyConstants.BROWSE_ERROR_HEADER in result.message
        assert "slow.example.com" in result.message

    def test_browse_tool_timeout_exceeds_request_timeout(self):
        """BrowseTool.timeout must exceed the per-URL BROWSE_REQUEST_TIMEOUT.

        Ensures the inner per-URL timeout fires before the outer executor
        timeout, so hung URLs produce graceful error sections instead of
        cancelling the entire tool call.
        """
        tool = BrowseTool(max_calls=3)
        assert tool.timeout is not None
        assert tool.timeout > PennyConstants.BROWSE_REQUEST_TIMEOUT


class TestSearchResultTrimming:
    """Tests for _trim_search_result: strips search pages to links + context."""

    def test_trims_to_lines_near_links(self):
        """Lines far from markdown links are removed."""
        content = "\n".join(
            [
                "Lots of preamble text here",
                "More preamble",
                "Even more preamble",
                "Still going",
                "Yet more preamble",
                "### NASA Article",
                "[nasa.gov/artemis](https://www.nasa.gov/artemis/)",
                "Some snippet text",
                "More snippet",
                "Filler line 1",
                "Filler line 2",
                "Filler line 3",
                "Filler line 4",
                "Filler line 5",
                "### Space.com Article",
                "[space.com/artemis](https://www.space.com/artemis)",
                "Another snippet",
            ]
        )
        result = _trim_search_result(content)
        assert "titles and links only" in result
        assert "nasa.gov/artemis" in result
        assert "space.com/artemis" in result
        assert "### NASA Article" in result
        assert "Lots of preamble" not in result
        assert "Filler line 3" not in result

    def test_returns_original_when_no_links(self):
        """Content with no markdown links passes through unchanged."""
        content = "Just plain text\nwith no links\nat all"
        result = _trim_search_result(content)
        assert result == content

    def test_strips_knowledge_panel_prose_with_inline_links(self):
        """Wikipedia-style prose with inline links is excluded."""
        content = "\n".join(
            [
                "A [fantasy](https://x.org/a) [drama](https://x.org/b).",
                "",
                "Ira [Martin](https://x.org/m)",
                "",
                "[Alice](https://x.org/a1) [Bob](https://x.org/b1)",
                "",
                "Genre",
                "Created by",
                "",
                "### Show Title",
                "",
                "[en.example.org/show](https://en.example.org/show)",
                "",
                "An American fantasy drama television series.",
                "### Show - IMDb",
                "",
                "[imdb.com/title/tt1](https://imdb.com/title/tt1)",
                "",
                "A delightful return to the world.",
            ]
        )
        result = _trim_search_result(content)
        # Real search result links and their context are kept
        assert "en.example.org/show" in result
        assert "imdb.com/title/tt1" in result
        assert "### Show Title" in result
        # Knowledge panel prose with inline links is stripped
        assert "[fantasy]" not in result
        # Multi-link metadata lines are stripped
        assert "[Alice]" not in result
        assert "Ira [Martin]" not in result

    def test_caps_at_max_search_links(self):
        """Only the first PennyConstants.MAX_SEARCH_LINKS standalone links are kept."""
        lines: list[str] = []
        for i in range(PennyConstants.MAX_SEARCH_LINKS + 5):
            lines.append(f"### Result {i}")
            lines.append(f"[example.com/page{i}](https://example.com/page{i})")
            lines.append(f"Snippet for result {i}")
        content = "\n".join(lines)
        result = _trim_search_result(content)
        # First 10 kept
        assert f"example.com/page{PennyConstants.MAX_SEARCH_LINKS - 1}" in result
        # 11th and beyond dropped
        assert f"example.com/page{PennyConstants.MAX_SEARCH_LINKS}" not in result
        assert f"example.com/page{PennyConstants.MAX_SEARCH_LINKS + 4}" not in result

    def test_header_injected(self):
        """Trimmed results start with the search result header."""
        content = "### Title\n[example](https://example.com)\nSnippet"
        result = _trim_search_result(content)
        assert result.startswith("These are search results")


class TestEmptyContentAfterToolCalls:
    """Tests for combined empty-content fixes: nudge prompts, think tag stripping,
    retry counter reset, and fallback response."""

    @pytest.mark.asyncio
    async def test_final_step_empty_content_gets_strong_nudge(self, test_db, mock_llm):
        """When model returns empty on final step, retry uses strong nudge."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_llm._make_text_response(request, "")
            return mock_llm._make_text_response(request, "Here's what I found!")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "Here's what I found!"

        retry_messages = mock_llm.requests[2]["messages"]
        last_user = next(m for m in reversed(retry_messages) if m["role"] == "user")
        assert "STOP" in last_user["content"]
        assert "test question" in last_user["content"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_mid_loop_empty_content_gets_continue_nudge(self, test_db, mock_llm):
        """When model returns empty mid-loop (tools still available), uses continue nudge."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                # Mid-loop: model returns empty (tools still available, not final step)
                return mock_llm._make_text_response(request, "")
            if count == 2:
                # Retry after continue nudge — model responds
                return mock_llm._make_text_response(request, "here's my answer")
            return mock_llm._make_text_response(request, "fallback")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "here's my answer"

        retry_messages = mock_llm.requests[1]["messages"]
        last_user = next(m for m in reversed(retry_messages) if m["role"] == "user")
        assert last_user["content"] == "Please provide your response."
        # Should NOT have the strong nudge — model still has tools available
        assert "STOP" not in last_user["content"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_think_only_response_triggers_retry(self, test_db, mock_llm):
        """Model returning only <think> tags with no body triggers retry."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_llm._make_text_response(
                    request, "<think>Let me reason about this...</think>"
                )
            return mock_llm._make_text_response(request, "here's the answer")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "here's the answer"
        assert len(mock_llm.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_counter_resets_after_tool_calls(self, test_db, mock_llm):
        """After nudge fires, tools are stripped so model must synthesize."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)
        # Mock tool executor so tool calls don't fail
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="search result"))

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "first"})
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "second"})
            if count == 3:
                # Empty content triggers nudge — tools stripped on next call
                return mock_llm._make_text_response(request, "")
            # count 4: tools stripped, model must produce text
            return mock_llm._make_text_response(request, "synthesized answer")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True
        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "synthesized answer"

        await agent.close()

    @pytest.mark.asyncio
    async def test_fallback_response_after_tool_calls(self, test_db, mock_llm):
        """FALLBACK_RESPONSE (not AGENT_EMPTY_RESPONSE) when empty after tool calls."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_large_tool_results_pass_through_untruncated(self, test_db, mock_llm):
        """Large tool results are not truncated — client enforces per-page limits."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)
        large_result = "x" * 100_000

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            messages = request["messages"]
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            content = tool_messages[0]["content"]
            # Body passes through whole (the Tool.format_result frame adds a
            # short header prefix, but the 100k payload is untouched).
            assert large_result in content
            assert "[truncated]" not in content
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        with patch.object(agent._tool_executor, "execute") as mock_exec:
            mock_exec.return_value = ToolResult(message=large_result)
            response = await agent.run("test", max_steps=max_steps)

        assert response.answer == "done"
        await agent.close()


class TestStrongNudgeUsesLastQuestion:
    """Test that the strong nudge references the current question, not prior history."""

    @pytest.mark.asyncio
    async def test_nudge_references_current_question_not_history(
        self,
        test_db,
        mock_llm,
    ):
        """When the agentic loop exhausts tool calls and fires a strong nudge,
        the nudge must reference the latest user question — not an earlier one
        from conversation history.

        Regression: _build_strong_nudge used next() (first user message) instead
        of the last, so with conversation history it would reference a prior question.
        """
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)

        history = [
            ("user", "what are some good 40k novels?"),
            ("assistant", "Here are some novels..."),
            ("user", "who were the dark mechanicum leaders?"),
            ("assistant", "Here are the leaders..."),
            ("user", "who starred in judge dredd 1995?"),
            ("assistant", "Here is the cast..."),
        ]

        current_question = "what else was joan chen in?"
        nudge_content = None

        def handler(request, count):
            nonlocal nudge_content
            messages = request["messages"]
            user_msgs = [m for m in messages if m.get("role") == "user"]
            last_user = user_msgs[-1]["content"] if user_msgs else ""

            if "STOP" in last_user:
                nudge_content = last_user
                return mock_llm._make_text_response(request, "Joan Chen was in Twin Peaks")

            # After 4 tool calls, return empty to trigger strong nudge
            if count >= 5:
                return mock_llm._make_text_response(request, "")

            return mock_llm._make_tool_call_response(
                request, "search", {"query": f"joan chen filmography {count}"}
            )

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True
        response = await agent.run(
            current_question,
            max_steps=max_steps,
            history=history,
        )

        assert nudge_content is not None, "Strong nudge should have fired"
        assert "joan chen" in nudge_content.lower(), (
            f"Nudge should reference 'joan chen' but got: {nudge_content}"
        )
        assert "dark mechanicum" not in nudge_content.lower(), (
            f"Nudge should NOT reference prior question but got: {nudge_content}"
        )
        assert response.answer == "Joan Chen was in Twin Peaks"

        await agent.close()


class TestRefusalRetry:
    """Test that model refusals trigger a retry nudge."""

    @pytest.mark.asyncio
    async def test_refusal_on_nonfinal_step_retries_with_nudge(self, test_db, mock_llm):
        """When model refuses on a non-final step, agent injects nudge and continues."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_llm._make_text_response(
                    request, "I'm sorry, but I can't help with that."
                )
            return mock_llm._make_text_response(request, "Here are the vegan smoothie recipes!")

        mock_llm.set_response_handler(handler)

        response = await agent.run("Give me a list of vegan smoothie recipes", max_steps=max_steps)
        assert response.answer == "Here are the vegan smoothie recipes!"
        assert len(mock_llm.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_refusal_on_final_step_retries_inline(self, test_db, mock_llm):
        """When model refuses on the final step, agent retries once inline."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, "I cannot help with that request.")
            return mock_llm._make_text_response(request, "Here is a helpful answer!")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        assert response.answer == "Here is a helpful answer!"
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_refusal_only_retried_once(self, test_db, mock_llm):
        """Refusal retry only fires once — second refusal is returned as-is."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            return mock_llm._make_text_response(request, "I'm sorry, I am unable to help.")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test question", max_steps=max_steps)
        # Should contain the refusal text (returned as-is after one retry)
        assert "sorry" in response.answer.lower() or "unable" in response.answer.lower()
        # Only two model calls: initial refusal + one retry
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_normal_response_not_retried(self, test_db, mock_llm):
        """Normal responses are not mistakenly flagged as refusals."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        mock_llm.set_response_handler(
            lambda req, count: mock_llm._make_text_response(req, "Here are your recipes!")
        )

        response = await agent.run("Give me vegan smoothie recipes", max_steps=max_steps)
        assert response.answer == "Here are your recipes!"
        assert len(mock_llm.requests) == 1

        await agent.close()


class TestUrlValidationSourceContext:
    """URL validation must accept URLs from system prompt and history, not only tool results.

    Each test runs a tool call first so `_tool_result_text` is populated and validation
    actually fires — the production bug only manifests after a real browse turn.
    """

    @pytest.mark.asyncio
    async def test_url_from_system_prompt_not_flagged(self, test_db, mock_llm):
        """A URL provided in the system prompt (e.g. knowledge section) is not hallucinated."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        knowledge_url = (
            "https://www.henryford.com/Blog/2022/11/Why-Some-People-Get-Colds-and-the-Flu"
        )
        system_prompt = (
            "You are Penny.\n\n### Related Knowledge\n"
            f"Why Some People Get Colds More Than Others\n{knowledge_url}\n"
            "Cold seasons trigger viral infections..."
        )
        answer = f"Here's what the research says: see {knowledge_url} for the full study."

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "colds"})
            return mock_llm._make_text_response(request, answer)

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "tell me about colds", max_steps=max_steps, system_prompt=system_prompt
        )

        assert response.answer == answer
        # Two model calls: tool call + text response. No retry.
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_url_from_history_not_flagged(self, test_db, mock_llm):
        """A URL the assistant cited earlier in conversation history is not hallucinated."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        prior_url = "https://pubmed.ncbi.nlm.nih.gov/26118561/"
        history = [
            ("user", "what's the data say"),
            ("assistant", f"I dug into a study at {prior_url} that covers exactly that."),
        ]
        answer = f"Following up — the same paper {prior_url} also notes immune signalling."

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "follow"})
            return mock_llm._make_text_response(request, answer)

        mock_llm.set_response_handler(handler)

        response = await agent.run("follow up", max_steps=max_steps, history=history)

        assert response.answer == answer
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_url_not_in_any_context_still_flagged(self, test_db, mock_llm):
        """URL with no source anywhere in messages still triggers a retry."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        bad = "Made-up source: https://totally-fake.example/never-seen"
        good = "Here's a clean answer with no URL."

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "x"})
            return mock_llm._make_text_response(request, bad if count == 2 else good)

        mock_llm.set_response_handler(handler)

        response = await agent.run("question", max_steps=max_steps)

        assert response.answer == good
        # Tool call + bad text + retry text = 3 model calls
        assert len(mock_llm.requests) == 3

        await agent.close()


class TestMalformedUrlCleaning:
    """Test that truncated or malformed URLs are stripped from final responses."""

    @pytest.mark.asyncio
    async def test_bare_truncated_url_removed(self, test_db, mock_llm):
        """Bare URL ending with a hyphen (truncated path) is removed from the response."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "Check this out: https://travelguide.com/destination- for details."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("tell me about travel", max_steps=max_steps)
        assert "https://travelguide.com/destination-" not in response.answer
        assert "Check this out:" in response.answer

        await agent.close()

    @pytest.mark.asyncio
    async def test_markdown_link_truncated_url_keeps_text(self, test_db, mock_llm):
        """Markdown link [text](bad_url) strips the URL but preserves the link text."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "Visit [Travel Guide](https://travelguide.com/destination-) for more info."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("travel info", max_steps=max_steps)
        assert "https://travelguide.com/destination-" not in response.answer
        assert "Travel Guide" in response.answer

        await agent.close()

    @pytest.mark.asyncio
    async def test_valid_url_unchanged(self, test_db, mock_llm):
        """A well-formed URL is not touched."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "See https://example.com/article for more."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("article link", max_steps=max_steps)
        assert "https://example.com/article" in response.answer

        await agent.close()

    @pytest.mark.asyncio
    async def test_source_url_appended_after_malformed_url_stripped(self, test_db, mock_llm):
        """When a malformed URL is stripped, source URL fallback appends a real URL."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(
                request, "Found something at https://bad.example/path-"
            )

        mock_llm.set_response_handler(handler)

        source_url = "https://real-source.com/article"
        with patch.object(agent._tool_executor, "execute") as mock_exec:
            mock_exec.return_value = ToolResult(message="result", source_urls=[source_url])
            response = await agent.run("test query", max_steps=max_steps)

        assert "https://bad.example/path-" not in response.answer
        assert source_url in response.answer

        await agent.close()


class TestAllToolsFailedAbort:
    """Test that the agentic loop aborts when all tool calls fail."""

    @pytest.mark.asyncio
    async def test_aborts_when_all_tool_calls_fail(self, test_db, mock_llm):
        """Loop aborts with AGENT_TOOLS_UNAVAILABLE when all tools return errors."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)
        # Mock tool executor to always return an error
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message="API unavailable", success=False)
        )

        def handler(request, count):
            # Model keeps trying tool calls — all fail
            return mock_llm._make_tool_call_response(
                request, "search", {"query": f"attempt {count}"}
            )

        mock_llm.set_response_handler(handler)
        response = await agent.run("what's the news?", max_steps=max_steps)
        assert response.answer.startswith("Sorry, I wasn't able to get results right now")
        assert "search" in response.answer

        await agent.close()

    @pytest.mark.asyncio
    async def test_no_abort_when_some_tools_succeed(self, test_db, mock_llm):
        """Loop continues when at least one tool call succeeds."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)

        call_count = 0

        async def alternating_executor(tool_call):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult(message="API unavailable", success=False)
            return ToolResult(message="found some results")

        agent._tool_executor.execute = alternating_executor

        def handler(request, count):
            if count <= 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": f"q{count}"})
            return mock_llm._make_text_response(request, "here are results")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "here are results"

        await agent.close()


class TestOnToolStartCallback:
    """Test that the on_tool_start callback fires before tool execution with all pending tools."""

    @pytest.mark.asyncio
    async def test_callback_called_once_per_step_with_all_tools(self, test_db, mock_llm):
        """on_tool_start fires once per step with a list of all tools in that step."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        captured: list[list[tuple[str, dict]]] = []

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            captured.append(tools)

        def handler(request, count):
            if count <= 2:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"query {count}"}
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test", max_steps=max_steps, on_tool_start=on_tool_start)
        assert response.answer == "done"
        # Two sequential single-tool steps → callback fires twice, each with one tool
        assert len(captured) == 2
        assert captured[0] == [("search", {"query": "query 1"})]
        assert captured[1] == [("search", {"query": "query 2"})]

        await agent.close()

    @pytest.mark.asyncio
    async def test_parallel_tools_fire_callback_once_with_both(self, test_db, mock_llm):
        """on_tool_start fires once for a parallel step, receiving both tools together."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        captured: list[list[tuple[str, dict]]] = []

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            captured.append(tools)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_parallel_tool_calls_response(
                    request,
                    [("search", {"query": "topic A"}), ("search", {"query": "topic B"})],
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test", max_steps=max_steps, on_tool_start=on_tool_start)
        assert response.answer == "done"
        # One step with two parallel tools → callback fires once with both
        assert len(captured) == 1
        assert captured[0] == [("search", {"query": "topic A"}), ("search", {"query": "topic B"})]

        await agent.close()

    @pytest.mark.asyncio
    async def test_callback_not_called_for_deduped_repeat(self, test_db, mock_llm):
        """on_tool_start does not fire when all tools in a step are deduplicated."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        captured: list[list[tuple[str, dict]]] = []

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            captured.append(tools)

        def handler(request, count):
            if count <= 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "same query"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        await agent.run("test", max_steps=max_steps, on_tool_start=on_tool_start)
        # Only the first step fires; the second is fully deduplicated so pending is empty
        assert len(captured) == 1

        await agent.close()

    @pytest.mark.asyncio
    async def test_failing_callback_does_not_abort_tool(self, test_db, mock_llm):
        """A callback that raises an exception does not prevent tool execution."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        async def on_tool_start(tools: list[tuple[str, dict]]) -> None:
            raise RuntimeError("callback exploded")

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps, on_tool_start=on_tool_start)
        assert response.answer == "done"
        assert len(response.tool_calls) == 1

        await agent.close()


class TestPromptLogAnnotations:
    """Test that prompt logs are annotated with agent_name and run_id."""

    @pytest.mark.asyncio
    async def test_agent_name_and_run_id_written_to_promptlog(self, test_db, mock_llm):
        """Every prompt in an agentic loop gets the agent's name and a shared run_id."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        # Track callback invocations
        callback_prompts: list[dict] = []
        db.messages._on_prompt_logged = lambda data: callback_prompts.append(data)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        await agent.run("test question", max_steps=max_steps)

        with Session(db.engine) as session:
            logs = session.exec(select(PromptLog)).all()

        assert len(logs) == 2
        # All logs share the same run_id
        run_ids = {log.run_id for log in logs}
        assert len(run_ids) == 1
        run_id = run_ids.pop()
        assert run_id is not None
        assert len(run_id) == 32  # uuid4 hex

        # All logs have the agent name
        assert all(log.agent_name == "Agent" for log in logs)

        # Callback fired for each prompt with run_id
        assert len(callback_prompts) == 2
        assert all(p["run_id"] == run_id for p in callback_prompts)
        assert all(p["agent_name"] == "Agent" for p in callback_prompts)
        assert "input_tokens" in callback_prompts[0]

        await agent.close()

    @pytest.mark.asyncio
    async def test_separate_runs_get_different_run_ids(self, test_db, mock_llm):
        """Two separate run() calls produce different run_ids."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, "done"))

        await agent.run("first", max_steps=max_steps)
        await agent.run("second", max_steps=max_steps)

        with Session(db.engine) as session:
            logs = session.exec(select(PromptLog)).all()

        assert len(logs) == 2
        assert logs[0].run_id != logs[1].run_id
        assert logs[0].agent_name == logs[1].agent_name == "Agent"

        await agent.close()


def _make_background_agent(test_db, *, max_steps=4):
    """A minimal BackgroundAgent (collector shape) for text-nudge testing.

    Built with both a work tool (search) and the ``done`` terminator in the
    registry so the model can either continue working or exit — exactly the two
    legal moves a collector has after a stray text response.  Keeps the default
    ``_keep_tools_on_final_step=True`` so tools stay available to exit with.
    """
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
        runtime=RuntimeParams(db=db, env_overrides={}),
    )
    client = LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.1,
    )
    agent = BackgroundAgent(
        system_prompt="test",
        model_client=client,
        tools=[StubSearchTool(), DoneTool()],
        db=db,
        config=config,
    )
    return agent, db, max_steps


class TestCollectorTextNudge:
    """A collector that emits plain text instead of a tool call gets nudged to
    re-emit as a tool call, rather than the loop treating the text as a final
    answer and ending the cycle without a ``done`` record."""

    @pytest.mark.asyncio
    async def test_text_bail_is_nudged_and_recovers_with_done(self, test_db, mock_llm):
        """Work, then prose ("Done. Summary: ...") → nudge → re-emits a real done().

        The realistic production shape (and what the eval's ``_InjectTextBail`` forces):
        the model does real work, THEN narrates completion as prose instead of
        calling done().  Work-first matters — a done() after the nudge is only
        honoured because a real tool call already ran; a bare done() with no work
        would itself be refused by the premature-done guard."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, "**Done. Summary: wrote the entry.**")
            return mock_llm._make_tool_call_response(
                request, "done", {"success": True, "summary": "wrote the entry"}
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # The model was called again after the stray text (nudged, not stopped).
        assert len(mock_llm.requests) == 3
        # The nudge was injected as the last user turn before the retry.
        last_user = [m for m in mock_llm.requests[2]["messages"] if m["role"] == "user"][-1]
        assert "tool call" in last_user["content"].lower()
        # The cycle closed with a real done() record (not a lost/failed cycle).
        assert any(record.tool == "done" for record in response.tool_calls)

        await agent.close()

    @pytest.mark.asyncio
    async def test_text_bail_can_recover_into_more_work(self, test_db, mock_llm):
        """Mid-work narration → nudge → the model continues with a work tool, not
        a premature done()."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(
                    request, "**Observation** the page has an entry."
                )
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "more"})
            return mock_llm._make_tool_call_response(
                request, "done", {"success": True, "summary": "done after more work"}
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # Nudged into continuing: it ran the work tool, then closed with done().
        tools_called = [record.tool for record in response.tool_calls]
        assert "search" in tools_called
        assert "done" in tools_called

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_text_bail_is_bounded_by_max_steps(self, test_db, mock_llm):
        """A model that keeps emitting text never loops forever — the nudge is
        bounded by max_steps and the cycle ends without a done record."""
        agent, db, max_steps = _make_background_agent(test_db, max_steps=3)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "still just talking")
        )

        response = await agent.run("", max_steps=max_steps)

        # Exactly max_steps calls — nudged each non-final step, then stopped.
        assert len(mock_llm.requests) == 3
        assert not any(record.tool == "done" for record in response.tool_calls)

        await agent.close()


class TestCollectorPrematureDone:
    """A collector whose very first tool call is done() — with no prior read /
    write / browse — is refused with an ERROR TOOL RESPONSE (not a text-step
    nudge: the model made a coherent tool call, so the correction goes back as
    that call's failed result).  A failed done() doesn't stop the loop, so the
    model sees the error and recovers with a real tool call first."""

    @pytest.mark.asyncio
    async def test_first_move_done_is_rejected_and_recovers(self, test_db, mock_llm):
        """done() as the opening move → error tool response → model reads, then
        legitimately closes with done()."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                # The production flavor: "no new matches" without reading anything.
                return mock_llm._make_tool_call_response(
                    request, "done", {"success": True, "summary": "no new matches this cycle"}
                )
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            return mock_llm._make_tool_call_response(
                request, "done", {"success": True, "summary": "wrote the entry"}
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # The premature done() did NOT stop the loop — the model was called again.
        assert len(mock_llm.requests) == 3
        # The rejection came back as a TOOL result (not a user-turn nudge).
        retry_messages = mock_llm.requests[1]["messages"]
        tool_results = [m for m in retry_messages if m["role"] == "tool"]
        assert tool_results, "premature done() should be refused via a tool result"
        assert "before doing anything" in tool_results[-1]["content"].lower()
        assert not any(
            m["role"] == "user" and "before doing" in m["content"] for m in retry_messages
        )
        # It recovered: a real work tool ran, then the cycle closed with done().
        tools_called = [record.tool for record in response.tool_calls]
        assert "search" in tools_called
        assert "done" in tools_called

        await agent.close()

    @pytest.mark.asyncio
    async def test_done_after_real_work_is_not_rejected(self, test_db, mock_llm):
        """A done() that follows a real tool call is legitimate — the guard only
        fires on a FIRST-move done(), so a normal read-then-done cycle is untouched."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            return mock_llm._make_tool_call_response(
                request, "done", {"success": True, "summary": "wrote the entry"}
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # Two calls only — the done() on step 2 closed cleanly, never refused.
        assert len(mock_llm.requests) == 2
        done_records = [r for r in response.tool_calls if r.tool == "done"]
        assert len(done_records) == 1
        assert done_records[0].failed is False

        await agent.close()

    @pytest.mark.asyncio
    async def test_batched_read_and_done_is_not_premature(self, test_db, mock_llm):
        """A single response that batches [search, done] is not a bail — a real
        tool call rode along, so the done() is honoured and closes the cycle."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            return mock_llm._make_parallel_tool_calls_response(
                request,
                [
                    ("search", {"query": "inputs"}),
                    ("done", {"success": True, "summary": "wrote the entry"}),
                ],
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # One model call: the batched done() stopped the loop, never refused.
        assert len(mock_llm.requests) == 1
        tools_called = [record.tool for record in response.tool_calls]
        assert "search" in tools_called and "done" in tools_called

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_first_move_done_is_bounded_by_max_steps(self, test_db, mock_llm):
        """A model that keeps opening with done() never loops forever — refused on
        each non-final step, bounded by max_steps.  On the final step there's no
        room to retry (like the text-bail fallback), so that done is honoured and
        is the only recorded one — the refused earlier ones leave no record."""
        agent, db, max_steps = _make_background_agent(test_db, max_steps=3)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_tool_call_response(
                request, "done", {"success": True, "summary": "no new matches this cycle"}
            )
        )

        response = await agent.run("", max_steps=max_steps)

        # Bounded at max_steps (didn't loop forever); only the final-step done()
        # was accepted — the two refused first-move dones left no record.
        assert len(mock_llm.requests) == 3
        assert len([r for r in response.tool_calls if r.tool == "done"]) == 1

        await agent.close()
