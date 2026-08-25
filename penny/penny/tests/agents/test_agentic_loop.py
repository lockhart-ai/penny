"""Tests for agentic loop changes: reasoning, last step, and after_step hook."""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from penny.agents.base import Agent, BackgroundAgent, _any_text
from penny.agents.chat import ChatAgent
from penny.agents.models import (
    REROLL_EXHAUSTED,
    MessageRole,
    ModelCallError,
    RunAbort,
    ToolCallRecord,
)
from penny.config import Config
from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm import LlmClient
from penny.llm.models import (
    LlmConnectionError,
    LlmMessage,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
)
from penny.prompts import Prompt
from penny.responses import PennyResponse

# The eval harness's loop-health probe, asserted against a REAL re-roll here: its whole premise
# is that a discarded draw leaves two promptlog rows on the same context, and a probe pinned
# only against hand-seeded rows would keep passing if the loop stopped leaving them (#1841).
from penny.tests.eval.conftest import draw_rerolled
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.text_validity import (
    half_formed_send_reason,
    has_leaked_harmony_envelope,
    is_call_as_text_bail,
    is_call_fragment_reply,
    is_degenerate_run,
    is_done_json_bail,
    is_empty_draw,
)
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool, _trim_search_result
from penny.tools.models import BrowseArgs, ToolArgs, ToolResult
from penny.validation import (
    ConditionKey,
    LoopContext,
    Proceed,
    Repair,
    Retry,
    run_validators,
)
from penny.validation.response_validators import (
    AppliedConfigurationValidator,
    HallucinatedToolCallRepair,
    HallucinatedUrlValidator,
    RefusalValidator,
    SkillNarrationValidator,
    WritesLandedValidator,
    XmlTagValidator,
)


class _StubSearchArgs(ToolArgs):
    """Args for the search stub — one required query, extras forbidden like a real tool."""

    query: str


class StubSearchTool(Tool):
    """Minimal stub tool for agentic loop testing."""

    name = "search"
    description = "Search for information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }
    args_model = _StubSearchArgs

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
        llm_embedding_model="test-embedding-model",
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
        embedding_model_client=client,
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
        # This run ended on a text reply, so every tool result already rode into a
        # later call and was written down there — nothing trails (#1778).  The record
        # of an ordinary run is byte-identical to before the tail existed.
        assert all(row.trailing_messages is None for row in db.messages.recent_prompts(limit=10))

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


class TestTerminalToolResultRecorded:
    """A run that ends the instant a tool returns still records that result (#1778).

    A tool result is otherwise durable only as a side effect of being fed back — it
    rides into the NEXT call's ``messages``.  A run with no next call (``max_steps``
    reached on a tool step, a write-gate STOP, a reroll abort, an exception) wrote its
    terminal outcome nowhere, and those are exactly the runs worth reading afterwards.
    """

    @pytest.mark.asyncio
    async def test_run_ending_on_a_tool_call_stamps_the_tail_on_its_record(self, test_db, mock_llm):
        """``max_steps`` reached on a tool step: the terminal call + its result land on
        the run's last prompt row, in the same wire shape the next call would have
        carried them in."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)
        # Keep tools available on the last step (the collector/terminator shape), so the
        # run can END on a tool call rather than be forced into a text answer.
        agent._keep_tools_on_final_step = True

        mock_llm.set_response_handler(
            lambda request, _count: mock_llm._make_tool_call_response(
                request, "search", {"query": "kites"}
            )
        )

        response = await agent.run("find kites", max_steps=max_steps, run_id="run-tail")
        assert response.answer == PennyResponse.AGENT_MAX_STEPS

        tail = db.messages.get_run_prompts("run-tail")[-1].get_trailing_messages()
        assert [message["role"] for message in tail] == ["assistant", "tool"]
        assert tail[0]["tool_calls"][0]["function"]["name"] == "search"
        assert tail[1]["tool_call_id"] == tail[0]["tool_calls"][0]["id"]
        assert "Mock search results for testing" in tail[1]["content"]
        # The structural per-call execution stamp rides along, so the terminal call's
        # success is a boolean read like every other call's.
        assert tail[1][PennyConstants.TOOL_RESULT_SUCCESS_KEY] is True

        await agent.close()

    @pytest.mark.asyncio
    async def test_exception_out_of_the_loop_still_stamps_the_tail(self, test_db, mock_llm):
        """The tail survives a run that DIED — the boundary box lives on the caller, so
        an exception thrown out of the loop body still closes the record."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._keep_tools_on_final_step = True

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "kites"})
            raise RuntimeError("the endpoint died mid-run")

        mock_llm.set_response_handler(handler)

        with pytest.raises(RuntimeError):
            await agent.run("find kites", max_steps=max_steps, run_id="run-died")

        tail = db.messages.get_run_prompts("run-died")[-1].get_trailing_messages()
        assert [message["role"] for message in tail] == ["assistant", "tool"]
        assert "Mock search results for testing" in tail[1]["content"]

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
    async def test_hallucinated_tool_call_is_stripped_and_the_run_closes_honestly(
        self, test_db, mock_llm
    ):
        """A final-step tool call, offered no tools, is stripped and the run REPORTS that
        it ended with nothing to say.

        The strong nudge that used to answer this retired with the empty-content
        validator (#1937): a draw with tool calls survives the reroll guard by
        construction, the repair strips them, and nothing left in the chain speaks to
        what remains — so the loop's own close states the outcome (FALLBACK_RESPONSE,
        since the run did call a tool first) instead of appending a user turn ordering
        the model to answer.  One model call per step, no extra round trip."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            # Final step: model hallucinates a tool call despite no tools offered
            return mock_llm._make_tool_call_response(request, "search", {"query": "more"})

        mock_llm.set_response_handler(handler)

        response = await agent.run("test query", max_steps=max_steps)
        assert response.answer == PennyResponse.FALLBACK_RESPONSE
        assert len(mock_llm.requests) == 2
        # The final step was offered no tools, and no nudge turn was appended for it.
        assert mock_llm.requests[1]["tools"] is None
        assert [m["content"] for m in mock_llm.requests[1]["messages"] if m["role"] == "user"] == [
            "test query"
        ]
        # The run's own work still travels with the close (#1776).
        assert [record.tool for record in response.tool_calls] == ["search"]

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


def _tool_result_messages(mock_llm, needle: str) -> list[str]:
    """All TOOL-role message contents (across recorded requests) containing ``needle``.

    The same rejection/result appears in every subsequent request's history, so callers
    read ``[0]`` — the identical copies collapse to one expected string.
    """
    return [
        message["content"]
        for request in mock_llm.requests
        for message in request["messages"]
        if message.get("role") == MessageRole.TOOL and needle in message["content"]
    ]


def _failed_dedup_frame(tool_name: str, first_line: str) -> str:
    """The whole OUTCOME-FIRST FAILED dedup-rejection render (#1673) for assertions —
    narration + the ``(<tool> result)`` machine tag + the FAILED body quoting the
    prior failure's first line."""
    narration = Prompt.DUPLICATE_CALL_NARRATION_FAILED.format(tool_name=tool_name)
    body = Prompt.DUPLICATE_CALL_REJECTION_FAILED.format(tool_name=tool_name, first_line=first_line)
    return f"{narration} ({tool_name} result)\n{body}"


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

        # The prior search SUCCEEDED, so the repeat is rejected with the OUTCOME-FIRST
        # SUCCEEDED frame (#1673): the narration states it succeeded and its result is
        # above, the retained (<tool> result) tag stays, and the reuse-the-result body
        # is preserved verbatim — narration + tag + preserved body.
        repeat_tool_messages = _tool_result_messages(
            mock_llm, "You already made this exact tool call"
        )
        assert repeat_tool_messages
        narration = Prompt.DUPLICATE_CALL_NARRATION_SUCCEEDED.format(tool_name="search")
        assert repeat_tool_messages[0] == (
            f"{narration} (search result)\n{Prompt.DUPLICATE_CALL_REJECTION_SUCCEEDED}"
        )
        assert "it succeeded" in repeat_tool_messages[0]
        assert "(search result)" in repeat_tool_messages[0]

        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_after_remediation_runs(self, test_db, mock_llm):
        """A FAILED write, a successful mutating remediation, then the IDENTICAL retry
        RUNS again (#1673) — and its result is the executed tool's, not a rejection."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=6)

        state = {"created": False}

        async def execute(tool_call):
            if tool_call.tool == "collection_set":
                state["created"] = True
                return ToolResult(message="Created collection widget-log", mutated=True)
            if tool_call.tool == "collection_write":
                if state["created"]:
                    return ToolResult(message="Wrote entry to widget-log", mutated=True)
                return ToolResult(
                    message="Memory widget-log not found — create it first", success=False
                )
            return ToolResult(message="ok")

        agent._tool_executor.execute = execute

        write_args = {"name": "widget-log", "content": "x"}

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "collection_set", {"name": "widget-log"}
                )
            if count == 3:
                # Byte-identical to count 1 — normally blocked, but the create mutated
                # since the failure, clearing the seen-calls cache, so the retry runs.
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"

        writes = [r for r in response.tool_calls if r.tool == "collection_write"]
        # The retry EXECUTED (two write records), rather than being blocked (one).
        assert len(writes) == 2
        assert writes[0].failed is True
        # The retry ran the real tool: its record is the executed result, not a rejection.
        assert writes[1].failed is False
        assert writes[1].mutated is True
        assert writes[1].result == "Wrote entry to widget-log"
        # No duplicate-rejection frame was injected for the write — it ran.
        assert _tool_result_messages(mock_llm, "nothing was saved") == []

        await agent.close()

    @pytest.mark.asyncio
    async def test_reread_after_write_runs(self, test_db, mock_llm):
        """The cache clears in the READ direction too (#1673): a read that FAILED, then
        a successful write to that key, then the IDENTICAL read RUNS again and succeeds —
        the mutation makes the previously-seen read forgettable, not just a failed write."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=6)

        state = {"written": False}

        async def execute(tool_call):
            if tool_call.tool == "collection_write":
                state["written"] = True
                return ToolResult(message="Wrote entry to widget-log", mutated=True)
            if tool_call.tool == "collection_read_latest":
                if state["written"]:
                    return ToolResult(message="1 entry: the value")
                return ToolResult(
                    message="Memory widget-log has no entry for key foo", success=False
                )
            return ToolResult(message="ok")

        agent._tool_executor.execute = execute

        read_args = {"name": "widget-log", "key": "foo"}
        write_args = {"name": "widget-log", "key": "foo", "content": "x"}

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "collection_read_latest", dict(read_args)
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            if count == 3:
                # Byte-identical to count 1 — the write mutated since, clearing the cache,
                # so the re-read RUNS instead of being blocked.
                return mock_llm._make_tool_call_response(
                    request, "collection_read_latest", dict(read_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"

        reads = [r for r in response.tool_calls if r.tool == "collection_read_latest"]
        # The re-read EXECUTED (two read records), rather than being blocked (one).
        assert len(reads) == 2
        assert reads[0].failed is True
        assert reads[1].failed is False
        assert reads[1].result == "1 entry: the value"
        # No duplicate-rejection frame — the re-read ran.
        assert _tool_result_messages(mock_llm, "nothing was saved") == []

        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_without_remediation_blocked_states_failure(self, test_db, mock_llm):
        """A FAILED call repeated with NO intervening mutation is blocked (#1673), and
        the rejection frame states the prior FAILURE and quotes its first line."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)

        failure = "Memory widget-log not found — create it first"

        async def execute(tool_call):
            if tool_call.tool == "collection_write":
                return ToolResult(message=failure, success=False)
            return ToolResult(message="ok")

        agent._tool_executor.execute = execute

        write_args = {"name": "widget-log", "content": "x"}

        def handler(request, count):
            if count <= 2:
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"

        writes = [r for r in response.tool_calls if r.tool == "collection_write"]
        # No mutation intervened, so the identical retry was blocked — one write only.
        assert len(writes) == 1

        rejections = _tool_result_messages(mock_llm, "nothing was saved")
        assert rejections
        # Whole-render: OUTCOME-FIRST FAILED frame quoting the prior failure's first line.
        assert rejections[0] == _failed_dedup_frame("collection_write", failure)

        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_after_remediation_is_loop_safe(self, test_db, mock_llm):
        """Loop-safety (#1673): an allowed retry that FAILS AGAIN adds no mutation, so a
        third identical attempt is blocked — the allowance can't drive an infinite loop."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=8)

        failure = "Memory widget-log not found — create it first"

        async def execute(tool_call):
            if tool_call.tool == "collection_set":
                return ToolResult(message="Created collection widget-log", mutated=True)
            if tool_call.tool == "collection_write":
                # The write keeps failing even after the create (a still-unmet precondition).
                return ToolResult(message=failure, success=False)
            return ToolResult(message="ok")

        agent._tool_executor.execute = execute

        write_args = {"name": "widget-log", "content": "x"}

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "collection_set", {"name": "widget-log"}
                )
            if count == 3:
                # Allowed retry (prior failed + create mutated) — runs, fails again.
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            if count == 4:
                # Third identical attempt: prior (the retry) failed, NO mutation since → blocked.
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "done"

        writes = [r for r in response.tool_calls if r.tool == "collection_write"]
        # W1 and the allowed retry executed; the THIRD attempt was blocked — two records.
        assert len(writes) == 2
        assert all(w.failed for w in writes)

        rejections = _tool_result_messages(mock_llm, "nothing was saved")
        assert rejections
        assert rejections[0] == _failed_dedup_frame("collection_write", failure)

        await agent.close()


class TestResultNarration:
    """`Tool.format_result` frames a tool result as tagged, first-person narration.

    The generic default: a first-person line + a retained ``(<tool> result)``
    machine tag + the body.  The tag is load-bearing — a live-model probe showed
    pure-prose narration with no tag raised the call-as-text bail rate, so it stays
    even as the header reads naturally.  Success and failure narrate differently,
    both keeping the tag.  Exercised via an *unregistered* tool name so it hits the
    generic default; per-tool overrides (browse #1480, memory #1481, …) are covered
    by each tool's own tests.
    """

    def test_success_result_carries_narration_tag_and_body(self):
        framed = Tool.format_result(
            "example_tool",
            {"queries": ["deepest lake"]},
            ToolResult(message="Lake Baikal is the deepest lake"),
        )
        assert framed == (
            "You used `example_tool` and here's the result: (example_tool result)\n"
            "Lake Baikal is the deepest lake"
        )

    def test_failure_result_narrates_honestly_and_keeps_the_tag(self):
        framed = Tool.format_result(
            "example_tool",
            {"queries": ["deepest lake"]},
            ToolResult(message="Error: no", success=False),
        )
        assert framed == (
            "You tried to use `example_tool` but it didn't work: (example_tool result)\nError: no"
        )


class TestModelErrorHandling:
    """`_invoke_model` swallows LlmError → returns AGENT_MODEL_ERROR; other exceptions propagate."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_agent_model_error(self, test_db, mock_llm):
        """Connection/response errors from the LLM result in AGENT_MODEL_ERROR, not a crash
        — and the aborted run names its own cause (#1909): the failing call writes no
        promptlog row, so the error's class and message live only on this record."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            raise LlmConnectionError("backend down")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert response.abort is not None
        assert response.abort == RunAbort(
            step=1,
            after_tool=None,
            error=ModelCallError(error_class="LlmConnectionError", message="backend down"),
        )
        # The first call died, so there is no tool to name — the reason says so rather
        # than claiming a step that never ran.
        assert response.abort.render() == (
            "model call failed at step 1: LlmConnectionError: backend down"
        )

        await agent.close()

    @pytest.mark.asyncio
    async def test_abort_names_the_step_and_the_last_successful_tool(self, test_db, mock_llm):
        """A run that dies MID-PROGRAM says where it had got to (#1909) — the step index
        plus the tool the last SUCCESSFUL step executed, which is the anchor the step
        number alone doesn't give.

        The last step here FAILED (a tool that doesn't exist), so naming the most recent
        call would point at the one thing the run never actually did; the anchor walks
        back to the read that landed."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "lakes"})
            if count == 2:
                return mock_llm._make_tool_call_response(request, "no_such_tool", {})
            raise LlmResponseError("500 Internal Server Error")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert response.abort is not None
        assert response.abort == RunAbort(
            step=3,
            after_tool="search",
            error=ModelCallError(
                error_class="LlmResponseError", message="500 Internal Server Error"
            ),
        )
        assert response.abort.render() == (
            "model call failed at step 3 after search: LlmResponseError: 500 Internal Server Error"
        )

        await agent.close()


class TestToolParseErrorReroll:
    """A 500 'error parsing tool call' is an INVALID DRAW (#1839): the backend refused
    to parse it as a call, so there is no usable output — the loop discards it and
    re-rolls the UNCHANGED context on the shared reroll budget.  No format nudge is
    injected, so nothing about the failed draw enters the conversation."""

    @pytest.mark.asyncio
    async def test_tool_parse_error_rerolls_the_unchanged_context(self, test_db, mock_llm):
        """The failed draw leaves NO trace: the redraw carries byte-identical messages,
        with no nudge user-turn appended."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                raise LlmToolParseError("error parsing tool call: raw='We need to produce...'")
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "here's what I found")

        mock_llm.set_response_handler(handler)

        await agent.run("test prompt", max_steps=max_steps)

        # The redraw is the SAME call again on byte-identical messages — the failed draw
        # left no assistant turn and no nudge user-turn behind.
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        assert "could not be parsed" not in str(mock_llm.requests[1]["messages"])

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_parse_error_recovers_and_completes(self, test_db, mock_llm):
        """A clean redraw proceeds as if the invalid one never happened — the reroll
        happens INSIDE the step, so the run still gets its full step budget."""

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
    async def test_persistent_tool_parse_error_fails_the_run(self, test_db, mock_llm):
        """Budget exhausted → the run fails honestly via the existing aborted-run path,
        after exactly DEGENERATE_REROLL_ATTEMPTS draws (the ONE shared budget) — and the
        abort names the tripped condition, once per discarded draw (#1909), so a run
        killed by parse failures is distinguishable from one killed by collapses."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            raise LlmToolParseError("error parsing tool call: raw='plain text again...'")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        assert response.abort is not None
        assert response.abort == RunAbort(
            step=1,
            after_tool=None,
            error=ModelCallError(
                error_class=REROLL_EXHAUSTED,
                message=("3 unusable draws: tool_parse_error, tool_parse_error, tool_parse_error"),
            ),
        )
        assert response.abort.render() == (
            "model call failed at step 1: reroll-exhausted: 3 unusable draws: "
            "tool_parse_error, tool_parse_error, tool_parse_error"
        )

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
        assert response.abort is not None
        assert response.abort.error == ModelCallError(
            error_class="LlmTimeoutError", message="Request timed out."
        )
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


class TestDegenerateOutputGuard:
    """gpt-oss occasionally collapses into a punctuation run ("...??…?..").  The
    loop discards that output and re-rolls on the UNCHANGED context (never appending
    the garbage — that's the contagion path), and throws the run out if it can't
    recover, so no poison is ever fed back to the model or reaches a tool call."""

    @pytest.mark.asyncio
    async def test_tool_arg_poison_discarded_and_rerolled(self, test_db, mock_llm):
        """A degenerate run inside a tool-call argument (the common case, which the
        validation chain never sees) is discarded and re-rolled — the poison turn is
        never appended, so the reroll re-sends the exact same context."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "the new Air‑...??…?..?????"}
                )
            return mock_llm._make_text_response(request, "recovered cleanly")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test query", max_steps=max_steps)
        assert response.answer == "recovered cleanly"
        # Exactly one reroll, both calls inside the same step.
        assert len(mock_llm.requests) == 2
        # The garbage was DISCARDED, not appended — it never reached the context.
        assert "?????" not in str(mock_llm.requests[-1]["messages"])

        await agent.close()

    @pytest.mark.asyncio
    async def test_content_poison_discarded_and_rerolled(self, test_db, mock_llm):
        """A degenerate run in plain text content is caught on the same path."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, "Here you go … … … … …")
            return mock_llm._make_text_response(request, "here is the real answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "here is the real answer"
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        # The PERSISTED twin of that identity: the discarded draw is logged before it is
        # inspected, so the re-roll leaves two promptlog rows carrying the same context.  That
        # repeat is the only trace it leaves — the eval harness's loop-health advisory reads it
        # (#1841), so the premise is pinned here against a real re-roll, not just seeded rows.
        assert draw_rerolled(db)

        await agent.close()

    @pytest.mark.asyncio
    async def test_call_fragment_reply_discarded_and_rerolled(self, test_db, mock_llm):
        """A would-be final reply that is a bare JSON call fragment
        (``{"memory": "rip? wait …"}`` — the #1570 field audit's observed leak)
        is discarded and re-rolled on the unchanged context, exactly like the
        other unusable-output conditions — it must never be SENT verbatim.

        Chat-only polarity: ``ChatAgent`` declares it in ``invalid_draw_conditions``;
        the base agent declares none, so plain text is a legitimate answer there.
        The full ``{"name": …, "arguments": …}`` envelope belongs to the sibling
        ``CALL_AS_TEXT`` condition, so the discarded draw is logged under the shape
        that actually describes it."""
        assert ChatAgent.invalid_draw_conditions == (
            (ConditionKey.CALL_AS_TEXT, is_call_as_text_bail),
            (ConditionKey.CALL_FRAGMENT_REPLY, is_call_fragment_reply),
            (ConditionKey.EMPTY, is_empty_draw),
        )
        assert Agent.invalid_draw_conditions == ()
        # Predicate edges: a bare fragment and a bare `{}` empty-object reply (the #1732
        # nudge-loop tail) match; the full call envelope and mid-prose JSON do not
        # (zero-false-positive discipline).
        assert is_call_fragment_reply('{"memory": "rip? wait we need entries"}') is True
        assert is_call_fragment_reply("{}") is True
        assert is_call_fragment_reply("{ }\n") is True
        assert is_call_fragment_reply('{"name": "done", "arguments": {}}') is False
        assert is_call_fragment_reply('I saved it as {"key": "price"} for you') is False
        assert is_call_fragment_reply("An empty JSON object is {}.") is False

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        # The ChatAgent polarity, declared on the test agent.
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, '{"memory": "rip? wait we need"}')
            return mock_llm._make_text_response(request, "here is the real answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps)
        assert response.answer == "here is the real answer"
        # Exactly one reroll on the unchanged context — the fragment never appended.
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_degenerate_tool_name_discarded_and_rerolled(self, test_db, mock_llm):
        """A collapse landing in the tool-call NAME field (an unregistered,
        collapse-shaped name like `Functions?????`) is the same poison as an
        argument collapse: the response is discarded and re-rolled on the
        unchanged context — no tool-not-found error result ever enters the
        conversation (that feedback is the contagion path)."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "Functions?????", {"query": "x"})
            return mock_llm._make_text_response(request, "recovered cleanly")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test query", max_steps=max_steps)
        assert response.answer == "recovered cleanly"
        # Exactly one reroll on the unchanged context.
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        # Neither the garbage name nor a tool-not-found result reached the context.
        reroll_messages = str(mock_llm.requests[-1]["messages"])
        assert "?????" not in reroll_messages
        assert "not found" not in reroll_messages.lower()

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_degeneration_aborts_run(self, test_db, mock_llm):
        """When every reroll is still degenerate, the run is thrown out with
        AGENT_MODEL_ERROR after exactly DEGENERATE_REROLL_ATTEMPTS calls — poison is
        never acted on or stored — and the abort names the collapse as the condition
        that spent the budget (#1909)."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            return mock_llm._make_tool_call_response(request, "search", {"query": "...??…?..?????"})

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        assert response.abort is not None
        assert response.abort.error == ModelCallError(
            error_class=REROLL_EXHAUSTED,
            message="3 unusable draws: degenerate_output, degenerate_output, degenerate_output",
        )

        await agent.close()

    @pytest.mark.asyncio
    async def test_reroll_guard_shadows_the_send_gate_on_quoted_collapse(self, test_db, mock_llm):
        """PINS the current shadowing behaviour (follow-up #1397).

        After #1386 the send gate (``half_formed_send_reason``) judges a message as a
        whole, so a substantive `quality` suggestion that QUOTES a degeneration-collapse
        ("......???") it observed would be DELIVERED — the send gate does not refuse it.
        But that suggestion never reaches the send gate: the agent-loop reroll guard runs
        ``is_degenerate_run`` on the SERIALIZED tool-call arguments of every call, so it
        discards + re-rolls the whole response upstream.  The two gates DISAGREE and the
        reroll guard wins — which is why fixing only the send gate cannot deliver a
        suggestion quoting a genuine collapse.  #1397 tracks closing that gap (paraphrase
        in the quality prompt, or a send-scoped whole-message check in the reroll guard —
        the corpus/entry-content substring check must stay strict either way)."""
        suggestion = (
            'The board-game-news collector sent "Hi there! ......???" before the real '
            "note. Fix: compose the complete message first, then send once."
        )
        # The send gate would ALLOW this substantive, quoting message post-#1386 ...
        assert half_formed_send_reason(suggestion) is None
        # ... but the reroll guard's predicate fires on the embedded collapse run.
        assert is_degenerate_run(suggestion) is True

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                # A send-shaped tool call whose content quotes the collapse.
                return mock_llm._make_tool_call_response(request, "search", {"query": suggestion})
            return mock_llm._make_text_response(request, "recovered cleanly")

        mock_llm.set_response_handler(handler)

        response = await agent.run("review the collector run", max_steps=max_steps)
        # The quoting response was DISCARDED and re-rolled (one reroll, poison never
        # appended) — it never reached the send gate that would have allowed it.
        assert response.answer == "recovered cleanly"
        assert len(mock_llm.requests) == 2
        assert "......???" not in str(mock_llm.requests[-1]["messages"])

        await agent.close()


# A leaked Harmony tool-call envelope in the text content — the whole call arrives
# as literal prose (generic `browse` tool) instead of parsed `tool_calls`, the
# shape some non-Ollama gpt-oss backends emit.
_HARMONY_LEAK = "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"


class TestHarmonyEnvelopeLeakGuard:
    """A backend that fails to parse gpt-oss's Harmony format leaks the whole tool
    call into ``message.content`` as literal control-token text; ``tool_calls`` is
    empty, so nothing downstream catches it and the raw envelope would be delivered
    to the user verbatim.  Reusing the degeneracy discard-and-reroll machinery, the
    loop discards that output and re-draws on the UNCHANGED context (the leak is
    intermittent), and throws the run out if it persists — never reconstructing the
    call from the envelope grammar."""

    def test_detector_has_no_false_positives(self):
        """The detector fires on a leaked envelope but NOT on ordinary prose or a
        code reply that merely contains an ellipsis — the same zero-false-positive
        discipline the degeneracy regex holds."""
        assert has_leaked_harmony_envelope(_HARMONY_LEAK) is True
        assert has_leaked_harmony_envelope("Sure, here's the answer.") is False
        assert (
            has_leaked_harmony_envelope("The slice is `nums[1:]` and `foo(...)` returns.") is False
        )

    @pytest.mark.asyncio
    async def test_leaked_envelope_discarded_and_rerolled(self, test_db, mock_llm):
        """A leaked Harmony envelope in the text content is discarded and re-rolled
        on the unchanged context — the raw tokens never reach the context (or the
        user), and the fresh draw comes back clean."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, _HARMONY_LEAK)
            return mock_llm._make_text_response(request, "here is the real answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what's the deepest lake?", max_steps=max_steps)
        assert response.answer == "here is the real answer"
        # Exactly one reroll on the unchanged context; the leak was never appended.
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        assert "<|call|>" not in str(mock_llm.requests[-1]["messages"])

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_leak_aborts_run(self, test_db, mock_llm):
        """When every reroll still leaks the envelope, the run is thrown out with
        AGENT_MODEL_ERROR after exactly DEGENERATE_REROLL_ATTEMPTS calls — raw
        Harmony tokens are never delivered — and the abort names the leak rather than
        the collapse the same budget also serves (#1909)."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            return mock_llm._make_text_response(request, _HARMONY_LEAK)

        mock_llm.set_response_handler(handler)

        response = await agent.run("what's the deepest lake?", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        assert response.abort is not None
        assert response.abort.error == ModelCallError(
            error_class=REROLL_EXHAUSTED,
            message="3 unusable draws: tool_call_leak, tool_call_leak, tool_call_leak",
        )

        await agent.close()


class TestEmptyContentFallback:
    """A final answer carrying no usable content closes HONESTLY (#1776).

    This is the base agent's shape: it declares no invalid draws, so an empty draw is
    not re-rolled and reaches the close.  Chat's empty draw never gets here — since
    #1937 it is discarded and re-rolled upstream (``TestChatEmptyDrawReroll``); what
    still lands here on the chat path is a tools-stripped final step whose hallucinated
    tool call the repair validator strips away, leaving nothing to say."""

    @pytest.mark.asyncio
    async def test_empty_response_returns_agent_empty_response(self, test_db, mock_llm):
        """When the model returns empty content, AGENT_EMPTY_RESPONSE is returned.

        One model call, not two: nothing asks the model to try again any more (#1937)."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE
        assert len(mock_llm.requests) == 1

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_response_after_tool_call(self, test_db, mock_llm):
        """FALLBACK_RESPONSE is returned when model returns empty after preceding tool calls.

        The distinction matters: FALLBACK_RESPONSE signals "worked but couldn't
        synthesise", AGENT_EMPTY_RESPONSE "never tried to answer".  The records travel
        with it (#1776) — a run that called tools and then said nothing must not record
        as a callless one, since everything reading a finished run reads ``tool_calls``.
        """
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)

        def handler(request, count):
            if count <= 3:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"query {count}"}
                )
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.FALLBACK_RESPONSE
        assert [record.tool for record in response.tool_calls] == ["search"] * 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_stripped_hallucinated_call_closes_honestly(self, test_db, mock_llm):
        """A final-step tool call (tools stripped) is repaired away, and what is left is
        an empty answer the run REPORTS rather than nudges about.

        ``HallucinatedToolCallRepair`` strips the calls and the content falls through a
        chain that no longer has anything to say about emptiness (#1937), so the close
        states that the run said nothing instead of appending a strong-nudge user turn."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_tool_call_response(
                request, "search", {"query": "x"}
            )
        )

        response = await agent.run("test prompt", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE
        assert len(mock_llm.requests) == 1

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
        # Each is wrapped by Tool.format_result — a tagged first-person narration
        # of the (successful) call plus its unchanged body.
        assert len(agent._tool_result_text) == 3
        assert agent._tool_result_text == [
            "You used `search` and here's the result: (search result)\nresult_A",
            "You used `search` and here's the result: (search result)\nresult_B",
            "You used `search` and here's the result: (search result)\nresult_C",
        ]

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

        tool = BrowseTool(
            max_calls=5,
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(queries=["best pizza toronto"], extract="the page content")

        assert len(browsed_urls) == 1
        search_url = list(browsed_urls.keys())[0]
        assert search_url.startswith("https://duckduckgo.com/?q=")
        assert "best%20pizza%20toronto" in search_url

    @pytest.mark.asyncio
    async def test_text_queries_fail_without_browser(self, test_db, mock_llm, monkeypatch):
        """A whole-channel outage (no browser connected) is named ONCE and binds the
        terminal move — NOT once per query as N page failures inviting variant retries.
        Three doomed queries render a single ``## browse error:`` outage banner, and
        because every query errored the result reports ``success=False`` so the failure
        is visible to structural accounting, not just the error text."""
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRIES", 0)
        monkeypatch.setattr(PennyConstants, "BROWSE_RETRY_DELAY", 0.0)
        tool = BrowseTool(
            max_calls=5,
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
        )

        result = await tool.execute(
            queries=["best pizza toronto", "https://example.test/a", "https://example.test/b"],
            extract="the page content",
        )

        assert result.success is False
        # Named once, not per-URL: three doomed queries → a single outage banner.
        assert result.message.count(PennyConstants.BROWSE_ERROR_HEADER) == 1
        assert "no browser is connected" in result.message
        # Binds the recovery instead of the per-page "try a different source" that
        # invites the doomed URL-variant retries.
        assert "won't help" in result.message
        assert "try a different source" not in result.message.lower()
        assert PennyConstants.BROWSE_PAGE_HEADER not in result.message

    @pytest.mark.asyncio
    async def test_converging_retry_after_failed_browses_keeps_its_steps(self, test_db, mock_llm):
        """Two browse calls that each fully fail (``success=False`` — the way BrowseTool
        reports an all-queries-failed call) no longer end the turn (#1776): the model
        reads the actionable errors, changes its arguments, and its THIRD call — the
        converging one — still runs and carries the answer.

        The removed all-tools-failed abort fired here, on failure COUNT alone, without
        looking at the arguments — so it could not tell a stuck run from a recovering
        one and cut this exact shape off with most of its steps unused.  Byte-identical
        repeats stay blocked by the duplicate-call cache, so the only behaviour the guard
        added on top was ending turns whose arguments were changing."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)
        attempts: list[str] = []

        async def failing_then_working(tool_call):
            attempts.append(tool_call.arguments["queries"][0])
            if len(attempts) <= 2:
                return ToolResult(
                    message=f"{PennyConstants.BROWSE_ERROR_HEADER}q\nCould not read this page.",
                    success=False,
                )
            return ToolResult(message="## browse https://example.test/c:\nthe page text")

        agent._tool_executor.execute = failing_then_working

        def handler(request, count):
            if count <= 3:
                # A DIFFERENT query each time — the model correcting, not repeating.
                return mock_llm._make_tool_call_response(
                    request, "browse", {"queries": [f"q{count}"]}
                )
            return mock_llm._make_text_response(request, "here's what the page said")

        mock_llm.set_response_handler(handler)

        response = await agent.run("look this up", max_steps=max_steps)

        # The turn reached the answer instead of stopping after the second failure.
        assert response.answer == "here's what the page said"
        assert attempts == ["q1", "q2", "q3"]
        assert [record.failed for record in response.tool_calls] == [True, True, False]

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_queries_rejected_at_arg_gate(self, test_db, mock_llm):
        """An empty ``queries`` list is rejected by ``BrowseArgs`` at the ``run``
        gate before ``execute`` runs — so an empty browse can't silently no-op —
        with an actionable message pointing at queries."""
        tool = BrowseTool(
            max_calls=5,
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
        )

        result = await tool.run(queries=[], extract="the page content")

        assert result.success is False
        assert "queries" in result.message
        assert "search query or URL" in result.message

        # extract is REQUIRED (#1570 — every browse routes through a micro-context;
        # the page never enters the main context whole).  A missing or blank extract
        # is rejected at the same gate with the fix, never treated as "whole page".
        missing = await tool.run(queries=["https://x.test"])
        assert missing.success is False and "extract" in missing.message
        blank = await tool.run(queries=["https://x.test"], extract="   ")
        assert blank.success is False and "extract" in blank.message
        assert BrowseArgs(queries=["q"], extract="the price").extract == "the price"

    @pytest.mark.asyncio
    async def test_urls_always_route_to_browse(self, test_db, mock_llm):
        """URLs always go to BrowseTool regardless of browser connection."""
        browsed_urls: list[str] = []

        async def fake_request(command, params):
            browsed_urls.append(params["url"])
            return (f"Page content from {params['url']}", None)

        request_fn = AsyncMock(side_effect=fake_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(
            max_calls=5,
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(
            queries=["https://example.com/page", "https://other.com"],
            extract="the page content",
        )

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

        tool = BrowseTool(
            max_calls=5,
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        result = await tool.execute(
            queries=["https://slow.example.com"], extract="the page content"
        )

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert PennyConstants.BROWSE_ERROR_HEADER in result.message
        assert "slow.example.com" in result.message

    def test_browse_tool_timeout_exceeds_request_timeout(self):
        """BrowseTool.timeout must exceed the per-URL BROWSE_REQUEST_TIMEOUT.

        Ensures the inner per-URL timeout fires before the outer executor
        timeout, so hung URLs produce graceful error sections instead of
        cancelling the entire tool call.
        """
        tool = BrowseTool(max_calls=3, embedding_client=cast(Any, MockLlmClient()))
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


class TestLargeToolResults:
    """A tool result is fed back whole — the client enforces per-page limits, the loop
    truncates nothing."""

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


class TestAllToolsFailedRunsOutItsSteps:
    """The all-tools-failed abort is GONE (#1776) — ``max_steps`` is the only bound on
    the loop, and an all-failed run closes honestly on the run record rather than being
    truncated mid-recovery."""

    @pytest.mark.asyncio
    async def test_every_tool_call_failing_uses_the_whole_step_budget(self, test_db, mock_llm):
        """A run whose every tool call fails keeps going instead of stopping at the old
        two-failure threshold, so the model spends the budget it was given trying to
        recover — and composes the close itself, over the failures it can see, rather
        than having the turn cut off under it."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=5)
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message="API unavailable", success=False)
        )

        def handler(request, count):
            # Model keeps trying tool calls, changing the query each time — all fail —
            # until the final step, where the loop has stripped its tools.
            if count <= max_steps - 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"attempt {count}"}
                )
            return mock_llm._make_text_response(request, "I couldn't reach any source for that")

        mock_llm.set_response_handler(handler)
        response = await agent.run("what's the news?", max_steps=max_steps)

        assert response.answer == "I couldn't reach any source for that"
        # A failing call on every step that HAS tools — the old guard stopped at 2.
        assert len(response.tool_calls) == max_steps - 1
        assert all(record.failed for record in response.tool_calls)

        await agent.close()

    @pytest.mark.asyncio
    async def test_failed_calls_stay_honest_in_the_context_the_model_reads(self, test_db, mock_llm):
        """Removing the abort must not let a failed run read as a successful one: every
        failed call is still framed to the model as a failure, verbatim, with its error
        body — the honesty the abort's canned answer used to stand in for lives here, in
        the per-call frames (whole-render pinned in ``TestToolResultFraming``) plus the
        ``error`` the run record carries, not in ending the turn early."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message="API unavailable", success=False)
        )
        events = []

        async def on_progress(event):
            events.append(event)

        def handler(request, count):
            return mock_llm._make_tool_call_response(
                request, "search", {"query": f"attempt {count}"}
            )

        mock_llm.set_response_handler(handler)
        response = await agent.run("what's the news?", max_steps=max_steps, on_progress=on_progress)

        assert response.tool_calls  # the run really did make failing calls
        failure_frame = "You tried to use `search` but it didn't work: (search result)\n"
        assert agent._tool_result_text == [f"{failure_frame}API unavailable"] * len(
            response.tool_calls
        )
        # And the run is classified an error from what actually happened, so nothing
        # downstream reads an all-failed run as a completed one.
        assert events[-1].event == "run_finished"
        assert events[-1].outcome == "error"

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

    @pytest.mark.asyncio
    async def test_structured_progress_covers_run_steps_tools_and_finish(self, test_db, mock_llm):
        """Progress events bracket the loop and identify each tool batch."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))
        events = []

        async def on_progress(event):
            events.append(event)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "topic"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        response = await agent.run("test", max_steps=max_steps, on_progress=on_progress)

        assert response.answer == "done"
        assert [event.event for event in events] == [
            "run_started",
            "step_started",
            "tools_started",
            "step_started",
            "run_finished",
        ]
        assert events[2].tools == (("search", {"query": "topic"}),)
        assert events[-1].outcome == "completed"

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


class _StubTerminator(Tool):
    """A terminator tool for the LOOP tests only.

    Production has none since #1911 — a chat turn ends by replying and a collector
    cycle ends when its program's calls are covered — but the loop mechanics these
    tests pin (the reroll of an invalid draw, batched-call dedup, the step budget) all
    need SOME call that ends a background run to be exercised at all.  Supplying one
    here keeps that coverage without asserting a surface production doesn't have."""

    name = "stub_terminator"
    description = "End the run."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(message="Cycle complete.")


class _TerminatingBackgroundAgent(BackgroundAgent):
    """A background agent that stops on :class:`_StubTerminator` — the seam a real
    ``Collector`` fills with its program-coverage read."""

    def should_stop_loop(self, records: list[ToolCallRecord]) -> bool:
        return any(record.tool == _StubTerminator.name and not record.failed for record in records)


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
        llm_embedding_model="test-embedding-model",
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
    agent = _TerminatingBackgroundAgent(
        system_prompt="test",
        model_client=client,
        embedding_model_client=client,
        tools=[StubSearchTool(), _StubTerminator()],
        db=db,
        config=config,
    )
    return agent, db, max_steps


class TestCollectorInvalidDrawReroll:
    """A collector acts ONLY through tool calls, so a draw carrying none is an INVALID
    DRAW (#1839): the loop discards it and re-rolls the UNCHANGED context on the shared
    reroll budget.  Nothing about the invalid draw enters the conversation — no stray
    assistant turn, no nudge user-turn, no marker on the run's prompt rows — so a
    recovered run is indistinguishable from one that never slipped."""

    @pytest.mark.asyncio
    async def test_prose_draw_is_discarded_and_the_redraw_proceeds(self, test_db, mock_llm):
        """Work, then prose ("Done. Summary: ...") → the prose is thrown away and a
        clean redraw closes the cycle with a real done().

        The realistic production shape: the model does real work, THEN narrates
        completion as prose instead of calling done()."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, "**Done. Summary: wrote the entry.**")
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        # Exactly one reroll, on the UNCHANGED context — the prose never appended and
        # no nudge user-turn injected.
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]
        redraw = str(mock_llm.requests[2]["messages"])
        assert "Done. Summary: wrote the entry." not in redraw
        assert "you act only through tool calls" not in redraw
        # The cycle closed with a real done() record (not a lost/failed cycle).
        assert any(record.tool == "stub_terminator" for record in response.tool_calls)

        await agent.close()

    @pytest.mark.asyncio
    async def test_redraw_can_be_more_work(self, test_db, mock_llm):
        """The redraw is a fresh draw on the same context, so it may be any legitimate
        move — here a work tool rather than a premature done()."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(
                    request, "**Observation** the page has an entry."
                )
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "more"})
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        tools_called = [record.tool for record in response.tool_calls]
        assert "search" in tools_called
        assert "stub_terminator" in tools_called

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_draw_is_the_same_invalid_draw(self, test_db, mock_llm):
        """An EMPTY draw (no text, no tool call) is a draw with no call like any other,
        so it takes the same reroll — the collector-flavoured empty-content nudge that
        used to answer it retired with the rest of the family."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, "")
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]
        assert "Please provide your response" not in str(mock_llm.requests[2]["messages"])
        assert any(record.tool == "stub_terminator" for record in response.tool_calls)

        await agent.close()

    @pytest.mark.asyncio
    async def test_done_as_json_text_is_the_same_invalid_draw(self, test_db, mock_llm):
        """The argless ``done()`` composed as a JSON envelope but never routed through
        the tool channel (gpt-oss's Harmony fallback, #1569) is discarded like any other
        call-less draw — the model's own real done() closes the cycle."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, '{"name": "done", "arguments": {}}')
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps)

        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]
        assert '{"name": "done"' not in str(mock_llm.requests[2]["messages"])
        done_records = [r for r in response.tool_calls if r.tool == "stub_terminator"]
        assert len(done_records) == 1
        assert done_records[0].arguments == {}

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_call_less_draws_fail_the_run(self, test_db, mock_llm):
        """A model that never makes a call never loops forever: the shared reroll budget
        bounds it and the run fails honestly, with no done record and no salvaged text."""
        agent, db, max_steps = _make_background_agent(test_db, max_steps=3)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "still just talking")
        )

        response = await agent.run("", max_steps=max_steps)

        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert not any(record.tool == "stub_terminator" for record in response.tool_calls)

        await agent.close()

    def test_the_condition_names_the_shape_where_it_can(self, test_db, mock_llm):
        """The reroll mechanism is shared, so the LOGGED condition must stay honest: a
        recognisable done envelope is ``DONE_JSON_BAIL``; everything else with no call
        is ``TEXT_INSTEAD_OF_TOOL``, keyed to the STATE (no call was made) and never to
        anything the prose happens to say."""
        agent, _db, _max_steps = _make_background_agent(test_db)
        assert BackgroundAgent.invalid_draw_conditions == (
            (ConditionKey.DONE_JSON_BAIL, is_done_json_bail),
            (ConditionKey.TEXT_INSTEAD_OF_TOOL, _any_text),
        )
        envelope = _text_response('{"name": "done", "arguments": {}}')
        assert agent._unusable_output_condition(envelope) == ConditionKey.DONE_JSON_BAIL
        for prose in ("Done. I wrote the entry.", "", '{"note": "not a done call"}'):
            assert (
                agent._unusable_output_condition(_text_response(prose))
                == ConditionKey.TEXT_INSTEAD_OF_TOOL
            ), prose
        # A draw that IS a call is never an invalid draw.
        assert agent._unusable_output_condition(_tool_response("search", {"query": "x"})) is None


class TestChatCallShapedTextReroll:
    """Chat's invalid draws are the ones that are not a REPLY (#1839/#1937).  A reply
    that is really a serialized tool call would be delivered to the user as raw
    machinery, so the loop discards it and re-rolls the unchanged context; an ordinary
    prose reply is chat's valid terminal state and is finalized untouched."""

    @pytest.mark.asyncio
    async def test_call_as_text_is_discarded_and_rerolled(self, test_db, mock_llm):
        """Real search, then a browse call emitted as JSON *text* → discarded, redrawn
        on the unchanged context, and the model's prose answer stands."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "deepest lake"}
                )
            if count == 2:
                return mock_llm._make_text_response(
                    request, '{"queries": ["deepest lake"], "reasoning": "read the page"}'
                )
            return mock_llm._make_text_response(request, "I couldn't find that app anywhere.")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what's the deepest lake?", max_steps=max_steps)

        # The JSON blob never became the reply and never entered the context.
        assert response.answer == "I couldn't find that app anywhere."
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]
        assert '"reasoning": "read the page"' not in str(mock_llm.requests[2]["messages"])

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_call_as_text_fails_the_turn(self, test_db, mock_llm):
        """Budget exhausted → the turn fails honestly rather than delivering the blob."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(
                request, '{"name": "browse", "arguments": {"queries": ["x"]}}'
            )
        )

        response = await agent.run("what's the deepest lake?", max_steps=max_steps)
        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS

        await agent.close()

    @pytest.mark.asyncio
    async def test_normal_prose_reply_is_valid(self, test_db, mock_llm):
        """A genuine prose reply is finalized as-is — chat's plain conversational reply
        stays a valid terminal state, so the guard must not fire on real answers."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "deepest lake"}
                )
            return mock_llm._make_text_response(request, "Lake Baikal is the deepest lake!")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what's the deepest lake?", max_steps=max_steps)
        assert response.answer == "Lake Baikal is the deepest lake!"
        assert len(mock_llm.requests) == 2  # no reroll

        await agent.close()

    def test_the_three_chat_shapes_and_their_edges(self, test_db, mock_llm):
        """Both Harmony fallback shapes and the say-nothing draw are invalid; a genuine
        reply is not.  The full envelope reads as ``CALL_AS_TEXT``, the mangled remainder
        as ``CALL_FRAGMENT_REPLY`` and a draw with nothing in it as ``EMPTY``, so the
        discarded draw is logged honestly.

        ORDER is the claim on the shared edge: a bare ``{}`` carries no letters either,
        so ``EMPTY`` is tried last and the ``{}`` tail keeps the truer name (#1732)."""
        agent, _db, _max_steps = _make_agent(test_db, mock_llm)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        assert is_call_as_text_bail('{"queries": ["x"], "reasoning": "y"}')
        assert is_call_as_text_bail('{"name": "browse", "arguments": {"queries": ["x"]}}')
        # A lone JSON object with no call markers, an envelope with a non-string name
        # or non-dict arguments, non-JSON prose, and prose that merely mentions JSON
        # are all NOT call-as-text bails (a genuine reply must pass through).
        for prose in (
            '{"note": "just data"}',  # no reasoning / not an envelope
            '{"name": 5, "arguments": {}}',  # non-string name
            '{"name": "browse", "arguments": "oops"}',  # non-dict arguments
            'The config looks like {"queries": [...]} roughly.',  # not a lone object
            "Lake Baikal is the deepest lake!",  # normal prose
            "",  # empty
        ):
            assert not is_call_as_text_bail(prose), prose

        envelope = _text_response('{"name": "browse", "arguments": {"queries": ["x"]}}')
        assert agent._unusable_output_condition(envelope) == ConditionKey.CALL_AS_TEXT
        fragment = _text_response('{"memory": "rip? wait we need"}')
        assert agent._unusable_output_condition(fragment) == ConditionKey.CALL_FRAGMENT_REPLY
        # The say-nothing family (#1937) — blank, separators, a lone emoji, a bare
        # ``<think>`` block — all read as EMPTY.
        for nothing in ("", "   \n ", "\n\n---", "🙂", "<think>reasoning only</think>"):
            assert is_empty_draw(nothing), nothing
            assert agent._unusable_output_condition(_text_response(nothing)) == ConditionKey.EMPTY
        # The shared edge: `{}` has no letters, but CALL_FRAGMENT_REPLY names it first.
        assert is_empty_draw("{}")
        assert (
            agent._unusable_output_condition(_text_response("{}"))
            == ConditionKey.CALL_FRAGMENT_REPLY
        )
        # An ordinary reply is VALID for chat, and so is a short one carrying real words.
        assert not is_empty_draw("Yes.")
        assert agent._unusable_output_condition(_text_response("Lake Baikal!")) is None
        assert agent._unusable_output_condition(_text_response("Yes.")) is None


class TestChatEmptyDrawReroll:
    """Chat's EMPTY draw is discarded and re-rolled like the rest of the family (#1937).

    It was the last invalid-output class handled on the VISIBLE path: the empty
    assistant turn plus a nudge user turn went into the conversation, taking the
    model's in-flight intent with them and leaving residue in the context every later
    reader of the run sees.  Now nothing about it enters the run."""

    @pytest.mark.asyncio
    async def test_the_production_sequence_never_grows_the_conversation(self, test_db, mock_llm):
        """The observed sequence, recreated: a tool call whose arguments collapsed is
        discarded (a true positive for the punctuation guard), the re-roll on the
        unchanged context comes back EMPTY, and the third draw is a clean call.

        All three draws are served on the SAME context — the two bad ones leave no
        assistant turn and no nudge user turn behind — and the clean call executes, so
        the write the model was in the middle of is not lost."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "the new Air‑...??…?..?????"}
                )
            if count == 2:
                return mock_llm._make_text_response(request, "")
            if count == 3:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "harbour lantern price"}
                )
            return mock_llm._make_text_response(request, "It is nine credits.")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what does the lantern cost?", max_steps=max_steps)

        # Two re-rolls on one budget, all three draws against a context that never grew.
        assert len(mock_llm.requests) == 4
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[0]["messages"]
        # Nothing from either bad draw reached the context: no collapse, no empty
        # assistant turn, no nudge user turn.
        redrawn = mock_llm.requests[2]["messages"]
        assert "?????" not in str(redrawn)
        assert [message["role"] for message in redrawn].count(MessageRole.USER) == 1
        assert not any(message.get("role") == MessageRole.ASSISTANT for message in redrawn)
        # The clean draw's call ran, and the turn ends on the model's own reply.
        assert [record.tool for record in response.tool_calls] == ["search"]
        assert response.tool_calls[0].arguments["query"] == "harbour lantern price"
        assert response.answer == "It is nine credits."
        # The re-roll's only durable trace: two persisted draws on one context (#1841).
        assert draw_rerolled(db)

        await agent.close()

    @pytest.mark.asyncio
    async def test_three_unusable_draws_abort_the_run(self, test_db, mock_llm):
        """The accepted trade-off: a re-roll on the unchanged context cannot unstick a
        deterministically-empty draw, so the budget burns and the run fails honestly via
        the #1909 abort path — never an empty reply and never a nudge."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "lantern ...??…?..?????"}
                )
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what does the lantern cost?", max_steps=max_steps)

        assert response.answer == PennyResponse.AGENT_MODEL_ERROR
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        assert response.tool_calls == []
        # The abort names every discarded draw in ORDER, so a run killed by three empties
        # and one killed by three collapses read as the different problems they are.
        assert response.abort is not None
        assert response.abort.error == ModelCallError(
            error_class=REROLL_EXHAUSTED,
            message="3 unusable draws: degenerate_output, empty, empty",
        )

        await agent.close()

    @pytest.mark.asyncio
    async def test_garbage_shaped_draw_after_a_tool_call_is_rerolled(self, test_db, mock_llm):
        """A draw that carries no real words — a bare markdown separator — is the same
        empty draw, so it is re-rolled rather than finalized over a real prior answer."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "lantern"})
            if count == 2:
                return mock_llm._make_text_response(request, "\n\n---")
            return mock_llm._make_text_response(request, "It is nine credits.")

        mock_llm.set_response_handler(handler)

        response = await agent.run("what does the lantern cost?", max_steps=max_steps)

        assert response.answer == "It is nine credits."
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]
        assert "---" not in str(mock_llm.requests[2]["messages"])

        await agent.close()


def _text_response(content: str) -> LlmResponse:
    return LlmResponse(message=LlmMessage(role="assistant", content=content))


def _tool_response(name: str, args: dict) -> LlmResponse:
    return LlmResponse(
        message=LlmMessage(
            role="assistant",
            content="",
            tool_calls=[
                LlmToolCall(id="c1", function=LlmToolCallFunction(name=name, arguments=args))
            ],
        )
    )


def _ctx(
    *,
    step: int = 0,
    is_final_step: bool = False,
    tools_available: bool = True,
    source_text: str = "",
    records: list[ToolCallRecord] | None = None,
    retried: set[ConditionKey] | None = None,
) -> LoopContext:
    return LoopContext(
        step=step,
        is_final_step=is_final_step,
        tools_available=tools_available,
        source_text=source_text,
        records=records if records is not None else [],
        retried=retried if retried is not None else set(),
    )


class TestResponseValidators:
    """Each validator owns one condition and returns its disposition — the unit
    contract behind the integration behaviour exercised above.  A new guard is a
    new validator with its own disposition, composed into an agent's chain."""

    def test_xml_validator_retries_then_proceeds_once_retried(self):
        resp = _text_response("<function=search>x</function>")
        outcome = XmlTagValidator().check(resp, _ctx())
        assert isinstance(outcome, Retry) and outcome.condition == ConditionKey.XML
        # Already retried → proceeds (retry-once-per-condition).
        assert isinstance(XmlTagValidator().check(resp, _ctx(retried={ConditionKey.XML})), Proceed)

    def test_retry_says_nothing_back_to_the_model(self):
        """A ``Retry`` carries the condition and nothing else (#1937).

        Its teaching user-turn retired with the last two families that used one — the
        call-shaped-text draws (#1839) and the empty draw — both of which are discarded
        and re-rolled before this chain runs.  The three conditions left correct by
        SHOWING the model its own bad draw, so a nudge field with no producer would be
        a dead parameter the loop still had to apply."""
        assert "nudge" not in Retry.model_fields

    def test_refusal_validator(self):
        resp = _text_response("I'm sorry, but I can't help with that.")
        outcome = RefusalValidator().check(resp, _ctx())
        assert isinstance(outcome, Retry) and outcome.condition == ConditionKey.REFUSAL

    def test_hallucinated_url_validator_uses_source_text(self):
        resp = _text_response("See https://made-up.example/never for details.")
        # No source text → nothing to check.
        assert isinstance(HallucinatedUrlValidator().check(resp, _ctx(source_text="")), Proceed)
        # URL absent from source → retry.
        bad = HallucinatedUrlValidator().check(resp, _ctx(source_text="unrelated text"))
        assert isinstance(bad, Retry) and bad.condition == ConditionKey.HALLUCINATED_URLS
        # URL present in source → proceed.
        ok = HallucinatedUrlValidator().check(
            resp, _ctx(source_text="ref https://made-up.example/never here")
        )
        assert isinstance(ok, Proceed)

    def test_hallucinated_tool_call_repair_strips_when_no_tools(self):
        resp = _tool_response("search", {"query": "x"})
        outcome = HallucinatedToolCallRepair().check(resp, _ctx(tools_available=False))
        assert isinstance(outcome, Repair)
        assert outcome.response.message.tool_calls is None
        # Original untouched (pure validator, deep copy).
        assert resp.message.tool_calls is not None
        # Tools available → no repair.
        assert isinstance(
            HallucinatedToolCallRepair().check(resp, _ctx(tools_available=True)), Proceed
        )

    def test_is_done_json_bail(self):
        # Detects only the argless done envelope; anything else is not a done bail.
        assert is_done_json_bail('{"name": "done", "arguments": {}}') is True
        assert is_done_json_bail('{"name": "done"}') is True
        assert is_done_json_bail('{"name": "done", "arguments": {"success": false}}') is True
        # Non-JSON, bare args (no name), wrong name, and a non-dict arguments → False.
        assert is_done_json_bail("not json") is False
        assert is_done_json_bail('{"success": true, "summary": "s"}') is False
        assert is_done_json_bail('{"name": "search", "arguments": {}}') is False
        assert is_done_json_bail('{"name": "done", "arguments": "oops"}') is False

    def test_chain_composition_is_one_list_entry_per_guard(self):
        """The response-shape chain is chat's and the base's; each agent shape declares
        its own run-shape guards and its own invalid-draw family.  A new guard = one
        more list entry, never a new branch in the loop."""
        assert Agent.response_validators[0].__class__ is HallucinatedToolCallRepair
        # What is left are the three draws that SAY something wrong.  The ones that say
        # nothing usable — call-shaped text (#1839) and the empty draw (#1937) — are
        # discarded and re-rolled upstream, so ``EmptyResponseValidator`` is GONE.
        chat_conditions = {
            XmlTagValidator,
            RefusalValidator,
            HallucinatedUrlValidator,
            HallucinatedToolCallRepair,
        }
        assert {v.__class__ for v in Agent.response_validators} == chat_conditions
        # The collector composes NO chain of its own (#1839): every call-less draw of
        # its shape is discarded upstream, so there is nothing left for a response-shape
        # guard to say and the collector-flavoured empty nudge retired with it.
        assert "response_validators" not in vars(BackgroundAgent)
        # THE WATCHED DELETION (#1911): the collector's one run-shape guard was the
        # premature-``done()`` refusal, and it retired with the tool it guarded — a
        # first-move close is UNAVAILABLE now rather than refused.
        assert BackgroundAgent.run_shape_validators == []
        # Base agent has no run-shape guards either, and declares no invalid draws (any
        # text is a legitimate answer there).
        assert Agent.run_shape_validators == []
        assert Agent.invalid_draw_conditions == ()
        # Chat's run-shape chain is the three narrate-from-the-RECORD nudges — what the run
        # LEARNED (#1658), what it SET RUNNING (#1869), and what it WROTE (#1946) — and its
        # invalid draws are the ones that are not a reply, so a plain conversational reply
        # stays valid.  The writes frame is LAST because it is the corrective one: the two
        # above ask for an account of the round, and it says which of that the store holds.
        assert [v.__class__ for v in ChatAgent.run_shape_validators] == [
            SkillNarrationValidator,
            AppliedConfigurationValidator,
            WritesLandedValidator,
        ]
        assert [condition for condition, _ in ChatAgent.invalid_draw_conditions] == [
            ConditionKey.CALL_AS_TEXT,
            ConditionKey.CALL_FRAGMENT_REPLY,
            ConditionKey.EMPTY,
        ]
        assert [condition for condition, _ in BackgroundAgent.invalid_draw_conditions] == [
            ConditionKey.DONE_JSON_BAIL,
            ConditionKey.TEXT_INSTEAD_OF_TOOL,
        ]
        # An invalid draw carries NO nudge (#1839/#1937) — it is rejected, not taught —
        # so the constants that used to be appended are gone with the validators that
        # appended them.  The last two are the empty draw's mid-loop and final-step
        # appends, retired with ``EmptyResponseValidator``.
        for retired in (
            "TOOL_FORMAT_NUDGE",
            "COLLECTOR_TOOL_CALL_NUDGE",
            "COLLECTOR_DONE_JSON_NUDGE",
            "CHAT_CALL_AS_TEXT_NUDGE",
            "COLLECTOR_CONTINUE_NUDGE",
            "CONTINUE_NUDGE",
            "FINAL_STEP_NUDGE",
        ):
            assert not hasattr(Prompt, retired), retired

    def test_run_validators_threads_repair_then_short_circuits(self):
        """A Repair threads its transformed response into the rest of the chain;
        the first non-proceed short-circuits and is returned."""
        resp = _tool_response("search", {"query": "x"})
        resp.message.content = "<function=search>x</function>"
        # Tools unavailable → repair strips the tool calls, then the XML content retries.
        outcome = run_validators(Agent.response_validators, resp, _ctx(tools_available=False))
        assert isinstance(outcome, Retry) and outcome.condition == ConditionKey.XML
        # Nothing left in the chain answers an EMPTY draw (#1937): the repaired,
        # contentless response simply proceeds, and the loop's own close reports it.
        stripped = run_validators(
            Agent.response_validators,
            _tool_response("search", {"query": "x"}),
            _ctx(tools_available=False),
        )
        assert isinstance(stripped, Proceed)
        assert stripped.response is not None and stripped.response.message.tool_calls is None
