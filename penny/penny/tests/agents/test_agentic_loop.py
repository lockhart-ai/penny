"""Tests for agentic loop changes: reasoning, last step, and after_step hook."""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from penny.agents.base import Agent, BackgroundAgent, _any_text
from penny.agents.chat import ChatAgent
from penny.agents.models import (
    BUDGET_EXCEEDED,
    REROLL_EXHAUSTED,
    MessageRole,
    ModelCallError,
    RunAbort,
    ToolCallRecord,
)
from penny.config import Config
from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PageReading, PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm import LlmClient
from penny.llm.call_link import CallLink
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
from penny.tools.browse import BrowseTool, _parse_search_results, _render_search_list
from penny.tools.link_handles import HandleTable
from penny.tools.models import BrowseArgs, BrowseItem, ToolArgs, ToolResult
from penny.validation import (
    ConditionKey,
    LoopContext,
    Proceed,
    Repair,
    Retry,
    run_validators,
)
from penny.validation.response_validators import (
    HallucinatedToolCallRepair,
    HallucinatedUrlValidator,
    RefusalValidator,
    XmlTagValidator,
    find_hallucinated_urls,
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

    async def execute(self, call_link: CallLink, **kwargs):
        return ToolResult(message="Mock search results for testing")


def _make_agent(test_db, mock_llm, *, max_steps=3, runtime_overrides=None, agent_cls=Agent):
    """Create a minimal Agent for loop testing.

    Returns (agent, db, max_steps) — max_steps must be passed to agent.run().
    Pass runtime_overrides={key: value} to override runtime config params for the test.
    Pass agent_cls to build a different polarity (e.g. ``_ChatRerollAgent`` below) — the
    class, not an instance, since all of this method's own setup still applies.
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
    agent = agent_cls(
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


class _ChatRerollAgent(ChatAgent):
    """A real ``ChatAgent``, built on ``_make_agent``'s plain test setup (#2790) —
    so a loop test exercises ``ChatAgent``'s own invalid-draw shapes AND notes, not
    a copy of them pasted onto a lighter base.  (A plain instance attribute can't
    stand in for ``_retry_note_for_condition``: its zero-arg ``super()`` call binds
    to the class it was textually defined on, which only a real subclass is.)"""


class TestReasoningStripped:
    """#2813 dropped the per-tool ``reasoning`` argument the framework used to
    inject — a model may still send a stray one out of habit, and the loop
    tolerates it (pops it before arg validation) without depending on its value:
    it is never stored or threaded anywhere, just discarded."""

    @pytest.mark.asyncio
    async def test_stray_reasoning_popped_and_discarded(self, test_db, mock_llm):
        """A tool call that still carries ``reasoning`` runs normally, and the key
        never reaches the tool's arguments or any stored record."""
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

        response = await agent.run(
            "what's the weather?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == "here's the weather!"
        assert len(response.tool_calls) == 1
        # reasoning should NOT be in the arguments dict
        assert "reasoning" not in response.tool_calls[0].arguments
        # This run ended on a text reply, so every tool result already rode into a
        # later call and was written down there — nothing trails (#1778).  The record
        # of an ordinary run is byte-identical to before the tail existed.
        assert all(row.trailing_messages is None for row in db.messages.recent_prompts(limit=10))

        await agent.close()

    @pytest.mark.asyncio
    async def test_call_without_reasoning_runs_the_same(self, test_db, mock_llm):
        """A tool call with no ``reasoning`` key at all — the ordinary case now
        that the framework no longer injects one — runs exactly the same."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "weather"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {"query": "weather"}

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
        carried them in.

        A background shape (#2370): a narrating agent hitting ``max_steps`` right after
        a tool call no longer has "no next call" — the one tools-off draw IS a next
        call, and its own request already carries the pair forward as ordinary history
        rather than as a stamped tail.  A background cycle skips that draw outright, so
        this scenario — truly no next call — still only happens there."""
        agent, db, max_steps = _make_background_agent(test_db, max_steps=1)

        mock_llm.set_response_handler(
            lambda request, _count: mock_llm._make_tool_call_response(
                request, "search", {"query": "kites"}
            )
        )

        response = await agent.run(
            "find kites", max_steps=max_steps, call_link=CallLink.top("run-tail")
        )
        assert response.ran_out_of_steps
        # No extra draw — one request only, confirming the skip (#2370).
        assert len(mock_llm.requests) == 1

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
            await agent.run("find kites", max_steps=max_steps, call_link=CallLink.top("run-died"))

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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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
        what remains — so the loop's close hands that fact to one more tools-off draw
        (#2370) rather than appending a user turn ordering the model to answer.  The
        mock keeps hallucinating a tool call on every further request, so that draw is
        unusable too, and the run closes on the deterministic account built from its
        one real tool call."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            # Final step (and the narrate draw after it): hallucinates a tool call
            # despite no tools offered, same as production might.
            return mock_llm._make_tool_call_response(request, "search", {"query": "more"})

        mock_llm.set_response_handler(handler)

        response = await agent.run("test query", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == (
            f"{PennyResponse.FACTS_EMPTY_WITH_TOOLS.format(count=1)} "
            f"{PennyResponse.ACCOUNT_TRIED_PREFIX}search."
        )
        # One extra call: the narrate draw after the final step came back empty (#2370).
        assert len(mock_llm.requests) == 3
        # The final step was offered no tools, and no nudge turn was appended for it.
        assert mock_llm.requests[1]["tools"] is None
        assert [m["content"] for m in mock_llm.requests[1]["messages"] if m["role"] == "user"] == [
            "test query"
        ]
        # The narrate draw is tools-off too.
        assert mock_llm.requests[2]["tools"] is None
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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


class _EchoToolCallIdArgs(ToolArgs):
    """``tag`` is unused by ``execute`` — it only exists so two parallel calls to
    this tool carry distinct arguments and so aren't blocked as duplicates by the
    agent loop's repeat-call guard (keyed on tool name + arguments)."""

    tag: str


class EchoToolCallIdTool(Tool):
    """A tool whose result IS the dispatched call's own id, read off its
    ``call_link`` (#2458) — lets a test observe the dispatcher's
    ``CallLink.for_tool_call`` directly, rather than inferring it from a nested
    model call's logged row."""

    name = "echo_tool_call_id"
    description = "Echo this dispatched call's own id"
    parameters = {"type": "object", "properties": {"tag": {"type": "string"}}}
    args_model = _EchoToolCallIdArgs

    async def execute(self, call_link: CallLink, **kwargs):
        return ToolResult(message=call_link.tool_call_id or "<none>")


class TestParallelToolCallsEachGetTheirOwnLink:
    """The dispatcher builds each dispatched call its OWN ``CallLink`` (#2458,
    ``call_link.for_tool_call(tool_call_id)``) — so N parallel tool calls in one
    step each see their OWN id, never a sibling's and never ``None``, with no
    ambient state or ``Task``-copy semantics required for the isolation."""

    @pytest.mark.asyncio
    async def test_two_parallel_calls_each_see_their_own_id(self, test_db, mock_llm):
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
            runtime=RuntimeParams(db=db),
        )
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
            tools=[EchoToolCallIdTool()],
            db=db,
            config=config,
        )
        agent._keep_tools_on_final_step = False

        def handler(request, count):
            if count == 1:
                return mock_llm._make_parallel_tool_calls_response(
                    request,
                    [
                        ("echo_tool_call_id", {"tag": "first"}),
                        ("echo_tool_call_id", {"tag": "second"}),
                    ],
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=3, call_link=CallLink.new_top())

        assert response.answer == "done"
        assert len(response.tool_calls) == 2
        results = {record.result for record in response.tool_calls}
        # Each call's own (provider-assigned) id, and no result is a sibling's or
        # ``None`` — each dispatched call got its own ``CallLink``.
        assert results == {"call_0", "call_1"}

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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        async def execute(tool_call, call_link=None):
            if tool_call.tool == "collection_configure":
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
                    request, "collection_configure", {"collection": "widget-log"}
                )
            if count == 3:
                # Byte-identical to count 1 — normally blocked, but the create mutated
                # since the failure, clearing the seen-calls cache, so the retry runs.
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        async def execute(tool_call, call_link=None):
            if tool_call.tool == "collection_write":
                state["written"] = True
                return ToolResult(message="Wrote entry to widget-log", mutated=True)
            if tool_call.tool == "collection_list":
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
                    request, "collection_list", dict(read_args)
                )
            if count == 2:
                return mock_llm._make_tool_call_response(
                    request, "collection_write", dict(write_args)
                )
            if count == 3:
                # Byte-identical to count 1 — the write mutated since, clearing the cache,
                # so the re-read RUNS instead of being blocked.
                return mock_llm._make_tool_call_response(
                    request, "collection_list", dict(read_args)
                )
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == "done"

        reads = [r for r in response.tool_calls if r.tool == "collection_list"]
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

        async def execute(tool_call, call_link=None):
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        async def execute(tool_call, call_link=None):
            if tool_call.tool == "collection_configure":
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
                    request, "collection_configure", {"collection": "widget-log"}
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
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

        await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())

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

        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                raise LlmToolParseError("error parsing tool call: raw='Let me reason first...'")
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "vitamins"})
            return mock_llm._make_text_response(request, "Here is the info about vitamins!")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "tell me about vitamins", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == "Here is the info about vitamins!"
        assert len(mock_llm.requests) == 3
        # A backend parse refusal never gets the discard mark: its own row is a
        # raised one, already distinguishable via call_outcome (#2702).
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 3
        assert rows[0].call_outcome == "raised"
        assert all(row.discard_reason is None for row in rows)

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_tool_parse_error_fails_the_run(self, test_db, mock_llm):
        """Budget exhausted → the reroll cause is REACHABLE (#2370), so the run takes
        one more tools-off draw rather than the canned error — and the mock keeps
        raising the same parse error on that draw too, so it falls to the
        deterministic account.  The abort still names the tripped condition, once per
        discarded draw (#1909), so a run killed by parse failures is distinguishable
        from one killed by collapses."""

        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            raise LlmToolParseError("error parsing tool call: raw='plain text again...'")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        # The reroll budget, plus the one narrate draw after it — also a parse error.
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS + 1
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
            response = await agent.run(
                "test prompt", max_steps=max_steps, call_link=CallLink.new_top()
            )

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
            await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())

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

        response = await agent.run("test query", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == "here is the real answer"
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        # The PERSISTED twin of that identity: the discarded draw is logged before it is
        # inspected, so the re-roll leaves two promptlog rows carrying the same context.  That
        # repeat is the only trace it leaves — the eval harness's loop-health advisory reads it
        # (#1841), so the premise is pinned here against a real re-roll, not just seeded rows.
        assert draw_rerolled(db)
        # The discarded draw's own row carries the reason its output was not
        # used (#2702); the accepted re-draw reads normally, unmarked.
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 2
        assert [row.discard_reason for row in rows if row.discard_reason] == ["degenerate_output"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_call_fragment_reply_discarded_and_rerolled(self, test_db, mock_llm):
        """A would-be final reply that is a bare JSON call fragment
        (``{"memory": "rip? wait …"}`` — the #1570 field audit's observed leak) is
        discarded and re-rolled — never SENT verbatim — and the redraw is shown the
        draft plus a note naming what was wrong (#2790): a model-choice shape, not a
        transport artifact, so it is not blind.

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

        # The ChatAgent polarity, built on the test agent, with its real notes.
        agent, _db, max_steps = _make_agent(
            test_db, mock_llm, max_steps=3, agent_cls=_ChatRerollAgent
        )
        draft = '{"memory": "rip? wait we need"}'

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, draft)
            return mock_llm._make_text_response(request, "here is the real answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == "here is the real answer"
        # Exactly one reroll, shown the draft plus the note (not blind, #2790) — the
        # fragment itself never becomes the redraw's OWN content.
        assert len(mock_llm.requests) == 2
        redraw = mock_llm.requests[1]["messages"]
        assert redraw[:-2] == mock_llm.requests[0]["messages"]
        draft_turn, note_turn = redraw[-2], redraw[-1]
        assert draft_turn["role"] == "assistant" and draft_turn["content"] == draft
        assert note_turn["role"] == "user"
        assert note_turn["content"] == Prompt.CALL_FRAGMENT_RETRY_NOTE

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

        response = await agent.run("test query", max_steps=max_steps, call_link=CallLink.new_top())
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
        """When every reroll is still degenerate, the reachable cause (#2370) takes one
        more tools-off draw — the mock keeps returning the same collapsed tool call, so
        that draw is unusable too (poison is never acted on or stored), and the run
        closes on the deterministic account.  The abort still names the collapse as the
        condition that spent the budget (#1909)."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            return mock_llm._make_tool_call_response(request, "search", {"query": "...??…?..?????"})

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS + 1
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

        response = await agent.run(
            "review the collector run", max_steps=max_steps, call_link=CallLink.new_top()
        )
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
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_text_response(request, _HARMONY_LEAK)
            return mock_llm._make_text_response(request, "here is the real answer")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what's the deepest lake?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == "here is the real answer"
        # Exactly one reroll on the unchanged context; the leak was never appended.
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        assert "<|call|>" not in str(mock_llm.requests[-1]["messages"])
        # The leaked draw's own row carries its condition (#2702).
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 2
        assert [row.discard_reason for row in rows if row.discard_reason] == ["tool_call_leak"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_leak_aborts_run(self, test_db, mock_llm):
        """When every reroll still leaks the envelope, the reachable cause (#2370)
        takes one more tools-off draw — the mock keeps leaking on that draw too, so
        raw Harmony tokens are still never delivered and the run closes on the
        deterministic account.  The abort still names the leak rather than the
        collapse the same budget also serves (#1909)."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            return mock_llm._make_text_response(request, _HARMONY_LEAK)

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what's the deepest lake?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS + 1
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
        """When the model returns empty content, one more tools-off draw is tried
        (#2370); the mock keeps answering empty, so the run closes on the
        deterministic account."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        def handler(request, count):
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == (
            f"{PennyResponse.FACTS_EMPTY_NO_TOOLS} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        )
        # The empty draw, plus the one narrate draw after it (#2370).
        assert len(mock_llm.requests) == 2
        # Both draws came back empty and both rows say so (#2702): the final
        # answer's empty draw, and the narration draw the account answered for.
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 2
        assert all(row.discard_reason == "empty" for row in rows)

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_response_after_tool_call(self, test_db, mock_llm):
        """An empty answer after preceding tool calls still gets the one narrate draw
        (#2370) — the mock keeps answering empty, so the account names what was tried.

        The distinction matters: a run that called tools and then said nothing must
        not record as a callless one, since everything reading a finished run reads
        ``tool_calls`` (#1776) — the records travel with the close either way."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=4)

        def handler(request, count):
            if count <= 3:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": f"query {count}"}
                )
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
        facts = PennyResponse.FACTS_EMPTY_WITH_TOOLS.format(count=3)
        assert response.answer == (
            f"{facts} {PennyResponse.ACCOUNT_TRIED_PREFIX}search, search, search."
        )
        assert [record.tool for record in response.tool_calls] == ["search"] * 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_stripped_hallucinated_call_closes_honestly(self, test_db, mock_llm):
        """A final-step tool call (tools stripped) is repaired away, and what is left is
        an empty answer the run narrates honestly rather than nudges about (#2370).

        ``HallucinatedToolCallRepair`` strips the calls and the content falls through a
        chain that no longer has anything to say about emptiness (#1937), so the close
        hands that fact to one more tools-off draw instead of appending a strong-nudge
        user turn.  The mock keeps hallucinating a tool call (ignoring that this new
        draw offers no tools either), so that draw is unusable too and the run closes
        on the deterministic account."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_tool_call_response(
                request, "search", {"query": "x"}
            )
        )

        response = await agent.run("test prompt", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == (
            f"{PennyResponse.FACTS_EMPTY_NO_TOOLS} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        )
        assert len(mock_llm.requests) == 2

        await agent.close()


def _make_bare_background_agent(test_db, mock_llm, *, max_steps=3, runtime_overrides=None):
    """Same shape as ``_make_agent``, but a ``BackgroundAgent`` — the collector's own
    draw-narration behaviour (skip every ordinary last draw, #2370) lives on that
    class, not on a flag ``_make_agent``'s plain ``Agent`` could be made to carry.
    Distinct from the other ``_make_background_agent`` below (a terminating shape for
    text-nudge testing) — this one carries only a work tool, no ``done``, since the
    budget-exceeded tests never reach a tool call at all."""
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
        embedding_model_client=client,
        tools=[StubSearchTool()],
        db=db,
        config=config,
    )
    agent.name = PennyConstants.COLLECTOR_AGENT_NAME
    return agent, db, max_steps


class _RecordingLlmClient:
    """A bare ``.chat()`` stand-in that only RECORDS what it was called with.

    ``mock_llm``/``MockLlmClient`` patches the OpenAI SDK a layer BELOW the real
    ``LlmClient`` (so that client's own request-building and retry logic genuinely
    runs) — which is the right harness for most of this file, but it means the
    ``max_tokens``/``extra_body`` kwargs the real client builds never reach back up
    to anything a test can read.  This stub sits exactly where ``Agent._invoke_model``
    calls ``self._model_client.chat(...)`` — the boundary #2475 adds the budget
    at — so a test built on it proves what THAT call site sent, not what the wire
    format happens to look like three layers down (``test_llm_client.py`` already
    covers that separately)."""

    def __init__(self, responses: list[LlmResponse]) -> None:
        self.model = "test-model"
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        *,
        call_link: CallLink,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_target: str | None = None,
        token_budget: int | None = None,
        thinking_enabled: bool = True,
    ) -> LlmResponse:
        self.calls.append(
            {
                "tools": tools,
                "token_budget": token_budget,
                "thinking_enabled": thinking_enabled,
            }
        )
        return self._responses.pop(0)


def _cut_draw() -> LlmResponse:
    """A draw cut by its own token budget — no tool call, no content, the shape
    both probed backends (#2474) came back with."""
    return LlmResponse(
        message=LlmMessage(role="assistant", content=""),
        model="test-model",
        finish_reason=PennyConstants.FINISH_REASON_LENGTH,
    )


class TestBudgetExceeded:
    """A draw cut by its own token budget (#2475, ``finish_reason ==
    FINISH_REASON_LENGTH``) is never handed to the discard-and-reroll machinery that
    every other unusable shape goes through — re-rolling on the SAME budget and
    thinking settings would almost certainly cut the next draw too.  It raises
    straight to the #2370 last-draw chokepoint instead, over one more draw with
    tools and thinking both off."""

    @pytest.mark.asyncio
    async def test_chat_shaped_agent_answers_with_the_last_draws_own_text(self, test_db, mock_llm):
        """The reply IS the thinking-off last draw's own text — not the canned
        ``AGENT_MODEL_ERROR`` (the model was reachable) and not a re-roll of the cut
        draw first (exactly two calls total)."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions
        agent.name = PennyConstants.CHAT_AGENT_NAME

        def handler(request, count):
            if count == 1:
                return _cut_draw()
            return mock_llm._make_text_response(
                request, "Here's what I had before I ran out of room to think."
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "tell me everything", max_steps=max_steps, call_link=CallLink.new_top()
        )

        assert response.answer == "Here's what I had before I ran out of room to think."
        # The cut draw, plus the ONE last draw — no reroll spent re-trying the same cap.
        assert len(mock_llm.requests) == 2
        assert response.abort == RunAbort(
            step=1,
            after_tool=None,
            error=ModelCallError(
                error_class=BUDGET_EXCEEDED,
                message="the draw ran past its token budget before reaching an answer",
            ),
        )
        # The cut draw's own row says it was set aside for its budget (#2702);
        # the thinking-off last draw reads normally, unmarked.
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 2
        assert [row.discard_reason for row in rows if row.discard_reason] == ["budget_cut"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_background_shaped_agent_draws_once_despite_ordinarily_skipping(
        self, test_db, mock_llm
    ):
        """A ``BackgroundAgent`` (the collector's shape) skips every ORDINARY last
        draw (#2370) — but a budget cut is the one shape it still draws for, per the
        code owner's ruling that every call kind ends in a live summary when the
        model can be reached."""
        agent, _db, max_steps = _make_bare_background_agent(test_db, mock_llm, max_steps=3)

        def handler(request, count):
            if count == 1:
                return _cut_draw()
            return mock_llm._make_text_response(
                request, "ran long — here's where the cycle had got to"
            )

        mock_llm.set_response_handler(handler)

        response = await agent.run("go", max_steps=max_steps, call_link=CallLink.new_top())

        assert response.answer == "ran long — here's where the cycle had got to"
        assert len(mock_llm.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_chat_and_collector_calls_each_send_their_own_budget(self, test_db):
        """Every draw a chat-shaped agent makes — the cut one AND the thinking-off
        last draw — carries ``token_budget_for(self.config.runtime, self.name)``'s
        own value; only the last draw turns thinking off and strips tools (#2474/#2475)."""
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
            runtime=RuntimeParams(db=db),
        )
        client = _RecordingLlmClient([_cut_draw(), _text_response("all set")])
        agent = Agent(
            system_prompt="test",
            model_client=cast(Any, client),
            embedding_model_client=cast(Any, client),
            tools=[StubSearchTool()],
            db=db,
            config=config,
        )
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions
        agent.name = PennyConstants.CHAT_AGENT_NAME
        agent._keep_tools_on_final_step = False

        response = await agent.run("hi", max_steps=3, call_link=CallLink.new_top())

        assert response.answer == "all set"
        chat_budget = token_budget_for(RuntimeParams(db=db), PennyConstants.CHAT_AGENT_NAME)
        assert client.calls[0]["token_budget"] == chat_budget
        assert client.calls[0]["thinking_enabled"] is True
        assert client.calls[0]["tools"]  # the ordinary step offers its tools
        assert client.calls[1]["token_budget"] == chat_budget
        assert client.calls[1]["thinking_enabled"] is False
        assert client.calls[1]["tools"] is None  # the last draw is tools-off (#2370)

        await agent.close()

        collector_client = _RecordingLlmClient([_cut_draw(), _text_response("summary")])
        background_agent = BackgroundAgent(
            system_prompt="test",
            model_client=cast(Any, collector_client),
            embedding_model_client=cast(Any, collector_client),
            db=db,
            config=config,
        )
        background_agent.name = PennyConstants.COLLECTOR_AGENT_NAME

        response = await background_agent.run("go", max_steps=3, call_link=CallLink.new_top())

        assert response.answer == "summary"
        collector_budget = token_budget_for(
            RuntimeParams(db=db), PennyConstants.COLLECTOR_AGENT_NAME
        )
        assert collector_client.calls[0]["token_budget"] == collector_budget
        assert collector_client.calls[0]["thinking_enabled"] is True
        assert collector_client.calls[1]["token_budget"] == collector_budget
        assert collector_client.calls[1]["thinking_enabled"] is False

        await background_agent.close()


class TestThinkTagStripping:
    """Test that <think>...</think> blocks are stripped from final responses."""

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_content(self, test_db, mock_llm):
        """<think>...</think> blocks in content are removed before sending to user."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "<think>Internal reasoning here.</think>\nHere is the real answer."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.thinking == "Step-by-step plan."
        assert response.answer == "Final response."

        await agent.close()

    def test_a_steps_thinking_rides_into_the_next_request(self):
        """A step is replayed to the model with its thinking, so the step after its
        tool result is drawn with the plan in view; a step with none adds no field."""
        call = LlmToolCall(
            id="call_1",
            function=LlmToolCallFunction(name="browse", arguments={"queries": ["x"]}),
        )
        planned = LlmMessage(
            role="assistant", tool_calls=[call], thinking="read the page, then save it"
        )
        assert planned.to_log_dict()["reasoning"] == "read the page, then save it"
        assert "reasoning" not in LlmMessage(role="assistant", content="done").to_log_dict()

    @pytest.mark.asyncio
    async def test_response_without_think_tags_unchanged(self, test_db, mock_llm):
        """Responses that contain no <think> tags are returned as-is."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        mock_llm.set_response_handler(
            lambda req, count: mock_llm._make_text_response(req, "Normal answer.")
        )

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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
                return mock_llm._make_tool_call_response(request, "search", {"query": "first"})
            if count == 2:
                return mock_llm._make_tool_call_response(request, "search", {"query": "second"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == "done"

        # Two steps with tool calls → two after_step calls
        assert len(captured_step_records) == 2
        assert len(captured_step_records[0]) == 1
        assert captured_step_records[0][0].arguments["query"] == "first"
        assert len(captured_step_records[1]) == 1
        assert captured_step_records[1][0].arguments["query"] == "second"

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

        await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())

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

        response = await agent.run(
            "test question", max_steps=max_steps, call_link=CallLink.new_top()
        )
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
            side_effect=lambda tool_call, call_link=None: ToolResult(
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

        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())

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
                    request,
                    "browse",
                    {
                        "items": [
                            {"query": "https://a.com", "looking_for": "the page content"},
                            {"query": "https://b.com", "looking_for": "the page content"},
                        ]
                    },
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
        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
        assert response.answer == "done"
        await agent.close()

    @pytest.mark.asyncio
    async def test_text_queries_route_to_search_url_when_browser_connected(self, test_db, mock_llm):
        """When a browser is connected, text queries become search URLs via BrowseTool."""
        browsed_urls: dict[str, str] = {}

        async def fake_request(command, params):
            url = params["url"]
            browsed_urls[url] = f"Results for {url}"
            return (browsed_urls[url], None, PageReading.ARTICLE)

        request_fn = AsyncMock(side_effect=fake_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(
            max_calls=5,
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(
            CallLink.new_top(),
            items=[{"query": "best pizza new york", "looking_for": "the page content"}],
        )

        assert len(browsed_urls) == 1
        search_url = list(browsed_urls.keys())[0]
        assert search_url.startswith("https://duckduckgo.com/?q=")
        assert "best%20pizza%20new%20york" in search_url

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
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )

        result = await tool.execute(
            CallLink.new_top(),
            items=[
                {"query": "best pizza new york", "looking_for": "the page content"},
                {"query": "https://example.test/a", "looking_for": "the page content"},
                {"query": "https://example.test/b", "looking_for": "the page content"},
            ],
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

        async def failing_then_working(tool_call, call_link=None):
            attempts.append(tool_call.arguments["items"][0]["query"])
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
                    request,
                    "browse",
                    {"items": [{"query": f"q{count}", "looking_for": "the page content"}]},
                )
            return mock_llm._make_text_response(request, "here's what the page said")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "look this up", max_steps=max_steps, call_link=CallLink.new_top()
        )

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
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )

        result = await tool.run(CallLink.new_top(), items=[])

        assert result.success is False
        assert "items" in result.message
        assert "at least one item" in result.message

        # Each item's looking_for is REQUIRED (#2718, carrying #1570's rule per
        # item — every browse routes page content through a micro-context).  A
        # missing or blank looking_for is rejected at the same gate with the fix.
        missing = await tool.run(CallLink.new_top(), items=[{"query": "https://x.test"}])
        assert missing.success is False and "looking_for" in missing.message
        blank = await tool.run(
            CallLink.new_top(), items=[{"query": "https://x.test", "looking_for": "   "}]
        )
        assert blank.success is False and "looking_for" in blank.message
        assert BrowseArgs(items=[BrowseItem(query="q", looking_for="the price")])
        items = [
            BrowseItem(query="https://a.test", looking_for="one"),
            BrowseItem(query="https://b.test", looking_for="two"),
        ]
        assert [item.looking_for for item in BrowseArgs(items=items).items] == ["one", "two"]

    @pytest.mark.asyncio
    async def test_urls_always_route_to_browse(self, test_db, mock_llm):
        """URLs always go to BrowseTool regardless of browser connection."""
        browsed_urls: list[str] = []

        async def fake_request(command, params):
            browsed_urls.append(params["url"])
            return (f"Page content from {params['url']}", None, PageReading.ARTICLE)

        request_fn = AsyncMock(side_effect=fake_request)
        mock_perm = MagicMock(check_domain=AsyncMock())

        tool = BrowseTool(
            max_calls=5,
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        await tool.execute(
            CallLink.new_top(),
            items=[
                {"query": "https://example.com/page", "looking_for": "the page content"},
                {"query": "https://other.com", "looking_for": "the page content"},
            ],
        )

        assert len(browsed_urls) == 2
        assert "https://example.com/page" in browsed_urls
        assert "https://other.com" in browsed_urls

    @pytest.mark.asyncio
    async def test_url_timeout_returns_error_section(self, test_db, monkeypatch):
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
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            model_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )
        tool.set_browse_provider(lambda: (request_fn, mock_perm))

        result = await tool.execute(
            CallLink.new_top(),
            items=[{"query": "https://slow.example.com", "looking_for": "the page content"}],
        )

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert PennyConstants.BROWSE_ERROR_HEADER in result.message
        # The error header shows the address as the model gave it (#2735) — no
        # number is spent on a page that was never read.
        assert f"{PennyConstants.BROWSE_ERROR_HEADER}https://slow.example.com" in result.message

    def test_browse_tool_timeout_exceeds_request_timeout(self, test_db):
        """BrowseTool.timeout must exceed the per-URL BROWSE_REQUEST_TIMEOUT.

        Ensures the inner per-URL timeout fires before the outer executor
        timeout, so hung URLs produce graceful error sections instead of
        cancelling the entire tool call.
        """
        tool = BrowseTool(
            max_calls=3,
            db=Database(test_db),
            embedding_client=cast(Any, MockLlmClient()),
            handles=HandleTable(),
        )
        assert tool.timeout is not None
        assert tool.timeout > PennyConstants.BROWSE_REQUEST_TIMEOUT


class TestSearchResultParsing:
    """Tests for _parse_search_results: parses a search page to titles + URLs
    (#2647, the picker's input — #2718)."""

    def test_lists_titles_and_urls(self):
        """Each result is its title and the link's own target; nothing else from the
        page — no snippets, no preamble."""
        content = "\n".join(
            [
                "Lots of preamble text here",
                "### NASA Article",
                "[nasa.gov/artemis](https://www.nasa.gov/artemis/)",
                "Some snippet text",
                "More snippet",
                "",
                "### Space.com Article",
                "[space.com/artemis](https://www.space.com/artemis)",
                "Another snippet",
            ]
        )
        assert _parse_search_results(content, "duckduckgo.com") == [
            ("NASA Article", "https://www.nasa.gov/artemis/"),
            ("Space.com Article", "https://www.space.com/artemis"),
        ]

    def test_title_is_taken_across_blank_separators(self):
        """A blank line between a result's title and its link is skipped; the title
        still names the result."""
        content = "\n".join(
            [
                "### The Watch Page",
                "",
                "[watch.example.com](https://watch.example.com/)",
                "Snippet text",
            ]
        )
        assert _parse_search_results(content, "duckduckgo.com") == [
            ("The Watch Page", "https://watch.example.com/")
        ]

    def test_returns_nothing_when_no_links(self):
        """Content with no markdown links yields no results — the "nothing found"
        path, and the picker has nothing to rank."""
        content = "Just plain text\nwith no links\nat all"
        assert _parse_search_results(content, "duckduckgo.com") == []

    def test_strips_knowledge_panel_prose_with_inline_links(self):
        """Wikipedia-style prose with inline links yields only the heading-backed
        results."""
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
        assert _parse_search_results(content, "duckduckgo.com") == [
            ("Show Title", "https://en.example.org/show"),
            ("Show - IMDb", "https://imdb.com/title/tt1"),
        ]

    def test_only_heading_backed_links_are_results(self):
        """A Kagi-shaped page — a knowledge-panel paragraph, a bare link, three
        infobox rows each followed by a link, then ordinary heading+link results —
        yields only the ordinary results (#2697).  The panel prose and infobox
        rows never become titles the picker ranks."""
        content = "\n".join(
            [
                "London is the capital and largest city of both England and the United",
                "Kingdom, standing on the River Thames in south-east England at its",
                "navigable estuary.",
                "",
                "[london.wikilike.org](https://london.wikilike.org/wiki/London)",
                "",
                "Website",
                "[cityoflondon.gov.example](https://www.cityoflondon.gov.example/)",
                "",
                "Country",
                "[uk.wikilike.org](https://uk.wikilike.org/wiki/United_Kingdom)",
                "",
                "• Type",
                "[citystatus.wikilike.org](https://citystatus.wikilike.org/wiki/City_status)",
                "",
                "### London - Wikilike",
                "[london.wikilike.org](https://london.wikilike.org/wiki/London)",
                "",
                "### Visit London — official guide",
                "[visitlondon.example.com](https://www.visitlondon.example.com/)",
            ]
        )
        assert _parse_search_results(content, "kagi.com") == [
            ("London - Wikilike", "https://london.wikilike.org/wiki/London"),
            ("Visit London — official guide", "https://www.visitlondon.example.com/"),
        ]

    def test_links_without_headings_list_by_link_text(self):
        """A page whose links all sit under non-heading lines — links but no
        headings at all — yields those links, each titled by its own link text
        (#2697): the picker ranks link titles, never raw page prose."""
        content = "\n".join(
            [
                "London is the capital of England.",
                "",
                "[london.wikilike.org](https://london.wikilike.org/wiki/London)",
                "",
                "Website",
                "[cityoflondon.gov.example](https://www.cityoflondon.gov.example/)",
            ]
        )
        assert _parse_search_results(content, "kagi.com") == [
            ("london.wikilike.org", "https://london.wikilike.org/wiki/London"),
            ("cityoflondon.gov.example", "https://www.cityoflondon.gov.example/"),
        ]

    def test_engine_affordance_links_are_not_results(self):
        """A standalone link back at the search engine's own host — its summariser,
        its maps — is the engine's affordance, not a result, and is dropped."""
        content = "\n".join(
            [
                "### First Result",
                "[example.com/a](https://example.com/a)",
                "Snippet one",
                "### Map Block",
                "[See Map](https://kagi.com/maps/search?q=watches)",
            ]
        )
        assert _parse_search_results(content, "kagi.com") == [
            ("First Result", "https://example.com/a")
        ]
        # The same link is a result when it does not point at the search host.
        assert _parse_search_results(content, "duckduckgo.com") == [
            ("First Result", "https://example.com/a"),
            ("Map Block", "https://kagi.com/maps/search?q=watches"),
        ]

    def test_caps_at_max_search_links(self):
        """Only the first PennyConstants.MAX_SEARCH_LINKS standalone links are parsed."""
        lines: list[str] = []
        for i in range(PennyConstants.MAX_SEARCH_LINKS + 5):
            lines.append(f"### Result {i}")
            lines.append(f"[example.com/page{i}](https://example.com/page{i})")
            lines.append(f"Snippet for result {i}")
        content = "\n".join(lines)
        results = _parse_search_results(content, "duckduckgo.com")
        assert len(results) == PennyConstants.MAX_SEARCH_LINKS
        assert f"page{PennyConstants.MAX_SEARCH_LINKS}" not in [url for _, url in results]

    def test_escape_render_lists_titles_with_handles(self):
        """The picker's escape render (#2718) is the trimmed list under the
        instruction header, each link its handle (#2717) — the one shape where
        the model still sees titles and numbers."""
        content = "\n".join(
            [
                "### NASA Article",
                "[nasa.gov/artemis](https://www.nasa.gov/artemis/)",
                "Some snippet text",
                "### Space.com Article",
                "[space.com/artemis](https://www.space.com/artemis)",
            ]
        )
        results = _parse_search_results(content, "duckduckgo.com")
        rendered = _render_search_list("artemis mission", results, HandleTable())
        assert 'You searched for "artemis mission"' in rendered
        assert "titles and link numbers only" in rendered
        assert "1. NASA Article [1]" in rendered
        assert "2. Space.com Article [2]" in rendered
        assert "https://www.nasa.gov" not in rendered
        assert "Some snippet text" not in rendered


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
            response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())

        assert response.answer == "done"
        await agent.close()


# One draft per condition the response chain retries: markup, a refusal, and a URL that
# appears nowhere in what the model was shown.
_RETRIED_DRAFTS = {
    ConditionKey.XML: "<answer>The deck is $499.</answer>",
    ConditionKey.REFUSAL: "I'm sorry, but I can't help with that.",
    ConditionKey.HALLUCINATED_URLS: "It's $499, see https://made-up.example/deck for details.",
}
_CLEAN_ANSWER = "The deck is $499."


class TestValidatorRetry:
    """A response validator's ``Retry`` discards the draft and re-draws on the UNCHANGED
    conversation (#2160).  The second call reads exactly what the first did, so the model
    is never shown its own finished reply with nothing after it to answer; the retry budget
    stays one retry per condition."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", list(_RETRIED_DRAFTS))
    async def test_retry_redraws_the_unchanged_conversation_and_ships_the_redraw(
        self, test_db, mock_llm, condition
    ):
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)
        draft = _RETRIED_DRAFTS[condition]

        def handler(request, count):
            return mock_llm._make_text_response(request, draft if count == 1 else _CLEAN_ANSWER)

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what does the deck cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == _CLEAN_ANSWER
        assert len(mock_llm.requests) == 2
        # No appended draft: the re-draw carries byte-identical messages.
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        assert draft not in str(mock_llm.requests[1]["messages"])
        # So the persisted ledger reads it the way it reads any re-roll: the same context, twice.
        assert draw_rerolled(db)
        # The rejected draft's own row carries the condition it was rejected
        # for (#2702); the accepted re-draw carries no mark.
        with Session(db.engine) as session:
            rows = list(session.exec(select(PromptLog)).all())
        assert len(rows) == 2
        assert [row.discard_reason for row in rows if row.discard_reason] == [condition.value]

        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", list(_RETRIED_DRAFTS))
    async def test_a_redraw_failing_the_same_condition_ships(self, test_db, mock_llm, condition):
        """The condition is spent after one retry, so a second draft with the same defect
        is the answer — two calls, the second on the unchanged conversation.  For
        HALLUCINATED_URLS the "answer" is the REPAIRED draft (#2721): the retry is
        spent, so the still-unsourced link is dropped and a notice appended, rather
        than shipping the bad link unchanged."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        draft = _RETRIED_DRAFTS[condition]

        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, draft))

        response = await agent.run(
            "what does the deck cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        if condition == ConditionKey.HALLUCINATED_URLS:
            assert response.answer == (
                f"It's $499, see  for details.\n\n{PennyResponse.LINK_DROPPED_NOTICE}"
            )
        else:
            assert response.answer == draft
        assert len(mock_llm.requests) == 2
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_each_condition_retries_once_all_on_the_same_conversation(
        self, test_db, mock_llm
    ):
        """Three different conditions in a row each spend their own retry, and every one of
        the four calls reads the same conversation."""
        agent, _db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)
        drafts = [*_RETRIED_DRAFTS.values(), _CLEAN_ANSWER]

        def handler(request, count):
            return mock_llm._make_text_response(request, drafts[count - 1])

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what does the deck cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == _CLEAN_ANSWER
        assert len(mock_llm.requests) == 4
        first = mock_llm.requests[0]["messages"]
        assert all(request["messages"] == first for request in mock_llm.requests)

        await agent.close()

    @pytest.mark.asyncio
    async def test_refusal_after_a_tool_step_redraws_that_steps_conversation(
        self, test_db, mock_llm
    ):
        """A retry mid-run re-draws on the conversation its step was drawn from, tool
        result included."""
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

        response = await agent.run(
            "Give me a list of vegan smoothie recipes",
            max_steps=max_steps,
            call_link=CallLink.new_top(),
        )
        assert response.answer == "Here are the vegan smoothie recipes!"
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]

        await agent.close()

    @pytest.mark.asyncio
    async def test_normal_response_not_retried(self, test_db, mock_llm):
        """Normal responses are not mistakenly flagged as refusals."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        mock_llm.set_response_handler(
            lambda req, count: mock_llm._make_text_response(req, "Here are your recipes!")
        )

        response = await agent.run(
            "Give me vegan smoothie recipes", max_steps=max_steps, call_link=CallLink.new_top()
        )
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
            "tell me about colds",
            max_steps=max_steps,
            system_prompt=system_prompt,
            call_link=CallLink.new_top(),
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

        response = await agent.run(
            "follow up", max_steps=max_steps, history=history, call_link=CallLink.new_top()
        )

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

        response = await agent.run("question", max_steps=max_steps, call_link=CallLink.new_top())

        assert response.answer == good
        # Tool call + bad text + retry text = 3 model calls, the retry on the unchanged context
        assert len(mock_llm.requests) == 3
        assert mock_llm.requests[2]["messages"] == mock_llm.requests[1]["messages"]

        await agent.close()


class TestMalformedUrlCleaning:
    """Test that truncated or malformed URLs are stripped from final responses."""

    @pytest.mark.asyncio
    async def test_bare_truncated_url_removed(self, test_db, mock_llm):
        """Bare URL ending with a hyphen (truncated path) is removed from the response.

        None of these four addresses ever appear in the source text either, so the
        hallucinated-URL validator retries once and then, on the same draft again,
        repairs it (#2721) — the same removal `clean_malformed_urls` would have done
        downstream, plus the dropped-link notice; nothing is left for the later
        malformed-URL cleanup pass to do."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        # The cleaner reads a URL as the address alone, the same way the hallucinated-URL
        # validator does (#2159): a wrapper can neither hide a truncated address nor go
        # with it — only the address is removed.
        raw = (
            "Check this out: https://travelguide.com/destination- for details. "
            "Also 【browse: https://travelguide.com/beach-】, **https://travelguide.com/city-** "
            "and (https://travelguide.com/island-)."
        )
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run(
            "tell me about travel", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == (
            "Check this out:  for details. Also 【browse: 】, **** and ().\n\n"
            f"{PennyResponse.LINK_DROPPED_NOTICE}"
        )

        await agent.close()

    @pytest.mark.asyncio
    async def test_markdown_link_truncated_url_keeps_text(self, test_db, mock_llm):
        """Markdown link [text](bad_url) strips the URL but preserves the link text."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        raw = "Visit [Travel Guide](https://travelguide.com/destination-) for more info."
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run("travel info", max_steps=max_steps, call_link=CallLink.new_top())
        assert "https://travelguide.com/destination-" not in response.answer
        assert "Travel Guide" in response.answer

        await agent.close()

    @pytest.mark.asyncio
    async def test_unsourced_well_formed_url_dropped_with_notice(self, test_db, mock_llm):
        """A well-formed URL is still a URL this synthetic test never sources, so the
        hallucinated-URL validator retries it once and then, unchanged on the second
        draft, repairs it (#2721): every address is removed — the same shape
        `clean_malformed_urls` produces for a truncated one — with the dropped-link
        notice appended.  Well-formed is not the same as sourced: a URL that really
        was read off a page the turn browsed is left alone by both the validator and
        the cleaner (#2159's wrapped-forms sweep above covers that case)."""
        agent, db, max_steps = _make_agent(test_db, mock_llm)

        # Nor is a wrapped one: the wrapper is not read into the address (#2159).
        raw = (
            "See https://example.com/article for more, "
            "【browse: https://example.com/article】, **https://example.com/article**, "
            "[https://example.com/article](https://example.com/article) and "
            "(https://en.wikipedia.org/wiki/Lantern_(lighting))."
        )
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, raw))

        response = await agent.run(
            "article link", max_steps=max_steps, call_link=CallLink.new_top()
        )
        assert response.answer == (
            f"See  for more, 【browse: 】, ****,  and ().\n\n{PennyResponse.LINK_DROPPED_NOTICE}"
        )

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
            response = await agent.run(
                "test query", max_steps=max_steps, call_link=CallLink.new_top()
            )

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
        response = await agent.run(
            "what's the news?", max_steps=max_steps, call_link=CallLink.new_top()
        )

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
        response = await agent.run(
            "what's the news?",
            max_steps=max_steps,
            on_progress=on_progress,
            call_link=CallLink.new_top(),
        )

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

        async def alternating_executor(tool_call, call_link=None):
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
        response = await agent.run("test", max_steps=max_steps, call_link=CallLink.new_top())
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

        response = await agent.run(
            "test", max_steps=max_steps, on_tool_start=on_tool_start, call_link=CallLink.new_top()
        )
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

        response = await agent.run(
            "test", max_steps=max_steps, on_tool_start=on_tool_start, call_link=CallLink.new_top()
        )
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

        await agent.run(
            "test", max_steps=max_steps, on_tool_start=on_tool_start, call_link=CallLink.new_top()
        )
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

        response = await agent.run(
            "test", max_steps=max_steps, on_tool_start=on_tool_start, call_link=CallLink.new_top()
        )
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
        response = await agent.run(
            "test", max_steps=max_steps, on_progress=on_progress, call_link=CallLink.new_top()
        )

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
        db.messages.subscribe_prompt_logged(lambda data: callback_prompts.append(data))

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        await agent.run("test question", max_steps=max_steps, call_link=CallLink.new_top())

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

        await agent.run("first", max_steps=max_steps, call_link=CallLink.new_top())
        await agent.run("second", max_steps=max_steps, call_link=CallLink.new_top())

        with Session(db.engine) as session:
            logs = session.exec(select(PromptLog)).all()

        assert len(logs) == 2
        assert logs[0].run_id != logs[1].run_id
        assert logs[0].agent_name == logs[1].agent_name == "Agent"

        await agent.close()


class TestLiveStepMarkers:
    """Live step markers (#2439): call_started/tool_started/tool_finished are
    published from the one shared chokepoint every caller already carries
    (``LlmClient.chat()``, the tool dispatcher) — no new per-call parameter,
    just the identity already flowing through ``CallLink``."""

    @pytest.mark.asyncio
    async def test_call_started_published_once_per_actual_model_call(self, test_db, mock_llm):
        """Each real chat() invocation — including a tool-call step and the
        final text step — publishes call_started BEFORE the network request,
        one for one with the promptlog row it will go on to write."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent._tool_executor.execute = AsyncMock(return_value=ToolResult(message="result"))

        started: list = []
        db.messages.subscribe_call_started(started.append)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        call_link = CallLink.new_top()
        await agent.run("test question", max_steps=max_steps, call_link=call_link)

        with Session(db.engine) as session:
            logs = session.exec(select(PromptLog)).all()

        assert len(started) == len(logs) == 2
        assert all(event.run_id == call_link.run_id for event in started)
        assert all(event.parent_run_id is None for event in started)
        assert all(event.parent_tool_call_id is None for event in started)
        assert all(event.agent_name == "Agent" for event in started)
        assert all(event.model == "test-model" for event in started)

        await agent.close()

    @pytest.mark.asyncio
    async def test_call_started_carries_a_dispatched_calls_own_link(self, test_db, mock_llm):
        """A model call made with a POSITIONED link (inside a dispatched tool
        call, as a micro-context draw would be) carries that position's
        parent_run_id/parent_tool_call_id on call_started too — the same
        identity the eventual logged row carries, never re-derived."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=1)
        mock_llm.set_response_handler(lambda req, count: mock_llm._make_text_response(req, "done"))

        started: list = []
        db.messages.subscribe_call_started(started.append)

        outer = CallLink.new_top()
        positioned = outer.for_tool_call("tool-call-1")
        draw_link = positioned.child()
        await agent.run("draw", max_steps=max_steps, call_link=draw_link)

        assert len(started) == 1
        assert started[0].run_id == draw_link.run_id
        assert started[0].parent_run_id == outer.run_id
        assert started[0].parent_tool_call_id == "tool-call-1"

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_started_and_finished_bracket_the_dispatch(self, test_db, mock_llm):
        """A dispatched tool call publishes tool_started before execution and
        tool_finished with its outcome right after — matched by the exact
        (run_id, tool_call_id) identity, never a guess."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message="result", success=True)
        )

        started: list = []
        finished: list = []
        db.messages.subscribe_tool_started(started.append)
        db.messages.subscribe_tool_finished(finished.append)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        call_link = CallLink.new_top()
        await agent.run("test question", max_steps=max_steps, call_link=call_link)

        assert len(started) == len(finished) == 1
        assert started[0].run_id == finished[0].run_id == call_link.run_id
        assert started[0].tool_call_id == finished[0].tool_call_id
        assert started[0].tool_name == finished[0].tool_name == "search"
        assert finished[0].success is True

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_finished_reports_a_failed_call(self, test_db, mock_llm):
        """A tool call that FAILS still publishes tool_finished — success is
        the tool's own structural result, not assumed."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=2)
        agent._tool_executor.execute = AsyncMock(
            return_value=ToolResult(message="broke", success=False)
        )

        finished: list = []
        db.messages.subscribe_tool_finished(finished.append)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "test"})
            return mock_llm._make_text_response(request, "done")

        mock_llm.set_response_handler(handler)

        await agent.run("test question", max_steps=max_steps, call_link=CallLink.new_top())

        assert len(finished) == 1
        assert finished[0].success is False

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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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
    DRAW (#1839): the loop discards it and re-rolls on the shared reroll budget — now
    shown the draft plus a note naming what was wrong (#2790), never a blind re-roll,
    since the model's own choice produced it.  Neither the draft nor the note is ever
    stored history: both live only in the in-memory ``messages`` resolving this one
    call, never what the next cycle's own fresh ``messages`` is built from — no marker
    on the run's prompt rows either, so a recovered run is otherwise indistinguishable
    from one that never slipped."""

    @pytest.mark.asyncio
    async def test_prose_draw_is_discarded_and_the_redraw_proceeds(self, test_db, mock_llm):
        """Work, then prose ("Done. Summary: ...") → discarded, the redraw shown the
        prose plus a note that nothing in it ran, and a clean redraw closes the cycle
        with a real done().

        The realistic production shape: the model does real work, THEN narrates
        completion as prose instead of calling done()."""
        agent, db, max_steps = _make_background_agent(test_db)
        draft = "**Done. Summary: wrote the entry.**"

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, draft)
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps, call_link=CallLink.new_top())

        # Exactly one reroll, shown the draft plus the note — never a nudge turn, and
        # never stored (#2790).
        assert len(mock_llm.requests) == 3
        redraw = mock_llm.requests[2]["messages"]
        assert redraw[:-2] == mock_llm.requests[1]["messages"]
        draft_turn, note_turn = redraw[-2], redraw[-1]
        assert draft_turn["role"] == "assistant" and draft_turn["content"] == draft
        assert note_turn["role"] == "user"
        assert note_turn["content"] == Prompt.TEXT_INSTEAD_OF_TOOL_RETRY_NOTE
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

        response = await agent.run("", max_steps=max_steps, call_link=CallLink.new_top())

        tools_called = [record.tool for record in response.tool_calls]
        assert "search" in tools_called
        assert "stub_terminator" in tools_called

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_draw_is_the_same_invalid_draw(self, test_db, mock_llm):
        """An EMPTY draw (no text, no tool call) is a draw with no call like any
        other — ``TEXT_INSTEAD_OF_TOOL``, the collector's own state-keyed catch-all,
        not a separate empty condition — so it gets the SAME note (#2790), not the
        old collector-flavoured empty-content nudge that retired with the rest of
        the family."""
        agent, db, max_steps = _make_background_agent(test_db)

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, "")
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps, call_link=CallLink.new_top())

        assert len(mock_llm.requests) == 3
        redraw = mock_llm.requests[2]["messages"]
        assert redraw[:-2] == mock_llm.requests[1]["messages"]
        draft_turn, note_turn = redraw[-2], redraw[-1]
        assert draft_turn["role"] == "assistant" and draft_turn["content"] == ""
        assert note_turn["role"] == "user"
        assert note_turn["content"] == Prompt.TEXT_INSTEAD_OF_TOOL_RETRY_NOTE
        assert "Please provide your response" not in str(redraw)
        assert any(record.tool == "stub_terminator" for record in response.tool_calls)

        await agent.close()

    @pytest.mark.asyncio
    async def test_done_as_json_text_is_the_same_invalid_draw(self, test_db, mock_llm):
        """The argless ``done()`` composed as a JSON envelope but never routed through
        the tool channel (gpt-oss's Harmony fallback, #1569) is discarded like any
        other call-less draw — shown the draft plus a note that the run hasn't ended
        (#2790) — and the model's own real done() closes the cycle."""
        agent, db, max_steps = _make_background_agent(test_db)
        draft = '{"name": "done", "arguments": {}}'

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "inputs"})
            if count == 2:
                return mock_llm._make_text_response(request, draft)
            return mock_llm._make_tool_call_response(request, "stub_terminator", {})

        mock_llm.set_response_handler(handler)

        response = await agent.run("", max_steps=max_steps, call_link=CallLink.new_top())

        assert len(mock_llm.requests) == 3
        redraw = mock_llm.requests[2]["messages"]
        assert redraw[:-2] == mock_llm.requests[1]["messages"]
        draft_turn, note_turn = redraw[-2], redraw[-1]
        assert draft_turn["role"] == "assistant" and draft_turn["content"] == draft
        assert note_turn["role"] == "user"
        assert note_turn["content"] == Prompt.DONE_JSON_BAIL_RETRY_NOTE
        done_records = [r for r in response.tool_calls if r.tool == "stub_terminator"]
        assert len(done_records) == 1
        assert done_records[0].arguments == {}

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_call_less_draws_fail_the_run(self, test_db, mock_llm):
        """A model that never makes a call never loops forever: the shared reroll budget
        bounds it and the run fails honestly, with no done record and no salvaged text.

        Reachable (#2370), but a background cycle has no user reply to write — its
        ``_draw_narration`` override skips the draw outright, so this still closes on
        the deterministic account with NO extra call spent (``len(mock_llm.requests)``
        stays at the reroll budget, proof the skip actually fired)."""
        agent, db, max_steps = _make_background_agent(test_db, max_steps=3)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "still just talking")
        )

        response = await agent.run("", max_steps=max_steps, call_link=CallLink.new_top())

        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
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
        """Real search, then a browse call emitted as JSON *text* → discarded, and the
        redraw is shown the draft plus a note (#2790) — not blind, since this is the
        model's own choice, not a transport artifact.  The model's prose answer
        stands. This draft is a bare-args fragment with no envelope and no tool
        named in it: before #2813 the framework's ``reasoning`` key alone
        identified this as a call-as-text bail; now that signal is gone, it's
        caught instead as CALL_FRAGMENT_REPLY (any ``{"…`` that isn't the full
        envelope) — still discarded and rerolled, just under that note."""
        agent, db, max_steps = _make_agent(
            test_db, mock_llm, max_steps=3, agent_cls=_ChatRerollAgent
        )
        draft = '{"queries": ["deepest lake"], "reasoning": "read the page"}'

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "deepest lake"}
                )
            if count == 2:
                return mock_llm._make_text_response(request, draft)
            return mock_llm._make_text_response(request, "I couldn't find that app anywhere.")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what's the deepest lake?", max_steps=max_steps, call_link=CallLink.new_top()
        )

        # The JSON blob never became the reply — the redraw shows it back as a draft
        # plus the note, and the clean prose answer is what the turn finalizes.
        assert response.answer == "I couldn't find that app anywhere."
        assert len(mock_llm.requests) == 3
        redraw = mock_llm.requests[2]["messages"]
        assert redraw[:-2] == mock_llm.requests[1]["messages"]
        draft_turn, note_turn = redraw[-2], redraw[-1]
        assert draft_turn["role"] == "assistant" and draft_turn["content"] == draft
        assert note_turn["role"] == "user"
        assert note_turn["content"] == Prompt.CALL_FRAGMENT_RETRY_NOTE

        await agent.close()

    @pytest.mark.asyncio
    async def test_persistent_call_as_text_fails_the_turn(self, test_db, mock_llm):
        """Budget exhausted → reachable (#2370), so one more tools-off draw is tried —
        the mock keeps emitting the same blob, which is unusable by the same rule, so
        the turn still fails honestly rather than delivering it, now via the
        deterministic account."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(
                request, '{"name": "browse", "arguments": {"queries": ["x"]}}'
            )
        )

        response = await agent.run(
            "what's the deepest lake?", max_steps=max_steps, call_link=CallLink.new_top()
        )
        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS + 1

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

        response = await agent.run(
            "what's the deepest lake?", max_steps=max_steps, call_link=CallLink.new_top()
        )
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

        assert is_call_as_text_bail('{"name": "browse", "arguments": {"queries": ["x"]}}')
        # A lone JSON object with no call markers, an envelope with a non-string name
        # or non-dict arguments, non-JSON prose, and prose that merely mentions JSON
        # are all NOT call-as-text bails (a genuine reply must pass through).  The
        # bare-args-plus-reasoning shape used to bail here too (keyed on the
        # framework-injected ``reasoning`` field); #2813 dropped that field and the
        # signal with it — it's still caught below as CALL_FRAGMENT_REPLY instead.
        for prose in (
            '{"queries": ["x"], "reasoning": "y"}',  # bare args, no envelope (#2813)
            '{"note": "just data"}',  # not an envelope
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
    """Chat's EMPTY draw is discarded and re-rolled like the rest of the family (#1937),
    with one addition since (#2790): the redraw is shown a note saying the last draft
    came out empty — no draft to echo back, so no assistant turn, just the note.

    #1937's guard still holds, stated precisely: the empty draw's nudge used to become
    part of the CONVERSATION's own turn structure — a stored assistant+user turn the
    state machine had to treat as shape, and every later reader of the run saw. The
    #2790 note is not that: it lives only in the in-memory ``messages`` resolving this
    one call (never what the next turn's ``_build_conversation`` reads back — that
    reads persisted ``messagelog`` rows, written separately), so nothing about it is
    ever stored history, and no assistant turn ever enters at all."""

    @pytest.mark.asyncio
    async def test_the_production_sequence_never_grows_the_conversation(self, test_db, mock_llm):
        """The observed sequence, recreated: a tool call whose arguments collapsed is
        discarded (a true positive for the punctuation guard, stays blind — a transport
        artifact, not the model's choice), the re-roll on the unchanged context comes
        back EMPTY (the model's own choice — shown a note, no draft), and the third
        draw is a clean call.

        The collapse leaves no trace at all; the EMPTY draw leaves exactly one note
        turn and no assistant turn — never the draft-plus-note shape the other
        model-choice conditions get, since there is no draft worth echoing back; and
        the clean call executes, so the write the model was in the middle of is not
        lost."""
        agent, db, max_steps = _make_agent(
            test_db, mock_llm, max_steps=3, agent_cls=_ChatRerollAgent
        )

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

        response = await agent.run(
            "what does the lantern cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )

        assert len(mock_llm.requests) == 4
        # The collapse (transport, blind) leaves the context exactly as it was.
        assert mock_llm.requests[1]["messages"] == mock_llm.requests[0]["messages"]
        # The EMPTY draw (the model's own choice) is shown ONLY the note — no draft,
        # since an empty draw has nothing worth echoing back (#2790).
        redrawn = mock_llm.requests[2]["messages"]
        assert redrawn[:-1] == mock_llm.requests[1]["messages"]
        note_turn = redrawn[-1]
        assert note_turn["role"] == "user" and note_turn["content"] == Prompt.EMPTY_DRAW_RETRY_NOTE
        # Still no collapse residue, and still no ASSISTANT turn from either bad draw —
        # only the note's own user turn is new.
        assert "?????" not in str(redrawn)
        assert not any(message.get("role") == MessageRole.ASSISTANT for message in redrawn)
        # The clean draw's call ran, and the turn ends on the model's own reply.
        assert [record.tool for record in response.tool_calls] == ["search"]
        assert response.tool_calls[0].arguments["query"] == "harbour lantern price"
        assert response.answer == "It is nine credits."
        # The collapse's own durable trace: two persisted draws on one context (#1841).
        # The EMPTY draw's redraw is no longer a byte-identical re-roll (it carries the
        # note), so this reads the collapse pair alone — accepted, the same shape #2698
        # established for the dangling-link note.
        assert draw_rerolled(db)

        await agent.close()

    @pytest.mark.asyncio
    async def test_three_unusable_draws_abort_the_run(self, test_db, mock_llm):
        """The accepted trade-off: a re-roll on the unchanged context cannot unstick a
        deterministically-empty draw, so the budget burns and the run fails via the
        #1909 abort path — reachable (#2370), so one more tools-off draw is tried; the
        mock keeps answering empty, so it closes on the deterministic account, never
        an apology and never a nudge."""
        agent, db, max_steps = _make_agent(test_db, mock_llm, max_steps=3)
        agent.invalid_draw_conditions = ChatAgent.invalid_draw_conditions

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(
                    request, "search", {"query": "lantern ...??…?..?????"}
                )
            return mock_llm._make_text_response(request, "")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what does the lantern cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )

        facts = PennyResponse.FACTS_REROLLS_EXHAUSTED.format(
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        )
        assert response.answer == f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
        assert len(mock_llm.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS + 1
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
        empty draw, so it is re-rolled (shown the note-only redraw, #2790) rather than
        finalized over a real prior answer."""
        agent, db, max_steps = _make_agent(
            test_db, mock_llm, max_steps=3, agent_cls=_ChatRerollAgent
        )

        def handler(request, count):
            if count == 1:
                return mock_llm._make_tool_call_response(request, "search", {"query": "lantern"})
            if count == 2:
                return mock_llm._make_text_response(request, "\n\n---")
            return mock_llm._make_text_response(request, "It is nine credits.")

        mock_llm.set_response_handler(handler)

        response = await agent.run(
            "what does the lantern cost?", max_steps=max_steps, call_link=CallLink.new_top()
        )

        assert response.answer == "It is nine credits."
        assert len(mock_llm.requests) == 3
        redrawn = mock_llm.requests[2]["messages"]
        assert redrawn[:-1] == mock_llm.requests[1]["messages"]
        note_turn = redrawn[-1]
        assert note_turn["role"] == "user" and note_turn["content"] == Prompt.EMPTY_DRAW_RETRY_NOTE
        assert "---" not in str(redrawn)

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
    handles: HandleTable | None = None,
) -> LoopContext:
    return LoopContext(
        step=step,
        is_final_step=is_final_step,
        tools_available=tools_available,
        source_text=source_text,
        records=records if records is not None else [],
        handles=handles if handles is not None else HandleTable(),
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
        """A ``Retry`` carries the condition and nothing else (#1937/#2160).

        The loop answers it by re-drawing on the unchanged conversation, writing nothing
        into it, so a nudge field would be a parameter with nothing to apply it to."""
        assert "nudge" not in Retry.model_fields

    def test_refusal_validator_catches_an_opening_refusal(self):
        resp = _text_response("I'm sorry, but I can't help with that.")
        outcome = RefusalValidator().check(resp, _ctx())
        assert isinstance(outcome, Retry) and outcome.condition == ConditionKey.REFUSAL

    def test_refusal_validator_lets_through_a_mid_text_hedge(self):
        """A good, substantive answer that only mentions "I can't" partway through is
        not a refusal (#2805) — ``is_refusal`` reads the opening sentence, not the
        whole reply, so this kind of good answer is never discarded and redrawn."""
        resp = _text_response(
            "Here's what I found on the topic. I can't fix the thundering herd problem "
            "myself, but I can suggest a few mitigations: add jitter to the retry "
            "backoff, cap the maximum number of concurrent connections, and consider a "
            "circuit breaker so one failing dependency doesn't cascade into every "
            "caller at once."
        )
        assert isinstance(RefusalValidator().check(resp, _ctx()), Proceed)

    def test_refusal_validator_lets_through_a_long_honest_apology(self):
        """An apology for a REAL failure that goes on to explain what happened is not a
        refusal (#2805), even though it opens with "I'm sorry" — the length bound is
        what tells an honest, substantive apology apart from a bare refusal."""
        resp = _text_response(
            "I'm sorry, the image server didn't answer when I tried to generate that "
            "picture. I checked LLM_IMAGE_API_URL and it looks like the server isn't "
            "reachable right now, so I can't produce an image until that's fixed. You "
            "might want to check whether the mflux server process is still running on "
            "the host."
        )
        assert isinstance(RefusalValidator().check(resp, _ctx()), Proceed)

    def test_refusal_validator_lets_through_a_short_honest_apology(self):
        """A SHORT apology for a real failure is still not a refusal (#2808) —
        "I'm sorry" carries no refusal phrase of its own, so a brief, honest "that
        failed" reply is never caught on length alone."""
        resp = _text_response("I'm sorry, the image server didn't answer.")
        assert isinstance(RefusalValidator().check(resp, _ctx()), Proceed)

    def test_refusal_validator_empty_content_is_not_a_refusal(self):
        resp = _text_response("")
        assert isinstance(RefusalValidator().check(resp, _ctx()), Proceed)

    def test_hallucinated_url_validator_uses_source_text(self):
        resp = _text_response("See https://made-up.example/never for details.")
        # No source text → nothing to check.
        assert isinstance(HallucinatedUrlValidator().check(resp, _ctx(source_text="")), Proceed)
        # URL absent from source → retry.
        bad = HallucinatedUrlValidator().check(resp, _ctx(source_text="unrelated text"))
        assert isinstance(bad, Retry) and bad.condition == ConditionKey.HALLUCINATED_URLS

        # A link citation the run's table does not hold is rejected with a note
        # naming which links DO exist (#2717) — the action-space repair beside the
        # raw-URL branch, which never fires for a handle-cited link.
        table = HandleTable()
        for i in range(1, 25):
            table.register(f"https://source.example/page-{i}")
        cited = _text_response("Read all about it, see [27].")
        dangling = HallucinatedUrlValidator().check(
            cited, _ctx(source_text="some source", handles=table)
        )
        assert isinstance(dangling, Retry) and dangling.condition == ConditionKey.HALLUCINATED_URLS
        assert dangling.note == (
            "There is no link [27] in this turn; the links you have are [1]–[24]."
        )
        # A citation the table holds proceeds — even with no raw URL in the source.
        ok_cite = _text_response("Read all about it, see [3].")
        assert isinstance(
            HallucinatedUrlValidator().check(
                ok_cite, _ctx(source_text="some source", handles=table)
            ),
            Proceed,
        )
        # With no links rendered this run, a bracketed number is prose, not a
        # failed citation — nothing to check.
        assert isinstance(
            HallucinatedUrlValidator().check(cited, _ctx(source_text="some source")), Proceed
        )
        # URL present in source → proceed.
        ok = HallucinatedUrlValidator().check(
            resp, _ctx(source_text="ref https://made-up.example/never here")
        )
        assert isinstance(ok, Proceed)

        # A URL the turn fetched is sourced however the reply wraps it (#2159): the URL is
        # read as the address alone.  The first three forms were observed — gpt-oss's
        # full-width citation bracket, markdown bold, and a link whose text is its own URL —
        # each read as part of the URL, each firing a Retry that threw a correct answer away.
        # The rest are here because the repair is the set of characters a URL may contain, a
        # closed set, rather than a list of the marks someone ran into: a test carrying only
        # what was seen would let the next wrapper regress in silence.
        cited = "https://lanternmuseum.org/admission"
        wiki = "https://en.wikipedia.org/wiki/Lantern_(lighting)"
        iri = "https://de.wikipedia.org/wiki/Laterne_Straße"
        source = f"browse {cited}: adult $18.50\nsee also {wiki} and {iri}"
        observed = [f"【browse: {cited}】", f"**{cited}**", f"[{cited}]({cited})"]
        swept = [
            f"{cited}.",
            f"{cited},",
            f"({cited})",
            f"[{cited}]",
            f"<{cited}>",
            f"`{cited}`",
            f'"{cited}"',
            f"'{cited}'",
            f"“{cited}”",
            f"«{cited}»",
            f"「{cited}」",
            f"（{cited}）",
            f"{cited}。",
            f"*{cited}*",
            f"__{cited}__",
            f"_{cited}_.",
            f"**{cited}**,",
            f"{cited}’s page",
            f"[{cited}]({cited}).",
            f"**[{cited}]({cited})**",
            f"[the museum]({cited})",
            f"[{cited}](/admission)",
            f"(see {wiki})",
            f"{iri}。",
        ]
        for form in observed + swept:
            reply = _text_response(f"The adult ticket is $18.50 {form}")
            outcome = HallucinatedUrlValidator().check(reply, _ctx(source_text=source))
            assert isinstance(outcome, Proceed), form

        # An invented URL is still caught in every position, read as its address alone.
        invented = "https://made-up.example/never"
        for form in (
            f"see {invented}",
            f"【browse: {invented}】",
            f"**{invented}**",
            f"({invented})",
            f"[{invented}]({invented})",
            f"[the museum]({invented})",
            f"[{invented}]({cited})",
            f"[{cited}]({invented})",
            f"[{cited}】]({invented})",
        ):
            assert find_hallucinated_urls(form, source) == [invented], form
            outcome = HallucinatedUrlValidator().check(
                _text_response(form), _ctx(source_text=source)
            )
            assert isinstance(outcome, Retry), form

    def test_hallucinated_url_validator_drops_the_link_once_the_retry_is_spent(self, caplog):
        """The SECOND draft (#2721): the retry-once condition is spent
        (``ctx.retried``), so a still-unsourced raw URL or a still-dangling
        handle no longer reaches the user silently — it is dropped (a
        ``Repair``) and the message says one was left out, logged the same
        way the chain's other repair records what it did.  A clean second
        draft proceeds untouched, same as a first draft that was never bad."""
        retried = {ConditionKey.HALLUCINATED_URLS}

        # Still-unsourced raw URL → Repair: the URL is gone, the sentence around
        # it stays, the notice is appended, and the original draft is untouched
        # (pure validator, deep copy).
        resp = _text_response("The price is $499, see https://made-up.example/deck for it.")
        with caplog.at_level(logging.WARNING, logger="penny.validation.response_validators"):
            outcome = HallucinatedUrlValidator().check(
                resp, _ctx(source_text="unrelated text", retried=retried)
            )
        assert isinstance(outcome, Repair)
        repaired_content = outcome.response.message.content
        assert "https://made-up.example/deck" not in repaired_content
        assert "The price is $499, see  for it." in repaired_content
        assert repaired_content.endswith(PennyResponse.LINK_DROPPED_NOTICE)
        assert resp.message.content == (
            "The price is $499, see https://made-up.example/deck for it."
        )
        assert any("Dropped unsourced URL" in r.message for r in caplog.records)

        # Still-dangling handle → Repair: the citation is gone, its text kept.
        table = HandleTable()
        table.register("https://source.example/page-1")
        dangling_resp = _text_response("Read about it, see [text][27].")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="penny.validation.response_validators"):
            outcome = HallucinatedUrlValidator().check(
                dangling_resp, _ctx(source_text="some source", handles=table, retried=retried)
            )
        assert isinstance(outcome, Repair)
        repaired_content = outcome.response.message.content
        assert "[27]" not in repaired_content
        assert "Read about it, see text." in repaired_content
        assert repaired_content.endswith(PennyResponse.LINK_DROPPED_NOTICE)
        assert any("Dropped dangling link handle" in r.message for r in caplog.records)

        # A clean second draft — nothing still bad — proceeds untouched.
        clean = _text_response("Read about it, see [1].")
        assert isinstance(
            HallucinatedUrlValidator().check(
                clean, _ctx(source_text="some source", handles=table, retried=retried)
            ),
            Proceed,
        )

        # A FIRST draft retries exactly as today — the retry-once condition is
        # not yet spent, so the same bad draft still re-draws rather than being
        # repaired in place (covered in full by test_hallucinated_url_validator_
        # uses_source_text above; asserted again here beside the spent-retry
        # case for contrast).
        first_draft = HallucinatedUrlValidator().check(resp, _ctx(source_text="unrelated text"))
        assert isinstance(first_draft, Retry)
        assert first_draft.condition == ConditionKey.HALLUCINATED_URLS

    def test_hallucinated_url_validator_drops_both_defects_in_one_second_draft(self, caplog):
        """A second draft can carry a dangling handle AND a separately typed
        unsourced raw URL at once — both are checked every time (#2721 review),
        so both are dropped in the one repair rather than the raw URL slipping
        through because the dangling check already matched."""
        table = HandleTable()
        table.register("https://source.example/page-1")
        both = _text_response(
            "Read about it, see [text][27]. Also check https://made-up.example/other for more."
        )
        with caplog.at_level(logging.WARNING, logger="penny.validation.response_validators"):
            outcome = HallucinatedUrlValidator().check(
                both,
                _ctx(
                    source_text="some source",
                    handles=table,
                    retried={ConditionKey.HALLUCINATED_URLS},
                ),
            )
        assert isinstance(outcome, Repair)
        repaired_content = outcome.response.message.content
        assert "[27]" not in repaired_content
        assert "https://made-up.example/other" not in repaired_content
        assert "Read about it, see text. Also check  for more." in repaired_content
        assert repaired_content.endswith(PennyResponse.LINK_DROPPED_NOTICE)
        messages = [r.message for r in caplog.records]
        assert any("Dropped dangling link handle" in m for m in messages)
        assert any("Dropped unsourced URL" in m for m in messages)

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
        # Chat's run-shape chain is EMPTY (#2455 naive baseline / #2473): what the run
        # WROTE is delivered straight onto the tool result the moment its call lands
        # (`ChatAgent.after_tool_step`), and there is no skill/round-framing machinery
        # left to narrate — so no text-branch guard is left to chain.  Its invalid draws
        # are still the ones that are not a reply, so a plain conversational reply stays
        # valid.
        assert ChatAgent.run_shape_validators == []
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
