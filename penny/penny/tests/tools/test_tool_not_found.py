"""Tests for handling tool calls with non-existent tool names and missing parameters."""

from typing import Any, cast

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.llm import LlmClient
from penny.llm.call_link import CallLink
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.base import Tool, ToolExecutor, ToolRegistry
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import DoneTool
from penny.tools.models import ToolArgs, ToolCall


class _StubSearchArgs(ToolArgs):
    """Args for the search stub — one required query, extras forbidden like a real tool."""

    query: str


class StubSearchTool(Tool):
    """Minimal stub tool for testing tool-not-found handling."""

    name = "search"
    description = "Search for information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }
    args_model = _StubSearchArgs

    async def execute(self, call_link: CallLink, **kwargs):
        return "Mock search results for testing"


class TestToolNotFound:
    """Test handling of tool calls for tools that don't exist."""

    @pytest.mark.asyncio
    async def test_agent_returns_helpful_error_for_nonexistent_tool(self, test_db, mock_llm):
        """Agent returns helpful error listing available tools for non-existent tool.

        Also pins the degeneracy-guard boundary: a PLAUSIBLE unknown name
        (`example_function_name` — no collapse characters) must reach the
        executor's tool-not-found path and have its error fed back to the model,
        never be discarded-and-rerolled as degenerate output (that path is
        reserved for collapse-shaped names like `Functions?????`)."""
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
        )
        search_tool = StubSearchTool()

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
            tools=[search_tool],
            db=db,
            config=config,
        )

        # Track messages sent to the model to verify error handling
        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # First call: return tool call with non-existent tool name
                return mock_llm._make_tool_call_response(
                    request, "example_function_name", {"query": "test"}
                )
            # Second call: return final response after receiving error
            return mock_llm._make_text_response(request, "Let me use the correct search tool.")

        mock_llm.set_response_handler(handler)

        # Agent should not crash - it should handle the error gracefully
        response = await agent.run("test prompt", max_steps=3, call_link=CallLink.new_top())

        # Verify that we got a response (not a crash)
        assert response.answer is not None

        # The error should have been sent back to the model as a tool result
        assert len(messages_sent) == 2  # Initial call + retry after error
        # The second call should include a TOOL role message with the error
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0

        # The framed result leads with the first-person framework-failure frame
        # (#1482) and retains the actionable remedy (the available-tools list).
        error_content = tool_messages[0]["content"]
        assert "there's no such tool" in error_content
        assert "available" in error_content.lower()
        assert "search" in error_content.lower()  # The actual tool name

        await agent.close()

    @pytest.mark.asyncio
    async def test_harmony_suffixed_tool_name_still_dispatches(self, test_db, mock_llm):
        """A backend that leaks Harmony control tokens into the tool name
        (``search<|channel|>commentary``) still resolves to the real tool —
        the name is normalized at the read-off boundary, so dispatch succeeds
        instead of logging 'Tool not found'."""
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
        )
        search_tool = StubSearchTool()
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
            tools=[search_tool],
            db=db,
            config=config,
        )

        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # Harmony-suffixed name — the leak this fix defends against.
                return mock_llm._make_tool_call_response(
                    request, "search<|channel|>commentary", {"query": "test"}
                )
            return mock_llm._make_text_response(request, "Done searching.")

        mock_llm.set_response_handler(handler)

        await agent.run("test prompt", max_steps=3, call_link=CallLink.new_top())

        # The tool result fed back must be the successful search output, not a
        # 'not found' error — i.e. the Harmony-suffixed name dispatched cleanly.
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        result_content = tool_messages[0]["content"]
        assert "Mock search results for testing" in result_content
        assert "not found" not in result_content.lower()

        await agent.close()


class StubReadLatestTool(Tool):
    """Stub for read_latest — used in close-match suggestion tests."""

    name = "read_latest"
    description = "Read the latest entries from a memory collection"
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "Memory name."},
        },
        "required": ["memory"],
    }

    async def execute(self, call_link: CallLink, **kwargs):
        return "entries"


class StubMemoryGetTool(Tool):
    """Stub for memory_get — the tool the retired ``collection_metadata`` became."""

    name = "memory_get"
    description = "Return the metadata fields for a single memory"
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string", "description": "Memory name."}},
        "required": ["memory"],
    }

    async def execute(self, call_link: CallLink, **kwargs):
        return "metadata"


class StubCollectionConfigureTool(Tool):
    """Stub for collection_configure — registered alongside ``memory_get`` because it is
    the character-NEAREST name to the retired ``collection_metadata`` and therefore the
    wrong answer the string leg gives on its own (#1944)."""

    name = "collection_configure"
    description = "Create or reconfigure a collection"
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Collection name."}},
        "required": ["name"],
    }

    async def execute(self, call_link: CallLink, **kwargs):
        return "set"


class StubSkillGetTool(Tool):
    """Stub for skill_get — the tool the retired ``skill_read`` became (#2305)."""

    name = "skill_get"
    description = "Read one skill by name"
    parameters = {
        "type": "object",
        "properties": {"skill": {"type": "string", "description": "Skill name."}},
        "required": ["skill"],
    }

    async def execute(self, call_link: CallLink, **kwargs):
        return "skill"


def _stub_browse_tool() -> BrowseTool:
    """A real ``BrowseTool``, not a same-named stub subclass (#2510).

    ``Tool.__init_subclass__`` registers every subclass into the CLASS-level,
    process-wide ``Tool._registry`` by its ``name`` the moment the class BODY
    executes — not per instance. A stub class written as ``class
    StubBrowseTool(Tool): name = "browse"`` would silently replace the real
    ``BrowseTool`` in that registry for the rest of the test process, breaking
    every other test that dispatches through it (progress-emoji reactions,
    result narration, status text). Constructing the real tool, as other
    suites already do, carries its real name without ever re-registering it."""
    return BrowseTool(max_calls=3, embedding_client=cast(Any, MockLlmClient()))


class TestToolNotFoundSuggestion:
    """Error message includes a 'did you mean' suggestion for close tool names."""

    @pytest.mark.asyncio
    async def test_did_you_mean_for_read_last(self):
        """'read_last' produces a 'Did you mean read_latest?' suggestion, carrying
        that tool's own one-line description (#2510) so the model reads how the
        tool it meant actually works, not just its name.

        Also the fall-through guard on the rename leg (#1944): ``read_last`` IS a
        renamed name, but it resolves to ``collection_list``, which this registry
        does not carry — so the suggestion must come from the string leg and name a tool
        that actually exists rather than one the table remembers."""
        registry = ToolRegistry()
        registry.register(StubReadLatestTool())
        executor = ToolExecutor(registry, timeout=30.0)

        tool_call = ToolCall(tool="read_last", arguments={})
        result = await executor.execute(tool_call, CallLink.new_top())

        assert result.success is False
        # The first-person frame is carried on ``narration`` (#1482) so ``format_result``
        # doesn't double-frame this no-registered-class result; the did-you-mean remedy
        # stays the body.
        assert result.narration == "You tried to use `read_last` but there's no such tool."
        assert "Did you mean 'read_latest'?" in result.message
        assert "Read the latest entries from a memory collection" in result.message

    @pytest.mark.asyncio
    async def test_retired_name_suggests_the_tool_it_became(self):
        """A name this project RENAMED resolves through the rename table, not through
        character distance (#1944).

        The measured miss: a chat turn guessed ``collection_metadata`` — retired by
        migration 0059 — and the string leg answered the character-nearest name, which
        shares the longer prefix but is a different tool entirely, while the one it
        wanted was ``memory_get``.  Both are registered here, so the string leg's wrong
        answer is available and the assertion is that meaning wins."""
        registry = ToolRegistry()
        registry.register(StubMemoryGetTool())
        registry.register(StubCollectionConfigureTool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="collection_metadata", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert result.narration == (
            "You tried to use `collection_metadata` but there's no such tool."
        )
        # Whole render: the suggestion leads the body, carrying memory_get's own
        # description (#2510), and the available list still names both tools — the
        # wrong answer is present as a choice and was not taken.
        assert result.message == (
            "Did you mean 'memory_get'? "
            "memory_get: Return the metadata fields for a single memory. "
            "Available tools: memory_get, collection_configure. "
            "You must ONLY use the tools listed above."
        )

    @pytest.mark.asyncio
    async def test_retired_skill_read_suggests_skill_get(self):
        """``skill_read`` resolves through the rename table to ``skill_get`` (#2305),
        the tool that replaced it."""
        registry = ToolRegistry()
        registry.register(StubSkillGetTool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="skill_read", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert result.message == (
            "Did you mean 'skill_get'? skill_get: Read one skill by name. "
            "Available tools: skill_get. "
            "You must ONLY use the tools listed above."
        )

    @pytest.mark.asyncio
    async def test_unknown_but_plausible_name_still_gets_the_string_leg(self):
        """The rename table is a first leg, not a replacement: a name nobody retired —
        an ordinary typo — still gets the nearest spelling (#1944)."""
        registry = ToolRegistry()
        registry.register(StubMemoryGetTool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="memory_gt", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert result.message == (
            "Did you mean 'memory_get'? "
            "memory_get: Return the metadata fields for a single memory. "
            "Available tools: memory_get. "
            "You must ONLY use the tools listed above."
        )

    @pytest.mark.asyncio
    async def test_no_suggestion_for_unrecognisable_tool_name(self):
        """No suggestion is added when no close match exists."""
        registry = ToolRegistry()
        registry.register(StubReadLatestTool())
        executor = ToolExecutor(registry, timeout=30.0)

        tool_call = ToolCall(tool="completely_unknown_xyz", arguments={})
        result = await executor.execute(tool_call, CallLink.new_top())

        assert result.success is False
        assert "Did you mean" not in result.message

    @pytest.mark.asyncio
    async def test_real_name_inside_a_namespaced_malformed_one_is_suggested(self):
        """A malformed emission that wraps a real tool name in namespacing
        (``functions.done``) still gets a suggestion — the containment leg
        (#2510) — carrying that tool's own description. The call itself is still
        REJECTED (principles.md §2 item 7): nothing here runs ``done``."""
        registry = ToolRegistry()
        registry.register(DoneTool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="functions.done", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert result.message == (
            "Did you mean 'done'? "
            "done: Call this when the cycle is finished, with notify set to true or false. "
            "Available tools: done. "
            "You must ONLY use the tools listed above."
        )

    @pytest.mark.asyncio
    async def test_real_name_inside_a_stray_argument_fragment_is_suggested(self):
        """A malformed emission trailing a stray argument fragment
        (``browse(extract: the price``) still gets a suggestion from the
        containment leg (#2510)."""
        registry = ToolRegistry()
        registry.register(_stub_browse_tool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="browse(extract: the price", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert "Did you mean 'browse'?" in result.message
        assert "browse: Look things up." in result.message

    @pytest.mark.asyncio
    async def test_two_available_names_inside_a_malformed_one_suggest_nothing(self):
        """Two available tool names both sit inside the requested one — picking
        between them would be a guess, so the containment leg stays silent and the
        call is rejected with no suggestion (#2510, principles.md §2 item 7)."""
        registry = ToolRegistry()
        registry.register(DoneTool())
        registry.register(_stub_browse_tool())
        executor = ToolExecutor(registry, timeout=30.0)

        result = await executor.execute(
            ToolCall(tool="functions.browse.done", arguments={}), CallLink.new_top()
        )

        assert result.success is False
        assert "Did you mean" not in result.message


class _StubDoneArgs(ToolArgs):
    """Two required typed fields — the shape the missing-required-parameter tests
    validate against (independent of the real, now-argless ``done`` tool)."""

    success: bool
    summary: str


class StubDoneTool(Tool):
    """Stub tool with two required typed+described parameters."""

    name = "stub_done"
    description = "Signal completion"
    parameters = {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the cycle succeeded.",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence description of what was done.",
            },
        },
        "required": ["success", "summary"],
    }
    args_model = _StubDoneArgs

    async def execute(self, call_link: CallLink, **kwargs):
        return "done"


class TestMissingRequiredParameters:
    """``Tool.run`` validates args against ``args_model`` before ``execute`` and,
    on failure, returns an actionable error tool response that names each bad
    field with the type + description hint from ``parameters``."""

    @pytest.mark.asyncio
    async def test_missing_params_error_includes_type_and_description(self):
        """Error names type and description for each missing required parameter."""
        result = await StubDoneTool().run(CallLink.new_top())

        assert result.success is False
        # The first-person frame is carried on ``narration`` (#1482); the per-field
        # reasons stay the actionable remedy body.
        assert result.narration == "You tried to use `stub_done` but the arguments were wrong:"
        error = result.message
        assert "success" in error
        assert "boolean" in error
        assert "True if the cycle succeeded" in error
        assert "summary" in error
        assert "string" in error
        assert "One-sentence description" in error

    @pytest.mark.asyncio
    async def test_missing_params_error_only_lists_absent_params(self):
        """Only the actually-missing parameter appears in the error."""
        result = await StubDoneTool().run(CallLink.new_top(), success=True)

        assert result.success is False
        assert "summary" in result.message
        assert "success" not in result.message

    @pytest.mark.asyncio
    async def test_no_error_when_all_required_params_present(self):
        """With all required params, validation passes and execute runs."""
        result = await StubDoneTool().run(CallLink.new_top(), success=True, summary="done")

        assert result == "done"  # the stub's execute output — no validation failure

    @pytest.mark.asyncio
    async def test_unknown_param_rejected_not_silently_ignored(self):
        """An unknown argument no longer runs the tool with default behaviour — it
        flows through the envelope naming the bad param and the accepted ones."""
        result = await StubDoneTool().run(
            CallLink.new_top(), success=True, summary="done", verbose=True
        )

        assert result.success is False
        assert "unknown parameter 'verbose'" in result.message
        # The accepted parameters are named so the model can correct the call.
        assert "success" in result.message
        assert "summary" in result.message

    @pytest.mark.asyncio
    async def test_unknown_param_suggests_closest_valid_param(self):
        """A near-miss misspelling gets a 'did you mean' pointer at the real field."""
        result = await StubDoneTool().run(CallLink.new_top(), success=True, sumary="done")

        assert result.success is False
        assert "unknown parameter 'sumary'" in result.message
        assert "did you mean 'summary'" in result.message

    @pytest.mark.asyncio
    async def test_agent_sends_hint_rich_error_to_model_on_missing_params(self, test_db, mock_llm):
        """Validation error with type hints is fed back to the model for retry."""
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
        )
        tool = StubDoneTool()
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
            tools=[tool],
            db=db,
            config=config,
        )

        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # Call done with no arguments
                return mock_llm._make_tool_call_response(request, "stub_done", {})
            return mock_llm._make_text_response(request, "Fixed.")

        mock_llm.set_response_handler(handler)

        await agent.run("test", max_steps=3, call_link=CallLink.new_top())

        # The error fed back to the model must include type hints
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        error_content = tool_messages[0]["content"]
        assert "boolean" in error_content
        assert "string" in error_content

        await agent.close()

    @pytest.mark.asyncio
    async def test_injected_reasoning_arg_survives_forbid_extra(self, test_db, mock_llm):
        """The framework injects a ``reasoning`` param into every tool schema; it is
        popped before arg validation, so ``extra="forbid"`` never rejects it — a real
        tool call carrying reasoning still executes instead of erroring."""
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
        )
        tool = StubDoneTool()
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
            tools=[tool],
            db=db,
            config=config,
        )

        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # A well-formed call that also carries the injected reasoning arg.
                return mock_llm._make_tool_call_response(
                    request,
                    "stub_done",
                    {"success": True, "summary": "did it", "reasoning": "thinking out loud"},
                )
            return mock_llm._make_text_response(request, "Done.")

        mock_llm.set_response_handler(handler)

        await agent.run("test", max_steps=3, call_link=CallLink.new_top())

        # The tool result fed back is the stub's success — not a forbid-extra rejection.
        tool_messages = [m for m in messages_sent[1] if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        result_content = tool_messages[0]["content"]
        assert "done" in result_content
        assert "unknown parameter" not in result_content
        assert "invalid arguments" not in result_content

        await agent.close()
