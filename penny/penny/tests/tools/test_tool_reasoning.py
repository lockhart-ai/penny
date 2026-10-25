"""Tests for the reasoning field injected into tool call schemas.

``to_ollama_tool`` adds a ``reasoning`` string property to every tool's schema
— a structured per-call rationale the run record captures for display and safe
re-exposure in logs the model later reads (raw thinking is never fed back).
One carve-out, pinned here: a tool that declares ``reasoning`` in its OWN
``parameters`` (browse) keeps its hand-written description — injection never
overwrites a declaration.

The one observed misuse — a terminal ``done`` called with ONLY ``reasoning``,
displacing the required ``success``/``summary`` — is taught, not restructured:
``done`` keeps the injected param, its description marks the two fields
REQUIRED and says ``reasoning`` alone is never valid, and the shared
invalid-args envelope already names both missing required fields with their
type + description hints for that exact shape (asserted below).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from penny.database import Database
from penny.llm.call_link import CallLink
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.generate_image import IMAGE_GENERATION_FAILED, GenerateImageTool
from penny.tools.link_handles import HandleTable
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionRemoveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    GetEventTool,
    LogAppendTool,
    LogCreateTool,
    LogReadTool,
    MemoryMetadataTool,
    ReadRunCallsTool,
)
from penny.tools.models import ToolResult


class _DummyTool(Tool):
    """Minimal tool for testing reasoning injection."""

    name = "dummy"
    description = "A test tool"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "test param"},
        },
        "required": ["query"],
    }

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        return ToolResult(message="ok")


class TestToolReasoningSchema:
    """Test that to_ollama_tool() injects a reasoning property."""

    def test_reasoning_property_injected(self):
        """Ollama tool schema includes a reasoning property."""
        tool = _DummyTool()
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert "reasoning" in props
        assert props["reasoning"]["type"] == "string"

    def test_original_properties_preserved(self):
        """Original tool properties are still present alongside reasoning."""
        tool = _DummyTool()
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["description"] == "test param"

    def test_original_parameters_not_mutated(self):
        """Injecting reasoning does not mutate the tool's own parameters dict."""
        tool = _DummyTool()
        tool.to_ollama_tool()
        # The tool's own parameters should NOT have reasoning
        assert "reasoning" not in tool.parameters["properties"]

    def test_tool_declared_reasoning_not_overwritten(self, db: Database):
        """A tool that declares reasoning in its OWN parameters (browse) keeps
        its hand-written description — injection never overwrites it."""
        tool = BrowseTool(max_calls=3, db=db, embedding_client=MagicMock(), handles=HandleTable())
        own_description = tool.parameters["properties"]["reasoning"]["description"]
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert props["reasoning"]["description"] == own_description
        assert "inner monologue" not in props["reasoning"]["description"]


# Note (#1569): the old "done with ONLY reasoning is an invalid-args displacement
# failure" contract is gone — ``done()`` is argless, so once the loop strips the
# framework-injected ``reasoning`` the remaining call is the valid argless sentinel.
# Argless behaviour + extra-arg rejection is covered in ``tests/tools/test_memory_tools.py``.


class TestBrowseResultNarration:
    """`BrowseTool.to_result_narration` (the #1480 per-tool override) summarises
    the WHOLE browse call in one first-person line — reflecting search-vs-read and
    the outcome — while the seam (`format_result`) adds the `(browse result)` tag
    and the per-page `## browse ...:` section headers stay in the body.  The
    override returns ONLY the sentence: the tag is the seam's job, not the tool's.

    The override answers for synthetic results (arguments only, no dispatched
    list behind them): a read narrates its query verbatim; the pages a SEARCH
    opened are named only by the tool-composed narration (#2718)."""

    def test_search_success_narrates_searched(self):
        narration = BrowseTool.to_result_narration(
            {"items": [{"query": "deepest lake", "looking_for": "the depth"}]},
            ToolResult(message="Lake Baikal"),
        )
        assert narration == 'You searched for "deepest lake"'
        assert "(browse result)" not in narration  # the tag is the seam's job

    def test_url_read_success_narrates_opened(self):
        narration = BrowseTool.to_result_narration(
            {"items": [{"query": "https://example.com/a", "looking_for": "the depth"}]},
            ToolResult(message="page text"),
        )
        assert narration == "You opened https://example.com/a"

    def test_mixed_search_and_read(self):
        narration = BrowseTool.to_result_narration(
            {
                "items": [
                    {"query": "deepest lake", "looking_for": "the depth"},
                    {"query": "https://example.com/a", "looking_for": "the depth"},
                ]
            },
            ToolResult(message="ok"),
        )
        assert narration == 'You searched for "deepest lake" and opened https://example.com/a'

    def test_total_failure_narrates_honestly(self):
        narration = BrowseTool.to_result_narration(
            {"items": [{"query": "deepest lake", "looking_for": "the depth"}]},
            ToolResult(message="## browse error: ...", success=False),
        )
        assert narration == (
            'You searched for "deepest lake" but couldn\'t read anything — no page was '
            "read, so there's nothing current to quote"
        )

    def test_url_read_failure_narrates_honestly(self):
        narration = BrowseTool.to_result_narration(
            {"items": [{"query": "https://example.com/a", "looking_for": "the depth"}]},
            ToolResult(message="## browse error: ...", success=False),
        )
        assert narration == (
            "You opened https://example.com/a but couldn't read anything — no page was "
            "read, so there's nothing current to quote"
        )

    def test_missing_items_falls_back(self):
        # No items in the args (an arg-validation failure still flows the raw dict
        # through format_result) — narrate the action generically, honestly.
        assert (
            BrowseTool.to_result_narration({}, ToolResult(message="ok")) == "You looked things up"
        )
        assert BrowseTool.to_result_narration({}, ToolResult(message="e", success=False)) == (
            "You looked things up but couldn't read anything — no page was read, so "
            "there's nothing current to quote"
        )

    def test_tool_composed_narration_names_opened_pages_as_handles(self, db: Database):
        """The narration a real call carries (``ToolResult.narration``, #2661) names
        the pages the call OPENED, in the handle form the model reads (#2717) —
        URL reads and picked search reads alike."""
        tool = BrowseTool(max_calls=3, db=db, embedding_client=MagicMock(), handles=HandleTable())
        narration = tool._narration(["deepest lake"], ["[3]"], True)
        assert narration == 'You searched for "deepest lake" and opened [3]'
        failed = tool._narration(["deepest lake"], [], False)
        assert failed == (
            'You searched for "deepest lake" but couldn\'t read anything — no page was '
            "read, so there's nothing current to quote"
        )


class TestGenerateImageResultNarration:
    """`GenerateImageTool.to_result_narration` (the #1481 per-tool override) names
    what Penny drew in one first-person line, branching on `result.success`, while
    the seam (`format_result`) adds the `(generate_image result)` tag.  The override
    returns ONLY the sentence — the tag is the seam's job, not the tool's."""

    def test_success_narrates_drew(self):
        narration = GenerateImageTool.to_result_narration(
            {"description": "a red fox"}, ToolResult(message="Generated an image of: a red fox.")
        )
        assert narration == 'You drew "a red fox":'
        assert "(generate_image result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        narration = GenerateImageTool.to_result_narration(
            {"description": "a red fox"}, ToolResult(message="couldn't draw", success=False)
        )
        assert narration == 'You tried to draw "a red fox" but it didn\'t work:'

    def test_missing_description_falls_back(self):
        # No description in the args (an arg-validation failure still flows the raw
        # dict through format_result) — narrate the action generically, honestly.
        assert (
            GenerateImageTool.to_result_narration({}, ToolResult(message="ok"))
            == "You drew your image:"
        )
        assert (
            GenerateImageTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You tried to draw your image but it didn't work:"
        )

    def test_format_result_wraps_narration_with_tag_and_body(self):
        """End-to-end through the seam: registry dispatch → generate_image override
        → `(generate_image result)` tag → body, in one framed string the model reads."""
        framed = Tool.format_result(
            "generate_image",
            {"description": "a red fox"},
            ToolResult(message="Generated an image of: a red fox."),
        )
        assert framed == (
            'You drew "a red fox": (generate_image result)\nGenerated an image of: a red fox.'
        )


class TestGenerateImageToolHandlesItsOwnFailure:
    """``generate_image``'s backend call is wrapped (#2368): a raised failure is caught and
    narrated — what happened, with no order attached — rather than falling through to the
    executor's generic "check the arguments" narration, which makes no sense for a backend
    outage."""

    async def test_a_raised_generation_returns_an_actionable_failure(self):
        image_client = MagicMock()
        image_client.generate_image = AsyncMock(side_effect=RuntimeError("endpoint unreachable"))
        tool = GenerateImageTool(image_client, MagicMock(), MagicMock())

        result = await tool.execute(CallLink.new_top(), description="a red fox")

        assert result.success is False
        assert result.message == IMAGE_GENERATION_FAILED.format(e="endpoint unreachable")
        image_client.generate_image.assert_called_once()


class TestMemoryReadNarration:
    """The #1481 per-tool overrides for the memory-READ tools each summarise the
    read in one first-person line branching on `result.success`; the seam adds the
    `(<tool> result)` tag, so the override returns ONLY the sentence.  The eval
    survival test (`test_memory_reads_recap.py`) proves the line reaches the reply;
    these pin the exact strings.
    """

    def test_log_read_narrates_read(self):
        narration = LogReadTool.to_result_narration(
            {"log": "browse-results"}, ToolResult(message="entries")
        )
        assert narration == "You read `browse-results`:"

    def test_log_read_failure_narrates_honestly(self):
        narration = LogReadTool.to_result_narration(
            {"log": "browse-results"}, ToolResult(message="err", success=False)
        )
        assert narration == "You tried to read `browse-results` but it didn't work:"

    def test_memory_get_narrates_check(self):
        narration = MemoryMetadataTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="metadata")
        )
        assert narration == "You checked the details of `likes`:"

    def test_format_result_wraps_memory_read_narration(self):
        """End-to-end through the seam: registry dispatch → log_read override →
        `(log_read result)` tag → body."""
        framed = Tool.format_result(
            "log_read",
            {"log": "browse-results"},
            ToolResult(message="1. really into chess"),
        )
        assert framed == ("You read `browse-results`: (log_read result)\n1. really into chess")


class TestMemoryWriteNarration:
    """The mutated-aware write tools narrate three outcomes: a real change, a
    no-op (``mutated=False`` — the keystone honesty branch: a dedup/missing-key
    call says so, never a false "saved"), and a failure."""

    def test_collection_write_saved(self):
        """The headline names what the write did (#2758) — the key it landed under,
        read off the result's own ``saved_keys``, never re-derived from the call's
        arguments."""
        args = {"collection": "likes", "entries": [{"key": "chess", "content": "x"}]}
        result = ToolResult(message="ok", mutated=True, saved_keys=["chess"])
        assert (
            CollectionWriteTool.to_result_narration(args, result) == 'You saved "chess" to `likes`:'
        )

    def test_collection_write_already_there(self):
        args = {"collection": "likes", "entries": [{"key": "chess", "content": "x"}]}
        assert (
            CollectionWriteTool.to_result_narration(args, ToolResult(message="dup", mutated=False))
            == "You didn't add anything new to `likes` — it was already there:"
        )

    def test_collection_write_failure(self):
        args = {"collection": "likes", "entries": [{"key": "chess", "content": "x"}]}
        assert (
            CollectionWriteTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to save to `likes` but it didn't work:"
        )

    def test_delete_entry_three_outcomes(self):
        args = {"collection": "likes", "key": "chess"}
        assert (
            CollectionRemoveTool.to_result_narration(args, ToolResult(message="", mutated=True))
            == 'You removed "chess" from `likes`:'
        )
        assert (
            CollectionRemoveTool.to_result_narration(args, ToolResult(message="", mutated=False))
            == 'You couldn\'t find "chess" to remove from `likes`:'
        )
        assert (
            CollectionRemoveTool.to_result_narration(args, ToolResult(message="", success=False))
            == 'You tried to remove "chess" from `likes` but it didn\'t work:'
        )

    def test_log_append(self):
        args = {"log": "events", "content": "x"}
        assert (
            LogAppendTool.to_result_narration(args, ToolResult(message="ok", mutated=True))
            == "You added an entry to `events`:"
        )
        assert (
            LogAppendTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to add an entry to `events` but it didn't work:"
        )


class TestMemoryLifecycleNarration:
    """Success + failure narration for the collection/log-lifecycle and
    introspection tools — every registered tool speaks (epic #1478)."""

    def test_collection_configure(self):
        args = {"collection": "games"}
        assert (
            CollectionCreateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You set up the `games` collection:"
        )
        assert (
            CollectionCreateTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to set up the `games` collection but it didn't work:"
        )

    def test_log_create(self):
        args = {"log": "events"}
        assert (
            LogCreateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You set up the `events` log:"
        )

    def test_archive_unarchive(self):
        """One tool, both directions (#2812): `archived` names which."""
        archive_args = {"collection": "games", "archived": True}
        assert (
            CollectionArchiveTool.to_result_narration(archive_args, ToolResult(message="ok"))
            == "You archived `games`:"
        )
        unarchive_args = {"collection": "games", "archived": False}
        assert (
            CollectionArchiveTool.to_result_narration(unarchive_args, ToolResult(message="ok"))
            == "You restored `games` from the archive:"
        )

    def test_collection_configure_settings(self):
        args = {"collection": "games"}
        assert (
            CollectionUpdateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You updated `games`'s settings:"
        )

    def test_collection_get(self):
        args = {"collection": "likes", "key": "chess"}
        assert (
            CollectionGetTool.to_result_narration(args, ToolResult(message="ok"))
            == 'You looked up "chess" in `likes`:'
        )

    def test_collection_read_keys(self):
        args = {"collection": "likes"}
        assert (
            CollectionKeysTool.to_result_narration(args, ToolResult(message="ok"))
            == "You listed the keys in `likes`:"
        )

    def test_run_list_and_run_get(self):
        assert (
            ReadRunCallsTool.to_result_narration({"source": "chat"}, ToolResult(message="ok"))
            == "You reviewed `chat`'s recent runs:"
        )
        assert (
            GetEventTool.to_result_narration({"run": "run 7f3a1b2c"}, ToolResult(message="ok"))
            == "You looked up `run 7f3a1b2c`:"
        )
        assert (
            GetEventTool.to_result_narration({}, ToolResult(message="x", success=False))
            == "You tried to look up a run but it didn't work:"
        )
