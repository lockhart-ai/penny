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
from unittest.mock import MagicMock

from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.generate_image import GenerateImageTool
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCatalogTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMergeTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    CollectorRunHistoryTool,
    DoneTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    LogReadTool,
    MemoryMetadataTool,
    ReadPublishedLatestTool,
    ReadRunCallsTool,
    ReadSimilarTool,
    TestExtractionPromptTool,
    UpdateEntryTool,
)
from penny.tools.models import ToolResult
from penny.tools.schedule_tools import (
    ScheduleCreateTool,
    ScheduleDeleteTool,
    ScheduleListTool,
)
from penny.tools.send_message import SendMessageTool


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

    async def execute(self, **kwargs: Any) -> ToolResult:
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

    def test_done_injected_like_every_tool(self):
        """The terminal done tool gets the injected reasoning param too — the
        schema is uniform; its misuse is taught (description + envelope), not
        restructured.  ``reasoning`` stays optional: required is unchanged."""
        tool = DoneTool()
        schema = tool.to_ollama_tool()
        params = schema["function"]["parameters"]
        assert "reasoning" in params["properties"]
        assert set(params["required"]) == {"success", "summary"}

    def test_tool_declared_reasoning_not_overwritten(self):
        """A tool that declares reasoning in its OWN parameters (browse) keeps
        its hand-written description — injection never overwrites it."""
        tool = BrowseTool(max_calls=3, embedding_client=MagicMock())
        own_description = tool.parameters["properties"]["reasoning"]["description"]
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert props["reasoning"]["description"] == own_description
        assert "inner monologue" not in props["reasoning"]["description"]


class TestDoneOnlyReasoningEnvelope:
    """The observed displacement failure gets an actionable invalid-args envelope."""

    async def test_done_with_only_reasoning_names_both_required_fields(self):
        """A done call carrying ONLY reasoning (the observed invalid-args
        failure — reasoning is stripped before validation, leaving no required
        args) is refused with the shared envelope naming BOTH missing required
        fields plus their type + description hints, and telling the model to
        call done again."""
        result = await DoneTool().run(reasoning="all sources failed, wrapping up")
        assert result.success is False
        # The first-person frame is carried on ``narration`` (#1482); the per-field
        # remedy stays the body verbatim.
        assert result.narration == "You tried to use `done` but the arguments were wrong:"
        message = result.message
        assert "success" in message and "boolean" in message
        assert "summary" in message and "string" in message
        assert "Call done(<valid arguments>) again." in message


class TestBrowseResultNarration:
    """`BrowseTool.to_result_narration` (the #1480 per-tool override) summarises
    the WHOLE browse call in one first-person line — reflecting search-vs-read and
    the outcome — while the seam (`format_result`) adds the `(browse result)` tag
    and the per-page `## browse ...:` section headers stay in the body.  The
    override returns ONLY the sentence: the tag is the seam's job, not the tool's.
    """

    def test_search_success_narrates_searched(self):
        narration = BrowseTool.to_result_narration(
            {"queries": ["quillpad version"]}, ToolResult(message="v4.2")
        )
        assert narration == 'You searched for "quillpad version"'
        assert "(browse result)" not in narration  # the tag is the seam's job

    def test_url_read_success_narrates_opened(self):
        narration = BrowseTool.to_result_narration(
            {"queries": ["https://example.com/a"]}, ToolResult(message="page text")
        )
        assert narration == "You opened https://example.com/a"

    def test_mixed_search_and_read(self):
        narration = BrowseTool.to_result_narration(
            {"queries": ["quillpad version", "https://example.com/a"]},
            ToolResult(message="ok"),
        )
        assert narration == 'You searched for "quillpad version" and opened https://example.com/a'

    def test_total_failure_narrates_honestly(self):
        narration = BrowseTool.to_result_narration(
            {"queries": ["quillpad version"]},
            ToolResult(message="## browse error: ...", success=False),
        )
        assert narration == 'You searched for "quillpad version" but couldn\'t read anything'

    def test_url_read_failure_narrates_honestly(self):
        narration = BrowseTool.to_result_narration(
            {"queries": ["https://example.com/a"]},
            ToolResult(message="## browse error: ...", success=False),
        )
        assert narration == "You opened https://example.com/a but couldn't read anything"

    def test_missing_queries_falls_back(self):
        # No queries in the args (an arg-validation failure still flows the raw dict
        # through format_result) — narrate the action generically, honestly.
        assert (
            BrowseTool.to_result_narration({}, ToolResult(message="ok")) == "You looked things up"
        )
        assert (
            BrowseTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You looked things up but couldn't read anything"
        )

    def test_format_result_wraps_narration_with_tag_and_body(self):
        """End-to-end through the seam: registry dispatch → browse override →
        `(browse result)` tag → body, in one framed string the model reads."""
        framed = Tool.format_result(
            "browse", {"queries": ["quillpad version"]}, ToolResult(message="v4.2 is out")
        )
        assert framed == 'You searched for "quillpad version" (browse result)\nv4.2 is out'


class TestScheduleResultNarration:
    """The schedule tools' `to_result_narration` overrides (#1481) each lead the
    result with one first-person line branching on `result.success`; the seam
    (`format_result`) adds the `(<tool> result)` tag and keeps the body.  The
    override returns ONLY the sentence — the tag is the seam's job, not the tool's.
    The live survival contract is `tests/eval/test_schedule_recap.py`; these pin the
    exact strings deterministically so a reverted narration turns `make check` red.
    """

    def test_create_success_narrates_setup(self):
        narration = ScheduleCreateTool.to_result_narration(
            {"request": "every morning summarize chess news"},
            ToolResult(message="Scheduled ...", mutated=True),
        )
        assert narration == "You set up a schedule to handle 'every morning summarize chess news':"
        assert "(schedule_create result)" not in narration  # the tag is the seam's job

    def test_create_failure_narrates_honestly(self):
        narration = ScheduleCreateTool.to_result_narration(
            {"request": "hourly"}, ToolResult(message="Could not parse ...", success=False)
        )
        assert narration == "You tried to set up a schedule for 'hourly' but it didn't work:"

    def test_create_missing_request_falls_back(self):
        # An arg-validation failure still flows the raw dict through format_result.
        assert (
            ScheduleCreateTool.to_result_narration({}, ToolResult(message="ok", mutated=True))
            == "You set up a schedule to handle what you asked:"
        )

    def test_delete_success_narrates_removal(self):
        narration = ScheduleDeleteTool.to_result_narration(
            {"description": "the morning news"},
            ToolResult(message="Removed ...", mutated=True),
        )
        assert narration == "You removed the schedule for 'the morning news':"

    def test_delete_no_match_narrates_honestly(self):
        # The none-scheduled / no-recipient no-op returns success=False, so the
        # honest "couldn't find a matching schedule" frame leads — never a claim
        # that something was removed.
        narration = ScheduleDeleteTool.to_result_narration(
            {"description": "the morning news"},
            ToolResult(message="There are no scheduled tasks ...", success=False),
        )
        assert (
            narration == "You couldn't find a matching schedule to remove for 'the morning news':"
        )

    def test_delete_missing_description_falls_back(self):
        assert (
            ScheduleDeleteTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You couldn't find a matching schedule to remove for a scheduled task:"
        )

    def test_list_narrates_check(self):
        assert (
            ScheduleListTool.to_result_narration({}, ToolResult(message="The user's tasks: ..."))
            == "You checked what you have scheduled:"
        )
        assert (
            ScheduleListTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You tried to check your schedules but it didn't work:"
        )

    def test_format_result_wraps_narration_with_tag_and_body(self):
        """End-to-end through the seam: registry dispatch → schedule override →
        `(schedule_create result)` tag → body, in one framed string the model reads."""
        framed = Tool.format_result(
            "schedule_create",
            {"request": "daily at 8am summarize chess news"},
            ToolResult(message="Scheduled 'chess' to run daily at 8am.", mutated=True),
        )
        assert framed == (
            "You set up a schedule to handle 'daily at 8am summarize chess news': "
            "(schedule_create result)\nScheduled 'chess' to run daily at 8am."
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


class TestMemoryReadNarration:
    """The #1481 per-tool overrides for the memory-READ tools each summarise the
    read in one first-person line branching on `result.success`; the seam adds the
    `(<tool> result)` tag, so the override returns ONLY the sentence.  The eval
    survival test (`test_memory_reads_recap.py`) proves the line reaches the reply;
    these pin the exact strings.
    """

    def test_read_similar_narrates_search(self):
        narration = ReadSimilarTool.to_result_narration(
            {"memory": "user-messages", "anchor": "chess"}, ToolResult(message="entries")
        )
        assert narration == 'You searched `user-messages` for "chess":'
        assert "result)" not in narration  # the tag is the seam's job

    def test_read_similar_failure_narrates_honestly(self):
        narration = ReadSimilarTool.to_result_narration(
            {"memory": "user-messages", "anchor": "chess"},
            ToolResult(message="transient error", success=False),
        )
        assert narration == "You tried to search `user-messages` but it didn't work:"

    def test_read_similar_missing_anchor_falls_back(self):
        assert (
            ReadSimilarTool.to_result_narration({}, ToolResult(message="ok"))
            == "You searched your memory:"
        )

    def test_log_read_narrates_read(self):
        narration = LogReadTool.to_result_narration(
            {"memory": "browse-results"}, ToolResult(message="entries")
        )
        assert narration == "You read `browse-results`:"

    def test_log_read_failure_narrates_honestly(self):
        narration = LogReadTool.to_result_narration(
            {"memory": "browse-results"}, ToolResult(message="err", success=False)
        )
        assert narration == "You tried to read `browse-results` but it didn't work:"

    def test_collection_catalog_narrates_review(self):
        narration = CollectionCatalogTool.to_result_narration({}, ToolResult(message="catalog"))
        assert narration == "You reviewed your collection catalog:"

    def test_memory_metadata_narrates_check(self):
        narration = MemoryMetadataTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="metadata")
        )
        assert narration == "You checked the details of `likes`:"

    def test_format_result_wraps_memory_read_narration(self):
        """End-to-end through the seam: registry dispatch → read_similar override →
        `(read_similar result)` tag → body."""
        framed = Tool.format_result(
            "read_similar",
            {"memory": "user-messages", "anchor": "chess"},
            ToolResult(message="1. really into chess"),
        )
        assert framed == (
            'You searched `user-messages` for "chess": (read_similar result)\n1. really into chess'
        )


class TestMemoryWriteNarration:
    """The mutated-aware write tools narrate three outcomes: a real change, a
    no-op (``mutated=False`` — the keystone honesty branch: a dedup/missing-key
    call says so, never a false "saved"), and a failure."""

    def test_collection_write_saved(self):
        args = {"memory": "likes", "entries": [{"key": "chess", "content": "x"}]}
        assert (
            CollectionWriteTool.to_result_narration(args, ToolResult(message="ok", mutated=True))
            == 'You saved "chess" to `likes`:'
        )

    def test_collection_write_already_there(self):
        args = {"memory": "likes", "entries": [{"key": "chess", "content": "x"}]}
        assert (
            CollectionWriteTool.to_result_narration(args, ToolResult(message="dup", mutated=False))
            == "You didn't add anything new to `likes` — it was already there:"
        )

    def test_collection_write_failure(self):
        args = {"memory": "likes", "entries": [{"key": "chess", "content": "x"}]}
        assert (
            CollectionWriteTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to save to `likes` but it didn't work:"
        )

    def test_update_entry_three_outcomes(self):
        args = {"memory": "likes", "key": "chess", "content": "x"}
        assert (
            UpdateEntryTool.to_result_narration(args, ToolResult(message="ok", mutated=True))
            == 'You updated "chess" in `likes`:'
        )
        assert (
            UpdateEntryTool.to_result_narration(args, ToolResult(message="miss", mutated=False))
            == 'You couldn\'t find "chess" to update in `likes`:'
        )
        assert (
            UpdateEntryTool.to_result_narration(args, ToolResult(message="e", success=False))
            == 'You tried to update "chess" in `likes` but it didn\'t work:'
        )

    def test_delete_entry_three_outcomes(self):
        args = {"memory": "likes", "key": "chess"}
        assert (
            CollectionDeleteEntryTool.to_result_narration(
                args, ToolResult(message="", mutated=True)
            )
            == 'You removed "chess" from `likes`:'
        )
        assert (
            CollectionDeleteEntryTool.to_result_narration(
                args, ToolResult(message="", mutated=False)
            )
            == 'You couldn\'t find "chess" to remove from `likes`:'
        )
        assert (
            CollectionDeleteEntryTool.to_result_narration(
                args, ToolResult(message="", success=False)
            )
            == 'You tried to remove "chess" from `likes` but it didn\'t work:'
        )

    def test_log_append(self):
        args = {"memory": "events", "content": "x"}
        assert (
            LogAppendTool.to_result_narration(args, ToolResult(message="ok", mutated=True))
            == "You added an entry to `events`:"
        )
        assert (
            LogAppendTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to add an entry to `events` but it didn't work:"
        )


class TestSendMessageNarration:
    """``send_message`` narrates a queued send (``mutated``), a correct no-op
    decline (mute/refusal — ``success`` but not ``mutated``, so she "held off"),
    and a failure."""

    def test_messaged(self):
        assert (
            SendMessageTool.to_result_narration(
                {}, ToolResult(message="Message sent.", mutated=True)
            )
            == "You messaged the user:"
        )

    def test_held_off(self):
        assert (
            SendMessageTool.to_result_narration({}, ToolResult(message="muted", mutated=False))
            == "You started to message the user but held off:"
        )

    def test_failure(self):
        assert (
            SendMessageTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You tried to message the user but it didn't work:"
        )


class TestMemoryLifecycleNarration:
    """Success + failure narration for the collection/log-lifecycle and
    introspection tools — every registered tool speaks (epic #1478)."""

    def test_collection_create(self):
        args = {"name": "games"}
        assert (
            CollectionCreateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You set up the `games` collection:"
        )
        assert (
            CollectionCreateTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to set up the `games` collection but it didn't work:"
        )

    def test_log_create(self):
        args = {"name": "events"}
        assert (
            LogCreateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You set up the `events` log:"
        )

    def test_archive_unarchive(self):
        args = {"memory": "games"}
        assert (
            CollectionArchiveTool.to_result_narration(args, ToolResult(message="ok"))
            == "You archived `games`:"
        )
        assert (
            CollectionUnarchiveTool.to_result_narration(args, ToolResult(message="ok"))
            == "You restored `games` from the archive:"
        )

    def test_collection_update_settings(self):
        args = {"name": "games"}
        assert (
            CollectionUpdateTool.to_result_narration(args, ToolResult(message="ok"))
            == "You updated `games`'s settings:"
        )

    def test_collection_merge(self):
        args = {"from_memory": "a", "to_memory": "b"}
        assert (
            CollectionMergeTool.to_result_narration(args, ToolResult(message="ok"))
            == "You merged `a` into `b`:"
        )
        assert (
            CollectionMergeTool.to_result_narration(args, ToolResult(message="e", success=False))
            == "You tried to merge `a` into `b` but it didn't work:"
        )

    def test_collection_get(self):
        args = {"memory": "likes", "key": "chess"}
        assert (
            CollectionGetTool.to_result_narration(args, ToolResult(message="ok"))
            == 'You looked up "chess" in `likes`:'
        )

    def test_collection_read_random_and_keys(self):
        args = {"memory": "likes"}
        assert (
            CollectionReadRandomTool.to_result_narration(args, ToolResult(message="ok"))
            == "You pulled a random sample from `likes`:"
        )
        assert (
            CollectionKeysTool.to_result_narration(args, ToolResult(message="ok"))
            == "You listed the keys in `likes`:"
        )

    def test_exists(self):
        assert (
            ExistsTool.to_result_narration(
                {"memories": ["likes"], "content": "c"}, ToolResult(message="no")
            )
            == "You checked whether that entry already exists:"
        )

    def test_read_run_calls_and_history(self):
        assert (
            ReadRunCallsTool.to_result_narration({"target": "chat"}, ToolResult(message="ok"))
            == "You reviewed `chat`'s recent runs:"
        )
        assert (
            CollectorRunHistoryTool.to_result_narration(
                {"collector": "likes"}, ToolResult(message="ok")
            )
            == "You reviewed `likes`'s run history:"
        )

    def test_read_published_latest(self):
        assert (
            ReadPublishedLatestTool.to_result_narration({}, ToolResult(message="ok"))
            == "You checked for new entries to share:"
        )

    def test_done_reflects_cycle_outcome(self):
        assert (
            DoneTool.to_result_narration(
                {"success": True, "summary": "s"}, ToolResult(message="ok")
            )
            == "You wrapped up the cycle:"
        )
        assert (
            DoneTool.to_result_narration(
                {"success": False, "summary": "s"}, ToolResult(message="ok")
            )
            == "You wrapped up the cycle, marking it unfinished:"
        )

    def test_test_extraction_prompt(self):
        args = {"memory": "games"}
        assert (
            TestExtractionPromptTool.to_result_narration(args, ToolResult(message="ok"))
            == "You ran the `games` collector to test it:"
        )
        assert (
            TestExtractionPromptTool.to_result_narration(
                args, ToolResult(message="e", success=False)
            )
            == "You ran the `games` collector to test it, but the cycle didn't succeed:"
        )
