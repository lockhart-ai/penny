"""Deterministic narration pins for the collector data/log tools (epic #1478, #1513).

`log_append` / `log_create` / `read_published_latest` / `collection_read_random` /
`collection_get` / `collection_keys` / `exists` each override
`to_result_narration` — one first-person line branching on `result.success`, the
result twin of `to_action_str`.  The seam (`format_result`) adds the
`(<tool> result)` tag and keeps the body; the override returns ONLY the sentence.

These tools are mostly collector-driven (the notifier drains `read_published_latest`,
producers `log_append`, dedup probes `exists` before writing), so their live
survival surface is the collector `done()` summary, which #1499's collector-done
mechanism already carries; `read_published_latest` gets a live done() survival case
in `tests/eval/test_collector_data_recap.py`.  These pin the exact strings
deterministically so a reverted narration turns `make check` red.
"""

from penny.tools.base import Tool
from penny.tools.memory_tools import (
    CollectionGetTool,
    CollectionKeysTool,
    CollectionReadRandomTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    ReadPublishedLatestTool,
)
from penny.tools.models import ToolResult


class TestLogCreateNarration:
    """`log_create` — a mutating create; success vs. an honest failure."""

    def test_success_narrates_created(self):
        narration = LogCreateTool.to_result_narration(
            {"name": "workout-log"}, ToolResult(message="Created log 'workout-log'.", mutated=True)
        )
        assert narration == "You created the log `workout-log`:"
        assert "(log_create result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        narration = LogCreateTool.to_result_narration(
            {"name": "workout-log"},
            ToolResult(message="A memory named 'workout-log' already exists.", success=False),
        )
        assert narration == "You tried to create the log `workout-log` but it didn't work:"

    def test_missing_name_falls_back(self):
        # An arg-validation failure still flows the raw dict through format_result.
        assert (
            LogCreateTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You tried to create the log a new log but it didn't work:"
        )


class TestLogAppendNarration:
    """`log_append` — a keyless append; success vs. the embed-failure refusal."""

    def test_success_narrates_added(self):
        narration = LogAppendTool.to_result_narration(
            {"memory": "browse-results"},
            ToolResult(message="Appended to 'browse-results'.", mutated=True),
        )
        assert narration == "You added an entry to `browse-results`:"
        assert "(log_append result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        narration = LogAppendTool.to_result_narration(
            {"memory": "browse-results"},
            ToolResult(message="Couldn't embed this entry ...", success=False),
        )
        assert narration == "You tried to add an entry to `browse-results` but it didn't work:"

    def test_missing_memory_falls_back(self):
        assert (
            LogAppendTool.to_result_narration({}, ToolResult(message="ok", mutated=True))
            == "You added an entry to a log:"
        )


class TestReadPublishedLatestNarration:
    """`read_published_latest` — the pub/sub consumer read; success vs. failure."""

    def test_success_narrates_checked(self):
        narration = ReadPublishedLatestTool.to_result_narration(
            {"n": 1}, ToolResult(message="1 new published entry (oldest first): ...")
        )
        assert narration == "You checked for new updates to share:"
        assert "(read_published_latest result)" not in narration  # the tag is the seam's job

    def test_no_new_still_narrates_checked(self):
        # The honest no-op body "(no new published entries)" is still a successful
        # read — the body carries the emptiness; the line names the action.
        narration = ReadPublishedLatestTool.to_result_narration(
            {}, ToolResult(message="(no new published entries)")
        )
        assert narration == "You checked for new updates to share:"

    def test_failure_narrates_honestly(self):
        narration = ReadPublishedLatestTool.to_result_narration(
            {"n": 1}, ToolResult(message="timed out", success=False)
        )
        assert narration == "You tried to check for new updates to share but it didn't work:"


class TestCollectionReadRandomNarration:
    """`collection_read_random` — a random-sample read; success vs. failure."""

    def test_success_narrates_sampled(self):
        narration = CollectionReadRandomTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="3 entries from `likes` (random sample): ...")
        )
        assert narration == "You sampled `likes` at random:"
        assert "(collection_read_random result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        narration = CollectionReadRandomTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="no such memory", success=False)
        )
        assert narration == "You tried to sample `likes` at random but it didn't work:"

    def test_missing_memory_falls_back(self):
        assert (
            CollectionReadRandomTool.to_result_narration({}, ToolResult(message="ok"))
            == "You sampled a collection at random:"
        )


class TestCollectionGetNarration:
    """`collection_get` — an exact-key lookup; success vs. failure, keyed narration."""

    def test_success_narrates_lookup(self):
        narration = CollectionGetTool.to_result_narration(
            {"memory": "likes", "key": "chess"},
            ToolResult(message="1 entry from `likes`: ..."),
        )
        assert narration == 'You looked up "chess" in `likes`:'
        assert "(collection_get result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        # A bracket-key rejection (`_bracket_key_rejection`) returns success=False.
        narration = CollectionGetTool.to_result_narration(
            {"memory": "likes", "key": "[chess]"},
            ToolResult(message="Key '[chess]' not found ...", success=False),
        )
        assert narration == 'You tried to look up "[chess]" in `likes` but it didn\'t work:'

    def test_missing_args_fall_back(self):
        assert (
            CollectionGetTool.to_result_narration({}, ToolResult(message="e", success=False))
            == "You tried to look up an entry in a collection but it didn't work:"
        )


class TestCollectionKeysNarration:
    """`collection_keys` — lists a collection's keys; success vs. failure."""

    def test_success_narrates_listed(self):
        narration = CollectionKeysTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="- chess\n- hiking")
        )
        assert narration == "You listed the keys in `likes`:"
        assert "(collection_keys result)" not in narration  # the tag is the seam's job

    def test_failure_narrates_honestly(self):
        narration = CollectionKeysTool.to_result_narration(
            {"memory": "likes"}, ToolResult(message="no such memory", success=False)
        )
        assert narration == "You tried to list the keys in `likes` but it didn't work:"

    def test_missing_memory_falls_back(self):
        assert (
            CollectionKeysTool.to_result_narration({}, ToolResult(message="ok"))
            == "You listed the keys in a collection:"
        )


class TestExistsNarration:
    """`exists` — the pre-write dedup probe; success vs. failure, named by content."""

    def test_success_narrates_checked(self):
        narration = ExistsTool.to_result_narration(
            {"memories": ["likes"], "content": "chess"}, ToolResult(message="yes")
        )
        assert narration == 'You checked whether "chess" is already saved:'
        assert "(exists result)" not in narration  # the tag is the seam's job

    def test_key_used_when_content_missing(self):
        narration = ExistsTool.to_result_narration(
            {"memories": ["likes"], "key": "chess"}, ToolResult(message="no")
        )
        assert narration == 'You checked whether "chess" is already saved:'

    def test_failure_narrates_honestly(self):
        narration = ExistsTool.to_result_narration(
            {"memories": ["likes"], "content": "chess"},
            ToolResult(message="no such memory", success=False),
        )
        assert (
            narration == 'You tried to check whether "chess" is already saved but it didn\'t work:'
        )

    def test_missing_probe_falls_back(self):
        assert (
            ExistsTool.to_result_narration({}, ToolResult(message="ok"))
            == "You checked whether that entry is already saved:"
        )


class TestFormatResultWrapsNarration:
    """End-to-end through the seam: registry dispatch → override → `(<tool> result)`
    tag → body, in one framed string the model reads."""

    def test_log_append_framed(self):
        framed = Tool.format_result(
            "log_append",
            {"memory": "browse-results"},
            ToolResult(message="Appended to 'browse-results'.", mutated=True),
        )
        assert framed == (
            "You added an entry to `browse-results`: (log_append result)\n"
            "Appended to 'browse-results'."
        )

    def test_read_published_latest_framed(self):
        framed = Tool.format_result(
            "read_published_latest",
            {"n": 1},
            ToolResult(message="(no new published entries)"),
        )
        assert framed == (
            "You checked for new updates to share: (read_published_latest result)\n"
            "(no new published entries)"
        )
