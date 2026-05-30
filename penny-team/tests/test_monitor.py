"""Tests for the Monitor agent.

The Monitor agent reads penny's production logs, extracts errors,
and uses Claude CLI to analyze and file bug issues. Unlike other
agents, it reads files instead of GitHub issues.

Unit tests cover the pure extraction functions. Integration tests
verify the full flow through has_work() and run().
"""

from __future__ import annotations

from pathlib import Path

from penny_team.monitor import (
    ErrorBlock,
    MonitorAgent,
    extract_error_signature,
    extract_errors,
    filter_known_errors,
    format_errors_for_prompt,
)

from tests.conftest import (
    MockGitHubAPI,
    TRUSTED_USERS,
    extract_prompt,
    make_issue_detail,
    make_pull_request,
    result_event,
)


# =============================================================================
# Helper: create a MonitorAgent with a temp log file
# =============================================================================


def make_monitor_agent(
    tmp_path: Path,
    log_content: str = "",
) -> tuple[MonitorAgent, Path]:
    """Create a MonitorAgent with a temporary log file and prompt."""
    log_file = tmp_path / "penny.log"
    if log_content:
        log_file.write_text(log_content)

    agent_dir = tmp_path / "penny_team" / "monitor"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "CLAUDE.md").write_text("# Monitor Agent Prompt\n\nYou are the monitor agent.\n")

    agent = MonitorAgent(
        name="monitor",
        log_path=str(log_file),
        interval_seconds=300,
        timeout_seconds=600,
        working_dir=tmp_path,
        trusted_users=TRUSTED_USERS,
    )
    agent.prompt_path = agent_dir / "CLAUDE.md"

    # Override state path to use tmp_path
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "monitor.state.json"
    type(agent)._state_path = property(lambda self, p=state_path: p)

    return agent, log_file


# =============================================================================
# extract_errors — unit tests for pure function
# =============================================================================


class TestExtractErrors:
    def test_empty_log_returns_empty(self):
        assert extract_errors("") == []

    def test_info_lines_ignored(self):
        log = "2024-01-15 14:23:45 - penny.agent - INFO - All good\n"
        assert extract_errors(log) == []

    def test_single_error_no_traceback(self):
        log = "2024-01-15 14:23:45 - penny.agent - ERROR - Something broke\n"
        errors = extract_errors(log)
        assert len(errors) == 1
        assert errors[0].timestamp == "2024-01-15 14:23:45"
        assert errors[0].module == "penny.agent"
        assert errors[0].level == "ERROR"
        assert errors[0].message == "Something broke"
        assert errors[0].traceback == ""

    def test_error_with_traceback(self):
        log = (
            "2024-01-15 14:23:45 - penny.agent - ERROR - Unhandled exception\n"
            "Traceback (most recent call last):\n"
            '  File "penny/agent/base.py", line 42, in run\n'
            "    result = await self.execute()\n"
            "ValueError: invalid input\n"
            "2024-01-15 14:23:46 - penny.agent - INFO - Recovered\n"
        )
        errors = extract_errors(log)
        assert len(errors) == 1
        assert "Traceback" in errors[0].traceback
        assert "ValueError: invalid input" in errors[0].traceback

    def test_critical_level_extracted(self):
        log = "2024-01-15 14:23:45 - penny.database - CRITICAL - DB corrupted\n"
        errors = extract_errors(log)
        assert len(errors) == 1
        assert errors[0].level == "CRITICAL"

    def test_multiple_errors(self):
        log = (
            "2024-01-15 14:23:45 - penny.agent - ERROR - First error\n"
            "2024-01-15 14:23:46 - penny.agent - INFO - Some info\n"
            "2024-01-15 14:23:47 - penny.tools - ERROR - Second error\n"
        )
        errors = extract_errors(log)
        assert len(errors) == 2
        assert errors[0].message == "First error"
        assert errors[1].message == "Second error"

    def test_warning_level_ignored(self):
        log = "2024-01-15 14:23:45 - penny.agent - WARNING - Just a warning\n"
        assert extract_errors(log) == []


# =============================================================================
# format_errors_for_prompt — unit test
# =============================================================================


class TestFormatErrors:
    def test_no_errors_returns_no_errors_message(self):
        result = format_errors_for_prompt([])
        assert "No errors found" in result

    def test_formats_error_with_traceback(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.agent",
            level="ERROR",
            message="Something broke",
            traceback="Traceback (most recent call last):\n  ValueError: bad",
        )
        result = format_errors_for_prompt([error])
        assert "penny.agent" in result
        assert "Something broke" in result
        assert "ValueError: bad" in result
        assert "Error 1" in result


# =============================================================================
# has_work() — integration tests
# =============================================================================


class TestMonitorHasWork:
    def test_no_log_file_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(tmp_path)
        # No content was written, so file wasn't created
        assert agent.has_work() is False

    def test_empty_log_file_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(tmp_path)
        log_file.write_text("")
        assert agent.has_work() is False

    def test_first_run_with_content_returns_true(self, tmp_path):
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )
        assert agent.has_work() is True

    def test_no_new_content_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )
        file_size = log_file.stat().st_size
        agent._save_offset(file_size)
        assert agent.has_work() is False

    def test_new_content_appended_returns_true(self, tmp_path):
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - old\n",
        )
        old_size = log_file.stat().st_size
        agent._save_offset(old_size)

        with open(log_file, "a") as f:
            f.write("2024-01-15 14:24:00 - penny.agent - ERROR - new error\n")

        assert agent.has_work() is True

    def test_log_rotation_detected_returns_true(self, tmp_path):
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="short",
        )
        agent._save_offset(999999)
        assert agent.has_work() is True


# =============================================================================
# run() — integration tests through agent.run()
# =============================================================================


class TestMonitorRun:
    def test_no_errors_in_log_skips_claude(self, tmp_path, capture_popen):
        """Log with only INFO lines -> no errors -> Claude CLI not called."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - All good\n",
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No errors in logs"
        assert len(calls) == 0

    def test_errors_in_log_triggers_claude(self, tmp_path, capture_popen):
        """Log with ERROR lines -> errors extracted -> Claude CLI called."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.agent - ERROR - Unhandled exception\n"
                "Traceback (most recent call last):\n"
                '  File "penny/agent/base.py", line 42, in run\n'
                "    result = await self.execute()\n"
                "ValueError: invalid input\n"
            ),
        )

        calls = capture_popen(stdout_lines=[result_event("Filed issue #100")], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Monitor Agent Prompt" in prompt
        assert "Unhandled exception" in prompt
        assert "ValueError: invalid input" in prompt

    def test_offset_advances_after_run(self, tmp_path, capture_popen):
        """After run(), the byte offset advances so same content isn't re-read."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test error\n",
        )

        capture_popen(stdout_lines=[result_event()], returncode=0)
        agent.run()

        assert agent._load_offset() == log_file.stat().st_size
        assert agent.has_work() is False

    def test_offset_advances_even_without_errors(self, tmp_path, capture_popen):
        """After run() with no errors, offset still advances past INFO lines."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - All good\n",
        )

        capture_popen(stdout_lines=[result_event()], returncode=0)
        agent.run()

        assert agent._load_offset() == log_file.stat().st_size

    def test_log_rotation_resets_offset(self, tmp_path, capture_popen):
        """If log file is smaller than saved offset, reads from start."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - after rotation\n",
        )
        agent._save_offset(999999)

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "after rotation" in prompt

    def test_first_run_reads_tail_of_large_log(self, tmp_path, capture_popen):
        """On first run with a large log, only reads last 100KB."""
        filler = "2024-01-15 14:23:45 - penny.agent - INFO - filler line here\n"
        # Each filler line is ~56 bytes, need ~2000 lines to exceed 100KB
        large_content = filler * 2000
        error_line = "2024-01-15 14:23:45 - penny.agent - ERROR - recent error\n"
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=large_content + error_line,
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "recent error" in prompt

    def test_missing_log_file_returns_failure(self, tmp_path, capture_popen):
        """If log file doesn't exist at run time, returns failure."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="some content",
        )
        log_file.unlink()

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is False
        assert "Failed to read log" in result.output
        assert len(calls) == 0

    def test_claude_cli_failure_returns_failure(self, tmp_path, capture_popen):
        """If Claude CLI returns non-zero, result.success is False."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )

        capture_popen(stdout_lines=[result_event("error")], returncode=1)
        result = agent.run()

        assert result.success is False

    def test_known_errors_filtered_before_claude(self, tmp_path, capture_popen):
        """Errors matching open bug issues are filtered out before Claude is called."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.tools.search - ERROR - Search failed\n"
                "Traceback (most recent call last):\n"
                "  File \"penny/tools/search.py\", line 42\n"
                "AuthenticationError: insufficient quota\n"
            ),
        )

        # Set up a mock GitHub API with an existing matching bug issue
        mock_api = MockGitHubAPI()
        mock_api.set_issues_detailed(
            "bug",
            [
                make_issue_detail(
                    number=100,
                    title="bug: Perplexity search fails with AuthenticationError",
                    body="Module: penny.tools.search\nAuthenticationError: insufficient quota",
                    labels=["bug"],
                )
            ],
        )
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert result.output == "All errors already have open issues"
        assert len(calls) == 0  # Claude CLI not called

    def test_error_matching_open_pr_filtered_before_claude(self, tmp_path, capture_popen):
        """Errors matching an open PR (not an issue) are still filtered out."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.tools.search - ERROR - Search failed\n"
                "Traceback (most recent call last):\n"
                '  File "penny/tools/search.py", line 42\n'
                "AuthenticationError: insufficient quota\n"
            ),
        )

        mock_api = MockGitHubAPI()
        mock_api.set_issues_detailed("bug", [])  # No matching issues
        mock_api.set_prs([
            make_pull_request(
                number=748,
                branch="issue-100-fix-search-auth",
                title="fix: handle Perplexity AuthenticationError in penny.tools.search",
                body="Catches AuthenticationError quota errors",
            )
        ])
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert result.output == "All errors already have open issues"
        assert len(calls) == 0  # Claude CLI not called

    def test_novel_errors_passed_to_claude(self, tmp_path, capture_popen):
        """Errors NOT matching any open issue are passed to Claude."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.database - ERROR - DB locked\n",
        )

        mock_api = MockGitHubAPI()
        mock_api.set_issues_detailed("bug", [])  # No open issues
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event("Filed issue #101")], returncode=0)
        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "DB locked" in prompt


# =============================================================================
# extract_error_signature / filter_known_errors — unit tests
# =============================================================================


class TestErrorDedup:
    def test_signature_with_traceback(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.tools.search",
            level="ERROR",
            message="Search failed",
            traceback="Traceback:\n  File ...\nAuthenticationError: bad key",
        )
        sig = extract_error_signature(error)
        assert sig == "penny.tools.search:authenticationerror"

    def test_signature_without_traceback(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.agent",
            level="ERROR",
            message="Connection refused to Ollama",
            traceback="",
        )
        sig = extract_error_signature(error)
        assert sig == "penny.agent:connection refused to ollama"

    def test_filter_removes_matching_error(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.tools.search",
            level="ERROR",
            message="Search failed",
            traceback="AuthenticationError: quota exceeded",
        )
        open_issues = [
            make_issue_detail(
                number=1,
                title="bug: search auth error",
                body="penny.tools.search AuthenticationError",
                labels=["bug"],
            )
        ]
        result = filter_known_errors([error], open_issues)
        assert result == []

    def test_filter_keeps_novel_error(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.database",
            level="ERROR",
            message="DB locked",
            traceback="sqlite3.OperationalError: database is locked",
        )
        open_issues = [
            make_issue_detail(
                number=1,
                title="bug: search auth error",
                body="penny.tools.search AuthenticationError",
                labels=["bug"],
            )
        ]
        result = filter_known_errors([error], open_issues)
        assert len(result) == 1
        assert result[0].message == "DB locked"

    def test_filter_removes_error_matching_open_pr(self):
        """Errors matching an open PR's title+body are filtered out."""
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.tools.search",
            level="ERROR",
            message="Search failed",
            traceback="AuthenticationError: quota exceeded",
        )
        open_prs = [
            make_pull_request(
                number=748,
                branch="issue-100-fix-search-auth",
                title="Fix search AuthenticationError in penny.tools.search",
                body="Handles AuthenticationError quota issue",
            )
        ]
        result = filter_known_errors([error], [], open_prs)
        assert result == []

    def test_filter_with_no_open_issues_keeps_all(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.agent",
            level="ERROR",
            message="Something broke",
            traceback="",
        )
        result = filter_known_errors([error], [])
        assert len(result) == 1


# =============================================================================
# In-review dedup — integration test
# =============================================================================


class TestEmbeddingDedup:
    """Embedding similarity supplements substring matching when available."""

    def test_embedding_catches_error_that_substring_misses(self):
        """When substring matching misses but embedding is similar, error is filtered."""
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.tools.search",
            level="ERROR",
            message="Search failed",
            traceback="AuthenticationError: quota exceeded",
        )
        # Issue text that does NOT contain "penny.tools.search" verbatim
        # (e.g., Claude rewrote it), so substring match would miss
        open_issues = [
            make_issue_detail(
                number=1,
                title="bug: Perplexity search authentication error",
                body="The search tool fails with an auth error when quota is exhausted.",
                labels=["bug"],
            )
        ]

        # Without embeddings: substring miss → error passes through
        result = filter_known_errors([error], open_issues)
        assert len(result) == 1

        # With embeddings: high similarity → error is filtered
        # Simulate both the error sig and issue text having similar embeddings
        existing_vecs = [[1.0, 0.0, 0.0]]  # one vec per text
        # Monkey-patch _embed_signatures to return a similar vector
        import penny_team.monitor as monitor_mod
        original = monitor_mod._embed_signatures
        monitor_mod._embed_signatures = lambda sigs: [[0.95, 0.05, 0.0]]
        try:
            result = filter_known_errors([error], open_issues, existing_vecs=existing_vecs)
            assert len(result) == 0
        finally:
            monitor_mod._embed_signatures = original


class TestInReviewDedup:
    """Errors matching in-review issues (not just bug) should be filtered."""

    def test_in_review_issue_filters_matching_error(self, tmp_path, capture_popen):
        """An error that matches an in-review issue is filtered before Claude."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.tools.search - ERROR - Search failed\n"
                "Traceback (most recent call last):\n"
                '  File "penny/tools/search.py", line 42\n'
                "AuthenticationError: insufficient quota\n"
            ),
        )

        mock_api = MockGitHubAPI()
        # No bug-labeled issues — the issue was already moved to in-review
        mock_api.set_issues_detailed("bug", [])
        mock_api.set_issues_detailed(
            "in-review",
            [
                make_issue_detail(
                    number=785,
                    title="bug: Perplexity search fails with AuthenticationError",
                    body="Module: penny.tools.search\nAuthenticationError: insufficient quota",
                    labels=["in-review"],
                )
            ],
        )
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert result.output == "All errors already have open issues"
        assert len(calls) == 0  # Claude CLI not called


class TestToolNotFoundNormalization:
    """``Tool not found:`` signatures collapse the variable name to a placeholder.

    Without this, every novel hallucinated name (``search_memory``,
    ``get_latest``, ``collection_search?``…) produces a unique signature, so a
    single tracked bug can never match the next one.
    """

    def test_signature_collapses_variable_tool_name(self):
        a = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.tools.base",
            level="ERROR",
            message="Tool not found: search_memory",
            traceback="",
        )
        b = ErrorBlock(
            timestamp="2024-01-15 14:24:00",
            module="penny.tools.base",
            level="ERROR",
            message="Tool not found: collection_read_latest",
            traceback="",
        )
        # Two different hallucinated names produce the SAME signature.
        assert extract_error_signature(a) == extract_error_signature(b)

    def test_signature_preserves_non_template_message(self):
        """Other error messages aren't collapsed — only the tool-not-found family."""
        a = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.agent",
            level="ERROR",
            message="Connection refused to Ollama",
            traceback="",
        )
        b = ErrorBlock(
            timestamp="2024-01-15 14:24:00",
            module="penny.agent",
            level="ERROR",
            message="DB locked",
            traceback="",
        )
        assert extract_error_signature(a) != extract_error_signature(b)


class TestClosedNotPlannedDedup:
    """Closed-as-not-planned bug issues are part of the dedup corpus.

    When the user closes a bug with reason ``not planned`` they're declaring
    a policy: don't refile this class again.  The monitor honours that by
    fetching closed-not-planned bugs alongside open ones.
    """

    def test_closed_not_planned_filters_matching_error(self, tmp_path, capture_popen):
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.tools.base - ERROR - "
                "Tool not found: search_memory\n"
            ),
        )

        mock_api = MockGitHubAPI()
        mock_api.set_issues_detailed("bug", [])
        mock_api.set_issues_detailed("in-review", [])
        mock_api.set_closed_not_planned(
            "bug",
            [
                make_issue_detail(
                    number=991,
                    title="bug: Tool not found for hallucinated name",
                    body="Module: penny.tools.base\nTool not found",
                    labels=["bug"],
                    state_reason="NOT_PLANNED",
                )
            ],
        )
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert result.output == "All errors already have open issues"
        assert len(calls) == 0  # Claude CLI not called — error pre-filtered

    def test_closed_not_planned_fetch_failure_is_fail_open(self, tmp_path, capture_popen):
        """A failure fetching closed-not-planned doesn't block the dedup pass."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.tools.base - ERROR - "
                "Tool not found: search_memory\n"
            ),
        )

        mock_api = MockGitHubAPI()
        mock_api.set_issues_detailed("bug", [])
        mock_api.set_issues_detailed("in-review", [])
        mock_api._list_closed_not_planned_fail = True
        agent.github_api = mock_api

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        # Error not pre-filtered → Claude is invoked (no matching tracked bug).
        assert result.success is True
        assert len(calls) == 1
