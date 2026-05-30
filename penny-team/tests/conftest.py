"""Shared test fixtures and helpers for penny-team tests.

Provides MockGitHubAPI for GitHub API interactions, subprocess mocking
for Claude CLI, agent factories, and data builders used across all test files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from penny_team.base import Agent
from github_api.api import (
    CheckStatus,
    CommentAuthor,
    IssueAuthor,
    IssueComment,
    IssueDetail,
    IssueLabel,
    IssueListItem,
    PRComment,
    PRReview,
    PullRequest,
    ReviewComment,
    ReviewCommentUser,
    WorkflowRun,
)

# Ensure penny-team package is importable (matches PYTHONPATH in Dockerfile)
PENNY_TEAM_ROOT = Path(__file__).parent.parent
if str(PENNY_TEAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PENNY_TEAM_ROOT))

# --- Constants ---

CODEOWNERS_CONTENT = "* @alice @bob\n"

BOT_SLUG = "penny-team"
BOT_LOGIN = "penny-team[bot]"
BOT_LOGINS = {BOT_SLUG, BOT_LOGIN}
TRUSTED_USERS = {"alice", "bob", BOT_SLUG, BOT_LOGIN}
CODEOWNERS_USERS = {"alice", "bob"}


# --- MockGitHubAPI ---


class MockGitHubAPI:
    """Mock GitHubAPI for tests — returns canned Pydantic model instances.

    Set up return values with set_* methods, then pass as github_api to agents.
    Tracks all method calls for assertion.
    """

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._issues: dict[str, list[IssueListItem]] = {}
        self._issues_detailed: dict[str, list[IssueDetail]] = {}
        self._closed_not_planned: dict[str, list[IssueDetail]] = {}
        self._prs: list[PullRequest] = []
        self._review_comments: dict[int, list[ReviewComment]] = {}
        self._failed_runs: dict[str, list[WorkflowRun]] = {}
        self._failed_logs: dict[int, str] = {}
        self._comment_issue_fail: bool = False
        self._list_issues_fail: bool = False
        self._list_issues_detailed_fail: bool = False
        self._list_closed_not_planned_fail: bool = False
        self._list_prs_fail: bool = False

    # --- Setup methods ---

    def set_issues(self, label: str, items: list[IssueListItem]) -> None:
        self._issues[label] = items

    def set_issues_detailed(self, label: str, items: list[IssueDetail]) -> None:
        self._issues_detailed[label] = items

    def set_closed_not_planned(self, label: str, items: list[IssueDetail]) -> None:
        self._closed_not_planned[label] = items

    def set_prs(self, prs: list[PullRequest]) -> None:
        self._prs = prs

    def set_review_comments(self, pr_number: int, comments: list[ReviewComment]) -> None:
        self._review_comments[pr_number] = comments

    def set_failed_runs(self, branch: str, runs: list[WorkflowRun]) -> None:
        self._failed_runs[branch] = runs

    def set_failed_log(self, run_id: int, log: str) -> None:
        self._failed_logs[run_id] = log

    # --- API methods (matching GitHubAPI interface) ---

    def list_issues(self, label: str, limit: int = 20) -> list[IssueListItem]:
        self.calls.append(("list_issues", (label,), {"limit": limit}))
        if self._list_issues_fail:
            raise RuntimeError("Mock list_issues failure")
        return self._issues.get(label, [])

    def list_issues_detailed(self, label: str, limit: int = 20) -> list[IssueDetail]:
        self.calls.append(("list_issues_detailed", (label,), {"limit": limit}))
        if self._list_issues_detailed_fail:
            raise RuntimeError("Mock list_issues_detailed failure")
        return self._issues_detailed.get(label, [])

    def list_closed_not_planned_issues(
        self, label: str, limit: int = 20
    ) -> list[IssueDetail]:
        self.calls.append(
            ("list_closed_not_planned_issues", (label,), {"limit": limit})
        )
        if self._list_closed_not_planned_fail:
            raise RuntimeError("Mock list_closed_not_planned_issues failure")
        return self._closed_not_planned.get(label, [])

    def comment_issue(self, number: int, body: str) -> None:
        self.calls.append(("comment_issue", (number, body), {}))
        if self._comment_issue_fail:
            raise RuntimeError("Mock comment_issue failure")

    def create_issue(self, title: str, body: str, labels: list[str]) -> str:
        self.calls.append(("create_issue", (title, body, labels), {}))
        return f"https://github.com/test/issues/{len(self.calls)}"

    def list_open_prs(self, limit: int = 20) -> list[PullRequest]:
        self.calls.append(("list_open_prs", (), {"limit": limit}))
        if self._list_prs_fail:
            raise RuntimeError("Mock list_open_prs failure")
        return self._prs

    def list_pr_review_comments(self, pr_number: int) -> list[ReviewComment]:
        self.calls.append(("list_pr_review_comments", (pr_number,), {}))
        return self._review_comments.get(pr_number, [])

    def list_failed_runs(self, branch: str, limit: int = 1) -> list[WorkflowRun]:
        self.calls.append(("list_failed_runs", (branch,), {"limit": limit}))
        return self._failed_runs.get(branch, [])

    def get_failed_job_log(self, run_id: int) -> str:
        self.calls.append(("get_failed_job_log", (run_id,), {}))
        return self._failed_logs.get(run_id, "")


# --- Agent factory ---


def make_agent(
    tmp_path: Path,
    name: str = "test-agent",
    required_labels: list[str] | None = None,
    interval: int = 300,
    timeout: int = 600,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    github_app: MagicMock | None = None,
    github_api: MockGitHubAPI | None = None,
    trusted_users: set[str] | None = TRUSTED_USERS,
    post_output_as_comment: bool = False,
    suppress_system_prompt: bool = True,
) -> Agent:
    """Create an agent with a temporary prompt file for integration testing."""
    agent_dir = tmp_path / "penny_team" / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_marker = f"# {name.title().replace('-', ' ')} Agent Prompt"
    (agent_dir / "CLAUDE.md").write_text(f"{prompt_marker}\n\nYou are the {name} agent.\n")

    agent = Agent(
        name=name,
        interval_seconds=interval,
        working_dir=tmp_path,
        timeout_seconds=timeout,
        model=model,
        allowed_tools=allowed_tools,
        required_labels=required_labels,
        github_app=github_app,
        github_api=github_api,
        trusted_users=trusted_users,
        post_output_as_comment=post_output_as_comment,
        suppress_system_prompt=suppress_system_prompt,
    )
    agent.prompt_path = agent_dir / "CLAUDE.md"
    return agent


# --- Data builders ---


def result_event(text: str = "Task completed") -> str:
    """Create a stream-json result event line."""
    return json.dumps({"type": "result", "result": text})


def make_issue_list_items(*numbers_and_timestamps: tuple[int, str]) -> list[IssueListItem]:
    """Create IssueListItem instances from (number, updatedAt) tuples."""
    return [
        IssueListItem(number=n, updated_at=ts)
        for n, ts in numbers_and_timestamps
    ]


def make_issue_detail(
    number: int = 42,
    title: str = "Add reminders feature",
    body: str = "Users should be able to set reminders via natural language.",
    author: str = "alice",
    labels: list[str] | None = None,
    comments: list[dict] | None = None,
    state_reason: str = "",
) -> IssueDetail:
    """Create an IssueDetail instance for testing."""
    issue_comments = []
    for c in (comments or []):
        issue_comments.append(
            IssueComment(
                author=IssueAuthor(login=c.get("author", {}).get("login", "")),
                body=c.get("body", ""),
                created_at=c.get("createdAt", ""),
            )
        )

    return IssueDetail(
        number=number,
        title=title,
        body=body,
        state_reason=state_reason,
        author=IssueAuthor(login=author),
        labels=[IssueLabel(name=l) for l in (labels or ["requirements"])],
        comments=issue_comments,
    )


def make_pull_request(
    number: int,
    branch: str,
    title: str = "",
    body: str = "",
    checks: list[CheckStatus] | None = None,
    mergeable: str = "MERGEABLE",
    reviews: list[PRReview] | None = None,
    comments: list[PRComment] | None = None,
) -> PullRequest:
    """Create a PullRequest instance for testing."""
    return PullRequest(
        number=number,
        title=title,
        body=body,
        head_ref_name=branch,
        status_check_rollup=checks or [],
        mergeable=mergeable,
        reviews=reviews or [],
        comments=comments or [],
    )


def make_check_status(
    name: str = "check",
    state: str = "COMPLETED",
    conclusion: str = "SUCCESS",
) -> CheckStatus:
    """Create a CheckStatus instance for testing."""
    return CheckStatus(name=name, state=state, conclusion=conclusion)


# --- Prompt extraction ---


def extract_prompt(calls: list[tuple[tuple, dict]]) -> str:
    """Extract the prompt string from captured Popen calls."""
    assert calls, "Expected Popen to be called, but it was not"
    cmd = calls[0][0][0]  # First call, positional args, first arg (command list)
    p_index = cmd.index("-p")
    return cmd[p_index + 1]


# --- Mock classes ---


class MockPopen:
    """Mock subprocess.Popen for Claude CLI stream-json output.

    Provides iterable stdout yielding JSON event lines,
    and standard process control methods.
    """

    def __init__(self, stdout_lines: list[str] | None = None, returncode: int = 0):
        self.stdout = iter(line + "\n" for line in (stdout_lines or []))
        self.returncode = returncode
        self.pid = 12345

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass

    def terminate(self):
        pass

    def poll(self):
        return self.returncode


# --- Fixtures ---


@pytest.fixture(autouse=True)
def isolate_state_dir(tmp_path, monkeypatch):
    """Isolate agent state files to tmp_path for all tests.

    Prevents _mark_processed and _save_state from writing to the real
    data directory, which would leak state between tests.
    """
    monkeypatch.setattr("penny_team.base.DATA_DIR", tmp_path)


@pytest.fixture
def mock_github_api():
    """Create a MockGitHubAPI instance for testing."""
    return MockGitHubAPI()


@pytest.fixture
def mock_popen(monkeypatch):
    """Provide a factory to create MockPopen instances and monkeypatch subprocess.Popen.

    Usage:
        popen = mock_popen(stdout_lines=['{"type":"result","result":"done"}'])
        # Now subprocess.Popen() returns the mock
    """
    mock_instance = None

    def factory(stdout_lines=None, returncode=0):
        nonlocal mock_instance
        mock_instance = MockPopen(stdout_lines=stdout_lines, returncode=returncode)
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_instance)
        return mock_instance

    return factory


@pytest.fixture
def capture_popen(monkeypatch):
    """Mock Popen that captures call args and returns canned stream-json output.

    Usage:
        calls = capture_popen(stdout_lines=['{"type":"result","result":"done"}'])
        agent.run()
        cmd = calls[0][0][0]  # First call, positional args, first arg (the command list)
        prompt = cmd[cmd.index("-p") + 1]
    """
    calls: list[tuple[tuple, dict]] = []

    def factory(stdout_lines=None, returncode=0):
        def popen_spy(*args, **kwargs):
            calls.append((args, kwargs))
            return MockPopen(stdout_lines=stdout_lines, returncode=returncode)

        monkeypatch.setattr(subprocess, "Popen", popen_spy)
        return calls

    return factory


@pytest.fixture
def project_root(tmp_path):
    """Create a temporary project root with a .github/CODEOWNERS file."""
    github_dir = tmp_path / ".github"
    github_dir.mkdir()
    (github_dir / "CODEOWNERS").write_text(CODEOWNERS_CONTENT)
    return tmp_path


@pytest.fixture
def mock_github_app():
    """Create a mock GitHubApp that doesn't make real API calls."""
    app = MagicMock()
    app.app_id = 12345
    app.installation_id = 67890
    app._fetch_slug.return_value = "penny-team"
    app.bot_name = "penny-team[bot]"
    app.bot_email = "12345+penny-team[bot]@users.noreply.github.com"
    app.get_token.return_value = "ghs_fake_token"
    app.get_env.return_value = {
        "GH_TOKEN": "ghs_fake_token",
        "GIT_AUTHOR_NAME": "penny-team[bot]",
        "GIT_AUTHOR_EMAIL": "12345+penny-team[bot]@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "penny-team[bot]",
        "GIT_COMMITTER_EMAIL": "12345+penny-team[bot]@users.noreply.github.com",
    }
    return app
