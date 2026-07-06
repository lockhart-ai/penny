"""GitHub API client for agent operations.

Consolidates all GitHub API interactions (issues, PRs, workflow runs)
behind typed Pydantic models, replacing scattered gh CLI subprocess calls
with direct urllib.request calls to REST and GraphQL endpoints.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# GitHub API constants
GITHUB_API = "https://api.github.com"

# API paths
API_ISSUE_COMMENTS = "/repos/{owner}/{repo}/issues/{number}/comments"
API_PR_REVIEW_COMMENTS = "/repos/{owner}/{repo}/pulls/{pr_number}/comments"
API_WORKFLOW_RUNS = "/repos/{owner}/{repo}/actions/runs"
API_RUN_JOBS = "/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
API_JOB_LOGS = "/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
API_ISSUES = "/repos/{owner}/{repo}/issues"

# GraphQL queries
GQL_ISSUES_LIGHTWEIGHT = """
query($owner: String!, $repo: String!, $label: String!, $limit: Int!) {
  repository(owner: $owner, name: $repo) {
    issues(labels: [$label], states: OPEN, first: $limit,
           orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number updatedAt }
    }
  }
}
"""

GQL_ISSUES_DETAILED = """
query($owner: String!, $repo: String!, $label: String!, $limit: Int!) {
  repository(owner: $owner, name: $repo) {
    issues(labels: [$label], states: OPEN, first: $limit,
           orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title body stateReason
        author { login }
        labels(first: 10) { nodes { name } }
        comments(first: 50) {
          nodes { author { login } body createdAt }
        }
      }
    }
  }
}
"""

# Same shape as GQL_ISSUES_DETAILED but for CLOSED issues — used to feed the
# monitor dedup with bugs the user has already triaged as "won't fix" so the
# same class isn't refiled every cycle.  Filtering by stateReason happens
# client-side (GitHub's filterBy argument doesn't expose stateReason).
GQL_ISSUES_DETAILED_CLOSED = """
query($owner: String!, $repo: String!, $label: String!, $limit: Int!) {
  repository(owner: $owner, name: $repo) {
    issues(labels: [$label], states: CLOSED, first: $limit,
           orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title body stateReason
        author { login }
        labels(first: 10) { nodes { name } }
        comments(first: 50) {
          nodes { author { login } body createdAt }
        }
      }
    }
  }
}
"""

GQL_OPEN_PRS = """
query($owner: String!, $repo: String!, $limit: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, first: $limit) {
      nodes {
        number title body headRefName mergeable
        reviews(first: 50) {
          nodes { author { login } state submittedAt }
        }
        comments(first: 50) {
          nodes { author { login } body createdAt }
        }
        commits(last: 1) {
          nodes {
            commit {
              statusCheckRollup {
                contexts(first: 50) {
                  nodes {
                    __typename
                    ... on CheckRun { name conclusion status }
                    ... on StatusContext { context state }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models — Issue
# =============================================================================


class IssueListItem(BaseModel):
    """Lightweight issue for has_work() timestamp comparison."""

    model_config = ConfigDict(populate_by_name=True)

    number: int
    updated_at: str = Field("", alias="updatedAt")


class IssueAuthor(BaseModel):
    """Author of a GitHub issue."""

    login: str = ""


class IssueLabel(BaseModel):
    """A label on a GitHub issue."""

    name: str


class IssueComment(BaseModel):
    """A comment on a GitHub issue."""

    model_config = ConfigDict(populate_by_name=True)

    author: IssueAuthor = IssueAuthor()
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class IssueDetail(BaseModel):
    """Full issue details from detailed query."""

    model_config = ConfigDict(populate_by_name=True)

    number: int
    title: str = ""
    body: str = ""
    state_reason: str = Field("", alias="stateReason")
    author: IssueAuthor = IssueAuthor()
    labels: list[IssueLabel] = []
    comments: list[IssueComment] = []


# =============================================================================
# Pydantic models — Pull Request
# =============================================================================


class CommentAuthor(BaseModel):
    """Author of a PR comment or review."""

    login: str = ""


class PRComment(BaseModel):
    """A top-level PR comment."""

    model_config = ConfigDict(populate_by_name=True)

    author: CommentAuthor = CommentAuthor()
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class PRReview(BaseModel):
    """A formal PR review."""

    model_config = ConfigDict(populate_by_name=True)

    author: CommentAuthor = CommentAuthor()
    state: str = ""
    submitted_at: str = Field("", alias="submittedAt")
    body: str = ""


class CheckStatus(BaseModel):
    """A single CI check from statusCheckRollup.

    Normalizes both CheckRun and StatusContext GraphQL types into
    a single model. For CheckRun: state=status, conclusion=conclusion.
    For StatusContext: state=state, conclusion="".
    """

    name: str
    state: str
    conclusion: str = ""


class PullRequest(BaseModel):
    """A pull request with checks, reviews, and comments."""

    model_config = ConfigDict(populate_by_name=True)

    number: int
    title: str = ""
    body: str = ""
    head_ref_name: str = Field("", alias="headRefName")
    mergeable: str = ""
    status_check_rollup: list[CheckStatus] = []
    reviews: list[PRReview] = []
    comments: list[PRComment] = []


# =============================================================================
# Pydantic models — PR Review Comments (inline code review)
# =============================================================================


class ReviewCommentUser(BaseModel):
    """Author of an inline PR review comment (REST API format)."""

    login: str = ""


class ReviewComment(BaseModel):
    """An inline review comment on a pull request (from REST API)."""

    user: ReviewCommentUser = ReviewCommentUser()
    body: str = ""
    path: str = ""
    created_at: str = ""


# =============================================================================
# Pydantic models — GitHub Actions
# =============================================================================


class WorkflowRun(BaseModel):
    """A GitHub Actions workflow run."""

    id: int


class WorkflowJob(BaseModel):
    """A job within a GitHub Actions workflow run."""

    id: int
    conclusion: str = ""


# =============================================================================
# Internal models — GraphQL response shapes
# =============================================================================
#
# These mirror the exact structure of GraphQL/REST responses so raw JSON
# is parsed into a Pydantic tree immediately. Public API models (above)
# are the normalized, flattened interface for consumers.


class _GqlAuthor(BaseModel):
    """Nullable author node in GraphQL responses."""

    login: str = ""


_NULL_AUTHOR = _GqlAuthor()


# --- Issues (lightweight) response ---


class _GqlIssueLwNodeList(BaseModel):
    nodes: list[IssueListItem] = []


class _GqlIssueLwRepo(BaseModel):
    issues: _GqlIssueLwNodeList = _GqlIssueLwNodeList()


class _GqlIssueLwData(BaseModel):
    repository: _GqlIssueLwRepo = _GqlIssueLwRepo()


class _GqlIssueLwResponse(BaseModel):
    data: _GqlIssueLwData = _GqlIssueLwData()


# --- Issues (detailed) response ---


class _GqlIssueCommentNode(BaseModel):
    author: _GqlAuthor | None = None
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class _GqlIssueCommentNodeList(BaseModel):
    nodes: list[_GqlIssueCommentNode] = []


class _GqlLabelNodeList(BaseModel):
    nodes: list[IssueLabel] = []


class _GqlIssueDetailNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    number: int
    title: str = ""
    body: str = ""
    state_reason: str = Field("", alias="stateReason")
    author: _GqlAuthor | None = None
    labels: _GqlLabelNodeList = _GqlLabelNodeList()
    comments: _GqlIssueCommentNodeList = _GqlIssueCommentNodeList()


class _GqlIssueDetailNodeList(BaseModel):
    nodes: list[_GqlIssueDetailNode] = []


class _GqlIssueDetailRepo(BaseModel):
    issues: _GqlIssueDetailNodeList = _GqlIssueDetailNodeList()


class _GqlIssueDetailData(BaseModel):
    repository: _GqlIssueDetailRepo = _GqlIssueDetailRepo()


class _GqlIssueDetailResponse(BaseModel):
    data: _GqlIssueDetailData = _GqlIssueDetailData()


# --- PRs response ---


class _GqlCheckContext(BaseModel):
    """Raw statusCheckRollup context — union of CheckRun | StatusContext."""

    model_config = ConfigDict(populate_by_name=True)

    typename: str = Field("", alias="__typename")
    # CheckRun fields
    name: str = ""
    conclusion: str | None = None
    status: str = ""
    # StatusContext fields
    context: str = ""
    state: str = ""


class _GqlCheckContextNodeList(BaseModel):
    nodes: list[_GqlCheckContext] = []


class _GqlRollup(BaseModel):
    contexts: _GqlCheckContextNodeList = _GqlCheckContextNodeList()


class _GqlCommitObj(BaseModel):
    status_check_rollup: _GqlRollup | None = Field(None, alias="statusCheckRollup")


class _GqlCommitNode(BaseModel):
    commit: _GqlCommitObj = _GqlCommitObj()


class _GqlCommitNodeList(BaseModel):
    nodes: list[_GqlCommitNode] = []


class _GqlReviewNode(BaseModel):
    author: _GqlAuthor | None = None
    state: str = ""
    submitted_at: str = Field("", alias="submittedAt")


class _GqlReviewNodeList(BaseModel):
    nodes: list[_GqlReviewNode] = []


class _GqlPRCommentNode(BaseModel):
    author: _GqlAuthor | None = None
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class _GqlPRCommentNodeList(BaseModel):
    nodes: list[_GqlPRCommentNode] = []


class _GqlPRNode(BaseModel):
    number: int
    title: str = ""
    body: str = ""
    head_ref_name: str = Field("", alias="headRefName")
    mergeable: str = ""
    reviews: _GqlReviewNodeList = _GqlReviewNodeList()
    comments: _GqlPRCommentNodeList = _GqlPRCommentNodeList()
    commits: _GqlCommitNodeList = _GqlCommitNodeList()


class _GqlPRNodeList(BaseModel):
    nodes: list[_GqlPRNode] = []


class _GqlPRRepo(BaseModel):
    pull_requests: _GqlPRNodeList = Field(_GqlPRNodeList(), alias="pullRequests")


class _GqlPRData(BaseModel):
    repository: _GqlPRRepo = _GqlPRRepo()


class _GqlPRResponse(BaseModel):
    data: _GqlPRData = _GqlPRData()


# --- REST response wrappers ---


class _RestWorkflowRunsResponse(BaseModel):
    workflow_runs: list[WorkflowRun] = []


class _RestJobsResponse(BaseModel):
    jobs: list[WorkflowJob] = []


# =============================================================================
# Conversion helpers — GraphQL response nodes → public API models
# =============================================================================


def _to_issue_comment(node: _GqlIssueCommentNode) -> IssueComment:
    author = node.author or _NULL_AUTHOR
    return IssueComment.model_validate(
        {"author": {"login": author.login}, "body": node.body, "createdAt": node.created_at}
    )


def _to_issue_detail(node: _GqlIssueDetailNode) -> IssueDetail:
    author = node.author or _NULL_AUTHOR
    return IssueDetail(
        number=node.number,
        title=node.title,
        body=node.body,
        state_reason=node.state_reason,
        author=IssueAuthor(login=author.login),
        labels=node.labels.nodes,
        comments=[_to_issue_comment(c) for c in node.comments.nodes],
    )


_TYPENAME_CHECK_RUN = "CheckRun"

# GitHub issue ``stateReason`` enum value for "won't fix" / "not planned"
# closures.  The other observed values are ``COMPLETED`` (issue resolved)
# and ``REOPENED``.
_STATE_REASON_NOT_PLANNED = "NOT_PLANNED"


def _to_check_status(ctx: _GqlCheckContext) -> CheckStatus:
    if ctx.typename == _TYPENAME_CHECK_RUN:
        return CheckStatus(name=ctx.name, state=ctx.status, conclusion=ctx.conclusion or "")
    return CheckStatus(name=ctx.context, state=ctx.state, conclusion="")


def _to_pr_review(node: _GqlReviewNode) -> PRReview:
    author = node.author or _NULL_AUTHOR
    return PRReview.model_validate(
        {"author": {"login": author.login}, "state": node.state, "submittedAt": node.submitted_at}
    )


def _to_pr_comment(node: _GqlPRCommentNode) -> PRComment:
    author = node.author or _NULL_AUTHOR
    return PRComment.model_validate(
        {"author": {"login": author.login}, "body": node.body, "createdAt": node.created_at}
    )


def _to_pull_request(node: _GqlPRNode) -> PullRequest:
    checks: list[CheckStatus] = []
    if node.commits.nodes:
        rollup = node.commits.nodes[0].commit.status_check_rollup
        if rollup:
            checks = [_to_check_status(ctx) for ctx in rollup.contexts.nodes]

    return PullRequest(
        number=node.number,
        title=node.title,
        body=node.body,
        head_ref_name=node.head_ref_name,  # type: ignore[unknown-argument]
        mergeable=node.mergeable,
        status_check_rollup=checks,
        reviews=[_to_pr_review(r) for r in node.reviews.nodes],
        comments=[_to_pr_comment(c) for c in node.comments.nodes],
    )


# =============================================================================
# GitHubAPI client
# =============================================================================


class GitHubAPI:
    """GitHub API client using urllib.request with token-based auth.

    Uses GraphQL for complex queries (issues with comments, PRs with
    checks/reviews) and REST for simple operations (posting comments,
    Actions API).
    """

    def __init__(
        self, token_provider: Callable[[], str], owner: str = "lockhart-ai", repo: str = "penny"
    ) -> None:
        self._get_token = token_provider
        self._owner = owner
        self._repo = repo

    # --- Low-level request methods ---

    def _rest_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: int = 15,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        """Make a REST API request to GitHub."""
        url = f"{GITHUB_API}{path}"
        token = self._get_token()
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", accept)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                return json.loads(raw)
            return raw.decode()

    def _graphql(
        self,
        query: str,
        variables: dict | None = None,
        timeout: int = 15,
    ) -> dict:
        """Make a GraphQL API request to GitHub."""
        body: dict[str, Any] = {"query": query}
        if variables:
            body["variables"] = variables
        result = self._rest_request("POST", "/graphql", body=body, timeout=timeout)
        if result and "errors" in result:
            errors = result["errors"]
            msg = errors[0].get("message", "Unknown GraphQL error") if errors else "Unknown"
            raise RuntimeError(f"GraphQL error: {msg}")
        return result

    # --- Issues (GraphQL) ---

    def list_issues(self, label: str, limit: int = 20) -> list[IssueListItem]:
        """List issues by label — lightweight, returns number + updatedAt only.

        Used by has_work() for fast timestamp comparison.
        """
        raw = self._graphql(
            GQL_ISSUES_LIGHTWEIGHT,
            variables={
                "owner": self._owner,
                "repo": self._repo,
                "label": label,
                "limit": limit,
            },
        )
        response = _GqlIssueLwResponse.model_validate(raw)
        return response.data.repository.issues.nodes

    def list_issues_detailed(self, label: str, limit: int = 20) -> list[IssueDetail]:
        """List issues by label with full details including comments.

        Replaces the N+1 pattern of gh issue list + N x gh issue view
        with a single GraphQL query per label.
        """
        raw = self._graphql(
            GQL_ISSUES_DETAILED,
            variables={
                "owner": self._owner,
                "repo": self._repo,
                "label": label,
                "limit": limit,
            },
        )
        response = _GqlIssueDetailResponse.model_validate(raw)
        return [_to_issue_detail(node) for node in response.data.repository.issues.nodes]

    def list_closed_not_planned_issues(
        self, label: str, limit: int = 20
    ) -> list[IssueDetail]:
        """List closed-as-not-planned issues by label.

        Used by the monitor to keep dedup memory of "won't fix" decisions
        the user made for prior reports — closing an alias / sanitization
        PR with reason ``not planned`` should silence the whole class of
        error on future runs, not just the one instance.

        Fetches recent CLOSED issues and filters client-side by
        ``stateReason == NOT_PLANNED`` (GitHub's ``filterBy`` argument
        doesn't expose stateReason).
        """
        raw = self._graphql(
            GQL_ISSUES_DETAILED_CLOSED,
            variables={
                "owner": self._owner,
                "repo": self._repo,
                "label": label,
                "limit": limit,
            },
        )
        response = _GqlIssueDetailResponse.model_validate(raw)
        return [
            _to_issue_detail(node)
            for node in response.data.repository.issues.nodes
            if node.state_reason == _STATE_REASON_NOT_PLANNED
        ]

    # --- Issues (REST) ---

    def comment_issue(self, number: int, body: str) -> None:
        """Post a comment on a GitHub issue.

        Raises on failure (caller handles error semantics).
        """
        path = API_ISSUE_COMMENTS.format(
            owner=self._owner,
            repo=self._repo,
            number=number,
        )
        self._rest_request("POST", path, body={"body": body}, timeout=30)

    def create_issue(self, title: str, body: str, labels: list[str]) -> str:
        """Create a GitHub issue and return its URL.

        Args:
            title: Issue title (max 256 chars, will be truncated)
            body: Issue body (markdown supported)
            labels: List of label names to apply

        Returns:
            Issue URL (e.g., https://github.com/owner/repo/issues/123)

        Raises:
            RuntimeError: If issue creation fails
        """
        path = API_ISSUES.format(
            owner=self._owner,
            repo=self._repo,
        )
        payload = {
            "title": title[:256],  # Enforce GitHub limit
            "body": body,
            "labels": labels,
        }
        result = self._rest_request("POST", path, body=payload, timeout=30)
        return result["html_url"]

    # --- PRs (GraphQL) ---

    def list_open_prs(self, limit: int = 20) -> list[PullRequest]:
        """Fetch open PRs with checks, reviews, and comments.

        The statusCheckRollup union type (CheckRun | StatusContext) is
        normalized into CheckStatus objects for uniform handling.
        """
        raw = self._graphql(
            GQL_OPEN_PRS,
            variables={
                "owner": self._owner,
                "repo": self._repo,
                "limit": limit,
            },
        )
        response = _GqlPRResponse.model_validate(raw)
        return [_to_pull_request(node) for node in response.data.repository.pull_requests.nodes]

    # --- PR Review Comments (REST) ---

    def list_pr_review_comments(self, pr_number: int) -> list[ReviewComment]:
        """Fetch inline review comments on a pull request."""
        path = API_PR_REVIEW_COMMENTS.format(
            owner=self._owner,
            repo=self._repo,
            pr_number=pr_number,
        )
        raw_comments = self._rest_request("GET", path)
        return [ReviewComment.model_validate(c) for c in (raw_comments or [])]

    # --- Actions (REST) ---

    def list_failed_runs(self, branch: str, limit: int = 1) -> list[WorkflowRun]:
        """List recent failed workflow runs for a branch."""
        path = API_WORKFLOW_RUNS.format(
            owner=self._owner,
            repo=self._repo,
        )
        params = f"?branch={urllib.parse.quote(branch)}&status=failure&per_page={limit}"
        data = self._rest_request("GET", f"{path}{params}")
        response = _RestWorkflowRunsResponse.model_validate(data or {})
        return response.workflow_runs[:limit]

    def get_failed_job_log(self, run_id: int) -> str:
        """Fetch logs from failed jobs in a workflow run.

        Equivalent to `gh run view <id> --log-failed`. Lists jobs for
        the run, finds failed ones, and fetches each job's log text.
        """
        jobs_path = API_RUN_JOBS.format(
            owner=self._owner,
            repo=self._repo,
            run_id=run_id,
        )
        jobs_data = self._rest_request("GET", jobs_path)
        response = _RestJobsResponse.model_validate(jobs_data or {})
        failed_jobs = [j for j in response.jobs if j.conclusion == "failure"]

        if not failed_jobs:
            return ""

        log_parts: list[str] = []
        for job in failed_jobs:
            try:
                log_path = API_JOB_LOGS.format(
                    owner=self._owner,
                    repo=self._repo,
                    job_id=job.id,
                )
                log_text = self._rest_request("GET", log_path, timeout=30)
                if log_text:
                    log_parts.append(str(log_text))
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to fetch log for job {job.id}: {e}")

        return "\n".join(log_parts)
