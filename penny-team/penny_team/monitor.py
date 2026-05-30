"""Monitor agent: reads penny production logs, detects errors, files bug issues.

Reads new content from penny's log file since the last run, extracts
ERROR/CRITICAL lines with their tracebacks, deduplicates against open
bug issues in Python, and uses Claude CLI to analyze remaining errors
and create new bug issues for the Worker agent to fix.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from github_api.api import GitHubAPI, IssueDetail, PullRequest
from similarity.embeddings import cosine_similarity

from penny_team.base import Agent, AgentRun
from penny_team.constants import TeamConstants
from penny_team.utils.ollama_embed import embed_batch

logger = logging.getLogger(__name__)


@dataclass
class ErrorBlock:
    """A single error extracted from logs: the ERROR line plus any traceback."""

    timestamp: str
    module: str
    level: str
    message: str
    traceback: str


# Pattern to match standard Python log lines:
# "2024-01-15 14:23:45 - penny.module - ERROR - Error message here"
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"  # timestamp
    r" - "
    r"([\w.]+)"  # module
    r" - "
    r"(\w+)"  # level
    r" - "
    r"(.*)$",  # message
)


def extract_errors(log_text: str) -> list[ErrorBlock]:
    """Extract ERROR/CRITICAL log entries with their tracebacks.

    Scans log lines for ERROR or CRITICAL level entries. When found,
    captures the line and any subsequent lines that are part of a
    traceback (indented lines, "Traceback" header, exception lines)
    until the next timestamped log line.
    """
    errors: list[ErrorBlock] = []
    lines = log_text.split("\n")
    i = 0

    while i < len(lines):
        match = _LOG_LINE_RE.match(lines[i])
        if match:
            timestamp, module, level, message = match.groups()
            if level in TeamConstants.LOG_LEVELS_ERROR:
                # Collect traceback lines that follow
                traceback_lines: list[str] = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # Stop at the next timestamped log line
                    if _LOG_LINE_RE.match(next_line):
                        break
                    traceback_lines.append(next_line)
                    j += 1

                errors.append(
                    ErrorBlock(
                        timestamp=timestamp,
                        module=module,
                        level=level,
                        message=message,
                        traceback="\n".join(traceback_lines).strip(),
                    )
                )
                i = j
                continue
        i += 1

    return errors


# Pattern to match the last exception line in a traceback (e.g., "ValueError: bad input")
_EXCEPTION_RE = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning|Fault))\b", re.MULTILINE)

# Variable tail of a "Tool not found: <name>" error message.  The model emits
# a different hallucinated name on every cycle (``read_next``, ``get_latest``,
# ``collection_search?``, ``<|special|>``…), so the raw message produces a
# fresh signature every time.  Collapsing the trailing name to a placeholder
# lets a single closed-as-not-planned tracker silence the whole class.
_TOOL_NOT_FOUND_RE = re.compile(r"^(Tool not found):\s*\S.*$", re.IGNORECASE)
_TOOL_NOT_FOUND_FAMILY = "Tool not found"


def _normalize_error_message(message: str) -> str:
    """Strip variable identifiers from known error-message templates.

    Currently handles the ``Tool not found: <NAME>`` family.  Add other
    templates here when a single class of error keeps minting unique
    signatures (validation-error field names, etc.).  We collapse to the
    bare template prefix (``"Tool not found"``) rather than a placeholder
    token like ``<TOOL>``, so the resulting signature substring still
    matches typical bug-issue titles ("Tool not found for ...",
    "Tool not found: search_memory") via the substring dedup path.
    """
    if _TOOL_NOT_FOUND_RE.match(message):
        return _TOOL_NOT_FOUND_FAMILY
    return message


def extract_error_signature(error: ErrorBlock) -> str:
    """Extract a normalized signature for dedup: 'module:ExceptionType'.

    Uses the module from the log line and the exception type from the
    traceback. Falls back to the first few words of the message if
    no exception type is found in the traceback.  When a known
    error-message template is detected (e.g. ``Tool not found:``) the
    variable name is collapsed to a placeholder so the signature is
    stable across hallucinated names.
    """
    exception_type = ""
    if error.traceback:
        matches = _EXCEPTION_RE.findall(error.traceback)
        if matches:
            exception_type = matches[-1]  # Last match is the actual exception

    if not exception_type:
        # Fall back to first few significant words of the (normalized) message
        words = _normalize_error_message(error.message).split()[:4]
        exception_type = " ".join(words)

    return f"{error.module}:{exception_type}".lower()


def filter_known_errors(
    errors: list[ErrorBlock],
    open_issues: list[IssueDetail],
    open_prs: list[PullRequest] | None = None,
    existing_vecs: list[list[float]] | None = None,
) -> list[ErrorBlock]:
    """Remove errors that already have a matching open bug issue or PR.

    Uses substring matching (module + exception in text) as the primary
    signal, with embedding similarity as a supplementary catch for cases
    where the text doesn't contain the raw error verbatim.
    """
    all_texts = [f"{issue.title}\n{issue.body}".lower() for issue in open_issues]
    all_texts += [f"{pr.title}\n{pr.body}".lower() for pr in (open_prs or [])]

    if not all_texts:
        return errors

    # Embed error signatures if embedding model is available
    sigs = [extract_error_signature(e) for e in errors]
    sig_vecs = _embed_signatures(sigs) if existing_vecs else None

    novel: list[ErrorBlock] = []
    for i, error in enumerate(errors):
        module_part, exception_part = sigs[i].split(":", 1)

        # Primary: substring match (module + exception both in text)
        is_known = any(module_part in text and exception_part in text for text in all_texts)

        # Supplementary: embedding similarity
        if not is_known and sig_vecs and sig_vecs[i] and existing_vecs:
            is_known = any(
                cosine_similarity(sig_vecs[i], ev) >= TeamConstants.EMBEDDING_DEDUP_THRESHOLD
                for ev in existing_vecs
            )

        if is_known:
            logger.info(f"[monitor] Skipping known error: {error.module} / {exception_part}")
        else:
            novel.append(error)

    return novel


def _embed_signatures(sigs: list[str]) -> list[list[float]] | None:
    """Embed error signatures via Ollama. Returns None if unavailable."""
    model = os.getenv(TeamConstants.ENV_OLLAMA_EMBEDDING_MODEL)
    if not model:
        return None
    url = os.getenv(TeamConstants.ENV_OLLAMA_URL, TeamConstants.OLLAMA_DEFAULT_URL)
    return embed_batch(sigs, url, model)


def format_errors_for_prompt(errors: list[ErrorBlock]) -> str:
    """Format extracted errors into a section for the Claude prompt."""
    if not errors:
        return "\n\n# Log Errors\n\nNo errors found in recent logs.\n"

    parts = [
        "\n\n# Log Errors\n\n"
        "The following errors were extracted from penny's production logs. "
        "Analyze these errors, deduplicate against existing bug issues, and "
        "create new bug issues for genuinely new problems.\n\n---\n"
    ]

    for idx, error in enumerate(errors, 1):
        section = f"\n## Error {idx}\n"
        section += f"**Timestamp**: {error.timestamp}\n"
        section += f"**Module**: {error.module}\n"
        section += f"**Level**: {error.level}\n"
        section += f"**Message**: {error.message}\n"
        if error.traceback:
            section += f"\n**Traceback**:\n```\n{error.traceback}\n```\n"
        section += "\n---\n"
        parts.append(section)

    return "".join(parts)


class MonitorAgent(Agent):
    """Agent that monitors penny's production logs and files bug issues.

    Overrides has_work() and run() to read log files instead of GitHub
    issues. Uses byte offset tracking to only process new log content.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        name: str = "monitor",
        interval_seconds: int = 300,
        working_dir: Path | None = None,
        timeout_seconds: int = 600,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        github_app=None,  # GitHub App instance, kept for backward compat
        github_api: GitHubAPI | None = None,
        trusted_users: set[str] | None = None,
    ) -> None:
        from penny_team.base import PROJECT_ROOT

        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            working_dir=working_dir or PROJECT_ROOT,
            timeout_seconds=timeout_seconds,
            model=model,
            allowed_tools=allowed_tools,
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted_users,
        )
        if log_path is not None:
            self.log_path = Path(log_path)
        else:
            self.log_path = PROJECT_ROOT / TeamConstants.PENNY_LOG_RELATIVE_PATH

    def _load_offset(self) -> int:
        """Load saved byte offset from state file."""
        state = self._load_state()
        return int(state.get(TeamConstants.MONITOR_STATE_OFFSET, "0"))

    def _save_offset(self, offset: int) -> None:
        """Persist byte offset to state file."""
        self._save_state({TeamConstants.MONITOR_STATE_OFFSET: str(offset)})

    def has_work(self) -> bool:
        """Check if the log file has new content since the last read.

        Returns True if the log file has grown beyond the saved offset,
        if log rotation is detected, or on first run. Returns False if
        the file doesn't exist, is empty, or hasn't changed.
        """
        try:
            if not self.log_path.exists():
                logger.info(f"[{self.name}] Log file not found: {self.log_path}")
                return False

            file_size = self.log_path.stat().st_size
            if file_size == 0:
                return False

            saved_offset = self._load_offset()

            if file_size < saved_offset:
                logger.info(f"[{self.name}] Log rotation detected, will read from start")
                return True

            if file_size > saved_offset:
                return True

            logger.info(f"[{self.name}] No new log content since last run")
            return False

        except OSError as e:
            logger.warning(f"[{self.name}] Error checking log file: {e}")
            return True  # Fail-open

    def _read_new_log_content(self) -> tuple[str, int]:
        """Read new log content since the last saved offset.

        On first run, reads the last MONITOR_FIRST_RUN_MAX_BYTES.
        On subsequent runs, reads from the saved offset to EOF.
        If file is smaller than saved offset (rotation), resets to 0.

        Returns (content, new_offset) tuple.
        """
        saved_offset = self._load_offset()
        file_size = self.log_path.stat().st_size

        if file_size < saved_offset:
            saved_offset = 0

        if saved_offset == 0 and file_size > TeamConstants.MONITOR_FIRST_RUN_MAX_BYTES:
            saved_offset = file_size - TeamConstants.MONITOR_FIRST_RUN_MAX_BYTES

        with open(self.log_path) as f:
            f.seek(saved_offset)
            content = f.read()

        new_offset = self.log_path.stat().st_size
        return content, new_offset

    def _fetch_dedup_issues(self) -> list[IssueDetail]:
        """Fetch open + closed-as-not-planned bug issues for dedup.

        Three sources contribute:
          - Open ``bug`` issues (active reports).
          - Open ``in-review`` issues — the Worker relabels bugs to
            ``in-review`` after pushing a PR; without this, the same error
            gets filed again once the original issue leaves the ``bug``
            label.
          - Closed-as-not-planned ``bug`` issues — when the user closes a
            bug with reason ``not planned`` it's a policy decision that
            the whole class shouldn't be filed again.  Without this, a
            cleanup pass that closes 40 alias PRs as won't-fix is silently
            undone the next time the model hallucinates a new tool name.

        Returns empty list on failure (fail-open).
        """
        if self.github_api is None:
            return []

        issues: list[IssueDetail] = []
        seen: set[int] = set()

        for label in (TeamConstants.Label.BUG, TeamConstants.Label.IN_REVIEW):
            try:
                batch = self.github_api.list_issues_detailed(label, limit=30)
                for issue in batch:
                    if issue.number not in seen:
                        issues.append(issue)
                        seen.add(issue.number)
            except (OSError, RuntimeError) as e:
                logger.warning(f"[{self.name}] Failed to fetch {label} issues: {e}")

        try:
            closed = self.github_api.list_closed_not_planned_issues(
                TeamConstants.Label.BUG, limit=50
            )
            for issue in closed:
                if issue.number not in seen:
                    issues.append(issue)
                    seen.add(issue.number)
        except (OSError, RuntimeError) as e:
            logger.warning(f"[{self.name}] Failed to fetch closed-not-planned bugs: {e}")

        return issues

    def _fetch_open_prs(self) -> list[PullRequest]:
        """Fetch open PRs for dedup. Returns empty list on failure."""
        if self.github_api is None:
            return []
        try:
            return self.github_api.list_open_prs(limit=30)
        except (OSError, RuntimeError) as e:
            logger.warning(f"[{self.name}] Failed to fetch open PRs: {e}")
            return []  # Fail-open: skip dedup rather than blocking

    @staticmethod
    def _embed_dedup_texts(
        issues: list[IssueDetail], prs: list[PullRequest]
    ) -> list[list[float]] | None:
        """Embed issue/PR titles for dedup. Returns None if unavailable."""
        model = os.getenv(TeamConstants.ENV_OLLAMA_EMBEDDING_MODEL)
        if not model:
            return None
        url = os.getenv(TeamConstants.ENV_OLLAMA_URL, TeamConstants.OLLAMA_DEFAULT_URL)
        texts = [f"{i.title} {i.body}" for i in issues]
        texts += [f"{p.title} {p.body}" for p in prs]
        if not texts:
            return None
        return embed_batch(texts, url, model)

    def run(self) -> AgentRun:
        """Read new log content, extract errors, dedup, and run Claude to file bug issues."""
        logger.info(f"[{self.name}] Starting cycle #{self.run_count + 1}")
        start = datetime.now()

        try:
            log_content, new_offset = self._read_new_log_content()
        except OSError as e:
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.error(f"[{self.name}] Failed to read log file: {e}")
            return AgentRun(
                agent_name=self.name,
                success=False,
                output=f"Failed to read log: {e}",
                duration=duration,
                timestamp=start,
            )

        errors = extract_errors(log_content)

        if not errors:
            self._save_offset(new_offset)
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.info(f"[{self.name}] No errors found in new log content")
            return AgentRun(
                agent_name=self.name,
                success=True,
                output="No errors in logs",
                duration=duration,
                timestamp=start,
            )

        logger.info(f"[{self.name}] Found {len(errors)} error(s) in logs")

        # Python-space dedup: filter out errors matching open bug/in-review issues or PRs
        open_issues = self._fetch_dedup_issues()
        open_prs = self._fetch_open_prs()
        existing_vecs = self._embed_dedup_texts(open_issues, open_prs)
        errors = filter_known_errors(errors, open_issues, open_prs, existing_vecs)

        if not errors:
            self._save_offset(new_offset)
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.info(f"[{self.name}] All errors matched existing bug issues")
            return AgentRun(
                agent_name=self.name,
                success=True,
                output="All errors already have open issues",
                duration=duration,
                timestamp=start,
            )

        prompt = self.prompt_path.read_text()
        error_section = format_errors_for_prompt(errors)

        if len(error_section) > TeamConstants.MONITOR_MAX_ERROR_CONTEXT:
            error_section = (
                error_section[: TeamConstants.MONITOR_MAX_ERROR_CONTEXT] + "\n\n... (truncated)\n"
            )

        prompt += error_section

        success, result_text = self._execute_claude(prompt)

        # Save offset after execution so errors aren't re-processed,
        # regardless of Claude success (avoid infinite retry loops)
        self._save_offset(new_offset)

        duration = (datetime.now() - start).total_seconds()
        self.last_run = datetime.now()
        self.run_count += 1

        level = logging.INFO if success else logging.ERROR
        status = "OK" if success else "FAILED"
        logger.log(level, f"[{self.name}] Cycle #{self.run_count} {status} in {duration:.1f}s")

        return AgentRun(
            agent_name=self.name,
            success=success,
            output=result_text,
            duration=duration,
            timestamp=start,
        )
