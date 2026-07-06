"""Shared constants for the penny-team orchestrator."""

from __future__ import annotations

import os
from enum import StrEnum


class TeamConstants:
    """All constants for the penny-team orchestrator."""

    class Label(StrEnum):
        """GitHub issue labels — each maps to exactly one agent."""

        REQUIREMENTS = "requirements"
        SPECIFICATION = "specification"
        IN_PROGRESS = "in-progress"
        IN_REVIEW = "in-review"
        BUG = "bug"

    # Labels where external state (CI checks, merge conflicts, reviews) can change
    # without updating issue timestamps
    LABELS_WITH_EXTERNAL_STATE = {Label.IN_REVIEW}

    # =========================================================================
    # CLI tools
    # =========================================================================

    CLAUDE_CLI = os.getenv("CLAUDE_CLI", "claude")
    GH_CLI = os.getenv("GH_CLI", "gh")

    # =========================================================================
    # Agent names
    # =========================================================================

    AGENT_PM = "product-manager"
    AGENT_ARCHITECT = "architect"
    AGENT_WORKER = "worker"
    AGENT_MONITOR = "monitor"
    AGENT_QUALITY = "quality"

    # =========================================================================
    # Agent timing (seconds)
    # =========================================================================

    PM_INTERVAL = 300
    PM_TIMEOUT = 600
    ARCHITECT_INTERVAL = 300
    ARCHITECT_TIMEOUT = 600
    WORKER_INTERVAL = 300
    WORKER_TIMEOUT = 1800
    MONITOR_INTERVAL = 300
    MONITOR_TIMEOUT = 600
    QUALITY_INTERVAL = 3600
    QUALITY_TIMEOUT = 600

    # =========================================================================
    # CI / PR status
    # =========================================================================

    CI_STATUS_PASSING = "passing"
    CI_STATUS_FAILING = "failing"

    # GitHub check conclusions that count as passing
    PASSING_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED", ""}

    # statusCheckRollup states that mean "still running"
    PENDING_STATES = {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED"}

    # GitHub review states that indicate feedback needing attention
    REVIEW_STATE_CHANGES_REQUESTED = "CHANGES_REQUESTED"

    # GitHub merge status
    MERGE_STATUS_CONFLICTING = "CONFLICTING"

    # Max characters of failure log to include in prompt
    MAX_LOG_CHARS = 3000

    # Max times the worker will attempt to fix CI before pausing for human help
    MAX_CI_FIX_ATTEMPTS = 3

    # =========================================================================
    # Stream-JSON event types (Claude CLI --output-format stream-json)
    # =========================================================================

    EVENT_ASSISTANT = "assistant"
    EVENT_RESULT = "result"

    # Stream-JSON content block types
    BLOCK_TEXT = "text"
    BLOCK_TOOL_USE = "tool_use"

    # =========================================================================
    # File names
    # =========================================================================

    PROMPT_FILENAME = "CLAUDE.md"
    ENV_FILENAME = ".env"
    ORCHESTRATOR_LOG = "orchestrator.log"

    # Standard CODEOWNERS file locations
    CODEOWNERS_PATHS = [
        ".github/CODEOWNERS",
        "CODEOWNERS",
        "docs/CODEOWNERS",
    ]

    # =========================================================================
    # GitHub App (auth constants live in github_api/auth.py)
    # =========================================================================

    GITHUB_REPO_OWNER = "lockhart-ai"
    GITHUB_REPO_NAME = "penny"
    BOT_SUFFIX = "[bot]"
    APP_PREFIX = "app/"
    NOREPLY_DOMAIN = "users.noreply.github.com"

    # Environment variable keys for GitHub App config
    ENV_APP_ID = "GITHUB_APP_ID"
    ENV_KEY_PATH = "GITHUB_APP_PRIVATE_KEY_PATH"
    ENV_INSTALL_ID = "GITHUB_APP_INSTALLATION_ID"

    # =========================================================================
    # Monitor agent — log parsing
    # =========================================================================

    # Log levels that indicate errors worth filing issues for
    LOG_LEVELS_ERROR = {"ERROR", "CRITICAL"}

    # Maximum bytes to read on first run (tail of log)
    MONITOR_FIRST_RUN_MAX_BYTES = 100 * 1024  # 100KB

    # Maximum bytes of error context to pass to Claude per cycle
    MONITOR_MAX_ERROR_CONTEXT = 50 * 1024  # 50KB

    # State key for byte offset
    MONITOR_STATE_OFFSET = "byte_offset"

    # =========================================================================
    # Quality agent — response evaluation
    # =========================================================================

    # Labels applied to quality-filed issues
    QUALITY_LABELS = ["bug", "quality"]

    # Maximum issues to file per cycle (safety cap)
    QUALITY_MAX_ISSUES_PER_CYCLE = 3

    # State key for last processed timestamp
    QUALITY_STATE_TIMESTAMP = "last_processed_at"

    # Maximum lookback on first run (no saved state) — avoids re-evaluating
    # the entire message history, which may contain already-fixed issues
    QUALITY_MAX_LOOKBACK_HOURS = 48

    # Minimum message length for privacy substring checks (shorter messages
    # like "yes" or "thanks" would cause false positives)
    QUALITY_PRIVACY_MIN_LENGTH = 20

    # Default Ollama API URL
    OLLAMA_DEFAULT_URL = "http://host.docker.internal:11434"

    # Ollama API endpoint for chat completions
    OLLAMA_CHAT_ENDPOINT = "/api/chat"

    # Environment variable keys for Ollama config
    ENV_OLLAMA_URL = "OLLAMA_API_URL"
    ENV_OLLAMA_MODEL = "OLLAMA_BACKGROUND_MODEL"
    ENV_OLLAMA_EMBEDDING_MODEL = "OLLAMA_EMBEDDING_MODEL"

    # Dedup thresholds (TCR + embedding, used by monitor and quality agents)
    TCR_DEDUP_THRESHOLD = 0.6
    EMBEDDING_DEDUP_THRESHOLD = 0.75

    # Data directory layout (relative to project root)
    PENNY_DB_RELATIVE_PATH = "data/penny/penny.db"
    PENNY_LOG_RELATIVE_PATH = "data/penny/logs/penny.log"
    TEAM_STATE_DIR = "data/penny-team/state"
    TEAM_LOG_DIR = "data/penny-team/logs"
