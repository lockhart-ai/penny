"""Configuration management for Penny."""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams

if TYPE_CHECKING:
    from penny.database import Database


def _load_dotenv() -> None:
    """Load .env file from project root or container path."""
    env_paths = [
        Path.cwd() / ".env",
        Path("/penny/.env"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


def _detect_channel_type() -> str:
    """Detect or read channel type from environment."""
    signal_number = os.getenv("SIGNAL_NUMBER")
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
    discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
    ios_enabled = os.getenv("IOS_ENABLED", "").lower() in ("1", "true", "yes")

    channel_type = os.getenv("CHANNEL_TYPE", "").lower()
    if channel_type:
        return channel_type

    has_discord = (
        discord_bot_token and discord_bot_token != "your-bot-token-here" and discord_channel_id
    )
    has_signal = signal_number and signal_number != "+1234567890"

    if has_discord and not has_signal:
        return "discord"
    if has_signal:
        return "signal"
    if ios_enabled:
        return "ios"
    raise ValueError(
        "No channel configured. Set either SIGNAL_NUMBER or "
        "DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID, or IOS_ENABLED=true in .env"
    )


def _validate_ios_config() -> None:
    """Validate iOS sidecar/APNs configuration."""
    # APNs credentials are optional so websocket-only local development works.
    # If any APNs field is provided, require the complete token-auth set.
    apns_fields = (
        os.getenv("IOS_APNS_TEAM_ID"),
        os.getenv("IOS_APNS_KEY_ID"),
        os.getenv("IOS_APNS_KEY_PATH"),
        os.getenv("IOS_BUNDLE_ID"),
    )
    if any(apns_fields) and not all(apns_fields):
        raise ValueError(
            "IOS_APNS_TEAM_ID, IOS_APNS_KEY_ID, IOS_APNS_KEY_PATH, and "
            "IOS_BUNDLE_ID are all required when APNs is configured"
        )


def _validate_channel_config(channel_type: str, ios_enabled: bool) -> None:
    """Validate required fields for the selected channel type."""
    if channel_type == "signal" and not os.getenv("SIGNAL_NUMBER"):
        raise ValueError("SIGNAL_NUMBER is required for Signal channel")
    if channel_type == "discord":
        discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_bot_token or discord_bot_token == "your-bot-token-here":
            raise ValueError(
                "DISCORD_BOT_TOKEN is required for Discord channel. "
                "Get your bot token from https://discord.com/developers/applications"
            )
        if not os.getenv("DISCORD_CHANNEL_ID"):
            raise ValueError("DISCORD_CHANNEL_ID is required for Discord channel")
    if channel_type == "ios" or ios_enabled:
        _validate_ios_config()


def _validate_embedding_config() -> None:
    """Ensure the required embedding model is configured.

    Embeddings back Penny's memory — preference dedup and similarity recall.
    Without one those features silently no-op, so the embedding model is a hard
    prerequisite rather than an optional degraded mode. Fail fast at startup
    with an actionable message instead of running memory-blind.
    """
    if not os.getenv("LLM_EMBEDDING_MODEL"):
        raise ValueError(
            "LLM_EMBEDDING_MODEL is required — Penny's memory (preference dedup and "
            "similarity recall) depends on it. Set LLM_EMBEDDING_MODEL in .env to a "
            "dedicated embedding model (e.g. embeddinggemma) and pull it with "
            "`ollama pull embeddinggemma`."
        )


def _collect_env_vars(channel_type: str) -> dict:
    """Read all config environment variables and return as constructor kwargs."""
    ios_enabled = os.getenv("IOS_ENABLED", "").lower() in ("1", "true", "yes")
    return {
        "channel_type": channel_type,
        "signal_number": os.getenv("SIGNAL_NUMBER"),
        "signal_api_url": os.getenv("SIGNAL_API_URL", "http://localhost:8080"),
        "discord_bot_token": os.getenv("DISCORD_BOT_TOKEN"),
        "discord_channel_id": os.getenv("DISCORD_CHANNEL_ID"),
        "llm_api_url": os.getenv("LLM_API_URL", "http://host.docker.internal:11434"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-oss:20b"),
        "llm_api_key": os.getenv("LLM_API_KEY", "not-needed"),
        "llm_vision_model": os.getenv("LLM_VISION_MODEL"),
        "llm_vision_api_url": os.getenv("LLM_VISION_API_URL"),
        "llm_vision_api_key": os.getenv("LLM_VISION_API_KEY"),
        "llm_image_model": os.getenv("LLM_IMAGE_MODEL"),
        "llm_embedding_model": os.getenv("LLM_EMBEDDING_MODEL"),
        "llm_embedding_api_url": os.getenv("LLM_EMBEDDING_API_URL"),
        "llm_embedding_api_key": os.getenv("LLM_EMBEDDING_API_KEY"),
        "image_api_url": os.getenv("LLM_IMAGE_API_URL", "http://host.docker.internal:11434"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "db_path": os.getenv("DB_PATH", "/penny/data/penny/penny.db"),
        "log_file": os.getenv("LOG_FILE"),
        "log_max_bytes": int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
        "log_backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
        "tool_timeout": float(os.getenv("TOOL_TIMEOUT", "120.0")),
        "llm_timeout": float(env_llm_timeout)
        if (env_llm_timeout := os.getenv("LLM_TIMEOUT"))
        else None,
        "fastmail_api_token": os.getenv("FASTMAIL_API_TOKEN"),
        "zoho_api_id": os.getenv("ZOHO_API_ID"),
        "zoho_api_secret": os.getenv("ZOHO_API_SECRET"),
        "zoho_refresh_token": os.getenv("ZOHO_REFRESH_TOKEN"),
        "browser_enabled": os.getenv("BROWSER_ENABLED", "").lower() in ("1", "true", "yes"),
        "browser_host": os.getenv("BROWSER_HOST", "localhost"),
        "browser_port": int(os.getenv("BROWSER_PORT", "9090")),
        "ios_enabled": ios_enabled,
        "ios_host": os.getenv("IOS_HOST", "0.0.0.0"),
        "ios_port": int(os.getenv("IOS_PORT", "9091")),
        "ios_pairing_token": os.getenv("IOS_PAIRING_TOKEN"),
        "ios_apns_team_id": os.getenv("IOS_APNS_TEAM_ID"),
        "ios_apns_key_id": os.getenv("IOS_APNS_KEY_ID"),
        "ios_apns_key_path": os.getenv("IOS_APNS_KEY_PATH"),
        "ios_bundle_id": os.getenv("IOS_BUNDLE_ID"),
        "ios_apns_sandbox": os.getenv("IOS_APNS_SANDBOX", "true").lower() in ("1", "true", "yes"),
    }


def _build_runtime_params(db: Database | None) -> RuntimeParams:
    """Build runtime params with env overrides."""
    env_overrides: dict[str, int | float | str] = {}
    for key, param in RUNTIME_CONFIG_PARAMS.items():
        env_val = os.getenv(key)
        if env_val is not None:
            with contextlib.suppress(ValueError):
                env_overrides[key] = param.validator(env_val)
    return RuntimeParams(db=db, env_overrides=env_overrides)


@dataclass
class Config:
    """Application configuration loaded from .env file."""

    # Channel type: "signal" or "discord"
    channel_type: str

    # Signal configuration (required if channel_type is "signal")
    signal_number: str | None
    signal_api_url: str

    # Discord configuration (required if channel_type is "discord")
    discord_bot_token: str | None
    discord_channel_id: str | None

    # LLM configuration (works with Ollama, omlx, or any OpenAI-compatible API)
    llm_api_url: str
    llm_model: str  # Text model for all agents
    llm_embedding_model: str  # Required embedding model (e.g. embeddinggemma) — backs memory

    # Logging configuration
    log_level: str

    # Database configuration
    db_path: str

    # Optional fields with defaults
    llm_api_key: str = "not-needed"
    log_file: str | None = None
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5
    llm_vision_model: str | None = None  # Vision model for image understanding
    llm_vision_api_url: str | None = None  # Override API URL for vision model
    llm_vision_api_key: str | None = None  # Override API key for vision model
    llm_image_model: str | None = None  # Image generation model (e.g., x/z-image-turbo)
    llm_embedding_api_url: str | None = None  # Override API URL for embedding model
    llm_embedding_api_key: str | None = None  # Override API key for embedding model
    image_api_url: str = "http://host.docker.internal:11434"  # Ollama REST API for /draw

    # LLM retry configuration
    llm_max_retries: int = 3
    llm_retry_delay: float = 0.5

    # Tool execution timeout (seconds)
    tool_timeout: float = 120.0

    # LLM read timeout in seconds (None = SDK default 600s). Use LLM_TIMEOUT env var to tune
    # for hardware where models take longer to respond. Connect timeout is always 5s.
    llm_timeout: float | None = None

    # Scheduler tick interval (seconds)
    scheduler_tick_interval: float = 1.0

    # Zoho API configuration (optional, enables /zoho command)
    zoho_api_id: str | None = None
    zoho_api_secret: str | None = None
    zoho_refresh_token: str | None = None

    # Fastmail JMAP configuration (optional, enables /email command)
    fastmail_api_token: str | None = None
    email_max_steps: int = 5

    # Browser extension server (runs alongside primary channel)
    browser_enabled: bool = False
    browser_host: str = "localhost"
    browser_port: int = 9090

    # iOS channel (primary when channel_type is "ios", sidecar when ios_enabled)
    ios_enabled: bool = False
    ios_host: str = "0.0.0.0"
    ios_port: int = 9091
    ios_pairing_token: str | None = None
    ios_apns_team_id: str | None = None
    ios_apns_key_id: str | None = None
    ios_apns_key_path: str | None = None
    ios_bundle_id: str | None = None
    ios_apns_sandbox: bool = True

    # Runtime-configurable params (DB override → env override → default)
    runtime: RuntimeParams = field(default_factory=RuntimeParams)

    @classmethod
    def load(cls, db: Database | None = None) -> Config:
        """Load configuration from .env file."""
        _load_dotenv()
        channel_type = _detect_channel_type()
        ios_enabled = os.getenv("IOS_ENABLED", "").lower() in ("1", "true", "yes")
        _validate_channel_config(channel_type, ios_enabled)
        _validate_embedding_config()
        return cls(**_collect_env_vars(channel_type), runtime=_build_runtime_params(db))


def setup_logging(
    log_level: str,
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If provided, logs to both file and console.
        max_bytes: Maximum log file size in bytes before rotation (default 10 MB).
        backup_count: Number of rotated backup files to keep (default 5).
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler (if log_file specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info("Logging to file: %s", log_file)

    # Silence noisy third-party loggers
    for name in (
        "httpcore",
        "httpx",
        "websockets",
        "perplexity",
        "primp",
        "rquest",
        "rustls",
        "reqwest",
        "hyper_util",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
