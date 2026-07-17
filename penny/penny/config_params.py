"""Runtime configuration parameter definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from penny.database import Database

# Auto-populated by ConfigParam.__post_init__
RUNTIME_CONFIG_PARAMS: dict[str, ConfigParam] = {}

# Group names (display order)
GROUP_CHAT = "Chat"
GROUP_BACKGROUND = "Background"
GROUP_MEMORY = "Memory"
GROUP_BROWSE = "Browse"
GROUP_SEND = "Send"
GROUP_EMAIL = "Email"

# Ordered list for display
CONFIG_GROUPS: list[str] = [
    GROUP_CHAT,
    GROUP_BACKGROUND,
    GROUP_MEMORY,
    GROUP_BROWSE,
    GROUP_SEND,
    GROUP_EMAIL,
]


@dataclass
class ConfigParam:
    """Definition of a runtime-configurable parameter.

    Automatically registers itself into RUNTIME_CONFIG_PARAMS on creation.
    """

    key: str
    description: str
    type: type  # int, float, or str
    default: int | float | str  # Default value (single source of truth)
    validator: Callable[[str], int | float | str]  # Parses and validates value from string
    group: str = GROUP_CHAT  # Display group for /config listing

    def __post_init__(self) -> None:
        RUNTIME_CONFIG_PARAMS[self.key] = self


def get_params_by_group() -> list[tuple[str, list[ConfigParam]]]:
    """Return params grouped by category in display order.

    Within each group, params are sorted alphabetically by key.
    """
    groups: dict[str, list[ConfigParam]] = {g: [] for g in CONFIG_GROUPS}
    for param in RUNTIME_CONFIG_PARAMS.values():
        groups[param.group].append(param)
    return [(g, sorted(groups[g], key=lambda p: p.key)) for g in CONFIG_GROUPS if groups[g]]


def _validate_positive_int(value: str) -> int:
    """Validate positive integer."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError("must be a positive integer") from e

    if parsed <= 0:
        raise ValueError("must be a positive integer")

    return parsed


def _validate_non_negative_int(value: str) -> int:
    """Validate a non-negative integer (0 allowed — e.g. to disable a feature)."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError("must be a non-negative integer") from e

    if parsed < 0:
        raise ValueError("must be a non-negative integer")

    return parsed


def _validate_positive_float(value: str) -> float:
    """Validate positive float."""
    try:
        parsed = float(value)
    except ValueError as e:
        raise ValueError("must be a positive number") from e

    if parsed <= 0:
        raise ValueError("must be a positive number")

    return parsed


def _validate_non_empty_string(value: str) -> str:
    """Validate non-empty string."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty string")
    return stripped


DOMAIN_MODE_RESTRICT = "restrict"
DOMAIN_MODE_ALLOW_ALL = "allow_all"
_VALID_DOMAIN_MODES = {DOMAIN_MODE_RESTRICT, DOMAIN_MODE_ALLOW_ALL}


def _validate_domain_mode(value: str) -> str:
    """Validate domain permission mode."""
    stripped = value.strip().lower()
    if stripped not in _VALID_DOMAIN_MODES:
        raise ValueError(f"must be one of: {', '.join(sorted(_VALID_DOMAIN_MODES))}")
    return stripped


def _validate_unit_float(value: str) -> float:
    """Validate float in (0.0, 1.0] range for similarity thresholds."""
    try:
        parsed = float(value)
    except ValueError as e:
        raise ValueError("must be a number between 0 and 1") from e

    if not (0.0 < parsed <= 1.0):
        raise ValueError("must be a number between 0 and 1")

    return parsed


# ── Chat — foreground conversation ───────────────────────────────────────────

ConfigParam(
    key="MAX_STEPS",
    description=(
        "Max agent loop steps per chat message cycle. Equal to "
        "BACKGROUND_MAX_STEPS by default: teaching (the demonstration the "
        "run-end skill extractor snapshots) runs in a chat turn, so any sequence "
        "longer than chat's budget would be unteachable while a collector "
        "could still execute it — equal budgets give both entry points the "
        "same power."
    ),
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

ConfigParam(
    key="MESSAGE_CONTEXT_LIMIT",
    description="Max recent conversation messages injected into chat context",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

# ── Background — every background agent (thinking, notify, extractors) ───────

ConfigParam(
    key="BACKGROUND_MAX_STEPS",
    description=(
        "Max agent loop steps per background-agent cycle. Equal to the chat "
        "MAX_STEPS by default — teaching happens in chat, so the two entry "
        "points share one step budget (teachable == executable)."
    ),
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="COLLECTOR_TICK_INTERVAL",
    description=(
        "Seconds between Collector dispatcher ticks (idle-gated).  Each tick "
        "the dispatcher checks which collection is most overdue based on its "
        "per-row collector_interval_seconds and runs that one.  Should be "
        "smaller than the smallest per-collection interval — otherwise that "
        "collection waits up to TICK_INTERVAL past its readiness."
    ),
    type=float,
    default=30.0,
    validator=_validate_positive_float,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="COLLECTOR_THROTTLE_AFTER",
    description=(
        "Consecutive idle cycles (no entries written / messages sent) before a "
        "collector backs off — its interval doubles, then the counter resets.  "
        "A productive cycle snaps the interval back to the user's set cadence.  "
        "0 disables auto-throttle."
    ),
    type=int,
    default=3,
    validator=_validate_non_negative_int,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="COLLECTOR_MAX_INTERVAL",
    description=(
        "Ceiling (seconds) for auto-throttle backoff — a collector's interval "
        "never doubles past this.  Default 604800 (one week)."
    ),
    type=int,
    default=604800,
    validator=_validate_positive_int,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="COLLECTOR_RUN_HISTORY",
    description=(
        "How many of a collector's own recent run summaries (the ``done`` summary "
        "of each past cycle) to show it at the top of every cycle, so it knows what "
        "its previous invocations did.  0 disables the section."
    ),
    type=int,
    default=10,
    validator=_validate_non_negative_int,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="IDLE_SECONDS",
    description="Seconds of silence before idle-gated background agents become eligible",
    type=float,
    default=60.0,
    validator=_validate_positive_float,
    group=GROUP_BACKGROUND,
)

ConfigParam(
    key="EMBEDDING_BACKFILL_BATCH_LIMIT",
    description="Max items per embedding backfill cycle on startup",
    type=int,
    default=50,
    validator=_validate_positive_int,
    group=GROUP_BACKGROUND,
)

# ── Memory tool — collection dedup thresholds ────────────────────────────────
#
# A candidate write is a duplicate if ANY signal hits its strict threshold,
# OR if any TWO signals hit their relaxed thresholds.  Three signals:
#   1. key TCR (token-containment ratio; lexical)
#   2. key embedding cosine (paraphrase)
#   3. content embedding cosine
# Strict catches obvious matches on one axis; relaxed catches weak-on-every-
# axis matches a single-signal gate would miss.

ConfigParam(
    key="MEMORY_DEDUP_KEY_TCR_STRICT",
    description="Strict key token-containment threshold for memory dedup",
    type=float,
    default=1.0,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

ConfigParam(
    key="MEMORY_DEDUP_KEY_TCR_RELAXED",
    description=(
        "Relaxed key token-containment threshold (catches abbreviation pairs "
        "like 'applied ai conference' / 'applied ai conf' at exactly 2/3)"
    ),
    type=float,
    default=0.65,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

ConfigParam(
    key="MEMORY_DEDUP_KEY_SIM_STRICT",
    description="Strict key embedding cosine threshold for memory dedup",
    type=float,
    default=0.90,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

ConfigParam(
    key="MEMORY_DEDUP_KEY_SIM_RELAXED",
    description="Relaxed key embedding cosine threshold for memory dedup",
    type=float,
    default=0.75,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

ConfigParam(
    key="MEMORY_DEDUP_CONTENT_SIM_STRICT",
    description="Strict content embedding cosine threshold for memory dedup",
    type=float,
    default=0.90,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

ConfigParam(
    key="MEMORY_DEDUP_CONTENT_SIM_RELAXED",
    description="Relaxed content embedding cosine threshold for memory dedup",
    type=float,
    default=0.75,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

# ── Browse tool ──────────────────────────────────────────────────────────────

ConfigParam(
    key="MAX_QUERIES",
    description="Max parallel queries per browse tool call",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_BROWSE,
)

ConfigParam(
    key="SEARCH_URL",
    description="Base URL for text searches (encoded query is appended)",
    type=str,
    default="https://duckduckgo.com/?q=",
    validator=_validate_non_empty_string,
    group=GROUP_BROWSE,
)

ConfigParam(
    key="DOMAIN_PERMISSION_MODE",
    description="Domain mode: restrict (prompt) or allow_all (auto-allow unknown)",
    type=str,
    default=DOMAIN_MODE_RESTRICT,
    validator=_validate_domain_mode,
    group=GROUP_BROWSE,
)

# ── Send tool — outbound message rate limiting ───────────────────────────────

ConfigParam(
    key="SEND_COOLDOWN_SECONDS",
    description=(
        "Flat cooldown in seconds between autonomous ``send_message`` calls. "
        "Bypassed when the user has replied since the agent's last send (the "
        "next send is conversational, not autonomous)."
    ),
    type=float,
    default=600.0,
    validator=_validate_positive_float,
    group=GROUP_SEND,
)

# ── Email tools ──────────────────────────────────────────────────────────────

ConfigParam(
    key="EMAIL_BODY_MAX_LENGTH",
    description="Max character length for email body content",
    type=int,
    default=4000,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="EMAIL_SEARCH_LIMIT",
    description="Max email results returned by the search_emails tool",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="EMAIL_LIST_LIMIT",
    description="Max email results returned by the list_emails tool",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="JMAP_REQUEST_TIMEOUT",
    description="Timeout in seconds for email API requests",
    type=float,
    default=30.0,
    validator=_validate_positive_float,
    group=GROUP_EMAIL,
)


class RuntimeParams:
    """Accessor for runtime-configurable parameters.

    Lookup chain: DB override → env override → ConfigParam.default.
    Supports attribute access with uppercase keys: config.runtime.IDLE_SECONDS
    """

    def __init__(
        self,
        db: Database | None = None,
        env_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._db = db
        self._env_overrides = env_overrides or {}

    def __getattr__(self, name: str) -> Any:
        key = name.upper()
        if key not in RUNTIME_CONFIG_PARAMS:
            raise AttributeError(f"No runtime config param: {name}")

        # 1. Check database
        if self._db is not None:
            db_value = self._get_db_value(key)
            if db_value is not None:
                return db_value

        # 2. Check env overrides (from Config.load)
        if key in self._env_overrides:
            return self._env_overrides[key]

        # 3. Fall back to default
        return RUNTIME_CONFIG_PARAMS[key].default

    def _get_db_value(self, key: str) -> Any:
        """Look up a runtime config override from the database."""
        assert self._db is not None  # Caller guards with `if self._db is not None`
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        with Session(self._db.engine) as session:
            result = session.exec(select(RuntimeConfig).where(RuntimeConfig.key == key)).first()

        if result is None:
            return None

        param = RUNTIME_CONFIG_PARAMS[key]
        try:
            return param.validator(result.value)
        except ValueError:
            return None
