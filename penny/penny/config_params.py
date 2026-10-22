"""Runtime configuration parameter definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants

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
GROUP_IOS_ATTACHMENTS = "iOS Attachments"
GROUP_TOKEN_BUDGETS = "Token budgets"

# Ordered list for display
CONFIG_GROUPS: list[str] = [
    GROUP_CHAT,
    GROUP_BACKGROUND,
    GROUP_MEMORY,
    GROUP_BROWSE,
    GROUP_SEND,
    GROUP_EMAIL,
    GROUP_IOS_ATTACHMENTS,
    GROUP_TOKEN_BUDGETS,
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
    # Closed value set, when the param has one — the browser addon renders a
    # toggle or segmented editor from it instead of a free-text input (#2569).
    choices: tuple[str, ...] = ()

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


_SWITCH_CHOICES = ("0", "1")


def _validate_switch(value: str) -> int:
    """Parse a runtime switch without accepting other integer values."""
    if value not in _SWITCH_CHOICES:
        raise ValueError("must be 0 or 1")
    return int(value)


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
    description=(
        "Raw recent-conversation messages fetched as the pool behind the chat "
        "history window (HISTORY_WINDOW_TURNS user turns are then kept). Set "
        "well above the window so multi-message replies never starve it."
    ),
    type=int,
    default=60,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

ConfigParam(
    key="CHAT_HISTORY_TURNS",
    description=(
        "How many recent exchanges a chat turn sees as conversation history — "
        "the last N of the user's turns, each with its reply. Older turns stay "
        "reachable through log_read, find, and the self-state's recent activity."
    ),
    type=int,
    default=5,
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
        "the dispatcher checks which collection's schedule is most overdue and "
        "runs that one.  Should be shorter than the shortest gap any collection's "
        "schedule asks for — otherwise that collection waits up to TICK_INTERVAL "
        "past the moment its rule came round."
    ),
    type=float,
    default=30.0,
    validator=_validate_positive_float,
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

# ── Memory tool — fuzzy-name match gate ─────────────────────────────────
#
# The meaning leg of a did-you-mean suggestion (a mistyped collection/log name
# resolved against stored description embeddings) clears this cosine; the
# entry-identity ladder does not read it — identity is decided by rules, a
# shortlist, and the decision model (#2715), not by a similarity threshold.
ConfigParam(
    key="MEMORY_DEDUP_CONTENT_SIM_RELAXED",
    description="Cosine a fuzzy-name meaning match must clear for a did-you-mean suggestion",
    type=float,
    default=0.75,
    validator=_validate_unit_float,
    group=GROUP_MEMORY,
)

# ── Browse tool ──────────────────────────────────────────────────────────────

ConfigParam(
    key="MAX_QUERIES",
    description="Max queries one browse tool call fetches in parallel (the batch size)",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_BROWSE,
)

ConfigParam(
    key="BROWSE_MAX_QUERIES_PER_CALL",
    description=(
        "Total queries one browse tool call may run; queries past this come back "
        "in a dropped section. Fetched in batches of MAX_QUERIES."
    ),
    type=int,
    default=12,
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
    choices=(DOMAIN_MODE_RESTRICT, DOMAIN_MODE_ALLOW_ALL),
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

# ── iOS push preview ─────────────────────────────────────────────────────────

ConfigParam(
    key="PUSH_PREVIEW_CHARS",
    description=(
        "Max characters of an outgoing message the iOS push notification's preview carries"
    ),
    type=int,
    default=140,
    validator=_validate_positive_int,
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


for _key, _description in (
    ("IOS_AUTOMATIC_IMAGES", "Attach automatic images to new iOS replies"),
    ("IOS_CITED_PAGE_IMAGES", "Include automatic images from exact cited pages"),
    ("IOS_SAME_SITE_IMAGES", "Include automatic images from other pages on cited sites"),
    ("IOS_RELATED_IMAGES", "Include broader related images, including previously generated images"),
):
    ConfigParam(
        key=_key,
        description=_description,
        type=int,
        default=1,
        validator=_validate_switch,
        choices=_SWITCH_CHOICES,
        group=GROUP_IOS_ATTACHMENTS,
    )

# ── Token budgets — one per model-call kind (#2474) ──────────────────────────
#
# #2468's survey read 169 gemma eval runs' and production's own promptlog usage and
# set these at roughly p99 plus headroom — the browse extractor pinned BELOW its eval
# p99 on purpose, because the draws above it are #2453's own known spiral, not a
# healthy call this cap should allow.  Runtime-configurable like every other param
# here (DB override → env override → this default), so a threshold that turns out
# wrong is a `/config` edit, not a code change.  What a crossed cap DOES is the
# follow-up issues' (#2475/#2476): this one only sends the cap and reads
# ``finish_reason`` (see ``penny.llm.client.LlmClient.chat``).
ConfigParam(
    key="TOKEN_BUDGET_CHAT",
    description="Max completion tokens for a chat-turn call",
    type=int,
    default=16000,
    validator=_validate_positive_int,
    group=GROUP_TOKEN_BUDGETS,
)

ConfigParam(
    key="TOKEN_BUDGET_COLLECTOR",
    description="Max completion tokens for a collector-cycle call",
    type=int,
    default=16000,
    validator=_validate_positive_int,
    group=GROUP_TOKEN_BUDGETS,
)

ConfigParam(
    key="TOKEN_BUDGET_BROWSE_EXTRACT",
    description="Max completion tokens for a browse-extraction micro-context call",
    type=int,
    default=16000,
    validator=_validate_positive_int,
    group=GROUP_TOKEN_BUDGETS,
)

# Which ConfigParam names which call kind's budget.  A kind absent from this table — a
# future agent, or a test's made-up name — gets no cap: ``token_budget_for`` returns
# ``None`` rather than inventing a number nobody chose for it.  (The notify-composer
# kind retired with the composing job in #2656: the delivery turn is a chat turn and
# rides TOKEN_BUDGET_CHAT.)
TOKEN_BUDGET_PARAM_BY_AGENT_NAME: dict[str, str] = {
    PennyConstants.CHAT_AGENT_NAME: "TOKEN_BUDGET_CHAT",
    PennyConstants.COLLECTOR_AGENT_NAME: "TOKEN_BUDGET_COLLECTOR",
    PennyConstants.BROWSE_EXTRACT_AGENT_NAME: "TOKEN_BUDGET_BROWSE_EXTRACT",
}


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
        result = self._db.config.get(key)

        if result is None:
            return None

        param = RUNTIME_CONFIG_PARAMS[key]
        try:
            return param.validator(result.value)
        except ValueError:
            return None


def token_budget_for(runtime: RuntimeParams, agent_name: str | None) -> int | None:
    """The configured token budget for one call kind (#2474), or ``None`` when
    ``agent_name`` names no kind ``TOKEN_BUDGET_PARAM_BY_AGENT_NAME`` knows — an
    unrecognised or absent kind sends no cap, the honest "we don't know" rather
    than a guessed number.  Reads through ``runtime`` the same way every other
    runtime param does, so a budget set via ``/config`` or the environment takes
    effect the same way ``MAX_STEPS`` does."""
    key = TOKEN_BUDGET_PARAM_BY_AGENT_NAME.get(agent_name) if agent_name else None
    if key is None:
        return None
    return int(getattr(runtime, key))
