"""Constants for Penny agent."""

from enum import StrEnum


class ChannelType(StrEnum):
    """Communication channel types."""

    SIGNAL = "signal"
    DISCORD = "discord"
    BROWSER = "browser"


class DomainPermissionValue(StrEnum):
    """Domain access permission states."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"


class ValidationReason(StrEnum):
    """Reasons a model response failed validation."""

    XML = "xml"
    EMPTY = "empty"
    REFUSAL = "refusal"
    HALLUCINATED_URLS = "hallucinated_urls"
    TOOL_PARSE_ERROR = "tool_parse_error"


class RunOutcome(StrEnum):
    """The first-class outcome of a collector cycle — one determination, stored
    on ``promptlog.run_outcome`` and surfaced everywhere (UI badge, the
    ``collector-runs`` log Penny reads, the auto-throttle).  Replaces the old
    ``run_success`` bool, which couldn't tell a clean no-op from real work.

    ``failed`` (errored / hit max steps / no ``done()``) ·
    ``no_work`` (completed cleanly, changed nothing) ·
    ``worked`` (completed and changed something — a write / update / move /
    delete / message) · ``cancelled`` (preempted by a foreground message — not
    a failure, not work; the throttle ignores it).
    """

    FAILED = "failed"
    NO_WORK = "no_work"
    WORKED = "worked"
    CANCELLED = "cancelled"


class ProgressEmoji(StrEnum):
    """Emojis used by ProgressTracker implementations to surface in-flight work.

    Channels that show progress as reactions on the user's message (e.g.
    SignalChannel) post one of these and morph between them as the agent's
    tool calls fire. Tools pick which one applies to their work via
    ``Tool.to_progress_emoji``.
    """

    THINKING = "\U0001f4ad"  # 💭 — initial state, before any tool calls
    SEARCHING = "\U0001f50d"  # 🔍 — running a text search
    READING = "\U0001f4d6"  # 📖 — reading a specific URL
    WORKING = "\u2699\ufe0f"  # ⚙️ — generic fallback for other tools


class ChatPromptType(StrEnum):
    """Prompt types emitted by ChatAgent flows. Logged to promptlog.prompt_type."""

    USER_MESSAGE = "user_message"
    VISION_MESSAGE = "vision_message"
    VISION_CAPTION = "vision_caption"


class PennyConstants:
    """All constants for the Penny agent."""

    class MessageDirection(StrEnum):
        """Direction of a logged message."""

        INCOMING = "incoming"
        OUTGOING = "outgoing"

    class MessageAuthor(StrEnum):
        """Conversational author of a message-log/run entry.

        A message has two conversational authors — the user (incoming) or Penny
        (outgoing); the message-log facades derive these from direction.
        ``COLLECTOR`` tags the synthesized ``collector-runs`` records.
        """

        USER = "user"
        PENNY = "penny"
        COLLECTOR = "collector"

    class SearchTrigger(StrEnum):
        """What triggered a search."""

        USER_MESSAGE = "user_message"
        PENNY_ENRICHMENT = "penny_enrichment"

    # Browse tool constants
    URL_BLOCKLIST_DOMAINS = (
        "play.google.com",
        "apps.apple.com",
    )
    BROWSE_RETRIES = 4
    BROWSE_RETRY_DELAY = 1.0
    BROWSE_REQUEST_TIMEOUT = 30.0

    # ``log_read`` window-mode look-back (seconds) for chat/schedule reads — the
    # "what just happened" range.  1 hour.
    LOG_READ_WINDOW_SECONDS = 3600

    # Connect timeout for the OpenAI-compatible LLM HTTP client.  Tunes only the
    # TCP-handshake / TLS deadline — the per-request read/write deadline is the
    # separately configurable ``LLM_TIMEOUT``.
    LLM_CONNECT_TIMEOUT_SECONDS = 5.0
    MAX_SEARCH_LINKS = 10
    BROWSE_SEARCH_HEADER = "## browse search: "
    BROWSE_PAGE_HEADER = "## browse: "
    BROWSE_ERROR_HEADER = "## browse error: "
    BROWSE_TITLE_PREFIX = "Title: "
    BROWSE_URL_PREFIX = "URL: "
    SECTION_SEPARATOR = "\n\n---\n\n"
    DISLIKE_FILTER_THRESHOLD = 0.8

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"

    # Zoho Mail API constants
    ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
    ZOHO_ACCOUNTS_URL = "https://mail.zoho.com/api/accounts"
    ZOHO_API_BASE = "https://mail.zoho.com/api"

    # Send queue — how often the drainer polls for a deliverable message.  The
    # actual send spacing is governed by SEND_COOLDOWN_SECONDS; this is just the
    # poll granularity (the drainer checks ~once a minute and sends at most one).
    SEND_QUEUE_DRAIN_INTERVAL = 60.0

    # Signal API connectivity validation
    SIGNAL_VALIDATE_MAX_ATTEMPTS = 12
    SIGNAL_VALIDATE_RETRY_DELAY = 5.0
    SIGNAL_VALIDATE_HTTP_TIMEOUT = 5.0

    # GitHub constants
    GITHUB_REPO_OWNER = "jaredlockhart"
    GITHUB_REPO_NAME = "penny"

    class PreferenceValence(StrEnum):
        """Valence of a user preference."""

        POSITIVE = "positive"
        NEGATIVE = "negative"

    class PreferenceSource(StrEnum):
        """How a preference was created."""

        MANUAL = "manual"
        EXTRACTED = "extracted"

    POSITIVE_REACTION_EMOJIS = frozenset(
        {
            "\U0001f44d",  # 👍
            "\u2764\ufe0f",  # ❤️
            "\U0001f525",  # 🔥
            "\U0001f44f",  # 👏
            "\U0001f60d",  # 😍
            "\U0001f64c",  # 🙌
            "\U0001f4af",  # 💯
            "\u2b50",  # ⭐
            "\U0001f60a",  # 😊
            "\U0001f389",  # 🎉
            "\U0001f4aa",  # 💪
            "\u2705",  # ✅
            "\U0001f929",  # 🤩
        }
    )

    # Vision constants
    VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    # Agent loop constants
    VISION_MAX_STEPS = 1
    RESPONSE_VALIDATION_RETRIES = 5
    # Minimum count of alphabetic characters for a model response to be
    # considered substantive. Catches garbage shapes — bare separators
    # (`---`), lone punctuation, emoji-only, runs of stars/dashes — without
    # enumerating them, while still allowing terse legit replies like "done"
    # or "yes". Anything below this is treated as EMPTY and retried.
    MIN_RESPONSE_LETTERS = 3
    TOOL_FAILURE_ABORT_THRESHOLD = 2

    # Thinking constants
    MIN_THOUGHT_WORDS = 50
    SUMMARY_URL_RETRIES = 2

    # Browser channel constants
    TOOL_REQUEST_TIMEOUT = 60.0
    PERMISSION_PROMPT_TIMEOUT = 60.0

    # System log memories (created by migration 0026) that the channel
    # adapter and browse tool side-effect-write to on every turn.
    MEMORY_USER_MESSAGES_LOG = "user-messages"
    MEMORY_COLLECTOR_RUNS_LOG = "collector-runs"

    # ``log_read`` cursor-mode batch bound — entries returned per call for a
    # collector.  Applies to every call: the first read (no cursor → most-recent
    # N, not the whole history) and later reads (the next N since the cursor).
    # The cursor advances by what was returned, so a backlog is worked through in
    # bounded batches across cycles instead of flooding one agentic loop with
    # hundreds of entries it can't reason over.
    LOG_READ_LIMIT = 10
    MEMORY_PENNY_MESSAGES_LOG = "penny-messages"
    MEMORY_BROWSE_RESULTS_LOG = "browse-results"

    # The system logs are populated exclusively by Python side-effects —
    # channel ingress/egress (``user-messages`` / ``penny-messages``), the
    # browse tool (``browse-results``), and the collector dispatcher
    # (``collector-runs``).  Agents may *read* them but must never append via
    # the ``log_append`` tool: a model-authored entry would corrupt the
    # conversation-turn reconstruction or forge an audit row.  Enforced in
    # ``LogAppendTool.execute``.
    SYSTEM_LOGS = frozenset(
        {
            MEMORY_USER_MESSAGES_LOG,
            MEMORY_PENNY_MESSAGES_LOG,
            MEMORY_BROWSE_RESULTS_LOG,
            MEMORY_COLLECTOR_RUNS_LOG,
        }
    )

    # System collections (created by migration 0027) that agents read and
    # write through the memory tool surface.
    MEMORY_UNNOTIFIED_THOUGHTS = "unnotified-thoughts"
    MEMORY_NOTIFIED_THOUGHTS = "notified-thoughts"

    # The self-correcting collector (seeded by migration 0055): reviews Penny's
    # own runs/messages against each collection's intent and rewrites drifted
    # extraction_prompts.  The collector gates the ``prompt_test`` dry-run tool
    # into the surface only for this collection's cycles.
    MEMORY_QUALITY_COLLECTION = "quality"

    # Centroid-proxy penalty applied during similarity-ranked retrieval:
    # ``adjusted = max(weighted, current_cos) - α * cos(entry, corpus_centroid)``.
    # The proxy is rank-equivalent to mean cosine to every other entry in the
    # same corpus (true centrality) up to an O(1/N) constant, so it acts as
    # the same centroid-magnet penalty without the O(N²) precompute — one
    # mean and one matrix-vector product per query, folded into ``_score``.
    MEMORY_RELEVANT_CENTRALITY_PENALTY = 0.5
    # Cluster-strength gate: top_head_mean / top_sample_mean must exceed this
    # for any entries to be returned — separates real clusters from flat
    # noise plateaus.
    MEMORY_RELEVANT_CLUSTER_GATE = 1.05
    # Cutoff is ``max(top_head_mean * RELATIVE_RATIO, ABSOLUTE_FLOOR)``.
    # The relative band adapts cluster width to cluster height; the
    # absolute floor is the empirical noise ceiling below which adjusted
    # scores are statistically indistinguishable from random.
    MEMORY_RELEVANT_RELATIVE_RATIO = 0.85
    MEMORY_RELEVANT_ABSOLUTE_FLOOR = 0.25
    # Number of top candidates averaged to estimate the cluster center
    # (numerator of the gate ratio).
    MEMORY_RELEVANT_GATE_HEAD_SIZE = 5
    # Number of top candidates averaged to estimate the broader noise floor
    # (denominator of the gate ratio).  Also doubles as the cold-start
    # threshold — below this we skip the gate and use just the absolute floor.
    MEMORY_RELEVANT_GATE_SAMPLE_SIZE = 20
    # Temporal neighbor expansion window for ``relevant``-mode log reads:
    # after similarity hits are selected, expand each by ±N minutes of
    # surrounding entries from the same log.  Captures conversational
    # follow-ups that share no entity overlap with the current message
    # but live in the same conversation as a real hit.
    MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES = 5

    # Low-information filter: entries in **log-shaped memories** with
    # fewer than this many word tokens are excluded from the similarity
    # corpus before scoring.  Empty strings, lone punctuation ("?", "…"),
    # stock greetings ("hi penny", "Hey! 😄"), and bare-URL fragments
    # otherwise dominate the cosine ranking on short keyword queries —
    # they don't carry topical content, but their tiny vocabulary
    # collides geometrically with any short anchor.
    #
    # Collections are NOT filtered: they have keyed entries where short
    # content is deliberate (the user's `likes` collection includes
    # entries like "anime", "cyberpunk", "video games").  Filtering them
    # would wipe out 75%+ of the user's actual stated preferences.
    MEMORY_RELEVANT_MIN_WORDS = 5
