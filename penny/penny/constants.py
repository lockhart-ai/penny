"""Constants for Penny agent."""

from enum import StrEnum


class ChannelType(StrEnum):
    """Communication channel types."""

    SIGNAL = "signal"
    DISCORD = "discord"
    BROWSER = "browser"
    IOS = "ios"


class DomainPermissionValue(StrEnum):
    """Domain access permission states."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"


class RunOutcome(StrEnum):
    """The first-class outcome of a collector cycle — one determination, stored
    on ``promptlog.run_outcome`` and surfaced everywhere (UI badge, the
    ``collector-runs`` log Penny reads, the auto-throttle).  Replaces the old
    ``run_success`` bool, which couldn't tell a clean no-op from real work.

    ``failed`` (errored, or ended with no successful ``done()`` AND did no real
    work — a true bail) ·
    ``no_work`` (completed cleanly, changed nothing) ·
    ``worked`` (completed and changed something — a write / update / move /
    delete / message) ·
    ``incomplete`` (did real work but never closed with a successful ``done()`` —
    typically hit max steps mid-cycle; the work is durable so the read cursor
    still advances and the throttle counts it as productive, but it's flagged
    distinctly so a too-tight step budget stays visible) ·
    ``cancelled`` (preempted by a foreground message — not a failure, not work;
    the throttle ignores it).
    """

    FAILED = "failed"
    NO_WORK = "no_work"
    WORKED = "worked"
    INCOMPLETE = "incomplete"
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

    # Egress image matching (side-channel media attach).  When an outgoing message
    # links no source page, we fall back to embedding-nearest and pick uniformly
    # at random among the top-K so a centroid "magnet" image can't repeat on
    # consecutive messages.  Exact-URL and same-domain matches are deterministic
    # (the cited page's own image is the right one) — jitter applies only here.
    MEDIA_MATCH_JITTER_TOPK = 5

    # ``log_read`` window-mode look-back (seconds) for chat/schedule reads — the
    # "what just happened" range.  1 hour.
    LOG_READ_WINDOW_SECONDS = 3600

    # Connect timeout for the OpenAI-compatible LLM HTTP client.  Tunes only the
    # TCP-handshake / TLS deadline — the per-request read/write deadline is the
    # separately configurable ``LLM_TIMEOUT``.
    LLM_CONNECT_TIMEOUT_SECONDS = 5.0
    # Total deadline for the lightweight model-list preflight probes, so the
    # timeout budget is explicit and consistent with the SDK path rather than
    # riding on httpx's implicit default.
    LLM_MODEL_LIST_TIMEOUT_SECONDS = 10.0
    # Provider-specific endpoint some OpenAI-compatible backends (e.g. openrouter)
    # use to list embedding-capable models that ``/v1/models`` omits.
    LLM_EMBEDDING_MODELS_ENDPOINT = "/v1/embeddings/models"
    MAX_SEARCH_LINKS = 10
    BROWSE_SEARCH_HEADER = "## browse search: "
    BROWSE_PAGE_HEADER = "## browse: "
    BROWSE_ERROR_HEADER = "## browse error: "
    # Disclosure header for queries dropped past the per-call cap.  Deliberately
    # distinct from the ok/error headers so the run-health I/O tally never counts
    # a dropped-queries note as a browse ok/failure.
    BROWSE_DROPPED_HEADER = "## browse dropped: "
    BROWSE_TITLE_PREFIX = "Title: "
    BROWSE_URL_PREFIX = "URL: "
    SECTION_SEPARATOR = "\n\n---\n\n"
    DISLIKE_FILTER_THRESHOLD = 0.8

    # Current date/time anchor — the single "Current date and time: <stamp>" line
    # handed to the model, shared by the agent-loop envelope and every ad-hoc
    # one-shot LLM flow (schedule/profile parse, startup announcement, email
    # summarize).  Rendered via ``datetime_utils.current_datetime_line``.
    CURRENT_DATETIME_FORMAT = "%A, %B %d, %Y at %I:%M %p %Z"
    CURRENT_DATETIME_PREFIX = "Current date and time: "

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
    # How many times the loop re-rolls a degenerate (punctuation-collapse) model
    # output before throwing out the whole run.  The bad output is DISCARDED, never
    # appended — a re-roll on the unchanged context, since the collapse is a
    # sampling artifact that a fresh draw usually clears.  Kept small: each re-roll
    # is a full model call, and a run that collapses 3× in a row is stuck (the
    # context is too large — see the ~4K-token cliff) and better abandoned than fed
    # poison downstream.
    DEGENERATE_REROLL_ATTEMPTS = 3
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
    PERMISSION_PROMPT_TIMEOUT = 60.0
    # Max inbound WebSocket frame size for the browser channel.  The websockets
    # default is 1 MiB, which a browse tool response overflows once it carries a
    # page's base64 image data URI (observed ~1.7 MB) — the library then rejects
    # the frame with a 1009 "message too big" close, dropping the connection
    # mid-browse.  16 MiB leaves generous headroom for image-bearing responses.
    BROWSER_WS_MAX_FRAME_BYTES = 16 * 1024 * 1024
    # A tool connection counts as live only while the addon keeps sending its
    # app-level heartbeat (HEARTBEAT_INTERVAL_MS = 15s in the extension).  Past
    # this window with no heartbeat the socket is treated as dead even if TCP is
    # still open: Firefox answers the WebSocket ping/pong at the network layer
    # while a suspended background script never processes the tool request, so
    # the protocol-level ping cannot detect it.  ~3 missed beats of slack.
    BROWSER_HEARTBEAT_TIMEOUT_SECONDS = 45.0

    # System log memories (created by migration 0026) that the channel
    # adapter and browse tool side-effect-write to on every turn.
    MEMORY_USER_MESSAGES_LOG = "user-messages"
    MEMORY_COLLECTOR_RUNS_LOG = "collector-runs"
    # ``promptlog.agent_name`` stamped on every chat-agent prompt — the structural
    # marker the ``read_run_calls`` tool uses to find conversational runs (a turn's
    # user message → the tool calls it drove).  Mirrors ``ChatAgent.name``.
    CHAT_AGENT_NAME = "chat"
    # How many recent conversational runs ``read_run_calls`` returns per batch —
    # bounded like every other cursored log read (``LOG_READ_LIMIT``).
    RUN_CALLS_LIMIT = 10

    # ``log_read`` cursor-mode batch bound — entries returned per call for a
    # collector.  Applies to every call: the first read (no cursor → most-recent
    # N, not the whole history) and later reads (the next N since the cursor).
    # The cursor advances by what was returned, so a backlog is worked through in
    # bounded batches across cycles instead of flooding one agentic loop with
    # hundreds of entries it can't reason over.
    LOG_READ_LIMIT = 10
    # How many of ONE collector's recent runs the ``collector_run_history`` tool
    # returns — full rendered run records (counts line + flags + tool trace), not
    # the one-line ``done`` summaries the cycle-start own-history block shows.
    # Fixed in Python (the model passes the collector name, never the count) and
    # bounded below ``COLLECTOR_RUN_HISTORY`` (10) because each record is heavy:
    # enough cycles to judge "one-off vs. persistent pattern" without flooding.
    RUN_HISTORY_RECORDS = 8
    # Cold-start window for a consumer draining a published collection it has no
    # cursor for yet.  Rather than replay the whole backlog (a flood) or skip it
    # entirely, the consumer starts one week back — the user-chosen window: the
    # last week's finds get delivered once, anything older counts as already-seen.
    PUBLISHED_COLDSTART_LOOKBACK_SECONDS = 7 * 86400
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

    # The self-correcting collector (seeded by migration 0055): reviews Penny's
    # own runs/messages against each collection's intent and rewrites drifted
    # extraction_prompts directly.
    MEMORY_QUALITY_COLLECTION = "quality"

    # The skills collector (seeded by migration 0043): distils reusable workflow
    # patterns from the real collections that exist, surfaced to chat via recall.
    MEMORY_SKILLS_COLLECTION = "skills"
    # The pub/sub notifier consumer (seeded by migration 0067): drains every
    # ``published`` collection's new entries and delivers them to the user.
    MEMORY_NOTIFIER_COLLECTION = "notifier"
    # Built-in preference / knowledge / inner-monologue extractors, seeded by
    # migration (0027/0031/0068) — Penny's own machinery, not collections the
    # user built.
    MEMORY_LIKES_COLLECTION = "likes"
    MEMORY_DISLIKES_COLLECTION = "dislikes"
    MEMORY_KNOWLEDGE_COLLECTION = "knowledge"
    MEMORY_THOUGHTS_COLLECTION = "thoughts"

    # Built-in framework collections, seeded by migration rather than created by
    # the user.  ``collection_catalog`` hides them: the skills collector distils
    # reusable patterns from the collections the *user* builds, and these are
    # Penny's own machinery (the skills/self-correction/notification loops and
    # the preference/knowledge/thought extractors) — distilling skills from them
    # would only mint meta-noise.  Parallels ``SYSTEM_LOGS``.
    SYSTEM_COLLECTIONS = frozenset(
        {
            MEMORY_SKILLS_COLLECTION,
            MEMORY_QUALITY_COLLECTION,
            MEMORY_NOTIFIER_COLLECTION,
            MEMORY_LIKES_COLLECTION,
            MEMORY_DISLIKES_COLLECTION,
            MEMORY_KNOWLEDGE_COLLECTION,
            MEMORY_THOUGHTS_COLLECTION,
        }
    )

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

    # Bounded neighbor expansion for ``relevant``-mode LOG recall.  Without a
    # cap the expansion is unbounded: each hybrid hit pulls in *every* entry
    # within ±MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES, so a dense burst of
    # messages around a single hit can drag dozens of entries (often huge
    # replies) into the prompt.  We take at most MEMORY_NEIGHBOR_HIT_LIMIT
    # hybrid hits, then expand each to at most MEMORY_NEIGHBOR_PER_HIT entries
    # (the hit plus its nearest-in-time neighbors inside the window) — a hard
    # ceiling of HIT_LIMIT × PER_HIT entries.  Logs only; collections never
    # expand.
    MEMORY_NEIGHBOR_HIT_LIMIT = 3
    MEMORY_NEIGHBOR_PER_HIT = 3

    # Length normalization for the lexical leg of hybrid recall ranking.
    # ``lexical_coverage`` is the IDF-weighted fraction of the query's tokens an
    # entry contains — but a long entry has a big token set, so it coincidentally
    # covers more of *any* query and wins the lexical leg on surface area alone
    # (the classic long-document bias BM25 corrects with a length term).
    # Coverage is divided by ``(1-b) + b*sqrt(len/avglen)``: a sub-linear penalty
    # that demotes coincidental long-doc matches without unseating genuinely
    # on-topic long entries (near-full coverage + strong cosine keep their slot).
    # sqrt compresses the wide length spread of message logs, so the penalty is
    # active where lengths vary (logs) and ~flat — effectively inert — where they
    # are uniform (collections).  b=0 disables it.
    MEMORY_LEXICAL_LENGTH_B = 0.5

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
