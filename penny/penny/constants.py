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


class WriteGateOutcome(StrEnum):
    """The closed, deterministic outcome of one ``collection_write`` entry at the
    write chokepoint — the change-gate (#1587, epic #1554 via mini-epic #1562).

    Python computes it by comparing the written value against the stored baseline
    per key; it is never a model judgment, and it supersedes the old ``WriteOutcome``
    ("written"/"duplicate"/"rejected") three-way split.  The union is derived from
    what the write path actually does, one member per reachable state:

    ``NEW_KEY`` — the key did not exist; the entry was written (baseline set) ·
    ``KEY_EXISTS_CHANGED`` — the exact key existed with *different* content: the
    observed value changed, so the write gate **auto-refreshes the stored baseline
    itself** in place (through the update path — same validation, degeneracy screen,
    and ``last_written_by_run_id`` stamp), and the run's only remaining job is to
    notify.  No ``update_entry`` call is needed — the refresh already happened, so
    the next observation of the same value reads ``KEY_EXISTS_UNCHANGED`` (#1633) ·
    ``KEY_EXISTS_UNCHANGED`` — the exact key existed with *identical* content: the
    value has not changed, so there is nothing further to do — the watch's "no
    change" signal, which carries STOP semantics (see ``WRITE_GATE_STOP_REASONS``) ·
    ``DUPLICATE`` — the content (or a near key) collided with a *different* existing
    key via the similarity dedup disjunction ·
    ``DEGENERATE`` — the content was rejected as degenerate (blank, punctuation
    collapse, bare URL, bail-out phrase).

    ``UNEXPECTED`` is the honest escape label: a state the gate could not classify.
    The write path is total, so it is never produced today; it exists so consumers
    (the STOP table, the run-record render) match the union exhaustively and any
    future unclassified state flags for review rather than being forced into a wrong
    box (the visible-degradation principle).
    """

    NEW_KEY = "new_key"
    KEY_EXISTS_CHANGED = "key_exists_changed"
    KEY_EXISTS_UNCHANGED = "key_exists_unchanged"
    DUPLICATE = "duplicate"
    DEGENERATE = "degenerate"
    UNEXPECTED = "unexpected"


# The declared STOP table (#1587): which write-gate outcomes end a must-act
# (collector) run at the write chokepoint, mapped to the run's stamped stop reason.
#
# STAGE ① (the conservative core): only the unambiguous "value unchanged" case
# stops — the watch that looked and found nothing changed.  ``NEW_KEY`` /
# ``KEY_EXISTS_CHANGED`` never stop (an accumulator keeps going mid-script), and
# ``DUPLICATE`` / ``DEGENERATE`` are surfaced-but-recoverable, not clean stops.
# Later stages add per-collection gate shape as DATA that extends THIS table (e.g.
# an accumulator that stops when its whole batch is unchanged), never new loop
# code; a collection whose gate shape isn't declared yet falls back to this
# conservative default.  Membership here is what makes an outcome STOP-worthy.
WRITE_GATE_STOP_REASONS: dict[WriteGateOutcome, str] = {
    WriteGateOutcome.KEY_EXISTS_UNCHANGED: "the value was unchanged since the last observation",
}


# The write-gate outcomes that changed durable state — either a genuinely new key
# landed (``NEW_KEY``) or an existing key's baseline was auto-refreshed in place
# (``KEY_EXISTS_CHANGED``, #1633).  Read by the write path's change-notify and the
# tool result's ``mutated`` flag (the throttle's work signal), so "did this write
# change anything?" is one definition, not two that can drift.
WRITE_GATE_MUTATING_OUTCOMES: frozenset[WriteGateOutcome] = frozenset(
    {WriteGateOutcome.NEW_KEY, WriteGateOutcome.KEY_EXISTS_CHANGED}
)


class MutationAction(StrEnum):
    """The kind of registry-entity lifecycle change a mutation event records
    (#1560).  Each create / update / archive / unarchive of a collection writes
    one ``mutation_event`` row, so "when was this archived, and by what?" is a
    read, not a memory the model re-asserts from its own past narration."""

    CREATED = "created"
    UPDATED = "updated"
    ARCHIVED = "archived"
    UNARCHIVED = "unarchived"


class MutationActor(StrEnum):
    """Who caused a registry mutation (#1560).

    ``USER_RUN`` — a chat turn's run did it (the user asked, the model acted);
    the run id is the join key into the ledger.  ``SYSTEM`` — the scheduler did
    it with no model in the loop (a ``max_runs`` / ``expires_at`` archive reading
    columns), so its cause is a policy, carried in the event's detail note."""

    USER_RUN = "user-run"
    SYSTEM = "system"


class MutationEntityType(StrEnum):
    """The kind of registry entity a mutation event points at (#1560).

    Only ``COLLECTION`` today — post-#1556 the collection is the one background
    mechanism.  Declared as an enum so a future first-class ``skill`` (its
    versioning is #1562/#1471) slots in without reshaping the event."""

    COLLECTION = "collection"


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

    # The framework-internal per-call execution stamp (#1600).  Written onto each
    # framework-authored tool-RESULT message dict (beside ``content`` /
    # ``tool_call_id``) at execution time from the tool's structured
    # ``ToolResult.success`` — the STRUCTURAL "did this call work?" bit the run-end
    # skill extractor's certification reads instead of parsing result-frame prose.
    # It lives in
    # ``promptlog.messages`` (round-trips via ``json.dumps``), never in
    # ``promptlog.response`` (the model's verbatim output), and is stripped from the
    # wire in ``LlmClient._translate_messages`` so the model never sees it.
    TOOL_RESULT_SUCCESS_KEY = "tool_success"

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
    # The leading marker of a ``generate_image`` tool result — names the stored
    # media row's id so the id is an addressable part of the run's egress/media
    # trace (#1560).  Single source of truth: ``GenerateImageTool`` formats with
    # it and ``render_run_calls`` parses it back, so the two can't drift.
    GENERATED_IMAGE_RESULT_PREFIX = "Generated image #"
    SECTION_SEPARATOR = "\n\n---\n\n"
    DISLIKE_FILTER_THRESHOLD = 0.8

    # Current date/time anchor — the single "Current date and time: <stamp>" line
    # handed to the model, shared by the agent-loop envelope and every ad-hoc
    # one-shot LLM flow (the /profile parse, startup announcement, email
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
    # The cycle-terminator tool's name.  Only the collector shapes carry it; the
    # chat agent has no ``done`` tool, so failure envelopes that suggest calling
    # it gate that suggestion on the tool actually being registered.
    DONE_TOOL_NAME = "done"
    # The ledger identity of a browse micro-context extraction — a fresh
    # single-shot model call (content + instruction, no tools) that runs when a
    # ``browse`` carries an ``extract`` argument.  It logs its own promptlog rows
    # under this agent/prompt type so run traces attribute it honestly, while the
    # bulk page content never enters the parent run's context.
    BROWSE_EXTRACT_AGENT_NAME = "browse-extract"
    BROWSE_MICRO_CONTEXT_PROMPT_TYPE = "browse_micro_context"
    # How many recent conversational runs ``read_run_calls`` returns per batch —
    # bounded like every other cursored log read (``LOG_READ_LIMIT``).
    RUN_CALLS_LIMIT = 10
    # The type tag a rendered activity-log run anchor carries — ``run <id>`` (the
    # self-state header's run/mutation lines and ``render_run_calls``'s header emit
    # it verbatim).  ``get_event`` strips it to route the typed id to the run case,
    # so the token a surface renders IS the argument the tool takes (the n≤1 anchor
    # discipline: format and parse share this one constant, never a magic string).
    RUN_EVENT_PREFIX = "run "

    # ``log_read`` cursor-mode batch bound — entries returned per call for a
    # collector.  Applies to every call: the first read (no cursor → most-recent
    # N, not the whole history) and later reads (the next N since the cursor).
    # The cursor advances by what was returned, so a backlog is worked through in
    # bounded batches across cycles instead of flooding one agentic loop with
    # hundreds of entries it can't reason over.
    LOG_READ_LIMIT = 10
    # How many recent registry-change events ``memory_metadata`` renders in its
    # "Recent changes" block (``db.mutations.history``) — bounded like every other
    # history read so a config-change trail stays readable without flooding.
    RUN_HISTORY_RECORDS = 8
    # How many resolve-by-meaning hits ``find`` returns, best-first (#1558,
    # #1640).  Bounded like every other read so an ambiguous query surfaces the
    # top candidates without flooding the model; the model narrows further by
    # exact name or type.  All candidates are ranked; only the head is shown.
    FIND_MATCH_LIMIT = 5
    # Self-state header caps (#1555).  The chat agent's system prompt opens with a
    # deterministically-rendered header of Penny's own operational situation
    # (mechanisms · recent activity · the store map · durable user facts).  Each
    # section is bounded to a fixed number of newest/named rows so the ambient
    # budget stays flat as history grows; when a section overflows, a visible
    # "+N more — <tool>" tail names the fetch tool, so nothing is silently
    # dropped and n≤1 still holds (the overflow is one named call away).  These
    # are prompt-budget bounds with a recoverable overflow, NOT silent
    # truncations — deliberately generous; sizing is tunable later.
    SELF_STATE_MECHANISMS_LIMIT = 12
    SELF_STATE_ACTIVITY_LIMIT = 8
    SELF_STATE_MAP_LIMIT = 20
    # Keys named before the "…" tail in a multi-write run line's writes clause
    # (#1641): a run that wrote several entries shows the count plus this many
    # sample keys, so the clause stays one line.  Wholesale bound, tunable later.
    SELF_STATE_WRITES_KEY_SAMPLE = 2
    MEMORY_PENNY_MESSAGES_LOG = "penny-messages"
    MEMORY_BROWSE_RESULTS_LOG = "browse-results"
    # Typed-id separator for an entry handle (``<memory>#<id>``).  A browse
    # micro-context returns this handle to the main loop so the full stored page
    # content stays retrievable (``Memory.entry_by_id``) without the bulk body
    # ever entering the run context — the anchor discipline.
    MEMORY_HANDLE_SEPARATOR = "#"

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

    # The RETIRED self-correcting collector (seeded by migration 0055, archived by
    # #1569/migration 0089): it reviewed Penny's own runs against each collection's
    # intent and proposed prompt fixes.  It existed to correct drift in prompts
    # GENERATED FROM PROSE — the model improvising an extraction_prompt from a
    # description.  That authoring channel is gone (#1590/#1591): a collector's
    # prompt is now a deterministic render of a taught skill, and a wrong prompt is
    # fixed by the USER re-teaching the skill (re-teach REPLACES it; the collection
    # re-renders) — no prose-generation step left to review, so the reviewer
    # retired with the failure mode.  Retained here to keep the archived shell
    # hidden from the catalog (via ``SYSTEM_COLLECTIONS``).
    MEMORY_QUALITY_COLLECTION = "quality"

    # The retired skills collection (seeded by migration 0043, ARCHIVED by 0092,
    # #1624).  It carried prose skill recipes — a reconcile collector maintained
    # them by model judgment, and the self-state section rendered them as standing
    # rules.  Both duties are superseded by the structural path: there is exactly
    # ONE skills store now, the ``skill`` TABLE (taught #1590, instantiated #1591,
    # fired ambiently #1621, re-rendered #1620).  0092 archived the collection
    # (visible tombstone) and deleted its seeded rule entries (never demonstrated,
    # so they cannot enter the certified-by-execution table; needed behaviors get
    # re-taught live).  Retained here to keep the archived shell hidden from the
    # catalog (via ``SYSTEM_COLLECTIONS``), same as quality/notifier.
    MEMORY_SKILLS_COLLECTION = "skills"
    # The retired pub/sub notifier consumer (seeded by migration 0067, archived by
    # #1557): it drained every ``published`` collection's new entries and delivered
    # them to the user.  #1557 replaced it with emission-as-property (the ``notify``
    # flag + the run-time notify suffix), archiving the row.  Retained here to keep
    # the archived shell hidden from the catalog (via ``SYSTEM_COLLECTIONS``) and to
    # classify historical notifier-sent messages on the iOS surface.
    MEMORY_NOTIFIER_COLLECTION = "notifier"
    # Built-in preference / knowledge / inner-monologue extractors, seeded by
    # migration (0027/0031/0068) — Penny's own machinery, not collections the
    # user built.
    MEMORY_LIKES_COLLECTION = "likes"
    MEMORY_DISLIKES_COLLECTION = "dislikes"
    MEMORY_KNOWLEDGE_COLLECTION = "knowledge"
    MEMORY_THOUGHTS_COLLECTION = "thoughts"

    # Built-in framework collections, seeded by migration rather than created by
    # the user.  ``collection_catalog`` hides them: these are Penny's own machinery
    # (the preference/knowledge/thought extractors and the retired
    # skills/notification/self-correction shells), not collections the *user*
    # built, so the catalog — which surfaces user-built collections — leaves them
    # out.  Parallels ``SYSTEM_LOGS``.
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
