"""Constants for Penny agent."""

from __future__ import annotations

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
    ``incomplete`` (did real work but never closed with a successful ``done()``).
    **No collector cycle records this any more (#1936)**: a cycle that never reached
    a healthy end has its writes thrown away, so it changed nothing durable and is
    the ``failed`` bail it now honestly is.  The member stays for the historical
    rows that carry it and the ``⚠ INCOMPLETE`` flag that renders them ·
    ``cancelled`` (preempted by a foreground message — not a failure, not work;
    the throttle ignores it).
    """

    FAILED = "failed"
    NO_WORK = "no_work"
    WORKED = "worked"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"


class CycleTrigger(StrEnum):
    """What set a collector cycle running (#1939).

    ``cadence`` — the dispatcher picked the collection up because its schedule
    said so.  Nobody is waiting; a notification it produces rides the full
    autonomous-send cooldown, and its attempts at one fire share one retry budget.

    ``on_demand`` — the user pressed "run this now" in an addon surface
    (``Collector.run_for``).  Someone is sitting in front of the result, so what
    the cycle sends is conversational rather than autonomous, and its attempts are
    the user's rather than the schedule's.

    Read in three places, which is why it is one word rather than three flags: the
    collector keys its per-occurrence retry budget on it, the send queue stores it
    on the row it queues (``send_queue.origin``), and the drainer reads that row to
    decide which delivery lane the message is in.
    """

    CADENCE = "cadence"
    ON_DEMAND = "on_demand"


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
    ``DUPLICATE_UNCHANGED`` — the dedup disjunction matched a *different* existing key
    AND the strict CONTENT signal fired: the stored entry already says what this write
    says, however either is worded, so the cycle observed nothing new.  The same no-news
    as ``KEY_EXISTS_UNCHANGED``, reached under a different key, and STOP-worthy for the
    same reason (#1919) ·
    ``DUPLICATE`` — the content (or a near key) collided with a *different* existing
    key WITHOUT the content signal firing strictly: the write carries a DIFFERENT value
    for something already stored, which is news, so it keeps the recoverable rejection
    that binds the matched key into an ``update_entry`` call and never stops the run ·
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
    DUPLICATE_UNCHANGED = "duplicate_unchanged"
    DUPLICATE = "duplicate"
    DEGENERATE = "degenerate"
    UNEXPECTED = "unexpected"


# The declared STOP table (#1587): which write-gate outcomes end a must-act
# (collector) run at the write chokepoint, mapped to the run's stamped stop reason.
#
# The rule is the unambiguous "value unchanged" case, under EITHER of the two ways a
# write can arrive at an entry that already holds its value: the exact key
# (``KEY_EXISTS_UNCHANGED``) or the dedup disjunction's strict CONTENT match under a
# different key (``DUPLICATE_UNCHANGED``, #1919).  Both are a watch that looked and found
# nothing changed, so both close the cycle at the chokepoint.
#
# The second one is here because reaching the same entry under a reworded key made the
# same no-news read as a recoverable rejection instead: the cycle carried on, and every
# measured sample "recovered" by writing a DIFFERENT entry — rational, since the surface
# it was answering asked it to try something else.  Splitting the outcome is what makes
# no-news structural rather than something the model is asked not to do.
#
# ``NEW_KEY`` / ``KEY_EXISTS_CHANGED`` never stop (an accumulator keeps going
# mid-script), and ``DUPLICATE`` / ``DEGENERATE`` stay surfaced-but-recoverable — a
# divergent-value collision is NEWS about a stored entry, so stopping on it would
# silence the change it carries.  Later stages add per-collection gate shape as DATA that
# extends THIS table (e.g. an accumulator that stops when its whole batch is unchanged),
# never new loop code.  Membership here is what makes an outcome STOP-worthy.
WRITE_GATE_STOP_REASONS: dict[WriteGateOutcome, str] = {
    WriteGateOutcome.KEY_EXISTS_UNCHANGED: "the value was unchanged since the last observation",
    WriteGateOutcome.DUPLICATE_UNCHANGED: (
        "the value was already recorded, under a different key, since the last observation"
    ),
}


# ``COLLECTOR_COVERED_REASON`` is RETIRED (#1916, with the coverage exit it named).  A
# cycle closes on ``done()`` again, and the render tells a clean close from an abandoned
# one by the ledger's own ``done`` record rather than by a stamped phrase — so a clean
# close carries no reason of its own, and the record's header falls back to the outcome
# enum exactly as it did before #1911.  What a clean close DOES carry, when the
# collection notifies, is what telling the user came to (``NOTIFICATION_NOTES``).

# The reason a cycle gets when its stored program names no call the collector could run
# — a purely prose prompt.  Its surface is the terminator alone, so there is nothing it
# could carry out, and the record says so rather than leaving the state to be diagnosed
# by exclusion (visible degradation).
COLLECTOR_UNREADABLE_PROGRAM_REASON = (
    "cycle ended without a done() call, and its program names no runnable call to read "
    "completion from"
)


class CycleEnd(StrEnum):
    """How a collector cycle ENDED — the closed set that decides both whether the
    cycle's staged entry writes land and whether its schedule occurrence is spent
    (#1936).

    One set read twice, because they are one question: a cycle that reached its
    conclusion did the job its fire was for, and a cycle that did not must be retried
    from the state it started in.  Declared as data like ``WRITE_GATE_STOP_REASONS`` —
    a later shape joins the table, not the code.

    ``STOPPED`` — a write-gate STOP closed the cycle at the chokepoint.  An early
    clean no-news exit IS an end point, so whatever landed before it commits ·
    ``CLOSED_QUIET`` — closed with ``done()`` on a collection that does not notify ·
    ``CLOSED_NOTIFIED`` — closed with ``done()``, and the notification was QUEUED ·
    ``CLOSED_DECLINED`` — closed with ``done()``, and delivery was deliberately
    declined (the user is muted, or there is no registered recipient): a retry cannot
    change either, so the cycle ended and the decline is stamped on the run record ·
    ``NOTIFICATION_NOT_DRAWN`` — closed, but the compose micro-context exhausted its
    rerolls, so the cycle never reached the point of reporting anything ·
    ``ABORTED`` — a model call died mid-run (#1909) ·
    ``CANCELLED`` — foreground activity preempted the cycle ·
    ``UNFINISHED`` — every other way of stopping short: the step cap, a model that
    trailed off, an exception out of the loop.  Not a named cause but the same state,
    and the ruling covers it in as many words — a cycle that "never reaches its
    healthy conclusion is retried until it does".
    """

    STOPPED = "stopped"
    CLOSED_QUIET = "closed_quiet"
    CLOSED_NOTIFIED = "closed_notified"
    CLOSED_DECLINED = "closed_declined"
    NOTIFICATION_NOT_DRAWN = "notification_not_drawn"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    UNFINISHED = "unfinished"


# The HEALTHY ends (#1936, code-owner ratified).  A cycle that ended one of these four
# ways reached its conclusion: its staged entry writes COMMIT, in one short transaction,
# and its schedule occurrence is SPENT.
#
# Membership is what makes an end healthy — everything outside this table discards and
# leaves the occurrence due for #1935's bounded retry, so an end nobody enumerated is
# retried rather than half-persisted.  That default is the ruling's own: a partial run
# must not persist state later cycles have to reason about.
HEALTHY_CYCLE_ENDS: frozenset[CycleEnd] = frozenset(
    {
        CycleEnd.STOPPED,
        CycleEnd.CLOSED_QUIET,
        CycleEnd.CLOSED_NOTIFIED,
        CycleEnd.CLOSED_DECLINED,
    }
)


# Why each UNHEALTHY end leaves its occurrence due — the line the retry logs (#1935's
# ``_retry_reason``, now read off this table).  ``ABORTED`` renders the abort's own
# cause when the run carries one (#1909) and falls back to this line when it does not.
# A foreground message cancels the cycle wherever it happens to be — the timing is the
# user's, not the collection's — so each of these, attempted a moment later, is a
# different draw.
CYCLE_END_RETRY_REASONS: dict[CycleEnd, str] = {
    CycleEnd.NOTIFICATION_NOT_DRAWN: "no message the cycle could send was ever written",
    CycleEnd.ABORTED: "the cycle's model call died mid-run",
    CycleEnd.CANCELLED: "the cycle was preempted by foreground activity",
    CycleEnd.UNFINISHED: "the cycle stopped short of a close",
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
    read, not a memory the model re-asserts from its own past narration.

    ``DELETED`` is the SKILL registry's retirement (#1902): that table is
    versionless and carries no archived flag, so a routine a bail takes back
    LEAVES rather than becoming a tombstone row — and the event is the only
    trace there is.  Naming it as its own action rather than reusing
    ``ARCHIVED`` keeps the render honest: a reader following an archived
    collection finds it, a reader following a deleted routine does not."""

    CREATED = "created"
    UPDATED = "updated"
    ARCHIVED = "archived"
    UNARCHIVED = "unarchived"
    DELETED = "deleted"


class MutationActor(StrEnum):
    """Who caused a registry mutation (#1560).

    ``USER_RUN`` — a chat turn's run did it (the user asked, the model acted);
    the run id is the join key into the ledger.  ``SYSTEM`` — the scheduler did
    it with no model in the loop (a ``max_runs`` / ``expires_at`` archive reading
    columns), so its cause is a policy, carried in the event's detail note."""

    USER_RUN = "user-run"
    SYSTEM = "system"


class TransitionCause(StrEnum):
    """What moved the conversation state machine (#1706).

    ``CLASSIFIER`` — a scoped micro-context draw over the current state's
    out-edges decided it; the row carries the run whose promptlog holds the
    draw.  ``STRUCTURAL`` — no model was in the loop (the post-apply reset,
    which is a fact about the edge table, not a judgment).  The same split the
    mutation ledger draws between a user-run and a system actor: it is what
    lets per-edge classifier accuracy be scored over production history without
    structural moves inflating it.
    """

    CLASSIFIER = "classifier"
    STRUCTURAL = "structural"


class MutationEntityType(StrEnum):
    """The kind of registry entity a mutation event points at (#1560).

    ``COLLECTION`` is the background mechanism post-#1556.  ``SKILL`` is the
    second customer the enum was declared for (#1902): a round that ends in idle
    takes its routine back — deleted outright when the round minted the name,
    reverted to its pre-round content when the round was re-teaching one that
    already stood — and the registry is AMBIENT, so a routine vanishing or
    reverting between turns has to appear in the recent-changes render like any
    other configuration change."""

    COLLECTION = "collection"
    SKILL = "skill"


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
    ROLLING = "\U0001f3b2"  # 🎲 — making a fair random choice
    WORKING = "\u2699\ufe0f"  # ⚙️ — generic fallback for other tools


class PermissionResolution(StrEnum):
    """How a domain-permission prompt ended.

    A prompt is broadcast to every channel and answered on whichever one the
    user reaches first (or not at all), so every other channel has to be told
    it is over — and *how* it ended is what each channel acknowledges. The
    third member is a real outcome, not an error: the prompt simply expired.
    """

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    TIMED_OUT = "timed_out"

    @classmethod
    def from_decision(cls, allowed: bool | None) -> PermissionResolution:
        """The resolution a prompt's outcome (``None`` = no answer) stands for."""
        if allowed is None:
            return cls.TIMED_OUT
        return cls.ALLOWED if allowed else cls.BLOCKED

    @property
    def emoji(self) -> str:
        """The mark a channel acknowledges this resolution with."""
        return PERMISSION_RESOLUTION_EMOJIS[self]


# The acknowledgment mark per resolution. Signal has no silent removal — deleting
# the prompt leaves a "This message was deleted" tombstone on every client — so a
# resolved prompt is marked in place with one of these instead.
PERMISSION_RESOLUTION_EMOJIS: dict[PermissionResolution, str] = {
    PermissionResolution.ALLOWED: "\u2705",  # ✅
    PermissionResolution.BLOCKED: "\u274c",  # ❌
    PermissionResolution.TIMED_OUT: "\u23f3",  # ⏳ — distinct from either answer
}


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

    # Email-rule provider tag — the ``email_rule.provider`` value the Zoho plugin
    # writes and filters on (a future non-Zoho email backend gets its own tag).
    PROVIDER_ZOHO = "zoho"

    # Zoho Mail API constants
    ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
    ZOHO_ACCOUNTS_URL = "https://mail.zoho.com/api/accounts"
    ZOHO_API_BASE = "https://mail.zoho.com/api"

    # Zoho Calendar API constants
    ZOHO_CALENDAR_API_BASE = "https://calendar.zoho.com/api/v1"
    # Calendar endpoint path fragments (relative to ZOHO_CALENDAR_API_BASE);
    # the ``{...}`` placeholder forms are filled with ``str.format(...)``.
    ZOHO_CALENDAR_CALENDARS_PATH = "/calendars"
    ZOHO_CALENDAR_EVENTS_PATH = "/calendars/{caluid}/events"
    ZOHO_CALENDAR_EVENT_PATH = "/calendars/{caluid}/events/{event_uid}"
    ZOHO_CALENDAR_FREEBUSY_PATH = "/calendars/freebusy"
    ZOHO_CALENDAR_FREESLOTS_PATH = "/freebusy/freeslots"

    # Zoho Projects API constants (v3)
    ZOHO_PROJECTS_API_BASE = "https://projectsapi.zoho.com/api/v3"
    # Projects endpoint path fragments (relative to ZOHO_PROJECTS_API_BASE);
    # the ``{...}`` placeholder forms are filled with ``str.format(...)``.
    ZOHO_PROJECTS_PORTALS_PATH = "/portals"
    ZOHO_PROJECTS_PROJECTS_PATH = "/portal/{portal_id}/projects"
    ZOHO_PROJECTS_TASKLISTS_PATH = "/portal/{portal_id}/projects/{project_id}/tasklists"
    ZOHO_PROJECTS_TASKS_PATH = "/portal/{portal_id}/projects/{project_id}/tasks"
    ZOHO_PROJECTS_TASK_PATH = "/portal/{portal_id}/projects/{project_id}/tasks/{task_id}"
    # Default task list a task is filed under when the caller names none.
    ZOHO_PROJECTS_DEFAULT_TASKLIST = "General"

    # InvoiceNinja v5 API endpoint path fragments (relative to the configured
    # base URL); the ``{...}`` placeholder form is filled with ``str.format(...)``.
    INVOICENINJA_HEALTH_CHECK_PATH = "/api/v1/health_check"
    INVOICENINJA_INVOICES_PATH = "/api/v1/invoices"
    INVOICENINJA_EXPENSES_PATH = "/api/v1/expenses"
    INVOICENINJA_EXPENSE_PATH = "/api/v1/expenses/{expense_id}"
    INVOICENINJA_EXPENSE_CATEGORIES_PATH = "/api/v1/expense_categories"
    INVOICENINJA_EXPENSE_CATEGORY_PATH = "/api/v1/expense_categories/{category_id}"

    # Default per-request timeout (seconds) for the Zoho Calendar + Projects
    # HTTP clients — the value both hardcoded before it was named here.
    ZOHO_CLIENT_TIMEOUT = 30.0

    # Send queue — how often the drainer polls for a deliverable message.  The
    # actual send spacing is governed by SEND_COOLDOWN_SECONDS; this is just the
    # poll granularity (the drainer checks ~once a minute and sends at most one).
    SEND_QUEUE_DRAIN_INTERVAL = 60.0

    # Signal API connectivity validation
    SIGNAL_VALIDATE_MAX_ATTEMPTS = 12
    SIGNAL_VALIDATE_RETRY_DELAY = 5.0
    SIGNAL_VALIDATE_HTTP_TIMEOUT = 5.0

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
    # The name of the RETIRED terminator tool (#1911).  No surface carries it: a
    # collector cycle ends when its program's calls are covered, and a chat turn ends
    # by replying.  The constant survives because the NAME still appears where no live
    # tool does — the ``{"name": "done"}`` JSON envelope the model emits from prior,
    # which the invalid-draw guard discards by name, and the ledger rows written before
    # the retirement, which the run renders and the skill extractor still have to
    # recognise.  History is never rewritten.
    DONE_TOOL_NAME = "done"
    # The ledger identity of a browse micro-context extraction — a fresh
    # single-shot model call (content + instruction, no tools) that runs when a
    # ``browse`` carries an ``extract`` argument.  It logs its own promptlog rows
    # under this agent/prompt type so run traces attribute it honestly, while the
    # bulk page content never enters the parent run's context.
    BROWSE_EXTRACT_AGENT_NAME = "browse-extract"
    BROWSE_MICRO_CONTEXT_PROMPT_TYPE = "browse_micro_context"
    # The ledger identity of a run-end skill-naming micro-context (#1665) — the
    # SECOND customer of the micro-context machinery.  After a qualifying chat run
    # is distilled, one single-shot model call writes a GENERIC name + description
    # for the routine (the tagged NAME:/DESCRIPTION: contract); it logs its own
    # promptlog rows under this agent/prompt type so run traces attribute it.
    SKILL_NAMING_AGENT_NAME = "skill-namer"
    SKILL_NAMING_PROMPT_TYPE = "skill_naming"
    # The ledger identity of a conversation-state classification (#1706) — the
    # THIRD customer of the micro-context machinery.  Once per incoming message a
    # single-shot model call picks the machine's next state from the CURRENT
    # state's out-edges (the tagged STATE: contract); it logs its own promptlog
    # rows under this agent/prompt type so every transition is attributable and
    # replayable from production history.
    STATE_CLASSIFIER_AGENT_NAME = "state-classifier"
    STATE_CLASSIFIER_PROMPT_TYPE = "state_classifier"
    # The ledger identity of a run-end skill-FRAMING micro-context (#1830) — the
    # FOURTH customer of the micro-context machinery.  Beside the labeller, and from
    # the user's ask ALONE, one single-shot model call writes the routine's public
    # INTERFACE: its generic name, its one-line description, and the parameter(s) the
    # user would have to say again (the tagged NAME:/DESCRIPTION:/PARAMETER contract).
    # Its own agent/prompt type, so a run trace shows the two draws as the two
    # questions they are — implementation and interface, sharing no evidence (#1824).
    SKILL_FRAME_AGENT_NAME = "skill-framer"
    SKILL_FRAME_PROMPT_TYPE = "skill_frame"
    # The ledger identity of a skill-BINDING micro-context (#1867) — the FIFTH
    # customer of the micro-context machinery.  Given a routine that ALREADY exists
    # and the user's own words asking for it on a new occasion, one single-shot
    # model call fills each declared parameter from those words (the tagged
    # VALUE/MISSING contract).  Its own agent/prompt type, so the routing draw and
    # the filling draw stay the two separate questions they are (#1803).
    SKILL_BIND_AGENT_NAME = "skill-binder"
    SKILL_BIND_PROMPT_TYPE = "skill_bind"
    # The ledger identity of the NOTIFY-COMPOSING micro-context (#1911) — the SIXTH
    # customer, and the first that closes a collector cycle rather than a chat turn.
    # Its own agent/prompt type so a run trace shows the message-writing draw apart
    # from the cycle's own calls: they are two contexts, and the whole point of the
    # split is that the second one is short and carries no tool channel.
    NOTIFY_COMPOSE_AGENT_NAME = "notify-composer"
    NOTIFY_COMPOSE_PROMPT_TYPE = "notify_compose"
    # How many nearest past messages the notify document carries from each of the two
    # message logs (#1911) — the ``k=5`` the retired notify steps asked for, kept
    # because a handful is what a callback line can be judged from and the document
    # stays short.  Since #1934 it is also the RENDER's own cap on that section, so the
    # bound holds wherever the lines came from rather than only where they were fetched.
    NOTIFY_RELATED_MESSAGES = 5
    # The notify document's section budgets (#1934).  The document is assembled whole,
    # framework-side, and the first live post-reset cycle assembled 50,805 characters of
    # it for an ordinary three-page news round — on a model whose degeneration onset is
    # ~4K prompt tokens.  Three things carried the bulk, and each has a budget below: a
    # browse call's RESULT inlined whole (31,725), the rendered CALL notation restating
    # the whole write payload in its arguments (14,923 on ONE line), and the entries the
    # cycle wrote (14,989 — the same content a third time).
    #
    # These are prompt-budget bounds with a VISIBLE overflow, not silent truncations:
    # every cut states the characters or the items it left out, so a draw reading a
    # condensed result knows it is reading part of one.  Proportions follow what the
    # message is written FROM — the entries the cycle wrote are the payload and get the
    # largest share, the calls are how it got there, and the earlier conversation is
    # background.  Together they hold an ordinary document near ~10K characters (~2.5K
    # tokens), comfortably under the onset, with the frame on top.  Deliberately
    # generous within that; sizing is tunable later.
    #
    # EVERY free-text field the document repeats is bounded, not just the three that
    # carried the measured bulk — a ceiling one unbounded field can defeat is not a
    # ceiling, and an entry KEY is model-authored with no length gate anywhere and
    # renders once per written entry.  The two fields deliberately left WHOLE are the
    # collection's name and the routine's: they render once each, they are bounded
    # upstream where they are derived (``DERIVED_NAME_MAX_LENGTH`` / the framer's mint),
    # and they are ANCHORS a message may need to copy verbatim.
    NOTIFY_WRITTEN_ENTRIES = 8
    NOTIFY_WRITTEN_CONTENT_CHARS = 500
    NOTIFY_ENTRY_KEY_CHARS = 120
    NOTIFY_CYCLE_CALLS = 10
    NOTIFY_CALL_CHARS = 150
    NOTIFY_CALL_RESULT_CHARS = 300
    NOTIFY_RELATED_LINE_CHARS = 200
    NOTIFY_DESCRIPTION_CHARS = 300
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
    # Did-you-mean suggestion gate (#1674): the minimum stdlib string similarity
    # (``difflib.SequenceMatcher`` ratio, 0..1) between a missed memory name / entry
    # key and an EXISTING one for the miss to lead its error with "did you mean
    # '<it>'?".  Conservative — a suggestion must be a near spelling of a real
    # name/key, so a genuinely-unrelated guess gets NO misleading suggestion and the
    # message stays byte-identical to before.  'aurora-deone' → 'aurora-deck-2'
    # (the motivating typo) clears it (ratio ≈ 0.72); two unrelated names don't.  No
    # house string-distance constant existed to reuse, so this one is named here (per
    # the ticket); the meaning leg reuses the dedup thresholds instead of a new
    # number.  0.6 is ``difflib``'s own documented default cutoff for close matches.
    DID_YOU_MEAN_STRING_CUTOFF = 0.6
    # Write-target REDIRECT gates (#1570 journey): a chat collection_write whose
    # target name doesn't exist is silently-but-NARRATEDLY redirected into an
    # existing collection when the miss is near-certainly a name variant of it.
    # Calibrated against the archived journey-eval run DBs (every missed write
    # target + every same-sample distinct-collection pair):
    #   distinct collections (different teams) cap at char 0.727 / TCR 0.667;
    #   observed typo class (transpositions, hyphen joins) lands char 0.90-0.92;
    #   same-intent extension ('team-news' → 'team-news-alerts') is TCR 1.0 at
    #   char 0.72.  So: char ≥ 0.85 (margin above 0.727, slack under 0.90) OR
    #   full token containment (TCR == 1.0, vs ≤ 0.667 for negatives) redirects;
    #   anything between falls to the did-you-mean refusal; nothing close at all
    #   AUTO-CREATES the collection (the dominant observed miss: no collection
    #   existed yet).
    MEMORY_NAME_REDIRECT_CHAR_RATIO = 0.85
    MEMORY_NAME_REDIRECT_TCR = 1.0
    # The auto-created collection's placeholder description prefix — shared by the
    # write tool (which stamps it) and the run-end auto-attach (which recognizes it
    # and backfills the skill's generic description as the real meaning anchor).
    AUTO_CREATED_DESCRIPTION_PREFIX = "auto-created to hold"
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
    # How many of the bound collection's entries a collector cycle's prompt renders
    # (#1914), newest first, in the same spirit as the three above: a prompt-budget
    # bound with a VISIBLE overflow, deliberately generous, tunable later.  20 keeps a
    # daily routine's recent three weeks readable — far past the point where a
    # re-observing routine can see the key it wrote under — while the block stays a
    # list a cycle skims rather than a corpus it re-reads.  The overflow states its own
    # COUNT rather than naming a fetch tool the way the self-state sections do: a
    # cycle's surface is scoped to its program's own calls, so a read tool named here
    # might not be on it, and an instruction that cannot be followed is worse than the
    # honest number.
    COLLECTOR_HOLDINGS_LIMIT = 20
    # How many times a collection's occurrence may be re-attempted before the
    # dispatcher stamps it anyway and waits for the next one (#1935).  A cycle that
    # ended on a STOCHASTIC cause and changed nothing — preempted by a foreground
    # message, or aborted on a failed model call — did not spend its occurrence, so
    # stamping it would silently skip a day of a daily job.  The bound is what keeps
    # the stamp's original job (a persistently-failing collection must not re-attempt
    # on every tick) intact: after this many attempts the occurrence is consumed
    # whatever happened.  3 covers a burst of foreground activity or a transport
    # wobble without letting a collection that fails every time hold the dispatcher.
    COLLECTOR_RETRY_ATTEMPTS = 3
    # How far ahead a stated end date stops being a date and starts being a
    # date-shaped way of writing "forever" (#1944).  An apply turn asked to set up a
    # job that runs indefinitely wrote ``expires_at = 2099-12-31``; the sentinel then
    # rendered in the self-state header as the job's end condition and was faithfully
    # copied back on every later ``collection_set``, so invented config became sticky.
    # The threshold is a HORIZON measured from now, never a fixed calendar date — a
    # fixed one is itself a sentinel that rots as the clock passes it.  It is a
    # WHOLESALE bound, not a calibrated one (there is no corpus of end dates to
    # calibrate against, and one built from a fresh deployment would be a corpus of
    # this very defect): twenty years is chosen to sit in the gap between two
    # populations that do not overlap anywhere near it — a real end condition is
    # something a person can point at (a contract, a course, a season, a trial) and
    # lands within a few years, while the dates a model reaches for when it means "no
    # end date" (2099-12-31, 9999-12-31) land decades or millennia out.  Tunable; the
    # over-correction it is sized against is a genuine end date read as a sentinel, so
    # it errs high, and both directions are pinned by tests.
    # Anything past it is normalised to no expiry, with the result naming what happened
    # — normalised, never silently, and never rejected: the job the user asked for is
    # exactly the unbounded one, so refusing the call would cost the turn over a value
    # the framework can read correctly.
    SENTINEL_EXPIRY_HORIZON_DAYS = 365 * 20
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

    # The retired pub/sub notifier consumer (seeded by migration 0067, archived by
    # #1557, then NUKED entirely by migration 0097/#1676).  Its ``memory`` row is
    # gone, but historical ``messagelog`` rows it sent survive (history is never
    # rewritten), so this key is retained SOLELY to classify those notifier-sent
    # messages on the iOS surface (``channels/ios/channel.py``).  No longer a member
    # of ``SYSTEM_COLLECTIONS`` — there is no archived shell left to hide.
    MEMORY_NOTIFIER_COLLECTION = "notifier"
    # NOTHING is pre-seeded any more (#1911, migration 0108) — the soft reboot.
    # ``dislikes`` was the last migration-seeded collection: 0097 removed the eight
    # generic catch-alls and kept it as "very narrow and specific", and the ruling
    # retires that exemption too, on the principle that no intermediate legacy
    # structure should be left standing.  So there is no ``SYSTEM_COLLECTIONS`` set
    # and no ``MEMORY_DISLIKES_COLLECTION``: every collection in the registry is one
    # the USER built, which is what let the catalog's hide-list and the duplicate
    # check's skip-list go with them.  ``SYSTEM_LOGS`` above is untouched — the four
    # logs are Python-populated perception, not collections anybody would rebuild.
