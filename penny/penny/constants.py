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
    a healthy end has its entry writes reverted, so it changed nothing durable and is
    the ``failed`` bail it now honestly is.  The member stays for the historical rows
    that carry it and the ``⚠ INCOMPLETE`` flag that renders them ·
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
    ``KEY_EXISTS_CHANGED`` — the write resolved to an existing entry whose content
    DIFFERS from what it held: the observed value changed, so the write gate
    **auto-refreshes the stored baseline itself** in place (through the update
    path — same validation, degeneracy screen, and ``last_written_by_run_id``
    stamp), and the run's only remaining job is to notify.  No further call is
    needed — the refresh already happened, so the next observation of the same
    value reads ``KEY_EXISTS_UNCHANGED`` (#1633).  Reached two ways since the
    identity ladder (#2715): the exact key with different content, and a
    different key the ladder matched as the same thing — a merge, whose reply
    shows the prior value and the one call that restores it ·
    ``KEY_EXISTS_UNCHANGED`` — the exact key existed with *identical* content: the
    value has not changed, so there is nothing further to do — the watch's "no
    change" signal.  It carried STOP semantics until #2385 emptied the STOP table:
    the unchanged write now returns normally and the cycle goes on to its close ·
    ``DUPLICATE_UNCHANGED`` — the identity ladder matched a *different* existing key
    AND the stored entry already says what this write says, however either is worded,
    so the cycle observed nothing new.  The same no-news
    as ``KEY_EXISTS_UNCHANGED``, reached under a different key — likewise no longer
    STOP-worthy since #2385 (#1919's split of the outcome itself stands) ·
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
    DEGENERATE = "degenerate"
    UNEXPECTED = "unexpected"


# The declared STOP table (#1587): the write-gate outcomes that CAN end a must-act
# (collector) run at the write chokepoint, mapped to the run's stamped stop reason.
#
# The two members are the "value unchanged" case, under EITHER of the two ways a write
# can arrive at an entry that already holds its value: the exact key
# (``KEY_EXISTS_UNCHANGED``) or the dedup disjunction's strict CONTENT match under a
# different key (``DUPLICATE_UNCHANGED``, #1919).
#
# NEITHER ends a cycle any more (#2385, the code owner's ruling): an unchanged write
# is a quiet OBSERVATION, not a verdict.  The write gate still dedups and its result
# still tells the model the value was unchanged, but the call returns normally so the
# model reaches its close and records with ``done(notify=…)`` whether the user should
# hear about the cycle — whether telling the user is the cycle's decision to make and
# be measured on, not a verdict the gate issues before the close exists to carry it.
# Nothing stops a cycle on a write-gate outcome any more (#2538 removed the always-empty
# ``WRITE_GATE_STOP_WORTHY_OUTCOMES`` membership check); this table keeps the two reasons
# for the LEGACY rows they stamped (the run-record renders read the stamped reason off
# the declared table) and for the eval cases that read it verbatim.
#
# ``NEW_KEY`` / ``KEY_EXISTS_CHANGED`` never stopped (an accumulator keeps going
# mid-script), and ``DEGENERATE`` stays surfaced-but-recoverable — a rejected
# entry can be re-written substantively on the next call.  (``DUPLICATE`` retired
# with the three-signal rule: the identity ladder merges or writes-through a
# look-alike pair, so there is no divergent-value collision left to reject.)
WRITE_GATE_STOP_REASONS: dict[WriteGateOutcome, str] = {
    WriteGateOutcome.KEY_EXISTS_UNCHANGED: "the value was unchanged since the last observation",
    WriteGateOutcome.DUPLICATE_UNCHANGED: (
        "the value was already recorded, under a different key, since the last observation"
    ),
}


class CollectionConfigureOutcome(StrEnum):
    """Which path a ``collection_configure`` call took — set by the dispatcher
    (``CollectionSetTool``), never re-derived from the call's own arguments or
    guessed from which inner class happened to answer the tool's registered name
    (#2768).  Three classes in ``tools/memory_tools.py`` once all declared
    ``name = "collection_configure"``, so the class registry's last-defined-wins
    collision made every call — create included — narrate as an update.  Now
    ``CollectionSetTool`` alone owns the name, and it stamps this field on the
    result it returns so its own ``to_result_narration`` names the real outcome."""

    CREATED = "created"
    UPDATED = "updated"


# ``COLLECTOR_COVERED_REASON`` is RETIRED (#1916, with the coverage exit it named).  A
# cycle closes on ``done()`` again, and the render tells a clean close from an abandoned
# one by the ledger's own ``done`` record rather than by a stamped phrase — so a clean
# close carries no reason of its own, and the record's header falls back to the outcome
# enum exactly as it did before #1911.  What a clean close DOES carry, when the
# collection notifies, is what telling the user came to (``NOTIFICATION_NOTES``).


class CycleEnd(StrEnum):
    """How a collector cycle ENDED — the closed set that decides both whether the
    cycle's entry writes SURVIVE and whether its schedule occurrence is spent (#1936).

    One set read twice, because they are one question: a cycle that reached its
    conclusion did the job its fire was for, and a cycle that did not must be retried
    from the state it started in.  Declared as data like ``WRITE_GATE_STOP_REASONS`` —
    a later shape joins the table, not the code.

    ``STOPPED`` — a write-gate STOP closed the cycle at the chokepoint.  An early
    clean no-news exit IS an end point, so whatever landed before it stands ·
    ``CLOSED_QUIET`` — closed with ``done()`` on a collection that does not notify ·
    ``CLOSED_NOTIFIED`` — closed with ``done()``, and the report was QUEUED for the
    delivery turn ·
    ``CLOSED_DECLINED`` — closed with ``done()``, and delivery was deliberately
    declined (the user is muted, or there is no registered recipient): a retry cannot
    change either, so the cycle ended and the decline is stamped on the run record ·
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
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    UNFINISHED = "unfinished"


# The HEALTHY ends (#1936, code-owner ratified).  A cycle that ended one of these four
# ways reached its conclusion: its entry writes STAND and its schedule occurrence is
# SPENT.
#
# Membership is what makes an end healthy — everything outside this table is REVERTED
# from the cycle's undo journal and leaves the occurrence due for #1935's bounded retry,
# so an end nobody enumerated is retried rather than half-persisted.  That default is
# the ruling's own: a partial run must not persist state later cycles have to reason
# about.
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

    ``COLLECTION`` is the background mechanism post-#1556 — the only registry
    this enum currently tracks."""

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
    ROLLING = "\U0001f3b2"  # 🎲 — making a fair random choice
    WORKING = "\u2699\ufe0f"  # ⚙️ — generic fallback for other tools


class PageReading(StrEnum):
    """How a browsed page was READ (#2388) — structural, set once by the extension's
    own rule (``isLinkIndex``/``richer`` in ``page_text.ts``) and carried unchanged
    through the wire, the browse tool, and into the extractor's own input.  Never
    guessed downstream from the text's shape: the extension is the one place that
    reads the live DOM, so it is the one place that gets to decide.

    A two-member enum rather than a bool — a third reading (e.g. a tabular page) could
    join this set later without renaming anything that already reads it.  ``ARTICLE``
    is also the honest default for an older extension that never sends the field at
    all (logged, not silent — see ``BrowserChannel._handle_tool_response``)."""

    ARTICLE = "article"
    LINK_LIST = "link_list"


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
    """Prompt types emitted by ChatAgent flows. Logged to promptlog.prompt_type.

    The separate ``vision_message``/``vision_caption`` types retired with the
    caption step (#2635) — an image turn is an ordinary ``user_message`` run.
    """

    USER_MESSAGE = "user_message"
    # The system-originated delivery turn a notifying job's report rides (#2656):
    # the framework renders the report (the job's name, description, and the full
    # written entries) and the chat agent writes the message from it with its whole
    # context.  Its own type so a run trace shows the delivery apart from the
    # user-initiated turns on the same agent.
    JOB_REPORT = "job_report"


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
    # The finish_reason a provider returns when a token cap cut the draw.
    FINISH_REASON_LENGTH = "length"
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

    # The conversation-record marker written where an image attachment was (#2635):
    # the image itself rides only its turn's own user message and history carries
    # this line instead — written by the channel at the log site, never by the model.
    IMAGE_HISTORY_MARKER = "[sent a photo]"

    # Agent loop constants
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
    # How often the liveness sweep probes the connections that have gone quiet
    # past that window.  Dead-socket detection used to be write-driven — nothing
    # noticed a socket was gone until something happened to write to it, so browse
    # requests went into a corpse for ~2.4 minutes (four retry rounds, twelve
    # unanswered requests) until a write finally failed with code 1006.  Sweeping
    # puts that on a clock: a half-open socket often absorbs one probe before its
    # next write raises, so detection costs a sweep or two rather than however long
    # it happens to be until something writes.  It costs nothing on a healthy
    # addon, which heartbeats every ~15s and so is never quiet enough to be probed.
    BROWSER_LIVENESS_SWEEP_SECONDS = 15.0
    # How long a single liveness probe may take before the socket is judged
    # unreachable.  A send awaits flow-control drain, so a half-open peer that has
    # stopped READING never raises — it just never returns, which is the one shape
    # a write-driven check cannot tell from a slow one.  Without a bound, a probe
    # of exactly the socket this sweep exists to find would wedge the sweep itself.
    BROWSER_LIVENESS_PROBE_TIMEOUT_SECONDS = 5.0
    # How a browser socket renders before it has told us which device it is.
    BROWSER_UNREGISTERED_DEVICE = "unregistered"
    # What an in-flight tool request is told when the socket it was sent on turns
    # out to be gone.  It names the connection AND the remedy, because it is read
    # in the log by whoever is working out why a browse round did nothing; the
    # browse tool consumes the ConnectionError itself, and retries onto whatever
    # connection is live rather than waiting out its own timeout for the same news.
    BROWSER_STRANDED_REQUEST_ERROR = (
        "the browser connection '{device}' closed with this request still in flight — "
        "retry: the next attempt routes to whatever addon connection is live"
    )

    # System log memories (created by migration 0026) that the channel
    # adapter and browse tool side-effect-write to on every turn.
    MEMORY_USER_MESSAGES_LOG = "user-messages"
    MEMORY_COLLECTOR_RUNS_LOG = "collector-runs"
    # ``promptlog.agent_name`` stamped on every chat-agent prompt — the structural
    # marker the ``run_list`` tool uses to find conversational runs (a turn's
    # user message → the tool calls it drove).  Mirrors ``ChatAgent.name``.
    CHAT_AGENT_NAME = "chat"
    # The ledger identity of the one background Collector dispatcher (every
    # collection cycle, whichever target it's bound to).  ``Collector.name`` reads
    # this constant rather than repeating the literal, so the two call-kind names
    # that gate a token budget (#2474) are both declared here, not split between
    # this module and ``agents/collector.py``.
    COLLECTOR_AGENT_NAME = "collector"
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
    # The real tool names a logged call is classified by when a run record is
    # rendered (``objects.py``'s ``render_tool_call`` / ``_run_io_tally`` /
    # ``classify_run``) — named here rather than compared as bare literals, the
    # same shape ``DONE_TOOL_NAME`` already set.  ``SEND_MESSAGE_TOOL_NAME``
    # stays even though the tool itself is retired (#1911): pre-retirement
    # ``promptlog`` rows still carry the name, and history is never rewritten.
    COLLECTION_WRITE_TOOL_NAME = "collection_write"
    COLLECTION_REMOVE_TOOL_NAME = "collection_remove"
    LOG_READ_TOOL_NAME = "log_read"
    LOG_APPEND_TOOL_NAME = "log_append"
    SEND_MESSAGE_TOOL_NAME = "send_message"
    BROWSE_TOOL_NAME = "browse"
    # The ledger identity of a browse micro-context extraction — a fresh
    # single-shot model call (content + instruction, no tools) that runs when a
    # ``browse`` carries an ``extract`` argument.  It logs its own promptlog rows
    # under this agent/prompt type so run traces attribute it honestly, while the
    # bulk page content never enters the parent run's context.
    BROWSE_EXTRACT_AGENT_NAME = "browse-extract"
    BROWSE_MICRO_CONTEXT_PROMPT_TYPE = "browse_micro_context"
    # The ledger identity of a browse picker draw — the degraded arm's pick
    # (shared model, thinking off) when no decision model is configured.  The
    # decision-model arm rides /v1/systemone and has no promptlog row.
    BROWSE_PICK_AGENT_NAME = "browse-pick"
    BROWSE_PICK_PROMPT_TYPE = "browse_pick"
    # How many ranked results one search item reads before it gives up (#2718):
    # the picker's picks, falling through while a read finds nothing it was
    # asked for.  Sized to the lookup cases' own shape — a search page carries
    # up to four results and the requirement can sit on a lower-ranked one
    # (the settling page ranked third on the measured lookup), so two reads
    # stopped short of the answer the list held.
    BROWSE_MAX_PICKED_READS = 4
    # The longest extract value that enters the main loop (#2718).  A longer
    # value is cut at the last line boundary under this and the render says how
    # much was left out, with the fetch handle to the stored full page.
    BROWSE_EXTRACT_MAX_CHARS = 4000
    # How many recent conversational runs ``run_list`` returns per batch —
    # bounded like every other cursored log read (``LOG_READ_LIMIT``).
    RUN_CALLS_LIMIT = 10
    # The type tag a rendered activity-log run anchor carries — ``run <id>`` (the
    # self-state header's run/mutation lines and ``render_run_calls``'s header emit
    # it verbatim).  ``run_get`` strips it to route the typed id to the run case,
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
    # ``log_read``'s day-read bound (#2752) — the optional ``day`` argument
    # reads ONE local day of a message log, oldest first, capped at this.  A
    # day is what the conversation map's anchors name, and a day's own volume
    # (10–23 messages on this deployment, ~52 at the recorded peak) is what the
    # cap has to hold — the 10-entry cursor batch would cut a day's exchange
    # in half.
    LOG_READ_DAY_LIMIT = 60
    # How many recent registry-change events ``memory_get`` renders in its
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
    # The auto-created collection's placeholder description prefix — shared by the
    # write tool (which stamps it) and the run-end auto-attach (which recognizes it
    # and backfills the skill's generic description as the real meaning anchor).
    AUTO_CREATED_DESCRIPTION_PREFIX = "auto-created to hold"
    # Self-state header caps (#1555).  The chat agent's system prompt opens with a
    # deterministically-rendered header of Penny's own operational situation
    # (mechanisms · recent activity · the store map · durable user facts).
    # Mechanisms and the store map render EVERYTHING — every row, no cap (#2288) —
    # since both index her own situation, which is the thing this header exists to show in full.
    # Activity stays bounded to a fixed number of newest rows, because it is a
    # ledger of HISTORY rather than an index of what currently exists, and a
    # visible "+ older activity — <tool>" tail names the fetch tool so nothing is
    # silently dropped and n≤1 still holds.
    SELF_STATE_ACTIVITY_LIMIT = 8
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
    # How many of a learned routine's steps the learn-close narration's step-shape line
    # renders (#1943), in order, before it states the count it left off.  The line exists
    # to be SKIMMED in one pass by the person who knows what the routine was meant to do,
    # so the bound is a READING budget like the three above: eight segments is roughly a
    # line of prose, and a demonstration longer than that is one whose whole shape no
    # single line was ever going to carry.  The overflow states its own COUNT and names
    # no fetch tool for the same reason the holdings block does not — the whole recipe is
    # one `skill_get(skill=<name>)` call away, so the honest number is the whole job of
    # the overflow.
    SKILL_SHAPE_STEPS = 8
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
    # copied back on every later ``collection_configure``, so invented config became sticky.
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
    # How many characters of a job's prompt the self-state header's instructions line
    # renders (#2495).  The line hangs under the job's mechanism row and shows what it
    # reads and follows — a READING budget like the activity cap above, not a hard
    # guarantee: a longer prompt renders its first sentence (hard-cut at a word
    # boundary inside the cap if even that is longer) followed by a note counting the
    # sentences cut and naming the ``memory_get('<name>')`` hop to the whole prompt, so
    # nothing is cut silently and the job's page renders whole regardless.  160 is
    # roughly the first sentence of a real job's prompt with room to spare; the full
    # prompt is always one anchored read away.
    SELF_STATE_JOB_PROMPT_CHARS = 160
    # How many characters of each redacted tool argument an iOS agent-progress
    # event carries (#2642).  The progress payload is a transient preview, not the
    # data — the run record and the tool's own result render the argument whole.
    IOS_PROGRESS_REDACT_PREVIEW_CHARS = 120
    # Discord's own hard message-length limit (#2642): a longer outgoing message is
    # split into chunks of this size, so nothing is ever lost.
    DISCORD_MESSAGE_LIMIT = 2000
    # A restart announcement is one line (#2642): an LLM-drafted announcement past
    # this length is rejected whole (never truncated) in favour of the deterministic
    # fallback note, because a longer draft is not an announcement.
    RESTART_ANNOUNCEMENT_MAX_CHARS = 150
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
    # The one deliberate exception since (#2633, migration 0119): the seeded
    # ``rules`` collection — standing instructions the user has given ("always X",
    # "never Y").  A seeded row with a known key is universal, so the migration is
    # allowed; the ``### Standing rules`` section (``agents/self_state.py``) renders
    # its entries in full wherever Penny acts, so this name is the anchor those
    # renders and the write verbs share.
    MEMORY_RULES_COLLECTION = "rules"
