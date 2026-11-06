"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with web browsing and memory tools.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.agents.base import Agent, InvalidDraw, ProgressCallback, ProgressScope
from penny.agents.conversation_map import ConversationMap, map_horizon_start
from penny.agents.models import (
    NO_INBOX,
    ControllerResponse,
    HistoryTurn,
    MessageRole,
    ToolCallRecord,
    TurnInbox,
)
from penny.agents.related_exchanges import RelatedExchanges
from penny.agents.self_state import SelfStateHeader, standing_rules_instruction
from penny.changelog import ChangelogEntry, render_changelog_announcement
from penny.channels.base import PageContext
from penny.constants import ChatPromptType, PennyConstants, RunOutcome
from penny.database.memory.identity import WriteDecisionCache
from penny.database.message_store import SessionAnchor
from penny.datetime_utils import current_datetime_line, stored_as_utc
from penny.llm.call_link import CallLink
from penny.llm.models import LlmError, LlmMessage, LlmToolCall, LlmToolCallFunction
from penny.notification import (
    JobFailedReport,
    JobReport,
    JobRetiredReport,
    SandboxReport,
    render_job_failed_report,
    render_job_report,
    render_job_retired_report,
    render_sandbox_report,
)
from penny.prompts import Prompt
from penny.text_validity import (
    is_call_as_text_bail,
    is_call_fragment_reply,
    is_embedded_call_as_text,
    is_empty_draw,
)
from penny.tools import Tool
from penny.tools.browse import BrowseTool
from penny.tools.collection_sandbox_run import CollectionSandboxRunTool, SandboxRunner
from penny.tools.generate_image import GenerateImageTool
from penny.tools.link_handles import PAGE_CITATION_LINE, HandleTable
from penny.tools.memory_tools import render_nothing_landed, render_writes_landed
from penny.tools.notifications import NotificationsTool
from penny.validation import ConditionKey

if TYPE_CHECKING:
    from datetime import datetime

    from penny.llm.image_client import MfluxImageClient

logger = logging.getLogger(__name__)


def _system_notification_trigger(facts: str) -> str:
    """The trigger a system-originated delivery turn is given (#2974): the
    framework's own facts, prefixed with ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX``
    so the live turn reads as a notification, never as the user's words or an
    instruction to send anything.  The ONE place the prefix is set — the same
    message ``Agent._run_trigger_message`` later locates in the run's own
    promptlog by that prefix, byte-identical."""
    return f"{PennyConstants.SYSTEM_NOTIFICATION_PREFIX}{facts}"


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Three context mechanisms, kept independent:

    - Deploy-stable identity → the LEADING system message (#2745): Penny's
      persona/voice, the user's standing rules, and her instructions — nothing
      that changes except on a deploy or a config edit.
    - Operational self-state → the ONE session-state system message (#3089,
      previously split across the leading and a per-turn message, #2756/
      #2814/#3055): a deterministic ``SelfStateHeader`` (active mechanisms ·
      recent activity · the store map · durable user facts) plus the
      **conversation map** (#2752), bounded by the session's own start
      (``_session_as_of``) so it is byte-identical turn to turn within one
      session — only an idle rotation rebuilds it.  The speculative
      user-content recall is gone (the ambient inversion, #1555): user content
      is fetched on demand via the tools the header's pointers name.
    - Chat turns → messages array as alternating user/assistant turns
      via ``_build_conversation`` and ``history=``.

    The request is ``[leading system][session-state][history][dynamic
    system][user]`` (#3089): the leading message carries identity + rules +
    instructions, reused across every session; the tool schemas render right
    after it (gemma4's raw template).  The session-state message — Jobs, Your
    memory, Recent activity, About the user, the map — comes next, its own
    system turn (confirmed #3089: the vendored template special-cases only the
    very FIRST message for the tool-schema block, so a second leading system
    message renders as its own turn with no tools re-declared), reused for
    every turn of the same session; the growing, append-only history follows
    it untouched.  The dynamic message — the clock, the page hint and the
    related earlier exchanges — sits right BEFORE the user's own turn: the one
    MID-CONVERSATION system placement prod's patched oMLX and OpenRouter
    (eval) both render in place, so eval and prod read the same prompt (see
    ``Agent._build_messages``).  Every turn — image or not — runs the
    standard agentic loop; an image rides the turn's own user message
    (#2635) for the chat model to read.
    """

    name: str = "chat"
    system_prompt = Prompt.CONVERSATION_PROMPT
    # The related earlier exchanges section (#3010) — rendered ONCE per turn in
    # ``_dispatch_turn`` (the live message embeds there, once) and read by
    # ``_dynamic_system_section`` on every loop step; cleared with the rest of
    # the per-turn state in its ``finally``.  ``None`` on the class is the
    # never-dispatched default, so an agent built outside ``_dispatch_turn``
    # renders the section's quiet absence.
    _related_exchanges: str | None = None
    # This turn's ONE ``ConversationMap`` instance (#3085) — built lazily by
    # whichever of ``_related_exchanges_section``/``_session_state_section`` runs
    # FIRST in ``_dispatch_turn`` (``_conversation_map_for``) and reused by the
    # other, so the session-state message's own render() and the
    # related-exchanges retrieval's one-to-one id set never run the
    # horizon/window pipeline through two different instances.  Cleared with
    # the rest of the per-turn state in ``_dispatch_turn``'s ``finally``;
    # ``None`` on the class is the never-dispatched default, same shape as
    # ``_related_exchanges`` above.
    _conversation_map: ConversationMap | None = None
    # The once-per-session state message (#3089) — Jobs + Your memory + Recent
    # activity + About the user + the conversation map, rendered ONCE per turn
    # in ``_dispatch_turn`` and read by ``_build_messages`` to insert as its own
    # system message right after the leading system message's tool schemas and
    # before the growing history.  Cleared with the rest of the per-turn state
    # in its ``finally``; ``None`` on the class is the never-dispatched default.
    _session_state: str | None = None
    # Chat's run-shape chain (base Agent's is empty).  Empty here too (#2455 naive
    # baseline / #2473): what a run wrote and what it configured are both delivered
    # earlier, straight onto the tool result the moment their call lands (see
    # ``after_tool_step`` below), so there is nothing left for a text-branch
    # validator to narrate.  A new chat shape guard is one more entry here — never a
    # branch in the loop.
    run_shape_validators = []

    # Chat's invalid draws are the ones that are not a REPLY (#1839/#1937): a plain
    # conversational reply IS chat's valid terminal state, so nothing about ordinary
    # prose is rejected.  Three shapes, most specific first, because the condition the
    # discarded draw is logged under must name what it actually was:
    #  - CALL_AS_TEXT: a tool call that reached the text channel — the full serialized
    #    call (``{"name": …, "arguments": …}``), or real args naming a registered tool
    #    as a string value / textual call syntax (#2787).  It would be delivered to
    #    the user as raw machinery.
    #  - CALL_FRAGMENT_REPLY: its mangled remainder — a bare argument fragment or the
    #    empty ``{}`` tail of a forced call attempt (#1570/#1732).
    #  - EMPTY: a draw that says nothing at all (#1937).  It was the last invalid-output
    #    class still handled on the VISIBLE path, where the empty turn plus a nudge went
    #    into the conversation and the model's in-flight intent was lost with them.  It
    #    is tried LAST because the ``{}`` tail above has no letters either, and that
    #    shape has a truer name.
    invalid_draw_conditions: tuple[InvalidDraw, ...] = (
        (ConditionKey.CALL_AS_TEXT, is_call_as_text_bail),
        (ConditionKey.CALL_FRAGMENT_REPLY, is_call_fragment_reply),
        (ConditionKey.EMPTY, is_empty_draw),
    )
    # Stable id linking the synthetic page-context tool-call to its tool-result
    # so the injection rides the standard OpenAI ``tool_call_id`` envelope, not
    # an ad-hoc ``tool_name`` field.
    PAGE_CONTEXT_TOOL_CALL_ID = "page-context"

    def __init__(
        self,
        image_client: MfluxImageClient | None = None,
        email_tools_builder: Callable[[str, str], list[Tool]] | None = None,
        plugin_tools: list[Tool] | None = None,
        *,
        collector: SandboxRunner,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # The capability behind ``collection_sandbox_run`` (#2964) — REQUIRED, no
        # default and no ``None`` that would quietly mean "not wired" (CLAUDE.md,
        # "Pass parameters, don't swap state"): every caller, production or test,
        # states explicitly what previews a job's cycle.  Typed against the
        # ``SandboxRunner`` protocol, never the concrete ``Collector``, so this
        # module never imports it (``penny.agents.collector`` already imports
        # from ``penny.tools.memory_tools``, and importing ``Collector`` back
        # from a tools-adjacent module would close that cycle).
        self._collector: SandboxRunner = collector
        self._pending_page_context: PageContext | None = None
        # The current user message, held for the span of one turn so the
        # per-turn tool builders (email read summarisation) can see what the
        # user just asked.  Set in ``handle`` and cleared in its ``finally``.
        self._current_message: str | None = None
        # Static plugin tools registered at startup. These are shared integrations
        # (Zoho calendar/projects, InvoiceNinja, etc.) that do not need per-turn
        # rebuilding, unlike email tools.
        self._plugin_tools: list[Tool] = plugin_tools or []
        # Chat replies via final text — tools are stripped on the final
        # agentic step to force the model to produce its reply.  Background
        # agents inherit the True default to keep tools available so they
        # can call ``done`` / ``send_message`` on the final step.
        self._keep_tools_on_final_step = False
        # Present only when an image model is configured — mirrors the retired
        # /draw command's conditionality; enables the generate_image tool.
        self._image_client = image_client
        # Present only when a mailbox is configured (Fastmail or Zoho) — mirrors
        # the retired /email + /zoho commands' conditionality.  Builds the email
        # tools fresh per turn (read_emails summarises against the current
        # message + date), so it takes ``(user_query, today)``.
        self._email_tools_builder = email_tools_builder
        # The run whose writes-landed record was already computed this turn — the
        # structural once-per-run guard, so the post-narration re-reply never re-reads
        # the store a second time (chat turns are sequential, so one field suffices).
        self._extraction_run_id: str | None = None

    def _invalid_draw_condition(self, content: str) -> ConditionKey | None:
        """Chat's shape check (#1839), extended to catch a tool call written as text
        ANYWHERE in the reply (#2787) — prose followed by the call object, which the
        declared ``invalid_draw_conditions`` tuple can't see: its ``CALL_AS_TEXT``
        entry (:func:`is_call_as_text_bail`) only judges the WHOLE reply, and a
        content-only predicate has no way to read the live tool registry anyway.
        Checked first, against the tools this turn actually installed
        (``_install_tools`` runs before every call, #2458) — so a plugin tool is
        covered with no change here — then falls through to the base walk, which
        keeps the lone-envelope / bare-args / fragment / empty shapes exactly as
        they were."""
        registered_tools = frozenset(tool.name for tool in self._tool_registry.get_all())
        if is_embedded_call_as_text(content, registered_tools):
            return ConditionKey.CALL_AS_TEXT
        return super()._invalid_draw_condition(content)

    def get_tools(
        self,
        run_id: str | None = None,
        handles: HandleTable | None = None,
        decision_cache: WriteDecisionCache | None = None,
    ) -> list[Tool]:
        tools = super().get_tools(run_id, handles, decision_cache)
        # Notification mute/unmute is a chat-driven action over the MuteState row
        # (the retired /mute + /unmute commands, then the retired mute/unmute tool
        # pair, #2812), so the tool lives on the chat surface — the model
        # dispatches to it from natural language.
        tools.append(NotificationsTool(self.db))
        # Previewing a job's own cycle (#2964) is chat-only, like notifications —
        # a collector cycle never gets this tool (it would let a preview preview
        # itself), and it is never reachable without an actual collector wired.
        tools.append(CollectionSandboxRunTool(self._collector))
        if self._image_client is not None:
            tools.append(
                GenerateImageTool(self._image_client, self.db, self._embedding_model_client)
            )
        tools.extend(self._email_tools())
        tools.extend(self._plugin_tools)
        return tools

    def _email_tools(self) -> list[Tool]:
        """Config-gated email tools, built fresh for this turn.

        Empty unless a mailbox (Fastmail or Zoho) is configured.  The builder
        wraps the long-lived email client with the current message + date so
        ``read_emails`` can summarise against what the user just asked — the
        retired /email + /zoho commands' search → read → answer surface, now
        driven from natural language.
        """
        if self._email_tools_builder is None:
            return []
        # During a live turn ``_current_message`` is always set (``handle`` sets it
        # before installing tools).  It is None only when ``get_tools`` is
        # enumerated outside a turn (surface inspection), where the tools are never
        # executed — so an empty query is the correct "no active question" value,
        # not a masked missing input.
        return self._email_tools_builder(
            self._current_message or "", current_datetime_line(self.db)
        )

    # ── Writes-landed narration, delivered with the tool step (#2473) ─────

    async def after_tool_step(
        self,
        step_records: list[ToolCallRecord],
        records: list[ToolCallRecord],
        messages: list[LlmMessage],
        run_id: str,
    ) -> None:
        """Append what this run has WRITTEN so far to the end of the step's own tool
        result — before any draft of the reply exists (#2455 naive baseline drops this
        chain's other former member, the applied-configuration frame, #1869 — it reads
        the round framing this branch no longer has).

        The record exists the moment its tool call lands (it is a read of the
        store, not of the run's own memory of itself), so recomputing them and
        appending whichever are non-``None`` here means the model's first reply is
        drawn with them already in view.  There is no draft to discard and no second
        draw: the old mechanism dropped a reply written before the record and redrew
        with it spliced into the system prompt, which is the two-reply regression this
        ticket closes (#2473).

        Recomputed only on a step whose OWN calls could plausibly have changed the
        record (a store-asking write) — ``_may_change_a_record`` gates it on
        ``step_records`` alone, so a step that only read or browsed costs no extra
        ledger query: the record can't change from a read.

        Appended to the TOOL result, never edited into the system prompt: the system
        prompt is this run's prefix, and editing it mid-run would invalidate a local
        build's prefix cache on every later step.  A later step's addendum states the
        record AS OF NOW — the whole picture, not a delta — so a run with two writes
        never leaves an earlier message understating what has landed."""
        if not self._may_change_a_record(step_records):
            return
        addendum = self._writes_landed_frame(records, run_id)
        if addendum:
            self._append_to_last_tool_result(messages, addendum)

    @staticmethod
    def _may_change_a_record(step_records: list[ToolCallRecord]) -> bool:
        """Whether this step's OWN calls could have moved the writes record — the
        same tool mark the record is read off (``Tool.asks_to_store``), so a step
        that only read or browsed is never worth a ledger query: the record can't
        change from a read."""
        return any(Tool.asks_to_store(record.tool) for record in step_records)

    @staticmethod
    def _append_to_last_tool_result(messages: list[LlmMessage], addendum: str) -> None:
        """Append ``addendum`` to the content of the LAST message in ``messages`` — the
        step's own tool result, since this is only ever called right after a tool step's
        results have been absorbed into it (#2473)."""
        last = messages[-1]
        messages[-1] = last.model_copy(update={"content": f"{last.content}\n\n{addendum}"})

    def _writes_landed_frame(self, records: list[ToolCallRecord], run_id: str) -> str | None:
        """The record of what this run SAVED, rendered for the reply to be written after
        (#1946/#2185) — ``None`` when the run neither saved anything nor tried to.

        What landed is read off the ledger's own entry stamps (``last_written_by_run_id``),
        so it is a read of the store rather than a tally of the run's tool calls: a write
        drawn and then discarded by the reroll guard, and one the change-gate refused, are
        both absent there and present in the run's memory of itself.

        When nothing landed, the record is each call this run made to a tool that DECLARES
        it keeps entries (``Tool.stores_entries``), with that tool's own answer — so a run
        whose every try was refused is told so instead of being left to narrate the save it
        meant.  A run that never tried gets no record: there is no ledger fact to report,
        and deciding it SHOULD have tried would mean reading the ask's words.  Nothing in
        the path names a tool, so a plugin's verb is covered by declaring the mark."""
        landed = render_writes_landed(
            self.db.memories.entries_written_by_run(run_id, exclude_system_logs=False)
        )
        record = landed or render_nothing_landed(self._store_attempts(records))
        if record is None:
            return None
        return Prompt.WRITES_LANDED_RECORD.format(record=record)

    @staticmethod
    def _store_attempts(records: list[ToolCallRecord]) -> list[ToolCallRecord]:
        """This run's calls that asked the store to keep an entry, in the order made —
        read off each call's own tool's declared mark, never off its name."""
        return [record for record in records if Tool.asks_to_store(record.tool)]

    # ── Message handling ───────────────────────────────────────────────

    async def handle(
        self,
        content: str,
        sender: str,
        *,
        call_link: CallLink,
        inbox: TurnInbox,
        images: list[str] | None = None,
        page_context: PageContext | None = None,
        thread_parent_id: int | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
        on_interim_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, runs agentic loop. Images ride the turn's own user
        message (#2635).

        ``call_link`` (#2458) is REQUIRED — a top-level run starter in its own right,
        built by the channel (``CallLink.top(run_id)``) before this turn is even
        dispatched, with no ambient state and no fallback.  ``run_id`` (for DB/
        mutation purposes — stamped on every promptlog row, recorded on any
        collection the turn creates via ``created_by_run_id``, #1566) is read off it.

        ``inbox`` (#2985) is REQUIRED, no default — the turn's join queue, drained
        by the loop after each tool step.  A foreground chat turn passes its own;
        ``NO_INBOX`` for a caller with no live user to drain for.

        ``thread_parent_id`` is the quote-reply's resolved parent row id — resolved
        by the channel from the quote's own anchor, never re-guessed here — and
        selects the thread history for the turn (``get_history``).

        ``on_interim_message`` (#2837) sends a step's own prose to the user the
        moment it is validated, before that step's tool calls run — the live
        channel's own hook, built with a user and a run to send to.  ``None`` for
        every other caller (``handle_job_report``'s delivery turn has no live
        channel dispatch of its own; the drainer gates and queues its reply).

        There is no conversation machine (#2455 naive baseline): every turn runs the
        same ``Prompt.CONVERSATION_PROMPT`` — there is no second state to pick an
        instruction for.
        """
        logger.info("Handling message from %s (conversation mode)", sender)
        return await self._dispatch_turn(
            content,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.USER_MESSAGE,
            progress_scope=ProgressScope.FOREGROUND,
            inbox=inbox,
            images=images,
            page_context=page_context,
            thread_parent_id=thread_parent_id,
            on_tool_start=on_tool_start,
            on_progress=on_progress,
            on_interim_message=on_interim_message,
        )

    async def handle_job_report(
        self, report: JobReport, sender: str, *, call_link: CallLink
    ) -> ControllerResponse:
        """The system-originated delivery turn (#2656, trigger reworked #2974):
        a turn exactly like any other chat turn, given the finished job's report
        as a fact of her situation rather than an instruction.

        The input is the framework-rendered report (``render_job_report``) — the
        job's name, what it is for, why it ran, and the FULL content of every entry
        the cycle wrote — never anything model-written.  It rides as the live turn's
        own USER message prefixed with ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX``
        (#2974) — a notification stating only what happened, never words from the
        user; she decides for herself whether, how, and how many times to
        reply, exactly as on any ordinary turn, over the same ambient state: her
        persona and voice, the standing rules, the conversation history (a past
        report is there to be read, and a formatting ask in it holds here as
        anywhere), and the full chat tool surface — so "show me yesterday's" shape
        follow-ups stay one ``collection_get`` away from the rendered job name.

        ``sender`` is the user the message is written for (the primary sender, read
        by the caller); ``call_link`` (#2458) is REQUIRED, as on every run starter.
        Whatever she sends is NOT delivered here — the caller (the send-queue
        drainer) gates it and queues it, so it is logged and delivered exactly like
        any other autonomous send."""
        # The delivery turn's own link table (#2717): the report's entries keep
        # real URLs, so the render converts them to THIS turn's handles — the
        # same table the turn is dispatched with, so a number the model cites
        # back expands to its URL at send time.
        handles = HandleTable()
        prompt = _system_notification_trigger(render_job_report(report, handles))
        logger.info("Handling job report for %s (delivery turn)", report.collection)
        return await self._dispatch_turn(
            prompt,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.JOB_REPORT,
            progress_scope=ProgressScope.BACKGROUND,
            inbox=NO_INBOX,
            handles=handles,
        )

    async def handle_sandbox_report(
        self, report: SandboxReport, sender: str, *, call_link: CallLink
    ) -> ControllerResponse:
        """The system-originated delivery turn for a SANDBOX preview (#2964,
        trigger reworked #2974) — the sibling of ``handle_job_report``.

        The input is the framework-rendered report (``render_sandbox_report``) — the
        job's name, what it is for, and the previewed cycle's own tool-call trace
        (never written entries — nothing it did was real) — never anything
        model-written.  It rides as the live turn's own USER message prefixed with
        ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX`` (#2974), stating only what
        the preview showed; everything else the turn needs is the same ambient
        state any delivery turn gets.

        ``sender``/``call_link`` are exactly as ``handle_job_report``'s.  Whatever
        she sends is NOT delivered here — the caller (the send-queue drainer) gates
        it and queues it, logged and delivered exactly like any other autonomous
        send.  No link table: ``render_sandbox_report``'s trace carries no handle
        conversion of its own (the same shape ``run_get``/``run_list`` already
        render their own trace in)."""
        prompt = _system_notification_trigger(render_sandbox_report(report))
        logger.info("Handling sandbox report for %s (delivery turn)", report.collection)
        return await self._dispatch_turn(
            prompt,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.SANDBOX_REPORT,
            progress_scope=ProgressScope.BACKGROUND,
            inbox=NO_INBOX,
        )

    async def handle_job_event_report(
        self,
        report: JobRetiredReport | JobFailedReport,
        sender: str,
        *,
        call_link: CallLink,
    ) -> ControllerResponse:
        """The system-originated delivery turn for the autonomous job events
        (#2975) — the sibling of ``handle_job_report``: a job the system archived
        (a ``JobRetiredReport`` — the run limit spent out or the expiry passed),
        or a cadence run that failed for good (a ``JobFailedReport`` — the
        occurrence-retry budget spent out).

        The input is the framework-rendered fact set (``render_job_retired_report``
        / ``render_job_failed_report``) — assembled from the chokepoints that
        already recorded the event (the mutation ledger's system-actor archive
        note, the failed run's own outcome stamp), never anything model-written.
        It rides as the live turn's own USER message prefixed with
        ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX`` (#2974), stating only what
        happened; everything else the turn needs is the same ambient state any
        delivery turn gets.  Dispatched on the report's own type — one turn, two
        fact sets, the same freedom to send nothing or several messages.

        ``sender``/``call_link`` are exactly as ``handle_job_report``'s.  Whatever
        she sends is NOT delivered here — the caller (the send-queue drainer) gates
        it and queues it, logged and delivered exactly like any other autonomous
        send.  No link table: the facts carry no URLs."""
        if isinstance(report, JobRetiredReport):
            facts = render_job_retired_report(report)
        else:
            facts = render_job_failed_report(report)
        prompt = _system_notification_trigger(facts)
        logger.info("Handling job event for %s (delivery turn)", report.collection)
        return await self._dispatch_turn(
            prompt,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.JOB_EVENT,
            progress_scope=ProgressScope.BACKGROUND,
            inbox=NO_INBOX,
        )

    async def handle_changelog_report(
        self,
        entries: list[ChangelogEntry],
        sender: str,
        *,
        call_link: CallLink,
    ) -> ControllerResponse:
        """The startup delivery turn (#2846, #2890, trigger reworked #2974): a turn
        exactly like any other chat turn, given this deploy's own changelog entries
        as a fact of her situation — or, when there are none, the fact that she's
        just restarted and nothing changed.

        The input is the framework-rendered entries (``render_changelog_announcement``)
        — each PR number and its text, never anything model-written, or (with no
        entries) the framework's own "nothing changed" render.  The notice announces
        only the changes that deployed (#3003) — the preflight's soft warnings ride
        nothing.  It rides as the live
        turn's own USER message prefixed with ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX``
        (#2974), stating only what happened — never an instruction to send anything —
        so she decides for herself whether, how, and how many times to reply, over
        the same ambient state any chat turn
        gets: her persona and voice, the standing rules, and the full chat tool
        surface — so a change that names something she can look at stays reachable
        before she writes about it.

        ``sender`` is the user the message is written for; ``call_link`` (#2458) is
        REQUIRED, as on every run starter.  Whatever she sends is NOT delivered here
        — the caller (``Penny._send_changelog_announcement``) gates and enqueues it,
        so it is logged and delivered exactly like any other autonomous send.  No
        link table: changelog entries are authored, reviewed prose about Penny's own
        tools — never user-sourced content carrying real URLs."""
        prompt = _system_notification_trigger(render_changelog_announcement(entries))
        logger.info("Handling changelog announcement (%d entries)", len(entries))
        return await self._dispatch_turn(
            prompt,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.CHANGELOG,
            progress_scope=ProgressScope.BACKGROUND,
            inbox=NO_INBOX,
        )

    async def _dispatch_turn(
        self,
        prompt: str,
        sender: str,
        *,
        call_link: CallLink,
        prompt_type: ChatPromptType,
        progress_scope: str,
        images: list[str] | None = None,
        page_context: PageContext | None = None,
        thread_parent_id: int | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
        on_interim_message: Callable[[str], Awaitable[None]] | None = None,
        handles: HandleTable | None = None,
        inbox: TurnInbox,
    ) -> ControllerResponse:
        """The one turn both entry points ride — the template method under
        ``handle`` and ``handle_job_report``: per-turn state held for the span of
        the turn and cleared in its ``finally``, context built (thread history when
        a quoted parent resolves, the ambient self-state header always), tools
        installed, loop run.

        The two entry points differ only in what they hand in: a user turn carries
        its attachments, its page context and its progress plumbing, runs in the
        foreground scope, and its ``prompt`` is the user's own words; a delivery
        turn carries none of that, runs in the background scope, and its
        ``prompt`` is a system notification (``_system_notification_trigger``,
        #2974) stating only what happened.  Beyond that one text, it is the SAME
        turn — same history, same self-state, same full tool surface, same freedom
        to call tools, send nothing, or send several messages. There is no second
        state to pick an instruction for."""
        self._current_user = sender
        self._pending_page_context = page_context
        run_id = call_link.run_id
        # The session anchor (#3078) — read ONCE here, off whatever the store
        # already holds (this turn's own live row may already be stored, on
        # the flows that write it before their run), and passed explicitly to
        # both reads below instead of each re-reading it. The chat pre-warm's
        # own anchor (``MessageStore.anchor_for_arrival``) is the sibling read
        # for a turn that has not arrived yet.
        anchor = self.db.messages.session_anchor(
            sender, int(self.config.runtime.CHAT_SESSION_IDLE_SECONDS)
        )
        # The related earlier exchanges (#3010) — computed once per turn here,
        # where the live message's own embedding happens, and read by
        # ``_dynamic_system_section`` on every loop step.  A delivery turn's
        # trigger is the framework's, never the user's words, so no retrieval
        # runs for one (``_related_exchanges_section`` gates on the prompt type).
        self._related_exchanges = await self._related_exchanges_section(prompt, sender, prompt_type)
        # The once-per-session state message (#3089) — computed once per turn
        # here too, read by ``_build_messages`` to insert right after the
        # leading system message.
        self._session_state = self._session_state_section(sender, anchor)
        # One numbered link table per turn (#2717) — created here, beside the run
        # identity, and passed to the history render, the tool surface and the
        # loop alike; the reply is expanded through it before it is returned.
        table = handles if handles is not None else HandleTable()
        try:
            self._current_message = prompt
            history = self._history_for_turn(sender, thread_parent_id, anchor, table)
            # A chat turn is never sandboxed (#2964) — only a collector cycle can be.
            self._install_tools(self.get_tools(run_id=run_id, handles=table), forced_plan=False)
            system_prompt = await self._build_system_prompt(sender)
            response = await self.run(
                prompt=prompt,
                max_steps=self.get_max_steps(),
                call_link=call_link,
                history=history,
                system_prompt=system_prompt,
                images=images,
                on_tool_start=on_tool_start,
                on_progress=on_progress,
                on_interim_message=on_interim_message,
                progress_scope=progress_scope,
                prompt_type=prompt_type,
                handles=table,
                inbox=inbox,
            )
            # Send-time expansion (#2717): every handle the reply cites becomes
            # its URL, so the conversation record and the wire carry real links.
            if response.answer:
                response = response.model_copy(update={"answer": table.expand(response.answer)})
            return response
        finally:
            self._current_user = None
            self._pending_page_context = None
            self._current_message = None
            self._related_exchanges = None
            self._conversation_map = None
            self._session_state = None

    def _record_turn_failure(self, run_id: str, reason: str) -> None:
        """Stamp a chat turn that needed the honest-fallback machinery (#2355 part
        D), the way a collector cycle stamps its own run — ``Agent``'s default is a
        no-op, since a chat turn has no second, richer stamp coming (unlike a
        collector cycle): this is the only one it gets.  ``run_target`` stays
        unset — a chat run names no collection, which is exactly what keeps it out
        of the collector-only activity queries (``recent_collector_runs``) and
        routes it instead through ``recent_failed_chat_turns``, the self-state
        header's own read of this stamp."""
        self.db.messages.set_run_outcome(run_id, RunOutcome.FAILED.value, reason)

    # ── Message building ────────────────────────────────────────────────

    def _build_session_prefix_messages(
        self,
        history: list[HistoryTurn] | None,
        system_prompt: str | None,
        session_state: str | None,
    ) -> list[LlmMessage]:
        """The reused PREFIX half (#3078): the leading system message, the
        once-per-session state message, and history — the SAME method the
        real turn (``_build_messages`` below) and the chat pre-warm both call.

        A DIFFERENT NAME from the base ``Agent._build_prefix_messages`` on
        purpose — not an override of it (which would need the SAME
        signature; a required third parameter here would violate the
        Liskov substitution rule the base class's own internal call relies
        on): this method calls ``super()._build_prefix_messages`` itself and
        is never reached polymorphically through ``Agent._build_messages``.

        ``session_state`` is a REQUIRED, explicit parameter rather than an
        implicit read of ``self._session_state`` — the pre-warm computes its
        own text (off its own, possibly-rotated anchor) and passes it here
        directly, never touching ``self`` at all, so nothing races a
        foreground turn's own ``self._session_state`` (#3078). The real
        turn passes its own value the same way, from ``_build_messages``.

        ``super()._build_prefix_messages`` returns ``[system][…history]``
        (#2756) — the leading system message, which carries the tool
        schemas right behind it (``raw_template.render_prompt``), then the
        growing history. ``session_state`` is inserted right after index 0
        and before any history, so the shape is
        ``[system][session-state][…history]``: a SECOND leading system
        message, confirmed (#3089) to render as its OWN system turn on
        gemma's raw-template transport (the one every provider — prod's
        oMLX and OpenRouter alike — renders from, #2829), with no tool
        re-declaration, regardless of what sits on either side of it. Only
        the degraded chat-completions FALLBACK (used when the
        raw-template faithfulness check fails) merges two leading system
        messages into one before rendering — oMLX's own request
        normalisation, confirmed by reading ``omlx/api/utils.py``'s
        ``prepare_system_messages_for_template`` — which loses the cache
        boundary between them but not the content (joined with a blank
        line), and only on that named degraded path."""
        messages = super()._build_prefix_messages(history, system_prompt)
        if session_state:
            messages.insert(1, LlmMessage(role=MessageRole.SYSTEM, content=session_state))
        return messages

    def _build_messages(
        self,
        prompt: str,
        history: list[HistoryTurn] | None = None,
        system_prompt: str | None = None,
        images: list[str] | None = None,
        handles: HandleTable | None = None,
    ) -> list[LlmMessage]:
        """Build messages, inserting the session-state message and injecting
        page context as a synthetic tools result.

        Composes the reused prefix (``_build_session_prefix_messages`` above,
        passed THIS turn's own ``self._session_state`` explicitly — never
        read implicitly inside it) with the per-turn dynamic system message
        and the live turn (#2756). The synthetic page-context assistant-call
        + tool-result appended below land AFTER everything, so the full
        shape is
        ``[system][session-state][…history][state][live][assistant][tool]``.
        Page context only ever rides an ordinary turn in practice — a
        delivery turn never carries one."""
        messages = self._build_session_prefix_messages(history, system_prompt, self._session_state)
        dynamic = self._dynamic_system_section(handles=handles)
        if dynamic:
            messages.append(LlmMessage(role=MessageRole.SYSTEM, content=dynamic))
        messages.append(LlmMessage(role=MessageRole.USER, content=prompt, images=images or None))
        if self._pending_page_context:
            self._inject_page_context(
                messages,
                self._pending_page_context,
                handles if handles is not None else HandleTable(),
            )
        return messages

    @staticmethod
    def _inject_page_context(
        messages: list[LlmMessage], page_context: PageContext, handles: HandleTable
    ) -> None:
        """Inject a synthetic search call + result for page context.

        Uses the BrowseTool format so the synthetic history matches the tool
        the model actually sees in its tool definitions, the standard
        ``tool_call_id`` envelope (linked to the synthetic call by
        ``PAGE_CONTEXT_TOOL_CALL_ID``), and the same tagged first-person envelope
        every real tool result gets (via ``Agent._frame_injected_result``) — with
        bespoke narration (``Prompt.PAGE_CONTEXT_NARRATION``) naming this as the page
        the user is currently viewing, so the content is unmistakably a browse result,
        never a fresh instruction to the model.
        """
        if not page_context.text:
            return

        # A URL the model READS always arrives as a handle (#2717): the page's
        # own address and every link in its text render as this turn's handles,
        # so the model cites the page — or re-reads it — by number.
        number = handles.register(page_context.url)
        page_content = (
            f"Title: {page_context.title}\nURL: {handles.plain(number)}\n\n"
            f"{handles.convert(page_context.text)}\n\n{PAGE_CITATION_LINE}"
        )

        # Assistant "called" fetch with the page's handle in queries — built
        # through the typed models like every other tool call, so its
        # ``arguments`` wire-encodes as a JSON string (#2604; it was a bare dict
        # before this stage).
        messages.append(
            LlmMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    LlmToolCall(
                        id=ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID,
                        function=LlmToolCallFunction(
                            name=BrowseTool.name,
                            arguments={"queries": [handles.plain(number)]},
                        ),
                    )
                ],
            )
        )
        # Tool "returned" the page content — framed like every real tool result via
        # the same tagged first-person envelope, but with bespoke narration naming this
        # for what it is: the page the user is currently viewing, not a search Penny ran.
        messages.append(
            LlmMessage(
                role=MessageRole.TOOL,
                content=Agent._frame_injected_result(
                    BrowseTool.name,
                    Prompt.PAGE_CONTEXT_NARRATION,
                    page_content,
                ),
                tool_call_id=ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID,
            )
        )

    async def _inject_joined_page_context(
        self, messages: list[LlmMessage], page_context: PageContext, handles: HandleTable
    ) -> None:
        """The page a JOINED message was sent from (#2985), rendered by the same
        synthetic browse-pair builder the triggering message's own context uses —
        the join loses nothing an ordinary user message carries."""
        if page_context:
            self._inject_page_context(messages, page_context, handles)

    # ── System prompt ──────────────────────────────────────────────────────

    async def _build_system_prompt(
        self,
        user: str | None,
        instructions: str | None = None,
        *,
        handles: HandleTable | None = None,
    ) -> str:
        """Identity + standing rules + instructions — the LEADING, DEPLOY-STABLE
        system message (#2745; reordered relative to history/state #2756;
        Jobs/the store map/the conversation map moved out into their own
        once-per-session message, #3089 — previously carried here, #2814).

        This message is the prefix the model server reuses across EVERY
        session (along with the tool schemas gemma4's renderer puts right
        after it), so it carries only what changes on a deploy or a config
        edit: Penny's persona/voice, then the user's standing rules — in the
        instruction part, right after the voice sections, where she reads
        instructions — then the instructions.  Nothing here changes on a job
        run, a memory write, or even an idle rotation — only #3089's own
        session-state message, right behind this one, does that.

        The ambient inversion (#1555): the chat prompt injects no
        **speculative user-content recall** at all.  User content is fetched on
        demand via the tools the header's pointers name (anchored by the user's
        message).  Background agents keep the base envelope (profile + inventory)
        and never see the header — it is the chat entry point's opening state.
        """
        return "\n\n".join(
            section
            for section in [
                self._identity_section(),
                standing_rules_instruction(self.db),
                self._instructions_section(instructions),
            ]
            if section
        )

    def _session_state_section(self, user: str | None, anchor: SessionAnchor) -> str | None:
        """Jobs + Your memory + Recent activity + About the user
        (``SelfStateHeader.render``) + the conversation map — the ONE
        session-state system message (#3089), inserted by ``_build_messages``
        right after the leading system message's tool schemas and before the
        growing history.  Bounded by the session's own start
        (``_session_as_of``), so it is byte-identical turn to turn within one
        session — a chat-side write, a job run, or a mutation already shows up
        as a tool result or a delivery turn in history, so repeating it here
        would be the duplication #3089 rules out.  Only an idle rotation (a
        NEW session, a new ``as_of``) changes what this says.

        ``anchor`` (#3078) is REQUIRED, no default (``CLAUDE.md``, "pass
        parameters, don't swap state") — the caller's own one read of the
        session anchor (``_dispatch_turn``, which reads it ONCE and passes
        it here and to ``get_history`` rather than each re-reading it).

        ``None`` for a caller with no user (there's no session to bound
        against, and no store map worth showing without one) — a delivery
        turn always has one (the primary sender), so this is never absent in
        practice.

        Recent activity's own emission cutoff (``map_horizon_start``, #3079)
        reads the LIVE clock, same as the conversation map's own day-fold
        does — the one narrow exception to the ``as_of`` bound above, shared
        with and no wider than the map's own already-accepted day-rotation
        (at most once a day, never mid-turn).  The conversation map itself is
        THIS turn's one ``ConversationMap`` instance (#3085,
        ``_conversation_map_for``), the same one ``_related_exchanges_section``
        reads its one-to-one line ids off, never a second map."""
        if user is None:
            return None
        return self._render_session_state(user, anchor, self._conversation_map_for(user))

    def _render_session_state(
        self, user: str, anchor: SessionAnchor, conversation_map: ConversationMap
    ) -> str | None:
        """The session-state text for an explicit anchor and map (#3078) — the
        ONE render (``SelfStateHeader`` + ``ConversationMap``) both
        ``_session_state_section`` above (passing THIS turn's own cached
        ``_conversation_map_for`` instance) and the chat pre-warm (passing a
        fresh, local one — ``ConversationMap`` caches nothing itself, so a
        fresh instance renders byte-identical text; only the INSTANCE isn't
        shared, never the data) call, so there is no second renderer."""
        as_of = self._session_as_of(anchor)
        map_horizon = map_horizon_start(self.db, self.config.runtime)
        return "\n\n".join(
            section
            for section in [
                SelfStateHeader(self.db, user, as_of, map_horizon=map_horizon).render(),
                conversation_map.render(),
            ]
            if section
        )

    def _conversation_map_for(self, user: str | None) -> ConversationMap:
        """This turn's ONE ``ConversationMap`` instance (#3085): built here on
        first use and cached on ``self`` for the rest of the turn, so this
        method and ``_related_exchanges_section`` — whichever runs first in
        ``_dispatch_turn`` — never build two.  ``ConversationMap`` itself
        caches nothing (every read is fresh off ``self.db``), so reusing the
        instance across this turn's two reads, or across repeated calls in a
        test, changes nothing about what either one returns."""
        if self._conversation_map is None:
            self._conversation_map = ConversationMap(self.db, user, self.config.runtime)
        return self._conversation_map

    def _session_as_of(self, anchor: SessionAnchor) -> datetime:
        """The instant every once-per-session section is bounded by (#3089):
        the current session's own first turn — ``anchor.session_started_at``,
        a pure read of the CALLER's own anchor (#3078: never a second
        ``MessageStore.session_anchor``/``anchor_for_arrival`` read here, so
        the session-state message and the history window — and, for the chat
        pre-warm, its own hypothetical "if the next turn arrived now" —
        always agree on the same anchor their caller already read once).
        Falls back to the live clock only when the store holds no turn at all
        yet (``session_started_at`` is ``None``) — there is nothing earlier to
        freeze against on the very first turn ever.

        Always timezone-AWARE (``stored_as_utc``): ``session_started_at`` is
        read straight off a stored column (naive IS UTC, same convention
        every other reader of a stored stamp follows), and ``Clock.now()`` is
        already aware — so every comparison this bound feeds (``MemoryStore.
        list_all``'s ``row.updated_at`` check, ``describe_next_run``'s
        ``now``) compares two aware values, never a naive one against an
        aware one."""
        return stored_as_utc(anchor.session_started_at or self.db.clock.now())

    def _dynamic_system_section(self, handles: HandleTable | None = None) -> str:
        """The per-turn system message's content, placed right BEFORE the
        live user turn (#2756): the clock, the page hint, then the related
        earlier exchanges (#3010, computed once per turn in
        ``_dispatch_turn``) — in that order, everything that may change
        between turns, none of it able to touch the reused prefix (the
        leading system message, the session-state message, and history) that
        precedes it."""
        return "\n\n".join(
            section
            for section in [
                current_datetime_line(self.db),
                self._page_hint_section(),
                self._related_exchanges,
            ]
            if section
        )

    async def _related_exchanges_section(
        self, prompt: str, sender: str, prompt_type: ChatPromptType
    ) -> str | None:
        """The turn's related earlier exchanges, or ``None`` when there are none
        (#3010).  Keyed to the user's own words: an ordinary user turn only — a
        delivery turn's trigger is the framework's notification, never the user
        asking, so no retrieval runs for one.  Embeds the live message once,
        here, so every later loop step reads the section this computed.

        Reads the conversation map's own one-to-one line ids off THIS turn's
        ONE ``ConversationMap`` instance (#3085, ``_conversation_map_for`` —
        the same one ``_build_system_prompt`` renders from, never a second
        map) and hands them to ``RelatedExchanges`` so a match whose user
        words the map already shows whole never renders them a second time."""
        if prompt_type is not ChatPromptType.USER_MESSAGE:
            return None
        if not prompt.strip():
            return None
        try:
            vectors = await self._embedding_model_client.embed([prompt])
        except LlmError as error:
            logger.warning("Related-exchanges retrieval skipped — embed failed: %s", error)
            return None
        map_one_to_one_ids = self._conversation_map_for(sender).one_to_one_user_ids()
        rendered = RelatedExchanges(
            self.db,
            sender,
            self.config.runtime,
            vectors[0],
            map_one_to_one_ids,
        ).render()
        if not rendered:
            return None
        return rendered

    def _page_hint_section(self) -> str | None:
        """Minimal hint about what page the user is currently viewing."""
        context = self._pending_page_context
        if not context or not context.url:
            return None
        return f"### Current Browser Page\n{context.title}\n{context.url}"

    def get_history(
        self,
        user: str,
        thread_parent_id: int | None = None,
        *,
        anchor: SessionAnchor,
    ) -> list[HistoryTurn] | None:
        """Recent conversation messages for chat continuity.

        When a quote-reply's thread parent is known (resolved by the channel
        from the quote's own anchor, never re-guessed here), walks that
        specific thread, capped like the standard window — the last
        ``CHAT_HISTORY_TURNS`` user turns with their replies (#2746), with
        NO anchor (ruled on #3077: a quote-reply's chain has no session of
        its own, so it keeps its own smaller, short-chain approximation).
        Falls back to the standard recent-message window.

        ``anchor`` (#3078) is REQUIRED, no default (``CLAUDE.md``, "pass
        parameters, don't swap state") — the caller's own one read of the
        session anchor (``_dispatch_turn``'s ``session_anchor``, shared with
        ``_session_state_section``; the chat pre-warm's own
        ``anchor_for_arrival``), never a second read taken here. Keyword-only
        so ``thread_parent_id`` keeps its own optional, positional shape.
        """
        if thread_parent_id is not None:
            thread_history = self.db.messages.get_thread_history(thread_parent_id)
            if thread_history:
                return self._trim_history_window(thread_history)
        return self._build_conversation(user, anchor)

    def _history_for_turn(
        self,
        sender: str,
        thread_parent_id: int | None,
        anchor: SessionAnchor,
        table: HandleTable,
    ) -> list[HistoryTurn] | None:
        """This turn's history, with every past assistant reply's real URLs
        converted to THIS turn's own handles (#2717) — the ONE method both
        the real turn (``_dispatch_turn``) and the chat pre-warm
        (``build_prewarm_prefix``, passing a fresh ``HandleTable()``) call,
        so the warm's rendered history can never diverge from the real
        turn's at the first linked reply in it. A reply citing a link used
        to render raw URLs in the warm's own history (it called
        ``get_history`` directly, skipping this conversion), so its prefix
        missed the server's cache from that point on in prod, where almost
        every reply cites one (#3078 fix). The handle numbering is
        deterministic per table — the same history walked in the same
        order always assigns the same numbers, whichever fresh table does
        the walking, so the warm's own ``HandleTable()`` and the real
        turn's agree."""
        history = self.get_history(sender, thread_parent_id, anchor=anchor)
        if not history:
            return history
        return [
            turn._replace(
                content=table.convert(turn.content)
                if turn.role == str(MessageRole.ASSISTANT)
                else turn.content
            )
            for turn in history
        ]

    # ── Chat pre-warm (#3078) ────────────────────────────────────────────

    async def build_prewarm_prefix(
        self, user: str, idle_seconds: int, now: datetime
    ) -> tuple[list[LlmMessage], list[dict]]:
        """The next chat turn's rendered PREFIX — the leading system message,
        the session-state message, and history — built through the exact
        render methods a real turn's own ``_build_messages`` uses (#3078),
        for the chat pre-warm's background task.

        Touches NO shared agent state: a fresh, local ``ConversationMap``
        (never ``self._conversation_map_for``) and a fresh, local
        ``HandleTable``/``ToolRegistry`` (never ``self._install_tools`` /
        ``self._tool_registry``), and no write to ``self._session_state`` /
        ``self._related_exchanges`` / ``self._current_message`` anywhere.  A
        foreground turn starting mid-build and this call racing it can
        never corrupt either one's state — there is nothing of this call's
        own left lying on ``self`` to corrupt.

        ``now`` is the one clock read: the anchor is "the next turn's own,
        if it arrived at ``now``" (``MessageStore.anchor_for_arrival``),
        which structurally comes back already rotated once the idle gap has
        passed — no separate rotated/mid-session branch anywhere here.

        The history and the tool schemas are built through the SAME methods
        a real turn's own ``_dispatch_turn`` calls (``_history_for_turn``,
        ``_tool_schemas``) over a fresh ``HandleTable()`` — a real turn's
        past replies are stored with real URLs, converted to numbered
        handles per turn (#2717); skipping that conversion here would have
        rendered raw URLs in the warm's history where the real turn renders
        handles, diverging the two prefixes at the first linked reply.

        Excludes the per-turn dynamic system message and the live turn on
        purpose (the caller renders only up to here): those differ every
        turn and sit after the cache boundary, so warming them would warm
        nothing reusable.
        """
        anchor = self.db.messages.anchor_for_arrival(user, idle_seconds, now)
        session_state = self._render_session_state(
            user, anchor, ConversationMap(self.db, user, self.config.runtime)
        )
        table = HandleTable()
        history = self._history_for_turn(user, None, anchor, table)
        system_prompt = await self._build_system_prompt(user)
        messages = self._build_session_prefix_messages(history, system_prompt, session_state)
        return messages, self._tool_schemas(table)

    def _tool_schemas(self, table: HandleTable) -> list[dict]:
        """The chat pre-warm's own read of the tool-schema block (#3078):
        the SAME registry mechanics a real turn's own ``_install_tools`` +
        ``self._tool_registry.get_ollama_tools()`` apply
        (``Agent._tool_registry_for``), run on a FRESH, local
        ``ToolRegistry`` instead of ``self._tool_registry`` — shared agent
        state a background warm must never touch while a foreground turn
        might be installing its own. Never a parallel list comprehension
        that only happens to match whatever ordering/dedup-by-name the
        registry applies. ``table`` mirrors the real turn's own
        ``get_tools(run_id=run_id, handles=table)`` call exactly, even
        though no tool's declared schema reads the handle table today."""
        return self._tool_registry_for(self.get_tools(handles=table)).get_ollama_tools()

    # ── Memory description embedding ──────────────────────────────────────

    async def embed_description(self, text: str) -> list[float] | None:
        """Embed a memory's description into its relevance anchor.

        Exposed for the browser + iOS channels' memory create/edit path, which —
        unlike the chat tool surface — compute embeddings through the agent rather
        than a memory tool.  The anchor stays live for on-demand relevance reads
        (``read_similar``) even though the chat prompt no longer injects ambient
        recall (the inversion, #1555).  Returns None only on a transient embed
        failure.
        """
        try:
            vecs = await self._embedding_model_client.embed([text])
            return vecs[0]
        except LlmError:
            logger.warning("Failed to embed memory description")
            return None

    # ── Vision ────────────────────────────────────────────────────────────

    # Retired with the caption step (#2635): an image rides the turn's own user
    # message (``Agent.run(images=…)``) for the chat model to read directly —
    # there is no separate caption call and no vision model client.
