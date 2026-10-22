"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with web browsing and memory tools.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.agents.base import Agent, InvalidDraw, ProgressCallback
from penny.agents.models import ControllerResponse, MessageRole, ToolCallRecord
from penny.agents.self_state import SelfStateHeader, standing_rules_instruction
from penny.channels.base import PageContext
from penny.constants import ChatPromptType, RunOutcome
from penny.datetime_utils import current_datetime_line
from penny.llm.call_link import CallLink
from penny.llm.models import LlmError, LlmMessage, LlmToolCall, LlmToolCallFunction
from penny.notification import JobReport, render_job_report
from penny.prompts import Prompt
from penny.text_validity import is_call_as_text_bail, is_call_fragment_reply, is_empty_draw
from penny.tools import Tool
from penny.tools.browse import BrowseTool
from penny.tools.generate_image import GenerateImageTool
from penny.tools.link_handles import PAGE_CITATION_LINE, HandleTable
from penny.tools.memory_tools import render_nothing_landed, render_writes_landed
from penny.tools.notifications import NotificationsMuteTool, NotificationsUnmuteTool
from penny.validation import ConditionKey

if TYPE_CHECKING:
    from penny.llm.image_client import OllamaImageClient

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Two context mechanisms, kept independent:

    - Operational self-state → the DYNAMIC system message (the second of the
      request's two system messages, #2745): a deterministic ``SelfStateHeader``
      (active mechanisms · recent activity · the store map · durable user
      facts), rendered from the registry + ledger, never a relevance guess.
      The speculative user-content recall is gone (the ambient inversion,
      #1555): user content is fetched on demand via the tools the header's
      pointers name.
    - Chat turns → messages array as alternating user/assistant turns
      via ``_build_conversation`` and ``history=``.

    The request is two system messages, static first (#2745): the first carries
    identity + the user's standing rules + instructions (never changes between
    turns, so the server reuses it and the tool schemas rendered after it); the
    second carries the clock, the page hint and the self-state header.  Every
    turn — image or not — runs the standard agentic loop; an image rides the
    turn's own user message (#2635) for the chat model to read.
    """

    name: str = "chat"
    system_prompt = Prompt.CONVERSATION_PROMPT
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
    #    call (``{"name": …, "arguments": …}``) or real args carrying the framework's
    #    ``reasoning`` field.  It would be delivered to the user as raw machinery.
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
        image_client: OllamaImageClient | None = None,
        email_tools_builder: Callable[[str, str], list[Tool]] | None = None,
        plugin_tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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

    def get_tools(
        self, run_id: str | None = None, handles: HandleTable | None = None
    ) -> list[Tool]:
        tools = super().get_tools(run_id, handles)
        # Notification mute/unmute is a chat-driven action over the MuteState row
        # (the retired /mute + /unmute commands), so both tools live on the chat
        # surface — the model dispatches to them from natural language.
        tools.append(NotificationsMuteTool(self.db))
        tools.append(NotificationsUnmuteTool(self.db))
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
        landed = render_writes_landed(self.db.memories.entries_written_by_run(run_id))
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
        images: list[str] | None = None,
        page_context: PageContext | None = None,
        thread_parent_id: int | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, runs agentic loop. Images ride the turn's own user
        message (#2635).

        ``call_link`` (#2458) is REQUIRED — a top-level run starter in its own right,
        built by the channel (``CallLink.top(run_id)``) before this turn is even
        dispatched, with no ambient state and no fallback.  ``run_id`` (for DB/
        mutation purposes — stamped on every promptlog row, recorded on any
        collection the turn creates via ``created_by_run_id``, #1566) is read off it.

        ``thread_parent_id`` is the quote-reply's resolved parent row id — resolved
        by the channel from the quote's own anchor, never re-guessed here — and
        selects the thread history for the turn (``get_history``).

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
            progress_scope="foreground",
            images=images,
            page_context=page_context,
            thread_parent_id=thread_parent_id,
            on_tool_start=on_tool_start,
            on_progress=on_progress,
        )

    async def handle_job_report(
        self, report: JobReport, sender: str, *, call_link: CallLink
    ) -> ControllerResponse:
        """The system-originated delivery turn (#2656): write the message that tells
        the user what a finished job's cycle found.

        The input is the framework-rendered report (``render_job_report``) — the
        job's name, what it is for, why it ran, and the FULL content of every entry
        the cycle wrote — never anything model-written.  Everything else the message
        needs is the turn's own ambient state: her persona and voice, the standing
        rules, the conversation history (a past report is there to be read, and a
        formatting ask in it holds here as anywhere), and the full chat tool surface
        — so "show me yesterday's" shape follow-ups stay one ``collection_get`` away
        from the rendered job name.

        ``sender`` is the user the message is written for (the primary sender, read
        by the caller); ``call_link`` (#2458) is REQUIRED, as on every run starter.
        The reply is NOT delivered here — the caller (the send-queue drainer) gates
        it and queues it, so it is logged and delivered exactly like any other
        autonomous send."""
        # The delivery turn's own link table (#2717): the report's entries keep
        # real URLs, so the render converts them to THIS turn's handles — the
        # same table the turn is dispatched with, so a number the model cites
        # back expands to its URL at send time.
        handles = HandleTable()
        prompt = render_job_report(report, handles)
        logger.info("Handling job report for %s (delivery turn)", report.collection)
        return await self._dispatch_turn(
            prompt,
            sender,
            call_link=call_link,
            prompt_type=ChatPromptType.JOB_REPORT,
            progress_scope="background",
            handles=handles,
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
        handles: HandleTable | None = None,
    ) -> ControllerResponse:
        """The one turn both entry points ride — the template method under
        ``handle`` and ``handle_job_report``: per-turn state held for the span of
        the turn and cleared in its ``finally``, context built (thread history when
        a quoted parent resolves, the ambient self-state header always), tools
        installed, loop run.

        The two entry points differ only in what they hand in: a user turn carries
        its attachments, its page context and its progress plumbing and runs in the
        foreground scope; a delivery turn carries none of that and runs in the
        background scope.  There is no second state to pick an instruction for."""
        self._current_user = sender
        self._pending_page_context = page_context
        run_id = call_link.run_id
        # One numbered link table per turn (#2717) — created here, beside the run
        # identity, and passed to the history render, the tool surface and the
        # loop alike; the reply is expanded through it before it is returned.
        table = handles if handles is not None else HandleTable()
        try:
            self._current_message = prompt
            history = self.get_history(sender, thread_parent_id=thread_parent_id)
            # Past replies are stored with real URLs — converted to this turn's
            # handles here, so a URL the model READS always arrives as a handle
            # (#2717).  User turns are the user's own words and stay as typed.
            if history:
                history = [
                    (
                        role,
                        table.convert(content) if role == str(MessageRole.ASSISTANT) else content,
                    )
                    for role, content in history
                ]
            self._install_tools(self.get_tools(run_id=run_id, handles=table))
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
                progress_scope=progress_scope,
                prompt_type=prompt_type,
                handles=table,
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

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        images: list[str] | None = None,
        handles: HandleTable | None = None,
    ) -> list[LlmMessage]:
        """Build messages, injecting page context as a synthetic tools result."""
        messages = super()._build_messages(prompt, history, system_prompt, images=images)
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
            f"Title: {page_context.title}\nURL: [{number}]\n\n"
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
                            arguments={"queries": [f"[{number}]"]},
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

    # ── System prompt ──────────────────────────────────────────────────────

    async def _build_system_prompt(
        self,
        user: str | None,
        instructions: str | None = None,
        *,
        handles: HandleTable | None = None,
    ) -> str:
        """Identity + standing rules + instructions — the STATIC system message
        (#2745).

        This message is the prefix the model server reuses across turns, so it
        carries only what never changes between them: Penny's persona/voice, then
        the user's standing rules — in the instruction part, right after the voice
        sections, where she reads instructions — then the instructions.  Everything
        per-turn — the clock, the current browser page, and the deterministic
        **self-state header** (active mechanisms · recent activity · the store map ·
        durable user facts, rendered from the registry (#1566) + ledger (#1560),
        never a relevance guess) — rides the SECOND system message
        (``_dynamic_system_section``).

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

    def _dynamic_system_section(self, handles: HandleTable | None = None) -> str:
        """The per-turn system message's content (#2745): the clock, the page
        hint, then Penny's own state — in that order, everything that may change
        between turns, none of it able to touch the reused static prefix ahead
        of it."""
        return "\n\n".join(
            section
            for section in [
                current_datetime_line(self.db),
                self._page_hint_section(),
                SelfStateHeader(self.db, self._current_user).render(),
            ]
            if section
        )

    def _page_hint_section(self) -> str | None:
        """Minimal hint about what page the user is currently viewing."""
        context = self._pending_page_context
        if not context or not context.url:
            return None
        return f"### Current Browser Page\n{context.title}\n{context.url}"

    def get_history(
        self, user: str, thread_parent_id: int | None = None
    ) -> list[tuple[str, str]] | None:
        """Recent conversation messages for chat continuity.

        When a quote-reply's thread parent is known (resolved by the channel
        from the quote's own anchor, never re-guessed here), walks that
        specific thread, capped like the standard window — the last
        ``CHAT_HISTORY_TURNS`` user turns with their replies (#2746).
        Falls back to the standard recent-message window.
        """
        if thread_parent_id is not None:
            thread_history = self.db.messages.get_thread_history(thread_parent_id)
            if thread_history:
                return self._trim_history_window(thread_history)
        return self._build_conversation(user)

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
