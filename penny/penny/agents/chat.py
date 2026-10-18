"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with web browsing and memory tools.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.agents.base import Agent, InvalidDraw, ProgressCallback
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.agents.self_state import SelfStateHeader
from penny.channels.base import PageContext
from penny.constants import ChatPromptType, PennyConstants, RunOutcome
from penny.datetime_utils import current_datetime_line
from penny.llm.call_link import CallLink
from penny.llm.models import LlmError, LlmMessage
from penny.prompts import Prompt
from penny.text_validity import is_call_as_text_bail, is_call_fragment_reply, is_empty_draw
from penny.tools import Tool
from penny.tools.browse import BrowseTool
from penny.tools.generate_image import GenerateImageTool
from penny.tools.memory_tools import render_nothing_landed, render_writes_landed
from penny.tools.notifications import NotificationsMuteTool, NotificationsUnmuteTool
from penny.validation import ConditionKey

if TYPE_CHECKING:
    from penny.llm.image_client import OllamaImageClient

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Two context mechanisms, kept independent:

    - Operational self-state → the dynamic tail of the system prompt: a
      deterministic ``SelfStateHeader`` (active mechanisms · recent activity ·
      the store map · durable user facts), rendered from the registry + ledger,
      never a relevance guess.  The speculative user-content recall is gone (the
      ambient inversion, #1555): user content is fetched on demand via the tools
      the header's pointers name.
    - Chat turns → messages array as alternating user/assistant turns
      via ``_build_conversation`` and ``history=``.

    The system prompt is identity + (page hint) + instructions + the self-state
    header.  Vision messages bypass the tool surface and use the captioner;
    everything else runs the standard agentic loop.
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

    def get_tools(self, run_id: str | None = None) -> list[Tool]:
        tools = super().get_tools(run_id)
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
        messages: list[dict],
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
    def _append_to_last_tool_result(messages: list[dict], addendum: str) -> None:
        """Append ``addendum`` to the content of the LAST message in ``messages`` — the
        step's own tool result, since this is only ever called right after a tool step's
        results have been absorbed into it (#2473)."""
        last = messages[-1]
        messages[-1] = {**last, "content": f"{last['content']}\n\n{addendum}"}

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
        quoted_text: str | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, processes images, runs agentic loop.

        ``call_link`` (#2458) is REQUIRED — a top-level run starter in its own right,
        built by the channel (``CallLink.top(run_id)``) before this turn is even
        dispatched, with no ambient state and no fallback.  ``run_id`` (for DB/
        mutation purposes — stamped on every promptlog row, recorded on any
        collection the turn creates via ``created_by_run_id``, #1566) is read off it.

        There is no conversation machine (#2455 naive baseline): every turn runs the
        same ``Prompt.CONVERSATION_PROMPT`` — there is no second state to pick an
        instruction for.
        """
        self._current_user = sender
        self._pending_page_context = page_context
        run_id = call_link.run_id
        try:
            content, has_images = await self._process_images(content, images, call_link)
            self._current_message = content
            history = self.get_history(sender, quoted_text=quoted_text)

            if has_images:
                logger.info("Handling vision message from %s", sender)
                self._install_tools([])
                system_prompt = await self._build_system_prompt(
                    sender, instructions=Prompt.VISION_RESPONSE_PROMPT
                )
                return await self.run(
                    prompt=content,
                    history=history,
                    max_steps=PennyConstants.VISION_MAX_STEPS,
                    call_link=call_link,
                    system_prompt=system_prompt,
                    on_progress=on_progress,
                    progress_scope="foreground",
                    prompt_type=ChatPromptType.VISION_MESSAGE,
                )

            logger.info("Handling message from %s (conversation mode)", sender)
            self._install_tools(self.get_tools(run_id=run_id))
            system_prompt = await self._build_system_prompt(sender)
            return await self.run(
                prompt=content,
                max_steps=self.get_max_steps(),
                call_link=call_link,
                history=history,
                system_prompt=system_prompt,
                on_tool_start=on_tool_start,
                on_progress=on_progress,
                progress_scope="foreground",
                prompt_type=ChatPromptType.USER_MESSAGE,
            )
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
    ) -> list[dict]:
        """Build messages, injecting page context as a synthetic tools result."""
        messages = super()._build_messages(prompt, history, system_prompt)
        if self._pending_page_context:
            self._inject_page_context(messages, self._pending_page_context)
        return messages

    @staticmethod
    def _inject_page_context(messages: list[dict], page_context: PageContext) -> None:
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

        page_content = (
            f"Title: {page_context.title}\nURL: {page_context.url}\n\n{page_context.text}"
        )

        # Assistant "called" fetch with the URL in queries
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID,
                        "type": "function",
                        "function": {
                            "name": BrowseTool.name,
                            "arguments": {
                                "queries": [page_context.url],
                            },
                        },
                    }
                ],
            }
        )
        # Tool "returned" the page content — framed like every real tool result via
        # the same tagged first-person envelope, but with bespoke narration naming this
        # for what it is: the page the user is currently viewing, not a search Penny ran.
        messages.append(
            {
                "role": "tool",
                "content": Agent._frame_injected_result(
                    BrowseTool.name,
                    Prompt.PAGE_CONTEXT_NARRATION,
                    page_content,
                ),
                "tool_call_id": ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID,
            }
        )

    # ── System prompt ──────────────────────────────────────────────────────

    async def _build_system_prompt(
        self,
        user: str | None,
        instructions: str | None = None,
    ) -> str:
        """Identity + (page hint) + instructions + self-state header (#1555).

        The chat prompt opens with Penny's persona/instructions and closes with a
        deterministic **self-state header** — the dynamic tail — rendering her own
        operational situation (active mechanisms · recent activity · the store map
        · durable user facts) purely from the registry
        (#1566) + ledger (#1560), never a relevance guess.

        The ambient inversion (#1555): the chat prompt no longer injects
        **speculative user-content recall** at all.  User content is fetched on
        demand via the tools the header's pointers name (anchored by the user's
        message).
        Background agents keep the base envelope (profile + inventory) and never
        see this header — it is the chat entry point's opening state.
        """
        return "\n\n".join(
            section
            for section in [
                self._identity_section(),
                self._context_block(self._page_hint_section()),
                self._instructions_section(instructions),
                SelfStateHeader(self.db, user).render(),
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
        self, user: str, quoted_text: str | None = None
    ) -> list[tuple[str, str]] | None:
        """Recent conversation messages for chat continuity.

        When a quote-reply is present, walks the parent chain for that
        specific thread. Falls back to the standard recent-message window.
        """
        if quoted_text:
            _, thread_history = self.db.messages.get_thread_context(quoted_text)
            if thread_history:
                return thread_history
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

    async def caption_image(self, image_b64: str, call_link: CallLink) -> str:
        """Caption an image using the vision model.

        The channel layer rejects image messages before they reach this
        method when ``LLM_VISION_MODEL`` is unset, so the client is
        guaranteed to exist by the time we get here. Guard explicitly so
        the type narrows without an assert.

        ``call_link`` (#2458) is REQUIRED — the caller's own turn link when
        called from inside a turn (``_process_images``, its only production
        caller); a direct caller with no turn in progress supplies its own
        (``CallLink.new_top()``). This caption is its own ledger-visible run —
        ``call_link.child()`` mints its own run id, linking back to whatever
        ``call_link`` names as parent.
        """
        if self._vision_model_client is None:
            raise RuntimeError(
                "caption_image called without a vision model client — "
                "channel-layer validation should have rejected this message"
            )
        messages = [
            LlmMessage(role="user", content=Prompt.VISION_AUTO_DESCRIBE_PROMPT, images=[image_b64]),
        ]
        response = await self._vision_model_client.chat(
            messages=messages,
            agent_name=self.name,
            prompt_type=ChatPromptType.VISION_CAPTION,
            call_link=call_link.child(),
        )
        return response.content.strip()

    # ── Image processing ──────────────────────────────────────────────────

    async def _process_images(
        self, content: str, images: list[str] | None, call_link: CallLink
    ) -> tuple[str, bool]:
        """Caption images with vision model and build combined text prompt."""
        if not images:
            return content, False

        captions = [await self.caption_image(img, call_link) for img in images]
        caption = ", ".join(captions)
        if content:
            content = Prompt.VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
        else:
            content = Prompt.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
        logger.info("Built vision prompt: %s", content[:200])
        return content, True
