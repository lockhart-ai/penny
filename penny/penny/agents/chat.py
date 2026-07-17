"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with web browsing and memory tools.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.agents.base import Agent, ProgressCallback
from penny.agents.models import ControllerResponse
from penny.agents.self_state import SelfStateHeader
from penny.channels.base import PageContext
from penny.constants import ChatPromptType, PennyConstants
from penny.datetime_utils import current_datetime_line
from penny.llm.models import LlmError
from penny.prompts import Prompt
from penny.skill_extraction import NoExtraction, SkillExtracted, SkillExtractor
from penny.tools import Tool
from penny.tools.browse import BrowseTool
from penny.tools.generate_image import GenerateImageTool
from penny.tools.memory_tools import TestExtractionPromptTool, collector_tool_surface
from penny.tools.notifications import NotificationsMuteTool, NotificationsUnmuteTool
from penny.tools.skill_tools import render_skill_full
from penny.validation.outcomes import LoopContext
from penny.validation.response_validators import CallAsTextValidator, SkillNarrationValidator

if TYPE_CHECKING:
    from penny.agents.collector import Collector
    from penny.llm.image_client import OllamaImageClient
    from penny.llm.models import LlmResponse

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
    # Chat's run-shape chain (base Agent's is empty).  A new chat shape guard is one
    # more entry here — never a branch in the loop.
    #  - SkillNarrationValidator: on a run that just auto-extracted a skill (#1658),
    #    nudge the model to tell the user what it learned FROM the rendered frame the
    #    text-branch prep stamped on the ctx (SAID==DID).  First, so narration wins.
    #  - CallAsTextValidator: a text response that is really a serialized tool call
    #    (gpt-oss's Harmony call-as-text bail) would be sent to the user as a raw
    #    JSON blob; catch it and nudge the model to re-emit the real call or reply in
    #    plain words.
    run_shape_validators = [SkillNarrationValidator(), CallAsTextValidator()]
    # Stable id linking the synthetic page-context tool-call to its tool-result
    # so the injection rides the standard OpenAI ``tool_call_id`` envelope, not
    # an ad-hoc ``tool_name`` field.
    PAGE_CONTEXT_TOOL_CALL_ID = "page-context"

    def __init__(
        self,
        image_client: OllamaImageClient | None = None,
        email_tools_builder: Callable[[str, str], list[Tool]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pending_page_context: PageContext | None = None
        # The current user message, held for the span of one turn so the
        # per-turn tool builders (email read summarisation) can see what the
        # user just asked.  Set in ``handle`` and cleared in its ``finally``.
        self._current_message: str | None = None
        # Chat replies via final text — tools are stripped on the final
        # agentic step to force the model to produce its reply.  Background
        # agents inherit the True default to keep tools available so they
        # can call ``done`` / ``send_message`` on the final step.
        self._keep_tools_on_final_step = False
        self._collector: Collector | None = None
        # Present only when an image model is configured — mirrors the retired
        # /draw command's conditionality; enables the generate_image tool.
        self._image_client = image_client
        # Present only when a mailbox is configured (Fastmail or Zoho) — mirrors
        # the retired /email + /zoho commands' conditionality.  Builds the email
        # tools fresh per turn (read_emails summarises against the current
        # message + date), so it takes ``(user_query, today)``.
        self._email_tools_builder = email_tools_builder
        # Automatic skill extraction at run end (#1658): the chat run's own ledger
        # is distilled into a skill when the run qualifies (read + write, healthy).
        # There is no ``skill_create`` tool — the framework does this deterministically.
        self._skill_extractor = SkillExtractor(
            self.db,
            self._embedding_model_client,
            self._model_client,
            agent_name=self.name,
            # The collector-runnable tool surface (#1668) — single-sourced, so a
            # captured step a collector could never run (a lifecycle call the demo
            # made) is dropped from the recipe instead of baked into an
            # uninstantiable skill.
            collector_tool_surface=collector_tool_surface(self.db, self._model_client),
        )
        # The run whose extraction was already attempted this turn — the structural
        # once-per-run guard, so the post-narration re-reply never re-extracts or
        # re-narrates (chat turns are sequential, so one field suffices; no leak).
        self._extraction_run_id: str | None = None

    def set_collector(self, collector: Collector) -> None:
        """Bind the Collector so test_extraction_prompt is available in chat."""
        self._collector = collector

    def get_tools(self, run_id: str | None = None) -> list[Tool]:
        tools = super().get_tools(run_id)
        # Notification mute/unmute is a chat-driven action over the MuteState row
        # (the retired /mute + /unmute commands), so both tools live on the chat
        # surface — the model dispatches to them from natural language.
        tools.append(NotificationsMuteTool(self.db))
        tools.append(NotificationsUnmuteTool(self.db))
        if self._collector is not None:
            tools.append(TestExtractionPromptTool(self._collector))
        if self._image_client is not None:
            tools.append(
                GenerateImageTool(self._image_client, self.db, self._embedding_model_client)
            )
        tools.extend(self._email_tools())
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

    # ── Automatic skill extraction + narration (#1658) ──────────────────

    async def _prepare_text_shape(
        self, response: LlmResponse, ctx: LoopContext, run_id: str
    ) -> LoopContext:
        """When the chat run emits final text, run automatic skill extraction over
        this run's completed ledger (Python-space, the run-end chokepoint) and, on a
        qualifying run, stamp the learned skill's rendered frame onto the ctx so the
        ``SkillNarrationValidator`` narrates it in the same turn.

        Extraction runs at most once per run (``_extraction_run_id``): the first
        final text extracts + narrates; the model's post-narration re-reply finds the
        run already attempted and falls through to the real final answer.  A
        non-qualifying run stamps nothing (the ctx passes through unchanged)."""
        if run_id == self._extraction_run_id:
            return ctx
        self._extraction_run_id = run_id
        frame = await self._extract_and_frame_skill(run_id)
        if frame is None:
            return ctx
        return ctx.model_copy(update={"learned_skill_frame": frame})

    async def _extract_and_frame_skill(self, run_id: str) -> str | None:
        """Extract a skill from this run and, on success, build the narration frame
        (the same ``render_skill_full`` render the read surface shows) so the model
        narrates from the render, not from memory.  ``None`` when the run did not
        qualify — the gate is logged, never silently swallowed."""
        result = await self._skill_extractor.extract(run_id)
        match result:
            case SkillExtracted(skill=skill, origin_message=origin):
                return Prompt.SKILL_LEARNED_NARRATION.format(
                    skill=render_skill_full(skill), demonstrated_on=origin
                )
            case NoExtraction(gate=gate):
                logger.debug("No skill extracted from run %s (%s)", run_id, gate)
                return None

    # ── Message handling ───────────────────────────────────────────────

    async def handle(
        self,
        content: str,
        sender: str,
        images: list[str] | None = None,
        page_context: PageContext | None = None,
        quoted_text: str | None = None,
        run_id: str | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, processes images, runs agentic loop.

        ``run_id`` is the turn's run id, minted by the channel so the same id
        both stamps every promptlog row and is recorded on any collection the
        turn creates (``created_by_run_id``, #1566) — the channel later links the
        spawning message to that run.  Passed as an explicit parameter down the
        call chain, never held as ambient state; when a direct caller omits it,
        one is minted here.
        """
        self._current_user = sender
        self._pending_page_context = page_context
        run_id = run_id or uuid.uuid4().hex
        try:
            content, has_images = await self._process_images(content, images)
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
                    system_prompt=system_prompt,
                    run_id=run_id,
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
                history=history,
                system_prompt=system_prompt,
                run_id=run_id,
                on_tool_start=on_tool_start,
                on_progress=on_progress,
                progress_scope="foreground",
                prompt_type=ChatPromptType.USER_MESSAGE,
            )
        finally:
            self._current_user = None
            self._pending_page_context = None
            self._current_message = None

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
        · taught skills · durable user facts) purely from the registry
        (#1566) + ledger (#1560), never a relevance guess.

        The ambient inversion (#1555): the chat prompt no longer injects
        **speculative user-content recall** at all.  User content is fetched on
        demand via the tools the header's pointers name (anchored by the user's
        message).  Taught-behavior firing — dark while #1555 removed ambient
        ``skills`` recall — is re-homed by #1471 onto the header's **Skills and
        rules** section: the taught-skill registry (``db.skills``, the sole
        skills store — the legacy ``skills`` collection retired with #1624)
        renders deterministically, so a taught behavior fires ambiently (0
        calls) with its full recipe one guess-free ``skill_read`` hop away.
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

    async def caption_image(self, image_b64: str) -> str:
        """Caption an image using the vision model.

        The channel layer rejects image messages before they reach this
        method when ``LLM_VISION_MODEL`` is unset, so the client is
        guaranteed to exist by the time we get here. Guard explicitly so
        the type narrows without an assert.
        """
        if self._vision_model_client is None:
            raise RuntimeError(
                "caption_image called without a vision model client — "
                "channel-layer validation should have rejected this message"
            )
        messages = [
            {"role": "user", "content": Prompt.VISION_AUTO_DESCRIBE_PROMPT, "images": [image_b64]},
        ]
        response = await self._vision_model_client.chat(
            messages=messages,
            agent_name=self.name,
            prompt_type=ChatPromptType.VISION_CAPTION,
            run_id=uuid.uuid4().hex,
        )
        return response.content.strip()

    # ── Image processing ──────────────────────────────────────────────────

    async def _process_images(self, content: str, images: list[str] | None) -> tuple[str, bool]:
        """Caption images with vision model and build combined text prompt."""
        if not images:
            return content, False

        captions = [await self.caption_image(img) for img in images]
        caption = ", ".join(captions)
        if content:
            content = Prompt.VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
        else:
            content = Prompt.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
        logger.info("Built vision prompt: %s", content[:200])
        return content, True
