"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with web browsing and memory tools.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from similarity.embeddings import cosine_similarity, deserialize_embedding

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.channels.base import PageContext
from penny.constants import ChatPromptType, PennyConstants
from penny.database.memory import Inclusion, Memory, RecallMode
from penny.database.models import MemoryEntry
from penny.datetime_utils import format_log_timestamp
from penny.llm.models import LlmError
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import TestExtractionPromptTool

if TYPE_CHECKING:
    from penny.agents.collector import Collector

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Two context mechanisms, kept independent:

    - Memory stores → system prompt via the recall block: stage-1
      ``inclusion`` routing (always / relevant / never) then stage-2
      ``recall`` entry rendering (all / relevant / recent).
    - Chat turns → messages array as alternating user/assistant turns
      via ``_build_conversation`` and ``history=``.

    The system prompt is identity + (profile + recall + page hint)
    + instructions.  Vision messages bypass the tool surface and use
    the captioner; everything else runs the standard agentic loop.
    """

    name: str = "chat"
    system_prompt = Prompt.CONVERSATION_PROMPT

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending_page_context: PageContext | None = None
        # Chat replies via final text — tools are stripped on the final
        # agentic step to force the model to produce its reply.  Background
        # agents inherit the True default to keep tools available so they
        # can call ``done`` / ``send_message`` on the final step.
        self._keep_tools_on_final_step = False
        self._collector: Collector | None = None

    def set_collector(self, collector: Collector) -> None:
        """Bind the Collector so test_extraction_prompt is available in chat."""
        self._collector = collector

    def get_tools(self) -> list[Tool]:
        tools = super().get_tools()
        if self._collector is not None:
            tools.append(TestExtractionPromptTool(self._collector))
        return tools

    # ── Message handling ───────────────────────────────────────────────

    async def handle(
        self,
        content: str,
        sender: str,
        images: list[str] | None = None,
        page_context: PageContext | None = None,
        quoted_text: str | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, processes images, runs agentic loop.
        """
        self._current_user = sender
        self._pending_page_context = page_context
        try:
            content, has_images = await self._process_images(content, images)
            history = self.get_history(sender, quoted_text=quoted_text)

            if has_images:
                logger.info("Handling vision message from %s", sender)
                self._install_tools([])
                system_prompt = await self._build_system_prompt(
                    sender, instructions=Prompt.VISION_RESPONSE_PROMPT
                )
                injected_context = await self._build_injected_context(sender, content)
                return await self.run(
                    prompt=content,
                    history=history,
                    max_steps=PennyConstants.VISION_MAX_STEPS,
                    system_prompt=system_prompt,
                    injected_context=injected_context,
                    prompt_type=ChatPromptType.VISION_MESSAGE,
                )

            logger.info("Handling message from %s (conversation mode)", sender)
            self._install_tools(self.get_tools())
            system_prompt = await self._build_system_prompt(sender)
            injected_context = await self._build_injected_context(sender, content)
            return await self.run(
                prompt=content,
                max_steps=self.get_max_steps(),
                history=history,
                system_prompt=system_prompt,
                injected_context=injected_context,
                on_tool_start=on_tool_start,
                prompt_type=ChatPromptType.USER_MESSAGE,
            )
        finally:
            self._current_user = None
            self._pending_page_context = None

    # ── Message building ────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        injected_context: str = "",
    ) -> list[dict]:
        """Build messages, injecting page context as a synthetic tools result."""
        messages = super()._build_messages(prompt, history, system_prompt, injected_context)
        if self._pending_page_context:
            self._inject_page_context(messages, self._pending_page_context)
        return messages

    @staticmethod
    def _inject_page_context(messages: list[dict], page_context: PageContext) -> None:
        """Inject a synthetic search call + result for page context.

        Uses the BrowseTool format so the synthetic history matches the tool
        the model actually sees in its tool definitions.
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
        # Tool "returned" the page content
        messages.append(
            {
                "role": "tool",
                "content": page_content,
                "tool_name": BrowseTool.name,
            }
        )

    # ── System prompt ──────────────────────────────────────────────────────

    async def _build_system_prompt(
        self,
        user: str | None,
        instructions: str | None = None,
    ) -> str:
        """Static chat system prompt: identity + (profile + inventory) + instructions.

        Ambient recall and the browser page hint are *volatile* per-turn
        context — they ride in the Live-context turn via
        ``_build_injected_context``, not here — so this body stays byte-stable
        across turns and the local KV cache keeps it warm.
        """
        return "\n\n".join(
            s
            for s in [
                self._identity_section(),
                self._context_block(
                    self._profile_section(user),
                    self._memory_inventory_section(),
                ),
                self._instructions_section(instructions),
            ]
            if s
        )

    async def _build_injected_context(self, user: str | None, content: str | None) -> str:
        """Volatile chat context for the Live-context turn: recall + page hint.

        - **Ambient recall**: two-stage — stage-1 ``inclusion`` routing
          (always / relevant-by-description-anchor / never) decides which
          memories participate, then stage-2 ``recall`` (all / relevant /
          recent) renders their entries; relevant-mode ranks entries against
          the conversation window with hybrid cosine+lexical similarity.
        - **Browser page hint**: when the user is on a page with the
          extension active.

        Both change per turn, so they live here rather than in the static
        system prompt.
        """
        history_texts = [text for _, text in self._build_conversation(user)] if user else []
        recall = await self._recall_section(
            current_message=content,
            conversation_history=history_texts,
            limit=int(self.config.runtime.RECALL_LIMIT),
        )
        return "\n\n".join(s for s in [recall, self._page_hint_section()] if s)

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

    # ── Ambient recall ────────────────────────────────────────────────────

    async def _recall_section(
        self,
        current_message: str | None,
        conversation_history: list[str] | None = None,
        limit: int = 99,
    ) -> str | None:
        """Ambient recall content, assembled in two stages.

        Stage 1 (collection routing) — each active memory's ``inclusion`` flag
        decides whether it participates: ``always`` unconditionally,
        ``relevant`` only when the conversation embeds close to the memory's
        description anchor, ``never`` not at all (already excluded).

        Stage 2 (entry rendering) — for each included memory, its ``recall``
        mode picks which entries surface:

          recent   — newest-first slice (``memory.newest_entries``)
          relevant — hybrid cosine+lexical ranking over the conversation window
                     (``memory.read_similar_hybrid``; skipped without embedding)
          all      — full set in insertion order (``memory.read_all``)

        Each memory is a polymorphic ``Memory`` object from
        ``db.memories.active_memories()`` — the renderers call methods on it and
        log-only behaviour (temporal-neighbor expansion) is the object's own
        override, so this path never branches on the memory's shape.
        """
        anchors = await self._embed_conversation_anchors(current_message, conversation_history)
        anchor_contents = self._anchor_contents(current_message, conversation_history)
        query_text = " ".join(
            t for t in [*(conversation_history or []), current_message or ""] if t
        )
        sections: list[str] = []
        for memory in self._active_memories():
            if not self._passes_inclusion(memory, anchors):
                continue
            section = self._render_recall_memory(
                memory, anchors, query_text, limit, anchor_contents
            )
            if section:
                sections.append(section)
        return "\n\n".join(sections) if sections else None

    def _active_memories(self) -> list[Memory]:
        """Memory objects for every non-archived, routable memory (inclusion != 'never')."""
        return self.db.memories.active_memories()

    def _passes_inclusion(self, memory: Memory, anchors: list[list[float]] | None) -> bool:
        """Stage-1 gate: does this memory participate in recall for this turn.

        ``always`` always passes; ``relevant`` passes only when the best cosine
        between the conversation window and the memory's description anchor
        clears ``MEMORY_INCLUSION_THRESHOLD``.  Fails open (includes) when there
        is no embedding to compare — no anchors (no embedding model / cold
        message) or no description anchor yet (pre-backfill) — so a missing
        vector never silently drops a collection.
        """
        inclusion = Inclusion(memory.inclusion)
        if inclusion == Inclusion.ALWAYS:
            return True
        if inclusion == Inclusion.NEVER:
            return False
        if anchors is None or memory.description_embedding is None:
            return True
        description_anchor = deserialize_embedding(memory.description_embedding)
        threshold = float(self.config.runtime.MEMORY_INCLUSION_THRESHOLD)
        return max(cosine_similarity(a, description_anchor) for a in anchors) >= threshold

    async def embed_description(self, text: str) -> list[float] | None:
        """Embed a memory description into its stage-1 routing anchor.

        Exposed for the browser channel's memory create/edit path, which —
        unlike the chat tool surface — has no embedding client of its own.
        Returns None when no embedding model is configured or the call fails.
        """
        if self._embedding_model_client is None:
            return None
        try:
            vecs = await self._embedding_model_client.embed([text])
            return vecs[0]
        except LlmError:
            logger.warning("Failed to embed memory description")
            return None

    async def _embed_conversation_anchors(
        self, current_message: str | None, history: list[str] | None
    ) -> list[list[float]] | None:
        """Embed history + current_message as ordered anchors (oldest→newest).

        Returns ``None`` when no current message is available, when no
        embedding client is configured, or when the embed call fails.
        Empty history is fine — the result is just ``[current_embedding]``.
        """
        if not current_message or self._embedding_model_client is None:
            return None
        texts = [*(history or []), current_message]
        try:
            return await self._embedding_model_client.embed(texts)
        except LlmError:
            logger.warning("Skipping relevant recall — conversation embedding failed")
            return None

    @staticmethod
    def _anchor_contents(current_message: str | None, history: list[str] | None) -> set[str]:
        """Texts being used as anchors — filtered out of the corpus before scoring.

        Channel ingress writes both the current incoming message and prior
        user/assistant turns into log memories before recall runs, so any
        of those would otherwise self-match its own anchor at cosine ≈ 1.0
        and dominate scoring.  Anchors stay anchors, never retrievals.
        """
        contents: set[str] = set()
        if current_message:
            contents.add(current_message)
        if history:
            contents.update(t for t in history if t)
        return contents

    def _render_recall_memory(
        self,
        memory: Memory,
        anchors: list[list[float]] | None,
        query_text: str,
        limit: int,
        anchor_contents: set[str],
    ) -> str | None:
        """Dispatch to the correct renderer for a single memory's recall mode."""
        mode = RecallMode(memory.recall)
        if mode == RecallMode.RECENT:
            entries = memory.newest_entries(k=limit)
        elif mode == RecallMode.RELEVANT:
            entries = self._relevant_entries(memory, anchors, query_text, limit, anchor_contents)
        elif mode == RecallMode.ALL:
            entries = memory.read_all()[:limit]
        else:
            return None
        if not entries:
            return None
        return self._format_recall_section(memory, entries)

    def _relevant_entries(
        self,
        memory: Memory,
        anchors: list[list[float]] | None,
        query_text: str,
        limit: int,
        anchor_contents: set[str],
    ) -> list[MemoryEntry]:
        """Run stage-2 hybrid ranking, expanding logs with temporal neighbors.

        For log-shaped memories the hybrid hits are augmented with every entry
        within ±``MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES`` of any hit's
        timestamp, so a single keyword match pulls in the surrounding
        conversation rather than a single line stripped of context.

        ``anchor_contents`` (the texts being used as anchors) are filtered out
        of the corpus before scoring — channel ingress writes user/penny
        messages into log memories, so without exclusion any anchor that
        matches an existing entry would self-match at cosine ≈ 1.0 and dominate
        the hit list.  Collections aren't written to from channel ingress, so
        the filter is a no-op for them.
        """
        if not anchors:
            return []
        # Logs expand each hit with its temporal neighbors, so they take fewer
        # hits and cap each hit's window — a hard HIT_LIMIT × PER_HIT ceiling on
        # an otherwise unbounded fan-out.  Collections don't expand, so they keep
        # the full ``limit`` of hits.
        hit_limit = min(limit, PennyConstants.MEMORY_NEIGHBOR_HIT_LIMIT) if memory.is_log else limit
        hits = memory.read_similar_hybrid(
            anchors,
            query_text,
            k=hit_limit,
            exclude_contents=anchor_contents or None,
        )
        # Collections return their hits unchanged; logs expand each hit with its
        # surrounding conversation (the polymorphic no-op vs. override).
        return memory.expand_with_temporal_neighbors(
            hits,
            PennyConstants.MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES,
            PennyConstants.MEMORY_NEIGHBOR_PER_HIT,
        )

    @staticmethod
    def _format_recall_section(memory: Memory, entries: list[MemoryEntry]) -> str:
        """Render a single memory's header + entries as a context subsection.

        Each entry gets its own ``####`` sub-header carrying the entry's key
        (when keyed) and the ``created_at`` timestamp, followed by the
        verbatim content on the next line.  This isolates entries
        visually in the prompt — without per-entry headers, multi-line
        contents (especially long Penny replies) blob together as one
        unbroken paragraph.  The timestamp also lets the model reason
        about temporal context ("we talked about this last week" vs
        "earlier today") without needing an extra tool call.
        """
        lines = [f"### {memory.name}", memory.description]
        for entry in entries:
            timestamp = format_log_timestamp(entry.created_at)
            header = f"#### [{entry.key}] · {timestamp}" if entry.key else f"#### {timestamp}"
            lines.append("")
            lines.append(header)
            lines.append(entry.content)
        return "\n".join(lines)

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
            content = PennyResponse.VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
        else:
            content = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
        logger.info("Built vision prompt: %s", content[:200])
        return content, True
