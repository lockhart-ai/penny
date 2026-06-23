"""Base Agent class with agentic loop and context building."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse as _urlparse
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import PennyConstants, ValidationReason
from penny.database import Database
from penny.llm import LlmClient
from penny.llm.models import LlmError, LlmTimeoutError, LlmToolParseError
from penny.llm.refusal import is_refusal
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import CursorReadTool, DoneTool, build_memory_tools
from penny.tools.send_message import SendMessageTool

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel

logger = logging.getLogger(__name__)


# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)

# Matches <think>...</think> blocks emitted inline by some models (e.g. DeepSeek-R1, Qwen3)
_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Matches markdown links [text](url) and bare URLs for validation
_MARKDOWN_LINK_URL_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
_BARE_URL_PATTERN = re.compile(r"(?<!\()(https?://\S+)")


def _is_url_truncated(url: str) -> bool:
    """Return True if url appears truncated or malformed.

    Checks for missing host and trailing hyphen (the most common sign of a cut-off path).
    Strips trailing prose punctuation before validation so sentence-ending periods
    don't cause false positives.
    """
    cleaned = url.rstrip(".,;:!?\"')>}]")
    try:
        parsed = _urlparse.urlparse(cleaned)
    except Exception:
        return True
    if not parsed.netloc or "." not in parsed.netloc:
        return True
    return cleaned.endswith("-")


def _clean_malformed_urls(content: str) -> str:
    """Remove truncated or malformed URLs from model-generated content.

    For markdown links [text](bad_url), the link text is preserved.
    For bare malformed URLs, the URL token is removed entirely.
    Valid URLs are left unchanged.
    """

    def fix_md_link(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        if _is_url_truncated(url):
            logger.warning("Stripped malformed URL from markdown link: %.120s", url)
            return text
        return m.group(0)

    def fix_bare_url(m: re.Match) -> str:
        url = m.group(1)
        if _is_url_truncated(url):
            logger.warning("Stripped malformed bare URL: %.120s", url)
            return ""
        return m.group(0)

    content = _MARKDOWN_LINK_URL_PATTERN.sub(fix_md_link, content)
    content = _BARE_URL_PATTERN.sub(fix_bare_url, content)
    return content


def _has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


def _strip_think_tags(content: str) -> tuple[str, str | None]:
    """Strip <think>...</think> blocks from content.

    Returns (cleaned_content, extracted_thinking) where extracted_thinking
    contains the concatenated text from all stripped blocks.
    """
    thinking_parts: list[str] = []

    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    cleaned = _THINK_TAG_PATTERN.sub(_collect, content).strip()
    extracted = "\n\n".join(thinking_parts) if thinking_parts else None
    return cleaned, extracted


def _parse_text_form_done(content: str) -> dict | None:
    """Recover an intended ``done(...)`` call from text content.

    Models occasionally emit the done args as plain content instead of a
    structured tool call.  Two observed shapes:
      * ``done({"success": true, "summary": "..."})``  (wrapped form)
      * ``{"success": true, "summary": "..."}``        (raw args JSON)

    Returns the parsed args dict if the content matches either shape and
    contains at least ``success`` or ``summary``, else ``None``.  Used in
    ``BackgroundAgent._run_cycle`` to synthesise a real ``ToolCallRecord``
    so the cycle's intent isn't lost when the model flubs the tool call.
    """
    text = content.strip()
    if text.startswith("done(") and text.endswith(")"):
        text = text[5:-1].strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        args = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    if "success" not in args and "summary" not in args:
        return None
    return args


def _build_strong_nudge(messages: list[dict]) -> str:
    """Build a context-aware nudge that includes the original user question.

    Called when many preceding tool calls may have saturated the model's context.
    Uses forceful language to break the model out of search-fixation loops.
    Including the original question gives the model a clear target after heavy tool use.
    """
    user_messages = [
        m["content"]
        for m in messages
        if m.get("role") == MessageRole.USER and not m["content"].startswith("STOP")
    ]
    original_question = user_messages[-1]
    return Prompt.FINAL_STEP_NUDGE.format(original_question=original_question)


@dataclass
class CycleResult:
    """Outcome of a single ``_run_cycle`` invocation.

    Returned to the caller (``execute``) so subclass cleanup can read the
    cycle's response without fishing it off ``self``.  ``run_id`` is owned
    by the caller and not part of this struct — every promptlog row from
    the cycle already carries it, and the caller passes the same UUID
    back into ``set_run_outcome`` directly.
    """

    success: bool
    response: ControllerResponse


@dataclass
class _StepResult:
    """Result of processing all tool calls in one agentic loop step."""

    messages: list[dict]
    records: list[ToolCallRecord]
    source_urls: list[str]


class Agent:
    """
    AI agent with a specific persona and capabilities.

    Agents receive shared LlmClient instances — foreground (fast, user-facing)
    and background (smart, processing). Callers create and own the clients;
    agents just hold references.
    """

    _instances: list[Agent] = []

    name: str = "Agent"

    # Static system prompt for this agent.  Subclasses with a fixed prompt
    # set this class attribute (e.g. ``system_prompt = Prompt.NOTIFY_SYSTEM_PROMPT``).
    # Agents that build their prompt dynamically (e.g. ChatAgent's
    # _build_system_prompt) leave this empty and pass ``system_prompt=``
    # explicitly to ``run()``.
    system_prompt: str = ""

    # Tool name that signals a successful cycle exit.  ``done`` is the
    # default; ``send_message`` for agents that signal completion by
    # delivering a message (notify).
    terminator_tool: str = DoneTool.name

    def __init__(
        self,
        model_client: LlmClient,
        db: Database,
        config: Config,
        vision_model_client: LlmClient | None = None,
        embedding_model_client: LlmClient | None = None,
        allow_repeat_tools: bool = False,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
    ):
        """Configure the agent.

        Long-lived subagents (chat, notify, thinking, extractors) declare
        ``system_prompt`` as a class attribute and let ``get_tools()`` build
        their surface fresh per cycle, so they don't pass ``system_prompt``
        or ``tools`` here.  Ad-hoc command agents (email, zoho) pass both
        because their prompt and tool set vary per invocation.
        """
        self.config = config
        self.db = db
        self.allow_repeat_tools = allow_repeat_tools

        self._model_client = model_client
        self._vision_model_client = vision_model_client
        self._embedding_model_client = embedding_model_client

        self._browse_tool: BrowseTool | None = None
        self._browse_provider: Callable[[], Any] | None = None
        self._channel: MessageChannel | None = None
        self._current_user: str | None = None
        self._tool_result_text: list[str] = []

        if system_prompt is not None:
            self.system_prompt = system_prompt

        # Long-lived agents leave the registry empty here and let
        # ``_install_tools(self.get_tools(...))`` rebuild it before each
        # cycle.  Ad-hoc agents pass their fixed tool list at construction.
        self._tool_registry = ToolRegistry()
        if tools is not None:
            for tool in tools:
                self._tool_registry.register(tool)
        self._tool_executor = ToolExecutor(self._tool_registry, timeout=config.tool_timeout)
        # Background subagents exit via the terminator tool (`done` / `send_message`)
        # — they keep tools available on the final step so the model can call its
        # exit tool. Chat overrides to False because its terminator is final
        # text output, not a tool call.
        self._keep_tools_on_final_step = True
        self._on_tool_start_factory: (
            Callable[
                [],
                tuple[
                    Callable[[list[tuple[str, dict]]], Awaitable[None]],
                    Callable[[], Awaitable[None]],
                ],
            ]
            | None
        ) = None

        Agent._instances.append(self)

        logger.info(
            "Initialized agent %s: model=%s",
            self.name,
            self._model_client.model,
        )

    # ── Top-level execution ──────────────────────────────────────────────

    async def execute(self) -> bool:
        """Run a scheduled cycle.

        Penny is single-user — every agent gets the identical tool surface
        (memory + browse + send_message), and ``send_message`` resolves the
        primary recipient itself at execute time.  No per-agent user
        binding is plumbed through here.
        """
        run_id = uuid.uuid4().hex
        result = await self._run_cycle(run_id)
        return result.success

    async def _run_cycle(self, run_id: str) -> CycleResult:
        """Generic agentic shell: install tools, run the loop, commit cursor.

        Builds the system prompt via ``_build_system_prompt(user)`` so
        background agents get the full envelope (identity + profile +
        memory inventory + task instructions).  ``user`` is the primary
        Penny user so notify can address them by name and the profile
        section reads correctly.  Reads ``self.name`` (class attr — also
        the prompt type identifier in promptlog) and ``self.terminator_tool``
        (class attr) to drive the cycle.  Every cursored read in the surface
        (``CursorReadTool`` — ``log_read`` and ``read_published_latest``) has
        its pending cursor committed on success and discarded on failure.

        ``run_id`` is supplied by the caller — the same UUID stamps every
        promptlog row this cycle produces and is what subclass cleanup
        passes back to ``set_run_outcome``.  Returning the response
        alongside ``success`` keeps the call chain explicit; no
        per-cycle state lives on ``self``.
        """
        tools = self.get_tools()
        cursor_tools = [t for t in tools if isinstance(t, CursorReadTool)]
        self._install_tools(tools)

        primary_user = self.db.users.get_primary_sender()
        system_prompt = await self._build_system_prompt(primary_user)

        response = await self.run(
            prompt="",
            max_steps=self.get_max_steps(),
            system_prompt=system_prompt,
            run_id=run_id,
            prompt_type=self.name,
        )
        # Recover from a text-form ``done(...)`` — model occasionally
        # emits the args as plain content instead of a structured tool
        # call, especially toward the end of a long cycle.  Synthesising
        # the missing record means cleanup (audit log, promptlog tag,
        # success bool) sees the model's intent rather than reporting a
        # spurious ``"max steps exceeded"``.
        if (
            self.terminator_tool == DoneTool.name
            and response.answer
            and not any(r.tool == DoneTool.name for r in response.tool_calls)
        ):
            args = _parse_text_form_done(response.answer)
            if args is not None:
                logger.info("Recovered text-form %s() call for run %s", DoneTool.name, run_id)
                response.tool_calls.append(ToolCallRecord(tool=DoneTool.name, arguments=args))
        success = any(record.tool == self.terminator_tool for record in response.tool_calls)

        # Commit every cursored read's pending advance on a productive cycle,
        # discard on a failed one — uniform across log_read and the published
        # fan-in read, so a cursor only moves over input actually processed.
        committed = self._consumed_input(success, response)
        for cursor_tool in cursor_tools:
            if committed:
                cursor_tool.commit_pending()
            else:
                cursor_tool.discard_pending()

        return CycleResult(success=success, response=response)

    @staticmethod
    def _consumed_input(success: bool, response: ControllerResponse) -> bool:
        """Did this cycle consume the input it read (→ commit the read cursor)?

        Yes if it closed via the terminator (``success``), OR if it changed
        durable state — a cycle that wrote/sent but then hit max steps or trailed
        off without a ``done()`` still genuinely processed its input, so the
        cursor must advance.  Otherwise the next tick re-reads the same batch,
        re-attempts the already-landed work, and dedup-rejects it — a wasted
        cycle, and (if the batch always blows the step budget) a collection stuck
        forever.  Reads/refusals/no-ops carry ``mutated=False`` and don't count.
        """
        return success or any(record.mutated for record in response.tool_calls)

    # ── Override hooks ───────────────────────────────────────────────────

    def get_max_steps(self) -> int:
        """Cap on agentic loop iterations — reads the shared runtime config."""
        return int(self.config.runtime.MAX_STEPS)

    # ── Agentic loop entry ───────────────────────────────────────────────

    async def run(
        self,
        prompt: str,
        max_steps: int,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        run_id: str | None = None,
        prompt_type: str | None = None,
    ) -> ControllerResponse:
        """Run the agentic loop — prompt in, response out."""
        if run_id is None:
            run_id = uuid.uuid4().hex
        self._tool_result_text = []
        messages = self._build_messages(prompt, history, system_prompt)
        tools = self._tool_registry.get_ollama_tools()
        return await self._run_agentic_loop(
            messages, tools, max_steps, on_tool_start, run_id, prompt_type
        )

    # ── Agentic loop internals ───────────────────────────────────────────

    async def _run_agentic_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        steps: int,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
        run_id: str | None = None,
        prompt_type: str | None = None,
    ) -> ControllerResponse:
        """Execute the step loop: call model, process tool calls, or return final answer."""
        source_urls: list[str] = []
        called_tools: set[tuple[str, ...]] = set()
        tool_call_records: list[ToolCallRecord] = []

        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)
            # Force final step early when batched tool calls accumulate to the cap,
            # preventing context growth beyond what the 1-per-step case allows.
            is_final_step = step == steps - 1 or len(tool_call_records) >= steps - 1
            step_tools = self._tools_for_step(tools, is_final_step)

            response = await self._call_model_validated(messages, step_tools, run_id, prompt_type)
            if response is None:
                return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

            if response.has_tool_calls:
                if await self.handle_premature_terminator(
                    response, messages, tool_call_records, is_final_step
                ):
                    continue
                result = await self._process_tool_calls(response, called_tools, on_tool_start)
                self._absorb_tool_step_result(result, messages, tool_call_records, source_urls)
                await self.after_step(result.records, result.messages, messages)
                if self.should_stop_loop(result.records):
                    logger.info("Loop stop requested after step %d/%d", step + 1, steps)
                    return ControllerResponse(answer="", tool_calls=tool_call_records)
                abort = self._abort_if_all_tools_failed(tool_call_records)
                if abort is not None:
                    return abort
                continue

            if await self.handle_text_step(response, messages, step, is_final_step):
                continue

            return self._build_final_response(response, source_urls, tool_call_records)

        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer=PennyResponse.AGENT_MAX_STEPS, tool_calls=tool_call_records
        )

    def _tools_for_step(self, tools: list[dict], is_final_step: bool) -> list[dict]:
        """Strip tools on the final step unless the agent keeps them available."""
        if is_final_step and not self._keep_tools_on_final_step:
            logger.debug("Final step — tools removed, model must produce text")
            return []
        return tools

    def _absorb_tool_step_result(
        self,
        result: Any,
        messages: list[dict],
        tool_call_records: list[ToolCallRecord],
        source_urls: list[str],
    ) -> None:
        """Append a step's tool-call output into the running loop state."""
        messages.extend(result.messages)
        tool_call_records.extend(result.records)
        source_urls.extend(result.source_urls)

    def _abort_if_all_tools_failed(
        self, tool_call_records: list[ToolCallRecord]
    ) -> ControllerResponse | None:
        """Return an early-exit response if every tool call so far has failed."""
        if len(tool_call_records) < PennyConstants.TOOL_FAILURE_ABORT_THRESHOLD:
            return None
        if not all(r.failed for r in tool_call_records):
            return None
        failed_tools = sorted({r.tool for r in tool_call_records})
        logger.warning(
            "All %d tool call(s) failed — aborting: %s",
            len(tool_call_records),
            ", ".join(failed_tools),
        )
        return ControllerResponse(
            answer=PennyResponse.AGENT_TOOLS_UNAVAILABLE.format(tools=", ".join(failed_tools)),
            tool_calls=tool_call_records,
        )

    def on_response(self, response) -> None:
        """Hook called after every model response, before tool/text branching.

        Override to capture content from all responses (e.g. inner monologue).
        """

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Handle a text-only model response. Return True to continue, False to stop.

        Base returns False — text response = final answer.
        Override to inject continuation messages and keep the loop going.
        """
        return False

    async def handle_premature_terminator(
        self,
        response,
        messages: list[dict],
        tool_call_records: list[ToolCallRecord],
        is_final: bool,
    ) -> bool:
        """Reject a terminator tool call made before any real work. Return True to
        continue, False to process the call normally.

        Base returns False — no agent shape forbids an early terminator.
        ``BackgroundAgent`` overrides this to refuse a first-move ``done()``.
        """
        return False

    async def after_step(
        self,
        step_records: list[ToolCallRecord],
        step_messages: list[dict],
        conversation: list[dict] | None = None,
    ) -> None:
        """Capture tool result text for URL validation. Override in subclasses (call super)."""
        for message in step_messages:
            if message.get("role") == MessageRole.TOOL:
                content = message.get("content", "")
                if content:
                    self._tool_result_text.append(content)

    def should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """Check if the loop should stop early.

        Default: any *successful* call to the ``done`` tool is a graceful
        terminator.  A done call whose args failed validation (missing
        required ``success``/``summary`` fields) keeps the loop going so
        the model sees the validation error and can retry with the full
        triple — otherwise the cycle would exit with a recorded-but-
        empty done and produce a misleading audit row.
        """
        return any(record.tool == DoneTool.name and not record.failed for record in step_records)

    async def _call_model_validated(
        self,
        messages: list[dict],
        tools: list[dict],
        run_id: str | None = None,
        prompt_type: str | None = None,
    ):
        """Call the model, retrying on invalid outputs.

        Checks for (in order): XML markup, empty content, refusal, hallucinated URLs,
        tool parse errors (500 plain-text-instead-of-JSON).
        Each invalid output type gets one retry. Tool call responses are returned
        immediately without validation. When tools are stripped (None) but the model
        hallucinates tool calls, they are cleared and content falls through to
        normal validation — which triggers the appropriate nudge for empty responses.
        """
        max_retries = PennyConstants.RESPONSE_VALIDATION_RETRIES
        effective_tools = tools if tools else None
        retried: set[ValidationReason] = set()
        response = None

        for attempt in range(max_retries):
            try:
                response = await self._invoke_model(messages, effective_tools, run_id, prompt_type)
            except LlmToolParseError:
                if ValidationReason.TOOL_PARSE_ERROR not in retried:
                    retried.add(ValidationReason.TOOL_PARSE_ERROR)
                    logger.warning(
                        "Tool parse error on attempt %d/%d — retrying with format nudge",
                        attempt + 1,
                        max_retries,
                    )
                    messages.append({"role": MessageRole.USER, "content": Prompt.TOOL_FORMAT_NUDGE})
                    continue
                logger.error("Tool parse error on repeated attempt — aborting")
                return None
            if response is None:
                return None

            if response.has_tool_calls and effective_tools is not None:
                return response
            if response.has_tool_calls and effective_tools is None:
                logger.warning("Model hallucinated tool calls without tools — stripping")
                response.message.tool_calls = None

            self.on_response(response)
            reason = self._check_response(response.content.strip(), retried, messages)
            if reason is None:
                return response

            retried.add(reason)
            logger.warning(
                "Invalid response (%s) on attempt %d/%d", reason, attempt + 1, max_retries
            )
            self._append_retry_nudge(messages, response, reason, effective_tools)

        return response

    async def _invoke_model(
        self,
        messages: list[dict],
        effective_tools: list[dict] | None,
        run_id: str | None,
        prompt_type: str | None,
    ):
        """Call the LLM, returning ``None`` on connection/response errors.

        Re-raises ``LlmToolParseError`` so ``_call_model_validated`` can inject a
        format nudge and retry — the model needs a different message, not the same one.

        Timeouts are logged at WARNING — they're transient (the model may be slow
        or temporarily busy) and are already retried by the LLM client before
        this method is called.  Other LlmErrors (connection refused, server error,
        model not found) are logged at ERROR.
        """
        try:
            return await self._model_client.chat(
                messages=messages,
                tools=effective_tools,
                agent_name=self.name,
                prompt_type=prompt_type,
                run_id=run_id,
                # The bound collection (collectors) / None (chat, schedule) is
                # known from the first prompt — stamp it on every row so the run
                # is identifiable at write time, not retroactively at cycle end.
                run_target=self._memory_scope(),
            )
        except LlmToolParseError:
            raise
        except LlmTimeoutError as exception:
            logger.warning("LLM request timed out (model slow or temporarily busy): %s", exception)
            return None
        except LlmError as exception:
            logger.error("LLM chat failed: %s", exception)
            return None

    def _append_retry_nudge(
        self,
        messages: list[dict],
        response: Any,
        reason: ValidationReason,
        effective_tools: list[dict] | None,
    ) -> None:
        """Append the bad response and, for empty content, a nudge prompting synthesis."""
        messages.append(response.message.to_input_message())
        if reason != ValidationReason.EMPTY:
            return
        # Empty content: nudge depends on whether the model still has tools.
        # Final step (tools stripped) gets a strong synthesis demand; mid-loop
        # gets a gentle continue.
        nudge = _build_strong_nudge(messages) if effective_tools is None else Prompt.CONTINUE_NUDGE
        messages.append({"role": MessageRole.USER, "content": nudge})

    def _check_response(
        self,
        content: str,
        already_retried: set[ValidationReason],
        messages: list[dict] | None = None,
    ) -> ValidationReason | None:
        """Check a text response for problems. Returns reason or None if valid."""
        if _has_xml_tags(content) and ValidationReason.XML not in already_retried:
            return ValidationReason.XML

        effective_content, _ = _strip_think_tags(content)
        letter_count = sum(1 for c in effective_content if c.isalpha())
        if (
            letter_count < PennyConstants.MIN_RESPONSE_LETTERS
            and ValidationReason.EMPTY not in already_retried
        ):
            return ValidationReason.EMPTY

        if (
            effective_content
            and is_refusal(effective_content)
            and ValidationReason.REFUSAL not in already_retried
        ):
            return ValidationReason.REFUSAL

        source_text = self._get_source_text(messages)
        if source_text and effective_content:
            bad_urls = self._find_hallucinated_urls(effective_content, source_text)
            if bad_urls and ValidationReason.HALLUCINATED_URLS not in already_retried:
                logger.warning(
                    "Hallucinated URL(s): %s",
                    ", ".join(url[:80] for url in bad_urls),
                )
                return ValidationReason.HALLUCINATED_URLS

        return None

    def _build_final_response(
        self,
        response,
        source_urls: list[str],
        tool_call_records: list[ToolCallRecord],
    ) -> ControllerResponse:
        """Build the ControllerResponse from the model's final (non-tool) answer."""
        content = response.content.strip()

        if not content:
            logger.error(
                "Model returned empty content! model=%s, preceding_tool_calls=%d",
                self._model_client.model,
                len(tool_call_records),
            )
            fallback = (
                PennyResponse.FALLBACK_RESPONSE
                if tool_call_records
                else PennyResponse.AGENT_EMPTY_RESPONSE
            )
            return ControllerResponse(answer=fallback)

        thinking = response.thinking or response.message.thinking

        # Strip <think>...</think> blocks emitted inline by some models.
        # Move extracted content to the thinking field if not already populated.
        content, inline_thinking = _strip_think_tags(content)
        if not thinking and inline_thinking:
            thinking = inline_thinking

        if thinking:
            logger.info("Extracted thinking text (length: %d)", len(thinking))

        if not content:
            logger.error("Model returned empty content after stripping think tags!")
            fallback = (
                PennyResponse.FALLBACK_RESPONSE
                if tool_call_records
                else PennyResponse.AGENT_EMPTY_RESPONSE
            )
            return ControllerResponse(answer=fallback)

        content = _clean_malformed_urls(content)

        if source_urls and "http" not in content:
            content = f"{content}\n\n{source_urls[0]}"

        word_count = len(content.split())
        if word_count < 10:
            logger.warning("Short response detected (word_count=%d): %s", word_count, content[:100])
        logger.info("Got final answer (length: %d)", len(content))
        return ControllerResponse(
            answer=content,
            thinking=thinking,
            tool_calls=tool_call_records,
        )

    # ── Tool management ──────────────────────────────────────────────────

    def set_channel(self, channel: MessageChannel) -> None:
        """Bind a channel so this agent can send messages via SendMessageTool."""
        self._channel = channel

    def _memory_scope(self) -> str | None:
        """Bind this agent's entry-mutation tools to a single collection.

        Default: no scope — chat-style agents see the full chat surface
        (lifecycle + reads, no entry mutations).  ``Collector`` overrides
        to return its current target's name, so ``build_memory_tools``
        returns the collector surface (entry mutations pinned to that
        target + log_append + reads).
        """
        return None

    def get_tools(self) -> list[Tool]:
        """Tool surface — memory + browse, dispatched by ``_memory_scope``.

        ``BackgroundAgent.get_tools`` extends this with ``done`` and
        (optionally) ``send_message`` for agents that terminate via a
        terminator tool or deliver outbound to the user.

        Builds fresh each cycle so runtime config changes take effect
        immediately and the underlying ``BrowseTool``'s author + cursor
        identity match the agent's current ``name``.
        """
        scope = self._memory_scope()
        # Key the memory tools (read cursors + entry author) on the bound
        # collection, not the constant agent identity.  The Collector drives
        # every collection under one ``name`` ("collector"), so keying on
        # ``self.name`` collapsed all collections that read the same log onto
        # a single shared cursor — whichever ran first consumed the new
        # entries and starved the rest.  ``scope`` is the bound collection for
        # collectors and None for chat/schedule agents (which keep self.name).
        tools: list[Tool] = build_memory_tools(
            self.db,
            self._embedding_model_client,
            agent_name=scope or self.name,
            scope=scope,
        )
        tools.append(self._build_browse_tool(author=self.name))
        return tools

    def _build_browse_tool(self, author: str) -> BrowseTool:
        """Build a fresh BrowseTool from config, updating self._browse_tool."""
        max_calls = int(self.config.runtime.MAX_QUERIES)
        search_url = str(self.config.runtime.SEARCH_URL)
        tool = BrowseTool(
            max_calls=max_calls,
            search_url=search_url,
            db=self.db,
            embedding_client=self._embedding_model_client,
            author=author,
        )
        if self._browse_provider:
            tool.set_browse_provider(self._browse_provider)
        self._browse_tool = tool
        return tool

    def _install_tools(self, tools: list[Tool]) -> None:
        """Replace the agent's tool registry and executor."""
        self._tool_registry = ToolRegistry()
        for tool in tools:
            self._tool_registry.register(tool)
        self._tool_executor = ToolExecutor(self._tool_registry, timeout=self.config.tool_timeout)
        logger.debug(
            "Installed %d tool(s) for %s: %s",
            len(tools),
            self.name,
            ", ".join(t.name for t in tools),
        )

    async def _process_tool_calls(
        self,
        response,
        called_tools: set[tuple[str, ...]],
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
    ) -> _StepResult:
        """Process all tool calls from a model response, executing valid ones in parallel."""
        logger.info("Model requested %d tool call(s)", len(response.message.tool_calls or []))
        messages: list[dict] = [response.message.to_input_message()]
        records: list[ToolCallRecord] = []
        source_urls: list[str] = []

        pending = self._dedup_tool_calls(response, called_tools, messages)
        await self._notify_tool_start(on_tool_start, pending)

        results = await asyncio.gather(
            *[
                self._execute_single_tool(name, args, reasoning)
                for _, name, args, reasoning in pending
            ]
        )

        self._collect_tool_results(pending, results, messages, records, source_urls)

        return _StepResult(
            messages=messages,
            records=records,
            source_urls=source_urls,
        )

    def _dedup_tool_calls(
        self,
        response: Any,
        called_tools: set[tuple[str, ...]],
        messages: list[dict],
    ) -> list[tuple[str, str, dict, str | None]]:
        """Filter out repeat tool calls, returning the pending ones to execute.

        Repeats append a "you already called this" message in place so the
        model sees the rejection. Mutates ``called_tools`` and ``messages``.
        """
        pending: list[tuple[str, str, dict, str | None]] = []
        for tool_call in response.message.tool_calls or []:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            # Pop reasoning before dedup (same args + different reasoning = repeat)
            reasoning = arguments.pop("reasoning", None)
            call_key = self._make_call_key(tool_name, arguments)

            if not self.allow_repeat_tools and call_key in called_tools:
                logger.info("Skipping repeat: %s(%s)", tool_name, arguments)
                repeat_msg = "You already made this exact tool call. Try a different query or tool."
                messages.append(
                    {"role": MessageRole.TOOL, "content": repeat_msg, "tool_call_id": tool_call_id}
                )
                continue

            called_tools.add(call_key)
            pending.append((tool_call_id, tool_name, arguments, reasoning))
        return pending

    async def _notify_tool_start(
        self,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None,
        pending: list[tuple[str, str, dict, str | None]],
    ) -> None:
        """Fire on_tool_start once with the full pending batch so UI can show combined status."""
        if not on_tool_start or not pending:
            return
        try:
            await on_tool_start([(name, dict(args)) for _, name, args, _ in pending])
        except RuntimeError, ValueError:
            logger.debug("on_tool_start callback failed")

    def _collect_tool_results(
        self,
        pending: list[tuple[str, str, dict, str | None]],
        results: list[tuple[str, ToolCallRecord, list[str]]],
        messages: list[dict],
        records: list[ToolCallRecord],
        source_urls: list[str],
    ) -> None:
        """Append each tool result to messages and accumulate records/urls.

        Frames the model-facing content via ``Tool.format_result`` so every
        result is unmistakably the response to the model's own call.  Framing
        happens here, not in ``_execute_single_tool``, so ``record.failed``
        (computed on the raw string by ``startswith`` checks) is unaffected.
        """
        for (tool_call_id, tool_name, _, _), (result_str, record, urls) in zip(
            pending, results, strict=True
        ):
            records.append(record)
            source_urls.extend(urls)
            messages.append(
                {
                    "role": MessageRole.TOOL,
                    "content": Tool.format_result(tool_name, result_str),
                    "tool_call_id": tool_call_id,
                }
            )

    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict,
        reasoning: str | None,
    ) -> tuple[str, ToolCallRecord, list[str]]:
        """Execute one tool call. Returns (result_str, record, source_urls)."""
        logger.info("Executing tool: %s", tool_name)
        if reasoning:
            logger.debug("Tool reasoning: %s", reasoning[:200])

        record = ToolCallRecord(tool=tool_name, arguments=arguments, reasoning=reasoning)
        tool_call = ToolCall(tool=tool_name, arguments=arguments)

        # The executor always hands back a structured ToolResult — a tool's own
        # return, or a synthesised failed one for framework errors (not-found,
        # timeout, crash).  One branch, no string-prefix guessing; success and
        # mutated are authoritative.
        result = await self._tool_executor.execute(tool_call)
        record.failed = not result.success
        record.mutated = result.mutated
        record.result = result.message
        logger.debug(
            "Tool result (success=%s mutated=%s): %s",
            result.success,
            result.mutated,
            result.message[:200],
        )
        return result.message, record, result.source_urls

    @staticmethod
    def _make_call_key(tool_name: str, arguments: dict) -> tuple[str, ...]:
        """Build a hashable key from tool name + arguments for dedup."""
        arg_parts = tuple(f"{k}={v}" for k, v in sorted(arguments.items()))
        return (tool_name, *arg_parts)

    # ── URL validation ──────────────────────────────────────────────────

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """Extract all URLs from text (both markdown links and bare URLs)."""
        md_urls = [m.group(2) for m in _MARKDOWN_LINK_URL_PATTERN.finditer(text)]
        bare_urls = [m.group(1) for m in _BARE_URL_PATTERN.finditer(text)]
        seen: set[str] = set()
        urls: list[str] = []
        for url in md_urls + bare_urls:
            cleaned = url.rstrip(".,;:!?\"')>}]")
            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
        return urls

    @classmethod
    def _find_hallucinated_urls(cls, text: str, source_text: str) -> list[str]:
        """Return URLs in text that don't appear verbatim in the source text."""
        urls = cls._extract_urls(text)
        if not urls:
            return []
        return [url for url in urls if url not in source_text]

    def _get_source_text(self, messages: list[dict] | None = None) -> str:
        """Combined source text for URL validation.

        Includes the full message context (system prompt, conversation history,
        tool results) so URLs the model was legitimately shown — e.g. in the
        knowledge section of the system prompt or in a prior assistant turn —
        are not flagged as hallucinated.
        """
        parts = list(self._tool_result_text)
        if messages:
            for message in messages:
                content = message.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
        return "\n".join(parts)

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build message list for Ollama chat API.

        The system_prompt is the full prompt body (identity, context,
        instructions) built by each agent's _build_system_prompt method.
        This method only prepends the timestamp.
        """
        effective = system_prompt or self.system_prompt
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")
        system_content = f"Current date and time: {now}\n\n{effective}"

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_content).to_dict()]

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        messages.append(ChatMessage(role=MessageRole.USER, content=prompt).to_dict())
        return messages

    # ── System prompt building (template method pattern) ─────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """Build the full system prompt body — used by background agents.

        Envelope: identity + (profile + memory inventory) + task instructions.
        ChatAgent overrides this to add ambient recall and a browser page
        hint between profile and inventory.

        Both chat and background include identity + profile because both
        types of agent can dispatch messages to the user; both include
        the inventory so the model can discover memories without calling
        ``list_memories``.

        ``user`` is ``None`` on a fresh install where no primary sender
        is configured yet — the profile section is just omitted in that
        case.  The timestamp is prepended by ``_build_messages`` — don't
        include it here.
        """
        sections = [
            self._identity_section(),
            self._context_block(
                self._profile_section(user),
                self._memory_inventory_section(),
            ),
            self._instructions_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ── Building blocks ───────────────────────────────────────────────────

    def _identity_section(self) -> str:
        """## Identity — Penny's voice and personality."""
        return f"## Identity\n{Prompt.PENNY_IDENTITY}"

    def _instructions_section(self, override: str | None = None) -> str:
        """## Instructions — agent-specific prompt body."""
        prompt = override or self.system_prompt
        return f"## Instructions\n{prompt}"

    @staticmethod
    def _context_block(*sections: str | None) -> str | None:
        """Wrap non-None sections under a ## Context header."""
        parts = [s for s in sections if s]
        if not parts:
            return None
        joined = "\n\n".join(parts)
        return f"## Context\n{joined}"

    def _profile_section(self, sender: str | None) -> str | None:
        """### User Profile — user name.

        Returns ``None`` when no primary user is configured (fresh install)
        or when the sender has no recorded ``UserInfo`` row yet.
        """
        if sender is None:
            return None
        user_info = self.db.users.get_info(sender)
        if user_info is None:
            return None
        logger.debug("Built profile context for %s", sender)
        return f"### User Profile\nThe user's name is {user_info.name}."

    def _memory_inventory_section(self) -> str | None:
        """### Memory Inventory — every non-archived memory by name, type, description, count.

        Includes memories with ``recall=off`` so the model knows what
        tool calls are possible for on-demand reads.  Sorted
        alphabetically by name for stable prompt structure.  Each line
        ends with the entry count so the model has a sense of which
        collections / logs are worth pulling from.  Goes in every
        agent's system prompt — chat and background alike — so the model
        never needs to call ``list_memories``.
        """
        memories = sorted(
            (m for m in self.db.memories.list_all() if not m.archived),
            key=lambda m: m.name,
        )
        if not memories:
            return None
        counts = self.db.memories.entry_counts()
        lines = ["### Memory Inventory"]
        for memory in memories:
            count = counts.get(memory.name, 0)
            lines.append(f"- {memory.name} ({memory.type}, {count} entries) — {memory.description}")
        return "\n".join(lines)

    def _build_conversation(self, sender: str) -> list[tuple[str, str]]:
        """Build conversation history as strict user/assistant alternation.

        Fetches the last N messages (no time boundary). Consecutive same-role
        messages are merged with newlines to maintain valid turn structure.
        """
        conversation: list[tuple[str, str]] = []
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            messages = self.db.messages.get_messages_since(sender, since=datetime.min, limit=limit)
            for msg in messages:
                role = (
                    MessageRole.USER
                    if msg.direction == PennyConstants.MessageDirection.INCOMING
                    else MessageRole.ASSISTANT
                )
                if conversation and conversation[-1][0] == role:
                    prev_role, prev_content = conversation[-1]
                    conversation[-1] = (prev_role, f"{prev_content}\n{msg.content}")
                else:
                    conversation.append((role, msg.content))
            if conversation:
                logger.debug("Built conversation (%d turns)", len(conversation))
        except Exception:
            logger.warning("Conversation building failed, proceeding without")
        return conversation

    # ── Utilities ────────────────────────────────────────────────────────

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Remove this agent from the instance registry."""
        if self in Agent._instances:
            Agent._instances.remove(self)

    @classmethod
    async def close_all(cls) -> None:
        """Close all agent instances."""
        for agent in cls._instances[:]:
            await agent.close()


class BackgroundAgent(Agent):
    """Subagent shape — thinking, notify, extractors.

    Reads ``BACKGROUND_MAX_STEPS`` instead of the chat ``MAX_STEPS`` cap,
    since background agents navigate the unified tool surface end-to-end
    (read inputs → process → write outputs → done) and need more loop
    iterations than a single chat turn.

    Adds ``done`` and ``send_message`` to the chat-style tool surface
    so background flows have a way to terminate and deliver to the
    user.  Chat agents reply inline via final text and don't need
    either — having ``done`` available there causes the model to call
    it instead of producing a reply.
    """

    def get_max_steps(self) -> int:
        return int(self.config.runtime.BACKGROUND_MAX_STEPS)

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """A collector acts only through tool calls — a text-only response is a bail.

        The model is meant to drive the whole cycle through tools and exit via
        ``done()``; when it instead emits prose (a "Done. Summary: ..." narration,
        a mid-work observation, a tool call written as text) the loop would
        otherwise treat that text as a final answer and stop, leaving the cycle
        with no ``done`` record — marked ``failed``, cursor uncommitted, re-run
        next tick.  Since the slip is stochastic (the same context usually
        produces a clean tool call), append the stray text + a nudge and keep the
        loop going so the model recovers with a real tool call — ``done()`` if it
        was finished, otherwise the next work tool.  Bounded by ``max_steps``: on
        the final step there's no room to retry, so let the loop end and fall
        through to the post-loop ``_parse_text_form_done`` recovery.
        """
        if is_final:
            return False
        messages.append(response.message.to_input_message())
        messages.append({"role": MessageRole.USER, "content": Prompt.COLLECTOR_TOOL_CALL_NUDGE})
        return True

    async def handle_premature_terminator(
        self,
        response,
        messages: list[dict],
        tool_call_records: list[ToolCallRecord],
        is_final: bool,
    ) -> bool:
        """Refuse a first-move ``done()`` — a collector must do real work first.

        A cycle whose very first tool call is ``done()`` (with no prior read /
        write / browse) is the ``⚠ NO WORK DONE`` bail: the model declared the
        cycle finished without even checking its inputs.  The model made a
        *coherent* tool call, so the correction is an **error tool response**
        (not a text-step nudge): append the assistant turn + a failed tool-result
        for the done call explaining why it was refused, exactly as
        ``_dedup_tool_calls`` rejects a repeat in place.  A failed ``done`` doesn't
        stop the loop (see ``should_stop_loop``), so the model sees the error and
        retries with a real tool call.  Bounded by ``max_steps``: on the final
        step there's no room to retry, so let the done close the cycle.

        Premature only when (a) this step's calls are all ``done`` and (b) no
        non-``done`` tool call has run yet this cycle — the same "no real work"
        test the run-health classifier uses for ``bailed``.  A batched
        ``[log_read, done]`` or a ``done`` after any earlier read is left alone.
        """
        if is_final:
            return False
        calls = response.message.tool_calls or []
        if not calls or any(call.function.name != DoneTool.name for call in calls):
            return False
        if any(record.tool != DoneTool.name for record in tool_call_records):
            return False
        messages.append(response.message.to_input_message())
        for call in calls:
            messages.append(
                {
                    "role": MessageRole.TOOL,
                    "content": Tool.format_result(
                        DoneTool.name, Prompt.COLLECTOR_PREMATURE_DONE_REJECTION
                    ),
                    "tool_call_id": call.id,
                }
            )
        logger.info("Rejected premature done() (no prior work) for %s", self.name)
        return True

    def get_tools(self) -> list[Tool]:
        tools = super().get_tools()
        tools.append(DoneTool())
        # send_message only enters the surface when a channel is wired, since the
        # drain schedule needs one to deliver.  The tool itself only enqueues, so
        # it takes no channel — it's attributed to the bound collection
        # (``_memory_scope()``) so the queue records which collector queued it.
        if self._channel is not None:
            tools.append(
                SendMessageTool(
                    agent_name=self._memory_scope() or self.name,
                    db=self.db,
                )
            )
        return tools
