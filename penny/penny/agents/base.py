"""Base Agent class with agentic loop and context building."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, assert_never

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.datetime_utils import current_datetime_line
from penny.llm import LlmClient
from penny.llm.models import LlmError, LlmResponse, LlmTimeoutError, LlmToolParseError
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.text_validity import (
    has_leaked_harmony_envelope,
    is_degenerate_run,
    is_degenerate_tool_name,
)
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry, ToolResult
from penny.tools.base import RESULT_TAG
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import CursorReadTool, DoneTool, build_memory_tools
from penny.tools.send_message import SendMessageTool
from penny.validation import (
    ConditionKey,
    LoopContext,
    NudgeContinue,
    Proceed,
    RejectToolCall,
    Repair,
    ResponseValidator,
    Retry,
    Stop,
    run_validators,
)
from penny.validation.response_validators import (
    DoneJsonBailValidator,
    EmptyResponseValidator,
    HallucinatedToolCallRepair,
    HallucinatedUrlValidator,
    PrematureDoneValidator,
    RefusalValidator,
    TextInsteadOfToolValidator,
    XmlTagValidator,
    build_strong_nudge,
    clean_malformed_urls,
    strip_think_tags,
)

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel

logger = logging.getLogger(__name__)


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

    # Recovery move bound into a browse channel-outage error (no browser
    # connected).  Chat answers from memory or tells the user; ``BackgroundAgent``
    # overrides this to bind ``done(success=false, ...)`` — the terminator it
    # actually has (chat has none).
    channel_outage_recovery: str = Prompt.BROWSE_OUTAGE_RECOVERY_CHAT

    # The composable response-validation chain — one validator per live
    # condition, run in order by ``run_validators`` each model call.  Reads
    # like a table of contents: a future guard is one more entry here, not a
    # new branch in the loop.  ``ChatAgent`` and ``BackgroundAgent`` compose
    # their own chains; the base agent (ad-hoc command agents) inherits this
    # response-shape set.  ``HallucinatedToolCallRepair`` runs first so a
    # tools-stripped final-step hallucination is cleaned before the
    # content-shape validators see it.
    response_validators: list[ResponseValidator] = [
        HallucinatedToolCallRepair(),
        XmlTagValidator(),
        EmptyResponseValidator(),
        RefusalValidator(),
        HallucinatedUrlValidator(),
    ]

    # The run-shape chain — guards that depend on the run's tool-call history
    # (premature ``done()``, prose-instead-of-tool), applied at the loop's
    # tool-call / text branch points.  Empty on the base agent (no shape forbids
    # an early terminator or a text answer); ``BackgroundAgent`` adds the
    # collector guards.  A future shape guard is one more entry here.
    run_shape_validators: list[ResponseValidator] = []

    def __init__(
        self,
        model_client: LlmClient,
        db: Database,
        config: Config,
        vision_model_client: LlmClient | None = None,
        allow_repeat_tools: bool = False,
        *,
        embedding_model_client: LlmClient,
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
        promptlog row this cycle produces, threads to the tool surface as the
        writing run (so this cycle's entry writes cite it, #1560), and is what
        subclass cleanup passes back to ``set_run_outcome``.  Returning the
        response alongside ``success`` keeps the call chain explicit; no
        per-cycle state lives on ``self``.
        """
        tools = self.get_tools(run_id=run_id)
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
        # A cycle ends successfully only on a real ``done()`` tool call.  A
        # model that signals completion as prose instead of calling the tool is
        # not accommodated (no text-form parsing) — the cycle is not successful,
        # its cursor doesn't commit, and it re-runs next tick; the model is
        # guided toward a structured ``done()`` by the in-loop tool-call nudge.
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

            ctx = self._loop_context(step, is_final_step, step_tools, messages, tool_call_records)
            if response.has_tool_calls:
                if self._reject_premature_terminator(response, messages, ctx):
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

            if self._nudge_text_step(response, messages, ctx):
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

    def _loop_context(
        self,
        step: int,
        is_final_step: bool,
        step_tools: list[dict],
        messages: list[dict],
        records: list[ToolCallRecord],
    ) -> LoopContext:
        """Snapshot the run state a validator reads — built fresh per branch so
        ``records`` reflects the work done before this response.  ``retried`` is
        managed inside ``_call_model_validated`` (per model call), so the
        run-shape disposition context carries the empty default."""
        return LoopContext(
            step=step,
            is_final_step=is_final_step,
            tools_available=bool(step_tools),
            source_text=self._get_source_text(messages),
            records=list(records),
        )

    def _reject_premature_terminator(
        self, response: LlmResponse, messages: list[dict], ctx: LoopContext
    ) -> bool:
        """Run the run-shape chain over a tool-call response; apply a
        ``RejectToolCall`` (premature first-move ``done()``) in place.

        Returns True when the loop should ``continue`` (the call was refused with
        an error tool-result), False to process the tool calls normally."""
        match run_validators(self.run_shape_validators, response, ctx):
            case RejectToolCall(message=message):
                self._append_rejected_tool_calls(response, messages, message)
                logger.info("Rejected premature done() (no prior work) for %s", self.name)
                return True
            case Proceed():
                return False
            case Retry() | Repair() | NudgeContinue() | Stop():
                raise AssertionError(
                    "run-shape validators produced an unexpected disposition on a "
                    "tool-call response"
                )
            case unreachable:
                assert_never(unreachable)

    def _nudge_text_step(
        self, response: LlmResponse, messages: list[dict], ctx: LoopContext
    ) -> bool:
        """Run the run-shape chain over a text-only response; apply a
        ``NudgeContinue`` (collector narrated prose where a tool call was due).

        Returns True when the loop should ``continue`` (response + nudge appended),
        False to treat the text as the final answer."""
        match run_validators(self.run_shape_validators, response, ctx):
            case NudgeContinue(message=message):
                messages.append(response.message.to_input_message())
                messages.append({"role": MessageRole.USER, "content": message})
                return True
            case Proceed():
                return False
            case Retry() | Repair() | RejectToolCall() | Stop():
                raise AssertionError(
                    "run-shape validators produced an unexpected disposition on a text response"
                )
            case unreachable:
                assert_never(unreachable)

    @staticmethod
    def _frame_injected_result(tool_name: str, narration: str, body: str) -> str:
        """Frame a tool-SHAPED injection (page context, a deduped/rejected call) in the
        same tagged first-person envelope real tool results get from
        ``Tool.format_result`` — a bespoke call-site narration + the retained
        ``(<tool> result)`` machine tag + the body — so the whole tool-result surface
        reads as one voice.

        These sites aren't real registered tool calls, so the narration is supplied
        here (epic #1478 / #1485) rather than dispatched through ``to_result_narration``.
        The tag is single-sourced from ``RESULT_TAG`` so the machine-tag invariant that
        keeps gpt-oss parsing a prose body as a tool result (not a fresh instruction)
        holds uniformly across the real and the injected sites.
        """
        return f"{narration} {RESULT_TAG.format(tool_name=tool_name)}\n{body}"

    @staticmethod
    def _append_rejected_tool_calls(
        response: LlmResponse, messages: list[dict], message: str
    ) -> None:
        """Append the assistant turn + a failed tool-result for each call, exactly
        as ``_dedup_tool_calls`` rejects a repeat in place."""
        messages.append(response.message.to_input_message())
        for call in response.message.tool_calls or []:
            messages.append(
                {
                    "role": MessageRole.TOOL,
                    "content": Agent._frame_injected_result(
                        call.function.name,
                        Prompt.REJECTED_CALL_NARRATION.format(tool_name=call.function.name),
                        message,
                    ),
                    "tool_call_id": call.id,
                }
            )

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
        """Call the model, driving the response-validation chain on each output.

        Builds a ``LoopContext`` and runs ``self.response_validators`` via
        ``run_validators`` (XML / empty / refusal / hallucinated-URL, with the
        no-tools tool-call strip as a ``Repair``).  A ``Retry`` appends the bad
        response + its nudge and re-calls — once per condition (``retried``).  A
        ``Proceed`` returns the (possibly-repaired) response.  Tool-call responses
        with tools available short-circuit unvalidated.  ``LlmToolParseError``
        (no response to inspect) routes through the same retried-set bookkeeping
        as a ``TOOL_PARSE_ERROR`` retry.
        """
        max_retries = PennyConstants.RESPONSE_VALIDATION_RETRIES
        effective_tools = tools if tools else None
        retried: set[ConditionKey] = set()
        response = None

        for attempt in range(max_retries):
            try:
                response = await self._invoke_nondegenerate(
                    messages, effective_tools, run_id, prompt_type
                )
            except LlmToolParseError:
                if self._retry_tool_parse_error(messages, retried, attempt, max_retries):
                    continue
                return None
            if response is None:
                return None

            if response.has_tool_calls and effective_tools is not None:
                return response

            self.on_response(response)
            ctx = LoopContext(
                step=0,
                is_final_step=False,
                tools_available=effective_tools is not None,
                source_text=self._get_source_text(messages),
                retried=retried,
            )
            match run_validators(self.response_validators, response, ctx):
                case Proceed(response=validated):
                    return validated if validated is not None else response
                case Retry(condition=condition, nudge=nudge):
                    # Append the post-repair response (tool calls stripped when
                    # tools were unavailable) — the chain may have stripped a
                    # hallucinated call before a content validator asked to retry.
                    appended = self._repaired_for_append(response, effective_tools)
                    self._apply_retry(
                        messages, appended, condition, nudge, retried, attempt, max_retries
                    )
                case Repair() | RejectToolCall() | NudgeContinue() | Stop():
                    raise AssertionError("response validators produced an unexpected disposition")
                case unreachable:
                    assert_never(unreachable)

        # Retries exhausted — return the (tool-stripped, if no tools) last response
        # so the loop's text/tool branching matches the validated form.
        return self._repaired_for_append(response, effective_tools) if response else response

    @staticmethod
    def _repaired_for_append(
        response: LlmResponse, effective_tools: list[dict] | None
    ) -> LlmResponse:
        """The response form to re-append on a retry — tool calls stripped when no
        tools were available, mirroring ``HallucinatedToolCallRepair``."""
        if effective_tools is not None or not response.has_tool_calls:
            return response
        repaired = response.model_copy(deep=True)
        repaired.message.tool_calls = None
        return repaired

    def _retry_tool_parse_error(
        self,
        messages: list[dict],
        retried: set[ConditionKey],
        attempt: int,
        max_retries: int,
    ) -> bool:
        """Inject a format nudge and signal a retry for a tool-parse 500, once.

        Returns True to retry (nudge appended), False to abort — the error has no
        response to inspect, so it's keyed into the same retried set the
        response-validator chain uses (one retry per condition)."""
        if ConditionKey.TOOL_PARSE_ERROR in retried:
            logger.error("Tool parse error on repeated attempt — aborting")
            return False
        retried.add(ConditionKey.TOOL_PARSE_ERROR)
        logger.warning(
            "Tool parse error on attempt %d/%d — retrying with format nudge",
            attempt + 1,
            max_retries,
        )
        messages.append({"role": MessageRole.USER, "content": Prompt.TOOL_FORMAT_NUDGE})
        return True

    def _apply_retry(
        self,
        messages: list[dict],
        response: LlmResponse,
        condition: ConditionKey,
        nudge: str,
        retried: set[ConditionKey],
        attempt: int,
        max_retries: int,
    ) -> None:
        """Apply a ``Retry`` disposition: record the condition, append the bad
        response, then its nudge (if any).

        ``EMPTY`` carries the empty validator's per-agent mid-loop nudge
        (``CONTINUE_NUDGE`` for chat, ``COLLECTOR_CONTINUE_NUDGE`` for collectors)
        and an empty nudge on the final step (tools stripped); the loop substitutes
        the forceful ``build_strong_nudge`` there, since the strong builder needs
        the message history a pure validator can't hold.  Other conditions just
        re-append the response (empty nudge)."""
        retried.add(condition)
        logger.warning(
            "Invalid response (%s) on attempt %d/%d", condition, attempt + 1, max_retries
        )
        messages.append(response.message.to_input_message())
        if condition == ConditionKey.EMPTY and not nudge:
            nudge = build_strong_nudge(messages)
        if nudge:
            messages.append({"role": MessageRole.USER, "content": nudge})

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
                # The bound collection (collectors) / None (chat) is
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

    async def _invoke_nondegenerate(
        self,
        messages: list[dict],
        effective_tools: list[dict] | None,
        run_id: str | None,
        prompt_type: str | None,
    ) -> LlmResponse | None:
        """Call the model, discarding UNUSABLE raw output and re-rolling on the
        *unchanged* context.

        Two kinds of unusable output share this one discard-and-reroll machinery,
        because both are transport artifacts a fresh draw usually clears:
          * a gpt-oss punctuation collapse (``…?.``) — most often inside a tool-call
            argument, which the validation chain never sees (tool-call responses
            short-circuit it); and
          * a leaked Harmony tool-call envelope in the text content — a backend that
            failed to parse the envelope, so the whole call arrives as literal prose
            that would otherwise be delivered to the user verbatim.
        So the check runs here, on the raw output of every call, before the loop
        parses or acts on it.  The bad response is DROPPED, never appended:
        re-appending a collapse feeds it back into the conversation (a poisoned step
        makes the next ~4× more likely to collapse too), and re-appending a leaked
        envelope would ship raw tokens.  After ``DEGENERATE_REROLL_ATTEMPTS`` it
        returns ``None`` so the caller throws out the whole run rather than act on,
        or store, the poison.  The warning names WHICH condition tripped (the reroll
        mechanism is shared; the log label must stay honest).  ``LlmToolParseError``
        propagates unchanged — the format-nudge retry in ``_call_model_validated``
        still owns that path.
        """
        attempts = PennyConstants.DEGENERATE_REROLL_ATTEMPTS
        for attempt in range(attempts):
            response = await self._invoke_model(messages, effective_tools, run_id, prompt_type)
            if response is None:
                return response
            condition = self._unusable_output_condition(response)
            if condition is None:
                return response
            logger.warning(
                "Unusable model output (%s) — discarding and re-rolling %d/%d",
                condition,
                attempt + 1,
                attempts,
            )
        logger.error("Model output still unusable after %d re-rolls — aborting run", attempts)
        return None

    def _unusable_output_condition(self, response: LlmResponse) -> ConditionKey | None:
        """The condition that makes this raw output unusable, or ``None`` if it's
        clean.  ``TOOL_CALL_LEAK`` — a leaked Harmony tool-call envelope in the text
        content (or a serialised argument) — or ``DEGENERATE_OUTPUT`` — a punctuation
        collapse in the content, any tool-call argument, or an unregistered
        collapse-shaped NAME field (``Functions?????``, which would otherwise flow to
        a tool-not-found error that keeps the poison in context).  Serialising the
        tool-call arguments is what lets the guard catch the common collapse case,
        where it lands in a ``collection_write`` / ``done`` argument rather than in
        visible prose."""
        parts = [response.message.content or ""]
        for call in response.message.tool_calls or []:
            if self._is_degenerate_tool_call_name(call.function.name):
                return ConditionKey.DEGENERATE_OUTPUT
            parts.append(json.dumps(call.function.arguments, ensure_ascii=False))
        if any(has_leaked_harmony_envelope(part) for part in parts):
            return ConditionKey.TOOL_CALL_LEAK
        if any(is_degenerate_run(part) for part in parts):
            return ConditionKey.DEGENERATE_OUTPUT
        return None

    def _is_degenerate_tool_call_name(self, name: str) -> bool:
        """A tool-call name that is UNREGISTERED and collapse-shaped is poison.

        Ordering mirrors the dispatch layering: a registered name is a real call
        (dispatch as normal — never rerolled); an unregistered one that carries
        collapse characters (``funcs.done?``, ``read_simpar?``) is the same
        degeneration as content poison, so the response is discarded and
        re-rolled.  An unregistered but plausible identifier (a near-miss like
        ``collection_metadata``, or a Harmony-token-wrapped valid name — the
        future Harmony strip's repair case, not ours) falls through to the
        executor's tool-not-found error with its "Did you mean X?" hint."""
        return self._tool_registry.get(name) is None and is_degenerate_tool_name(name)

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
        content, inline_thinking = strip_think_tags(content)
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

        content = clean_malformed_urls(content)

        if source_urls and "http" not in content:
            content = f"{content}\n\n{source_urls[0]}"

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

    def get_tools(self, run_id: str | None = None) -> list[Tool]:
        """Tool surface — memory + browse, dispatched by ``_memory_scope``.

        ``BackgroundAgent.get_tools`` extends this with ``done`` and
        (optionally) ``send_message`` for agents that terminate via a
        terminator tool or deliver outbound to the user.

        Builds fresh each cycle so runtime config changes take effect
        immediately and the underlying ``BrowseTool``'s author + cursor
        identity match the agent's current ``name``.

        ``run_id`` is the id of the run building this surface — the chat turn's or
        the collector cycle's — threaded to every write/mutation-capable tool as
        the executing run (#1560): entry writes stamp it, ``collection_create``
        records it as the new mechanism's creating run (#1566), and each registry
        mutation records it as the change's cause.  Passed as an explicit
        parameter, never ambient state.
        """
        scope = self._memory_scope()
        # Key the memory tools (read cursors + entry author) on the bound
        # collection, not the constant agent identity.  The Collector drives
        # every collection under one ``name`` ("collector"), so keying on
        # ``self.name`` collapsed all collections that read the same log onto
        # a single shared cursor — whichever ran first consumed the new
        # entries and starved the rest.  ``scope`` is the bound collection for
        # collectors and None for chat agents (which keep self.name).
        tools: list[Tool] = build_memory_tools(
            self.db,
            self._embedding_model_client,
            agent_name=scope or self.name,
            scope=scope,
            run_id=run_id,
            include_lifecycle=self._include_lifecycle_tools(),
        )
        tools.append(self._build_browse_tool(author=self.name))
        return tools

    def _include_lifecycle_tools(self) -> bool:
        """Whether this agent may reshape the registry — create / update / merge /
        archive / unarchive collections and create logs.

        Chat-style agents do: the user evolves collections through them.  A
        cadence-fired collector run does NOT (``Collector`` overrides to False,
        #1556) — a background poll has no business re-architecting the system, so
        those tools are structurally ABSENT from its surface, not merely
        discouraged in its prompt.  Declared here as a template method (the run
        type decides), never as an if/else on run type inside the loop.
        """
        return True

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
            channel_outage_recovery=self.channel_outage_recovery,
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
                repeat_message = Prompt.DUPLICATE_CALL_REJECTION
                messages.append(
                    {
                        "role": MessageRole.TOOL,
                        "content": self._frame_injected_result(
                            tool_name,
                            Prompt.DUPLICATE_CALL_NARRATION.format(tool_name=tool_name),
                            repeat_message,
                        ),
                        "tool_call_id": tool_call_id,
                    }
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
        results: list[tuple[ToolResult, ToolCallRecord, list[str]]],
        messages: list[dict],
        records: list[ToolCallRecord],
        source_urls: list[str],
    ) -> None:
        """Append each tool result to messages and accumulate records/urls.

        Frames the model-facing content via ``Tool.format_result`` — a tagged,
        first-person narration of the call plus its body — so every result reads
        as the reply to the model's own call.  The call ``arguments`` and the whole
        ``ToolResult`` (not just its string body) are threaded through so the
        narration can name the action and branch on success/failure.
        """
        for (tool_call_id, tool_name, arguments, _), (result, record, urls) in zip(
            pending, results, strict=True
        ):
            records.append(record)
            source_urls.extend(urls)
            messages.append(
                {
                    "role": MessageRole.TOOL,
                    "content": Tool.format_result(tool_name, arguments, result),
                    "tool_call_id": tool_call_id,
                }
            )

    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict,
        reasoning: str | None,
    ) -> tuple[ToolResult, ToolCallRecord, list[str]]:
        """Execute one tool call. Returns (result, record, source_urls).

        Returns the whole ``ToolResult`` (not just its message string) so the
        framing site (``_collect_tool_results``) can narrate the outcome and branch
        on ``result.success``.  ``record.failed`` is still computed here on the raw
        result, before any framing, so failure detection is unaffected.
        """
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
        # A media row this call created (generate_image) rides to egress on the
        # record → ControllerResponse.generated_media_ids → deterministic attach.
        record.media_id = result.media_id
        logger.debug(
            "Tool result (success=%s mutated=%s): %s",
            result.success,
            result.mutated,
            result.message[:200],
        )
        return result, record, result.source_urls

    @staticmethod
    def _make_call_key(tool_name: str, arguments: dict) -> tuple[str, ...]:
        """Build a hashable key from tool name + arguments for dedup."""
        arg_parts = tuple(f"{k}={v}" for k, v in sorted(arguments.items()))
        return (tool_name, *arg_parts)

    # ── URL validation ──────────────────────────────────────────────────

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
        system_content = f"{current_datetime_line(self.db)}\n\n{effective}"

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
        ChatAgent overrides this entirely: it drops the speculative user-content
        recall (the ambient inversion, #1555) and closes the prompt with a
        deterministic self-state header (mechanisms · activity · store map ·
        durable user facts) in the dynamic tail instead.  Background agents keep
        this base envelope — they read memory explicitly per task and never open
        on a chat entry point's self-state.

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

    # The collector response-shape chain — same guards as the base/chat chain, but
    # the empty-response validator carries ``COLLECTOR_CONTINUE_NUDGE`` instead of
    # the chat ``CONTINUE_NUDGE``: a collector acts only through tool calls, so
    # "provide your response" invites an unparseable prose reply — its mid-loop
    # empty-content nudge must demand a tool call.  Composed explicitly (not
    # inherited) so the swap reads here as a table of contents, one differing entry.
    response_validators: list[ResponseValidator] = [
        HallucinatedToolCallRepair(),
        XmlTagValidator(),
        EmptyResponseValidator(continue_nudge=Prompt.COLLECTOR_CONTINUE_NUDGE),
        RefusalValidator(),
        HallucinatedUrlValidator(),
    ]

    # A collector acts only through tool calls, so three run-shape guards apply
    # that don't on chat: ``done()``'s arguments emitted as bare JSON text
    # (``DoneJsonBailValidator`` → shape-specific teaching ``NudgeContinue``,
    # ordered BEFORE the generic guard so the specific teaching outranks it), a
    # prose answer where a tool call was due (``TextInsteadOfToolValidator`` →
    # ``NudgeContinue``), and a first-move ``done()`` before any real work
    # (``PrematureDoneValidator`` → ``RejectToolCall``).  Applied at the loop's
    # text / tool-call branch points; all honour ``max_steps`` (no retry room on
    # the final step).
    run_shape_validators: list[ResponseValidator] = [
        PrematureDoneValidator(),
        DoneJsonBailValidator(),
        TextInsteadOfToolValidator(),
    ]

    # A collector closes with ``done()``, so its channel-outage recovery binds that
    # terminator instead of the chat "answer the user" move.
    channel_outage_recovery: str = Prompt.BROWSE_OUTAGE_RECOVERY_COLLECTOR

    def get_max_steps(self) -> int:
        return int(self.config.runtime.BACKGROUND_MAX_STEPS)

    def get_tools(self, run_id: str | None = None) -> list[Tool]:
        tools = super().get_tools(run_id)
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
