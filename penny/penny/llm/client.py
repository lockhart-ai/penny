"""LLM client for chat completions and embeddings.

Uses the OpenAI Python SDK, which works with any OpenAI-compatible API:
Ollama, omlx, OpenAI cloud, etc. Just change the base_url.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx
import openai

from penny.constants import PennyConstants
from penny.llm import raw_template, thinking_loop
from penny.llm.call_link import CallLink

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
from penny.llm.models import (
    FAULT_LOG_FIELD,
    PROVIDER_LOG_FIELD,
    PROVIDER_REQUEST_FIELD,
    PROVIDER_RESPONSE_FIELD,
    UNREPORTED_PROVIDER,
    CallOutcome,
    LlmConnectionError,
    LlmError,
    LlmFault,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
    PromptTransport,
    ProviderPin,
    RawFaithfulnessOutcome,
    RawFaithfulnessResult,
    UsageTiming,
    fault_for_status,
    usage_timing_from_response,
)

# NOT a top-level import (#2842): ``penny.validation``'s package init pulls in
# ``outcomes.py`` -> ``penny.agents.models`` -> ``penny.agents``'s own init ->
# ``agents/base.py``, which imports ``penny.llm`` for ``LlmClient`` -- a real
# cycle back to THIS module while it is still mid-import. Deferred into
# ``_log_thinking_loop_cut`` itself, where every module above is already
# fully loaded by the time anything actually calls it.

logger = logging.getLogger(__name__)

# Default API key for local inference servers that require one but don't check it
_DEFAULT_API_KEY = "not-needed"

# Reasoning is ON for every call, everywhere — not a setting.  A model that is not
# reasoning makes visibly worse decisions on the work Penny does, and the failure is quiet:
# it still answers, still calls tools, and only the DECISIONS degrade.
#
# It is hardcoded here rather than configured because the same weights reason or do not
# depending on who is serving them, which is a difference nothing in a run records. Ollama
# runs a hybrid model with thinking ON by default; a gateway serving the same model
# defaults it OFF. A full eval suite was spent comparing a model against itself that way —
# non-thinking remote against thinking local — and every conclusion drawn from it was
# about the switch rather than the model.
#
# Both bodies below carry TWO fields because the switch is read differently per runner
# (#2761): ``reasoning``/``reasoning_effort`` are what OpenRouter and Ollama's
# OpenAI-compatible endpoint read; oMLX ignores both and switches Gemma 4 thinking only
# through the chat template, ``chat_template_kwargs.enable_thinking`` — a field the other
# two runners silently ignore. Sending both on every call is what makes one body portable
# to all three without a backend-specific branch; prod was already getting this on oMLX
# only because a host-side proxy rewrote every request to add the kwarg.
#
# Read-only: sent as-is, never mutated. A runner that does not know a field ignores it
# (verified against Ollama's OpenAI-compatible endpoint), so no runner's path changes
# just because another runner's field rides alongside it.
REASONING_ENABLED_BODY = {
    "reasoning": {"enabled": True},
    "chat_template_kwargs": {"enable_thinking": True},
}

# The per-call override that turns reasoning OFF for one draw (#2474's probes, then
# #2607's follow-up). Only ``reasoning_effort: "none"`` — no ``reasoning`` key at all.
# #2474 found ``reasoning_effort: "none"`` alone disabled thinking on Ollama AND
# OpenRouter, but shipped it alongside ``reasoning: {enabled: False}`` anyway, never
# having probed the two fields TOGETHER. #2607's probe did: on local Ollama
# (gemma4:26b-mlx), a ``reasoning`` object PRESENT AT ALL keeps thinking on, whatever
# its ``enabled`` value — Ollama reads the key's presence, not its content, and that
# read overrides ``reasoning_effort``. A production extractor draw hit this: the
# thinking-off retry (#2476) sent both fields, thought for 31k characters, and was cut
# at its token budget exactly like the draw it was retrying. So this body must omit
# ``reasoning`` entirely, not set it to ``False`` — see ``_build_chat_kwargs``, which
# drops the always-on key for this call rather than updating it. ``chat_template_kwargs``
# carries the same off switch for oMLX (#2761) — it has no key-presence quirk of its own,
# so unlike ``reasoning`` it is simply set ``False`` rather than omitted.
THINKING_OFF_BODY = {
    "reasoning_effort": "none",
    "chat_template_kwargs": {"enable_thinking": False},
}

# Ask for embedding vectors as plain floats rather than the SDK's base64 default.
# Not every backend accepts base64: Google AI Studio refuses it outright ("embeddings do
# not support base64 encoding_format"), and the failure is the quiet kind — the write path
# stores a NULL vector and carries on, so a whole memory suite ran with no embeddings at
# all and still scored ~1.00. Floats are what every OpenAI-compatible backend accepts, and
# the SDK decodes either into the same list, so nothing downstream can tell the difference.
EMBEDDING_ENCODING_FORMAT = "float"

# Fallback when an error response carries no content-type header.
_UNKNOWN_CONTENT_TYPE = "unknown"

# Joins a provider's tool-call id to the position of a call split out of it.
_SPLIT_CALL_ID_SEPARATOR = "_"

# The fields of a dumped completion that hold its tool calls.
_CHOICES_FIELD = "choices"
_MESSAGE_FIELD = "message"
_TOOL_CALLS_FIELD = "tool_calls"
_FUNCTION_FIELD = "function"
_ARGUMENTS_FIELD = "arguments"
_ID_FIELD = "id"


def _summarize_llm_error(error: Exception) -> str:
    """Summarize an LLM API error for logging without dumping the raw body.

    Some OpenAI-compatible endpoints serve a non-JSON error (e.g. a 404 as an
    HTML page); logging the raw body dumps thousands of characters per line and
    buries real signal.  A structured JSON error surfaces its short ``message``
    field; an HTML / non-JSON body is reported by its content type and length
    only — never its content.
    """
    status = getattr(error, "status_code", None)
    prefix = f"HTTP {status}" if status is not None else type(error).__name__
    return f"{prefix}: {_llm_error_detail(error)}"


def _llm_error_detail(error: Exception) -> str:
    """Extract a short, body-free detail string from an LLM API error."""
    response = getattr(error, "response", None)
    if response is None:
        # Connection/timeout errors carry no HTTP body; their own message is short.
        return str(error)
    message = _json_error_message(getattr(error, "body", None))
    if message is not None:
        return message
    return _non_json_error_detail(response)


def _non_json_error_detail(response: httpx.Response) -> str:
    """Describe a non-JSON error body by type + length only, never its content."""
    content_type = response.headers.get("content-type", _UNKNOWN_CONTENT_TYPE)
    return f"non-JSON error body (content-type={content_type}, {len(response.text)} chars)"


def _json_error_message(body: Any) -> str | None:
    """Pull the ``message`` field from a structured OpenAI-style JSON error body."""
    if not isinstance(body, dict):
        return None
    error_field = body.get("error")
    if isinstance(error_field, dict) and isinstance(error_field.get("message"), str):
        return error_field["message"]
    message = body.get("message")
    return message if isinstance(message, str) else None


def _no_payload_detail(raw: Any, field: str) -> str:
    """Describe a 200 response that carried no payload, quoting the provider's own reason.

    A gateway in front of a model (OpenRouter, a proxy) can answer 200 with an ERROR
    payload where a completion belongs — the SDK parses it into a model whose ``choices``
    / ``data`` is None. That is not an ``openai`` error and never reaches the retry loop:
    it surfaces as a ``TypeError`` at the first subscript, kills the run, and leaves no
    record of what the provider actually said. Measured against OpenRouter, this ended 4
    of 10 samples with no reply at all.
    """
    message = _json_error_message(getattr(raw, "model_extra", None))
    detail = f": {message}" if message else ""
    return f"response carried no {field}{detail}"


def _provider_of(raw: Any) -> str | None:
    """Which upstream served this response, when the endpoint is a gateway that says so.

    A routing gateway puts the serving provider in a top-level field the SDK does not
    model, so it lands in ``model_extra``.  A direct endpoint sends nothing and gets
    ``None`` — the honest answer, and the one the tally renders as "unreported".
    """
    extra = getattr(raw, "model_extra", None)
    if not isinstance(extra, dict):
        return None
    provider = extra.get(PROVIDER_RESPONSE_FIELD)
    return provider if isinstance(provider, str) and provider else None


def _summarize_httpx_error(response: httpx.Response) -> str:
    """Summarize an httpx response error without dumping provider HTML."""
    try:
        message = _json_error_message(response.json())
    except ValueError:
        message = None
    if message:
        return f"HTTP {response.status_code}: {message}"
    return f"HTTP {response.status_code}: {_non_json_error_detail(response)}"


def _extract_model_ids(payload: Any) -> list[str]:
    """Extract model ids from common model-list JSON shapes."""
    items: Any = (
        payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else payload
    )

    if not isinstance(items, list):
        raise LlmResponseError("embedding model list response did not contain a list")

    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            ids.append(item["name"])
    return ids


# Either wire shape's own raw SDK response, by whichever transport
# (#2829) served the call — a chat completion, or a raw completions
# response, read from the plain OpenAI SDK, never a bespoke client.
_RawCompletion = openai.types.chat.ChatCompletion | openai.types.completion.Completion


class _ThinkingLoopCutLog(NamedTuple):
    """What ``chat()``'s own logging needs to write the discarded draw a
    thinking-loop cut threw away as its OWN ``promptlog`` row (#2842) —
    everything but the ids/snapshot ``chat()`` already has from its own
    ``call_link``/``messages_snapshot``. ``kept_thinking`` is the thought text
    kept up to the cut (this row's own ``thinking`` column); ``detail`` is the
    discard detail naming the repeat (window/threshold, characters kept);
    ``duration_ms`` is the CUT attempt's own wall time, not the
    continuation's."""

    kept_thinking: str
    detail: str
    duration_ms: int


class _RawTemplateDraw(NamedTuple):
    """One raw-template attempt's result (#2842): ``raw`` is the completion
    ``_attempt_chat_request`` parses as usual — the CONTINUATION's, when a
    loop was cut, so the caller never has to tell the two apart. ``cut`` is
    the discarded draw's own logging detail, or ``None`` when nothing was
    cut."""

    raw: openai.types.completion.Completion
    cut: _ThinkingLoopCutLog | None


class _StreamOutcome(NamedTuple):
    """One streamed raw-completions call's own outcome (#2842/#2843): ``raw``
    is ``None`` exactly when ``cut`` fired (the stream was closed before a
    real completion ever arrived) — never both at once. ``cut`` is a
    ``ThinkingLoopCut`` on the FIRST call's own ``ThoughtStreamTracker``
    (thought only), or a ``RepeatCut`` on the CONTINUATION's own
    ``RepeatStreamTracker`` (whole output) — the caller tells them apart by
    type, since they're handled completely differently (build a second
    continuation vs. hand off to the existing budget-cut path).
    ``last_chunk``/``last_choice`` are kept even when ``cut`` fired, so a
    caller that needs to build ITS OWN completion from the kept text (the
    budget-cut hand-off) has a real chunk to ``model_copy`` from."""

    raw: openai.types.completion.Completion | None
    cut: thinking_loop.ThinkingLoopCut | thinking_loop.RepeatCut | None
    duration_ms: int
    last_chunk: openai.types.completion.Completion | None = None
    last_choice: openai.types.completion_choice.CompletionChoice | None = None


def _completed_raw(outcome: _StreamOutcome) -> openai.types.completion.Completion:
    """``outcome.raw`` narrowed to non-``None`` (#2842) — true by construction
    whenever ``outcome.cut`` is ``None`` (``_stream_raw_completion`` never
    returns one without the other), but the type checker has no way to read
    that invariant off a plain ``NamedTuple``; this is the one place that
    narrowing is spelled out, rather than an unchecked cast at each call site.
    Raises, never asserts (an ``assert`` is stripped under ``python -O``,
    which would let a broken invariant through as a typed ``None``) — this
    branch is unreachable today, so a raise here names a real defect in
    ``_stream_raw_completion`` itself, not a caller's mistake.
    """
    if outcome.raw is None:
        raise LlmResponseError("a stream outcome with no cut must carry a completion")
    return outcome.raw


def _budget_cut_raw(outcome: _StreamOutcome) -> openai.types.completion.Completion:
    """Build a completion reporting AS IF the token budget had been hit
    (#2843) — the CONTINUATION looped a SECOND time; rather than a third
    cut-and-continue (open-ended recursion the issue doesn't want), this
    hands the draw to Penny's EXISTING budget-cut discard-and-reroll path
    (``agents/base.py``'s own ``finish_reason == FINISH_REASON_LENGTH`` ->
    ``ConditionKey.BUDGET_CUT``) by reporting the SAME transport fact a real
    budget cut reports, with the cleanly-trimmed kept text as the final
    content. Bounds a repeat to two cycles instead of the full budget.
    """
    cut = outcome.cut
    if not isinstance(cut, thinking_loop.RepeatCut):
        raise LlmResponseError("a RepeatStreamTracker must only ever produce a RepeatCut")
    if outcome.last_chunk is None or outcome.last_choice is None:
        raise LlmResponseError("a repeated continuation must carry at least one streamed chunk")
    choice = outcome.last_choice.model_copy(
        update={"text": cut.kept_text, "finish_reason": PennyConstants.FINISH_REASON_LENGTH}
    )
    return outcome.last_chunk.model_copy(update={"choices": [choice]})


class _ChatAttemptResult(NamedTuple):
    """One chat attempt's own outcome: the raw completion the attempt drew, and
    its parsed response — or ``None`` for ``response`` when the completion
    carried no choices at all. ``_chat_attempts`` reads ``response`` to tell
    the two apart rather than the shape of what came back, so a successful
    draw and a 200-with-no-completion draw are the same return type.
    ``thinking_loop_cut`` (#2842) is the raw-template transport's own discarded
    cut draw, logged separately by ``chat()`` alongside this attempt's real
    response — ``None`` on the chat transport, which never streams, and on a
    raw-template attempt that never looped.
    """

    raw: _RawCompletion
    response: LlmResponse | None
    thinking_loop_cut: _ThinkingLoopCutLog | None = None


class _RawTemplateParseError(Exception):
    """Internal-only (#2833): carries the raw completion alongside the
    ``LlmToolParseError`` it wraps, from the point in ``_attempt_chat_request``
    where both are in scope, up to ``chat()``'s own logging — the one place
    with the ids/snapshot a ``promptlog`` row needs.

    Never escapes ``chat()``: it is caught there, the original
    ``LlmToolParseError`` (unchanged — same message, same ``fault``) is what
    the caller actually sees raised. Not a public error class, and not a new
    fault — purely plumbing so the prompt-log row can keep the raw completion
    text instead of discarding it.
    """

    def __init__(self, error: LlmToolParseError, raw: _RawCompletion) -> None:
        super().__init__(str(error))
        self.error = error
        self.raw = raw


class LlmClient:
    """Client for LLM inference via OpenAI-compatible APIs.

    Works with Ollama, omlx, or any OpenAI-compatible server.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        db: Any = None,
        *,
        max_retries: int,
        retry_delay: float,
        api_key: str = _DEFAULT_API_KEY,
        timeout: float | None = None,
        provider_pin: ProviderPin | None = None,
        runtime: RuntimeParams | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.db = db
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        self.provider_pin = provider_pin
        # The deployment's ONE ``RuntimeParams`` (#2474) — not a fresh instance built
        # from ``db`` alone, which would carry the DB tier but never the env tier (the
        # env overrides are collected once, in ``Config.load()``, into this exact
        # object).  A micro-context customer (``tools/micro_context.py``) reads its own
        # call kind's token budget off ``self._model_client.runtime`` rather than
        # building its own, so all eight budgets resolve DB → env → default alike —
        # the same chain ``agents/base.py`` already gets through ``self.config.runtime``.
        # ``None`` only where a caller builds a client with no runtime at all (a test, a
        # one-off script); the resolver then falls back to a default-only lookup.
        self.runtime = runtime

        client_kwargs: dict[str, Any] = {
            "base_url": f"{self.api_url}/v1",
            "api_key": api_key,
            "max_retries": 0,  # We handle retries ourselves
        }
        if timeout is not None:
            # Keep connect timeout short; only extend read/write for slow models.
            client_kwargs["timeout"] = httpx.Timeout(
                timeout=timeout, connect=PennyConstants.LLM_CONNECT_TIMEOUT_SECONDS
            )

        self.client = openai.AsyncOpenAI(**client_kwargs)

        # Structural transport dispatch (#2829): a model with a vendored chat
        # template renders it and calls raw completions; any other model uses
        # the chat-completions API, as before.  Mutable only through
        # ``disable_raw_template_transport`` — the startup preflight's
        # raw-faithfulness check names the one case this flips at runtime.
        self._raw_template_enabled = raw_template.has_vendored_template(model)

        logger.info("Initialized LLM client: url=%s, model=%s", api_url, model)

    def disable_raw_template_transport(self, reason: str) -> None:
        """Fall back this client to the chat transport for the rest of its life.

        Called by the startup preflight when the raw-faithfulness check finds
        this deployment's endpoint does not serve the model's raw completions
        faithfully (a chat-wrapping provider pinned to a model this client
        would otherwise route raw) — a named degraded mode, never a silent
        one (``docs/principles.md``'s visible-degradation principle).
        """
        self._raw_template_enabled = False
        logger.warning("Raw template transport disabled for %s: %s", self.model, reason)

    @property
    def raw_template_enabled(self) -> bool:
        """Whether this client currently routes this model through the
        raw-template transport (#2829) — read by the startup preflight to
        decide whether the raw-faithfulness check applies at all."""
        return self._raw_template_enabled

    # ── Summary ──────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[LlmMessage],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        *,
        call_link: CallLink,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_target: str | None = None,
        token_budget: int | None = None,
        thinking_enabled: bool = True,
    ) -> LlmResponse:
        """Generate a chat completion with optional tool calling.

        ``messages`` is typed conversation history (#2563) — the one place it
        becomes a dict is right here, via ``LlmMessage.to_log_dict()``, before
        anything below (the retry loop, ``_translate_messages``, the prompt log)
        runs. Everything from here down is unchanged: it always dealt in dicts
        and still does — only the one hand-off at this boundary is new.

        On the raw-template transport (#2829), a looping thought is cut mid-stream
        and the SAME model turn continues in one more raw call (#2842,
        ``_send_raw_template_request``) — invisibly to this method's own
        interface: exactly one ``LlmResponse`` still comes out, and the retry
        loop sees exactly one attempt. The discarded cut draw is logged as its
        OWN, separately discarded ``promptlog`` row right before this call's
        normal completed one (``_log_thinking_loop_cut``), reusing the same
        discard machinery a budget cut is marked with — no new column.

        Logs ONE ``promptlog`` row for this call, however it ends — completed,
        cancelled, timed out, or raised (#2355 part A) — written once at the END,
        whichever way it ends, rather than inserted at send and completed later: that
        covers every case seen and adds no "in flight" state a reader has to handle.
        ``duration_ms`` is the call's TOTAL wall time from entry, across every
        retried attempt — not just the one that happened to succeed — so a call
        retried twice before answering reports its real cost, which is what the eval
        harness's token-cost accounting reads.

        ``call_link`` (#2458) is REQUIRED — no ambient state, no ``X or <mint>``
        fallback here (the code owner's ruling: the type checker is the
        enforcement). It names this call's own ``run_id`` and, for a draw made on
        behalf of an enclosing run (a micro-context, which mints its own ``run_id``
        via :meth:`~penny.llm.call_link.CallLink.child` for its own ledger
        attribution) or from inside a dispatched tool call, that run's/call's id —
        read straight off the link, never re-derived here.

        ``token_budget`` (#2474) is the caller's own cap for this call KIND — read off
        ``penny.config_params.token_budget_for`` by every production call site, never
        hardcoded here — sent under both ``max_tokens`` and ``max_completion_tokens``
        (see ``_build_chat_kwargs``) so a draw that runs long is cut rather than left
        to a wall-clock timeout, which on a slow local model turns 9 minutes of
        deliberation into a dropped connection with no record of why. ``None`` sends
        no cap at all (today's behaviour). What a CUT draw does next — reacting to
        ``finish_reason="length"`` on the returned response — is the follow-up
        issues' (#2475/#2476); this only sends the cap and reads it back.

        ``thinking_enabled`` (#2474) is the per-call reasoning switch the probes on
        this ticket found: unlike the always-on default (``REASONING_ENABLED_BODY``),
        a caller can ask for one draw with thinking off.  No production caller passes
        it yet — it exists for the follow-up issues that give a capped call a second,
        reasoning-off attempt to answer with what it has.

        Reads top to bottom as the call's own table of contents: resolve this
        call's ids and announce it, run the retry loop, then log however it
        ended.
        """
        resolved_run_id, parent_run_id, parent_tool_call_id = self._start_call(
            call_link, agent_name, prompt_type, run_target
        )
        start = time.time()
        # Decided once, structurally, for the whole call (#2829) — never
        # re-derived per attempt, and named on BOTH the success and failure
        # log rows, so a reader can tell which wire shape a raised/cancelled
        # call was even attempting.
        transport = self._resolve_transport(messages, format)
        # The one hand-off (#2563): typed history becomes the internal dict shape
        # here, once, before the retry loop and the prompt log — both unchanged,
        # both still dealing purely in dicts, exactly as before this call's entry.
        messages_snapshot = [message.to_log_dict() for message in messages]
        try:
            response, raw, thinking_loop_cut = await self._chat_attempts(
                messages_snapshot, tools, format, token_budget, thinking_enabled, transport
            )
        except asyncio.CancelledError:
            self._log_failed_call(
                messages_snapshot,
                tools,
                start,
                agent_name,
                prompt_type,
                resolved_run_id,
                run_target,
                parent_run_id,
                parent_tool_call_id,
                transport,
                CallOutcome.CANCELLED,
            )
            raise
        except _RawTemplateParseError as failure:
            # The raw completion was already fetched before parsing failed —
            # kept as this call's logged response instead of the usual "no
            # response" `{}` (#2833), so what the model actually wrote survives
            # for diagnosis. The error the caller sees is the original
            # `LlmToolParseError`, unchanged — this wrapper never escapes here.
            self._log_failed_call(
                messages_snapshot,
                tools,
                start,
                agent_name,
                prompt_type,
                resolved_run_id,
                run_target,
                parent_run_id,
                parent_tool_call_id,
                transport,
                CallOutcome.RAISED,
                error=failure.error,
                raw=failure.raw,
            )
            raise failure.error from failure.__cause__
        except LlmError as error:
            outcome = (
                CallOutcome.TIMED_OUT if isinstance(error, LlmTimeoutError) else CallOutcome.RAISED
            )
            self._log_failed_call(
                messages_snapshot,
                tools,
                start,
                agent_name,
                prompt_type,
                resolved_run_id,
                run_target,
                parent_run_id,
                parent_tool_call_id,
                transport,
                outcome,
                error=error,
            )
            raise

        if thinking_loop_cut is not None:
            self._log_thinking_loop_cut(
                thinking_loop_cut,
                messages_snapshot,
                tools,
                agent_name,
                prompt_type,
                resolved_run_id,
                run_target,
                parent_run_id,
                parent_tool_call_id,
            )
        return self._log_successful_call(
            response,
            raw,
            messages_snapshot,
            tools,
            start,
            agent_name,
            prompt_type,
            resolved_run_id,
            run_target,
            parent_run_id,
            parent_tool_call_id,
            transport,
        )

    def _resolve_transport(
        self, messages: list[LlmMessage], format: dict | str | None
    ) -> PromptTransport:
        """Which wire shape this call uses — decided once, before the retry
        loop, from facts that never change between attempts: the model's own
        identity, and whether THIS call carries an image or a structured
        ``format`` (#2829).  Neither has a raw-completions equivalent — an
        image needs the provider's own vision processor, and ``format`` is an
        OpenAI-API-level field gemma's raw template has no construct for — so
        both force the chat transport regardless of the model.
        """
        if not self._raw_template_enabled:
            return PromptTransport.CHAT
        if format is not None:
            return PromptTransport.CHAT
        if any(message.images for message in messages):
            return PromptTransport.CHAT
        return PromptTransport.RAW_TEMPLATE

    # ── Chat: call setup and logging ────────────────────────────────────

    def _start_call(
        self,
        call_link: CallLink,
        agent_name: str | None,
        prompt_type: str | None,
        run_target: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Resolve this call's own run/parent ids off ``call_link`` (#2458 — no
        ambient state, read straight off the link) and announce it (#2439),
        before the network request — a live Prompts tab's only signal that
        anything is happening until the call's own ``promptlog`` row eventually
        lands.  Returns the ids ``chat()`` threads through its own logging.
        """
        resolved_run_id = call_link.run_id
        parent_run_id = call_link.parent_run_id
        parent_tool_call_id = call_link.parent_tool_call_id
        self._publish_call_started(
            resolved_run_id, parent_run_id, parent_tool_call_id, agent_name, prompt_type, run_target
        )
        return resolved_run_id, parent_run_id, parent_tool_call_id

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        """Milliseconds since ``start`` (``time.time()`` at the top of ``chat()``)."""
        return int((time.time() - start) * 1000)

    def _log_failed_call(
        self,
        messages_snapshot: list[dict],
        tools: list[dict] | None,
        start: float,
        agent_name: str | None,
        prompt_type: str | None,
        resolved_run_id: str,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        transport: PromptTransport,
        outcome: CallOutcome,
        *,
        error: Exception | None = None,
        raw: _RawCompletion | None = None,
    ) -> None:
        """Log this call's ONE ``promptlog`` row for a cancelled, timed-out, or
        raised call (#2355 part A) — no parsed response to persist, so
        ``thinking`` goes through as ``None``.

        ``raw`` is ``None`` for every outcome except a raw-template tool-call
        parse failure (#2833), whose caller passes the completion already
        fetched before parsing failed — ``_log_call`` dumps it unreshaped
        (no parsed ``LlmResponse`` to reshape it around), so the row keeps the
        raw completion text instead of the usual ``{}``.
        """
        self._log_call(
            messages_snapshot,
            raw,
            None,
            tools,
            None,
            self._elapsed_ms(start),
            agent_name,
            prompt_type,
            resolved_run_id,
            run_target,
            parent_run_id,
            parent_tool_call_id,
            transport,
            outcome,
            error=error,
        )

    def _log_thinking_loop_cut(
        self,
        cut: _ThinkingLoopCutLog,
        messages_snapshot: list[dict],
        tools: list[dict] | None,
        agent_name: str | None,
        prompt_type: str | None,
        run_id: str | None,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
    ) -> None:
        """Log the draw a thinking-loop cut threw away as its OWN ``promptlog``
        row (#2842) — reusing the same discard machinery a budget cut is
        marked with (``ConditionKey``/``mark_prompt_discarded``), not a new
        column: this row's ``call_outcome`` is COMPLETED (a real draw came
        back, it was just not the one used), and ``discard_reason``/
        ``discard_detail`` name why. The CONTINUATION's own row is the normal
        one ``_log_successful_call`` writes right after this one, carrying the
        response ``chat()``'s caller actually gets.
        """
        if not self.db:
            return
        row_id = self.db.messages.log_prompt(
            **self._thinking_loop_cut_log_kwargs(
                cut,
                messages_snapshot,
                tools,
                agent_name,
                prompt_type,
                run_id,
                run_target,
                parent_run_id,
                parent_tool_call_id,
            )
        )
        if row_id is not None:
            self._mark_thinking_loop_discarded(row_id, cut.detail)

    def _thinking_loop_cut_log_kwargs(
        self,
        cut: _ThinkingLoopCutLog,
        messages_snapshot: list[dict],
        tools: list[dict] | None,
        agent_name: str | None,
        prompt_type: str | None,
        run_id: str | None,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
    ) -> dict[str, Any]:
        """The cut draw's own ``log_prompt`` kwargs (#2842) — ``call_outcome``
        is COMPLETED (a real draw came back) with no usage/timing fields: the
        stream was closed before any usage ever arrived, so there is none to
        report, not a guessed zero."""
        return {
            "model": self.model,
            "messages": messages_snapshot,
            "response": _thinking_loop_cut_dump(cut.kept_thinking),
            "tools": tools,
            "thinking": cut.kept_thinking,
            "duration_ms": cut.duration_ms,
            "agent_name": agent_name,
            "prompt_type": prompt_type,
            "run_id": run_id,
            "run_target": run_target,
            "parent_run_id": parent_run_id,
            "parent_tool_call_id": parent_tool_call_id,
            "transport": PromptTransport.RAW_TEMPLATE.value,
            "call_outcome": CallOutcome.COMPLETED.value,
        }

    def _mark_thinking_loop_discarded(self, row_id: int, detail: str) -> None:
        """Stamp the cut draw's own row with the shared discard machinery
        (#2842) — the same ``ConditionKey``/``mark_prompt_discarded`` path a
        budget cut is marked with, no new column.

        Deferred import: ``penny.validation``'s package init pulls in
        ``outcomes.py`` -> ``penny.agents.models`` -> ``penny.agents``'s own
        init -> ``agents/base.py``, which imports ``penny.llm`` for
        ``LlmClient`` — a real cycle back to THIS module while it is still
        mid-import. Safe here: every module above is already fully loaded by
        the time anything actually calls this.
        """
        from penny.validation.conditions import ConditionKey

        # Only ever called right after this same call's own successful
        # ``db.messages.log_prompt`` (see ``_log_thinking_loop_cut``), so
        # ``self.db`` is already known truthy here — no re-check needed.
        self.db.messages.mark_prompt_discarded(row_id, ConditionKey.THINKING_LOOP.value, detail)

    def _log_successful_call(
        self,
        response: LlmResponse,
        raw: _RawCompletion,
        messages_snapshot: list[dict],
        tools: list[dict] | None,
        start: float,
        agent_name: str | None,
        prompt_type: str | None,
        resolved_run_id: str,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        transport: PromptTransport,
    ) -> LlmResponse:
        """Log the response and this call's ONE completed ``promptlog`` row
        (#2355 part A), then hand the response back to ``chat()``'s caller.

        The row's id is stamped on the response (#2702) — logging identity only,
        so a rejection site that discards this draw's output can mark its own
        row afterwards; ``None`` when no row was written (no ``db``)."""
        thinking = response.thinking or response.message.thinking
        self._log_response(response, thinking)
        response.prompt_log_id = self._log_call(
            messages_snapshot,
            raw,
            response,
            tools,
            thinking,
            self._elapsed_ms(start),
            agent_name,
            prompt_type,
            resolved_run_id,
            run_target,
            parent_run_id,
            parent_tool_call_id,
            transport,
            CallOutcome.COMPLETED,
        )
        return response

    # ── Chat: the retry loop ─────────────────────────────────────────────

    async def _backoff(self, attempt: int) -> None:
        """Wait before re-attempting a failed call, doubling with each attempt.

        A remote endpoint's failures arrive in bursts — a provider draining a queue, a
        node cycling — so a flat delay re-attempts three times into the same bad moment
        and reports the burst as three separate failures.  Doubling spends the same first
        wait and gives the last attempt a materially later one.
        """
        await asyncio.sleep(self.retry_delay * 2**attempt)

    def _log_attempt(
        self, attempt: int, *, fault: LlmFault | None, provider: str | None, detail: str
    ) -> None:
        """Record ONE chat attempt's outcome, with the fault class and provider as VALUES.

        One record per attempt, at one chokepoint, because the alternative is what was
        there before: a different sentence per failure branch, counted only by a human
        grepping afterwards.  The message stays human; ``llm_fault`` and ``llm_provider``
        ride alongside it as structured fields, so a tally reads them off the record and
        never has to recognise a phrasing.  A success is DEBUG (it is not news), a fault
        is WARNING (it is).
        """
        logger.log(
            logging.DEBUG if fault is None else logging.WARNING,
            "LLM chat attempt %d/%d: %s via %s%s",
            attempt + 1,
            self.max_retries,
            fault.value if fault is not None else "ok",
            provider or UNREPORTED_PROVIDER,
            f" — {detail}" if detail else "",
            extra={FAULT_LOG_FIELD: fault, PROVIDER_LOG_FIELD: provider},
        )

    async def _chat_attempts(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        format: dict | str | None,
        token_budget: int | None,
        thinking_enabled: bool,
        transport: PromptTransport,
    ) -> tuple[LlmResponse, _RawCompletion, _ThinkingLoopCutLog | None]:
        """The retry loop over individual network attempts.

        Unchanged in substance from before #2355 — each attempt's own outcome
        still goes through ``_log_attempt`` (the human-readable DEBUG/WARNING line,
        one per attempt) — except it no longer writes the call's ``promptlog`` row
        itself: ``chat()`` does that once, for the WHOLE call, from whichever way
        this returns or raises.  Returns the parsed response alongside the raw
        completion the caller persists, and the attempt's own discarded
        thinking-loop cut, if any (#2842 — ``None`` on every attempt that didn't
        loop, or that failed outright: a failed CONTINUATION call goes through
        this same retry loop with no special path, so its own cut draw is
        simply never logged, exactly like any other failed attempt's work);
        raises an ``LlmError`` subclass (or lets a ``CancelledError`` through
        untouched) when no attempt produced one.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = await self._attempt_chat_request(
                    messages, tools, format, token_budget, thinking_enabled, attempt, transport
                )
            except LlmError:
                raise
            except openai.NotFoundError as error:
                raise self._not_found_error(attempt, error) from error
            except openai.APITimeoutError as error:
                last_error = await self._retry_after_timeout(attempt, error)
            except openai.APIConnectionError as error:
                last_error = await self._retry_after_connection_error(attempt, error)
            except openai.OpenAIError as error:
                last_error = await self._retry_after_openai_error(attempt, error)
            else:
                if result.response is not None:
                    return result.response, result.raw, result.thinking_loop_cut
                last_error = await self._retry_after_no_choices(attempt, result.raw)

        logger.error("LLM chat failed after %d attempts: %s", self.max_retries, last_error)
        if last_error is None:
            raise LlmResponseError("LLM chat exhausted retries without a recorded error")
        raise last_error

    async def _attempt_chat_request(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        format: dict | str | None,
        token_budget: int | None,
        thinking_enabled: bool,
        attempt: int,
        transport: PromptTransport,
    ) -> _ChatAttemptResult:
        """Send and parse ONE request, over whichever transport this call resolved to.

        ``result.response`` is ``None`` when the 200 carried no completion at
        all — ``_retry_after_no_choices`` is what classifies that as a
        retryable fault: an ``LlmError`` raised from here would re-raise
        untried (the first ``except`` clause in ``_chat_attempts``), and a
        provider answering 200-with-no-completion is exactly the transient
        another attempt usually gets past.
        """
        logger.debug("Sending chat request (attempt %d/%d)", attempt + 1, self.max_retries)
        translated_messages = self._translate_messages(messages)
        thinking_loop_cut: _ThinkingLoopCutLog | None = None
        if transport is PromptTransport.RAW_TEMPLATE:
            draw = await self._send_raw_template_request(
                translated_messages, tools, token_budget, thinking_enabled
            )
            raw = draw.raw
            thinking_loop_cut = draw.cut
            if not raw.choices:
                return _ChatAttemptResult(
                    raw=raw, response=None, thinking_loop_cut=thinking_loop_cut
                )
            try:
                response = self._parse_raw_template_response(raw)
            except LlmToolParseError as error:
                # The agent loop's own ``except LlmToolParseError`` already logs
                # this DEBUG line (``penny/agents/base.py``) — nothing to add
                # here. ``error``'s message carries ~200 characters of context
                # around the failure point (#2833, ``raw_template._context``),
                # so that line names the offending text with no change needed
                # on its side. Wrapped so the raw completion, already fetched,
                # reaches ``chat()``'s own logging instead of being discarded.
                raise _RawTemplateParseError(error, raw) from error
        else:
            kwargs = self._build_chat_kwargs(
                translated_messages, tools, format, token_budget, thinking_enabled
            )
            raw = await self.client.chat.completions.create(**kwargs)
            if not raw.choices:
                return _ChatAttemptResult(raw=raw, response=None)
            response = self._parse_response(raw)

        self._log_attempt(attempt, fault=None, provider=response.provider, detail="")
        return _ChatAttemptResult(raw=raw, response=response, thinking_loop_cut=thinking_loop_cut)

    async def _send_raw_template_request(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        token_budget: int | None,
        thinking_enabled: bool,
    ) -> _RawTemplateDraw:
        """Render the vendored template and STREAM the result as a raw
        completion (#2829, streamed since #2842): streaming lets a looping
        thought be cut mid-generation (both oMLX and OpenRouter/CoreWeave
        confirmed stopping real generation the moment the stream closes,
        rather than finishing it unseen) instead of running to its token
        budget. On a cut, one more raw call continues the SAME model turn
        from the kept thinking, closed — ``draw.raw`` is always that
        continuation's completion when a cut happened, so
        ``_attempt_chat_request`` never has to tell the two apart.
        ``thinking_enabled`` is already baked into the rendered prompt's own
        bytes, so it is not sent as a separate field here.

        The continuation is ITSELF guarded (#2843, ``RepeatStreamTracker`` —
        its whole output, content included, not just thinking): a cut
        landing mid-draft let a real prod continuation just resume an
        unfinished line and loop again under its own token budget. A second
        loop there is not given a THIRD call — it hands off to Penny's
        existing budget-cut path instead (:func:`_budget_cut_raw`), bounding
        a repeat to two cycles rather than the full budget.
        """
        translated_tools = self._translate_tools(tools) if tools else None
        prompt = raw_template.render_prompt(messages, translated_tools, thinking_enabled)
        outcome = await self._stream_raw_completion(
            await self._create_raw_completion(prompt, token_budget),
            tracker=thinking_loop.ThoughtStreamTracker(),
        )
        if outcome.cut is None:
            return _RawTemplateDraw(raw=_completed_raw(outcome), cut=None)
        if not isinstance(outcome.cut, thinking_loop.ThinkingLoopCut):
            raise LlmResponseError(
                "a ThoughtStreamTracker must only ever produce a ThinkingLoopCut"
            )
        continuation_prompt = prompt + outcome.cut.continuation_suffix()
        continuation = await self._stream_raw_completion(
            await self._create_raw_completion(continuation_prompt, token_budget),
            tracker=thinking_loop.RepeatStreamTracker(),
        )
        cut_log = _ThinkingLoopCutLog(
            kept_thinking=outcome.cut.kept_body,
            detail=outcome.cut.discard_detail(),
            duration_ms=outcome.duration_ms,
        )
        if continuation.cut is not None:
            return _RawTemplateDraw(raw=_budget_cut_raw(continuation), cut=cut_log)
        return _RawTemplateDraw(raw=_completed_raw(continuation), cut=cut_log)

    async def _create_raw_completion(
        self, prompt: str, token_budget: int | None
    ) -> openai.AsyncStream[openai.types.completion.Completion]:
        """One streaming raw-completions request (#2842) — the provider pin
        rides the same ``extra_body`` passthrough as the chat transport
        (:meth:`_provider_pin_extra_body`); ``stream_options.include_usage``
        asks for a final usage-bearing chunk so a non-cut draw still reports
        token counts and timing, exactly as the non-streamed call used to.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "stream_options": {"include_usage": True},
            "stop": list(raw_template.STOP_SEQUENCES),
            "extra_body": self._provider_pin_extra_body(),
        }
        if token_budget is not None:
            kwargs["max_tokens"] = token_budget
        return await self.client.completions.create(**kwargs)

    async def _stream_raw_completion(
        self,
        stream: openai.AsyncStream[openai.types.completion.Completion],
        *,
        tracker: thinking_loop.ThoughtStreamTracker | thinking_loop.RepeatStreamTracker,
    ) -> _StreamOutcome:
        """Consume one streamed raw-completions call chunk by chunk (#2842),
        closing it (``async with`` — the SDK's own ``AsyncStream.close()``)
        the moment ``tracker`` detects a cut rather than reading to the end.

        ``tracker`` names which guard applies: a ``ThoughtStreamTracker`` on
        the FIRST call (thought only), a ``RepeatStreamTracker`` on the
        CONTINUATION (#2843 — its whole output, content included, since a
        cut landing mid-draft is exactly what let a real prod continuation
        loop again under its own token budget). Either way this is never
        chased past one cut per call — the caller (``_send_raw_template_request``)
        decides what a cut on EACH tracker means, never a THIRD call.
        """
        start = time.time()
        last_chunk, last_choice, buffer, cut = await self._consume_stream(stream, tracker)
        if cut is not None:
            return _StreamOutcome(
                raw=None,
                cut=cut,
                duration_ms=self._elapsed_ms(start),
                last_chunk=last_chunk,
                last_choice=last_choice,
            )
        if last_chunk is None:
            # A stream that closed with literally no chunks at all — not even
            # a trailing usage-only one. Never seen against oMLX or CoreWeave
            # (confirmed both always send at least one); raised rather than
            # guessed at, the same way any other genuinely empty response is
            # (``_no_payload_detail``) rather than silently treated as a cut.
            raise LlmResponseError(
                "raw-template stream closed with no chunks at all", fault=LlmFault.NO_CHOICES
            )
        return _StreamOutcome(
            raw=_reconstruct_completion(last_chunk, last_choice, buffer),
            cut=None,
            duration_ms=self._elapsed_ms(start),
        )

    @staticmethod
    async def _consume_stream(
        stream: openai.AsyncStream[openai.types.completion.Completion],
        tracker: thinking_loop.ThoughtStreamTracker | thinking_loop.RepeatStreamTracker,
    ) -> tuple[
        openai.types.completion.Completion | None,
        openai.types.completion_choice.CompletionChoice | None,
        str,
        thinking_loop.ThinkingLoopCut | thinking_loop.RepeatCut | None,
    ]:
        """Read ``stream`` chunk by chunk, closing it (``async with`` —
        ``_stream_raw_completion``'s own docstring covers why) the moment
        ``tracker`` detects a cut rather than reading to the end. Returns the
        last chunk/choice seen (``None``/``None`` for a stream with no chunks
        at all), the accumulated text, and the cut if one fired — the four
        pieces ``_stream_raw_completion`` needs to build its own result,
        whichever way this ends.

        Runs ``tracker.final_check`` once after the stream ends with no cut
        found mid-way (#2843): the stream almost never ends exactly on a
        ``CHECK_INTERVAL_CHARS`` boundary, so without this a loop sitting in
        that last, sub-interval stretch would go undetected entirely.
        """
        buffer = ""
        last_chunk: openai.types.completion.Completion | None = None
        last_choice: openai.types.completion_choice.CompletionChoice | None = None
        async with stream:
            async for chunk in stream:
                last_chunk = chunk
                if chunk.choices:
                    last_choice = chunk.choices[0]
                    buffer += last_choice.text
                cut = tracker.feed(buffer)
                if cut is not None:
                    return last_chunk, last_choice, buffer, cut
        return last_chunk, last_choice, buffer, tracker.final_check(buffer)

    def _provider_pin_extra_body(self) -> dict[str, Any]:
        """The provider pin's own ``extra_body`` passthrough (#2829) — the ONE
        place that builds it, so the raw-template request and the
        raw-faithfulness check can never drift apart on whether the pin rides
        along.  (They did: the check called ``completions.create`` with no
        pin at all, so an OpenRouter run pinned to CoreWeave could still be
        answered by any provider on the check — usually a chat-wrapping one —
        while the real raw-template calls stayed correctly pinned.)  Empty
        when no pin is configured, same as every other passthrough here.
        """
        extra_body: dict[str, Any] = {}
        if self.provider_pin is not None:
            extra_body[PROVIDER_REQUEST_FIELD] = self.provider_pin.as_request_field()
        return extra_body

    @staticmethod
    def _parse_raw_template_response(raw: openai.types.completion.Completion) -> LlmResponse:
        """Parse one raw completion's text into our ``LlmResponse`` (#2829) —
        the raw-template counterpart of ``_parse_response``."""
        choice = raw.choices[0]
        thinking, tool_calls, content = raw_template.parse_completion(choice.text or "")
        return LlmResponse(
            message=LlmMessage(
                role="assistant", content=content, tool_calls=tool_calls, thinking=thinking
            ),
            thinking=thinking,
            model=raw.model,
            provider=_provider_of(raw),
            finish_reason=choice.finish_reason,
        )

    def _not_found_error(self, attempt: int, error: openai.NotFoundError) -> LlmNotFoundError:
        """Log, then build the terminal error for a model that doesn't exist — never retried."""
        summary = _summarize_llm_error(error)
        self._log_attempt(attempt, fault=LlmFault.NOT_FOUND, provider=None, detail=summary)
        logger.error("LLM chat failed (model not found, no retry): %s", summary)
        return LlmNotFoundError(summary)

    async def _retry_after_timeout(
        self, attempt: int, error: openai.APITimeoutError
    ) -> LlmTimeoutError:
        """Log a request timeout, then back off before the next attempt, if one remains."""
        last_error = LlmTimeoutError(str(error))
        self._log_attempt(attempt, fault=LlmFault.TIMEOUT, provider=None, detail=str(error))
        if attempt < self.max_retries - 1:
            await self._backoff(attempt)
        return last_error

    async def _retry_after_connection_error(
        self, attempt: int, error: openai.APIConnectionError
    ) -> LlmConnectionError:
        """Log a connection failure, then back off before the next attempt, if one remains."""
        last_error = LlmConnectionError(str(error))
        self._log_attempt(attempt, fault=LlmFault.CONNECTION, provider=None, detail=str(error))
        if attempt < self.max_retries - 1:
            await self._backoff(attempt)
        return last_error

    async def _retry_after_openai_error(
        self, attempt: int, error: openai.OpenAIError
    ) -> LlmResponseError:
        """Classify a generic OpenAI error, raising immediately for an unretryable
        tool-parse failure; otherwise log it and back off before the next
        attempt, if one remains.
        """
        summary = _summarize_llm_error(error)
        # 500 "error parsing tool call" means the model produced plain text
        # instead of a JSON tool call. Retrying with the same messages won't
        # help — raise immediately so the agent can inject a format nudge.
        if getattr(error, "status_code", None) == 500 and "error parsing tool call" in str(error):
            self._log_attempt(attempt, fault=LlmFault.TOOL_PARSE, provider=None, detail=summary)
            logger.warning("Tool parse error — model returned plain text instead of JSON tool call")
            raise LlmToolParseError(summary) from error
        fault = fault_for_status(getattr(error, "status_code", None))
        last_error = LlmResponseError(summary, fault=fault)
        self._log_attempt(attempt, fault=fault, provider=None, detail=summary)
        if attempt < self.max_retries - 1:
            await self._backoff(attempt)
        return last_error

    async def _retry_after_no_choices(
        self, attempt: int, raw: openai.types.chat.ChatCompletion
    ) -> LlmResponseError:
        """Log a 200 that carried no completion as a retryable fault — checked
        BEFORE parsing rather than raised (see ``_attempt_chat_request``) —
        then back off before the next attempt, if one remains.
        """
        last_error = LlmResponseError(_no_payload_detail(raw, "choices"), fault=LlmFault.NO_CHOICES)
        self._log_attempt(
            attempt, fault=LlmFault.NO_CHOICES, provider=_provider_of(raw), detail=str(last_error)
        )
        if attempt < self.max_retries - 1:
            await self._backoff(attempt)
        return last_error

    # ── Generate (chat wrapper) ──────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        *,
        call_link: CallLink,
        agent_name: str | None = None,
        prompt_type: str | None = None,
    ) -> LlmResponse:
        """Generate a completion for a prompt (converts to chat format internally)."""
        messages = [LlmMessage(role="user", content=prompt)]
        return await self.chat(
            messages,
            tools,
            format,
            call_link=call_link,
            agent_name=agent_name,
            prompt_type=prompt_type,
        )

    # ── Embeddings ───────────────────────────────────────────────────────

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Generate embeddings for one or more texts."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Sending embed request (attempt %d/%d)", attempt + 1, self.max_retries)

                response = await self.client.embeddings.create(
                    model=self.model, input=text, encoding_format=EMBEDDING_ENCODING_FORMAT
                )
                if not response.data:
                    last_error = LlmResponseError(
                        _no_payload_detail(response, "data"), fault=LlmFault.NO_CHOICES
                    )
                    logger.warning(
                        "LLM embed returned no data (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        last_error,
                    )
                    if attempt < self.max_retries - 1:
                        await self._backoff(attempt)
                    continue
                embeddings = [list(item.embedding) for item in response.data]

                logger.debug(
                    "Generated %d embedding(s), dim=%d", len(embeddings), len(embeddings[0])
                )
                return embeddings

            except LlmError:
                raise
            except openai.NotFoundError as error:
                summary = _summarize_llm_error(error)
                logger.error("LLM embed failed (model not found, no retry): %s", summary)
                raise LlmNotFoundError(summary) from error
            except openai.APIConnectionError as error:
                last_error = LlmConnectionError(str(error))
                logger.warning(
                    "LLM embed error (attempt %d/%d): %s", attempt + 1, self.max_retries, error
                )
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt)
            except openai.OpenAIError as error:
                summary = _summarize_llm_error(error)
                last_error = LlmResponseError(
                    summary, fault=fault_for_status(getattr(error, "status_code", None))
                )
                logger.warning(
                    "LLM embed error (attempt %d/%d): %s", attempt + 1, self.max_retries, summary
                )
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt)

        logger.error("LLM embed failed after %d attempts: %s", self.max_retries, last_error)
        if last_error is None:
            raise LlmResponseError("LLM embed exhausted retries without a recorded error")
        raise last_error

    # ── Model listing ────────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        """List model ids available at the endpoint (OpenAI-compatible /v1/models).

        Used by the startup preflight to verify an endpoint is reachable and a
        configured model resolves. Translates SDK errors into the ``LlmError``
        hierarchy so callers can distinguish an unreachable endpoint
        (``LlmConnectionError`` / ``LlmTimeoutError``) from a reachable one
        whose model listing failed (``LlmResponseError``).
        """
        try:
            response = await self.client.models.list()
        except openai.APITimeoutError as error:
            raise LlmTimeoutError(str(error)) from error
        except openai.APIConnectionError as error:
            raise LlmConnectionError(str(error)) from error
        except openai.OpenAIError as error:
            raise LlmResponseError(
                _summarize_llm_error(error),
                fault=fault_for_status(getattr(error, "status_code", None)),
            ) from error
        return [model.id for model in response.data]

    async def check_raw_template_faithfulness(self) -> RawFaithfulnessResult:
        """Whether this endpoint serves this model's RAW completions faithfully
        (#2829 build decision 3): one call of the bare BOS token, ``max_tokens``
        1, must report exactly one prompt token — a provider whose
        ``/v1/completions`` secretly wraps the prompt as a chat turn (Makora,
        Cloudflare, Google per #2828/#2829 step 1) reports the wrap's own
        turn-opening tokens on top (14, on one provider probed).

        Carries the SAME provider pin as the real raw-template request
        (:meth:`_provider_pin_extra_body`) — without it, a gateway routes this
        check to any provider in the pool while the real calls stay pinned,
        so a perfectly faithful CoreWeave answers nothing and an unrelated
        chat-wrapping provider answers this check instead.  A transient fault
        (429/5xx/timeout/connection) is retried on the client's OWN
        retry/backoff helpers (the same ones ``_chat_attempts`` uses, not a
        second loop) up to ``max_retries`` attempts: a run measured 71 of
        1,275 samples hitting a single un-retried 429 here, which a naive
        read-as-unfaithful turned into every later call on that sample
        routing to the chat endpoint and 404ing on a model it never serves.

        Never raises — exhausting every attempt still erroring answers
        :attr:`RawFaithfulnessOutcome.UNVERIFIED` rather than propagating,
        because an error here is a MOMENT, not a verdict, and a real call is
        what surfaces a genuinely dead endpoint's own honest error (visible
        degradation, not a guess made on its behalf).  Used by the startup
        preflight (keeps the raw transport on ``UNVERIFIED``, falls back to
        chat only on a real ``UNFAITHFUL`` answer, both named, never silent)
        and the eval suite's pre-run endpoint smoke check (which MAY still
        refuse on ``UNVERIFIED``, since it is the gate protecting a whole run).
        """
        last_error: LlmError | None = None
        for attempt in range(self.max_retries):
            try:
                raw = await self.client.completions.create(
                    model=self.model,
                    prompt=raw_template.RAW_FAITHFULNESS_PROMPT,
                    max_tokens=raw_template.RAW_FAITHFULNESS_MAX_TOKENS,
                    extra_body=self._provider_pin_extra_body(),
                )
            except openai.APITimeoutError as error:
                last_error = await self._retry_after_timeout(attempt, error)
                continue
            except openai.APIConnectionError as error:
                last_error = await self._retry_after_connection_error(attempt, error)
                continue
            except openai.OpenAIError as error:
                last_error = await self._retry_after_openai_error(attempt, error)
                continue
            prompt_tokens = raw.usage.prompt_tokens if raw.usage else None
            outcome = (
                RawFaithfulnessOutcome.FAITHFUL
                if prompt_tokens == raw_template.RAW_FAITHFULNESS_EXPECTED_PROMPT_TOKENS
                else RawFaithfulnessOutcome.UNFAITHFUL
            )
            return RawFaithfulnessResult(outcome=outcome, provider=_provider_of(raw))

        return RawFaithfulnessResult(
            outcome=RawFaithfulnessOutcome.UNVERIFIED,
            detail=str(last_error) if last_error is not None else None,
        )

    async def list_embedding_models(self) -> list[str]:
        """List embedding model ids from provider-specific /v1/embeddings/models.

        Some OpenAI-compatible providers expose embedding-capable models through
        a narrower endpoint even when ``/v1/models`` omits them. This is only
        used as an embedding preflight fallback.
        """
        url = f"{self.api_url}{PennyConstants.LLM_EMBEDDING_MODELS_ENDPOINT}"
        timeout = httpx.Timeout(
            PennyConstants.LLM_MODEL_LIST_TIMEOUT_SECONDS,
            connect=PennyConstants.LLM_CONNECT_TIMEOUT_SECONDS,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
        except httpx.TimeoutException as error:
            raise LlmTimeoutError(str(error)) from error
        # A NETWORK error is the connection failing — refused, or reset mid-request — which is
        # the same moment ``list_models`` reports as a connection error through the SDK.
        except httpx.NetworkError as error:
            raise LlmConnectionError(str(error)) from error
        except httpx.RequestError as error:
            raise LlmResponseError(str(error)) from error
        except httpx.HTTPStatusError as error:
            raise LlmResponseError(
                _summarize_httpx_error(error.response),
                fault=fault_for_status(error.response.status_code),
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise LlmResponseError("embedding model list returned non-JSON response") from error
        return _extract_model_ids(payload)

    # ── Cleanup ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()
        logger.info("LLM client closed")

    # ── Internal: request building ───────────────────────────────────────

    def _build_chat_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        format: dict | str | None,
        token_budget: int | None = None,
        thinking_enabled: bool = True,
    ) -> dict:
        """Build kwargs for the OpenAI chat completions call."""
        # The provider pin rides the same passthrough, added only when one was
        # CONFIGURED, so a client without one sends the body it always sent — and a
        # direct endpoint that has never heard of the field ignores it.  Annotated
        # ``dict[str, Any]`` rather than left inferred: the two bodies below carry
        # different value shapes (a nested dict for ``REASONING_ENABLED_BODY``, a bare
        # string for ``THINKING_OFF_BODY``), which a narrower inferred type would
        # reject on whichever one is assigned second.
        #
        # ``thinking_enabled=False`` does not UPDATE the always-on body with
        # ``THINKING_OFF_BODY`` — it starts from an empty body and sends only
        # ``THINKING_OFF_BODY``.  On local Ollama a ``reasoning`` key's PRESENCE keeps
        # thinking on regardless of its value (#2607), so the always-on ``reasoning:
        # {enabled: True}`` must never ride alongside the off switch — not even as a
        # stale key a naive ``.update()`` would leave in place from a prior merge.
        extra_body: dict[str, Any] = (
            dict(THINKING_OFF_BODY) if not thinking_enabled else dict(REASONING_ENABLED_BODY)
        )
        if self.provider_pin is not None:
            extra_body[PROVIDER_REQUEST_FIELD] = self.provider_pin.as_request_field()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Not an OpenAI-schema field, so it rides in the vendor passthrough the SDK
            # provides. Reasoning comes back where ``_parse_response`` already reads it.
            "extra_body": extra_body,
        }
        if token_budget is not None:
            # Sent under BOTH names (#2474's probes, against local Ollama's
            # gemma4:26b-mlx and OpenRouter's gemma pinned to Cloudflare): local
            # Ollama's OpenAI-compatible endpoint silently ignores
            # ``max_completion_tokens`` — the modern OpenAI field — and only honours
            # the deprecated ``max_tokens``, while OpenRouter honours
            # ``max_completion_tokens`` (confirmed; ``max_tokens`` alone untested
            # there, so both ride together rather than risk it).  One body that cuts
            # the draw on both, with no backend-specific branch here.
            kwargs["max_tokens"] = token_budget
            kwargs["max_completion_tokens"] = token_budget
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        if format is not None:
            kwargs["response_format"] = self._translate_format(format)
        return kwargs

    @staticmethod
    def _translate_messages(messages: list[dict]) -> list[dict]:
        """Translate messages to OpenAI format, handling vision images.

        Also drops the framework-internal per-call success stamp (#1600) from
        tool-result messages before the wire: it's logged (it stays on the
        ``messages_snapshot`` the caller keeps) but is not a wire message field,
        so it never reaches the model — a copy-drop, so the logged dict keeps it.
        """
        translated = []
        for message in messages:
            if "images" in message:
                translated.append(_translate_vision_message(message))
            elif PennyConstants.TOOL_RESULT_SUCCESS_KEY in message:
                translated.append(
                    {
                        key: value
                        for key, value in message.items()
                        if key != PennyConstants.TOOL_RESULT_SUCCESS_KEY
                    }
                )
            else:
                translated.append(message)
        return translated

    @staticmethod
    def _translate_tools(tools: list[dict]) -> list[dict]:
        """Wrap bare function definitions in OpenAI tool format if needed."""
        translated = []
        for tool in tools:
            if "type" in tool:
                translated.append(tool)
            else:
                translated.append({"type": "function", "function": tool})
        return translated

    @staticmethod
    def _translate_format(format: dict | str) -> dict:
        """Translate format param to OpenAI response_format."""
        if format == "json":
            return {"type": "json_object"}
        if isinstance(format, dict):
            return {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": format},
            }
        return {"type": format}

    # ── Internal: response parsing ───────────────────────────────────────

    def _parse_response(self, raw: openai.types.chat.ChatCompletion) -> LlmResponse:
        """Parse an OpenAI ChatCompletion into our LlmResponse model."""
        choice = raw.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [call for tc in message.tool_calls for call in self._parse_tool_calls(tc)]

        # Reasoning fields are non-standard extensions (Ollama uses
        # ``reasoning_content``, newer OpenAI ``reasoning``). They land in
        # the pydantic model_extra dict because the SDK allows extras.
        extras = message.model_extra or {}
        thinking = extras.get("reasoning_content") or extras.get("reasoning")

        return LlmResponse(
            message=LlmMessage(
                role=message.role,
                content=message.content or "",
                tool_calls=tool_calls,
                thinking=thinking,
            ),
            thinking=thinking,
            model=raw.model,
            provider=_provider_of(raw),
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def _parse_tool_calls(
        tool_call: openai.types.chat.ChatCompletionMessageToolCall,
    ) -> list[LlmToolCall]:
        """The calls ONE OpenAI tool call carries — see ``_split_parallel_call``."""
        name = tool_call.function.name
        calls = _split_parallel_call(tool_call.id, tool_call.function.arguments)
        if len(calls) > 1:
            logger.warning(
                "Tool call %s to %s carried %d argument objects — dispatching each as its own call",
                tool_call.id,
                name,
                len(calls),
            )
        return [LlmClient._parse_tool_call(call_id, name, raw) for call_id, raw in calls]

    @staticmethod
    def _parse_tool_call(call_id: str, name: str, raw: str) -> LlmToolCall:
        """Parse a single tool call, deserializing JSON arguments."""
        arguments = {}
        if raw:
            logger.debug("Tool call raw arguments: %.500s", raw)
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed tool call arguments, extracting via regex: %s", raw[:200])
                arguments = LlmClient._extract_malformed_arguments(raw)

        return LlmToolCall(id=call_id, function=LlmToolCallFunction(name=name, arguments=arguments))

    # Regex to extract quoted strings from a queries array (browse tool)
    _QUERY_PATTERN = re.compile(r'"queries"\s*:\s*\[([^\]]*)', re.DOTALL)
    _QUOTED_STRING = re.compile(r'"([^"]+)"')

    # Regex to extract key/content pairs from an entries array (collection_write tool).
    # Handles both proper objects and unescaped-quote stringified objects where the
    # outer JSON itself is malformed (json.loads raises JSONDecodeError).
    _ENTRIES_SECTION_PATTERN = re.compile(r'"entries"\s*:\s*\[', re.DOTALL)
    _ENTRY_KV_PATTERN = re.compile(
        r'"key"\s*:\s*"([^"]*)".*?"content"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    _COLLECTION_FIELD_PATTERN = re.compile(r'"collection"\s*:\s*"([^"]+)"')

    @staticmethod
    def _extract_malformed_arguments(raw: str) -> dict[str, Any]:
        """Best-effort extraction of tool arguments from malformed JSON.

        Handles two known malformation patterns:
        - ``queries`` array (browse tool): items are plain strings.
        - ``entries`` array (collection_write): items are objects whose inner
          quotes are unescaped, making json.loads fail with JSONDecodeError.
          Key/content pairs are recovered via regex.

        Falls back to empty dict if nothing can be extracted.
        """
        # Browse tool: queries array of plain strings
        match = LlmClient._QUERY_PATTERN.search(raw)
        if match:
            items = LlmClient._QUOTED_STRING.findall(match.group(1))
            if items:
                return {"queries": items}

        # collection_write tool: entries array of {key, content} objects
        entries_match = LlmClient._ENTRIES_SECTION_PATTERN.search(raw)
        if entries_match:
            pairs = LlmClient._ENTRY_KV_PATTERN.findall(raw[entries_match.end() :])
            entries = [{"key": k, "content": c} for k, c in pairs]
            if entries:
                result: dict[str, Any] = {"entries": entries}
                collection_match = LlmClient._COLLECTION_FIELD_PATTERN.search(raw)
                if collection_match:
                    result["collection"] = collection_match.group(1)
                return result

        return {}

    # ── Internal: logging ────────────────────────────────────────────────

    @staticmethod
    def _log_response(response: LlmResponse, thinking: str | None) -> None:
        """Log response details at appropriate levels."""
        if response.has_tool_calls:
            logger.info("Received %d tool call(s)", len(response.message.tool_calls or []))
        if thinking:
            logger.debug("Model thinking: %s", thinking[:200])
        logger.debug("Response content: %s", response.content)

    def _publish_call_started(
        self,
        run_id: str,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        agent_name: str | None,
        prompt_type: str | None,
        run_target: str | None,
    ) -> None:
        """Announce this call is about to be made (#2439), before the network
        request — a live Prompts tab's only signal that anything is happening
        until the call's own ``promptlog`` row eventually lands.  Every reroll
        attempt and every validator retry is its own real call here, so this
        fires once per actual request, matching 1:1 with the row ``_log_call``
        will write for it however it ends."""
        if not self.db:
            return
        self.db.messages.publish_call_started(
            run_id=run_id,
            parent_run_id=parent_run_id,
            parent_tool_call_id=parent_tool_call_id,
            agent_name=agent_name,
            prompt_type=prompt_type,
            model=self.model,
            run_target=run_target,
        )

    def _log_call(
        self,
        messages_snapshot: list[dict],
        raw: _RawCompletion | None,
        response: LlmResponse | None,
        tools: list[dict] | None,
        thinking: str | None,
        duration_ms: int,
        agent_name: str | None,
        prompt_type: str | None,
        run_id: str | None,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        transport: PromptTransport,
        outcome: CallOutcome,
        *,
        error: Exception | None = None,
    ) -> int | None:
        """Log this call's ONE ``promptlog`` row, however it ended (#2355 part A).

        Returns the row's id (#2702) — read only by the success path, which
        stamps it on the returned response so a rejection site can mark it.

        ``raw`` is the dumped completion on :attr:`CallOutcome.COMPLETED`, and also
        on a raised raw-template tool-call parse failure (#2833, via
        ``_log_failed_call``'s own ``raw`` — the completion was already fetched
        before parsing it failed), ``None`` on every other non-completed outcome
        — ``log_prompt`` writes the literal ``"{}"`` for a call with genuinely no
        response, which every existing reader already treats exactly as it
        treats a falsy ``response`` (no change needed on their side).  On the
        raw-template transport, a successfully PARSED call's dump is RESHAPED to
        the chat endpoint's own ``choices[].message`` shape (#2829,
        ``_raw_template_log_shape``) — every existing reader of
        ``promptlog.response`` (the self-state header, the data browser, the eval
        harness's own tool-call read) assumes that shape, which the raw
        completions wire shape (``choices[].text``) does not carry; an unparsed
        (``response=None``) dump is left AS ITS RAW SHAPE instead — there is no
        parsed ``LlmResponse`` to reshape it around, and the raw ``choices[].text``
        is exactly the text this is for: keeping what the model actually wrote.

        ``error.fault`` (an ``LlmFault`` — #2526) is persisted alongside the error's own
        class and message: every ``LlmError`` already carries this classification (set
        the moment it raises, from the response's own status), so this is a read, not a
        new computation — it is what lets a reader tell a provider-side refusal
        (429/5xx) from Penny's own malformed or oversized request (4xx) without
        re-parsing the free-text message.
        """
        if not self.db:
            return None
        dumped = self._dumped_for_log(raw, response, transport) if raw is not None else None
        fault = getattr(error, "fault", None)
        timing = usage_timing_from_response(dumped) if dumped is not None else UsageTiming()
        return self.db.messages.log_prompt(
            model=self.model,
            messages=messages_snapshot,
            response=dumped,
            tools=tools,
            thinking=thinking,
            duration_ms=duration_ms,
            agent_name=agent_name,
            prompt_type=prompt_type,
            run_id=run_id,
            run_target=run_target,
            parent_run_id=parent_run_id,
            parent_tool_call_id=parent_tool_call_id,
            transport=transport.value,
            call_outcome=outcome.value,
            call_error_class=type(error).__name__ if error is not None else None,
            call_error_message=str(error) if error is not None else None,
            call_error_fault=fault.value if fault is not None else None,
            prompt_tokens=timing.prompt_tokens,
            cached_tokens=timing.cached_tokens,
            completion_tokens=timing.completion_tokens,
            prefill_ms=timing.prefill_ms,
            generation_ms=timing.generation_ms,
        )

    @staticmethod
    def _dumped_for_log(
        raw: _RawCompletion, response: LlmResponse | None, transport: PromptTransport
    ) -> dict:
        """``raw``'s own dumped dict, shaped for the prompt log by transport."""
        dumped = raw.model_dump()
        if transport is not PromptTransport.RAW_TEMPLATE:
            return _dispatched_response(dumped)
        return _raw_template_log_shape(dumped, response)


def _split_parallel_call(call_id: str, arguments: str | None) -> list[tuple[str, str]]:
    """The calls ONE provider tool call carries, as ``(id, arguments)`` pairs.

    A provider can return parallel calls to one tool as a SINGLE call whose
    arguments are several JSON objects back to back (``{…}{…}``). That string is
    not one JSON value, so read whole it either reached the tool as no arguments
    at all or was salvaged by a regex that paired a key from one object with the
    content of the next — a write the model never made. Each complete leading
    object is its own call; whatever follows the last one is a call too, so a
    malformed tail is salvaged, or refused, on its own text alone.

    One object — or arguments that open with no complete object — is one call,
    returned untouched. The first call keeps the provider's id and each later one
    takes it with its position appended, so every tool result still answers
    exactly one call.
    """
    raw = arguments or ""
    pieces = _argument_objects(raw)
    if len(pieces) < 2:
        return [(call_id, raw)]
    return [
        (call_id if position == 1 else f"{call_id}{_SPLIT_CALL_ID_SEPARATOR}{position}", piece)
        for position, piece in enumerate(pieces, start=1)
    ]


def _argument_objects(raw: str) -> list[str]:
    """Each complete JSON object opening ``raw``, in order, then any text left over."""
    decoder = json.JSONDecoder()
    pieces: list[str] = []
    start = _after_whitespace(raw, 0)
    while start < len(raw):
        try:
            value, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        pieces.append(raw[start:end])
        start = _after_whitespace(raw, end)
    if pieces and start < len(raw):
        pieces.append(raw[start:])
    return pieces


def _after_whitespace(raw: str, position: int) -> int:
    """The index of the first non-whitespace character at or after ``position``."""
    while position < len(raw) and raw[position].isspace():
        position += 1
    return position


def _dispatched_response(response: dict) -> dict:
    """A dumped completion as the prompt log records it: the calls that were dispatched.

    Every reader of a run — its rendered calls, the skill extractor, the eval
    harness — takes a run's tool calls from this record and pairs them with the
    tool results that follow, so a provider call that was dispatched as several
    is recorded as those several, under the ids their results answer. A call
    that was not split is recorded exactly as the provider sent it.
    """
    for choice in response.get(_CHOICES_FIELD) or []:
        message = choice.get(_MESSAGE_FIELD) or {}
        if message.get(_TOOL_CALLS_FIELD):
            message[_TOOL_CALLS_FIELD] = [
                dispatched
                for call in message[_TOOL_CALLS_FIELD]
                for dispatched in _dispatched_calls(call)
            ]
    return response


def _dispatched_calls(call: dict) -> list[dict]:
    """One dumped tool call as the call(s) it was dispatched as."""
    function = call.get(_FUNCTION_FIELD) or {}
    calls = _split_parallel_call(call.get(_ID_FIELD) or "", function.get(_ARGUMENTS_FIELD))
    if len(calls) < 2:
        return [call]
    return [
        {**call, _ID_FIELD: call_id, _FUNCTION_FIELD: {**function, _ARGUMENTS_FIELD: arguments}}
        for call_id, arguments in calls
    ]


def _raw_template_log_shape(dumped: dict, response: LlmResponse | None) -> dict:
    """The raw completion's dump, reshaped to the chat endpoint's own
    ``choices[].message`` shape for the prompt log (#2829).

    The raw wire shape (``choices[].text``) carries no ``message`` at all, so
    every existing reader of ``promptlog.response`` that assumes the chat
    shape — the self-state header, the data browser, the eval harness's own
    tool-call read — would see nothing.  Built from ``response`` (already
    parsed by ``_parse_raw_template_response``, so this never re-parses and
    never fails where that call already succeeded); ``response`` is ``None``
    only when ``dumped`` carries no choices at all, in which case there is
    nothing to reshape.
    """
    choices = dumped.get(_CHOICES_FIELD) or []
    if response is None or not choices:
        return dumped
    message: dict[str, Any] = {"role": "assistant", "content": response.message.content}
    if response.message.tool_calls:
        message[_TOOL_CALLS_FIELD] = [
            {
                _ID_FIELD: call.id,
                "type": "function",
                _FUNCTION_FIELD: {
                    "name": call.function.name,
                    _ARGUMENTS_FIELD: json.dumps(call.function.arguments),
                },
            }
            for call in response.message.tool_calls
        ]
    if response.thinking:
        message["reasoning_content"] = response.thinking
    return {**dumped, _CHOICES_FIELD: [{**choices[0], _MESSAGE_FIELD: message}]}


def _reconstruct_completion(
    last_chunk: openai.types.completion.Completion,
    last_choice: openai.types.completion_choice.CompletionChoice | None,
    full_text: str,
) -> openai.types.completion.Completion:
    """The one faithful ``Completion`` a streamed raw-template call produces
    (#2842) — every delta chunk's own text concatenated into ``full_text``,
    everything else (id/model/usage/``finish_reason``) read off the LAST
    chunk that carried it. ``last_choice`` is tracked separately from
    ``last_chunk`` because a provider's own trailing usage-only chunk (sent
    only when ``stream_options.include_usage`` is set) can carry an EMPTY
    ``choices`` alongside the real usage, with the ``finish_reason``'s own
    owning chunk one step earlier.
    """
    if last_choice is None:
        return last_chunk
    choice = last_choice.model_copy(update={"text": full_text})
    return last_chunk.model_copy(update={"choices": [choice]})


def _thinking_loop_cut_dump(kept_thinking: str) -> dict:
    """The discarded cut draw's own ``promptlog.response`` dump (#2842) —
    already in the chat-shape ``_raw_template_log_shape`` reshapes a real
    completion INTO, built directly since there is no raw completion object
    for a stream closed by Penny itself mid-generation (no ``choices[].text``
    worth keeping — the text that mattered is ``kept_thinking``, already the
    row's own ``thinking`` column)."""
    return {
        _CHOICES_FIELD: [
            {
                "index": 0,
                "finish_reason": None,
                _MESSAGE_FIELD: {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": kept_thinking,
                },
            }
        ]
    }


# Base64 prefixes of each supported image format's magic bytes — the first bytes
# of the file, read off the base64 string itself (base64 maps 3 bytes to 4 chars, so
# a 3-4 byte magic shows up as a fixed prefix). An image that carries none of them
# is assumed to be a JPEG, the most common capture format.
_B64_IMAGE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/9j/", "image/jpeg"),
    ("iVBOR", "image/png"),
    ("R0lGOD", "image/gif"),
    ("UklGR", "image/webp"),
)
_DEFAULT_IMAGE_MIME = "image/jpeg"


def _image_data_uri(image: str) -> str:
    """Normalize one image to an OpenAI ``image_url`` data URI.

    A ``data:`` URI (what the channels hand over, carrying the attachment's own
    mime type) passes through as-is; bare base64 gets its mime type detected from
    the image's magic bytes.
    """
    if image.startswith("data:"):
        return image
    mime = next(
        (detected for prefix, detected in _B64_IMAGE_PREFIXES if image.startswith(prefix)),
        _DEFAULT_IMAGE_MIME,
    )
    return f"data:{mime};base64,{image}"


def _translate_vision_message(message: dict) -> dict:
    """Translate a message carrying images to OpenAI content-parts format.

    Each image becomes one ``image_url`` part whose URL is a data URI with the
    image's real mime type (#2635) — never a hardcoded label.
    """
    content_parts: list[dict] = [{"type": "text", "text": message.get("content", "")}]
    for image_b64 in message.get("images", []):
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(image_b64)},
            }
        )
    return {"role": message["role"], "content": content_parts}
