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
from typing import TYPE_CHECKING, Any

import httpx
import openai

from penny.constants import PennyConstants
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
    ProviderPin,
    fault_for_status,
)

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
# Read-only: sent as-is, never mutated. Backends that do not know the field ignore it
# (verified against Ollama's OpenAI-compatible endpoint), so the local path is unchanged
# and the remote path stops silently differing from it.
REASONING_ENABLED_BODY = {"reasoning": {"enabled": True}}

# The per-call override that turns reasoning OFF for one draw (#2474's probes,
# against local Ollama's gemma4:26b-mlx and OpenRouter's gemma pinned to Cloudflare).
# Two fields, because no single one disabled it on BOTH: on Ollama, ``reasoning:
# {enabled: false}`` alone was silently ignored (the model still reasoned in full);
# ``reasoning_effort: "none"`` alone worked on Ollama AND on OpenRouter, where
# ``reasoning: {enabled: false}`` alone also worked. Sending both is what makes one
# body portable without asking the caller which backend is live; no production
# caller passes this yet — see ``LlmClient.chat``'s ``thinking_enabled`` parameter.
THINKING_OFF_BODY = {"reasoning": {"enabled": False}, "reasoning_effort": "none"}

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

        logger.info("Initialized LLM client: url=%s, model=%s", api_url, model)

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

    # ── Chat ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
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
        """
        resolved_run_id = call_link.run_id
        parent_run_id = call_link.parent_run_id
        parent_tool_call_id = call_link.parent_tool_call_id
        self._publish_call_started(
            resolved_run_id, parent_run_id, parent_tool_call_id, agent_name, prompt_type, run_target
        )
        start = time.time()
        messages_snapshot = list(messages)
        try:
            response, raw = await self._chat_attempts(
                messages, tools, format, token_budget, thinking_enabled
            )
        except asyncio.CancelledError:
            self._log_call(
                messages_snapshot,
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
                CallOutcome.CANCELLED,
            )
            raise
        except LlmError as error:
            outcome = (
                CallOutcome.TIMED_OUT if isinstance(error, LlmTimeoutError) else CallOutcome.RAISED
            )
            self._log_call(
                messages_snapshot,
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
                outcome,
                error=error,
            )
            raise

        thinking = response.thinking or response.message.thinking
        self._log_response(response, thinking)
        self._log_call(
            messages_snapshot,
            raw,
            tools,
            thinking,
            self._elapsed_ms(start),
            agent_name,
            prompt_type,
            resolved_run_id,
            run_target,
            parent_run_id,
            parent_tool_call_id,
            CallOutcome.COMPLETED,
        )
        return response

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        """Milliseconds since ``start`` (``time.time()`` at the top of ``chat()``)."""
        return int((time.time() - start) * 1000)

    async def _chat_attempts(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        format: dict | str | None,
        token_budget: int | None = None,
        thinking_enabled: bool = True,
    ) -> tuple[LlmResponse, openai.types.chat.ChatCompletion]:
        """The retry loop over individual network attempts.

        Unchanged in substance from before #2355 — each attempt's own outcome
        still goes through ``_log_attempt`` (the human-readable DEBUG/WARNING line,
        one per attempt) — except it no longer writes the call's ``promptlog`` row
        itself: ``chat()`` does that once, for the WHOLE call, from whichever way
        this returns or raises.  Returns the parsed response alongside the raw
        completion the caller persists; raises an ``LlmError`` subclass (or lets a
        ``CancelledError`` through untouched) when no attempt produced one.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Sending chat request (attempt %d/%d)", attempt + 1, self.max_retries)

                translated_messages = self._translate_messages(messages)
                kwargs = self._build_chat_kwargs(
                    translated_messages, tools, format, token_budget, thinking_enabled
                )
                raw = await self.client.chat.completions.create(**kwargs)

                # Checked BEFORE parsing, and treated as a retryable fault rather than
                # raised: an LlmError raised from here would re-raise untried (see the
                # first except clause), and a provider answering 200-with-no-completion
                # is exactly the transient another attempt usually gets past.
                if not raw.choices:
                    last_error = LlmResponseError(
                        _no_payload_detail(raw, "choices"), fault=LlmFault.NO_CHOICES
                    )
                    self._log_attempt(
                        attempt,
                        fault=LlmFault.NO_CHOICES,
                        provider=_provider_of(raw),
                        detail=str(last_error),
                    )
                    if attempt < self.max_retries - 1:
                        await self._backoff(attempt)
                    continue

                response = self._parse_response(raw)
                self._log_attempt(attempt, fault=None, provider=response.provider, detail="")
                return response, raw

            except LlmError:
                raise
            except openai.NotFoundError as error:
                summary = _summarize_llm_error(error)
                self._log_attempt(attempt, fault=LlmFault.NOT_FOUND, provider=None, detail=summary)
                logger.error("LLM chat failed (model not found, no retry): %s", summary)
                raise LlmNotFoundError(summary) from error
            except openai.APITimeoutError as error:
                last_error = LlmTimeoutError(str(error))
                self._log_attempt(attempt, fault=LlmFault.TIMEOUT, provider=None, detail=str(error))
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt)
            except openai.APIConnectionError as error:
                last_error = LlmConnectionError(str(error))
                self._log_attempt(
                    attempt, fault=LlmFault.CONNECTION, provider=None, detail=str(error)
                )
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt)
            except openai.OpenAIError as error:
                summary = _summarize_llm_error(error)
                # 500 "error parsing tool call" means the model produced plain text
                # instead of a JSON tool call. Retrying with the same messages won't
                # help — raise immediately so the agent can inject a format nudge.
                if getattr(error, "status_code", None) == 500 and "error parsing tool call" in str(
                    error
                ):
                    self._log_attempt(
                        attempt, fault=LlmFault.TOOL_PARSE, provider=None, detail=summary
                    )
                    logger.warning(
                        "Tool parse error — model returned plain text instead of JSON tool call"
                    )
                    raise LlmToolParseError(summary) from error
                fault = fault_for_status(getattr(error, "status_code", None))
                last_error = LlmResponseError(summary, fault=fault)
                self._log_attempt(attempt, fault=fault, provider=None, detail=summary)
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt)

        logger.error("LLM chat failed after %d attempts: %s", self.max_retries, last_error)
        if last_error is None:
            raise LlmResponseError("LLM chat exhausted retries without a recorded error")
        raise last_error

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
        messages = [{"role": "user", "content": prompt}]
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
        # Both vendor fields ride the same passthrough.  The provider pin is added only
        # when one was CONFIGURED, so a client without one sends the body it always sent —
        # and a direct endpoint that has never heard of the field ignores it.  Annotated
        # ``dict[str, Any]`` rather than left inferred: ``REASONING_ENABLED_BODY``'s own
        # narrow shape (one key, one bool-valued nested dict) would otherwise make ty
        # reject ``.update(THINKING_OFF_BODY)`` below, whose two keys carry different
        # value shapes (a nested dict AND a bare string).
        extra_body: dict[str, Any] = dict(REASONING_ENABLED_BODY)
        if self.provider_pin is not None:
            extra_body[PROVIDER_REQUEST_FIELD] = self.provider_pin.as_request_field()
        if not thinking_enabled:
            extra_body.update(THINKING_OFF_BODY)
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
        raw: openai.types.chat.ChatCompletion | None,
        tools: list[dict] | None,
        thinking: str | None,
        duration_ms: int,
        agent_name: str | None,
        prompt_type: str | None,
        run_id: str | None,
        run_target: str | None,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        outcome: CallOutcome,
        *,
        error: Exception | None = None,
    ) -> None:
        """Log this call's ONE ``promptlog`` row, however it ended (#2355 part A).

        ``raw`` is the dumped completion on :attr:`CallOutcome.COMPLETED`, ``None``
        on every other outcome — ``log_prompt`` writes the literal ``"{}"`` for a
        call with no response, which every existing reader already treats exactly
        as it treats a falsy ``response`` (no change needed on their side).
        """
        if not self.db:
            return
        response = _dispatched_response(raw.model_dump()) if raw is not None else None
        self.db.messages.log_prompt(
            model=self.model,
            messages=messages_snapshot,
            response=response,
            tools=tools,
            thinking=thinking,
            duration_ms=duration_ms,
            agent_name=agent_name,
            prompt_type=prompt_type,
            run_id=run_id,
            run_target=run_target,
            parent_run_id=parent_run_id,
            parent_tool_call_id=parent_tool_call_id,
            call_outcome=outcome.value,
            call_error_class=type(error).__name__ if error is not None else None,
            call_error_message=str(error) if error is not None else None,
        )


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


def _translate_vision_message(message: dict) -> dict:
    """Translate Ollama-style vision message to OpenAI content-parts format."""
    content_parts: list[dict] = [{"type": "text", "text": message.get("content", "")}]
    for image_b64 in message.get("images", []):
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
    return {"role": message["role"], "content": content_parts}
