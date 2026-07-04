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
from typing import Any

import httpx
import openai

from penny.constants import PennyConstants
from penny.llm.models import (
    LlmConnectionError,
    LlmError,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
)

logger = logging.getLogger(__name__)

# Default API key for local inference servers that require one but don't check it
_DEFAULT_API_KEY = "not-needed"

# Fallback when an error response carries no content-type header.
_UNKNOWN_CONTENT_TYPE = "unknown"


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
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.db = db
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

    # ── Chat ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
        run_target: str | None = None,
    ) -> LlmResponse:
        """Generate a chat completion with optional tool calling."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Sending chat request (attempt %d/%d)", attempt + 1, self.max_retries)

                start = time.time()
                messages_snapshot = list(messages)
                translated_messages = self._translate_messages(messages)

                kwargs = self._build_chat_kwargs(translated_messages, tools, format)
                raw = await self.client.chat.completions.create(**kwargs)
                duration_ms = int((time.time() - start) * 1000)

                response = self._parse_response(raw)
                thinking = response.thinking or response.message.thinking
                self._log_response(response, thinking)
                self._log_to_database(
                    messages_snapshot,
                    raw,
                    tools,
                    thinking,
                    duration_ms,
                    agent_name,
                    prompt_type,
                    run_id,
                    run_target,
                )

                return response

            except LlmError:
                raise
            except openai.NotFoundError as error:
                summary = _summarize_llm_error(error)
                logger.error("LLM chat failed (model not found, no retry): %s", summary)
                raise LlmNotFoundError(summary) from error
            except openai.APITimeoutError as error:
                last_error = LlmTimeoutError(str(error))
                logger.warning(
                    "LLM request timed out (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except openai.APIConnectionError as error:
                last_error = LlmConnectionError(str(error))
                logger.warning(
                    "LLM chat error (attempt %d/%d): %s", attempt + 1, self.max_retries, error
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except openai.OpenAIError as error:
                summary = _summarize_llm_error(error)
                # 500 "error parsing tool call" means the model produced plain text
                # instead of a JSON tool call. Retrying with the same messages won't
                # help — raise immediately so the agent can inject a format nudge.
                if getattr(error, "status_code", None) == 500 and "error parsing tool call" in str(
                    error
                ):
                    logger.warning(
                        "Tool parse error — model returned plain text instead of JSON tool call"
                    )
                    raise LlmToolParseError(summary) from error
                last_error = LlmResponseError(summary)
                logger.warning(
                    "LLM chat error (attempt %d/%d): %s", attempt + 1, self.max_retries, summary
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

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
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
    ) -> LlmResponse:
        """Generate a completion for a prompt (converts to chat format internally)."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(
            messages,
            tools,
            format,
            agent_name=agent_name,
            prompt_type=prompt_type,
            run_id=run_id,
        )

    # ── Embeddings ───────────────────────────────────────────────────────

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Generate embeddings for one or more texts."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Sending embed request (attempt %d/%d)", attempt + 1, self.max_retries)

                response = await self.client.embeddings.create(model=self.model, input=text)
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
                    await asyncio.sleep(self.retry_delay)
            except openai.OpenAIError as error:
                summary = _summarize_llm_error(error)
                last_error = LlmResponseError(summary)
                logger.warning(
                    "LLM embed error (attempt %d/%d): %s", attempt + 1, self.max_retries, summary
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("LLM embed failed after %d attempts: %s", self.max_retries, last_error)
        if last_error is None:
            raise LlmResponseError("LLM embed exhausted retries without a recorded error")
        raise last_error

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
    ) -> dict:
        """Build kwargs for the OpenAI chat completions call."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        if format is not None:
            kwargs["response_format"] = self._translate_format(format)
        return kwargs

    @staticmethod
    def _translate_messages(messages: list[dict]) -> list[dict]:
        """Translate messages to OpenAI format, handling vision images."""
        translated = []
        for message in messages:
            if "images" in message:
                translated.append(_translate_vision_message(message))
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
            tool_calls = [self._parse_tool_call(tc) for tc in message.tool_calls]

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
        )

    @staticmethod
    def _parse_tool_call(tool_call: openai.types.chat.ChatCompletionMessageToolCall) -> LlmToolCall:
        """Parse a single OpenAI tool call, deserializing JSON arguments."""
        arguments = {}
        if tool_call.function.arguments:
            logger.debug("Tool call raw arguments: %.500s", tool_call.function.arguments)
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed tool call arguments, extracting via regex: %s",
                    tool_call.function.arguments[:200],
                )
                arguments = LlmClient._extract_malformed_arguments(tool_call.function.arguments)

        return LlmToolCall(
            id=tool_call.id,
            function=LlmToolCallFunction(
                name=tool_call.function.name,
                arguments=arguments,
            ),
        )

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
    _MEMORY_FIELD_PATTERN = re.compile(r'"memory"\s*:\s*"([^"]+)"')

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
                memory_match = LlmClient._MEMORY_FIELD_PATTERN.search(raw)
                if memory_match:
                    result["memory"] = memory_match.group(1)
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

    def _log_to_database(
        self,
        messages_snapshot: list[dict],
        raw: openai.types.chat.ChatCompletion,
        tools: list[dict] | None,
        thinking: str | None,
        duration_ms: int,
        agent_name: str | None,
        prompt_type: str | None,
        run_id: str | None,
        run_target: str | None,
    ) -> None:
        """Log prompt exchange to database if available."""
        if not self.db:
            return
        self.db.messages.log_prompt(
            model=self.model,
            messages=messages_snapshot,
            response=raw.model_dump(),
            tools=tools,
            thinking=thinking,
            duration_ms=duration_ms,
            agent_name=agent_name,
            prompt_type=prompt_type,
            run_id=run_id,
            run_target=run_target,
        )


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
