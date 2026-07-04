"""Pydantic models for LLM client responses.

These are our own types, decoupled from any SDK. The LlmClient
translates provider-specific responses into these models.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Tool-name normalization ──────────────────────────────────────────────

# gpt-oss emits tool calls in the Harmony format, whose names are wrapped with
# control tokens like ``<|channel|>commentary``.  Local Ollama strips these
# before returning the tool call, but some remote OpenAI-compatible backends
# (e.g. OpenRouter serving gpt-oss) leak them, so the raw name arrives as
# e.g. ``done<|channel|>commentary`` or ``collection_read_latest<|channel|>``.
# The real tool name is always the leading identifier before the first control
# token, so everything from the first ``<|`` marker onward is stripped.  A
# legitimate tool name is a plain identifier and never contains ``<|``.
_HARMONY_CONTROL_TOKEN = re.compile(r"<\|.*", re.DOTALL)


def strip_harmony_control_tokens(name: str) -> str:
    """Strip leaked Harmony control tokens from a tool-call name.

    Defensive normalization so tool dispatch is robust to any backend that
    doesn't fully parse the Harmony format.  Applied where the tool name is
    read off the model response (``LlmToolCallFunction.name``), so every
    downstream consumer — registry lookup, done-detection, dedup, result
    framing — sees the clean identifier.
    """
    return _HARMONY_CONTROL_TOKEN.sub("", name).strip()


# ── Error types ──────────────────────────────────────────────────────────


class LlmError(Exception):
    """Base error for LLM client operations."""


class LlmNotFoundError(LlmError):
    """Model not found (404). Should not be retried."""


class LlmConnectionError(LlmError):
    """Could not connect to the LLM server."""


class LlmTimeoutError(LlmConnectionError):
    """LLM request timed out. Transient — model may be slow or temporarily busy."""


class LlmResponseError(LlmError):
    """Server returned an error response."""


class LlmToolParseError(LlmError):
    """Server could not parse the model's tool call output (plain text instead of JSON).

    This is a model formatting failure, not a transient server error.
    Retrying with the same messages won't help — the agent must re-prompt with a
    format reminder so the model knows to return only a valid JSON tool call.
    """


# ── Response types ───────────────────────────────────────────────────────


class LlmToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip_harmony_control_tokens(cls, value: str) -> str:
        """Normalize the tool name at the read-off boundary — see
        ``strip_harmony_control_tokens``.  This is the single point where a raw
        model-response tool name enters our models, so cleaning here keeps
        dispatch, done-detection, dedup, and result framing all consistent."""
        return strip_harmony_control_tokens(value)


class LlmToolCall(BaseModel):
    """A tool call from the model response."""

    id: str
    function: LlmToolCallFunction


class LlmMessage(BaseModel):
    """Message object from a chat response."""

    role: str
    content: str = ""
    tool_calls: list[LlmToolCall] | None = None
    thinking: str | None = None

    def to_input_message(self) -> dict[str, Any]:
        """Convert to input message format for the next request (excludes thinking)."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments),
                    },
                }
                for tool_call in self.tool_calls
            ]
        return message


class LlmResponse(BaseModel):
    """Response from an LLM chat call."""

    message: LlmMessage
    thinking: str | None = None
    model: str | None = None

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return bool(self.message.tool_calls)
