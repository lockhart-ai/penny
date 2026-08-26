"""Pydantic models for LLM client responses.

These are our own types, decoupled from any SDK. The LlmClient
translates provider-specific responses into these models.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, ClassVar

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


# ── Fault classes ────────────────────────────────────────────────────────

# The structured fields every chat-attempt log record carries.  A reader that wants to
# know what a run spent its calls on READS these off the record rather than re-parsing
# the sentence they were formatted into: the sentence is for a human, and a tally built
# by matching it would break the first time the wording improved.
FAULT_LOG_FIELD = "llm_fault"
PROVIDER_LOG_FIELD = "llm_provider"

# What an endpoint that names no upstream provider is called in a tally.  Local Ollama
# reports none and there is nothing wrong with that, so the tally says "unreported"
# rather than inventing a name or dropping the calls.
UNREPORTED_PROVIDER = "unreported"

# The response field an OpenAI-compatible GATEWAY uses to name the upstream that actually
# served the completion (OpenRouter does; a direct endpoint does not).  It is the only
# thing that tells one member of a routing pool from another, which is the difference
# between "this model is broken" and "one provider in the pool is".
PROVIDER_RESPONSE_FIELD = "provider"


class LlmFault(StrEnum):
    """What a failed model call failed OF — the class, not the sentence.

    Every failure already produced a message; none of them produced a value anything
    could count.  So a run that died 188 times of one cause and a run that died once of
    188 causes read identically, and telling them apart meant grepping logs by hand.
    The class is carried on the error and stamped on the attempt's log record, so the
    tally is a READ.

    ``transient`` is the second customer: it says whether another draw could plausibly
    succeed, which is what separates a provider having a bad minute (retry) from a model
    the endpoint will never serve (refuse now, and say why).
    """

    NO_CHOICES = "no choices"
    RATE_LIMITED = "429"
    SERVER_ERROR = "5xx"
    CLIENT_ERROR = "4xx"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    NOT_FOUND = "model not found"
    TOOL_PARSE = "tool parse"
    OTHER = "other"

    @property
    def transient(self) -> bool:
        """Could another draw get past this one?

        A 200 with no completion, a rate limit, a 5xx, a stall and a dropped connection
        are all moments rather than verdicts.  A 404, a rejected request, a model that
        answered in prose where JSON was required, and anything unrecognised are not —
        an unrecognised fault stays non-transient on purpose, so a gate whose whole job
        is to be decisive never becomes flaky over a class nobody has looked at yet.
        """
        return self in _TRANSIENT_FAULTS


_TRANSIENT_FAULTS = frozenset(
    {
        LlmFault.NO_CHOICES,
        LlmFault.RATE_LIMITED,
        LlmFault.SERVER_ERROR,
        LlmFault.TIMEOUT,
        LlmFault.CONNECTION,
    }
)

_RATE_LIMITED_STATUS = 429
_SERVER_ERROR_STATUS = 500
_CLIENT_ERROR_STATUS = 400


def fault_for_status(status: int | None) -> LlmFault:
    """The fault class an HTTP status names, or ``OTHER`` when it names none."""
    if status is None:
        return LlmFault.OTHER
    if status == _RATE_LIMITED_STATUS:
        return LlmFault.RATE_LIMITED
    if status >= _SERVER_ERROR_STATUS:
        return LlmFault.SERVER_ERROR
    if status >= _CLIENT_ERROR_STATUS:
        return LlmFault.CLIENT_ERROR
    return LlmFault.OTHER


# The request-body field an OpenAI-compatible GATEWAY reads its routing preference from.
# It travels as ``extra_body`` because it is not part of the OpenAI schema — a direct
# endpoint that has never heard of it ignores it.
PROVIDER_REQUEST_FIELD = "provider"


class ProviderPreference(BaseModel):
    """Which upstream a routing gateway should PREFER — and whether it may use another.

    Pinning is a PREFERENCE by default, never a wall.  Pinning hard (``allow_fallbacks``
    off) put 325 rate limits on ONE endpoint at a concurrency the same run handled with
    zero unpinned: forbidding every other upstream concentrates the whole run's load onto
    one of them, so hard pinning and concurrency are coupled and a run that pins hard has
    to lower its in-flight budget to match.  Preferring instead keeps the throughput the
    pool provides, and reproducibility becomes something the run OBSERVES rather than
    enforces — the answering provider is recorded per call, so a fallback shows up in the
    artifacts as the fact it is.
    """

    order: list[str]
    allow_fallbacks: bool = True

    @classmethod
    def prefer(cls, provider: str | None) -> ProviderPreference | None:
        """A preference for one named upstream, or ``None`` when none was configured."""
        return cls(order=[provider]) if provider else None

    def as_request_field(self) -> dict[str, Any]:
        """The gateway's own routing object, for the request body."""
        return {"order": list(self.order), "allow_fallbacks": self.allow_fallbacks}


# ── Error types ──────────────────────────────────────────────────────────


class LlmError(Exception):
    """Base error for LLM client operations.

    Carries its :class:`LlmFault` so a caller decides what to do by reading a value
    rather than re-parsing the message the client built for a human.  Each subclass
    names the fault it always is; the one shape that varies — a server error response,
    which is a 429 or a 5xx or a 4xx depending on the status — takes its fault explicitly
    from the client that read the status.
    """

    default_fault: ClassVar[LlmFault] = LlmFault.OTHER

    def __init__(self, *args: Any, fault: LlmFault | None = None) -> None:
        super().__init__(*args)
        self.fault = fault or self.default_fault


class LlmNotFoundError(LlmError):
    """Model not found (404). Should not be retried."""

    default_fault: ClassVar[LlmFault] = LlmFault.NOT_FOUND


class LlmConnectionError(LlmError):
    """Could not connect to the LLM server."""

    default_fault: ClassVar[LlmFault] = LlmFault.CONNECTION


class LlmTimeoutError(LlmConnectionError):
    """LLM request timed out. Transient — model may be slow or temporarily busy."""

    default_fault: ClassVar[LlmFault] = LlmFault.TIMEOUT


class LlmResponseError(LlmError):
    """Server returned an error response."""


class LlmToolParseError(LlmError):
    """Server could not parse the model's tool call output (plain text instead of JSON).

    This is a model formatting failure, not a transient server error.
    Retrying with the same messages won't help — the agent must re-prompt with a
    format reminder so the model knows to return only a valid JSON tool call.
    """

    default_fault: ClassVar[LlmFault] = LlmFault.TOOL_PARSE


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
    # Which upstream actually served this completion, when the endpoint is a gateway that
    # says so.  ``None`` from a direct endpoint, which reports no such thing.  It is here
    # beside ``model`` because the pair is the answer to "what answered me": a routing
    # pool serves one model from several providers, and only one of them need be broken
    # for a run to die while the model itself is fine.
    provider: str | None = None

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return bool(self.message.tool_calls)
