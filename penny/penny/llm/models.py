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

from penny.constants import PennyConstants

# ── Tool-name normalization ──────────────────────────────────────────────

# gpt-oss emits tool calls in the Harmony format, whose names are wrapped with
# control tokens like ``<|channel|>commentary``.  Local Ollama strips these
# before returning the tool call, but some remote OpenAI-compatible backends
# (e.g. OpenRouter serving gpt-oss) leak them, so the raw name arrives as
# e.g. ``done<|channel|>commentary`` or ``collection_list<|channel|>``.
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


class CallOutcome(StrEnum):
    """How ONE call to :meth:`LlmClient.chat` ended (#2355 part A) — stored on the
    one ``promptlog`` row every call now writes, however it ends, rather than only
    on success.

    ``COMPLETED`` — a usable response came back (today's only logged case).
    ``CANCELLED`` — an ``asyncio.CancelledError`` reached the call from outside (a
    tool's own timeout, a foreground preemption, shutdown) and was re-raised
    untouched; the client records only THAT it was cancelled — why is the
    canceller's to record, in its own result.
    ``TIMED_OUT`` — every retry exhausted on ``LlmTimeoutError`` (the endpoint's own
    request timing out), not an external cancellation.
    ``RAISED`` — any other exception escaped (connection refused, a non-retryable
    response, model not found, an unparseable tool call) — ``call_error_class`` /
    ``call_error_message`` carry the exception's own class and message.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RAISED = "raised"


class PromptTransport(StrEnum):
    """Which wire shape carried ONE call to :meth:`LlmClient.chat` (#2829) —
    stored on its ``promptlog`` row alongside ``call_outcome``, so a reader
    can tell which runner behaviour it got without re-deriving it from the
    model id.

    ``CHAT`` — the OpenAI chat-completions API; each provider applies its
    own copy of the model's chat template.
    ``RAW_TEMPLATE`` — Penny renders the model's own vendored chat template
    itself and sends the result as a raw completions prompt, so every
    runner serving the model sees byte-identical input.  Chosen structurally
    by model identity (:func:`penny.llm.raw_template.has_vendored_template`)
    and forced back to ``CHAT`` for a call carrying an image or a structured
    ``format`` (neither has a raw-completions equivalent), or when the
    raw-faithfulness check (the startup preflight) found this deployment's
    endpoint does not actually serve the model's raw completions faithfully.
    """

    CHAT = "chat"
    RAW_TEMPLATE = "raw_template"


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


# The request-body field an OpenAI-compatible GATEWAY reads its routing preferences from.
# It travels as ``extra_body`` because it is not part of the OpenAI schema — a direct
# endpoint that has never heard of it ignores it.
PROVIDER_REQUEST_FIELD = "provider"


class ProviderPin(BaseModel):
    """The gateway's routing object for a configured provider: a PIN to that one upstream.

    Every request goes to exactly the named provider, with ``allow_fallbacks`` off, so
    nothing is ever routed somewhere nobody chose.  Upstreams serving the same model are
    not interchangeable: one gateway adapter was measured replacing every plain-text tool
    result with ``{}``, so a run routed there measured a model that never saw its own tool
    results.  A pin makes the upstream a fact of the configuration rather than of the
    moment.  The cost is that a pinned run gets only what that one provider sustains, so
    concurrency is sized to it.  There is no unpinned mode: fallbacks are not a field
    here, because nothing is supposed to turn them on.
    """

    provider: str

    @classmethod
    def pin(cls, provider: str | None) -> ProviderPin | None:
        """A pin to one named upstream, or ``None`` when none was configured."""
        return cls(provider=provider) if provider else None

    def as_request_field(self) -> dict[str, Any]:
        """The gateway's own routing object, for the request body."""
        return {"order": [self.provider], "allow_fallbacks": False}


class RawFaithfulnessResult(BaseModel):
    """Whether a raw-completions endpoint served THIS model's raw template
    transport faithfully (#2829 build decision 3), and which upstream
    actually answered — read off the response itself (``_provider_of``),
    not assumed to be the configured pin, so a refusal or a degraded-mode
    message names whoever really answered.  A future mis-route (the pin not
    reaching the request, a gateway routing around it) then names itself
    instead of reporting the pin back at the caller unchanged."""

    faithful: bool
    provider: str | None = None


# ── Usage & timing ───────────────────────────────────────────────────────

# A duration field present only on a response whose ``usage`` carries oMLX's own
# shape (seconds) — its presence, not any passed-in runner identity, is what tells
# an oMLX response apart from Ollama's native one below, which names its own
# generation duration differently.
_OMLX_GENERATION_FIELD = "generation_duration"
_OMLX_PREFILL_FIELD = "prompt_eval_duration"

# Ollama's native API (``/api/chat``, not the OpenAI-compatible endpoint Penny's
# chat path actually calls today) reports the same two phases in nanoseconds,
# under its own field names — ``eval_duration`` only ever appears on that shape,
# so its presence is what tells the two apart.
_OLLAMA_GENERATION_FIELD = "eval_duration"
_OLLAMA_PREFILL_FIELD = "prompt_eval_duration"

_NANOSECONDS_PER_MS = 1_000_000
_MS_PER_SECOND = 1_000


class UsageTiming(BaseModel):
    """Token counts and per-phase timing read off one call's own ``usage`` object
    (#2793) — every field ``None`` where the serving runner didn't report it,
    never a guessed zero, so a reader can tell "ran instantly" from "not measured".

    ``prompt_tokens``/``completion_tokens``/``cached_tokens`` share one field shape
    across every runner this reads (OpenAI's own ``usage.prompt_tokens`` /
    ``usage.completion_tokens`` / ``usage.prompt_tokens_details.cached_tokens``).
    ``prefill_ms``/``generation_ms`` do not: each runner names and units its own
    phase split differently, so :func:`usage_timing_from_response` tells them apart
    structurally, by which duration field is actually present, never by a runner
    name passed in from outside.
    """

    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    prefill_ms: int | None = None
    generation_ms: int | None = None


def usage_timing_from_response(response: dict) -> UsageTiming:
    """:class:`UsageTiming` for one dumped completion's own ``usage`` object.

    Called from the one chokepoint that logs a ``promptlog`` row (``LlmClient``'s
    ``_log_call``) and, identically, from migration 0129's backfill over every
    existing row's stored ``response`` — one mapping, never two.
    """
    usage = response.get("usage") or {}
    prefill_ms, generation_ms = _duration_ms_pair(usage)
    return UsageTiming(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        prefill_ms=prefill_ms,
        generation_ms=generation_ms,
    )


def _duration_ms_pair(usage: dict) -> tuple[int | None, int | None]:
    """``(prefill_ms, generation_ms)`` for whichever runner shape ``usage`` carries.

    oMLX's ``generation_duration`` (seconds) and Ollama native's ``eval_duration``
    (nanoseconds) never both appear on the same response, so checking for one
    after the other — rather than requiring the caller to say which runner this
    is — is exact, not a guess. OpenRouter (and anything else reporting neither)
    returns ``(None, None)``: no prefill/generation split, not a zero one.
    """
    if _OMLX_GENERATION_FIELD in usage:
        return (
            _seconds_to_ms(usage.get(_OMLX_PREFILL_FIELD)),
            _seconds_to_ms(usage.get(_OMLX_GENERATION_FIELD)),
        )
    if _OLLAMA_GENERATION_FIELD in usage:
        return (
            _nanoseconds_to_ms(usage.get(_OLLAMA_PREFILL_FIELD)),
            _nanoseconds_to_ms(usage.get(_OLLAMA_GENERATION_FIELD)),
        )
    return None, None


def _seconds_to_ms(seconds: float | None) -> int | None:
    return round(seconds * _MS_PER_SECOND) if seconds is not None else None


def _nanoseconds_to_ms(nanoseconds: float | None) -> int | None:
    return round(nanoseconds / _NANOSECONDS_PER_MS) if nanoseconds is not None else None


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
    """One turn of conversation history (#2563).

    Carries every field any role in use today needs — a plain system/user turn
    only ever sets ``role``/``content``; an assistant turn adds ``tool_calls``
    and/or ``thinking``; a tool-result turn adds ``tool_call_id`` and
    ``tool_success``; a user turn with an attachment adds ``images``.  One model, not a family
    per role: the shapes differ only in which optional fields are populated, so
    a role union would fragment every call site into a ``match`` for no gain.
    """

    role: str
    content: str = ""
    tool_calls: list[LlmToolCall] | None = None
    thinking: str | None = None
    # The call this message answers — set only on a tool-result turn.
    tool_call_id: str | None = None
    # The tool's own structural success stamp (``PennyConstants.
    # TOOL_RESULT_SUCCESS_KEY``, #1600) — set only on a tool-result turn. Internal
    # only: ``LlmClient._translate_messages`` drops it before the wire, but it
    # stays on the logged/stored shape ``to_log_dict()`` produces.
    tool_success: bool | None = None
    # Image attachments as data URIs — set only on the user turn they arrived with
    # (#2635; translated to OpenAI ``image_url`` content parts by ``LlmClient.
    # _translate_messages`` at the wire boundary, never here).
    images: list[str] | None = None

    def to_log_dict(self) -> dict[str, Any]:
        """The internal (pre-wire) message-dict shape every agent-loop call site
        builds today and the prompt log stores verbatim — the inverse of
        :meth:`from_dict`.  Wire-level translation (folding ``images`` into
        content-parts, stripping ``tool_success``) is one step later, in
        ``LlmClient._translate_messages`` — not here.

        Carries the step's thinking as ``reasoning``, so the next step of the same turn
        is drawn with it in view. A plan made before the first call ("read the page,
        then write it down") is otherwise gone by the time the result comes back, and
        the step after a tool result is then decided without it.

        It rides only on the messages of the turn in flight: the next turn's history is
        built from the message log, which holds what was said and never what was
        thought, so nothing here crosses turns."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.thinking:
            message["reasoning"] = self.thinking
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
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_success is not None:
            message[PennyConstants.TOOL_RESULT_SUCCESS_KEY] = self.tool_success
        if self.images is not None:
            message["images"] = self.images
        return message

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LlmMessage:
        """The inverse of :meth:`to_log_dict` — reconstructs a message from the
        internal dict shape, e.g. off a run's own promptlog rows (#1778) or at a
        call-site boundary that still builds that shape directly (#2563)."""
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [cls._tool_call_from_dict(call) for call in data["tool_calls"]]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tool_calls,
            thinking=data.get("reasoning"),
            tool_call_id=data.get("tool_call_id"),
            tool_success=data.get(PennyConstants.TOOL_RESULT_SUCCESS_KEY),
            images=data.get("images"),
        )

    @staticmethod
    def _tool_call_from_dict(data: dict[str, Any]) -> LlmToolCall:
        """One ``tool_calls`` entry off the internal dict shape — ``arguments``
        is always a JSON string there (the page-context call builds its arguments
        the same way as every other call since #2604)."""
        return LlmToolCall(
            id=data["id"],
            function=LlmToolCallFunction(
                name=data["function"]["name"],
                arguments=json.loads(data["function"]["arguments"]),
            ),
        )


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
    # Why the completion stopped (``stop``, ``length``, ``tool_calls``, …), read
    # straight off the raw choice.  ``"length"`` is the one value that matters today:
    # it is how a provider says a token cap cut the draw off mid-thought rather than
    # the model choosing to stop, and ``LlmResponse`` carried no way to tell the two
    # apart before this field existed — a capped response decoded identically to a
    # finished one.  What happens on ``"length"`` is the follow-up issues' (#2475/#2476);
    # this is only the read.
    finish_reason: str | None = None
    # The ``promptlog`` row id the client logged this call under, set by
    # ``LlmClient`` when the row is written (#2702) — logging identity only,
    # never model-facing (``LlmResponse`` never enters the model's context).  A
    # rejection site reads it to mark its own row via
    # ``db.messages.mark_prompt_discarded``; ``None`` only where no row was
    # written (a client built without a ``db`` — a test, a one-off script).
    prompt_log_id: int | None = None

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return bool(self.message.tool_calls)
