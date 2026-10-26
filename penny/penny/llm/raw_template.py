"""Gemma 4's own chat template, rendered by Penny and sent as a raw completions
prompt (#2829), so every runner serving the model sees byte-identical input
instead of each provider's own chat-template application.

#2828 found the eval suite's pinned provider never opens a new thought after a
tool result, where prod's oMLX and two other providers do — traced to the
providers disagreeing about how the chat-completions API applies the chat
template.  Rendering it ourselves removes that variable: Penny decides the
exact bytes, once, and every runner just tokenizes them.

The vendored template (``templates/gemma4_chat_template.jinja``) is confirmed
byte-for-byte against oMLX's own ``/v1/chat/completions`` render for a real
later-step request (system/user/assistant-tool-call/tool-result), once oMLX's
own request-normalisation (merging consecutive leading system messages,
enriching an undocumented required tool parameter with a synthetic
description) is accounted for.  Neither of those is part of the template
itself, and this module does not replicate them — the whole point of
rendering once, ourselves, is one consistent input across providers, not
reproducing any one provider's own pre-processing.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import jinja2

from penny.llm.models import (
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILENAME = "gemma4_chat_template.jinja"

# Model ids this transport has a vendored template for, matched as a plain
# case-insensitive substring of the model id — so both oMLX's own id
# ("gemma-4-26B-A4B-it-QAT-MLX-4bit") and OpenRouter's
# ("google/gemma-4-26b-a4b-it") resolve the same way.  No config knob: a model
# either has a vendored template checked into the repo or it doesn't.
_VENDORED_MODEL_MARKERS = ("gemma-4-26b-a4b-it",)

BOS_TOKEN = "<bos>"
# The two markers that end a generation, read off gemma's own
# ``generation_config`` eos ids (1 = ``<eos>``, 106 = ``<turn|>``, 50 =
# ``<|tool_response>``): the model ends an ordinary turn with ``<turn|>`` and
# ends a TOOL-CALL turn with ``<|tool_response>`` instead (the template's own
# generation prompt opens that block for it — see the vendored template's
# final ``add_generation_prompt`` branch).  A provider honouring only the
# tokenizer's bare eos id would let the model run on past a tool call and
# hallucinate the tool's own response text, so both are sent as stop
# sequences, not just the turn marker.
STOP_SEQUENCE = "<turn|>"
TOOL_RESPONSE_STOP_SEQUENCE = "<|tool_response>"
STOP_SEQUENCES = (STOP_SEQUENCE, TOOL_RESPONSE_STOP_SEQUENCE)
EOS_TOKEN = "<eos>"
# Markers a provider can still return as literal TEXT rather than consuming
# as the stop sequence (CoreWeave's probe behaviour) — stripped from the end
# of the content the parser hands back, so one never leaks into a reply.
_TRAILING_MARKERS = (TOOL_RESPONSE_STOP_SEQUENCE, STOP_SEQUENCE, EOS_TOKEN)

# The raw-faithfulness check (#2829 build decision 3): a raw completion of the
# bare BOS token must report exactly one prompt token.  A provider whose
# `/v1/completions` secretly wraps the prompt as a chat turn (Makora,
# Cloudflare, Google — #2828/#2829 step 1) reports more (a chat-wrap's own
# turn-opening tokens, e.g. 14 on one provider probed) — this is what
# distinguishes a truly raw endpoint from one that only looks like one.
RAW_FAITHFULNESS_PROMPT = BOS_TOKEN
RAW_FAITHFULNESS_MAX_TOKENS = 1
RAW_FAITHFULNESS_EXPECTED_PROMPT_TOKENS = 1

# The startup preflight's named degraded-mode message (#2829 build decision
# 3) — never silent: a confirmed UNFAITHFUL answer falls the client back to
# the chat transport, and says so in exactly these words.
RAW_FAITHFULNESS_DEGRADED_MESSAGE = (
    "raw template unavailable on {endpoint}; using the chat transport"
)

# A transient fault (429/5xx/timeout/connection) is NOT a verdict — it says
# the endpoint had a bad moment, not that it wraps raw completions.  Every
# attempt erroring still keeps the raw transport (a real call surfaces its
# own availability error honestly); this names the state out loud, never
# silently.
RAW_FAITHFULNESS_UNVERIFIED_MESSAGE = (
    "raw template faithfulness unverified on {endpoint} ({detail}); keeping the raw transport"
)

_STRING_OPEN = '<|"|>'
_TOOL_CALL_OPEN = "<|tool_call>"
_TOOL_CALL_CLOSE = "<tool_call|>"
_TOOL_CALL_HEAD = re.compile(r"<\|tool_call>call:(?P<name>[^{]+)\{", re.DOTALL)
_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Exponent form included (``1e+21``, ``1.5e-10``) — the template's own
# ``format_argument`` macro has no numeric branch of its own: a plain
# int/float falls to its ``else: {{ argument }}`` and Jinja prints Python's
# own ``str()``, which switches a float to exponential notation outside a
# normal range (confirmed: ``str(1e21) == "1e+21"``).
_NUMBER_PATTERN = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_NULL_LITERAL = "None"

# Gemma's own tool-call syntax carries no id at all (unlike the OpenAI
# tool-call schema) — Penny mints one per call, globally unique, since
# history, dedup, and the prompt log's ``parent_tool_call_id``/``CallLink``
# tracing all key on it; a per-RESPONSE counter (``raw_call_1``, …) would
# repeat across every turn's first call and collide.
_RAW_CALL_ID_PREFIX = "raw_call_"

# Half-width of the context window every ``LlmToolParseError`` message embeds
# around the exact position parsing broke (#2833) — ~200 characters total, enough
# to read what the model actually wrote there (what came just before the break,
# not only what follows it), for the DEBUG log line client.py reads straight off
# the raised error's own message.
_CONTEXT_RADIUS = 100


def _new_call_id() -> str:
    return f"{_RAW_CALL_ID_PREFIX}{uuid.uuid4().hex}"


def _context(text: str, pos: int) -> str:
    """~200 characters of ``text`` centred on ``pos`` — the failure point."""
    return text[max(0, pos - _CONTEXT_RADIUS) : pos + _CONTEXT_RADIUS]


# The thought channel's two equivalent spellings: the raw marker text gemma's
# template itself emits (CoreWeave — confirmed byte-identical to the local
# tokenizer, #2829 step 1) and oMLX's own rendering of the same channel as
# `<think>...</think>` (`omlx/adapter/gemma4.py`'s `Gemma4OutputParserSession`,
# selected by MODEL identity and applied unconditionally for this model family
# — there is no request flag that turns it off).  A provider added later that
# strips the markers entirely (Parasail, per #2829) would need its own
# detection added here as a third pattern — not built, since none is pinned
# yet.
_THOUGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\A<\|channel>thought\n(?P<body>.*?)\n?<channel\|>", re.DOTALL),
    re.compile(r"\A<think>\n?(?P<body>.*?)\n?</think>\n?", re.DOTALL),
)

_template_cache: jinja2.Template | None = None


def has_vendored_template(model: str) -> bool:
    """Whether ``model`` has a vendored chat template this module can render."""
    lowered = model.lower()
    return any(marker in lowered for marker in _VENDORED_MODEL_MARKERS)


def _template() -> jinja2.Template:
    """The vendored template, parsed once and cached for the process's life."""
    global _template_cache
    if _template_cache is None:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)))
        _template_cache = env.get_template(_TEMPLATE_FILENAME)
    return _template_cache


# ── Rendering ─────────────────────────────────────────────────────────────


def render_prompt(messages: list[dict], tools: list[dict] | None, thinking_enabled: bool) -> str:
    """Render Penny's wire-shape messages (post ``LlmClient._translate_messages``,
    ``tools`` post ``_translate_tools``) through the vendored template — the
    exact bytes sent as the raw completions prompt.

    ``enable_thinking`` maps straight from ``thinking_enabled``: the template
    opens the thought channel at the top of the leading system turn when True,
    and otherwise appends an empty, already-closed thought channel to the
    generation prompt instead of leaving it open (gemma's own semantics for
    "thinking off" — a closed channel, not an absent one).
    """
    return _template().render(
        messages=[_with_parsed_tool_call_arguments(message) for message in messages],
        tools=tools,
        bos_token=BOS_TOKEN,
        enable_thinking=thinking_enabled,
        add_generation_prompt=True,
    )


def _with_parsed_tool_call_arguments(message: dict) -> dict:
    """A message with each tool call's ``arguments`` restored from the wire's
    JSON string to a mapping.

    The wire shape always serialises ``arguments`` to a JSON string (the
    OpenAI tool-call schema).  The template accepts a string there too, but
    inserts it VERBATIM — plain JSON double quotes, not gemma's own ``<|"|>``
    escaping — which would render Penny's own history turns in a different
    syntax than the model's own past tool calls use.  Confirmed against
    oMLX's real render: only the mapping form produced a byte-for-byte match.
    """
    if not message.get("tool_calls"):
        return message
    message = dict(message)
    message["tool_calls"] = [_parsed_tool_call(call) for call in message["tool_calls"]]
    return message


def _parsed_tool_call(call: dict) -> dict:
    function = dict(call["function"])
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        function["arguments"] = json.loads(arguments) if arguments.strip() else {}
    return {**call, "function": function}


# ── Parsing ───────────────────────────────────────────────────────────────


def parse_completion(text: str) -> tuple[str | None, list[LlmToolCall] | None, str]:
    """Parse one raw completion's text into ``(thinking, tool_calls, content)``.

    Mirrors what the chat transport's provider-side tool-call parser already
    gives ``LlmClient._parse_response`` — the same three values, read here
    off gemma's own grammar instead of an OpenAI-shaped response object.
    Raises :class:`LlmToolParseError` on a tool call that opens but does not
    parse, exactly the bucket a malformed provider-parsed tool call falls
    into on the chat path today (no retry — the agent's own format-nudge
    retry is what follows).
    """
    thinking, rest = _split_thought(text)
    tool_calls, rest = _split_tool_calls(rest)
    content = _strip_trailing_markers(rest).strip()
    return thinking, tool_calls, content


def _split_thought(text: str) -> tuple[str | None, str]:
    for pattern in _THOUGHT_PATTERNS:
        match = pattern.match(text)
        if match:
            body = match.group("body").strip()
            return (body or None), text[match.end() :]
    return None, text


def _split_tool_calls(text: str) -> tuple[list[LlmToolCall] | None, str]:
    """Every ``<|tool_call>...<tool_call|>`` span ANYWHERE in ``text`` (not
    only at its very start) — a reply that opens with a sentence before
    calling a tool keeps that sentence as content instead of being rerolled
    as unparsed text, the same leniency the chat path's provider-parsed tool
    calls already get.  Content is every span of text OUTSIDE a call,
    concatenated in order.
    """
    calls: list[LlmToolCall] = []
    content_parts: list[str] = []
    pos = 0
    while True:
        idx = text.find(_TOOL_CALL_OPEN, pos)
        if idx == -1:
            content_parts.append(text[pos:])
            break
        content_parts.append(text[pos:idx])
        pos, call = _read_one_tool_call(text, idx)
        calls.append(call)
    return (calls or None), "".join(content_parts)


def _read_one_tool_call(text: str, start: int) -> tuple[int, LlmToolCall]:
    """One ``<|tool_call>call:name{...}<tool_call|>`` span starting at ``start``
    (where ``text[start:]`` begins with ``<|tool_call>``) — returns the
    position just past its closing marker, and the parsed call.

    The id is minted fresh per call (:func:`_new_call_id`), never derived
    from position within the response — gemma's own syntax carries no id at
    all, so Penny mints its own for the dispatcher's own result-pairing.
    """
    match = _TOOL_CALL_HEAD.match(text, start)
    if not match:
        raise LlmToolParseError(f"malformed tool call opening: {_context(text, start)!r}")
    name = match.group("name")
    arguments, pos = _parse_object_body(text, match.end())
    if text[pos : pos + len(_TOOL_CALL_CLOSE)] != _TOOL_CALL_CLOSE:
        raise LlmToolParseError(
            f"tool call to {name!r} is missing its closing marker: {_context(text, pos)!r}"
        )
    call = LlmToolCall(
        id=_new_call_id(), function=LlmToolCallFunction(name=name, arguments=arguments)
    )
    return pos + len(_TOOL_CALL_CLOSE), call


def _strip_trailing_markers(text: str) -> str:
    """Strip a trailing stop marker a provider returned as literal text
    rather than consuming as the stop sequence — ``<|tool_response>`` (the
    tool-call turn's own end marker) and ``<turn|>``/``<eos>`` (an ordinary
    turn's), repeated in case more than one trails."""
    changed = True
    while changed:
        changed = False
        for marker in _TRAILING_MARKERS:
            if text.endswith(marker):
                text = text[: -len(marker)]
                changed = True
    return text


def _parse_value(text: str, pos: int) -> tuple[Any, int]:
    pos = _skip_ws(text, pos)
    if text.startswith(_STRING_OPEN, pos):
        return _parse_string(text, pos)
    if text.startswith("{", pos):
        return _parse_object_body(text, pos + 1)
    if text.startswith("[", pos):
        return _parse_array(text, pos)
    if text.startswith("true", pos):
        return True, pos + 4
    if text.startswith("false", pos):
        return False, pos + 5
    if text.startswith(_NULL_LITERAL, pos):
        return None, pos + len(_NULL_LITERAL)
    return _parse_number(text, pos)


def _parse_string(text: str, pos: int) -> tuple[str, int]:
    start = pos + len(_STRING_OPEN)
    end = text.find(_STRING_OPEN, start)
    if end == -1:
        raise LlmToolParseError(
            f"unterminated string in tool call arguments: {_context(text, start)!r}"
        )
    return text[start:end], end + len(_STRING_OPEN)


def _parse_object_body(text: str, pos: int) -> tuple[dict[str, Any], int]:
    """``pos`` points just past an opening ``{``; returns the parsed mapping
    and the position just past the matching closing ``}``."""
    result: dict[str, Any] = {}
    pos = _skip_ws(text, pos)
    if text.startswith("}", pos):
        return result, pos + 1
    while True:
        key_match = _KEY_PATTERN.match(text, pos)
        if not key_match:
            raise LlmToolParseError(f"malformed tool call argument key at: {_context(text, pos)!r}")
        key, pos = key_match.group(0), key_match.end()
        pos = _skip_ws(text, pos)
        if not text.startswith(":", pos):
            raise LlmToolParseError(
                f"expected ':' after key {key!r} in tool call arguments: {_context(text, pos)!r}"
            )
        pos = _skip_ws(text, pos + 1)
        value, pos = _parse_value(text, pos)
        result[key] = value
        pos = _skip_ws(text, pos)
        if text.startswith(",", pos):
            pos = _skip_ws(text, pos + 1)
            continue
        if text.startswith("}", pos):
            return result, pos + 1
        raise LlmToolParseError(
            f"malformed tool call arguments: expected ',' or '}}': {_context(text, pos)!r}"
        )


def _parse_array(text: str, pos: int) -> tuple[list[Any], int]:
    pos = _skip_ws(text, pos + 1)
    result: list[Any] = []
    if text.startswith("]", pos):
        return result, pos + 1
    while True:
        value, pos = _parse_value(text, pos)
        result.append(value)
        pos = _skip_ws(text, pos)
        if text.startswith(",", pos):
            pos = _skip_ws(text, pos + 1)
            continue
        if text.startswith("]", pos):
            return result, pos + 1
        raise LlmToolParseError(
            f"malformed tool call arguments: expected ',' or ']': {_context(text, pos)!r}"
        )


def _parse_number(text: str, pos: int) -> tuple[int | float, int]:
    match = _NUMBER_PATTERN.match(text, pos)
    if not match:
        raise LlmToolParseError(f"malformed tool call argument value at: {_context(text, pos)!r}")
    raw = match.group(0)
    is_float = "." in raw or "e" in raw or "E" in raw
    value: int | float = float(raw) if is_float else int(raw)
    return value, match.end()


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
