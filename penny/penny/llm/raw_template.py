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
STOP_SEQUENCE = "<turn|>"

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
# 3) — never silent: a failed raw-faithfulness check falls the client back
# to the chat transport, and says so in exactly these words.
RAW_FAITHFULNESS_DEGRADED_MESSAGE = (
    "raw template unavailable on {endpoint}; using the chat transport"
)

_STRING_OPEN = '<|"|>'
_TOOL_CALL_OPEN = "<|tool_call>"
_TOOL_CALL_CLOSE = "<tool_call|>"
_TOOL_CALL_HEAD = re.compile(r"<\|tool_call>call:(?P<name>[^{]+)\{", re.DOTALL)
_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_PATTERN = re.compile(r"-?\d+(\.\d+)?")

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
    content = _strip_trailing_turn(rest).strip()
    return thinking, tool_calls, content


def _split_thought(text: str) -> tuple[str | None, str]:
    for pattern in _THOUGHT_PATTERNS:
        match = pattern.match(text)
        if match:
            body = match.group("body").strip()
            return (body or None), text[match.end() :]
    return None, text


def _split_tool_calls(text: str) -> tuple[list[LlmToolCall] | None, str]:
    calls: list[LlmToolCall] = []
    rest = text
    while rest.startswith(_TOOL_CALL_OPEN):
        rest, call = _read_one_tool_call(rest, len(calls) + 1)
        calls.append(call)
    return (calls or None), rest


def _read_one_tool_call(text: str, position: int) -> tuple[str, LlmToolCall]:
    """One ``<|tool_call>call:name{...}<tool_call|>`` span, and the text after it.

    ``position`` numbers calls within this one response (1-based) to
    synthesise a stable id — gemma's own tool-call syntax carries no id at
    all, unlike the OpenAI tool-call schema, so Penny mints one for the
    dispatcher's own result-pairing to key on.
    """
    match = _TOOL_CALL_HEAD.match(text)
    if not match:
        raise LlmToolParseError(f"malformed tool call opening: {text[:80]!r}")
    name = match.group("name")
    arguments, pos = _parse_object_body(text, match.end())
    if text[pos : pos + len(_TOOL_CALL_CLOSE)] != _TOOL_CALL_CLOSE:
        raise LlmToolParseError(f"tool call to {name!r} is missing its closing marker")
    call = LlmToolCall(
        id=f"raw_call_{position}",
        function=LlmToolCallFunction(name=name, arguments=arguments),
    )
    return text[pos + len(_TOOL_CALL_CLOSE) :], call


def _strip_trailing_turn(text: str) -> str:
    return text[: -len(STOP_SEQUENCE)] if text.endswith(STOP_SEQUENCE) else text


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
    return _parse_number(text, pos)


def _parse_string(text: str, pos: int) -> tuple[str, int]:
    start = pos + len(_STRING_OPEN)
    end = text.find(_STRING_OPEN, start)
    if end == -1:
        raise LlmToolParseError("unterminated string in tool call arguments")
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
            raise LlmToolParseError(
                f"malformed tool call argument key at: {text[pos : pos + 40]!r}"
            )
        key, pos = key_match.group(0), key_match.end()
        pos = _skip_ws(text, pos)
        if not text.startswith(":", pos):
            raise LlmToolParseError(f"expected ':' after key {key!r} in tool call arguments")
        pos = _skip_ws(text, pos + 1)
        value, pos = _parse_value(text, pos)
        result[key] = value
        pos = _skip_ws(text, pos)
        if text.startswith(",", pos):
            pos = _skip_ws(text, pos + 1)
            continue
        if text.startswith("}", pos):
            return result, pos + 1
        raise LlmToolParseError("malformed tool call arguments: expected ',' or '}'")


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
        raise LlmToolParseError("malformed tool call arguments: expected ',' or ']'")


def _parse_number(text: str, pos: int) -> tuple[int | float, int]:
    match = _NUMBER_PATTERN.match(text, pos)
    if not match:
        raise LlmToolParseError(f"malformed tool call argument value at: {text[pos : pos + 40]!r}")
    raw = match.group(0)
    value: int | float = float(raw) if "." in raw else int(raw)
    return value, match.end()


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
