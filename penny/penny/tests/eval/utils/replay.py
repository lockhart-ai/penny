"""Verbatim prompt replay — recreate a real model failure from the promptlog.

This is the tool for the **log → test → fix loop** (the project's durable process for
correcting model behaviour — see "The log → test → fix loop" in `penny/CLAUDE.md`):

  1. find candidates in the DB (promptlog rows exhibiting the failure),
  2. pull the FULL verbatim input the model saw (this module — read a row BY ID),
  3. run it to confirm the failure reproduces verbatim,
  4. genericize PII/real-topic mentions into a committable fixtures.py case,
  5. run it again to confirm the genericized version still reproduces,
  6. then tweak the prompt to correct it.

It carries no prompt content of its own — it reads a row by id from a SQLite DB — so
it's privacy-safe to commit while the real prompts stay in the local (gitignored) DB.

The synthetic eval cases approximate a production failure on privacy-safe topics; this
replays the EXACT failing prompt straight out of a local promptlog so there's no
approximation gap.

What it measures: one model turn per sample.  A collector cycle's first turn either
reaches for ``log_read`` (it's doing the work) or jumps to ``done()`` (it bailed —
"the user said nothing").  Replaying the verbatim messages N times gives the bail
rate and the reasoning-token cost of the current all-in-system / empty-user-turn
structure.

``--mode split`` applies the structural experiment to the SAME bytes: the generic
framing + runtime rules stay in the (now static, cacheable) system message, and the
collection-specific body + date move into the user turn — then replays again so the
two structures are compared on identical content.

``--mode verbatim-request`` (#2875) is a DIFFERENT kind of replay from every other
mode here: it never re-renders anything. Every other mode re-derives its wire body
through today's code (``LlmClient.chat()`` — its own template, its own sampling
defaults, no token budget) from the row's ``messages``/``tools`` alone, so a replay is
always a reconstruction at whatever commit happens to be checked out. This mode
instead reads the row's own ``request`` column — the exact kwargs ``LlmClient`` handed
the OpenAI SDK at send time (#2875) — and sends that body straight through a bare
``openai.AsyncOpenAI`` of its own, overriding only what differs about the TARGET
runner: the model and the provider pin inside ``extra_body``, both resolved from
`EVAL_MODELS` through the roster (`roster.resolve_from_env`), never read from a
hand-set ``LLM_MODEL``/``LLM_PROVIDER`` (#3001). Everything else — messages/prompt,
tools, every sampling field, stop sequences, the token budget — rides unchanged,
proving the bytes match rather than approximating them.

``--mode ollama-native`` sends the SAME rendered raw-template prompt through Ollama's
native ``/api/generate`` (``raw: true``) instead of the OpenAI-compatible
``/v1/completions`` every other raw-template column uses — for a deployment where the
latter chat-wraps the prompt instead of serving it faithfully (see the module-level
comment by ``_OLLAMA_NATIVE_GENERATE_PATH``).

``--jsonl PATH`` writes one full JSON record per sample (tool calls with their
arguments, the FULL thinking, the FULL reply, per-sample timing, any error) as each
sample finishes — so a long, concurrent (``--concurrency N``) run keeps every sample
it completed even if it's killed partway through, and a later pass can classify the
samples without re-running them.

Run inside the penny container (the prod DB is mounted at /penny/data):

    docker compose run --rm --no-deps penny \
        env LLM_API_URL=http://host.docker.internal:11434 LLM_MODEL=gpt-oss:20b \
        python -m penny.tests.eval.utils.replay --id 138385 --samples 12 --mode both

    docker compose run --rm --no-deps penny \
        env LLM_API_URL=http://host.docker.internal:11434 LLM_MODEL=gpt-oss:20b \
        python -m penny.tests.eval.utils.replay --id 138385 --samples 5 \
            --mode verbatim-request

A remote provider (OpenRouter) additionally needs ``LLM_API_KEY`` — optional, read
the same way ``make eval-remote`` reads it, never hardcoded here. The model and its
pin come from `EVAL_MODELS` through the roster (`roster.resolve_from_env`, #3001):
the suite's model (the roster's first entry) unless ``LLM_MODEL`` names another
ROSTER entry, and a model or provider outside the roster refuses the run up front —
a hand-set ``LLM_PROVIDER`` that does not match the resolved entry is never silently
ignored and never used. ``make replay ID=<promptlog id>`` (the Makefile target,
#2917) wires this up end to end: it resolves the named model's entry the same way
``make eval-remote`` does (the roster, over ``EVAL_MODELS``), reads the key host-side
from the primary checkout's ``.env`` the same way too (``from_env``, never exported
to a shell or assumed to already be in the container's environment, since only the
primary checkout's ``.env`` is ever a real file — a worktree's own is a placeholder
directory), and writes ``--jsonl`` to the primary checkout's
``data/reports/replay/<id>.jsonl`` so it survives the worktree that ran it being
swept. Name the model explicitly — the primary checkout's ``EVAL_MODELS`` may not
list the suite's model first, so an unnamed ``make replay`` can resolve to
whichever roster entry is first instead. This run draws 12 samples at once,
concurrency 5, with the default ``--mode plain``:

    LLM_MODEL=google/gemma-4-26b-a4b-it make replay ID=138385 SAMPLES=12 CONCURRENCY=5

The supported A/B path is ``make replay ID=<row> VARIANTS=<spec.json> SAMPLES=<n>``
(#3001): the spec names each variant and its transforms of the stored request (a
system-prompt replacement or append, a ``last_system_append`` to the request's LAST
system message — the per-turn dynamic one (#3008), a ``last_tool_result`` replacing
the request's LAST tool message — the tool result the model reacts to, a swapped
message, thinking on or
off, and — applied FIRST, before any of those (#3058) — ``substitutions``: a list
of ``{pattern, replacement}`` regex pairs rewritten into EVERY message's text
content (its reasoning/thinking text included), every tool call's arguments
string, and the tools' JSON text, so a
notation that appears throughout a request (a link-handle format, say) can be
re-rendered wholesale per arm), every
arm is re-rendered through the byte-identity check for ``raw_template`` rows, and all
arms go through the SAME resolved roster entry in one SDK session. The recipe holds
the provider lock for the whole run and resolves the roster once for all arms — the
same lock and the same resolution every other recipe path uses; ``replay_variants``
is that recipe's Python step, not a separately importable entry point (kittens call
make targets, never harness helpers).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import os
import re
import sqlite3
import tempfile
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from sqlmodel import Session

# Import order workaround: ``penny.database`` and ``penny.agents`` import each
# other (``database.py`` -> ``memory.objects`` -> ``validation`` -> ``agents.models``
# -> ``agents.base`` -> ``from penny.database import Database``), and this module is
# the FIRST thing to import either package when run standalone (``python -m
# penny.tests.eval.utils.replay``). Importing ``penny.agents`` first means the cycle
# resolves the way it does from every other entry point (``penny.py`` imports agents
# before anything reaches back into ``penny.database``) instead of deadlocking on a
# partially-initialized ``penny.database`` package.
import penny.agents  # noqa: F401
from penny.database.database import Database
from penny.database.models import PromptLog
from penny.llm import raw_template
from penny.llm.call_link import CallLink
from penny.llm.client import REASONING_ENABLED_BODY, THINKING_OFF_BODY, LlmClient
from penny.llm.models import (
    PROVIDER_REQUEST_FIELD,
    LlmError,
    LlmMessage,
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
    PromptTransport,
    ProviderPin,
)
from penny.tests.eval.utils.roster import EvalModel, resolve_from_env

_DEFAULT_DB = "/penny/data/penny/penny.db"
# The A/B spec's env var (#3001): the make recipe reads the `VARIANTS` file host-side
# and forwards its JSON here (an env value never word-splits the way an argv JSON
# would, and a host path would not be visible in the container the tool runs in).
# Empty reads as unset, the same convention `env_seconds` uses for a
# recipe-forwarded-but-empty variable.
REPLAY_VARIANTS_SPEC_ENV = "REPLAY_VARIANTS_SPEC"
_RUNTIME_MARKER = "## Runtime rules"
_COLLECTOR_MARKER = "You are the collector"

# ``--mode ollama-native`` (#2884): Ollama's OpenAI-compatible
# ``/v1/completions`` was measured chat-wrapping the prompt on this deployment
# (a bare ``<bos>`` probe read 17 prompt tokens, not 1 — the same chat-wrap
# signature ``check_raw_template_faithfulness`` catches for a gateway provider),
# so it is not the faithful raw path here the way it is for oMLX/OpenRouter.
# Ollama's own NATIVE endpoint with ``raw: true`` IS faithful (confirmed: the
# same bare-BOS probe reads exactly 1 prompt token there) — this mode renders
# the SAME vendored template (``raw_template.render_prompt``) and sends it
# through that native endpoint instead, with the same sampling
# (temperature 1.0, top_p 0.95, top_k 64) and stop sequences every other
# column's raw-template transport uses.
_OLLAMA_NATIVE_GENERATE_PATH = "/api/generate"
_NANOSECONDS_PER_MS = 1_000_000


@dataclass
class _Tally:
    """First-turn outcomes across a replay batch.

    The cycle's first move is the whole signal: ``done()`` first = bailed (gave up
    without doing the work); any other tool = engaged (started the work); no tool
    call = the model emitted bare text instead of acting.
    """

    samples: int = 0
    bailed: int = 0  # first move was done()
    engaged: int = 0  # first move was a real tool (a read, a browse, …)
    no_tool: int = 0  # emitted text, no tool call
    errored: int = 0  # the call raised (e.g. an un-parseable tool-call argument)
    bail_thinking_chars: int = 0
    engaged_thinking_chars: int = 0
    first_tools: dict[str, int] = field(default_factory=dict)

    def record(self, first_tool: str | None, thinking: str | None) -> None:
        self.samples += 1
        chars = len(thinking or "")
        key = first_tool or "<none>"
        self.first_tools[key] = self.first_tools.get(key, 0) + 1
        if first_tool is None:
            self.no_tool += 1
        elif first_tool == "done":
            self.bailed += 1
            self.bail_thinking_chars += chars
        else:
            self.engaged += 1
            self.engaged_thinking_chars += chars

    def record_error(self) -> None:
        """An API error on this sample — the model produced output the backend
        couldn't parse (e.g. a tool-call argument wrapped in markdown backticks),
        which in production aborts the turn and lands no change."""
        self.samples += 1
        self.errored += 1
        self.first_tools["<error>"] = self.first_tools.get("<error>", 0) + 1

    def report(self, label: str) -> None:
        bail_rate = 100 * self.bailed / self.samples if self.samples else 0
        avg_engaged = self.engaged_thinking_chars / self.engaged if self.engaged else 0
        print(
            f"\n[{label}] {self.samples} samples · "
            f"BAILED {self.bailed} ({bail_rate:.0f}%) · engaged {self.engaged} · "
            f"no-tool {self.no_tool} · errored {self.errored} · "
            f"{avg_engaged:.0f} avg reasoning chars when engaged · "
            f"first-tools {self.first_tools}"
        )


def _load_prompt(db_path: str, prompt_id: int) -> tuple[list[dict], list[dict] | None]:
    """Read the verbatim messages + tools off one promptlog row.

    Opened read-only (URI ``mode=ro``): this script's whole point is to read a
    production row without any chance of writing to the DB it came from.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT messages, tools, agent_name, run_target FROM promptlog WHERE id = ?",
            (prompt_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise SystemExit(f"no promptlog row with id={prompt_id} in {db_path}")
    messages, tools, agent_name, run_target = row
    print(f"loaded id={prompt_id} agent={agent_name} target={run_target}")
    return json.loads(messages), (json.loads(tools) if tools else None)


def _load_request(db_path: str, prompt_id: int) -> dict:
    """Read the EXACT request off one promptlog row's ``request`` column (#2875) —
    the zlib-compressed JSON dump of the ``SentRequest`` ``LlmClient`` stamped at
    send time: ``{"transport", "endpoint", "model", "body", "header_names"}``,
    where ``body`` is literally the kwargs handed to the OpenAI SDK. Never a
    re-render — there is nothing here for this tool to reconstruct."""
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT request, agent_name, run_target FROM promptlog WHERE id = ?",
            (prompt_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise SystemExit(f"no promptlog row with id={prompt_id} in {db_path}")
    request_blob, agent_name, run_target = row
    if request_blob is None:
        raise SystemExit(
            f"promptlog row id={prompt_id} carries no stored request (#2875) — "
            "logged before the column existed, or a cancelled call that never "
            "got far enough to leave one behind"
        )
    print(f"loaded id={prompt_id} agent={agent_name} target={run_target}")
    return json.loads(zlib.decompress(request_blob))


def _client_for(api_url: str, api_key: str) -> openai.AsyncOpenAI:
    """The bare SDK client the verbatim and A/B transports send through — one seam,
    so the tests can stand in for the SDK boundary without a live endpoint."""
    return openai.AsyncOpenAI(base_url=f"{api_url.rstrip('/')}/v1", api_key=api_key)


def _retarget_body(body: dict, entry: EvalModel) -> dict:
    """Override ONLY what differs about the TARGET RUNNER (#2875, retarget always
    #3001) — ``model`` and the provider pin inside ``extra_body`` — leaving the
    prompt/messages, tools, every sampling field, and the stop sequences exactly as
    recorded. The target is ALWAYS the resolved roster entry: nothing may run against
    a model or upstream `EVAL_MODELS` does not define, so there is no
    "replay-against-the-stored-runner" mode left — a stored request for a different
    model is retargeted to the resolved one, and a stored pin from the original call
    is STRIPPED when the resolved entry names no provider, because leaving it behind
    would route the replay to the original call's upstream rather than the resolved
    one."""
    body = dict(body)
    body["model"] = entry.model
    extra_body = dict(body.get("extra_body") or {})
    if entry.provider:
        extra_body[PROVIDER_REQUEST_FIELD] = ProviderPin(provider=entry.provider).as_request_field()
        body["extra_body"] = extra_body
    elif PROVIDER_REQUEST_FIELD in extra_body or "extra_body" in body:
        extra_body.pop(PROVIDER_REQUEST_FIELD, None)
        if extra_body:
            body["extra_body"] = extra_body
        else:
            body.pop("extra_body", None)
    return body


async def _send_verbatim_request(
    client: openai.AsyncOpenAI, transport: str, body: dict
) -> tuple[str | None, list[LlmToolCall], str]:
    """Send ``body`` straight through the bare SDK client — no ``LlmClient``, no
    template re-render, no sampling default of this tool's own — and parse the
    response the SAME way prod parses it on this transport. Returns
    ``(thinking, tool_calls, reply_content)`` with the calls as typed
    ``LlmToolCall``s (arguments parsed off the wire into dicts, the record shape
    the JSONL carries whole).

    A raw-template body recorded with ``stream: true`` is sent with streaming
    turned OFF instead (``stream_options`` dropped along with it): sampling is
    unaffected either way (streaming only changes HOW the bytes arrive, never
    what is asked for), and this tool has no need for the mid-stream cut
    behaviour streaming exists to support in production — only the final text,
    parsed once, the same way a non-cut draw always was.
    """
    if transport == PromptTransport.RAW_TEMPLATE.value:
        body = dict(body)
        body.pop("stream_options", None)
        body["stream"] = False
        raw = await client.completions.create(**body)
        thinking, tool_calls, content = raw_template.parse_completion(raw.choices[0].text or "")
        return thinking, list(tool_calls or []), content
    raw = await client.chat.completions.create(**body)
    message = raw.choices[0].message
    extras = message.model_extra or {}
    thinking = extras.get("reasoning_content") or extras.get("reasoning")
    calls = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append(
            LlmToolCall(
                id=call.id,
                function=LlmToolCallFunction(name=call.function.name, arguments=arguments),
            )
        )
    return thinking, calls, message.content or ""


async def _replay_verbatim_request(
    db_path: str, prompt_id: int, samples: int, entry: EvalModel
) -> None:
    """``--mode verbatim-request`` (#2875): replay the row's own exact wire
    request, unchanged apart from the target runner, N times — see the module
    docstring for why this mode is not a re-render like every other one here.

    The target runner is the resolved roster entry (passed in — ``_main`` resolves
    ONCE for the whole invocation, #3001): ``LLM_API_URL``/``LLM_API_KEY`` pick where
    this sends it, and the model + pin come from `EVAL_MODELS` — never from a
    hand-set ``LLM_MODEL``/``LLM_PROVIDER``, which are only checked against the
    roster and otherwise refused.
    """
    sent_request = _load_request(db_path, prompt_id)
    transport = sent_request["transport"]
    body = _retarget_body(sent_request["body"], entry)
    api_url = os.environ.get("LLM_API_URL", "http://host.docker.internal:11434")
    client = _client_for(api_url, os.environ.get("LLM_API_KEY") or "not-needed")
    print(
        f"replaying id={prompt_id} transport={transport} model={body.get('model')} "
        f"against {api_url}"
    )
    for index in range(samples):
        try:
            thinking, calls, content = await _send_verbatim_request(client, transport, body)
        except openai.OpenAIError as error:
            print(f"  [{index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            continue
        names = [call.function.name for call in calls]
        tail = (thinking or "")[-200:]
        print(
            f"  [{index + 1}/{samples}] tools={names or '—'} "
            f"thinking_tail={tail!r} reply={content[:200]!r}"
        )


class Substitution(BaseModel):
    """One entry of a variant's ``substitutions`` list (#3058) — ``pattern`` (a
    Python regex) replaced by ``replacement`` (a ``re.sub`` repl string, so
    backreferences like ``\\1`` work). Unknown keys are refused, like the
    variant's own."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    replacement: str


class ReplayVariant(BaseModel):
    """One named arm of an A/B replay (#3001) — named transforms of ONE stored
    request, everything left unset riding as recorded. ``name`` is the arm's
    label in the JSONL and the tally report; duplicate names are refused, since
    two arms with one label could not be told apart in the output. Unknown keys
    are refused too (``extra="forbid"``), so a typo'd spec key errors loudly
    instead of silently running the arm untransformed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Regex substitutions applied FIRST (#3058), in list order, to every
    # message's text content (its reasoning/thinking text included), every tool
    # call's arguments string, and the tools' JSON text — the lever for a
    # notation that appears throughout a request (e.g. re-rendering every link
    # handle in a new format). A pattern that is not a valid regex refuses the
    # arm at validation, and with it the spec.
    substitutions: list[Substitution] = Field(default_factory=list)
    # Replaces the system message's content (the prompt A/B's usual lever).
    system_prompt: str | None = None
    # Appended after the recorded system content (a follow-up instruction A/B),
    # separated by a blank line — composes with `system_prompt` (replace first,
    # then append) and with leaving the recorded content alone.
    system_prompt_append: str | None = None
    # Replaces the final user message's content (the trigger-wording A/B).
    user_message: str | None = None
    # Appended after the LAST system message's content (#3008 — a chat request's
    # per-turn dynamic message just before the user's turn), separated by a blank
    # line; on a one-system request that is the same message the first-append
    # targets, so the two compose. Applied after the first-message transforms.
    # A request with no system message refuses the arm, as the first-message
    # transforms do.
    last_system_append: str | None = None
    # Replaces the LAST tool-role message's content — the tool result the model is
    # about to react to (the A/B lever for a tool-result wording). A request with
    # no tool message refuses the arm, as the other transforms do.
    last_tool_result: str | None = None
    # The reasoning channel on or off; None keeps the recorded state.
    thinking: bool | None = None

    @field_validator("substitutions")
    @classmethod
    def _patterns_compile(cls, value: list[Substitution]) -> list[Substitution]:
        """Every pattern compiles — an invalid regex refuses the arm (and with it
        the whole spec) before anything is loaded or sent, as every other spec
        error does."""
        for substitution in value:
            try:
                re.compile(substitution.pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid substitution regex {substitution.pattern!r}: {error}"
                ) from error
        return value


# The A/B spec file validates as a WHOLE list, so one bad entry refuses the run
# before anything is loaded or sent (the roster's own fail-fast shape).
_VARIANT_SPEC_ADAPTER = TypeAdapter(list[ReplayVariant])


def _substituted_text(text: str, subs: Sequence[tuple[re.Pattern[str], str]]) -> str:
    """One text through every substitution in list order (#3058)."""
    for regex, replacement in subs:
        text = regex.sub(replacement, text)
    return text


def _substituted_call(call: dict, subs: Sequence[tuple[re.Pattern[str], str]]) -> dict:
    """One tool call's deep copy with its arguments string substituted — a deep
    copy so the loaded row's own call dict is never mutated (the variants share
    the loaded messages, and a shallow copy would leak one arm's rewrite into
    the next)."""
    call = copy.deepcopy(call)
    arguments = (call.get("function") or {}).get("arguments")
    if isinstance(arguments, str):
        call["function"]["arguments"] = _substituted_text(arguments, subs)
    return call


def _substituted_messages_and_tools(
    messages: list[dict], tools: list[dict] | None, variant: ReplayVariant
) -> tuple[list[dict], list[dict] | None]:
    """The variant's substitutions applied (#3058) — BEFORE any other transform,
    in list order, to every message's text content (the ``reasoning``/
    ``thinking`` field included — on a raw_template row it renders into the
    prompt and quotes the very notation an arm re-renders), every tool call's
    arguments string, and the tools' JSON text (each tool schema goes through
    its JSON dump, so a substitution reaches every string field of every
    schema). Returns the inputs unchanged when the arm names none."""
    if not variant.substitutions:
        return messages, tools
    subs = [(re.compile(sub.pattern), sub.replacement) for sub in variant.substitutions]
    rebuilt = []
    for message in messages:
        message = dict(message)
        if isinstance(message.get("content"), str):
            message["content"] = _substituted_text(message["content"], subs)
        if isinstance(message.get("reasoning"), str):
            message["reasoning"] = _substituted_text(message["reasoning"], subs)
        if isinstance(message.get("thinking"), str):
            message["thinking"] = _substituted_text(message["thinking"], subs)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                _substituted_call(call, subs) for call in message["tool_calls"]
            ]
        rebuilt.append(message)
    if tools is not None:
        tools = json.loads(_substituted_text(json.dumps(tools), subs))
    return rebuilt, tools


def _transformed_messages(messages: list[dict], variant: ReplayVariant) -> list[dict]:
    """Apply the variant's message transforms — system replacement/append and a
    user content swap — to a copy of the messages, never mutating the loaded row."""
    rebuilt = [dict(message) for message in messages]
    if variant.system_prompt is not None or variant.system_prompt_append is not None:
        system_at = next((i for i, m in enumerate(rebuilt) if m["role"] == "system"), None)
        if system_at is None:
            raise ValueError(
                f"variant '{variant.name}': the stored request has no system message to transform"
            )
        content = rebuilt[system_at]["content"]
        if variant.system_prompt is not None:
            content = variant.system_prompt
        if variant.system_prompt_append is not None:
            content = (
                f"{content}\n\n{variant.system_prompt_append}"
                if content
                else variant.system_prompt_append
            )
        rebuilt[system_at]["content"] = content
    if variant.last_system_append is not None:
        last_system_at = next(
            (i for i in reversed(range(len(rebuilt))) if rebuilt[i]["role"] == "system"), None
        )
        if last_system_at is None:
            raise ValueError(
                f"variant '{variant.name}': the stored request has no system message to transform"
            )
        content = rebuilt[last_system_at]["content"]
        rebuilt[last_system_at]["content"] = (
            f"{content}\n\n{variant.last_system_append}" if content else variant.last_system_append
        )
    if variant.last_tool_result is not None:
        last_tool_at = next(
            (i for i in reversed(range(len(rebuilt))) if rebuilt[i]["role"] == "tool"), None
        )
        if last_tool_at is None:
            raise ValueError(
                f"variant '{variant.name}': the stored request has no tool message to replace"
            )
        rebuilt[last_tool_at]["content"] = variant.last_tool_result
    if variant.user_message is not None:
        user_at = next(
            (i for i in reversed(range(len(rebuilt))) if rebuilt[i]["role"] == "user"), None
        )
        if user_at is None:
            raise ValueError(
                f"variant '{variant.name}': the stored request has no user message to replace"
            )
        rebuilt[user_at]["content"] = variant.user_message
    return rebuilt


def _thinking_body_keys(body: dict, enabled: bool) -> dict:
    """The chat-transport thinking toggle — the SAME two building blocks
    ``LlmClient`` composes its body from (``REASONING_ENABLED_BODY`` /
    ``THINKING_OFF_BODY``), merged the same way: the off form must omit
    ``reasoning`` entirely (Ollama reads the key's presence, not its content),
    so every thinking key is dropped before the chosen block is applied."""
    body = dict(body)
    for key in ("reasoning", "reasoning_effort", "chat_template_kwargs"):
        body.pop(key, None)
    body.update(REASONING_ENABLED_BODY if enabled else THINKING_OFF_BODY)
    return body


def _recorded_thinking(messages: list[dict], tools: list[dict] | None, stored_prompt: str) -> bool:
    """Which thinking state the row was rendered with — the BYTE-IDENTITY CHECK
    for a raw_template row (#3001): the untransformed re-render must match the
    stored ``prompt`` for exactly one state. Neither matching means this
    re-render is not faithful to what prod sent (a thinking-loop continuation's
    prompt, say, or a template that has since changed), and an A/B re-rendered
    on top of it would compare against different bytes — refused."""
    for enabled in (True, False):
        rendered = raw_template.render_prompt(messages, tools, thinking_enabled=enabled)
        if rendered == stored_prompt:
            return enabled
    raise ValueError(
        "the stored request's prompt does not byte-match a re-render of the row's own "
        "messages under either thinking state — the re-render is not faithful to what "
        "prod sent (a continuation call's prompt, or a template that has since "
        "changed), so an A/B on it would compare against different bytes"
    )


def _variant_bodies(
    sent_request: dict, db_path: str, prompt_id: int, variants: Sequence[ReplayVariant]
) -> dict[str, dict]:
    """Each variant's wire body, per transport. A raw_template row re-renders its
    transformed messages through ``render_prompt`` — gated by the byte-identity
    check, which proves the re-render pipeline is faithful before any variant's
    render is trusted; a chat row transforms the stored body's own ``messages``
    (the exact wire bytes, nothing to re-derive). A variant's substitutions
    (#3058) run FIRST, on the messages and the tools, before any other transform."""
    transport = sent_request["transport"]
    base = sent_request["body"]
    bodies: dict[str, dict] = {}
    if transport == PromptTransport.RAW_TEMPLATE.value:
        messages, tools = _load_prompt(db_path, prompt_id)
        recorded = _recorded_thinking(messages, tools, base["prompt"])
        for variant in variants:
            thinking = variant.thinking if variant.thinking is not None else recorded
            subbed_messages, subbed_tools = _substituted_messages_and_tools(
                messages, tools, variant
            )
            body = dict(base)
            body["prompt"] = raw_template.render_prompt(
                _transformed_messages(subbed_messages, variant),
                subbed_tools,
                thinking_enabled=thinking,
            )
            bodies[variant.name] = body
        return bodies
    for variant in variants:
        body = (
            _thinking_body_keys(base, variant.thinking)
            if variant.thinking is not None
            else dict(base)
        )
        subbed_messages, subbed_tools = _substituted_messages_and_tools(
            body.get("messages") or [], body.get("tools"), variant
        )
        body["messages"] = _transformed_messages(subbed_messages, variant)
        if subbed_tools is not None:
            body["tools"] = subbed_tools
        bodies[variant.name] = body
    return bodies


async def _one_variant_sample(
    client: openai.AsyncOpenAI,
    transport: str,
    body: dict,
    variant: ReplayVariant,
    index: int,
    samples: int,
    semaphore: asyncio.Semaphore,
    handle: Any | None,
    write_lock: asyncio.Lock | None,
) -> _SampleRecord:
    """One sample of one arm, gated by the arm's shared semaphore, printing and
    recording in the same shape every other mode here prints. The record lands
    in the JSONL the moment it finishes (#2884 pause resilience), never batched."""
    async with semaphore:
        try:
            thinking, calls, content = await _send_verbatim_request(client, transport, body)
        except openai.OpenAIError as error:
            print(f"  [{variant.name} {index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            record = _SampleRecord(
                label=variant.name,
                sample=index + 1,
                error=f"{type(error).__name__}: {error}",
            )
            await _write_record(record, handle, write_lock)
            return record
    names = [call.function.name for call in calls]
    print(
        f"  [{variant.name} {index + 1}/{samples}] first={names[0] if names else '—'} tools={names}"
    )
    print(f"    thinking-tail: {(thinking or '')[-300:]!r}")
    print(f"    reply: {content!r}")
    record = _SampleRecord(
        label=variant.name,
        sample=index + 1,
        tool_calls=[
            _ToolCallRecord(name=call.function.name, arguments=call.function.arguments)
            for call in calls
        ],
        thinking=thinking,
        reply=content,
    )
    await _write_record(record, handle, write_lock)
    return record


async def _run_variant_arm(
    client: openai.AsyncOpenAI,
    transport: str,
    body: dict,
    variant: ReplayVariant,
    samples: int,
    concurrency: int,
    jsonl_path: str | None,
) -> _Tally:
    """One arm's full batch of samples — concurrent under the same semaphore
    shape as every other mode, JSONL written per-sample the moment it finishes
    (#2884), tallied exactly like the other modes' ``_replay``."""
    tally = _Tally()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(open(jsonl_path, "w")) if jsonl_path else None
        write_lock = asyncio.Lock() if handle is not None else None
        tasks = [
            _one_variant_sample(
                client, transport, body, variant, index, samples, semaphore, handle, write_lock
            )
            for index in range(samples)
        ]
        records = await asyncio.gather(*tasks)
    for record in records:
        if record.error is not None:
            tally.record_error()
        else:
            first = record.tool_calls[0].name if record.tool_calls else None
            tally.record(first, record.thinking)
    return tally


async def replay_variants(
    prompt_id: int,
    variants: Sequence[ReplayVariant],
    samples: int,
    *,
    db_path: str = _DEFAULT_DB,
    jsonl_base: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = 1,
) -> dict[str, _Tally]:
    """The A/B run behind ``make replay … VARIANTS=<spec.json>`` (#3001): N samples
    of each named variant of ONE stored request, every arm through the SAME
    resolved target, in one SDK session. NOT a separately importable entry point —
    the make recipe is the only caller, holds the provider lock for the whole run
    (the same shell lock every other recipe path uses), and resolves the roster
    for its own banner before dispatching here.

    The target is resolved ONCE from `EVAL_MODELS` through the roster
    (``resolve_from_env``) — a model or upstream the roster does not define
    refuses the run before anything is sent. Every arm's body is retargeted to
    that one entry (model + pin), so all arms in one call necessarily carry the
    same model and pin. ``jsonl_base`` writes one JSONL per arm
    (``<base>-<name>.jsonl``), per-sample, the moment each finishes.
    """
    if not variants:
        raise ValueError("replay_variants: at least one variant is required")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"replay_variants: duplicate variant names: {sorted(set(names))}")
    entry = resolve_from_env()
    sent_request = _load_request(db_path, prompt_id)
    transport = sent_request["transport"]
    bodies = _variant_bodies(sent_request, db_path, prompt_id, variants)
    retargeted = {name: _retarget_body(body, entry) for name, body in bodies.items()}
    url = api_url or os.environ.get("LLM_API_URL", "http://host.docker.internal:11434")
    client = _client_for(url, api_key or os.environ.get("LLM_API_KEY") or "not-needed")
    print(
        f"replaying id={prompt_id} transport={transport} model={entry.model} "
        f"against {url} · {len(variants)} variant(s) x {samples} sample(s)"
    )
    tallies: dict[str, _Tally] = {}
    for variant in variants:
        jsonl_path = f"{jsonl_base.rsplit('.', 1)[0]}-{variant.name}.jsonl" if jsonl_base else None
        tally = await _run_variant_arm(
            client,
            transport,
            retargeted[variant.name],
            variant,
            samples,
            concurrency,
            jsonl_path,
        )
        tally.report(variant.name)
        tallies[variant.name] = tally
    return tallies


def _is_collector_prompt(messages: list[dict]) -> bool:
    """True when the row is a collector cycle (has both structural markers).

    Only collector prompts carry the ``## Runtime rules`` + ``You are the collector``
    markers the placement experiment splits on.  Any other prompt (a chat turn, a
    startup announcement) is replayed verbatim instead — the loop's core move (read a
    row by id, replay it N times) works for every prompt; the system/user split does not.
    """
    system = next((message for message in messages if message["role"] == "system"), None)
    return (
        system is not None
        and _RUNTIME_MARKER in system["content"]
        and _COLLECTOR_MARKER in system["content"]
    )


def _parse_system(messages: list[dict]) -> tuple[str, str, str, list[dict]]:
    """Pull (date, collector_body, runtime_rules) out of the system message.

    Returns those three parts plus the non-system messages (the empty user turn,
    any tool turns) so a caller can recompose the prompt under either placement.
    """
    system = next((message for message in messages if message["role"] == "system"), None)
    if system is None or _RUNTIME_MARKER not in system["content"]:
        raise SystemExit("system message has no runtime-rules marker — not a collector prompt?")
    content = system["content"]
    collector_at = content.index(_COLLECTOR_MARKER)
    runtime_at = content.index(_RUNTIME_MARKER)
    date_part = content[:collector_at].strip()
    body = content[collector_at:runtime_at].strip()
    runtime = content[runtime_at:].strip()
    rest = [message for message in messages if message["role"] != "system"]
    return date_part, body, runtime, rest


def _compose(
    date: str,
    body: str,
    runtime: str,
    rest: list[dict],
    *,
    placement: str,
    nudge: str | None = None,
) -> list[dict]:
    """Rebuild the prompt with the collector ``body`` in ``system`` or ``user``.

    ``system`` reproduces the current production shape (everything in the system
    message, the user turn left empty) — UNLESS ``nudge`` is given, in which case
    the otherwise-empty user turn is filled with that generic string (the body
    stays in system).  ``user`` keeps only the static framing + runtime rules in
    system and moves the date + collection body into the user turn.
    """
    if placement == "system":
        system_content = f"{date}\n\n{body}\n\n{runtime}"
        rebuilt: list[dict] = [{"role": "system", "content": system_content}]
        for message in rest:
            if message["role"] == "user" and nudge is not None:
                rebuilt.append({"role": "user", "content": nudge})
            else:
                rebuilt.append(message)
        return rebuilt
    system_content = f"You are one of Penny's background collectors.\n\n{runtime}"
    user_content = f"{date}\n\n{body}\n\nRun your collection cycle now."
    rebuilt = [{"role": "system", "content": system_content}]
    for message in rest:
        if message["role"] == "user":
            rebuilt.append({"role": "user", "content": user_content})
        else:
            rebuilt.append(message)
    return rebuilt


class _TimingRecord(BaseModel):
    """One sample's own per-call usage/timing (#2884 performance column) —
    present (every field defaulting to ``None``) on every ``_SampleRecord``,
    never silently absent, so a reader can tell "no metrics configured" /
    "endpoint reported nothing" apart from "this field never existed"."""

    wall_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    prefill_ms: int | None = None
    generation_ms: int | None = None
    decode_tokens_per_sec: float | None = None


class _ToolCallRecord(BaseModel):
    """One tool call inside a ``_SampleRecord`` — name plus its full arguments,
    never truncated (the live ``print`` tail is for a human watching the run;
    the JSONL record is for a later pass to classify without re-running)."""

    name: str
    arguments: dict[str, Any]


class _SampleRecord(BaseModel):
    """One sample's full, structured draw (#2884) — the tool calls with their
    arguments, the FULL thinking, and the FULL reply — everything a later pass
    needs to classify the sample against a failure definition without
    re-running it. Written to ``--jsonl`` as one line per sample by
    ``_write_record``."""

    label: str
    sample: int
    error: str | None = None
    tool_calls: list[_ToolCallRecord] = Field(default_factory=list)
    thinking: str | None = None
    reply: str | None = None
    timing: _TimingRecord = Field(default_factory=_TimingRecord)


def _timing_for(metrics_db: Database | None, prompt_log_id: int | None) -> _TimingRecord:
    """This sample's own per-call usage/timing, read back off the ``promptlog``
    row ``LlmClient._log_call`` already wrote it to (#2793's ``UsageTiming``) —
    a READ, not a re-derivation, so this reports exactly what the serving
    runner's response carried, under the field names
    ``penny.llm.models.usage_timing_from_response`` already gives them (oMLX's
    ``generation_duration``/``prompt_eval_duration`` seconds, Ollama native's
    ``eval_duration``/``prompt_eval_duration`` nanoseconds, or nothing at all
    from a gateway that reports neither). An all-``None`` ``_TimingRecord`` when
    there is no row to read — no ``--metrics-db`` configured, or a
    cancelled/errored call that never logged one.
    """
    if metrics_db is None or prompt_log_id is None:
        return _TimingRecord()
    with Session(metrics_db.engine) as session:
        row = session.get(PromptLog, prompt_log_id)
    if row is None:
        return _TimingRecord()
    decode_tokens_per_sec = None
    if row.completion_tokens and row.generation_ms:
        decode_tokens_per_sec = round(row.completion_tokens / (row.generation_ms / 1000), 1)
    return _TimingRecord(
        wall_ms=row.duration_ms,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        cached_tokens=row.cached_tokens,
        prefill_ms=row.prefill_ms,
        generation_ms=row.generation_ms,
        decode_tokens_per_sec=decode_tokens_per_sec,
    )


def _sample_record(
    label: str,
    index: int,
    response,
    error: LlmError | None,
    timing: _TimingRecord | None = None,
) -> _SampleRecord:
    """Build this sample's ``_SampleRecord`` off either an ``LlmError`` or a
    completed ``LlmResponse`` — see ``_SampleRecord`` for what it carries.
    ``timing`` is this sample's own ``_timing_for`` read; an all-``None``
    ``_TimingRecord`` when the caller passes none (no ``--metrics-db``
    configured).
    """
    if error is not None:
        return _SampleRecord(
            label=label,
            sample=index + 1,
            error=f"{type(error).__name__}: {error}",
            timing=timing or _TimingRecord(),
        )
    calls = response.message.tool_calls or []
    thinking = response.thinking or response.message.thinking
    return _SampleRecord(
        label=label,
        sample=index + 1,
        tool_calls=[
            _ToolCallRecord(name=call.function.name, arguments=call.function.arguments)
            for call in calls
        ],
        thinking=thinking,
        reply=response.message.content or "",
        timing=timing or _TimingRecord(),
    )


def _print_thinking(thinking: str | None, dump_full: bool) -> None:
    """Print one sample's thinking (#2884) — the FULL trace when
    ``--dump-thinking`` is set, needed to read WHY a sample engaged the way it
    did and not just which tool it picked first; otherwise the short tail every
    mode already printed, unchanged."""
    if dump_full and thinking:
        print(f"    thinking: {thinking}")
    else:
        print(f"    thinking-tail: {(thinking or '')[-300:]!r}")


def _timing_from_native(data: dict, wall_ms: int) -> _TimingRecord:
    """This sample's own usage/timing off Ollama's NATIVE ``/api/generate``
    response (#2884) — its own field names (nanoseconds), named here the
    same way ``usage_timing_from_response`` already names oMLX's and the chat
    endpoint's: a READ off whatever the response actually reports."""
    completion_tokens = data.get("eval_count")
    generation_ns = data.get("eval_duration")
    generation_ms = (
        round(generation_ns / _NANOSECONDS_PER_MS) if generation_ns is not None else None
    )
    prefill_ns = data.get("prompt_eval_duration")
    decode_tokens_per_sec = None
    if completion_tokens and generation_ms:
        decode_tokens_per_sec = round(completion_tokens / (generation_ms / 1000), 1)
    return _TimingRecord(
        wall_ms=wall_ms,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=completion_tokens,
        cached_tokens=data.get("prompt_eval_cached_count"),
        prefill_ms=round(prefill_ns / _NANOSECONDS_PER_MS) if prefill_ns is not None else None,
        generation_ms=generation_ms,
        decode_tokens_per_sec=decode_tokens_per_sec,
    )


async def _send_ollama_native(
    http_client: httpx.AsyncClient, api_url: str, model: str, prompt: str
) -> tuple[dict, int]:
    """One call on Ollama's native raw completions path (#2884) — see the
    module-level comment by ``_OLLAMA_NATIVE_GENERATE_PATH`` for why this
    deployment needs it instead of the OpenAI-compatible ``/v1/completions``
    every other column's raw-template transport uses. ``num_predict: -1``
    (unlimited) matches every other column here, which calls ``LlmClient.chat()``
    with no ``token_budget`` either — nothing in this comparison caps a draw."""
    started = time.monotonic()
    response = await http_client.post(
        f"{api_url.rstrip('/')}{_OLLAMA_NATIVE_GENERATE_PATH}",
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "num_predict": -1,
                "stop": list(raw_template.STOP_SEQUENCES),
            },
        },
        timeout=httpx.Timeout(300.0, connect=10.0),
    )
    wall_ms = round((time.monotonic() - started) * 1000)
    response.raise_for_status()
    return response.json(), wall_ms


async def _write_record(
    record: _SampleRecord, handle: Any | None, write_lock: asyncio.Lock | None
) -> None:
    """Persist one sample's record the moment it finishes (#2884 pause
    resilience), not batched at the end of the row — so a kill mid-row loses
    at most the samples still in flight, never the ones already done. Writes
    land in COMPLETION order rather than sample-number order (the file no
    longer needs sorting after ``gather``); each line still carries its own
    ``sample`` number, so a reader identifies it by that field, not by
    position. Guarded by ``write_lock`` since several samples can finish
    concurrently onto the same open handle."""
    if handle is None:
        return
    assert write_lock is not None
    async with write_lock:
        handle.write(record.model_dump_json() + "\n")
        handle.flush()


async def _replay_one_native(
    http_client: httpx.AsyncClient,
    api_url: str,
    model: str,
    prompt: str,
    label: str,
    index: int,
    samples: int,
    semaphore: asyncio.Semaphore,
    handle: Any | None = None,
    write_lock: asyncio.Lock | None = None,
    dump_thinking: bool = False,
) -> _SampleRecord:
    """One sample on the Ollama-native raw path — the native-transport sibling
    of ``_replay_one``, producing the SAME record shape (#2884) so the
    classification/tally code downstream never has to know which transport a
    sample came from."""
    async with semaphore:
        try:
            data, wall_ms = await _send_ollama_native(http_client, api_url, model, prompt)
        except httpx.HTTPError as error:
            print(f"  [{label} {index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            record = _SampleRecord(
                label=label, sample=index + 1, error=f"{type(error).__name__}: {error}"
            )
            await _write_record(record, handle, write_lock)
            return record
    text = data.get("response", "")
    timing = _timing_from_native(data, wall_ms)
    try:
        thinking, tool_calls, content = raw_template.parse_completion(text)
    except LlmToolParseError as error:
        print(f"  [{label} {index + 1}/{samples}] PARSE ERROR: {error}")
        record = _SampleRecord(
            label=label,
            sample=index + 1,
            error=f"{type(error).__name__}: {error}",
            timing=timing,
        )
        await _write_record(record, handle, write_lock)
        return record
    calls = tool_calls or []
    names = [call.function.name for call in calls]
    print(f"  [{label} {index + 1}/{samples}] first={(names[0] if names else '—')} tools={names}")
    _print_thinking(thinking, dump_thinking)
    print(f"    reply: {content!r}")
    print(f"    timing: {timing}")
    record = _SampleRecord(
        label=label,
        sample=index + 1,
        tool_calls=[
            _ToolCallRecord(name=call.function.name, arguments=call.function.arguments)
            for call in calls
        ],
        thinking=thinking,
        reply=content,
        timing=timing,
    )
    await _write_record(record, handle, write_lock)
    return record


async def _replay_native(
    api_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    samples: int,
    label: str,
    *,
    concurrency: int = 1,
    jsonl_path: str | None = None,
    dump_thinking: bool = False,
) -> _Tally:
    """The native-transport sibling of ``_replay`` — same tally/JSONL shape,
    one rendered prompt (the vendored template, #2884) sent through
    Ollama's native endpoint per sample instead of ``LlmClient.chat()``."""
    tally = _Tally()
    prompt = raw_template.render_prompt(messages, tools, thinking_enabled=True)
    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(open(jsonl_path, "w")) if jsonl_path else None
        write_lock = asyncio.Lock() if handle is not None else None
        async with httpx.AsyncClient() as http_client:
            semaphore = asyncio.Semaphore(max(1, concurrency))
            tasks = [
                _replay_one_native(
                    http_client,
                    api_url,
                    model,
                    prompt,
                    label,
                    index,
                    samples,
                    semaphore,
                    handle,
                    write_lock,
                    dump_thinking,
                )
                for index in range(samples)
            ]
            records = await asyncio.gather(*tasks)
    for record in records:
        if record.error is not None:
            tally.record_error()
        else:
            first = record.tool_calls[0].name if record.tool_calls else None
            tally.record(first, record.thinking)
    return tally


async def _replay_one(
    client: LlmClient,
    typed_messages: list[LlmMessage],
    tools: list[dict] | None,
    label: str,
    index: int,
    samples: int,
    semaphore: asyncio.Semaphore,
    metrics_db: Database | None,
    handle: Any | None = None,
    write_lock: asyncio.Lock | None = None,
    dump_thinking: bool = False,
) -> _SampleRecord:
    """One sample's call, gated by ``semaphore`` (``--concurrency``) so a batch of
    samples can run several draws at once against a runner that can take it, while
    still printing and tallying in the SAME shape as the serial path."""
    async with semaphore:
        try:
            response = await client.chat(
                typed_messages,
                tools=tools,
                call_link=CallLink.new_top(),
                prompt_type=f"replay-{label}",
            )
        except LlmError as error:
            print(f"  [{label} {index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            record = _sample_record(label, index, None, error)
            await _write_record(record, handle, write_lock)
            return record
    calls = response.message.tool_calls or []
    names = [call.function.name for call in calls]
    first = names[0] if names else None
    thinking = response.thinking or response.message.thinking
    reply = response.message.content or ""
    timing = _timing_for(metrics_db, response.prompt_log_id)
    print(f"  [{label} {index + 1}/{samples}] first={first or '—'} tools={names}")
    _print_thinking(thinking, dump_thinking)
    print(f"    reply: {reply!r}")
    print(f"    timing: {timing}")
    record = _sample_record(label, index, response, None, timing)
    await _write_record(record, handle, write_lock)
    return record


async def _replay(
    client: LlmClient,
    messages: list[dict],
    tools: list[dict] | None,
    samples: int,
    label: str,
    *,
    concurrency: int = 1,
    jsonl_path: str | None = None,
    metrics_db: Database | None = None,
    dump_thinking: bool = False,
) -> _Tally:
    tally = _Tally()
    # The one conversion at this call site (#2563): this script works in the
    # internal dict shape throughout (loaded verbatim off the promptlog, recomposed
    # by ``_compose``) — ``from_dict`` lifts it to typed history only here, right
    # before the one place it reaches ``LlmClient.chat()``.
    typed_messages = [LlmMessage.from_dict(message) for message in messages]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(open(jsonl_path, "w")) if jsonl_path else None
        write_lock = asyncio.Lock() if handle is not None else None
        tasks = [
            _replay_one(
                client,
                typed_messages,
                tools,
                label,
                index,
                samples,
                semaphore,
                metrics_db,
                handle,
                write_lock,
                dump_thinking,
            )
            for index in range(samples)
        ]
        records = await asyncio.gather(*tasks)
    # Persisted as each sample FINISHES (#2884 — ``_write_record``), not
    # batched here — this loop only tallies, so order doesn't matter: `_Tally`
    # is a sum, indifferent to arrival order.
    for record in records:
        if record.error is not None:
            tally.record_error()
        else:
            first = record.tool_calls[0].name if record.tool_calls else None
            tally.record(first, record.thinking)
    return tally


def _open_metrics_db(path: str | None) -> Database:
    """A throwaway, writable ``Database`` (#2884 performance column) this
    run's calls log their per-call usage/timing to — never the read-only prod
    DB this tool reads its prompt from. A fresh tempfile path when the caller
    names none, so every invocation gets its own and nothing accumulates.

    ``mkstemp`` (not the race-prone, deprecated ``mktemp``) just to MINT the
    path — the fd closes immediately, since ``Database`` opens the file
    itself via SQLAlchemy."""
    db_path = path
    if db_path is None:
        descriptor, db_path = tempfile.mkstemp(suffix=".db", prefix="replay-metrics-")
        os.close(descriptor)
    metrics_db = Database(db_path)
    metrics_db.create_tables()
    print(f"metrics db: {db_path}")
    return metrics_db


def _variants_from_spec(spec_json: str) -> list[ReplayVariant]:
    """The A/B spec's variants, validated as a whole — unknown keys, a wrong
    shape, or an empty list are all refused before anything is loaded or sent.
    The spec arrives INLINE (the make recipe reads the ``VARIANTS`` file host-side
    and passes its contents here — a host path would not be visible inside the
    container the recipe runs the tool in)."""
    try:
        raw = json.loads(spec_json)
    except json.JSONDecodeError as error:
        raise SystemExit(f"the variants spec is not valid JSON: {error}") from error
    try:
        variants = _VARIANT_SPEC_ADAPTER.validate_python(raw)
    except ValidationError as error:
        raise SystemExit(
            "the variants spec is not a list of "
            '{"name": ..., "substitutions"?: [{"pattern": ..., "replacement": ...}], '
            '"system_prompt"?: ..., "system_prompt_append"?: ..., '
            '"last_system_append"?: ..., "last_tool_result"?: ..., '
            '"user_message"?: ..., "thinking"?: ...} entries: '
            f"{error}\nAn entry with no transform is the baseline arm."
        ) from error
    return variants


async def _main(args: argparse.Namespace) -> None:
    # The ONE resolution for the whole invocation (#3001): the model and its pin come
    # from `EVAL_MODELS` through the roster (`resolve_from_env`) — never from a
    # hand-set ``LLM_MODEL``/``LLM_PROVIDER``, which are only checked against the
    # roster and otherwise refused up front.
    entry = resolve_from_env()
    spec = os.environ.get(REPLAY_VARIANTS_SPEC_ENV)
    if spec:
        if args.mode != "plain" or args.body_file or args.nudge:
            raise SystemExit(
                "the A/B spec (REPLAY_VARIANTS_SPEC) is the whole run — it replaces "
                "--mode/--body-file/--nudge"
            )
        variants = _variants_from_spec(spec)
        await replay_variants(
            args.id,
            variants,
            args.samples,
            db_path=args.db,
            jsonl_base=args.jsonl,
            api_url=os.environ.get("LLM_API_URL", "http://host.docker.internal:11434"),
            api_key=os.environ.get("LLM_API_KEY") or "not-needed",
            concurrency=args.concurrency,
        )
        return
    if args.mode == "verbatim-request":
        await _replay_verbatim_request(args.db, args.id, args.samples, entry)
        return
    messages, tools = _load_prompt(args.db, args.id)
    if args.mode == "ollama-native":
        # No ``LlmClient``, no metrics db: this mode never goes through
        # ``LlmClient.chat()`` at all (see ``_replay_native``'s docstring) —
        # the collector placement split doesn't apply either, same as 'plain'.
        api_url = os.environ.get("LLM_API_URL", "http://host.docker.internal:11434")
        model = entry.model
        (
            await _replay_native(
                api_url,
                model,
                messages,
                tools,
                args.samples,
                "verbatim",
                concurrency=args.concurrency,
                jsonl_path=args.jsonl,
                dump_thinking=args.dump_thinking,
            )
        ).report("verbatim")
        return
    metrics_db = _open_metrics_db(args.metrics_db)
    client = LlmClient(
        api_url=os.environ.get("LLM_API_URL", "http://host.docker.internal:11434"),
        model=entry.model,
        api_key=os.environ.get("LLM_API_KEY") or "not-needed",
        # Required so ``LlmClient._log_call`` actually writes a ``promptlog`` row
        # at all (it's a no-op with ``db=None``) — without this, every sample's
        # ``_timing_for`` read finds no row and silently returns an all-``None``
        # ``_TimingRecord`` (#2884 perf comparison: this was missing and every
        # oMLX timing came back empty).
        db=metrics_db,
        # The resolved entry's own pin — ``None`` when the entry names no provider
        # (a direct local endpoint, which has never heard of the routing field and
        # ignores it). Never a hand-set ``LLM_PROVIDER`` (#3001).
        provider_pin=ProviderPin.pin(entry.provider),
        max_retries=3,
        retry_delay=1.0,
        timeout=120.0,
    )
    if args.mode == "plain" or not _is_collector_prompt(messages):
        # Non-collector (e.g. a chat turn): replay the verbatim messages as-is —
        # the placement split doesn't apply.  ``no_tool`` in the tally is the bail
        # signal here (the model narrated "done" instead of calling a tool).
        (
            await _replay(
                client,
                messages,
                tools,
                args.samples,
                "verbatim",
                concurrency=args.concurrency,
                jsonl_path=args.jsonl,
                metrics_db=metrics_db,
                dump_thinking=args.dump_thinking,
            )
        ).report("verbatim")
        return
    date, body, runtime, rest = _parse_system(messages)
    if args.body_file:
        with open(args.body_file) as handle:
            body = handle.read().strip()
        print(f"swapped collector body from {args.body_file} ({len(body)} chars)")
    placements = ("system", "user") if args.mode == "both" else (args.mode,)
    for placement in placements:
        composed = _compose(date, body, runtime, rest, placement=placement, nudge=args.nudge)
        label = f"{placement}+nudge" if (args.nudge and placement == "system") else placement
        jsonl_path = (
            f"{args.jsonl.rsplit('.', 1)[0]}-{label}.jsonl"
            if (args.jsonl and len(placements) > 1)
            else args.jsonl
        )
        (
            await _replay(
                client,
                composed,
                tools,
                args.samples,
                label,
                concurrency=args.concurrency,
                jsonl_path=jsonl_path,
                metrics_db=metrics_db,
                dump_thinking=args.dump_thinking,
            )
        ).report(label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=_DEFAULT_DB, help="SQLite DB with the promptlog")
    parser.add_argument("--id", type=int, required=True, help="promptlog row id to replay")
    parser.add_argument("--samples", type=int, default=12, help="replays per placement")
    parser.add_argument(
        "--mode",
        choices=("system", "user", "both", "plain", "verbatim-request", "ollama-native"),
        default="both",
        help="where the collector body goes: system (current shape) / user (split) / both; "
        "'plain' forces a verbatim replay of messages/tools, re-rendered through today's "
        "code (auto-selected for non-collector prompts); 'verbatim-request' (#2875) sends "
        "the row's own stored EXACT request unchanged — no re-render at all, see the "
        "module docstring; 'ollama-native' (#2884) is 'plain' with the SAME rendered "
        "prompt but sent through Ollama's native /api/generate (raw: true) instead of "
        "LlmClient.chat() — for a deployment where /v1/completions doesn't serve raw "
        "completions faithfully (see the module-level comment by "
        "_OLLAMA_NATIVE_GENERATE_PATH)",
    )
    parser.add_argument(
        "--body-file",
        help="local file whose contents replace the collector body (hold content "
        "constant while varying format — e.g. a numbered rewrite); never committed",
    )
    parser.add_argument(
        "--nudge",
        help="fill the otherwise-empty user turn with this generic string while the "
        "body stays in the system prompt (only applies to system placement)",
    )
    parser.add_argument(
        "--metrics-db",
        help="path for the throwaway, WRITABLE db each call's per-call usage/timing "
        "(#2884 performance column) is logged to and read back from — never "
        "the read-only --db this tool loads the prompt from; defaults to a fresh "
        "tempfile per run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many samples to run in flight at once against the runner (default "
        "1 — serial); size this to what the target endpoint actually sustains, "
        "never to the sample count",
    )
    parser.add_argument(
        "--jsonl",
        help="write one JSON object per sample (tool calls with arguments, the FULL "
        "thinking, the FULL reply — no truncation) to this path, one line per "
        "sample in sample-number order; with --mode both this is suffixed per "
        "placement (e.g. foo.jsonl -> foo-system.jsonl, foo-user.jsonl)",
    )
    parser.add_argument(
        "--dump-thinking",
        action="store_true",
        help="print each sample's FULL thinking to stdout, not just the 300-char "
        "tail every mode prints by default — needed to read WHY a sample engaged "
        "the way it did, not just which tool it picked first. The JSONL record "
        "(--jsonl) always carries the full thinking regardless of this flag",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
