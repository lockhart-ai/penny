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
``openai.AsyncOpenAI`` of its own, overriding only ``model`` (``--model``/``LLM_MODEL``)
and the provider pin inside ``extra_body`` (``--provider``/``LLM_PROVIDER``) when
retargeting a different runner than the one that originally served it. Everything
else — messages/prompt, tools, every sampling field, stop sequences, the token
budget — rides unchanged, proving the bytes match rather than approximating them.

Run inside the penny container (the prod DB is mounted at /penny/data):

    docker compose run --rm --no-deps penny \
        env LLM_API_URL=http://host.docker.internal:11434 LLM_MODEL=gpt-oss:20b \
        python -m penny.tests.eval.utils.replay --id 138385 --samples 12 --mode both

    docker compose run --rm --no-deps penny \
        env LLM_API_URL=http://host.docker.internal:11434 LLM_MODEL=gpt-oss:20b \
        python -m penny.tests.eval.utils.replay --id 138385 --samples 5 \
            --mode verbatim-request
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import zlib
from dataclasses import dataclass, field

import openai

from penny.llm import raw_template
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.models import (
    PROVIDER_REQUEST_FIELD,
    LlmError,
    LlmMessage,
    PromptTransport,
    ProviderPin,
)

_DEFAULT_DB = "/penny/data/penny/penny.db"
_RUNTIME_MARKER = "## Runtime rules"
_COLLECTOR_MARKER = "You are the collector"


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


def _retargeted_body(sent_request: dict, model: str | None, provider: str | None) -> dict:
    """Override ONLY what differs about the TARGET RUNNER (#2875) — ``model`` and
    the provider pin inside ``extra_body`` — leaving the prompt/messages, tools,
    every sampling field, and the stop sequences exactly as recorded. ``None``
    for either argument changes nothing (replay against the same runner that
    served the original call)."""
    body = dict(sent_request["body"])
    if model:
        body["model"] = model
    if provider:
        extra_body = dict(body.get("extra_body") or {})
        extra_body[PROVIDER_REQUEST_FIELD] = ProviderPin(provider=provider).as_request_field()
        body["extra_body"] = extra_body
    return body


async def _send_verbatim_request(
    client: openai.AsyncOpenAI, transport: str, body: dict
) -> tuple[str | None, list[str], str]:
    """Send ``body`` straight through the bare SDK client — no ``LlmClient``, no
    template re-render, no sampling default of this tool's own — and parse the
    response the SAME way prod parses it on this transport. Returns
    ``(thinking, tool_call_names, reply_content)``.

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
        return thinking, [call.function.name for call in (tool_calls or [])], content
    raw = await client.chat.completions.create(**body)
    message = raw.choices[0].message
    extras = message.model_extra or {}
    thinking = extras.get("reasoning_content") or extras.get("reasoning")
    names = [call.function.name for call in (message.tool_calls or [])]
    return thinking, names, message.content or ""


async def _replay_verbatim_request(db_path: str, prompt_id: int, samples: int) -> None:
    """``--mode verbatim-request`` (#2875): replay the row's own exact wire
    request, unchanged, N times — see the module docstring for why this mode
    is not a re-render like every other one here.

    The target runner comes from the SAME env vars every other mode in this
    module reads: ``LLM_API_URL``/``LLM_API_KEY`` pick where this sends it
    (required — there is no runner to default to); ``LLM_MODEL``/
    ``LLM_PROVIDER`` are overrides, read with NO default, so leaving either
    unset replays against the EXACT model/pin the row's own request already
    carries rather than silently retargeting it.
    """
    sent_request = _load_request(db_path, prompt_id)
    transport = sent_request["transport"]
    body = _retargeted_body(
        sent_request, os.environ.get("LLM_MODEL"), os.environ.get("LLM_PROVIDER")
    )
    api_url = os.environ.get("LLM_API_URL", "http://host.docker.internal:11434")
    client = openai.AsyncOpenAI(
        base_url=f"{api_url.rstrip('/')}/v1",
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    )
    print(
        f"replaying id={prompt_id} transport={transport} model={body.get('model')} "
        f"against {api_url}"
    )
    for index in range(samples):
        try:
            thinking, names, content = await _send_verbatim_request(client, transport, body)
        except openai.OpenAIError as error:
            print(f"  [{index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            continue
        tail = (thinking or "")[-200:]
        print(
            f"  [{index + 1}/{samples}] tools={names or '—'} "
            f"thinking_tail={tail!r} reply={content[:200]!r}"
        )


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


async def _replay(
    client: LlmClient, messages: list[dict], tools: list[dict] | None, samples: int, label: str
) -> _Tally:
    tally = _Tally()
    # The one conversion at this call site (#2563): this script works in the
    # internal dict shape throughout (loaded verbatim off the promptlog, recomposed
    # by ``_compose``) — ``from_dict`` lifts it to typed history only here, right
    # before the one place it reaches ``LlmClient.chat()``.
    typed_messages = [LlmMessage.from_dict(message) for message in messages]
    for index in range(samples):
        try:
            response = await client.chat(
                typed_messages,
                tools=tools,
                call_link=CallLink.new_top(),
                prompt_type=f"replay-{label}",
            )
        except LlmError as error:
            tally.record_error()
            print(f"  [{label} {index + 1}/{samples}] ERROR {type(error).__name__}: {error}")
            continue
        calls = response.message.tool_calls or []
        names = [call.function.name for call in calls]
        first = names[0] if names else None
        thinking = response.thinking or response.message.thinking
        tally.record(first, thinking)
        reply = response.message.content or ""
        thinking_tail = (thinking or "")[-300:]
        print(f"  [{label} {index + 1}/{samples}] first={first or '—'} tools={names}")
        print(f"    thinking-tail: {thinking_tail!r}")
        print(f"    reply: {reply!r}")
    return tally


async def _main(args: argparse.Namespace) -> None:
    if args.mode == "verbatim-request":
        await _replay_verbatim_request(args.db, args.id, args.samples)
        return
    messages, tools = _load_prompt(args.db, args.id)
    client = LlmClient(
        api_url=os.environ.get("LLM_API_URL", "http://host.docker.internal:11434"),
        model=os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        api_key=os.environ.get("LLM_API_KEY") or "not-needed",
        # A remote gateway (OpenRouter) pin — ``None`` on a direct local endpoint,
        # which has never heard of the routing field and ignores it.
        provider_pin=ProviderPin.pin(os.environ.get("LLM_PROVIDER")),
        max_retries=3,
        retry_delay=1.0,
        timeout=120.0,
    )
    if args.mode == "plain" or not _is_collector_prompt(messages):
        # Non-collector (e.g. a chat turn): replay the verbatim messages as-is —
        # the placement split doesn't apply.  ``no_tool`` in the tally is the bail
        # signal here (the model narrated "done" instead of calling a tool).
        (await _replay(client, messages, tools, args.samples, "verbatim")).report("verbatim")
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
        (await _replay(client, composed, tools, args.samples, label)).report(label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=_DEFAULT_DB, help="SQLite DB with the promptlog")
    parser.add_argument("--id", type=int, required=True, help="promptlog row id to replay")
    parser.add_argument("--samples", type=int, default=12, help="replays per placement")
    parser.add_argument(
        "--mode",
        choices=("system", "user", "both", "plain", "verbatim-request"),
        default="both",
        help="where the collector body goes: system (current shape) / user (split) / both; "
        "'plain' forces a verbatim replay of messages/tools, re-rendered through today's "
        "code (auto-selected for non-collector prompts); 'verbatim-request' (#2875) sends "
        "the row's own stored EXACT request unchanged — no re-render at all, see the "
        "module docstring",
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
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
