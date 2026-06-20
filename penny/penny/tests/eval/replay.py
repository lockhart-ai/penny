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

Run inside the penny container (the prod DB is mounted at /penny/data):

    docker compose run --rm --no-deps penny \
        env LLM_API_URL=http://host.docker.internal:11434 LLM_MODEL=gpt-oss:20b \
        python -m penny.tests.eval.replay --id 138385 --samples 12 --mode both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass, field

from penny.llm.client import LlmClient

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

    def report(self, label: str) -> None:
        bail_rate = 100 * self.bailed / self.samples if self.samples else 0
        avg_engaged = self.engaged_thinking_chars / self.engaged if self.engaged else 0
        print(
            f"\n[{label}] {self.samples} samples · "
            f"BAILED {self.bailed} ({bail_rate:.0f}%) · engaged {self.engaged} · "
            f"no-tool {self.no_tool} · {avg_engaged:.0f} avg reasoning chars when engaged · "
            f"first-tools {self.first_tools}"
        )


def _load_prompt(db_path: str, prompt_id: int) -> tuple[list[dict], list[dict] | None]:
    """Read the verbatim messages + tools off one promptlog row."""
    connection = sqlite3.connect(db_path)
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
    for index in range(samples):
        response = await client.chat(messages, tools=tools, prompt_type=f"replay-{label}")
        calls = response.message.tool_calls or []
        names = [call.function.name for call in calls]
        first = names[0] if names else None
        thinking = response.thinking or response.message.thinking
        tally.record(first, thinking)
        print(f"  [{label} {index + 1}/{samples}] first={first or '—'} tools={names}")
    return tally


async def _main(args: argparse.Namespace) -> None:
    messages, tools = _load_prompt(args.db, args.id)
    date, body, runtime, rest = _parse_system(messages)
    if args.body_file:
        with open(args.body_file) as handle:
            body = handle.read().strip()
        print(f"swapped collector body from {args.body_file} ({len(body)} chars)")
    client = LlmClient(
        api_url=os.environ.get("LLM_API_URL", "http://host.docker.internal:11434"),
        model=os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        max_retries=3,
        retry_delay=1.0,
        timeout=120.0,
    )
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
        choices=("system", "user", "both"),
        default="both",
        help="where the collector body goes: system (current shape) / user (split) / both",
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
