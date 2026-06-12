"""Shared infrastructure for the prompt-validation harness.

These are ad-hoc validation runs, NOT pytest — they need a live LLM
(gpt-oss) and embedding model (embeddinggemma) on Ollama and take
minutes, so they're invoked by hand, not by ``make check``.  See
README.md.

Everything here runs on the host (no Docker): it reads the real prompt
text out of the penny source via AST, talks to Ollama through the
OpenAI-compatible endpoint for both chat and embeddings, and imports the
stdlib-only ``similarity`` package for cosine/tokenisation.  It does NOT
import the ``penny`` package, so it stays runnable without the full dep
set installed on the host.

Run from the repo root, e.g.::

    PYTHONPATH=. uv run --python 3.12 --with openai \
        python scripts/prompt_validation/lifecycle.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
PENNY_PKG = REPO_ROOT / "penny" / "penny"


# ── Reading real prompt text from source (no import of penny) ───────────────


def class_attr(path: Path, class_name: str, attr: str, globs: dict | None = None) -> Any:
    """Extract a class attribute literal from a source file via AST.

    Pure literals use ``literal_eval``; f-strings / concatenations are
    eval'd with a caller-supplied namespace (to stub module constants
    the assignment references, e.g. ``_RECALL_MODES``).
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == attr:
                            try:
                                return ast.literal_eval(item.value)
                            except (ValueError, SyntaxError):
                                return eval(ast.unparse(item.value), globs or {}, {})
    raise RuntimeError(f"{class_name}.{attr} not found in {path}")


def conversation_prompt() -> str:
    return class_attr(PENNY_PKG / "prompts.py", "Prompt", "CONVERSATION_PROMPT")


def penny_identity() -> str:
    return class_attr(PENNY_PKG / "prompts.py", "Prompt", "PENNY_IDENTITY")


# Mirror the two production enums (memory_store.Inclusion / RecallMode) so
# the tool descriptions' f-strings and ``[m.value for m in ...]`` enum
# comprehensions evaluate at load time without importing the penny package.
_INCLUSION_VALUES = ["always", "relevant", "never"]
_RECALL_MODE_VALUES = ["recent", "relevant", "all"]


class _FakeEnum:
    """Iterable stub so ``[m.value for m in SomeEnum]`` evals at load time."""

    def __init__(self, values: list[str]):
        self._values = values

    def __iter__(self):
        from collections import namedtuple

        member = namedtuple("Member", ["value"])
        return iter(member(v) for v in self._values)


def load_tool(class_name: str) -> dict:
    """Build the OpenAI tool dict for a memory_tools Tool subclass from source.

    Reads the real ``name`` / ``description`` / ``parameters`` so suites test
    the production tool surface, not a hand-copied paraphrase.
    """
    path = PENNY_PKG / "tools" / "memory_tools.py"
    globs = {
        "_INCLUSION_MODES": ", ".join(_INCLUSION_VALUES),
        "_RECALL_MODES": ", ".join(_RECALL_MODE_VALUES),
        "Inclusion": _FakeEnum(_INCLUSION_VALUES),
        "RecallMode": _FakeEnum(_RECALL_MODE_VALUES),
    }
    return {
        "type": "function",
        "function": {
            "name": class_attr(path, class_name, "name", globs),
            "description": class_attr(path, class_name, "description", globs),
            "parameters": class_attr(path, class_name, "parameters", globs),
        },
    }


def browse_tool() -> dict:
    """Minimal browse tool (production's is dynamic; this matches its shape)."""
    return {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Look things up. Pass up to 3 queries and/or URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
                "required": ["queries"],
            },
        },
    }


def render_skills_recall(seed_skills: list[tuple[str, str]]) -> str:
    """Render seed skills as a recall block (mirrors chat.py:_format_recall_section).

    Injects all skills — the harness isolates skill *followability* from
    *retrievability*; production embedding recall surfaces the matching subset.
    """
    lines = ["## Recall context", "", "### skills",
             "Workflow patterns — how to compose tools to satisfy user intents"]
    for key, content in seed_skills:
        lines.append("")
        lines.append(f"#### [{key}] · 2026-06-01 00:00")
        lines.append(content.rstrip())
    return "\n".join(lines)


def _load_migration_module(filename: str):
    """Import a migration module directly — pure data, no DB side effects."""
    path = PENNY_PKG / "database" / "migrations" / filename
    spec = importlib.util.spec_from_file_location(f"_mig_{filename[:4]}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_seed_skills() -> tuple[list[tuple[str, str]], str]:
    """Load (SEED_SKILLS, SKILLS_EXTRACTION_PROMPT) at their migrated end-state.

    0043 seeds the skills; 0045 rewrites several for the inclusion/recall
    split.  Overlaying the 0045 replacements reproduces what a fully
    migrated database contains, so suites test the skills production
    actually serves — not the original seed text.
    """
    seed_mod = _load_migration_module("0043_seed_skills_collection.py")
    fix_mod = _load_migration_module("0045_update_skills_for_inclusion_split.py")
    skills = [
        (key, fix_mod.REPLACEMENTS.get(key, content)) for key, content in seed_mod.SEED_SKILLS
    ]
    return skills, seed_mod.SKILLS_EXTRACTION_PROMPT


# ── LLM access (Ollama via OpenAI-compatible endpoint) ──────────────────────


def _client() -> OpenAI:
    base = os.getenv("LLM_API_URL", "http://localhost:11434").rstrip("/") + "/v1"
    return OpenAI(base_url=base, api_key=os.getenv("LLM_API_KEY", "not-needed"))


def chat_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-oss:20b")


def embed_model() -> str:
    return os.getenv("LLM_EMBEDDING_MODEL", "embeddinggemma")


@dataclass
class CallMetrics:
    """Timing + token usage aggregated across LLM calls."""

    wall_s: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)

    def record(self, wall_s: float, usage: Any) -> None:
        self.wall_s.append(wall_s)
        if usage is not None:
            self.prompt_tokens.append(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens.append(getattr(usage, "completion_tokens", 0) or 0)

    def _avg(self, xs: list) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def summary(self) -> str:
        n = len(self.wall_s)
        if not n:
            return "(no calls)"
        avg_wall = self._avg(self.wall_s)
        avg_in = self._avg(self.prompt_tokens)
        avg_out = self._avg(self.completion_tokens)
        gen_tps = (sum(self.completion_tokens) / sum(self.wall_s)) if sum(self.wall_s) else 0.0
        return (
            f"{n} calls | avg {avg_wall:.1f}s | in {avg_in:.0f} tok | "
            f"out {avg_out:.0f} tok | gen {gen_tps:.1f} tok/s"
        )


class Harness:
    """One harness instance: client + shared call metrics."""

    def __init__(self, temperature: float = 0.7) -> None:
        self.client = _client()
        self.model = chat_model()
        self.temperature = temperature
        self.metrics = CallMetrics()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """One chat completion; records timing + usage. Returns the message."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        t0 = time.monotonic()
        resp = self.client.chat.completions.create(**kwargs)
        self.metrics.record(time.monotonic() - t0, getattr(resp, "usage", None))
        return resp.choices[0].message

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via the OpenAI-compatible embeddings endpoint."""
        resp = self.client.embeddings.create(model=embed_model(), input=texts)
        return [d.embedding for d in resp.data]


def extract_tool_calls(msg: Any) -> list[dict]:
    """Normalize a message's tool calls into [{name, args, id}]."""
    out = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {"_unparseable": tc.function.arguments}
        out.append({"name": tc.function.name, "args": args, "id": tc.id})
    return out


@dataclass
class Conversation:
    """Outcome of a multi-turn agentic run."""

    turns: list[dict]  # [{calls: [{name,args}], text}]
    terminal_call: dict | None  # first {name, args} the model committed to
    final_text: str

    def all_calls(self) -> list[dict]:
        return [c for t in self.turns for c in t["calls"]]


# Sentinel: a tool_result function returns this to mark a call terminal
# (the harness captures it and stops the loop, as production would hand
# off to the real side-effecting tool).
TERMINAL = object()


def converse(
    h: Harness,
    system: str,
    user_msg: str,
    tools: list[dict],
    tool_result: Callable[[str, dict], Any],
    max_steps: int = 6,
) -> Conversation:
    """Run the agentic loop, serving synthetic results for read-style tool
    calls and stopping at the first terminal call.

    ``tool_result(name, args)`` returns a string (a synthetic tool result —
    the loop continues) or the ``TERMINAL`` sentinel (capture the call and
    stop).  A plain text reply with no tool calls also stops the loop.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    turns: list[dict] = []
    terminal_call: dict | None = None
    final_text = ""

    for _ in range(max_steps):
        msg = h.chat(messages, tools=tools)
        calls = extract_tool_calls(msg)
        text = msg.content or ""
        turns.append({"calls": [{"name": c["name"], "args": c["args"]} for c in calls], "text": text})

        if not calls:
            final_text = text
            break

        # Serve each call exactly ONCE — stateful tool_result functions
        # (dedup sets, served-page flags) break if invoked twice per call,
        # because the model sees the SECOND invocation's value.  This bug
        # silently fed repeat-rejections instead of pages to every suite
        # case with a stateful serve, invalidating their measurements.
        served = [(c, tool_result(c["name"], c["args"])) for c in calls]
        terminal = next((c for c, result in served if result is TERMINAL), None)
        if terminal is not None:
            terminal_call = {"name": terminal["name"], "args": terminal["args"]}
            break

        # All read-style: append the served results and continue.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                for c in calls
            ],
        })
        for c, result in served:
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"],
                "content": str(result),
            })

    return Conversation(turns, terminal_call, final_text)


# ── Reporting ───────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    fails: list[str]


def report(results: list[CaseResult], metrics: CallMetrics | None = None) -> int:
    """Print a per-case summary table; return the failure count."""
    print(f"\n{'=' * 78}\n# Summary\n{'=' * 78}\n")
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)
    n_fail = 0
    print(f"{'case':32}  pass-rate  failure-modes")
    print(f"{'-' * 32}  ---------  -------------------------------------------")
    for case_id, rs in by_case.items():
        npass = sum(1 for r in rs if r.passed)
        n_fail += sum(1 for r in rs if not r.passed)
        fails = Counter(f for r in rs if not r.passed for f in r.fails)
        fail_str = "; ".join(f"{c}× {m[:54]}" for m, c in fails.most_common())
        print(f"{case_id:32}  {npass}/{len(rs):<7}  {fail_str}")
    total = len(results)
    print(f"\nTOTAL: {total - n_fail}/{total}")
    if metrics is not None:
        print(f"PERF:  {metrics.summary()}")
    return n_fail


def run_samples(
    case_id: str,
    n: int,
    run_fn: Callable[[], CaseResult],
    verbose: bool = True,
) -> list[CaseResult]:
    """Run a single case ``n`` times, printing per-sample status."""
    if verbose:
        print(f"\n{'=' * 78}\n## {case_id}  (×{n})\n{'=' * 78}")
    out = []
    for i in range(n):
        try:
            r = run_fn()
        except Exception as e:  # noqa: BLE001 — harness should not die on one case
            r = CaseResult(case_id, False, [f"runtime error: {type(e).__name__}: {e}"])
        out.append(r)
        if verbose:
            status = "✓" if r.passed else "✗"
            tail = "" if r.passed else f" — {'; '.join(r.fails)[:90]}"
            print(f"  [{i + 1}] {status}{tail}")
    return out
