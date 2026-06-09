"""Skills-extractor suite — the background collector that grows/tunes skills.

Runs the real SKILLS_EXTRACTION_PROMPT (migration 0043) + the collector
runtime rules (collector.py) against synthetic conversation logs, and
checks the entry-level outcome on the skills collection:

  teach       — explicit "from now on when I say X, do Y" → collection_write
  correct-sub — "stop telling me X"                       → update_entry (refine)
  correct-scope — "only do X for Y"                       → update_entry (narrow)
  deprecate   — "drop that rule entirely"                 → collection_delete_entry
  lift        — a one-off then taught as a rule           → collection_write
  quiet       — normal chat, no teaching                  → no write, done()

Topics are generic (recipes, a wiki-linking research habit) — no user data.
"""
from __future__ import annotations

import json

from scripts.prompt_validation._harness import (
    CaseResult,
    Harness,
    class_attr,
    extract_tool_calls,
    load_seed_skills,
    run_samples,
)
from scripts.prompt_validation._harness import PENNY_PKG

NAME = "skills_extractor"

_, EXTRACTION_PROMPT = load_seed_skills()
RUNTIME_RULES = class_attr(PENNY_PKG / "agents" / "collector.py", "Collector", "_RUNTIME_RULES")

SYSTEM = (
    "You are the collector for the `skills` collection.\n"
    "Description: Workflow patterns — how to compose tools to satisfy user intents\n\n"
    f"{EXTRACTION_PROMPT}\n\n{RUNTIME_RULES}"
)


def _tool(name: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": props, "required": required}}}

TOOLS = [
    _tool("log_read_next", {"memory": {"type": "string"}, "cap": {"type": "integer"}}, ["memory"]),
    _tool("read_similar", {"memory": {"type": "string"}, "anchor": {"type": "string"},
                           "k": {"type": "integer"}}, ["memory", "anchor"]),
    _tool("collection_write", {"memory": {"type": "string"}, "entries": {"type": "array", "items": {
        "type": "object", "properties": {"key": {"type": "string"}, "content": {"type": "string"}},
        "required": ["key", "content"]}}}, ["memory", "entries"]),
    _tool("update_entry", {"memory": {"type": "string"}, "key": {"type": "string"},
                           "content": {"type": "string"}}, ["memory", "key", "content"]),
    _tool("collection_delete_entry", {"memory": {"type": "string"}, "key": {"type": "string"}},
          ["memory", "key"]),
    _tool("send_message", {"content": {"type": "string"}}, ["content"]),
    _tool("done", {"success": {"type": "boolean"}, "summary": {"type": "string"}},
          ["success", "summary"]),
]

# Existing skills the read_similar probe can surface (correction/deprecation).
EXISTING = {
    "recipes-include-prep-and-difficulty": (
        "TRIGGER\nUser asks about recipes.\n\nSTEPS\n1. browse() for the recipe.\n"
        "2. Answer with ingredients, prep time, AND difficulty level."
    ),
    "research-with-wikipedia-link": (
        "TRIGGER\nUser asks you to research a topic.\n\nSTEPS\n1. browse() for the topic.\n"
        "2. Always include a Wikipedia link alongside other sources.\n3. Answer."
    ),
}

# (id, user_msgs, penny_msgs, expected, existing_key)
CASES = [
    ("teach",
     ["what's a good pasta recipe",
      "from now on when i ask about recipes, always include the prep time and difficulty"],
     [], "write", None),
    ("correct-sub",
     ["what's a good carbonara recipe",
      "wait, when i ask about recipes, stop telling me the difficulty — just give me prep time"],
     [], "update", "recipes-include-prep-and-difficulty"),
    ("correct-scope",
     ["research the latest wireless earbuds for me",
      "you don't need wikipedia for product comparisons — only do the wiki link for historical topics"],
     [], "update", "research-with-wikipedia-link"),
    ("deprecate",
     ["on second thought, drop the recipes skill entirely — never mind tracking that, just delete it"],
     [], "delete", "recipes-include-prep-and-difficulty"),
    ("lift",
     ["find me a good thai place near home",
      "from now on whenever i ask about restaurants, always include the price range"],
     ["Sukhumvit Garden — solid pad thai, ~10 min away."], "write", None),
    ("quiet",
     ["what's the weather today?", "thanks!"],
     ["Sunny and 72°F!", "anytime — let me know if you need anything else"], "no_op", None),
]


def _run_cycle(h: Harness, user_msgs, penny_msgs, existing_key, max_steps=10):
    def serve(name, args):
        if name == "log_read_next":
            msgs = user_msgs if args.get("memory") == "user-messages" else penny_msgs
            return "\n".join(f"- {m}" for m in msgs) if msgs else "(no entries)"
        if name == "read_similar":
            if args.get("memory") == "skills" and existing_key:
                return f"- [{existing_key}] {EXISTING[existing_key]}"
            return "(no entries)"
        if name in ("collection_write", "update_entry"):
            return "ok"
        if name == "collection_delete_entry":
            return f"Deleted '{args.get('key')}'."
        if name == "send_message":
            return "sent"
        return "ok"

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Begin cycle."}]
    writes, updates, deletes = [], [], []
    done = None
    for _ in range(max_steps):
        msg = h.chat(messages, tools=TOOLS)
        calls = extract_tool_calls(msg)
        if not calls:
            break
        for c in calls:
            if c["name"] == "collection_write":
                writes.append(c["args"])
            elif c["name"] == "update_entry":
                updates.append(c["args"])
            elif c["name"] == "collection_delete_entry":
                deletes.append(c["args"])
            elif c["name"] == "done":
                done = c["args"]
        if done is not None:
            break
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in calls]})
        for c in calls:
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": serve(c["name"], c["args"])})
    return writes, updates, deletes, done


def _score(expected, existing_key, writes, updates, deletes, done) -> list[str]:
    f = []
    if expected == "write":
        if not any(w.get("memory") == "skills" for w in writes):
            f.append("expected collection_write to skills, none happened")
        else:
            for w in writes:
                for e in w.get("entries", []):
                    if "TRIGGER" not in (e.get("content") or "").upper():
                        f.append(f"written skill {e.get('key')!r} lacks TRIGGER/STEPS shape")
    elif expected == "update":
        if not any(u.get("memory") == "skills" and u.get("key") == existing_key for u in updates):
            if writes:
                f.append(f"wrote a new skill instead of update_entry on {existing_key!r} (fragments)")
            else:
                f.append(f"correction lost — no update_entry on {existing_key!r}")
    elif expected == "delete":
        if not any(d.get("memory") == "skills" and d.get("key") == existing_key for d in deletes):
            f.append(f"deprecation lost — no delete of {existing_key!r}")
    elif expected == "no_op":
        if writes:
            f.append(f"wrote {len(writes)} skill(s) on a quiet cycle")
        if updates or deletes:
            f.append("mutated skills on a quiet cycle")
        if not done:
            f.append("no done() call")
    return f


def run(h: Harness, samples: int, only: str | None = None) -> list[CaseResult]:
    results: list[CaseResult] = []
    for cid, umsgs, pmsgs, expected, existing_key in CASES:
        if only and only != cid:
            continue

        def one(umsgs=umsgs, pmsgs=pmsgs, expected=expected, existing_key=existing_key, cid=cid):
            w, u, d, done = _run_cycle(h, umsgs, pmsgs, existing_key)
            fails = _score(expected, existing_key, w, u, d, done)
            return CaseResult(cid, not fails, fails)

        results.extend(run_samples(f"{NAME}:{cid}", samples, one))
    return results
