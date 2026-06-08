"""Collection-authoring dry-run for the two-stage recall design.

Validates that gpt-oss, given a research request, authors a
collection_create call with the fields the new design needs:

  1. a CONTENT-REFLECTIVE description — the stage-1 routing anchor; lists
     the actual topics/items so future messages about them match it, NOT a
     generic "a collection of X".
  2. an ``inclusion`` flag set sensibly (``relevant`` for topical research).
  3. the existing fields still correct (browse-first extraction_prompt,
     interval, silent handling).

Candidate tool/skill text is defined INLINE here so we can iterate wording
before porting to source.  Once the pass-rate is solid, the candidate text
graduates into ``memory_tools.py`` / the skills migration.

Run from repo root::

    PYTHONPATH=. uv run --python 3.12 --with openai \
        python scripts/prompt_validation/lifecycle.py
"""
from __future__ import annotations

import os
import re

from scripts.prompt_validation._harness import (
    CaseResult,
    Harness,
    conversation_prompt,
    extract_tool_calls,
    penny_identity,
    report,
    run_samples,
)

# ── Candidate collection_create tool (new design) ───────────────────────────

CREATE_DESC = (
    "Create a keyed collection memory with a background collector.\n"
    "\n"
    "Fields:\n"
    "- name: unique slug (lowercase, hyphens).\n"
    "- description: REQUIRED, and it is a RETRIEVAL ANCHOR, not a label.\n"
    "  Write what the collection will CONTAIN — the actual topics, subjects,\n"
    "  and item types — phrased so a future user message about any of them\n"
    "  matches it. List specifics. Never a generic container phrase.\n"
    '    GOOD: "Heavier euro-style strategy board games and modern tabletop\n'
    '    classics: worker-placement, engine-builders, 2-player duels."\n'
    '    BAD:  "A collection of board game recommendations."\n'
    "  It is embedded and used to decide WHEN to surface this collection, so\n"
    "  concrete topic words are what make it work.\n"
    "- inclusion: how the collection participates in recall.\n"
    "    relevant (default) — surfaces only when the message matches the\n"
    "      description. Use for topical research collections.\n"
    "    always — surfaces on every message. Only for instruction/rule\n"
    "      collections the agent must always consult.\n"
    "    never — never surfaces in chat (background-only data).\n"
    "- recall: how ENTRIES are selected once it surfaces "
    "(relevant / recent / all).\n"
    "- extraction_prompt: REQUIRED. Numbered browse-first collector steps.\n"
    "- collector_interval_seconds: REQUIRED. 1800/3600/21600/86400.\n"
)

CREATE_PARAMS = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {
            "type": "string",
            "description": "Content-reflective retrieval anchor — list actual topics/items.",
        },
        "inclusion": {"type": "string", "enum": ["always", "relevant", "never"]},
        "recall": {"type": "string", "enum": ["relevant", "recent", "all"]},
        "extraction_prompt": {"type": "string"},
        "collector_interval_seconds": {"type": "integer"},
    },
    "required": ["name", "description", "inclusion", "recall", "extraction_prompt",
                 "collector_interval_seconds"],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "collection_create", "description": CREATE_DESC, "parameters": CREATE_PARAMS}},
    {"type": "function", "function": {
        "name": "browse", "description": "Look things up. Pass up to 3 queries/URLs.",
        "parameters": {"type": "object", "properties": {
            "reasoning": {"type": "string"},
            "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
            "required": ["queries"]}}},
]

# ── Candidate research skill (new design) ───────────────────────────────────

RESEARCH_SKILL = """## Recall context

### skills
Workflow patterns

#### [Research collection — notify on new finds] · 2026-06-01 00:00
TRIGGER
User wants ongoing research with notifications on new finds. Example phrasings:
- "research X for me, ping me when you find stuff"
- "follow X and let me know about new things"
- "keep an eye on X, tell me when there's something new"

STEPS

Single-turn act-then-echo.

1. Call collection_create with:
   - name: short slug from the topic
   - description: a CONTENT-REFLECTIVE anchor — list the specific topics,
     subjects and item types this collection will hold, phrased so future
     messages about any of them match it. Not "a collection of X".
   - inclusion: "relevant"
   - recall: "relevant"
   - collector_interval_seconds: 3600 (or match cadence words)
   - extraction_prompt (numbered, name each tool):
     > Collect [topic] — [scope].
     > 1. browse(...) — queries targeting [scope]; read actual pages
     > 2. log_read_next("user-messages") — pick up corrections
     > 3. Each entry: key = item name; content = name + description + URL
     > 4. collection_write("[name]", entries=[...])
     > 5. If write succeeded, send_message: "found a new X" + URL
     > 6. done(). If nothing new, just done().

2. Summarize back from the echo and ask if they want tweaks.

#### [Research collection — silent] · 2026-06-01 00:00
TRIGGER
User wants ongoing research WITHOUT notifications — they'll check in
themselves. Example phrasings:
- "research X for me, silent, i'll check in"
- "research X but don't ping me"
- "track X quietly, no notifications"

STEPS

Single-turn act-then-echo. Same as the notify variant EXCEPT:
- inclusion: "relevant", recall: "off"
- extraction_prompt has NO send_message step (silent — the collector
  writes entries but never pings; the user reads the list when they want).

Summarize back, mention it's silent.
"""

SYSTEM = f"{penny_identity()}\n\n{RESEARCH_SKILL}\n\n{conversation_prompt()}"


# ── Cases: (id, request, topic tokens for the description, expect_silent) ────

CASES = [
    ("research-boardgames",
     "i got back into board games — research heavier euro-style strategy games and "
     "modern tabletop classics for me, ping me when you find good ones",
     {"board", "game", "strategy", "euro", "tabletop", "worker", "engine", "2-player",
      "2 player", "classic"}, False),
    ("research-espresso",
     "research espresso machines and grinders under $1000 for me, ping me with finds",
     {"espresso", "machine", "grinder", "coffee", "1000", "burr", "boiler"}, False),
    ("research-novels",
     "keep an eye on new hard sci-fi novel releases and let me know when good ones land",
     {"sci-fi", "scifi", "science fiction", "novel", "book", "hard", "release"}, False),
    ("research-silent-fountainpens",
     "research fountain pens and inks for me — silent, i'll check the list myself",
     {"fountain pen", "pen", "ink", "nib", "writing", "stationery"}, True),
]


def _score(args: dict | None, topic_tokens: set[str], expect_silent: bool) -> CaseResult:
    fails: list[str] = []
    if args is None:
        return CaseResult("", False, ["no collection_create call"])
    desc = (args.get("description") or "")
    desc_l = desc.lower()
    matched = sum(1 for t in topic_tokens if t in desc_l)
    if matched < 2:
        fails.append(f"description not content-reflective ({matched} topic words): {desc[:70]!r}")
    if re.match(r"\s*(a |an )?(collection|list|set) of\b", desc_l):
        fails.append(f"generic container description: {desc[:50]!r}")
    if args.get("inclusion") != "relevant":
        fails.append(f"inclusion expected 'relevant', got {args.get('inclusion')!r}")
    body = (args.get("extraction_prompt") or "").lower()
    if "browse" not in body:
        fails.append("extraction_prompt missing browse step")
    if expect_silent and "send_message" in body:
        fails.append("silent request but body has send_message")
    if not expect_silent and "send_message" not in body:
        fails.append("notify request but body missing send_message")
    return CaseResult("", len(fails) == 0, fails)


def main() -> None:
    n = int(os.getenv("N_SAMPLES", "5"))
    h = Harness()
    print(f"# Collection authoring (two-stage design) — {h.model}, {n} samples")

    all_results: list[CaseResult] = []
    for cid, req, toks, silent in CASES:
        def run_one(req=req, toks=toks, silent=silent, cid=cid) -> CaseResult:
            msg = h.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": req}],
                tools=TOOLS,
            )
            calls = extract_tool_calls(msg)
            create = next((c["args"] for c in calls if c["name"] == "collection_create"), None)
            r = _score(create, toks, silent)
            r.case_id = cid
            return r
        all_results.extend(run_samples(cid, n, run_one))

    report(all_results, h.metrics)


if __name__ == "__main__":
    main()
