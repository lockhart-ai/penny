"""Novel-pattern suite — does the substrate GENERALIZE past the seeded skills?

Requests with no matching skill.  The bar is "sensible behavior": improvise
a reasonable collection_create, OR ask a clarifying question, OR browse —
but don't silently do something clearly wrong.  Each case prints what the
model did; the pass check is lenient (did it take a coherent action).
"""
from __future__ import annotations

from scripts.prompt_validation._harness import (
    CaseResult,
    Conversation,
    Harness,
    browse_tool,
    conversation_prompt,
    converse,
    load_seed_skills,
    load_tool,
    penny_identity,
    render_skills_recall,
    run_samples,
)

NAME = "novel_patterns"

_SEED, _ = load_seed_skills()
TOOLS = [
    load_tool("CollectionCreateTool"),
    load_tool("CollectionMetadataTool"),
    load_tool("ReadLatestTool"),
    browse_tool(),
]
SYSTEM = f"{penny_identity()}\n\n{render_skills_recall(_SEED)}\n\n{conversation_prompt()}"


def _serve(name: str, args: dict):
    if name == "collection_create":
        return f"Created collection '{args.get('name')}'."
    if name == "browse":
        return "## search results\n~5 pages found (synthetic)."
    if name in ("read_latest", "collection_metadata"):
        return "(no entries)"
    return "ok"


CASES = [
    ("url-watcher",
     "watch this page https://example.com/news for changes weekly and tell me when it updates"),
    ("recurring-reminder",
     "remind me to water my plants every sunday morning"),
    ("chat-pattern-extraction",
     "every time i mention a book in our chats, quietly save the title to a list"),
    ("tool-gap-email",
     "summarize every email i get from my landlord and send me the summary"),
]


def _score(conv: Conversation) -> list[str]:
    """Lenient: a coherent response is enough — created a collection, browsed,
    or gave a substantive text reply (incl. gracefully declining a tool-gap).
    The only failure is doing nothing at all (empty / stuck)."""
    names = [c["name"] for c in conv.all_calls()]
    if "collection_create" in names or "browse" in names:
        return []
    if len((conv.final_text or "").strip()) >= 20:
        return []
    return ["no action and no substantive reply (possibly stuck)"]


def run(h: Harness, samples: int, only: str | None = None) -> list[CaseResult]:
    results: list[CaseResult] = []
    for cid, msg in CASES:
        if only and only != cid:
            continue

        def one(msg=msg, cid=cid):
            conv = converse(h, SYSTEM, msg, TOOLS, _serve)
            acted = [c["name"] for c in conv.all_calls()] or ["text-only"]
            print(f"      behavior: {acted}  reply={(conv.final_text or '')[:60]!r}")
            return CaseResult(cid, not _score(conv), _score(conv))

        results.extend(run_samples(f"{NAME}:{cid}", samples, one))
    return results
