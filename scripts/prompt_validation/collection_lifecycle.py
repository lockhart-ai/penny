"""Collection lifecycle suite — chat-agent authoring/operating collections.

Exercises the full lifecycle against the REAL production prompts (seed
skills from migration 0043, tool descriptions from memory_tools.py):

  create  — research+notify, research+silent, scheduled digest, cadence-in-request
  update  — add scope, focus-swap (drop A / focus B), flip silent↔notify, change cadence
  archive — done / cancelled phrasings
  query   — read entries / answer from ambient recall (no browse, no create)
  abstain — ambiguous one-shot, implicit trip-prep (should NOT create a collection)

Single-turn act-then-echo: the model calls the tool, we serve a synthetic
echo, it summarizes.  We score the captured tool call + the summary.
"""
from __future__ import annotations

from scripts.prompt_validation._harness import (
    TERMINAL,
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
)
from scripts.prompt_validation.fixtures import SYNTH_COLLECTIONS

NAME = "collection_lifecycle"

# Real tool surface + real seed skills.
_SEED, _ = load_seed_skills()
TOOLS = [
    load_tool("CollectionCreateTool"),
    load_tool("CollectionUpdateTool"),
    load_tool("CollectionArchiveTool"),
    load_tool("CollectionMetadataTool"),
    load_tool("ReadLatestTool"),
    load_tool("ReadSimilarTool"),
    browse_tool(),
]
SKILLS_RECALL = render_skills_recall(_SEED)

# An existing collection for update/query cases (synthetic, scrubbed).
_BG = next(c for c in SYNTH_COLLECTIONS if c.name == "board-games")
_BG_PROMPT = (
    "Collect heavier euro-style strategy board games and modern tabletop classics.\n"
    "1. browse the web for new strategy board games; read actual pages.\n"
    '2. log_read_next("user-messages") to pick up corrections.\n'
    "3. Each entry: key = game name; content = name + description + player count + URL.\n"
    '4. collection_write("board-games", entries=[...]).\n'
    "5. If write succeeded, send_message: a one-sentence \"found a new game\" note + URL.\n"
    "6. done(). If nothing new, just done()."
)
_BG_RECALL = "## Recall context (existing collection)\n\n### board-games\n" + _BG.description + "\n" + "\n".join(
    f"\n#### [{e.split(' — ')[0]}] · 2026-06-01 00:00\n{e}" for e in _BG.entries
)
_BG_METADATA = (
    f"name: board-games\ntype: collection\ndescription: {_BG.description}\n"
    f"inclusion: relevant\nrecall: relevant\narchived: False\ninterval: 3600s\n"
    f"extraction prompt: {_BG_PROMPT}"
)

SYSTEM_BASE = f"{penny_identity()}\n\n{SKILLS_RECALL}\n\n{conversation_prompt()}"
SYSTEM_WITH_BG = f"{penny_identity()}\n\n{SKILLS_RECALL}\n\n{_BG_RECALL}\n\n{conversation_prompt()}"


def _serve(name: str, args: dict):
    """Synthetic tool results — act-then-echo keeps everything non-terminal so
    the model proceeds to its summary; TERMINAL is unused here."""
    if name == "collection_metadata":
        return _BG_METADATA if args.get("memory") == "board-games" else f"'{args.get('memory')}' not found."
    if name in ("read_latest", "read_similar"):
        if args.get("memory") == "board-games":
            return "\n".join(f"- [{e.split(' — ')[0]}] {e}" for e in _BG.entries)
        return "(no entries)"
    if name == "collection_create":
        a = args
        return (
            f"Created collection '{a.get('name')}':\n  interval: {a.get('collector_interval_seconds')}s\n"
            f"  inclusion: {a.get('inclusion')}\n  recall: {a.get('recall')}\n"
            f"  description: {a.get('description', '')}\n"
            f"  extraction_prompt: {a.get('extraction_prompt', '')[:200]}"
        )
    if name == "collection_update":
        return f"Updated collection '{args.get('name')}': {', '.join(k for k in args if k != 'name')}"
    if name == "collection_archive":
        return f"Archived '{args.get('memory')}'."
    if name == "browse":
        return "## search results\n~5 relevant pages found (synthetic)."
    return TERMINAL  # unknown tool → stop


def _call(conv: Conversation, name: str) -> dict | None:
    return next((c["args"] for c in conv.all_calls() if c["name"] == name), None)


# ── Scorers ─────────────────────────────────────────────────────────────────

_REQ_CREATE = {
    "name",
    "description",
    "inclusion",
    "recall",
    "extraction_prompt",
    "collector_interval_seconds",
}


def _score_create(
    conv: Conversation, *, inclusion: str, send_message: bool, interval: int | None
) -> list[str]:
    """Score a create against the two-flag surface.

    ``inclusion`` is the load-bearing routing expectation (silent =
    ``never``).  ``recall`` is not asserted — for the create flows any
    valid entry mode is acceptable and the enum schema constrains it.
    """
    f = []
    a = _call(conv, "collection_create")
    if a is None:
        return ["no collection_create call"]
    missing = _REQ_CREATE - set(a.keys())
    if missing:
        f.append(f"missing fields: {sorted(missing)}")
    if a.get("inclusion") != inclusion:
        f.append(f"inclusion expected {inclusion!r}, got {a.get('inclusion')!r}")
    body = (a.get("extraction_prompt") or "").lower()
    if "browse" not in body:
        f.append("body missing browse step")
    if send_message and "send_message" not in body:
        f.append("notify but body missing send_message")
    if not send_message and "send_message" in body:
        f.append("silent but body has send_message")
    if interval is not None and a.get("collector_interval_seconds") != interval:
        f.append(f"interval expected {interval}, got {a.get('collector_interval_seconds')}")
    return f


def _score_update(conv: Conversation, *, added: tuple[str, ...] | None = None,
                  removed: str | None = None, inclusion: str | None = None,
                  interval: int | None = None, body_required: bool = True) -> list[str]:
    f = []
    a = _call(conv, "collection_update")
    if a is None:
        return ["no collection_update call"]
    if a.get("name") != "board-games":
        f.append(f"wrong name: {a.get('name')!r}")
    body = a.get("extraction_prompt") or ""
    if body_required and not body:
        f.append("expected extraction_prompt body rewrite")
    # ``added`` is a set of acceptable phrasings — any one present passes.
    if added and not any(t.lower() in body.lower() for t in added):
        f.append(f"body missing added scope (any of {added})")
    if removed and removed.lower() in body.lower():
        f.append(f"body still has removed scope {removed!r}")
    if inclusion and a.get("inclusion") != inclusion:
        f.append(f"inclusion expected {inclusion!r}, got {a.get('inclusion')!r}")
    if inclusion == "never" and "send_message" in body.lower():
        f.append("silent flip but body still has send_message")
    if interval is not None and a.get("collector_interval_seconds") != interval:
        f.append(f"interval expected {interval}, got {a.get('collector_interval_seconds')}")
    return f


def _score_archive(conv: Conversation) -> list[str]:
    a = _call(conv, "collection_archive")
    if a is None:
        return ["no collection_archive call"]
    return [] if a.get("memory") == "board-games" else [f"wrong memory: {a.get('memory')!r}"]


def _score_query(conv: Conversation) -> list[str]:
    f = []
    names = [c["name"] for c in conv.all_calls()]
    if "browse" in names:
        f.append("browsed instead of using the collection")
    if "collection_create" in names or "collection_update" in names:
        f.append("mutated on a read-only query")
    text = conv.final_text.lower()
    if text and not any(g.split(" — ")[0].lower() in text for g in _BG.entries):
        f.append("answer references no saved entry")
    return f


def _score_no_create(conv: Conversation, want_browse: bool) -> list[str]:
    f = []
    names = [c["name"] for c in conv.all_calls()]
    if "collection_create" in names:
        f.append("silently created a collection")
    if want_browse and "browse" not in names:
        f.append("did not browse the one-shot question")
    return f


# (id, system, user message, scorer)
CASES = [
    ("create-notify", SYSTEM_BASE,
     "research heavier euro-style strategy board games for me, ping me when you find good ones",
     lambda c: _score_create(c, inclusion="relevant", send_message=True, interval=None)),
    ("create-silent", SYSTEM_BASE,
     "research fountain pens and inks for me — silent, i'll check the list myself",
     lambda c: _score_create(c, inclusion="never", send_message=False, interval=None)),
    ("create-digest", SYSTEM_BASE,
     "research indie game releases for me — check hourly, send me a digest at 6pm",
     lambda c: _score_create(c, inclusion="relevant", send_message=True, interval=None)),
    ("create-cadence", SYSTEM_BASE,
     "research new sci-fi novels for me, check daily, ping me when good ones land",
     lambda c: _score_create(c, inclusion="relevant", send_message=True, interval=86400)),
    ("update-add-scope", SYSTEM_WITH_BG,
     "add solo/co-op board games to the board games collection too",
     lambda c: _score_update(c, added=("solo", "co-op", "cooperative"))),
    ("update-focus-swap", SYSTEM_WITH_BG,
     "actually drop the party games from board-games, just focus on heavy 2-player duels",
     lambda c: _score_update(c, added=("2-player", "two-player", "duel", "1v1", "head-to-head"),
                             removed="party")),
    ("update-silent-flip", SYSTEM_WITH_BG,
     "stop pinging me about new board game finds, i'll just check the collection myself",
     lambda c: _score_update(c, inclusion="never")),
    ("update-cadence", SYSTEM_WITH_BG,
     "check the board games collection every 30 minutes instead",
     lambda c: _score_update(c, interval=1800, body_required=False)),
    ("archive-done", SYSTEM_WITH_BG,
     "i'm done collecting board games, archive that one",
     _score_archive),
    ("archive-cancelled", SYSTEM_WITH_BG,
     "lost interest in board games, close the board-games collection please",
     _score_archive),
    ("query-entries", SYSTEM_WITH_BG,
     "remind me which board games we'd flagged as worth buying",
     _score_query),
    ("ambiguous-lookup", SYSTEM_BASE,
     "find me a good low-light houseplant that's hard to kill",
     lambda c: _score_no_create(c, want_browse=True)),
    ("implicit-prep", SYSTEM_BASE,
     "booked a cabin trip for october, 10 days off-grid. starting to plan.",
     lambda c: _score_no_create(c, want_browse=False)),
]


def run(h: Harness, samples: int, only: str | None = None) -> list[CaseResult]:
    from scripts.prompt_validation._harness import run_samples

    results: list[CaseResult] = []
    for cid, system, msg, scorer in CASES:
        if only and only != cid:
            continue

        def one(system=system, msg=msg, scorer=scorer, cid=cid) -> CaseResult:
            conv = converse(h, system, msg, TOOLS, _serve)
            fails = scorer(conv)
            return CaseResult(cid, not fails, fails)

        results.extend(run_samples(f"{NAME}:{cid}", samples, one))
    return results
