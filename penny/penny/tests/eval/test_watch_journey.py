"""The watch journey (#1570) — the epic's composed-behavior exit gate.

Each beat drives the REAL chat/collector loops against the live model with
NATURAL user language ("remember it", "let me know if it changes") and scores
persisted DB state — the NL→machinery mapping IS the contract (a script whose
user turns name tools or collections tests an actor reading stage directions,
not an assistant).  Fixture is fully synthetic: a fictional marketplace listing
("Aurora Deck 2" on faux-market.example) with a controllable price field.

Beat map (the case plan on #1570):
    1. elicit + teach      — this file's first case
    2. instantiate w/ expiry
    3. quiet cycles / the change
    4. refresh (re-teach)
    5. inspect (state + provenance)
    6. multi-instantiate + teardown
    7. self-termination

Cases start REPORT-ONLY (``min_pass_rate=None``) per the promote-later
discipline: first live runs locate where the model actually stumbles; gating
thresholds come once the scorer is verified against captured samples.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.skill_store import holes_from_json, steps_from_json
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    collection_entries,
    new_collections,
)
from penny.tests.eval.fixtures import CannedPage

pytestmark = pytest.mark.eval


# ── Fixture: the fictional listing (price is the controllable field) ─────────

LISTING_URL = "https://faux-market.example/aurora-deck-2"

AURORA_LISTING_499 = CannedPage(
    match="aurora-deck-2",
    text=(
        "Title: Aurora Deck 2 — handheld console | faux-market\n"
        f"{LISTING_URL}\n"
        "\n"
        "Aurora Deck 2 (open box, tested). Ships from a fictional warehouse.\n"
        "Price: $499\n"
        f"[Aurora Deck 2 listing]({LISTING_URL})\n"
        "Seller: nebula_resale (4.9 stars). Listing updated daily.\n"
    ),
)


# ── Beat 0: the atom — "remember X" → a durable write, proven by read-back ──
#
# Before teaching, watching, or notifying can compose, the primitive must hold:
# a natural "can you remember <fact> for me" maps to a collection write
# (create-a-container or write-into-an-appropriate-existing-one — the ROUTE is
# the model's choice; the OUTCOME is scored), and the fact is retrievable a
# turn later WITHOUT re-asking or browsing.  No browse fixture is installed —
# the user states the fact, so the read-back can only come from storage.

_BEAT0_TURNS = [
    "hey, can you remember that the aurora deck 2 is listed at $499 for me?",
    "thanks — what did I say the aurora deck 2 was listed at?",
]


def _all_collection_writes(db: Database, before: set[str]) -> dict[str, dict[str, str]]:
    """Entries of every non-log collection that could have received the fact —
    existing collections plus anything created this sample."""
    names = {
        row.name for row in db.memories.list_all() if row.type == "collection" and not row.archived
    } | {row.name for row in new_collections(db, before)}
    return {name: collection_entries(db, name) for name in names}


_READ_TOOLS = ("read_similar", "collection_read_latest", "collection_get")

# The chat loop's text-bail nudge (injected as a user turn when the model emits
# prose instead of a tool call) — its presence means the routing slipped, even
# if recovery then succeeded.  Loop-health visibility, not a behavior score.
_BAIL_NUDGE_MARKER = "could not be parsed as a tool call"


def _final_run_calls(db: Database) -> list[tuple[str, dict]]:
    """(tool, args) for every call in the LAST chat run — the turn-2 answer's
    actual evidence trail, read from the persisted promptlog."""
    import json as _json

    rows = [r for r in db.messages.recent_prompts(limit=200) if r.run_id]
    if not rows:
        return []
    rows.sort(key=lambda r: r.timestamp)
    last_run = rows[-1].run_id
    calls: list[tuple[str, dict]] = []
    for row in rows:
        if row.run_id != last_run or not row.response:
            continue
        response = _json.loads(row.response)
        message = response.get("choices", [{}])[0].get("message", {})
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            try:
                args = _json.loads(function.get("arguments") or "{}")
            except ValueError, TypeError:
                args = {}
            calls.append((function.get("name", ""), args))
    return calls


def _bail_nudge_fired(db: Database) -> bool:
    """True when any prompt's message array carries the injected text-bail nudge."""
    for row in db.messages.recent_prompts(limit=200):
        if row.messages and _BAIL_NUDGE_MARKER in row.messages:
            return True
    return False


def _score_beat0(db: Database, before: set[str], reply: str) -> list[Check]:
    created = new_collections(db, before)
    replies = _outgoing(db)
    stored = _all_collection_writes(db, before)
    fact_collections = {
        name
        for name, entries in stored.items()
        if any("499" in content for content in entries.values())
    }
    fact_stored = bool(fact_collections)
    first_reply = replies[0] if replies else ""
    final_reply = replies[-1] if replies else ""
    read_backed = any(
        (tool in _READ_TOOLS and args.get("memory") in fact_collections)
        # `find` (#1640) is the guess-free cross-memory entry search — a find
        # call in the answering run IS a storage read (the hit carries the
        # value); the separate $499 check still guards answer correctness.
        or tool == "find"
        for tool, args in _final_run_calls(db)
    )

    return [
        Check("the fact landed durably in a collection (any route)", fact_stored),
        Check("no runaway creation (at most one new collection)", len(created) <= 1),
        Check(
            "turn-1 reply confirms remembering (SAID == DID)",
            fact_stored
            == (
                "remember" in first_reply.lower()
                or "saved" in first_reply.lower()
                or "recorded" in first_reply.lower()
                or "noted" in first_reply.lower()
                or "got it" in first_reply.lower()
                or "jotted" in first_reply.lower()
            )
            if replies
            else False,
        ),
        Check("read-back states $499", "499" in final_reply),
        Check(
            "read-back BACKED BY a storage read (turn-2 run read the fact's collection)",
            read_backed,
        ),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat0_remember_and_recall(chat_eval: ChatEval):
    """Beat 0: the storage atom — a natural 'remember X' lands the fact in a
    collection and a follow-up retrieves it, with no browse available."""
    await chat_eval(
        case_id="journey-beat0-remember-recall",
        messages=_BEAT0_TURNS,
        score=_score_beat0,
        min_pass_rate=None,  # report-only until the scorer is sample-verified
    )


# ── Beat 1: elicit + teach ───────────────────────────────────────────────────
#
# The user asks in natural language with NO skill in the registry (fresh DB —
# the skill table ships empty).  The designed happy path (#1629/#1630): the
# NO_SKILL_FOUND elicitation → set up the inert container → the user walks her
# through ONCE (real browse + real write, certified by execution) → skill_create
# over that run.  The instantiation/attach is beat 2's job, so turn 3 here ends
# at the saved skill.

_BEAT1_TURNS = [
    # The natural ask — "remember it" must map to storage, "let me know if it
    # changes" to emission.  No tool names, no "collection".
    (
        f"can you go read the aurora deck 2 listing at {LISTING_URL}, "
        "find the price, and remember it? and let me know if it ever changes"
    ),
    # The walkthrough — still natural: the one value to extract, and a plain
    # "remember it as" for the write.
    (
        f"sure — read {LISTING_URL}, pull out just the price (nothing else), "
        "and remember it as 'Aurora Deck 2'"
    ),
    # The promotion.
    "perfect — save that as a skill so you can do this again",
]


def _outgoing(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = db.memory("penny-messages").read_recent(window_seconds=3600, cap=None)
    return [entry.content for entry in entries]


def _asks_for_demonstration(replies: list[str]) -> bool:
    """Broad semantic match for the elicitation being RELAYED to the user —
    the model paraphrases the walk-me-through-it-once literal, so match the
    intent, not the wording."""
    needles = (
        "walk me through",
        "walk you through",
        "show me how",
        "show me once",
        "teach me",
        "demonstrate",
        "walk me thru",
        "guide me through",
    )
    return any(needle in reply.lower() for reply in replies for needle in needles)


def _browsed_listing(db: Database) -> bool:
    """The demo browse is persisted in the browse-results log — score the
    durable record, not the call transcript."""
    entries = db.memory("browse-results").read_recent(window_seconds=3600, cap=None)
    return any("aurora-deck-2" in entry.content for entry in entries)


def _score_beat1(db: Database, before: set[str], reply: str) -> list[Check]:
    created = new_collections(db, before)
    container = created[0] if len(created) == 1 else None
    replies = _outgoing(db)
    entries = collection_entries(db, container.name) if container else {}
    wrote_price = any("499" in content for content in entries.values())

    skills = db.skills.list_all()
    skill = skills[0] if len(skills) == 1 else None
    steps = steps_from_json(skill.steps) if skill else []
    step_tools = [step.tool for step in steps]
    holes = holes_from_json(skill.holes) if skill else []

    return [
        Check("exactly one container created (remember → storage)", len(created) == 1),
        Check(
            "a reply asks for a demonstration (elicitation relayed)",
            _asks_for_demonstration(replies),
        ),
        Check("demo browse read the listing (persisted in browse-results)", _browsed_listing(db)),
        Check("demo write landed the price in the container", wrote_price),
        Check(
            "SAID == DID: a reply states the fixture price ($499)", any("499" in r for r in replies)
        ),
        Check("exactly one skill saved", skill is not None),
        Check(
            "skill steps are the certified demo calls (browse → collection_write)",
            step_tools == ["browse", "collection_write"],
        ),
        Check("skill records its source run", bool(skill and skill.source_run_id)),
        Check("at least one hole inferred from the utterance", len(holes) >= 1),
    ]


@pytest.mark.asyncio
async def test_beat1_elicit_and_teach(chat_eval: ChatEval):
    """Beat 1: a natural watch request with no skill in the registry elicits a
    walkthrough, the demonstration executes for real (browse + write into the
    container), and the run is promoted to a certified skill."""
    await chat_eval(
        case_id="journey-beat1-elicit-teach",
        messages=_BEAT1_TURNS,
        browse=[AURORA_LISTING_499],
        score=_score_beat1,
        min_pass_rate=None,  # report-only until the scorer is sample-verified
    )
