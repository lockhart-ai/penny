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

Beat-0 cases GATE at 0.8 (promoted 2026-07-16 after the matrix ran clean:
warm 0.96 · activity-window 1.00 · cold 1.00 · empty-registry 1.00 — a single
bail-recovery costs exactly one of five checks, so honest recoveries pass and
real breakage fails).  Later beats start REPORT-ONLY per the promote-later
discipline and gate once sample-verified.
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

    return [
        Check("the fact landed durably in a collection (any route)", fact_stored),
        Check("no runaway creation (at most one new collection)", len(created) <= 1),
        Check(
            # A word-list proved brittle (live sample: a valid confirmation
            # phrased outside the list).  The honest signal is the FACT: a
            # turn-1 reply that restates the stored value is an acknowledgment;
            # claiming the fact while storage failed is the dishonest case.
            "turn-1 reply acknowledges the fact it stored (SAID == DID)",
            fact_stored == ("499" in first_reply) if replies else False,
        ),
        Check("read-back states $499", "499" in final_reply),
        # NOTE: no hard provenance check here — answering a one-turn-old fact
        # from the conversation window is correct behavior (live sample 5).
        # The COLD variant below owns provenance absolutely.
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
        min_pass_rate=0.8,  # promoted 2026-07-16: sample-verified across the beat-0 matrix
    )


# ── Beat 0c: EMPTY registry — "remember X" with nowhere to put it ───────────
#
# Every seeded collection is deleted before the conversation: the store map is
# empty, there is no `knowledge` magnet, no container at all.  "Remember X"
# must drive CREATION (the #1630 skill-optional inert create) + the write —
# the create arm of remember → collection_create-or-collection_write.


def _delete_all_collections(db: Database) -> None:
    from sqlmodel import Session, delete, select

    from penny.database.models import MemoryEntry, MemoryRow

    with Session(db.engine) as session:
        names = [
            row.name
            for row in session.exec(select(MemoryRow).where(MemoryRow.type == "collection")).all()
        ]
        for name in names:
            session.exec(
                delete(MemoryEntry).where(MemoryEntry.memory_name == name)  # ty: ignore[invalid-argument-type]
            )
            session.exec(
                delete(MemoryRow).where(MemoryRow.name == name)  # ty: ignore[invalid-argument-type]
            )
        session.commit()


def _score_beat0_empty(db: Database, before: set[str], reply: str) -> list[Check]:
    created = new_collections(db, before)
    replies = _outgoing(db)
    final_reply = replies[-1] if replies else ""
    entries = collection_entries(db, created[0].name) if len(created) == 1 else {}
    fact_stored = any("499" in content for content in entries.values())

    return [
        Check("exactly one collection created (nowhere existed — she made one)", len(created) == 1),
        Check("the fact landed in the created collection", fact_stored),
        Check("read-back states $499", "499" in final_reply),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat0_empty_registry_creates(chat_eval: ChatEval):
    """Beat 0c: with ZERO collections in the registry, 'remember X' must create
    a container (skill-optional inert create) and write the fact into it."""
    await chat_eval(
        case_id="journey-beat0-empty-registry",
        messages=_BEAT0_TURNS,
        seed=_delete_all_collections,
        score=_score_beat0_empty,
        min_pass_rate=0.8,  # promoted 2026-07-16: sample-verified across the beat-0 matrix
    )


# ── Beat 0a: ACTIVITY-WINDOW recall — the write is ambient ───────────────────
#
# The fact was written by a RECENT run (no conversation carries it), so the
# self-state activity block renders the write ambiently (#1641):
#   run <id> · <when> · knowledge → worked (2 calls) · wrote 'aurora deck 2
#   price' → `knowledge`
# Awareness costs zero calls; retrieval is one call with both arguments
# consumable verbatim off the line.  Any storage read passes (code-owner
# ruling); the transcript shows whether she copied the rendered key.

_BEAT0A_TURN = "hey — remind me, what was the aurora deck 2 listed at?"


def _seed_recent_run_write(db: Database) -> None:
    import json as _json
    from datetime import UTC, datetime, timedelta

    from sqlmodel import Session

    from penny.database.models import MemoryEntry, PromptLog

    when = datetime.now(UTC) - timedelta(minutes=20)
    response = {"choices": [{"message": {"tool_calls": [{"id": "0"}, {"id": "1"}]}}]}
    with Session(db.engine) as session:
        session.add(
            PromptLog(
                model="test-model",
                messages="[]",
                response=_json.dumps(response),
                agent_name="chat",
                run_id="seedrun0a",
                run_outcome="worked",
                run_reason="",
                run_target="knowledge",
                timestamp=when,
            )
        )
        session.add(
            MemoryEntry(
                memory_name="knowledge",
                key="aurora deck 2 price",
                content="$499",
                author="chat",
                created_at=when,
                created_by_run_id="seedrun0a",
                last_written_by_run_id="seedrun0a",
            )
        )
        session.commit()


def _score_beat0a(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = _outgoing(db)
    final_reply = replies[-1] if replies else ""
    read_backed = any(
        (tool in _READ_TOOLS and args.get("memory") == "knowledge") or tool == "find"
        for tool, args in _final_run_calls(db)
    )

    return [
        Check("recall states $499 (the write is ambient, value is not)", "499" in final_reply),
        Check("answer BACKED by a storage read (any route)", read_backed),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat0a_activity_window_recall(chat_eval: ChatEval):
    """Beat 0a: a fact written by a recent run renders ambiently on the run
    line (key + collection, never the value) — retrieval is one call with
    arguments consumable verbatim off the line."""
    await chat_eval(
        case_id="journey-beat0a-activity-recall",
        message=_BEAT0A_TURN,
        seed=_seed_recent_run_write,
        score=_score_beat0a,
        min_pass_rate=0.8,  # promoted 2026-07-16: sample-verified across the beat-0 matrix
    )


# ── Beat 0b: COLD recall — storage is the only route ────────────────────────
#
# The fact was stored in a PREVIOUS session (seeded directly; no conversation
# history carries it), so conversation echo is impossible: the answer exists
# only in the store.  This is the n≤1 invariant's absolute test — the model
# must reach the entry via `find` (guess-free) or a correctly-aimed scoped
# read.  Provenance is a HARD check here, unlike the warm case above.

_BEAT0_COLD_TURN = (
    "hey — a while back I asked you to remember what the aurora deck 2 "
    "was listed at. what was the price?"
)


def _seed_cold_fact(db: Database) -> None:
    from penny.database.memory.types import EntryInput

    db.memory("knowledge").write(
        [EntryInput(key="aurora deck 2 price", content="$499")], author="chat"
    )


def _score_beat0_cold(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = _outgoing(db)
    final_reply = replies[-1] if replies else ""
    calls = _final_run_calls(db)
    read_backed = any(
        (tool in _READ_TOOLS and args.get("memory") == "knowledge") or tool == "find"
        for tool, args in calls
    )

    return [
        Check("cold recall states $499 (storage is the only route)", "499" in final_reply),
        Check("answer BACKED by a storage read (find or a scoped read)", read_backed),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat0_cold_recall(chat_eval: ChatEval):
    """Beat 0b: a fact stored in a previous session is retrieved with zero
    conversational trace — the absolute test of one-call reachability."""
    await chat_eval(
        case_id="journey-beat0-cold-recall",
        message=_BEAT0_COLD_TURN,
        seed=_seed_cold_fact,
        score=_score_beat0_cold,
        min_pass_rate=0.8,  # promoted 2026-07-16: sample-verified across the beat-0 matrix
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
    # The INSTIGATING ask — deliberately unfulfillable as stated: "watch this"
    # requires a job, a job's prompt only exists as a skill render, and no
    # skill exists.  The correct move is the honest gap: "I don't know how —
    # teach me."  (The earlier draft front-loaded the read/extract/store
    # instructions here; those are the TEACHING prompt and belong in turn 2,
    # as the user's RESPONSE to her ask.)
    (
        f"can you watch the aurora deck 2 listing at {LISTING_URL} "
        "and let me know if the price ever changes?"
    ),
    # The TEACHING walkthrough — the explicit instructions, where they belong.
    (
        f"sure — read {LISTING_URL}, pull out just the price (nothing else), "
        "and remember it as 'Aurora Deck 2'"
    ),
    # The promotion.  (Attaching the skill to make the watch RUN is beat 2.)
    "perfect — save that as a skill so you can do this again",
]


def _outgoing(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = db.memory("penny-messages").read_recent(window_seconds=3600, cap=None)
    return [entry.content for entry in entries]


def _asks_for_demonstration(replies: list[str]) -> bool:
    """Broad semantic match for the honest gap being voiced — "I don't know
    how, teach me" in any paraphrase.  Match the intent, not the wording."""
    needles = (
        "walk me through",
        "walk you through",
        "show me how",
        "show me once",
        "teach me",
        "teach it",
        "demonstrate",
        "walk me thru",
        "guide me through",
        "don't know how",
        "don't yet know how",
        "haven't learned",
        "no skill",
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

    no_watch_yet = all(
        row.extraction_prompt is None
        for row in db.memories.list_all()
        if row.type == "collection" and not row.archived
    )

    return [
        Check(
            "turn 1 voices the honest gap (asks to be taught, any paraphrase)",
            _asks_for_demonstration(replies),
        ),
        Check("exactly one container created", len(created) == 1),
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
        Check(
            "no dispatchable watch exists yet (attach is beat 2 — no faked watch)",
            no_watch_yet,
        ),
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
