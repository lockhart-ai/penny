"""The watch journey (#1570) — the epic's composed-behavior exit gate.

Each beat drives the REAL chat/collector loops against the live model with
NATURAL user language ("remember it", "let me know if it changes") and scores
persisted DB state — the NL→machinery mapping IS the contract (a script whose
user turns name tools or collections tests an actor reading stage directions,
not an assistant).  Fixture is fully synthetic: a fictional marketplace listing
("Aurora Deck 2" on faux-market.example) with a controllable price field.

Beat map (one beat at a time, each with an exact terminal state; #1658 made
skill authoring AUTOMATIC — no "save that as a skill" beat exists anymore):
    0. remember      — the storage atom (remember X → durable write → read-back)
    1. elicit        — empty registry → recognize the gap, ask to be taught
    2. demonstrate   — follow "read this url, find the price, remember it";
                       the skill AUTO-EXTRACTS from the run + she narrates it
    3. instantiate   — "keep watching it and let me know" → find hits the
                       learned skill → a LIVE watch (trigger, notify, retarget)
    4. quiet cycles / the change
    5. refresh (re-demonstrate — replaces the skill)
    6. inspect (state + provenance)
    7. multi-instantiate + teardown
    8. self-termination

Beat-0 cases GATE at 0.8 (promoted 2026-07-16 after the matrix ran clean:
warm 0.96 · activity-window 1.00 · cold 1.00 · empty-registry 1.00 — a single
bail-recovery costs exactly one of five checks, so honest recoveries pass and
real breakage fails).  Later beats start REPORT-ONLY per the promote-later
discipline and gate once sample-verified.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.skill_store import parameters_from_json, steps_from_json
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

# The chat loop's text-bail nudges (injected as a user turn when the model emits
# prose OR a call-shaped JSON blob instead of a real tool call) — their presence
# means the routing slipped, even if recovery then succeeded.  Loop-health
# visibility, not a behavior score.  TWO distinct nudges cover the family:
# TOOL_FORMAT_NUDGE (a parse-500 fallback) and CHAT_CALL_AS_TEXT_NUDGE (the
# call-as-text bail) — an earlier single marker silently missed the second and
# false-greened a real spiral.  Each marker is an ASCII, newline-free slice of
# one nudge, so it survives the JSON-escaping of row.messages (a full-string
# match would break on escaped newlines and — em-dashes).
_BAIL_NUDGE_MARKERS = (
    "could not be parsed as a tool call",  # Prompt.TOOL_FORMAT_NUDGE
    "wrote a tool call as plain text",  # Prompt.CHAT_CALL_AS_TEXT_NUDGE
)


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
    """True when any prompt's message array carries an injected text-bail nudge."""
    for row in db.messages.recent_prompts(limit=200):
        if row.messages and any(marker in row.messages for marker in _BAIL_NUDGE_MARKERS):
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


# ── Beat 1: elicit ───────────────────────────────────────────────────────────
#
# The user asks in natural language with NO skill in the registry (fresh DB —
# the skill table ships empty).  The teach loop breaks into three beats, each
# with its own terminal state (one-beat-at-a-time): beat 1 ELICITS (she
# recognizes the gap and asks to be taught, full stop), beat 2 DEMONSTRATES (the
# user walks her through one cycle; she browses, extracts, writes a baseline),
# beat 3 PROMOTES (skill_create over that run + attach — where the from_run
# distillation lives).  This case is beat 1 only.

# The INSTIGATING ask — deliberately unfulfillable as stated: "watch this"
# requires a job, a job's prompt only exists as a skill render, and no skill
# exists.  THIS BEAT'S TERMINAL STATE (one-beat-at-a-time): she recognizes she
# needs a skill she doesn't have and asks to be taught.  Full stop.  The
# teaching walkthrough and the promotion are LATER beats with their own cases.
_BEAT1_TURN = (
    f"can you watch the aurora deck 2 listing at {LISTING_URL} "
    "and let me know if the price ever changes?"
)


def _outgoing(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = db.memory("penny-messages").read_recent(window_seconds=3600, cap=None)
    return [entry.content for entry in entries]


def _score_beat1(db: Database, before: set[str], reply: str) -> list[Check]:
    """The four OBJECTIVE facts of the terminal state — she didn't fake it.

    The fifth, linguistic fact — "she voices the gap and asks to be taught" — is
    NOT scored here: it's one line of English with no structural DB signal, and a
    keyword matcher can't track the model's paraphrasing ("walking through it
    once", "quick demo together", "give me a quick example").  The harness dumps
    every reply verbatim (EVAL_REPORT_DIR), so that facet is read off the
    transcript by the reviewer, not approximated by a scorer that emits false red.
    What's automated here is exactly what's objective: no improvisation, no faked
    watch — the durable regression net that a future "she faked it" break trips."""
    created = new_collections(db, before)

    # Only collections SHE created this sample must be prompt-less (seeded
    # system collections legitimately carry prompts).
    no_watch_faked = all(row.extraction_prompt is None for row in created)
    seeded_writes = sum(
        len(collection_entries(db, name)) for name in ("likes", "dislikes", "knowledge", "thoughts")
    )

    return [
        Check(
            "no improvised stand-in writes (seeded collections untouched)",
            seeded_writes == 0,
        ),
        Check(
            "no dispatchable watch faked (her collections, if any, are inert)",
            no_watch_faked,
        ),
        Check(
            "at most one container created (the elicitation's step 1 is fine)", len(created) <= 1
        ),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat1_recognizes_the_skill_gap(chat_eval: ChatEval):
    """Beat 1, terminal state: an instigating watch request with no skill in
    the registry ends with Penny voicing the gap and asking to be taught —
    no improvisation, no faked watch.  Teaching and promotion are later beats.

    The scored checks cover the objective half (she didn't fake it); the "asks
    to be taught" verdict is read off the dumped transcript (see _score_beat1)."""
    await chat_eval(
        case_id="journey-beat1-skill-gap",
        message=_BEAT1_TURN,
        browse=[AURORA_LISTING_499],
        score=_score_beat1,
        min_pass_rate=None,  # report-only until the scorer is sample-verified
    )


# ── Beat 2: demonstrate ──────────────────────────────────────────────────────
#
# The user gives her the routine as three bare, enact-able steps — "read this
# url, find the price, remember it" — and THIS BEAT'S TERMINAL STATE
# (one-beat-at-a-time) is simply that she FOLLOWS them: browse the page, extract
# the price, write it down.  No "notify me" here — operationalizing the watch is
# beat 3; this beat only has to prove she reliably enacts a demonstrated routine.
#
# Two properties are load-bearing:
#   • The value must come from the BROWSE, not the turn — the step says "find the
#     price", never "$499" — so a stored "499" proves she actually read the page
#     (the inverse of beat 0's no-browse read-back).
#   • WHERE she writes is deliberately FREE.  The demonstrated write's collection
#     is captured in the run; at beat-3 distillation skill_create extracts it as
#     a PARAMETER (a hole), and instantiation must re-provide it (write-retarget).
#     So constraining the target here would break the very freedom the skill
#     substrate relies on — score "landed in SOME collection", never which one.

_BEAT2_TURN = (
    f"read the aurora deck 2 listing at {LISTING_URL}, find the current price, and remember it."
)


def _browsed_the_listing(db: Database) -> bool:
    """The walked-through fetch is persisted in the browse-results log — score
    the durable record, not the call transcript."""
    entries = db.memory("browse-results").read_recent(window_seconds=3600, cap=None)
    return any("aurora-deck-2" in entry.content for entry in entries)


def _score_beat2(db: Database, before: set[str], reply: str) -> list[Check]:
    """The objective facts of the demonstrate terminal state: she enacted the
    three steps (browsed → extracted → stored the browsed value), NARRATED them
    honestly, and the skill auto-extracted.  Narrated dishonesty is the failure
    that makes the whole journey collapse: the user teaches by watching what she
    says she did, and the skill is distilled from the actual run — a SAID≠DID
    gap poisons both."""
    stored = _all_collection_writes(db, before)
    value_stored = any(
        "499" in content for entries in stored.values() for content in entries.values()
    )
    replies = _outgoing(db)
    return [
        Check(
            "she browsed the listing (step 1 — the demonstrated fetch happened)",
            _browsed_the_listing(db),
        ),
        Check(
            "the browsed value ($499) landed durably in a collection (steps 2+3, any collection)",
            value_stored,
        ),
        Check(
            # SAID == DID via the acknowledge-the-fact pattern (verb lists proved
            # brittle three times).  The narration flow sends TWO replies per
            # teach turn (the routine report + the learned-skill narration), so
            # the echo is checked across ALL sent replies, not just the first.
            "a sent reply reports the browsed value it stored (SAID == DID)",
            (any("499" in reply for reply in replies) == value_stored) if replies else False,
        ),
        Check(
            # The auto-extraction (#1658): the demonstration run ITSELF yields the
            # skill — deterministically, at the run-end chokepoint, no authoring
            # tool.  The demonstrate turn's terminal state includes it.
            "a skill was auto-extracted from the demonstration (browse+write, parameterized)",
            _extracted_skill_shape_ok(db),
        ),
        Check(
            "the learned-skill narration frame fired (she narrates FROM the render)",
            _learned_frame_fired(db),
        ),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


# The SkillNarrationValidator's injected frame (Prompt.SKILL_LEARNED_NARRATION) —
# a stable slice; its presence in the prompt log proves the extraction fired and
# the model was handed the RENDERED recipe to narrate from.
_LEARNED_FRAME_MARKER = "You just learned a reusable skill"


def _learned_frame_fired(db: Database) -> bool:
    for row in db.messages.recent_prompts(limit=200):
        if row.messages and _LEARNED_FRAME_MARKER in row.messages:
            return True
    return False


def _extracted_skill_shape_ok(db: Database) -> bool:
    """Exactly one skill, carrying the demonstrated routine (browse + write among
    its steps) and parameterized (≥1 required hole — the URL/extract/key)."""
    skills = db.skills.list_all()
    if len(skills) != 1:
        return False
    step_tools = [step.tool for step in steps_from_json(skills[0].steps)]
    parameters = parameters_from_json(skills[0].parameters)
    return "browse" in step_tools and "collection_write" in step_tools and len(parameters) >= 1


@pytest.mark.asyncio
async def test_beat2_demonstrates_the_routine(chat_eval: ChatEval):
    """Beat 2, terminal state (as amended by #1658 auto-extraction): given the
    three bare steps ("read this url, find the price, remember it") the FIRST
    time — no dedicated container exists yet — Penny enacts them (browse →
    extract → write), the skill is AUTO-EXTRACTED from the run (no authoring
    tool), and she narrates what she learned from the injected render.  The
    '$499 came from the browse' guarantee holds by construction: the step names
    the price, never its value; the target collection is deliberately
    unconstrained, so we score "landed SOMEWHERE"."""
    await chat_eval(
        case_id="journey-beat2-demonstrate",
        message=_BEAT2_TURN,
        browse=[AURORA_LISTING_499],
        score=_score_beat2,
        min_pass_rate=None,  # report-only until sample-verified
        timeout=240.0,  # extraction + the narration re-reply (a live sample timed out at 120)
    )


# ── Beat 3: instantiate — the watch goes live, no machinery words ────────────
#
# The full simplified teach loop (#1658): instigate → teach-me → demonstrate
# (the skill AUTO-EXTRACTS from that run — no authoring tool, no "save that as
# a skill" handoff to trip on) → the user closes with pure INTENT: "keep
# watching it and let me know if the price ever changes."  She already knows
# how (find hits the auto-extracted skill / it renders ambiently), so THIS
# BEAT'S TERMINAL STATE: a LIVE dispatchable collection exists — the skill
# attached, its prompt rendered, writes retargeted to the new collection, a
# trigger set, notify on.  The user never says "skill" or "collection".

_BEAT3_TURNS = [
    _BEAT1_TURN,
    _BEAT2_TURN,
    "perfect — keep watching it and let me know if the price ever changes.",
]


def _live_watches(db: Database, before: set[str]) -> list:
    """Collections created THIS SAMPLE that are live, dispatchable watches: a
    skill attached AND a rendered prompt (inert storage has neither)."""
    return [
        row
        for row in new_collections(db, before)
        if row.skill_name is not None and row.extraction_prompt is not None
    ]


def _score_beat3(db: Database, before: set[str], reply: str) -> list[Check]:
    """The objective terminal state of the instantiate beat: the auto-extracted
    skill got attached to a live watch — trigger set, notify on, writes
    retargeted to the new collection.  The 'she says what the watch will do'
    verdict (the what-it-will-do echo narrated back) is read off the dumped
    transcript."""
    skills = db.skills.list_all()
    watches = _live_watches(db, before)
    watch = watches[0] if watches else None
    has_trigger = watch is not None and (
        watch.collector_interval_seconds is not None
        or watch.run_at is not None
        or watch.source_log is not None
    )
    retargeted = watch is not None and f"memory='{watch.name}'" in (watch.extraction_prompt or "")
    return [
        Check("the demonstration auto-extracted a skill (exactly one exists)", len(skills) == 1),
        Check(
            "ONE live watch was instantiated (skill attached, prompt rendered)", len(watches) == 1
        ),
        Check("the watch has a trigger (it will actually run)", has_trigger),
        Check("notify is on (the ask was 'let me know')", watch.notify if watch else False),
        Check(
            "writes retargeted to the new collection (the rendered program doesn't lie)",
            retargeted,
        ),
        Check("clean tool routing (no text-bail nudge fired)", not _bail_nudge_fired(db)),
    ]


@pytest.mark.asyncio
async def test_beat3_instantiates_the_watch(chat_eval: ChatEval):
    """Beat 3, terminal state (the #1658 world): after elicit → demonstrate
    (auto-extracted) → 'keep watching it and let me know', a LIVE watch exists —
    the skill attached to a new collection, prompt rendered with writes
    retargeted, a trigger set, notify on.  No machinery words anywhere in the
    user's turns; the skill and the collection are Penny's bookkeeping.

    The scored checks cover the objective terminal state; the 'she says what
    the watch will do' verdict is read off the dumped transcript."""
    await chat_eval(
        case_id="journey-beat3-instantiate",
        messages=_BEAT3_TURNS,
        browse=[AURORA_LISTING_499],
        score=_score_beat3,
        min_pass_rate=None,  # report-only until sample-verified
        timeout=240.0,  # the demonstrate turn carries extraction + the narration nudge
    )
