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
warm 0.96 · activity-window 1.00 · cold 1.00 · empty-registry 1.00).  Later
beats start REPORT-ONLY per the promote-later discipline and gate once
sample-verified.

UNIFORM CASE PATTERNS (2026-07-20, the code owner's consistency pass): every
check label carries one of three prefixes — ``state:`` (end DB/ledger facts),
``reply:`` (SAID==DID / reply-vs-state honesty), ``calls:`` (call provenance +
sequencing).  Scoring follows the state-is-core doctrine: state/reply checks
(and call checks that ARE a case's contract, e.g. dispatch fired) are scored;
sequencing annotations and the uniform loop-health check
(``routing_clean`` — bail + continue nudges, shared in conftest) are ADVISORY
(``Check(..., scored=False)`` — rendered in the report, excluded from the
score).  Since the routing check became advisory, a nudge-recovered beat-0
sample scores on its state/reply checks alone; real breakage still fails them.
Cases NEVER override prompts, clients, or tool surfaces — the harness runs the
real code and real prompts only (the artificial-prompt detour is the recorded
counter-example).
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    chat_run_tool_sequences,
    collection_entries,
    is_ordered_subsequence,
    new_collections,
    outgoing_replies,
    routing_clean,
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


def _score_beat0(db: Database, before: set[str], reply: str) -> list[Check]:
    created = new_collections(db, before)
    replies = outgoing_replies(db)
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
        Check(
            "state: the fact landed durably in a collection (any route)",
            fact_stored,
            kind="state",
        ),
        Check(
            "state: no runaway creation (at most one new collection)",
            len(created) <= 1,
            kind="state",
        ),
        Check(
            # A word-list proved brittle (live sample: a valid confirmation
            # phrased outside the list).  The honest signal is the FACT: a
            # turn-1 reply that restates the stored value is an acknowledgment;
            # claiming the fact while storage failed is the dishonest case.
            "reply: turn-1 reply acknowledges the fact it stored (SAID == DID)",
            fact_stored == ("499" in first_reply) if replies else False,
            kind="reply",
        ),
        Check("reply: read-back states $499", "499" in final_reply, kind="reply"),
        # NOTE: no hard provenance check here — answering a one-turn-old fact
        # from the conversation window is correct behavior (live sample 5).
        # The COLD variant below owns provenance absolutely.
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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
# empty — post-0097 this is nearly the DEFAULT world (no catch-alls).  "Remember X"
# must drive CREATION (the #1630 skill-optional inert create) + the write —
# the create arm of remember → collection_set-or-collection_write.


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
    replies = outgoing_replies(db)
    final_reply = replies[-1] if replies else ""
    entries = collection_entries(db, created[0].name) if len(created) == 1 else {}
    fact_stored = any("499" in content for content in entries.values())

    return [
        Check(
            "state: exactly one collection created (nowhere existed — she made one)",
            len(created) == 1,
            kind="state",
        ),
        Check("state: the fact landed in the created collection", fact_stored, kind="state"),
        Check("reply: read-back states $499", "499" in final_reply, kind="reply"),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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
#   run <id> · <when> · gear-notes → worked (2 calls) · wrote 'aurora deck 2
#   price' → `gear-notes`
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
    # Post-0097 there is no catch-all to lean on — the fact lives in a
    # user-shaped collection, created here like a real earlier turn would have.
    db.memories.create_collection("gear-notes", "notes about gear the user tracks")
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
                run_target="gear-notes",
                timestamp=when,
            )
        )
        session.add(
            MemoryEntry(
                memory_name="gear-notes",
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
    replies = outgoing_replies(db)
    final_reply = replies[-1] if replies else ""
    read_backed = any(
        (tool in _READ_TOOLS and args.get("memory") == "gear-notes") or tool == "find"
        for tool, args in _final_run_calls(db)
    )

    return [
        Check(
            "reply: recall states $499 (the write is ambient, value is not)",
            "499" in final_reply,
            kind="reply",
        ),
        Check(
            "calls: answer BACKED by a storage read (any route)",
            read_backed,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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

    db.memories.create_collection("gear-notes", "notes about gear the user tracks")
    db.memory("gear-notes").write(
        [EntryInput(key="aurora deck 2 price", content="$499")], author="chat"
    )


def _score_beat0_cold(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = outgoing_replies(db)
    final_reply = replies[-1] if replies else ""
    calls = _final_run_calls(db)
    read_backed = any(
        (tool in _READ_TOOLS and args.get("memory") == "gear-notes") or tool == "find"
        for tool, args in calls
    )

    return [
        Check(
            "reply: cold recall states $499 (storage is the only route)",
            "499" in final_reply,
            kind="reply",
        ),
        Check(
            "calls: answer BACKED by a storage read (find or a scoped read)",
            read_backed,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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


# ── Beat 1: the teach loop — no skill → elicit → demonstrate → live watch ────
#
# The instigating ask ("watch X and let me know when the price changes") names a
# WANT, not a routine, and the skill registry is EMPTY — so the designed shape is
# the full teach loop in one conversation (the code owner's canonical script,
# 2026-07-19):
#
#   1. user:  the watch ask
#   2. Penny: "I don't have a skill for that — teach me: what to read, what to
#      look for, what to remember"  (elicit; nothing enacted yet)
#   3. user:  the routine in ONE message ("first read <url>, then look for the
#      current price, then remember it")
#   4. Penny: enacts it — browse(url, extract) → collection_write(baseline) —
#      the framework auto-extracts + auto-attaches the skill at run end (#1658),
#      and she operationalizes the watch (collection_set: trigger + notify) and
#      reports honestly what now exists.
#
# Scoring follows the state-is-core doctrine: the SCORED checks are END DB STATE
# (what exists when the conversation ends); the call-spine checks are advisory
# flavour (scored=False — visible in the report, excluded from the score).  The
# reply's honesty — she claims exactly what the DB shows — is read off the dumped
# transcript in the joint review, never keyword-matched.

# Both turns carry the user's cadence word ('daily') — the learned notice's
# rule is "their own words are the trigger", so the words must exist to bind
# (2026-07-19: the run-2 s1 near-miss showed the notice correctly makes no
# call when the ask contains no cadence words).  The demo is FOUR steps:
# read the page, find the price, remember it, notify me if it changes daily.
_BEAT1_ASK = (
    f"can you watch the aurora deck 2 listing at {LISTING_URL} daily "
    "and let me know when the price changes?"
)
_BEAT1_DEMO = (
    f"sure — first read {LISTING_URL}, then look for the current price, "
    "then remember it, and notify me if it changes daily"
)

# Turn 1 is the elicitation: orientation reads are fine (checking for a matching
# skill is the right first instinct), but nothing may be enacted or configured
# before the user has taught the routine.
_ENACTING_TOOLS = (
    "browse",
    "collection_write",
    "collection_set",
    "update_entry",
    "collection_delete_entry",
    "log_append",
)


def _browse_extract_attributed(db: Database) -> bool:
    """The mechanism fingerprint: the price came through a browse-extract
    micro-context — an attributed ledger row, never page text the model pasted."""
    return any(
        row.agent_name == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
        for row in db.messages.recent_prompts(limit=200)
    )


def _score_beat1(db: Database, before: set[str], reply: str) -> list[Check]:
    """State is the pass/fail; the call spine is flavour.

    Scored checks read the END DB STATE — the durable facts of an operational
    watch (container, learned skill, attachment, trigger, notify, baseline entry,
    the browse-extract attribution).  scored=False checks read the per-run call
    sequences for phase discipline and routing health; they annotate the report
    without moving the score."""
    created = new_collections(db, before)
    watch = created[0] if len(created) == 1 else None
    entries = collection_entries(db, watch.name) if watch is not None else {}
    runs = chat_run_tool_sequences(db)
    first_run = runs[0] if runs else []
    demo_run = runs[1] if len(runs) > 1 else []
    trigger_set = watch is not None and (
        watch.collector_interval_seconds is not None
        or watch.cron_expression is not None
        or watch.run_at is not None
    )
    return [
        Check("state: exactly one collection created", len(created) == 1, kind="state"),
        Check(
            "state: a skill was learned (browse+write, parameterized)",
            _extracted_skill_shape_ok(db),
            kind="state",
        ),
        Check(
            "state: the skill is attached (collection carries skill + rendered prompt)",
            watch is not None
            and watch.skill_name is not None
            and watch.extraction_prompt is not None,
            anchor="collection_set(",
            kind="state",
        ),
        Check(
            "state: a trigger is set (the watch can actually run)",
            trigger_set,
            kind="state",
        ),
        Check(
            "state: notify is on (the user asked to hear about changes)",
            watch is not None and bool(watch.notify),
            kind="state",
        ),
        Check(
            "state: the baseline entry holds the browsed price",
            any("499" in content for content in entries.values()),
            anchor="collection_write(",
            kind="state",
        ),
        Check(
            "state: the price came through browse-extract (attributed ledger row)",
            _browse_extract_attributed(db),
            anchor="browse(",
            kind="state",
        ),
        Check(
            "state: the seeded collection untouched",
            not collection_entries(db, "dislikes"),
            kind="state",
        ),
        Check(
            "calls: turn 1 enacts nothing (elicit before the routine is taught)",
            not any(tool in _ENACTING_TOOLS for tool in first_run),
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: demo-turn spine browse → collection_write → collection_set",
            is_ordered_subsequence(["browse", "collection_write", "collection_set"], demo_run),
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_beat1_teach_loop(chat_eval: ChatEval):
    """Beat 1, the teach loop end-to-end: empty skill registry + a watch ask →
    Penny elicits the routine, the user demonstrates it in one message, she
    enacts it (the skill auto-extracts + auto-attaches at run end), and the
    watch goes live (trigger + notify).  See the canonical script above; the
    reply's honesty is reviewed off the transcript, not keyword-matched."""
    await chat_eval(
        case_id="journey-beat1-teach-loop",
        messages=[_BEAT1_ASK, _BEAT1_DEMO],
        browse=[AURORA_LISTING_499],
        score=_score_beat1,
        min_pass_rate=None,  # report-only until the rubric is jointly sample-verified
        timeout=240.0,  # the demo turn runs extraction + the narration continuation
    )


# ── Beat 1a: elicitation alone — the first exchange, REAL prompt ─────────────
#
# One exchange: the watch ask against an empty skill registry.  Terminal state:
# Penny voices the gap and asks to be taught — and the WORLD IS UNTOUCHED (no
# collection, no skill, no fetched page).  The ask itself is linguistic and is
# read off the transcript in the joint review; the scored checks are the end DB
# state, which for this beat is NOTHING CHANGED.
#
# HISTORY (2026-07-19): this case briefly ran under an ARTIFICIAL pared
# instruction block via the ``prepare`` hook.  The code owner's ruling ended
# that: the eval harness exists to test REAL Penny — real code, real prompts —
# and results under a synthetic prompt say nothing about production ("no more
# artificial prompt, that will never help us").  The levers learned in that
# detour inform the REAL prompt's clause review; nothing here swaps prompts.


def _score_beat1a(db: Database, before: set[str], reply: str) -> list[Check]:
    """Terminal state of the elicitation beat: the world is untouched.

    No container, no skill, no fetched page, no seeded-collection writes — she
    asked instead of acting.  Whether the reply IS the teach ask is read off the
    transcript in the joint review (one line of English, no structural signal)."""
    runs = chat_run_tool_sequences(db)
    first_run = runs[0] if runs else []
    fetched = db.memory("browse-results").read_recent(window_seconds=3600, cap=None)
    return [
        Check(
            "state: no collection created (nothing enacted)",
            not new_collections(db, before),
            kind="state",
        ),
        Check(
            "state: no skill learned (no round ran)",
            not db.skills.list_all(),
            kind="state",
        ),
        Check(
            "state: no page fetched (browse-results stayed empty)",
            not fetched,
            kind="state",
        ),
        Check(
            "state: the seeded collection untouched",
            not collection_entries(db, "dislikes"),
            kind="state",
        ),
        Check(
            "calls: no enacting calls (orientation reads only)",
            not any(tool in _ENACTING_TOOLS for tool in first_run),
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_beat1a_elicits(chat_eval: ChatEval):
    """Beat 1a, REAL prompt: with an empty skill registry, the watch ask ends
    in a teach-me reply and an untouched world — no browse, no write, no faked
    setup.  One exchange, the end."""
    await chat_eval(
        case_id="journey-beat1a-elicit",
        message=_BEAT1_ASK,
        browse=[AURORA_LISTING_499],
        score=_score_beat1a,
        min_pass_rate=None,  # report-only until the rubric is jointly sample-verified
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
    replies = outgoing_replies(db)
    return [
        Check(
            "state: she browsed the listing (step 1 — the demonstrated fetch happened)",
            _browsed_the_listing(db),
            kind="state",
        ),
        Check(
            "state: the browsed value ($499) landed durably in a collection "
            "(steps 2+3, any collection)",
            value_stored,
            kind="state",
        ),
        Check(
            # SAID == DID via the acknowledge-the-fact pattern (verb lists proved
            # brittle three times).  The narration flow sends TWO replies per
            # teach turn (the routine report + the learned-skill narration), so
            # the echo is checked across ALL sent replies, not just the first.
            "reply: a sent reply reports the browsed value it stored (SAID == DID)",
            (any("499" in reply for reply in replies) == value_stored) if replies else False,
            kind="reply",
        ),
        Check(
            # The auto-extraction (#1658): the demonstration run ITSELF yields the
            # skill — deterministically, at the run-end chokepoint, no authoring
            # tool.  The demonstrate turn's terminal state includes it.
            "state: a skill was auto-extracted from the demonstration "
            "(browse+write, parameterized)",
            _extracted_skill_shape_ok(db),
            kind="state",
        ),
        Check(
            "state: the learned-skill narration frame fired (she narrates FROM the render)",
            _learned_frame_fired(db),
            kind="state",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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
    _BEAT1_ASK,
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
        Check(
            "state: the demonstration auto-extracted a skill (exactly one exists)",
            len(skills) == 1,
            kind="state",
        ),
        Check(
            "state: ONE live watch was instantiated (skill attached, prompt rendered)",
            len(watches) == 1,
            kind="state",
        ),
        Check(
            "state: the watch has a trigger (it will actually run)",
            has_trigger,
            kind="state",
        ),
        Check(
            "state: notify is on (the ask was 'let me know')",
            watch.notify if watch else False,
            kind="state",
        ),
        Check(
            "state: writes retargeted to the new collection (the rendered program doesn't lie)",
            retargeted,
            kind="state",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
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


# ── Beat 2b: the fused ask DECOMPOSES into teach-then-schedule (field shape) ──
#
# The first external deployment exposed the FUSED ask our imperative-shaped
# beat 2 structurally couldn't handle: one message carrying sources, cadence,
# and filter — never an imperative.  Observed live: she elicited page mechanics
# ("the snippet and pattern for one of the URLs") and PLANNED forever; no round
# ever ran.  Four prompt iterations at demanding a SELF-STARTED round moved
# nothing (writes 1/8 — the conversational prior at a descriptive message is
# propose-and-confirm, and doctrine prose loses to it).  The adopted design
# HARNESSES that prior instead: teaching and instantiation are two separate
# things, so the correct response to a fused ask is to SPLIT IT OUT LOUD —
# "teach me the find first: give me the whole routine in one message (example
# modelled from THEIR parameters) — then I'll run it on your schedule."  The
# user's routine reply manufactures the imperative the enact machinery (beat 2,
# 0.90) fires on; the schedule binds at the attach step (beat 3, 1.00).
# (All fixture content here is synthetic; the shape is what's real.)
#
# THIS BEAT'S TERMINAL STATE, turn by turn: (1) the fused ask gets the
# decompose response — the routine requested in ONE message, example modelled
# from their sources, no mechanics demands, no schedule design; (2) the
# routine arrives → she runs it — browse, extract in plain language, WRITE —
# and the skill auto-extracts from that round; (3) "now do that morning and
# evening" → a live watch: skill attached, trigger set, notify on.

FOXES_URL = "https://www.ridgelinefoxes.com/news"
SEALS_URL = "https://www.harborseals.com/news"

FOXES_NEWS_PAGE = CannedPage(
    match="ridgelinefoxes",
    text=(
        "Title: Ridgeline Foxes | Official Site — Team News\n"
        f"{FOXES_URL}\n\n"
        "Foxes sign veteran goalie Aurelio Brandt to a two-year deal — the club "
        "confirmed the signing Thursday morning.\n"
        "Final score: Foxes 3, Rovers 2 (overtime).\n"
        "Training camp opens next month at Ridgeline Arena.\n"
    ),
)

SEALS_NEWS_PAGE = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor Seals | Official Site — Team News\n"
        f"{SEALS_URL}\n\n"
        "Seals name Petra Volk head of player development after a lengthy search.\n"
        "Final score: Seals 1, Gulls 4.\n"
        "Season ticket renewals open Friday.\n"
    ),
)

# Turn 1 = the FUSED ask (sources + filter + schedule, no imperative — the
# field shape verbatim).  Turn 2 = the user's routine, the answer the decompose
# ask requests (imperative steps; the URLs referenced, not retyped — a real
# user doesn't repeat themselves).  Turn 3 = pure schedule intent.
_BEAT2B_TURNS = [
    (
        "hey can you set up news alerts for my favourite teams? the ridgeline "
        f"foxes and the harbor seals — their news pages are {FOXES_URL} and "
        f"{SEALS_URL}. check them twice a day, and alert me about "
        "notable stuff like trades, signings, and injuries — not game scores."
    ),
    (
        "sure: 1. go to those two news pages 2. pull out any trades, signings, "
        "or injuries — skip game scores 3. remember the title plus a short "
        "blurb for each"
    ),
    ("perfect — now do that twice a day and let me know when something new shows up."),
]

# Page-mechanics demands — the observed stall vocabulary.  Present in a SENT
# reply = she's asking the user to do her reading for her.
# "snippet" deliberately absent: live replies use it benignly ("scan the
# headlines or snippets for keywords") — the field regression's demand shape is
# structural vocabulary (selectors/xpath/css/parse patterns), which never is.
_MECHANICS_DEMANDS = (
    "selector",
    "xpath",
    "css",
    "html pattern",
    "parse pattern",
    "laid out",
    "layout",
)


def _demanded_mechanics(replies: list[str]) -> bool:
    text = " ".join(replies).lower()
    return any(term in text for term in _MECHANICS_DEMANDS)


def _round_ran(db: Database) -> bool:
    """The self-started round's browse is persisted in browse-results."""
    entries = db.memory("browse-results").read_recent(window_seconds=3600, cap=None)
    return any(
        "ridgelinefoxes" in entry.content or "harborseals" in entry.content for entry in entries
    )


def _notable_written(db: Database, before: set[str]) -> bool:
    """A round's write carries page-derived content — 'Brandt' exists ONLY in the
    fixture pages, so a stored copy proves browse → extract → write."""
    stored = _all_collection_writes(db, before)
    # Page-ONLY tokens (never in the user's turns), so a truncated-but-real write
    # ("Foxes sign veteran…") still counts and a fabricated sample entry doesn't.
    page_tokens = ("brandt", "volk", "aurelio", "petra", "veteran", "goalie", "player development")
    return any(
        any(token in content.lower() for token in page_tokens)
        for entries in stored.values()
        for content in entries.values()
    )


# The teach-ask family — how she asks to be walked through the round.  Broad by
# design (live samples said "teach me a single round" / "a quick walkthrough of
# one round", never the scripted "one message" literally); the semantics scored
# is one-complete-pass elicitation, not the exact phrase.
_TEACH_ASK_TOKENS = (
    "teach me",
    "walk me through",
    "walkthrough",
    "one message",
    "single message",
    "one round",
    "single round",
    "one pass",
    "one complete pass",
)


def _teach_ask(text: str) -> bool:
    return any(t in text.lower() for t in _TEACH_ASK_TOKENS)


def _decompose_ask(replies: list[str]) -> tuple[bool, bool]:
    """Turn-1 verdicts on the FIRST reply: (she recognized she can't act yet and
    asked to be taught the round — the one-complete-pass elicitation; the example
    was modelled from THEIR sources, so 'yes, do that' is a complete answer)."""
    if not replies:
        return False, False
    first = replies[0].lower()
    modelled = any(t in first for t in ("ridgelinefoxes", "harborseals", "foxes", "seals"))
    return _teach_ask(first), modelled


def _score_beat2b(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = outgoing_replies(db)
    one_message, modelled = _decompose_ask(replies)
    watches = _live_watches(db, before)
    watch = watches[0] if watches else None
    has_trigger = watch is not None and (
        watch.collector_interval_seconds is not None
        or watch.run_at is not None
        or watch.source_log is not None
    )
    # The chain completing WITHOUT an ask is spontaneous one-shot success — the
    # end goal, not a failure.  The ask is scored as the fallback: required only
    # when she could not run the routine herself; the modelled-example facet
    # applies only to an ask that actually happened.
    chain_complete = _notable_written(db, before) and bool(watches)
    return [
        Check(
            "reply: she asked to be taught the round OR ran it herself to completion",
            one_message or chain_complete,
            kind="reply",
        ),
        Check(
            "reply: an ask that happened was modelled from THEIR sources",
            modelled if one_message else True,
            kind="reply",
        ),
        Check(
            "state: the routine ran on arrival (browsed the given sites)",
            _round_ran(db),
            kind="state",
        ),
        Check(
            "state: the round's write landed (page-derived content stored)",
            _notable_written(db, before),
            kind="state",
        ),
        Check(
            "state: a skill auto-extracted from the round",
            len(db.skills.list_all()) >= 1,
            kind="state",
        ),
        Check(
            "state: a live watch exists (skill attached, prompt rendered)",
            len(watches) >= 1,
            kind="state",
        ),
        Check(
            "state: the watch has a trigger and notify is on",
            has_trigger and bool(watch.notify if watch else False),
            kind="state",
        ),
        Check(
            "reply: no re-teach ask once the skill exists (final reply doesn't re-elicit)",
            not (_teach_ask(replies[-1]) if replies else False),
            kind="reply",
        ),
        Check(
            "reply: she never demanded page mechanics (snippets/selectors/patterns)",
            not _demanded_mechanics(replies),
            kind="reply",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_beat2b_fused_ask_decomposes(chat_eval: ChatEval):
    """Beat 2b, terminal state (the split sequencing): the FUSED field-shaped ask
    gets the decompose response — the routine requested in ONE message, the
    example modelled from their sources — the user's routine reply is enacted on
    arrival (browse → extract → write, skill auto-extracts), and the closing
    schedule intent instantiates a live watch (skill attached, trigger, notify).
    Teaching and instantiation are two separate things; the decompose ask is
    what routes a fused ask onto the two proven paths.  The 'learned it and
    ready to put it on a schedule' narration after the enact turn is read off
    the dumped transcript (the beat-1/beat-3 pattern for linguistic facets)."""
    await chat_eval(
        case_id="journey-beat2b-decompose",
        messages=_BEAT2B_TURNS,
        browse=[FOXES_NEWS_PAGE, SEALS_NEWS_PAGE],
        score=_score_beat2b,
        min_pass_rate=None,  # report-only until the reshaped sequencing is sample-verified
        timeout=300.0,  # three turns: decompose + enact-with-extraction + instantiate
    )


# ── Beat 2c: the ONE-SHOT — routine + schedule in a single message ───────────
#
# The fast path the split sequencing earns: when the user's single message
# already carries the routine AS STEPS plus the schedule, no decompose ask is
# needed — she enacts immediately (case a), the skill auto-extracts, the
# framework auto-attaches it to the collection the round created, and the
# learned notice hands her the one remaining decision: bind the user's schedule
# words as the trigger + notify.  Terminal state = a live watch in ONE turn.

_BEAT2C_TURN = (
    "hey penny — twice a day, check the ridgeline foxes and "
    f"harbor seals news pages ({FOXES_URL} and {SEALS_URL}): 1. go to both "
    "pages 2. pull out any trades, signings, or injuries — skip game scores "
    "3. remember the title plus a short blurb for each — and let me know when "
    "something new shows up."
)


def _score_beat2c(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = outgoing_replies(db)
    watches = _live_watches(db, before)
    watch = watches[0] if watches else None
    has_trigger = watch is not None and (
        watch.collector_interval_seconds is not None
        or watch.run_at is not None
        or watch.source_log is not None
    )
    return [
        Check(
            "state: the routine ran (browsed the given sites)",
            _round_ran(db),
            kind="state",
        ),
        Check(
            "state: the round's write landed (page-derived content stored)",
            _notable_written(db, before),
            kind="state",
        ),
        Check(
            "state: a skill auto-extracted from the round",
            len(db.skills.list_all()) >= 1,
            kind="state",
        ),
        Check(
            "state: a live watch exists (skill attached, prompt rendered)",
            len(watches) >= 1,
            kind="state",
        ),
        Check(
            "state: the watch has a trigger and notify is on",
            has_trigger and bool(watch.notify if watch else False),
            kind="state",
        ),
        Check(
            "reply: no teach-ask for a routine already in hand",
            not any(_teach_ask(r) for r in replies),
            kind="reply",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_beat2c_one_shot_fused_routine(chat_eval: ChatEval):
    """Beat 2c, terminal state: a single message carrying the routine as steps
    PLUS the schedule yields a live watch in ONE turn — enact (case a), skill
    auto-extracts, framework auto-attaches to the round's own collection, and
    the learned notice's config affordance binds trigger + notify.  The reverse
    detection direction is scored too: a routine already in hand must never be
    re-elicited."""
    await chat_eval(
        case_id="journey-beat2c-one-shot",
        message=_BEAT2C_TURN,
        browse=[FOXES_NEWS_PAGE, SEALS_NEWS_PAGE],
        score=_score_beat2c,
        min_pass_rate=None,  # report-only until sample-verified
        timeout=300.0,  # enact + extraction + auto-attach + the config continuation
    )
