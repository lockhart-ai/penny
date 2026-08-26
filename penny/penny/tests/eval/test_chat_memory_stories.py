"""The chat memory stories — what a user asks Penny's memory to do, end to end.

Each case drives the REAL chat path against the live model with the user's own words
and scores PERSISTED state: the collections, the entries and their run stamps, the
pages the browse log holds, the machine's own move, and the replies Penny sent.  Six
stories, in the order a user meets them:

    1. remember it        "remember X for me"  → it is stored, and it comes back later
                          (same conversation · a cold next session · off the ambient
                          activity line · across the whole store)
    4. look it up         "look it up and save it" → browse then write (one hop, two
                          hops, the check-then-fill, and the lookup that UPDATES)
    5. change it          "forget X" / "change my note" → exactly that entry, the rest
                          untouched
    6. leave it alone     a passing mention is not an instruction — nothing fetched,
                          nothing written
   11. two facts at once  one message carrying a like and a dislike → two writes, each
                          into the collection that fits it
   15. two sources        the fused ask for a routine over TWO pages — the estate's
                          only two-page routine: decompose, demonstrate, set running —
                          and the learn close, where the reply states what the routine
                          it just learned will RUN, for the one person who can tell a
                          captured step from a stray one (#1943)

The worlds are the CURRENT runtime's.  Nothing is pre-seeded any more (migration
0108), so every collection a case starts with is one this file creates the way the
user's own earlier turn would have — through ``create_collection`` and the
collection's own ``write``, stamped with a seeded run id so "what did THIS sample
write" stays answerable (``is_seeded_run``).  The old ``likes`` / ``dislikes``
catch-alls are gone and no case leans on them.

WHERE a write lands is deliberately open where the story does not fix it.  A turn
carrying an instruction ("remember it", "save it to my games list") lands the machine
in ``learn``, and a learn round is FRAMED on the way in — Penny already has a
container of the round's own before the first call — so a scorer that demanded one
named destination would report the framework's own routing as a model failure.  So
the durable checks read what THIS RUN wrote wherever it landed, and the destination is
reported beside them.  Story 11 is the exception, because there the destination IS the
story: two facts, two fitting places.

Check labels carry one of three prefixes — ``state:`` (end DB facts), ``reply:``
(what Penny said, against what she did), ``calls:`` (call provenance).  State and
reply checks are SCORED; the call spine, the state the machine landed in, and the
loop-health verdict are ADVISORY (``scored=False``) unless that row IS the story's own
claim — a lookup before a write is one, since a write with no browse is a fact that
came from nowhere, and the learn close's LANDING is the other, since a story about the
reply that ends a learn round has no subject at all outside that state (#1989).  Where
a scored claim only exists in a state a sample never reached, it reads ``Check.na(...)``
rather than ❌: a precondition nobody met is not a contract anybody failed.

Every case is REPORT-ONLY (``min_pass_rate=None``): the thresholds are the code
owner's to set once the numbers are read.  All content is synthetic — invented games,
invented teams, invented markets — because the repo is public.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest
from sqlmodel import Session

from penny.agents.self_state import SelfStateHeader
from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState, MachineSnapshot
from penny.database import Database
from penny.database.memory import EntryInput, MemoryType
from penny.database.models import MemoryEntry, MemoryRow, PromptLog, Skill
from penny.database.skill_store import steps_from_json
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory

# The enacting-tool set is read from the suite's shared fixtures, not restated here: the
# state machine's elicitation edge asks the same question of a turn (nothing acted on
# before it was taught), and one policy in two copies is two contracts.
from penny.tests.eval.cohort import (
    BASE_WORLD,
    CohortArm,
    unsourced_specifics,
)
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    asked_for_page_structure,
    chat_run_tool_sequences,
    collection_entries,
    describes,
    given_to_the_model,
    is_ordered_subsequence,
    is_seeded_run,
    new_collections,
    outgoing_replies,
    routing_clean,
    seeded_run_id,
    tool_call_arg_values,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    ENACTING_TOOLS,
    MULTIHOP_PAGES,
    CannedPage,
    SynthCollection,
)

# Standing a ROUND up before the measured turn is the transition suite's idiom, read from
# where that suite declares it rather than restated here: a seeded machine state, a seeded
# conversation turn and a seeded ledger row are one shape, and a second copy of it would be
# a second contract free to drift from the one every edge case is measured against.
#
# Its PROBES are deliberately restated instead (``_assert_parked_in_elicit`` below): that
# suite's are keyed to its own ``_ElicitRound`` case type, which this file has no shape
# for.  What a probe asserts is the seed it stands beside, so the honest cost of not
# widening a neighbour's fixture type is one restated probe, named here rather than left
# for a reader to notice.
from penny.tests.eval.test_state_transitions import (
    _drawn_state,
    _log_ask,
    _log_chat_step,
    _log_classifier_draw,
    _log_reply,
    _park,
    _seeded_response,
)
from penny.tools.skill_tools import render_skill_shape

pytestmark = pytest.mark.eval

_FAMILY = "chat-memory"

# The reads a scorer asks "was the answer backed by storage" through.  ``find`` is the
# guess-free route and carries no ``memory`` argument, so it is checked by name alone.
_SCOPED_READ_TOOLS = ("collection_read_latest", "collection_get", "read_similar")
_FIND_TOOL = "find"

_BROWSE_TOOL = "browse"
_WRITE_TOOL = "collection_write"
_UPDATE_TOOL = "update_entry"
_DELETE_TOOL = "collection_delete_entry"

# The run ids the seeded worlds write under: a collection the user built in an earlier
# session, and the recent run whose write the activity line renders.  Both are seeded
# ids, so ``live_prompts`` and the run-stamp readers below tell them from this sample's
# own work (#1846).
_EARLIER_SESSION = seeded_run_id("earlier-session")
_RECENT_RUN = seeded_run_id("recent-run")


# ── The collections the stories start from (all user-built) ──────────────────

_GAMES = SynthCollection(
    "games",
    "Video games the user is tracking or interested in: titles, release dates, and notes.",
    entries=(),
)

# The one game the lookup stories are about — invented, so it exists on no real site
# and cannot be answered from what the model already knows.  The key is the text before
# the em dash, the way every seeded entry in this suite is written.
_MISTFORGE_ENTRY = "Mistforge Tactics — a turn-based strategy game."

_GAMES_WITH_MISTFORGE = SynthCollection(
    _GAMES.name,
    _GAMES.description,
    entries=(_MISTFORGE_ENTRY,),
)

_INTO = SynthCollection(
    "things-im-into",
    "Things the user is into: hobbies, music, and the ways they like to spend their time.",
    entries=(
        "chess — enjoys playing chess",
        "hiking — loves weekend hikes",
        "jazz — a big fan of jazz records",
    ),
)

_AVOID = SynthCollection(
    "things-i-avoid",
    "Things the user would rather avoid: places, sounds, and food they dislike.",
    entries=("loud offices — can't focus in them",),
)

_GEAR_NOTES = SynthCollection(
    "gear-notes",
    "Notes about gear the user tracks: what a thing is, and what it was listed at.",
    entries=(),
)
_GEAR_PRICE_KEY = "aurora deck 2 price"
_GEAR_PRICE = "$499"
# The probe is the FIGURE, not the formatted value: a stored "499" and a stored "$499"
# are the same fact remembered, and a scorer that demanded the currency symbol would
# report a correct memory as a miss.
_PRICE_FACT = "499"


def _seed_collection(db: Database, synth: SynthCollection, *, run_id: str) -> None:
    """Lay down a collection the user built in an earlier session, through the
    production paths a chat turn would have used: the store's own
    ``create_collection`` and the collection's own ``write``.

    The run stamp is what gives the seed entrance-condition fidelity — production
    stamps every write with the run that made it (#1560), so an unstamped seed would
    make "stamped at all" mean "written by this sample" and every reader below would
    read a handed-down entry as this turn's work."""
    db.memories.create_collection(synth.name, synth.description, created_by_run_id=run_id)
    require_memory(db, synth.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in synth.entries],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=run_id,
    )


# ── Reading what the sample did ──────────────────────────────────────────────


def _written_by_this_run(entry: MemoryEntry) -> bool:
    """Whether THIS sample put an entry's current value there — created by a live run,
    or last rewritten by one.  Both stamps, because an edit of a seeded entry moves
    only ``last_written_by_run_id``."""
    stamps = (entry.created_by_run_id, entry.last_written_by_run_id)
    return any(stamp is not None and not is_seeded_run(stamp) for stamp in stamps)


def _entries_this_run_wrote(db: Database) -> list[tuple[str, MemoryEntry]]:
    """``(collection, entry)`` for every COLLECTION entry this sample wrote, wherever
    it landed — the run-id stamp answering "what did she store", so a case never has
    to guess which container a framed round used.

    Collections only: the browse log carries the fetched page, and counting that as a
    stored fact would let "she wrote it down" pass on a run that only read a page."""
    written: list[tuple[str, MemoryEntry]] = []
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [(row.name, entry) for entry in entries if _written_by_this_run(entry)]
    return written


def _normalize(text: str) -> str:
    """Fold the typography gpt-oss sprinkles into its output so a SEMANTIC substring
    probe isn't defeated by cosmetics: unicode hyphens → '-', nbsp/zero-width/narrow
    spaces → ' ', bold markers stripped, curly quotes straightened, lowercased.  (A
    0/N from an un-normalized probe is a scorer bug — the model wrote 'co‑op' /
    'Mist​forge', semantically right.)"""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for source, target in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(source, target)
    return folded


def _entry_text(entry: MemoryEntry) -> str:
    """An entry's KEY and CONTENT, normalized and joined — the probe for "did this
    fact land here", robust to which half the model put the fact in (one measured
    sample keyed ``mistforge_tactics`` and stylized the body, so contents alone
    missed it)."""
    return _normalize(" ".join(text for text in (entry.key, entry.content) if text))


def _stored_text(db: Database, name: str) -> str:
    """Every key and content a collection currently holds, normalized and joined."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _landed_in(db: Database, fact: str) -> set[str]:
    """The collections where THIS run's own writes carry ``fact``."""
    needle = _normalize(fact)
    return {name for name, entry in _entries_this_run_wrote(db) if needle in _entry_text(entry)}


def _pages_fetched(db: Database) -> list[MemoryEntry]:
    """Every page this sample read — the browse log's recent window."""
    return require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).read_recent(
        window_seconds=3600, cap=None
    )


def _fetched(db: Database, token: str) -> bool:
    """Whether a page carrying ``token`` was actually read this sample."""
    return any(token in entry.content for entry in _pages_fetched(db))


def _read_backed(db: Database, name: str) -> bool:
    """Whether the answer was backed by a storage read: ``find`` (the guess-free
    route) or a read aimed at the collection holding the fact."""
    if tool_was_called(db, _FIND_TOOL):
        return True
    targets = {
        target for tool in _SCOPED_READ_TOOLS for target in tool_call_arg_values(db, tool, "memory")
    }
    return name in targets


def _enacting_calls(db: Database) -> list[str]:
    """Every enacting call this sample made, in order — what a no-fire guard names
    when it fails, and what an elicitation turn must not carry."""
    return [tool for run in chat_run_tool_sequences(db) for tool in run if tool in ENACTING_TOOLS]


# ── The advisory rows every case carries ─────────────────────────────────────


def _landed_state(db: Database) -> str | None:
    latest = db.machine.latest_transition()
    return latest.to_state if latest else None


def _walked(db: Database) -> str:
    """The machine's walk this sample, oldest move first — ``idle→learn, learn→apply``."""
    moves = reversed(db.machine.recent_transitions(limit=20))
    return ", ".join(f"{move.from_state}→{move.to_state}" for move in moves) or "no move"


def _landing_advisory(
    db: Database, expected: ConversationState | None = None, *, scored: bool = False
) -> Check:
    """Where the machine ended up, reported beside the story's own checks.

    A turn carrying an instruction lands in ``learn`` and mints a routine at run end;
    a question lands in idle.  Which of those a given phrasing is belongs to the state
    definitions rather than to a memory story, so this row REPORTS the landing (and
    the whole walk) and only ``expected`` — set where the story does turn on it —
    makes it a verdict.

    ``scored`` promotes that verdict into the graded denominator, for the one story whose
    contract only EXISTS in the state it names (#1989): the learn close is a claim about
    the reply that ends a learn round, so where the machine landed is that story's
    precondition, and reporting it unscored beside scored reply checks put the accurate
    signal out of the score and the misleading one in it."""
    landed = _landed_state(db)
    ok = landed == expected.value if expected is not None else landed is not None
    return Check(
        "calls: where the machine landed",
        ok,
        rationale=f"walked {_walked(db)}",
        scored=scored,
        kind="spine",
    )


def _routing_advisory(db: Database) -> Check:
    return Check(
        "calls: clean routing (no re-rolled draw or continue nudge)",
        routing_clean(db),
        scored=False,
        kind="proc",
    )


def _reply_names(reply: str, tokens: tuple[str, ...]) -> list[Check]:
    """The reply must NAME what it acted on (the #1478 recap prong): one graded check
    per subject, anchored to the reply row, matched as a normalized substring — never
    against a wording list, which proved brittle three times over."""
    normalized = _normalize(reply)
    return [
        Check(
            f"reply: names '{token}'",
            _normalize(token) in normalized,
            kind="reply",
            anchor=REPLY_ANCHOR,
        )
        for token in tokens
    ]


# ═══ Story 1 — "remember it" ═════════════════════════════════════════════════
#
# The atom every other story stands on: a fact the USER states is stored, and comes
# back when it is asked for.  Four depths, in order of how little conversation is left
# to answer from — the same conversation, a previous session, an ambient run line, and
# a sweep of the whole store.  No browse fixture is installed for the first three, so
# a value in the reply can only have come out of storage or out of the conversation,
# never off a page.

_REMEMBER_TURNS = [
    "hey, can you remember that the aurora deck 2 is listed at $499 for me?",
    "thanks — what did I say the aurora deck 2 was listed at?",
]


def _score_remember_and_recall(db: Database, before: set[str], reply: str) -> list[Check]:
    """The registry starts EMPTY (nothing is pre-seeded since migration 0108), so
    there is nowhere for the fact to go and making somewhere is part of the story —
    which is what the deleted empty-registry variant used to measure separately.

    No provenance check on the read-back: answering a one-turn-old fact from the
    conversation window is correct behaviour, and the cold case below owns provenance
    absolutely."""
    created = new_collections(db, before)
    replies = outgoing_replies(db)
    stored = _landed_in(db, _PRICE_FACT)
    first_reply = replies[0] if replies else ""
    final_reply = replies[-1] if replies else ""
    return [
        Check(
            "state: the fact landed durably in a collection",
            bool(stored),
            rationale=None if stored else "no collection holds it",
            kind="state",
        ),
        Check(
            "state: exactly one collection was created (there was nowhere to put it)",
            len(created) == 1,
            rationale=f"created {[row.name for row in created]}",
            kind="state",
        ),
        Check(
            # A word list proved brittle (a valid confirmation phrased outside it).
            # The honest signal is the FACT: a turn-1 reply restating the stored value
            # is an acknowledgment, and claiming it while storage failed is the
            # dishonest case this scores against.
            "reply: turn 1 claims exactly what it stored",
            bool(replies) and (bool(stored) == (_PRICE_FACT in first_reply)),
            kind="reply",
        ),
        Check(
            "reply: the read-back states the price",
            _PRICE_FACT in final_reply,
            kind="reply",
        ),
        Check(
            "state: the fact landed in one place, not several",
            len(stored) <= 1,
            rationale=f"landed in {sorted(stored)}" if len(stored) > 1 else None,
            kind="state",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_remembered_fact_comes_back_in_the_same_conversation(
    chat_eval: ChatEval,
) -> None:
    """Story 1, the atom: "remember X" lands the fact somewhere durable and the next
    turn answers with it — with no page installed, so nothing could have been looked
    up instead."""
    await chat_eval(
        case_id="memory-remember-and-recall",
        messages=_REMEMBER_TURNS,
        score=_score_remember_and_recall,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ── 1b: the cold next session — storage is the only route ───────────────────
#
# The fact was stored in a PREVIOUS session, so no conversation carries it and an echo
# is impossible: the answer exists only in the store.  This is the n≤1 invariant's
# absolute test — the entry is reached through ``find`` or a correctly-aimed read —
# so unlike the warm case above, provenance is scored.

_COLD_TURN = (
    "hey — a while back I asked you to remember what the aurora deck 2 "
    "was listed at. what was the price?"
)


def _seed_cold_fact(db: Database) -> None:
    """The collection and the entry an earlier session left behind, stamped with that
    session's run."""
    _seed_collection(db, _GEAR_NOTES, run_id=_EARLIER_SESSION)
    require_memory(db, _GEAR_NOTES.name).write(
        [EntryInput(key=_GEAR_PRICE_KEY, content=_GEAR_PRICE)],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=_EARLIER_SESSION,
    )


def _score_cold_recall(db: Database, before: set[str], reply: str) -> list[Check]:
    backed = _read_backed(db, _GEAR_NOTES.name)
    return [
        Check(
            "reply: the recall states the price",
            _PRICE_FACT in reply,
            kind="reply",
            anchor=REPLY_ANCHOR,
        ),
        Check(
            "calls: the answer is backed by a storage read",
            backed,
            rationale=None if backed else f"called {tool_call_sequence(db)}",
            kind="spine",
        ),
        Check(
            "state: nothing was written to answer a question",
            not _entries_this_run_wrote(db),
            kind="state",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_fact_from_a_previous_session_comes_back(chat_eval: ChatEval) -> None:
    """Story 1, cold: a fact stored in a previous session is retrieved with zero
    conversational trace — the absolute test of one-call reachability."""
    await chat_eval(
        case_id="memory-cold-recall",
        message=_COLD_TURN,
        seed=_seed_cold_fact,
        score=_score_cold_recall,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ── 1c: off the ambient activity line ────────────────────────────────────────
#
# The fact was written by a RECENT RUN rather than by a conversation, so the self-state
# activity block renders the write ambiently (#1641):
#
#   run <id> · <when> · gear-notes → WORKED (2 calls) · wrote 'aurora deck 2 price'
#   → `gear-notes`
#
# Awareness costs zero calls and the retrieval is one call whose arguments are both
# on that line, verbatim.  The VALUE is not: the line names the key and the collection
# and nothing else, so a reply carrying the price had to read the entry.  The probe
# holds that premise against the SHIPPED render — if the clause ever stops rendering,
# this case must say so rather than quietly becoming a second cold-recall case.

_ACTIVITY_TURN = "hey — remind me, what was the aurora deck 2 listed at?"


def _seed_recent_run_write(db: Database) -> None:
    """A completed run 20 minutes ago that wrote the fact — the row the activity block
    renders, plus the entry it wrote, joined by the run id exactly as production joins
    them."""
    when = datetime.now(UTC) - timedelta(minutes=20)
    response = '{"choices": [{"message": {"tool_calls": [{"id": "0"}, {"id": "1"}]}}]}'
    _seed_collection(db, _GEAR_NOTES, run_id=_RECENT_RUN)
    with Session(db.engine) as session:
        session.add(
            PromptLog(
                model="test-model",
                messages="[]",
                response=response,
                agent_name=PennyConstants.CHAT_AGENT_NAME,
                run_id=_RECENT_RUN,
                run_outcome="worked",
                run_reason="",
                run_target=_GEAR_NOTES.name,
                timestamp=when,
            )
        )
        session.add(
            MemoryEntry(
                memory_name=_GEAR_NOTES.name,
                key=_GEAR_PRICE_KEY,
                content=_GEAR_PRICE,
                author=PennyConstants.CHAT_AGENT_NAME,
                created_at=when,
                created_by_run_id=_RECENT_RUN,
                last_written_by_run_id=_RECENT_RUN,
            )
        )
        session.commit()


def _probe_activity_line(penny: Penny) -> None:
    """The world really does render the write on the activity line — asserted through
    the shipped renderer, never a restatement of it, so the case cannot pass on a
    premise that has silently gone away."""
    rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
    clause = SelfStateHeader._collection_writes(_GEAR_NOTES.name, [_GEAR_PRICE_KEY])
    assert clause in rendered, (
        f"memory-activity-window-recall: the seeded write must render on the activity "
        f"line as {clause!r} — the header reads:\n{rendered}"
    )
    assert _GEAR_PRICE not in rendered, (
        "memory-activity-window-recall: the activity line must NOT carry the value — "
        "otherwise the reply proves nothing about the read"
    )


def _score_activity_window_recall(db: Database, before: set[str], reply: str) -> list[Check]:
    backed = _read_backed(db, _GEAR_NOTES.name)
    return [
        Check(
            "reply: the recall states the price (the write is ambient, the value is not)",
            _PRICE_FACT in reply,
            kind="reply",
            anchor=REPLY_ANCHOR,
        ),
        Check(
            "calls: the answer is backed by a storage read",
            backed,
            rationale=None if backed else f"called {tool_call_sequence(db)}",
            kind="spine",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_fact_a_recent_run_wrote_comes_back(chat_eval: ChatEval) -> None:
    """Story 1, ambient: the run line names the key and the collection, so the read is
    one call with both arguments copied rather than guessed — and the value, which the
    line withholds, comes back in the reply."""
    await chat_eval(
        case_id="memory-activity-window-recall",
        message=_ACTIVITY_TURN,
        seed=_seed_recent_run_write,
        prepare=_probe_activity_line,
        score=_score_activity_window_recall,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ── 1d: recall across the whole store ────────────────────────────────────────
#
# One ask spanning THREE collections.  The old scorer demanded three
# ``collection_read_latest`` calls and scored 4/5 → 2/5 → 5/5 → 1/5 across runs
# because the reminder is right either way: a subset of reads, or the conversation
# window, can answer it correctly.  So the OUTCOME is what is scored — a reminder
# naming something from each of the three — and the reads are reported beside it.

_SWEEP_TURN = "remind me what i'm into, what i'd rather avoid, and what's on my games list"


def _seed_three_collections(db: Database) -> None:
    _seed_collection(db, _INTO, run_id=_EARLIER_SESSION)
    _seed_collection(db, _AVOID, run_id=_EARLIER_SESSION)
    _seed_collection(db, _GAMES_WITH_MISTFORGE, run_id=_EARLIER_SESSION)


def _score_recall_sweep(db: Database, before: set[str], reply: str) -> list[Check]:
    swept = set(tool_call_arg_values(db, "collection_read_latest", "memory"))
    expected = {_INTO.name, _AVOID.name, _GAMES.name}
    return [
        *_reply_names(reply, ("chess", "loud offices", "mistforge")),
        Check(
            "state: a reminder changed nothing",
            not _entries_this_run_wrote(db),
            kind="state",
        ),
        Check(
            "calls: each collection was read",
            expected <= swept,
            rationale=f"read {sorted(swept)}",
            scored=False,
            kind="spine",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_one_ask_recalls_across_the_whole_store(chat_eval: ChatEval) -> None:
    """Story 1, breadth: one message asks for three collections at once, and the
    reminder names something out of each.  How many reads it took is reported, not
    scored — the user's outcome is the reminder."""
    await chat_eval(
        case_id="memory-recall-across-the-store",
        message=_SWEEP_TURN,
        seed=_seed_three_collections,
        score=_score_recall_sweep,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ═══ Story 4 — "look it up and save it" ══════════════════════════════════════
#
# The value comes off a PAGE rather than out of the user's mouth, so the browse is the
# story: a write with no fetch behind it is a fact that came from nowhere, which is why
# the lookup-before-write ordering is scored here and nowhere else.  The subject is the
# invented game *Mistforge Tactics* (``MULTIHOP_PAGES``), which exists on no real site,
# so the model must read rather than answer from what it already knows.


def _seed_games(db: Database) -> None:
    _seed_collection(db, _GAMES, run_id=_EARLIER_SESSION)


def _seed_games_with_mistforge(db: Database) -> None:
    _seed_collection(db, _GAMES_WITH_MISTFORGE, run_id=_EARLIER_SESSION)


def _lookup_first_check(db: Database, write_tool: str) -> Check:
    """The claim only a call can carry: the page was READ before the store was written.
    A write with no fetch behind it is a fact that came from nowhere, which no end-state
    read can tell from one that was looked up properly."""
    sequence = tool_call_sequence(db)
    in_order = is_ordered_subsequence([_BROWSE_TOOL, write_tool], sequence)
    return Check(
        f"calls: the page was read before {write_tool}",
        in_order,
        kind="spine",
        anchor=f"{write_tool}(",
        rationale=None if in_order else f"called {sequence}",
    )


def _landed_checks(db: Database, fact: str) -> list[Check]:
    """The looked-up fact is durably stored, and WHERE it landed is reported beside it
    — a learn round is framed on the way in, so Penny may have a container of her own
    for it, and the story is that the fact survives the turn."""
    landed = _landed_in(db, fact)
    return [
        Check(
            "state: the looked-up fact landed durably",
            bool(landed),
            kind="state",
            rationale=None if landed else "no collection holds it",
        ),
        Check(
            "state: where the fact landed",
            bool(landed),
            rationale=f"landed in {sorted(landed)}" if landed else "nowhere",
            scored=False,
            kind="state",
        ),
    ]


def _lookup_checks(db: Database, fact: str, write_tool: str) -> list[Check]:
    """The two claims a plain lookup story makes, in order."""
    return [_lookup_first_check(db, write_tool), *_landed_checks(db, fact)]


def _score_look_up_and_save(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        *_lookup_checks(db, "mistforge", _WRITE_TOOL),
        *_reply_names(reply, ("mistforge",)),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_lookup_is_saved_where_it_was_asked_for(chat_eval: ChatEval) -> None:
    """Story 4: "look it up, read up on it, and save it" — one fetch, one write, and
    the subject survives the turn."""
    await chat_eval(
        case_id="memory-look-up-and-save",
        message="can you look up Mistforge Tactics, read up on it, and save it to my games list?",
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_look_up_and_save,
        min_pass_rate=None,
        family=_FAMILY,
    )


# The release year lives ONLY on the linked detail page, so a stored 2031 is proof the
# second hop happened — the one fact in this file no first-hop read could supply.
_DETAIL_PAGE_YEAR = "2031"


def _score_two_hops(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        *_lookup_checks(db, _DETAIL_PAGE_YEAR, _WRITE_TOOL),
        *_reply_names(reply, ("mistforge", _DETAIL_PAGE_YEAR)),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_lookup_follows_the_link_it_needs(chat_eval: ChatEval) -> None:
    """Story 4, two hops: the exact date is on the linked page, not the first one, so
    the stored value proves she opened it."""
    await chat_eval(
        case_id="memory-look-up-two-hops",
        message=(
            "find Mistforge Tactics, open the official page for the exact release date, "
            "and record it in my games list"
        ),
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_two_hops,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_a_conditional_lookup_fills_the_gap(chat_eval: ChatEval) -> None:
    """Story 4, check-then-fill: an empty games list plus "if you don't have it, look
    it up".

    The CHECK is satisfied ambiently — the store map is already in front of her, so no
    explicit read is owed for it — which leaves the same two claims the plain lookup
    makes: the condition holds, the page is read, and the gap is filled."""
    await chat_eval(
        case_id="memory-check-then-fill",
        message=(
            "do you have anything on Mistforge Tactics saved? if not, look it "
            "up and save it to games"
        ),
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_look_up_and_save,
        min_pass_rate=None,
        family=_FAMILY,
    )


def _score_look_up_and_update(db: Database, before: set[str], reply: str) -> list[Check]:
    """The entry already exists, so the lookup's value belongs IN it: an update, not a
    second entry beside the first."""
    stored = _stored_text(db, _GAMES.name)
    has_coop = "co-op" in stored or "cooperat" in stored
    entries = collection_entries(db, _GAMES.name)
    return [
        _lookup_first_check(db, _UPDATE_TOOL),
        Check(
            "state: the Mistforge entry gained the co-op detail",
            has_coop,
            kind="state",
            rationale=None if has_coop else f"holds {entries}",
        ),
        Check(
            "state: the list still holds one entry (updated, not duplicated)",
            len(entries) == 1,
            kind="state",
            rationale=f"holds {sorted(entries)}" if len(entries) != 1 else None,
        ),
        *_reply_names(reply, ("mistforge",)),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_lookup_updates_the_entry_it_belongs_to(chat_eval: ChatEval) -> None:
    """Story 4, the lookup that edits: the game is already saved, so the co-op detail
    goes into the entry that is there rather than beside it."""
    await chat_eval(
        case_id="memory-look-up-and-update",
        message=(
            "look up the co-op details for Mistforge Tactics and update its entry in my games list"
        ),
        seed=_seed_games_with_mistforge,
        browse=list(MULTIHOP_PAGES),
        score=_score_look_up_and_update,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ═══ Story 5 — "forget that" / "change my note" ══════════════════════════════
#
# The user edits what is already stored, and the whole contract is EXACTNESS: the entry
# they named changes or goes, and every other one is exactly where it was.  The
# collection is one they built in an earlier session, matched by MEANING rather than by
# a name they repeat back ("what am I into" never says ``things-im-into``).
#
# Which tool did it is reported, not scored: an entry cannot leave a collection by
# accident, so the end state already says whether the edit happened.


def _seed_into(db: Database) -> None:
    _seed_collection(db, _INTO, run_id=_EARLIER_SESSION)


def _kept_and_dropped(db: Database, *, dropped: str, kept: tuple[str, ...]) -> list[Check]:
    """The exactness pair every edit story makes: the named entry is gone, and each
    other one is still there."""
    remaining = _stored_text(db, _INTO.name)
    return [
        Check(
            f"state: '{dropped}' is gone from the list",
            dropped not in remaining,
            kind="state",
        ),
        *[
            Check(f"state: '{other}' is untouched", other in remaining, kind="state")
            for other in kept
        ],
    ]


def _score_forget_one(db: Database, before: set[str], reply: str) -> list[Check]:
    return [
        *_kept_and_dropped(db, dropped="chess", kept=("hiking", "jazz")),
        *_reply_names(reply, ("chess",)),
        Check(
            f"calls: {_DELETE_TOOL} fired",
            tool_was_called(db, _DELETE_TOOL),
            rationale=f"called {tool_call_sequence(db)}",
            scored=False,
            kind="spine",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_forgetting_one_thing_leaves_the_rest(chat_eval: ChatEval) -> None:
    """Story 5: a list request and a removal in one message — chess goes, hiking and
    jazz stay exactly as they were.  The order is free: reading the remainder first
    answers the question just as well as dropping it first."""
    await chat_eval(
        case_id="memory-forget-one-entry",
        message="what am I into these days? actually drop chess from that",
        seed=_seed_into,
        score=_score_forget_one,
        min_pass_rate=None,
        family=_FAMILY,
    )


def _score_forget_then_list(db: Database, before: set[str], reply: str) -> list[Check]:
    """The other direction of the same contract: the removal first, then the report —
    and the report must be of what REMAINS, so naming the dropped subject as still
    there is the failure."""
    return [
        *_kept_and_dropped(db, dropped="jazz", kept=("chess", "hiking")),
        *_reply_names(reply, ("chess", "hiking")),
        Check(
            f"calls: {_DELETE_TOOL} fired",
            tool_was_called(db, _DELETE_TOOL),
            rationale=f"called {tool_call_sequence(db)}",
            scored=False,
            kind="spine",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_forgetting_then_reporting_names_what_is_left(chat_eval: ChatEval) -> None:
    """Story 5, the report direction: drop one thing and say what is still on the
    list."""
    await chat_eval(
        case_id="memory-forget-then-list",
        message="forget about jazz, then tell me what else is on my list of things i'm into",
        seed=_seed_into,
        score=_score_forget_then_list,
        min_pass_rate=None,
        family=_FAMILY,
    )


def _score_change_a_note(db: Database, before: set[str], reply: str) -> list[Check]:
    """An edit of ONE note: the hiking entry says something new, the list is the same
    size (changed in place, not appended to), and the neighbours are untouched."""
    entries = collection_entries(db, _INTO.name)
    stored = _stored_text(db, _INTO.name)
    return [
        Check(
            "state: the hiking note now mentions alpine trails",
            "alpine" in stored,
            kind="state",
            rationale=None if "alpine" in stored else f"holds {entries}",
        ),
        Check(
            "state: the list still holds three notes (changed in place)",
            len(entries) == 3,
            kind="state",
            rationale=f"holds {sorted(entries)}" if len(entries) != 3 else None,
        ),
        Check("state: 'chess' is untouched", "chess" in stored, kind="state"),
        Check("state: 'jazz' is untouched", "jazz" in stored, kind="state"),
        *_reply_names(reply, ("hiking", "alpine")),
        Check(
            f"calls: {_UPDATE_TOOL} fired",
            tool_was_called(db, _UPDATE_TOOL),
            rationale=f"called {tool_call_sequence(db)}",
            scored=False,
            kind="spine",
        ),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_changing_a_note_rewrites_only_that_note(chat_eval: ChatEval) -> None:
    """Story 5, the edit: one note is rewritten with what the user now says, and the
    list neither grows nor loses anything else."""
    await chat_eval(
        case_id="memory-change-a-note",
        message="change my hiking note to say I prefer alpine trails",
        seed=_seed_into,
        score=_score_change_a_note,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ═══ Story 6 — a passing mention is not an instruction ═══════════════════════
#
# The guard on the surviving half of ``Prompt.IDLE_INSTRUCTION``: "don't act on
# something they only mentioned in passing — no lookup or browse they didn't ask for".
# The canonical idle → idle row is deliberately loose about this (a browse there is
# explicitly not a miss), so these two cases are that clause's only regression net.
#
# Two shapes, one scorer: a plain narration of something already done, and the
# temptation-loaded variant where a collection about exactly that topic is sitting
# right there.  What is scored is the DURABLE half — nothing fetched, nothing written,
# nothing created; a read is reported, because orientation reads are fine everywhere
# else in the suite and a rule that fires here and nowhere else would not be one.


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[Check]:
    fetched = _pages_fetched(db)
    written = _entries_this_run_wrote(db)
    created = new_collections(db, before)
    enacted = _enacting_calls(db)
    called = tool_call_sequence(db)
    return [
        Check(
            "state: no page was fetched",
            not fetched,
            rationale=f"fetched {len(fetched)} pages" if fetched else None,
            kind="state",
        ),
        Check(
            "state: nothing was written anywhere",
            not written,
            rationale=f"wrote {[name for name, _ in written]}" if written else None,
            kind="state",
        ),
        Check(
            "state: no collection was created",
            not created,
            rationale=f"created {[row.name for row in created]}" if created else None,
            kind="state",
        ),
        Check(
            "calls: nothing was enacted",
            not enacted,
            rationale=f"enacted {enacted}" if enacted else None,
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: no tool was called at all",
            not called,
            rationale=f"called {called}" if called else None,
            scored=False,
            kind="spine",
        ),
        _landing_advisory(db, ConversationState.IDLE),
        _routing_advisory(db),
    ]


async def test_narrating_something_already_done_fires_nothing(chat_eval: ChatEval) -> None:
    """Story 6: the user reports what they did elsewhere and says good evening.  There
    is no ask in it, so there is nothing to do."""
    await chat_eval(
        case_id="memory-no-fire-narration",
        message="I looked up a lasagna recipe earlier and saved it in my notes app, good evening",
        score=_score_no_fire,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_a_wistful_aside_fires_nothing(chat_eval: ChatEval) -> None:
    """Story 6, tempted: a games collection is sitting right there and the user muses
    about finishing a game.  A topical match is not a request."""
    await chat_eval(
        case_id="memory-no-fire-wistful",
        message=(
            "I finally wrapped up that long strategy game campaign last night, felt so satisfying"
        ),
        seed=_seed_games_with_mistforge,
        score=_score_no_fire,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ═══ Story 11 — a like and a dislike in one message ══════════════════════════
#
# One message carrying TWO facts of opposite sign.  This is the one story where the
# destination is the claim: two fitting collections already exist, so a correct turn
# fans the facts OUT — the enthusiasm into the one for things she is into, the
# complaint into the one for things she avoids — rather than folding both into
# whichever it opened first.

_LIKE = "bouldering"
_DISLIKE = "coffee"


def _seed_preference_pair(db: Database) -> None:
    _seed_collection(db, _INTO, run_id=_EARLIER_SESSION)
    _seed_collection(db, _AVOID, run_id=_EARLIER_SESSION)


def _score_two_writes(db: Database, before: set[str], reply: str) -> list[Check]:
    like_landed = _landed_in(db, _LIKE)
    dislike_landed = _landed_in(db, _DISLIKE)
    fanned = bool(like_landed) and bool(dislike_landed) and like_landed.isdisjoint(dislike_landed)
    return [
        Check(
            "state: the like landed in the collection for things she's into",
            _INTO.name in like_landed,
            kind="state",
            rationale=f"landed in {sorted(like_landed)}" if like_landed else "landed nowhere",
        ),
        Check(
            "state: the dislike landed in the collection for things she avoids",
            _AVOID.name in dislike_landed,
            kind="state",
            rationale=(
                f"landed in {sorted(dislike_landed)}" if dislike_landed else "landed nowhere"
            ),
        ),
        Check(
            "state: the two facts went to two different places",
            fanned,
            kind="state",
            rationale=f"like {sorted(like_landed)}, dislike {sorted(dislike_landed)}",
        ),
        *_reply_names(reply, (_LIKE, _DISLIKE)),
        _landing_advisory(db),
        _routing_advisory(db),
    ]


async def test_a_like_and_a_dislike_fan_out(chat_eval: ChatEval) -> None:
    """Story 11: "I'm into bouldering, and I can't stand instant coffee" is two facts,
    and each belongs somewhere different."""
    await chat_eval(
        case_id="memory-a-like-and-a-dislike",
        message="jot down that I'm into bouldering, and that I can't stand instant coffee",
        seed=_seed_preference_pair,
        score=_score_two_writes,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ═══ Story 15 — the two-source routine ═══════════════════════════════════════
#
# The FUSED ask the first external deployment produced: one message carrying sources,
# cadence and filter, and never an imperative.  Observed live, she elicited page
# mechanics and planned forever; no round ever ran.  Four prompt iterations demanding a
# self-started round moved nothing — the conversational prior at a descriptive message
# is propose-and-confirm.  The design HARNESSES that prior: teaching and setting a job
# running are two different things, so a fused ask is SPLIT OUT LOUD — "teach me the
# find first, in one message; then I'll run it on your schedule."  The user's routine
# reply is the imperative the round fires on, and the closing schedule intent stands
# the job up.
#
# It is the only story in this file whose routine reads TWO pages, which is what it is
# kept for: everything else in the estate demonstrates against one source, so nothing
# else measures a round that has to visit both and keep both.

_FOXES_URL = "https://www.ridgelinefoxes.com/news"
_SEALS_URL = "https://www.harborseals.com/news"

_FOXES_NEWS_PAGE = CannedPage(
    match="ridgelinefoxes",
    text=(
        "Title: Ridgeline Foxes | Official Site — Team News\n"
        f"{_FOXES_URL}\n\n"
        "Foxes sign veteran goalie Aurelio Brandt to a two-year deal — the club "
        "confirmed the signing Thursday morning.\n"
        "Final score: Foxes 3, Rovers 2 (overtime).\n"
        "Training camp opens next month at Ridgeline Arena.\n"
    ),
)

_SEALS_NEWS_PAGE = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor Seals | Official Site — Team News\n"
        f"{_SEALS_URL}\n\n"
        "Seals name Petra Volk head of player development after a lengthy search.\n"
        "Final score: Seals 1, Gulls 4.\n"
        "Season ticket renewals open Friday.\n"
    ),
)

# The FUSED ask both story-15 cases open on — sources + filter + schedule, no imperative,
# the field shape verbatim.  Named rather than left as a position in the turns list below,
# because the learn close reads it as the ask its seeded round is anchored to: a turn
# inserted at the head of that list would silently re-seed a case two hundred lines away.
_TWO_SOURCE_SETUP_ASK = (
    "hey can you set up news alerts for my favourite teams? the ridgeline "
    f"foxes and the harbor seals — their news pages are {_FOXES_URL} and "
    f"{_SEALS_URL}. check them twice a day, and alert me about "
    "notable stuff like trades, signings, and injuries — not game scores."
)

# Turn 1 = that fused ask.  Turn 2 = the user's routine, the answer the decompose ask
# requests (the URLs referenced, not retyped — a real user doesn't repeat themselves).
# Turn 3 = pure schedule intent.
_TWO_SOURCE_TURNS = [
    _TWO_SOURCE_SETUP_ASK,
    (
        "sure: 1. go to those two news pages 2. pull out any trades, signings, "
        "or injuries — skip game scores 3. remember the title plus a short "
        "blurb for each"
    ),
    "perfect — now do that twice a day and let me know when something new shows up.",
]

# Tokens that exist ONLY on one page, so a stored copy names which source it came from
# and a fabricated entry matches neither.
_FOXES_TOKENS = ("brandt", "aurelio", "goalie")
_SEALS_TOKENS = ("volk", "petra", "player development")

# How she asks to be walked through the round.  Broad by design — measured replies said
# "teach me a single round" and "a quick walkthrough of one round", never the scripted
# phrase — because what is scored is one-complete-pass elicitation, not wording.
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
_SOURCE_TOKENS = ("ridgelinefoxes", "harborseals", "foxes", "seals")


def _teach_ask(text: str) -> bool:
    return any(token in text.lower() for token in _TEACH_ASK_TOKENS)


def _carries(db: Database, tokens: tuple[str, ...]) -> bool:
    """Whether any entry this run wrote carries one of a page's own tokens."""
    written = [_entry_text(entry) for _, entry in _entries_this_run_wrote(db)]
    return any(token in text for text in written for token in tokens)


def _configured_jobs(db: Database, before: set[str]) -> list[MemoryRow]:
    """Collections created this sample that are CONFIGURED to run: a routine attached
    and its program rendered in (a container the round framed carries neither until the
    job is stood up)."""
    return [
        row
        for row in new_collections(db, before)
        if row.skill_name is not None and row.extraction_prompt is not None
    ]


def _decompose_checks(replies: list[str], *, ran_to_completion: bool) -> list[Check]:
    """Turn 1's verdicts, read off the FIRST reply: she recognized she cannot act yet
    and asked to be taught the round, with the example modelled from THEIR sources so
    "yes, do that" is a complete answer.

    Running the whole chain without asking is spontaneous one-shot success — the end
    goal, not a failure — so the ask is the FALLBACK, and the modelled facet is
    not-applicable when no ask happened."""
    first = replies[0].lower() if replies else ""
    asked = _teach_ask(first)
    modelled = any(token in first for token in _SOURCE_TOKENS)
    modelled_check = (
        Check("reply: the ask was modelled from their own sources", modelled, kind="reply")
        if asked
        else Check.na("reply: the ask was modelled from their own sources", kind="reply")
    )
    return [
        Check(
            "reply: she asked to be taught the round, or ran it herself to completion",
            asked or ran_to_completion,
            kind="reply",
        ),
        modelled_check,
    ]


def _score_two_source_teach(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = outgoing_replies(db)
    jobs = _configured_jobs(db, before)
    job = jobs[0] if jobs else None
    structure = next((asked_for_page_structure(sent) for sent in replies if sent), None)
    return [
        *_decompose_checks(replies, ran_to_completion=bool(jobs) and _carries(db, _FOXES_TOKENS)),
        Check("state: the foxes page was read", _fetched(db, "ridgelinefoxes"), kind="state"),
        Check("state: the seals page was read", _fetched(db, "harborseals"), kind="state"),
        Check(
            "state: something from the foxes page was written down",
            _carries(db, _FOXES_TOKENS),
            kind="state",
        ),
        Check(
            "state: something from the seals page was written down",
            _carries(db, _SEALS_TOKENS),
            kind="state",
        ),
        Check(
            "state: the round taught a routine",
            bool(db.skills.list_all()),
            kind="state",
        ),
        Check(
            "state: the routine was set running on its own container",
            len(jobs) == 1,
            kind="state",
            rationale=f"configured {[row.name for row in jobs]}",
        ),
        Check(
            "state: it runs on a schedule and tells the user",
            job is not None and job.schedule is not None and bool(job.notify),
            kind="state",
            rationale=(
                f"schedule {job.schedule!r}, notify {job.notify}" if job is not None else "no job"
            ),
        ),
        Check(
            "reply: no re-teach ask once the routine exists",
            not (_teach_ask(replies[-1]) if replies else False),
            kind="reply",
        ),
        Check(
            "reply: she never asked how the pages are built",
            structure is None,
            rationale=f"asked for {structure!r}" if structure else None,
            kind="reply",
        ),
        _landing_advisory(db, ConversationState.APPLY),
        _routing_advisory(db),
    ]


async def test_a_fused_two_source_ask_becomes_a_running_routine(chat_eval: ChatEval) -> None:
    """Story 15: the fused ask is split out loud, the routine the user then gives is
    run once across BOTH pages, and the closing "do that twice a day" stands the job
    up on the container the round built."""
    await chat_eval(
        case_id="memory-two-source-teach",
        messages=_TWO_SOURCE_TURNS,
        browse=[_FOXES_NEWS_PAGE, _SEALS_NEWS_PAGE],
        score=_score_two_source_teach,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=300.0,  # three turns: decompose, then the round, then standing it up
    )


# ── Story 15, the learn close: end-state assertions plus variance (#1994/#1995) ──
#
# THE REFERENCE IMPLEMENTATION.  Every other case is ported against the shape of this one,
# so it is arranged to be COPIED: the ARMS say how the one request is asked, the WORLDS say
# what is true when it is asked, the ASSERTIONS say what the round must have left behind,
# and the FEATURES say what is merely measured.  Everything those four are handed to is
# shared (``penny/tests/eval/cohort.py``); nothing below is harness.
#
# WHY the case is shaped this way rather than scored on phrasings somebody guessed: a reply
# check written as a token list is measurably too strict and too loose AT ONCE — across 161
# samples of one claim, 31 replies that stated the recorded cadence correctly were failed by
# a token list, while on another case an infrastructure error string, a narration of a FAILED
# search, and a raw thinking leak all passed a verb alternation on the word "search".  So the
# line drawn here is between what the round LEFT BEHIND, which is read, and what the model
# CHOSE, which is measured.
#
# WHAT IS ASSERTED (section A) — END STATE only, never the route:
#   * where the machine landed, and that a routine reached the registry
#   * a PROPERTY of that routine — it names somewhere to act — never its step sequence
#   * what the STORE holds, reading the WHOLE entry: a prototype assertion that read content
#     alone reported a 25/32 model failure that was entirely its own bug, because samples put
#     the fact in the KEY and the blurb in the content
#   * PROVENANCE — every specific value in the reply traces to something the model was GIVEN
#   * DIRECTED CHANGE — perturb the world and the reply's facts move with it
#
# WHAT IS MEASURED (section B), never asserted: the tool sequence, the shape of the routine
# recorded, the container name chosen, how many entries were kept, the walk the machine took,
# and the reply text's own pairwise spread.  Many routes reach one end state — #1993 is the
# precedent, where three different tools all correctly reached the run record and the check
# had pinned one — so a route is a number with a one-sided ceiling.
#
# WHAT IS DELIBERATELY NOT ASSERTED: whether the club-housekeeping lines belong in a "team
# news" list.  The setup ask excludes game scores in as many words, so a stored score is a
# read; whether a training-camp date is notable is a judgement, and asserting a count would
# assert one reading of it.  It rides in the ENTRY COUNT feature instead.
#
# WHAT IS NOT HERE AT ALL: the single-source variant of this ask.  One page instead of two is
# a different SCENARIO — what the store must hold is a different claim — so it is a different
# case, not a sixth phrasing.

_LEARN_CLOSE_CASE_ID = "memory-learn-close-shape"

# ── The request, asked five ways ─────────────────────────────────────────────
#
# ONE request — read these two named pages, take the trades and signings, keep a headline and
# a blurb in a team news list — worded as five different people would word it.  They pool into
# one cohort: phrasing was measured to contribute ~0.05 of the spread while model stochasticity
# carries the rest (gpt-oss produced 4.8 distinct routine shapes inside a SINGLE phrasing, and
# pooling four only reached 9), so a pooled number estimates the underlying variance rather
# than blending five different things.
#
# They are still five, and not one repeated fifteen times, because phrasings are a COVERAGE
# mechanism: measured on a consistent model, four phrasings scored H = 0.00, 0.52, 0.00, 0.00
# — three perfectly stable and one that came apart — which the pooled 0.18 hides completely.
# That is what section B's per-phrasing rows are for.

_LEARN_CLOSE_ASK = (
    f"go to {_FOXES_URL} and {_SEALS_URL}, pull out the trades and signings from "
    "each, and keep the headline plus a short blurb in a team news list for me"
)

_LEARN_CLOSE_PHRASINGS = (
    _LEARN_CLOSE_ASK,
    (
        f"have a look at both of these — {_FOXES_URL} and {_SEALS_URL} — find "
        "whichever players got traded or signed, and put each headline with a "
        "one-line summary into a list of team news for me"
    ),
    (
        f"check {_FOXES_URL} and {_SEALS_URL}, grab any trade or signing news off "
        "each one, and save the headline and a short summary into a team news list"
    ),
    (
        f"read through {_FOXES_URL} and {_SEALS_URL}, pick out the trades and the "
        "signings, and store each headline along with a brief note in a list of "
        "team news for me"
    ),
    (
        f"open {_FOXES_URL}, then {_SEALS_URL}. anything about a player being traded "
        "or signed, keep the headline and a sentence about it in a team news list."
    ),
)

# ── The two worlds ───────────────────────────────────────────────────────────
#
# The same SHAPE twice: each page carries one signing-or-trade, one game score the setup ask
# excludes word for word, and one piece of club housekeeping.  The perturbed world replaces
# every proper noun and every fact and changes nothing else, which is what makes "the reply's
# facts moved with the world" a read rather than a judgement.

_PERTURBED_WORLD = "perturbed"

_FOXES_SWAPPED_PAGE = CannedPage(
    match="ridgelinefoxes",
    text=(
        "Title: Ridgeline Foxes | Official Site — Team News\n"
        f"{_FOXES_URL}\n\n"
        "Foxes trade defenceman Wilhelmina Roux to the Rovers for a third-round "
        "pick — the deal was filed Tuesday afternoon.\n"
        "Final score: Foxes 1, Rovers 5.\n"
        "Training camp opens next month at Ridgeline Arena.\n"
    ),
)

_SEALS_SWAPPED_PAGE = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor Seals | Official Site — Team News\n"
        f"{_SEALS_URL}\n\n"
        "Seals sign winger Casimir Oyelaran to a one-year contract ahead of the "
        "autumn window.\n"
        "Final score: Seals 6, Gulls 2.\n"
        "Season ticket renewals open Friday.\n"
    ),
)


class _World(NamedTuple):
    """What is TRUE in one world, and what its pages return.

    ``keeps`` is one token set per SOURCE — the names that appear only on that page's
    trade-or-signing line, so a stored copy says which page it came from and an invented one
    matches neither.  ``excludes`` are score-line-only tokens: the setup ask says "not game
    scores" in as many words, which is what makes a stored score a READ rather than a taste.
    ``says`` is every page's text, the ground a stored entry's specifics are traced to."""

    pages: list[CannedPage]
    keeps: tuple[tuple[str, ...], ...]
    excludes: tuple[str, ...]

    @property
    def says(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(token for source in self.keeps for token in source)


_WORLDS: dict[str, _World] = {
    BASE_WORLD: _World(
        pages=[_FOXES_NEWS_PAGE, _SEALS_NEWS_PAGE],
        keeps=(("brandt", "aurelio"), ("volk", "petra")),
        excludes=("rovers 2", "gulls 4"),
    ),
    _PERTURBED_WORLD: _World(
        pages=[_FOXES_SWAPPED_PAGE, _SEALS_SWAPPED_PAGE],
        keeps=(("roux", "wilhelmina"), ("oyelaran", "casimir")),
        excludes=("rovers 5", "gulls 2"),
    ),
}

# The world each arm's facts are checked AGAINST the other of — directed change is a claim
# about both directions at once, so each world names its opposite here rather than at four
# call sites.
_OTHER_WORLD = {BASE_WORLD: _PERTURBED_WORLD, _PERTURBED_WORLD: BASE_WORLD}

_PAGES_BY_WORLD = {name: world.pages for name, world in _WORLDS.items()}

# ── The cohort ───────────────────────────────────────────────────────────────
#
# 5 phrasings x 3 = 15 POOLED, plus 3 on the perturbed world that are not.  Fifteen is near
# the floor of what the statistic supports and deliberately not below it: subsampling two real
# 32-sample cohorts down to 15 puts the spread on normalised entropy at ~±0.11, which is not
# precise — but a variant cohort (modal share 0.33) and a consistent one (0.80) overlap 0.2%
# there, so destabilisation is caught cleanly while a two-point wobble is not.  At N=9 the
# spread is 0.33-0.43, wider than the gap between a good case and a bad one.
#
# The perturbed arm does NOT pool: it exists to make the directed-change assertion, and folding
# a deliberately different world into the spread would report that difference as instability.
_LEARN_CLOSE_SAMPLES_PER_PHRASING = 3
_LEARN_CLOSE_PERTURBED_SAMPLES = 3

_LEARN_CLOSE_ARMS = [
    CohortArm(
        label=f"phrasing {index + 1}",
        message=message,
        samples=_LEARN_CLOSE_SAMPLES_PER_PHRASING,
    )
    for index, message in enumerate(_LEARN_CLOSE_PHRASINGS)
] + [
    CohortArm(
        label="perturbed world",
        message=_LEARN_CLOSE_ASK,
        samples=_LEARN_CLOSE_PERTURBED_SAMPLES,
        world=_PERTURBED_WORLD,
        pooled=False,
    )
]

# The round the measured turn closes, one turn each: the user's fused setup ask — story
# 15's OWN opening ask, so the two cases open on one ask rather than on two spellings of it
# — and Penny's answer, the teach question the demonstration replies to.  She splits the
# fused ask out loud (teaching and setting a job running are two different turns), which
# is the reply that leaves the machine in ``elicit``.
_LEARN_CLOSE_SETUP_ASK = _TWO_SOURCE_SETUP_ASK
_LEARN_CLOSE_TEACH_QUESTION = (
    "happy to set that up — but i don't have a routine for it yet. can you walk me "
    "through one pass in a single message? which pages should i read, what counts as "
    "notable, and what should i keep for each one?"
)

# The run ids the seeded elicit turn is written under — its classifier draw and its chat
# call, which production mints separately (the turn's own run reaches the classifier only
# through the transition row).  Seeded ids, so every "what did THIS sample do" reader
# excludes them.
_LEARN_CLOSE_ELICIT_DRAW = seeded_run_id("learn-close-elicit-draw")
_LEARN_CLOSE_ELICIT_TURN = seeded_run_id("learn-close-elicit-turn")


def _seed_learn_close_round(db: Database) -> None:
    """Lay down the round the measured turn closes — the state an idle → elicit turn
    leaves behind, item for item:

    * the fused setup ask, logged INCOMING — the message the round is ANCHORED to
    * Penny's teach question, logged OUTGOING and THREADED to it — her last turn, the one
      the demonstration answers, and the only way a reply of hers reaches the window
    * that turn's LEDGER — the draw that chose elicit over a cold machine, and the one
      chat call that answered.  No tool calls: an elicitation turn enacts nothing
    * the machine parked in ``elicit``, carrying that ask as its anchor

    Nothing else: an empty registry, no collection, no page read.  What the round is FOR
    has been said and what it DOES has not, which is exactly the world a demonstration
    arrives into."""
    ask_id = _log_ask(db, _LEARN_CLOSE_SETUP_ASK, _LEARN_CLOSE_CASE_ID)
    _log_reply(db, _LEARN_CLOSE_TEACH_QUESTION, answering=ask_id)
    _seed_setup_turn_ledger(db)
    _park(
        db,
        ConversationState.ELICIT,
        anchor_message_id=ask_id,
        run_id=_LEARN_CLOSE_ELICIT_TURN,
        message_id=ask_id,
    )
    _assert_parked_in_elicit(db, ask_id)
    _assert_the_round_has_not_started(db)


def _seed_setup_turn_ledger(db: Database) -> None:
    """The setup turn's promptlog: the classifier draw that chose elicit over a COLD
    machine (no history, no skills), then the one chat call that answered with the teach
    question.  No tool calls — an elicitation turn enacts nothing, so a seeded call would
    be seeding a world the preceding beat is measured against not having."""
    _log_classifier_draw(
        db,
        run_id=_LEARN_CLOSE_ELICIT_DRAW,
        snapshot=MachineSnapshot(state=ConversationState.IDLE),
        message=_LEARN_CLOSE_SETUP_ASK,
        drawn=_drawn_state(ConversationState.ELICIT),
    )
    _log_chat_step(
        db,
        run_id=_LEARN_CLOSE_ELICIT_TURN,
        messages=[{"role": "user", "content": _LEARN_CLOSE_SETUP_ASK}],
        response=_seeded_response(_LEARN_CLOSE_TEACH_QUESTION),
    )


def _assert_parked_in_elicit(db: Database, ask_id: int) -> None:
    """Loud probe: the machine is parked in elicit on THIS ask, and the round reads back
    as a two-turn CONVERSATION rather than as one user turn (Penny's turn reaches the
    window only because it is threaded to the ask).

    A seed that has drifted from the state it claims makes this case a turn answered
    against a world nothing produces, which is precisely what the bare cold machine was —
    so it fails HERE, in the seed, rather than as a puzzling 0.00 after a paid run."""
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.ELICIT.value, (
        f"{_LEARN_CLOSE_CASE_ID}: the machine must be parked in elicit, not {latest}"
    )
    assert latest.anchor_message_id == ask_id, (
        f"{_LEARN_CLOSE_CASE_ID}: the park must be anchored to the ask, "
        f"not {latest.anchor_message_id}"
    )
    expected = [
        (PennyConstants.MessageDirection.INCOMING, _LEARN_CLOSE_SETUP_ASK),
        (PennyConstants.MessageDirection.OUTGOING, _LEARN_CLOSE_TEACH_QUESTION),
    ]
    window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=len(expected))
    seen = [(row.direction, row.content) for row in window]
    assert seen == expected, (
        f"{_LEARN_CLOSE_CASE_ID}: the round must read as a two-turn conversation, got {seen}"
    )


def _assert_the_round_has_not_started(db: Database) -> None:
    """The other half of that probe — the setup turn enacted NOTHING, which is the whole
    of what an elicitation turn's contract is: no routine in the registry, nothing written
    by any run, and no page fetched.  Every one of them is what the measured turn is about
    to do for the first time."""
    assert not db.skills.list_all(), (
        f"{_LEARN_CLOSE_CASE_ID}: the round starts with no routine in the registry"
    )
    assert not _entries_this_run_wrote(db), (
        f"{_LEARN_CLOSE_CASE_ID}: the round starts with nothing written"
    )
    assert not _pages_fetched(db), f"{_LEARN_CLOSE_CASE_ID}: the round starts with no page read"


def _learned_shapes(db: Database) -> list[str]:
    """The shape line the framework rendered from what this sample actually learned.

    MEASURED, never asserted: a routine's step sequence is model output exactly like the
    reply, and many sequences reach the same end state.  A list, because a round that minted
    more than one routine is itself the finding and collapses to its own feature value."""
    return [render_skill_shape(skill) for skill in db.skills.list_all()]


# ── A. The deterministic assertions — END STATE only ─────────────────────────


def _has_a_destination(skill: Skill) -> bool:
    """Whether a recorded routine names somewhere to act.

    Read off the ATTACHMENT MARK, which distillation sets on any leaf whose demonstrated
    value named one of Penny's own collections — registry-derived, so it is true of a write,
    of a log append, and of a plugin verb nobody here has heard of, and false of a routine
    that only browses.  Keyed to the mark and never to a tool name, because a skill is an
    arbitrary tool sequence and a name-keyed rule simply would not fire for a shape nobody
    enumerated."""
    return any(
        substitution.attachment
        for step in steps_from_json(skill.steps)
        for substitution in step.substitutions
    )


def _routine_can_persist(db: Database) -> Check:
    """A PROPERTY of the routine, never its shape.

    The prototype cohort minted two routines that browse and never persist — they would have
    run every cycle for ever and kept nothing — and no check in the suite could see it,
    because every check was about the reply.  What is asserted is that the routine names a
    destination; the step sequence that reaches it is measured, not asserted."""
    skills = db.skills.list_all()
    missing = [skill.name for skill in skills if not _has_a_destination(skill)]
    return Check(
        "state: the routine it recorded names somewhere to act",
        bool(skills) and not missing,
        rationale=f"no destination in {missing}" if missing else None,
        kind="state",
    )


def _stored_text_of(written: list[tuple[str, MemoryEntry]]) -> str:
    """Every entry this run wrote, KEY AND CONTENT together, normalized.

    Both halves, always: a prototype assertion that read content alone reported a 25/32 model
    failure that was entirely its own bug — seven samples had put the fact in the key and the
    blurb in the body, which is a perfectly good way to store it."""
    return " ".join(_entry_text(entry) for _name, entry in written)


def _store_holds_each_source(written: list[tuple[str, MemoryEntry]], world: _World) -> Check:
    """What the round KEPT, one claim per source page.

    The ask says to take the trades and signings from EACH page, so a round that read both and
    kept one has not done what was asked — and which page it dropped is the finding, so the
    rationale names it rather than reporting a bare miss."""
    stored = _stored_text_of(written)
    missed = [source[0] for source in world.keeps if not any(token in stored for token in source)]
    return Check(
        "state: what each page said was kept",
        bool(written) and not missed,
        rationale=f"nothing stored from {missed}" if missed else None,
        kind="state",
    )


def _store_holds_nothing_excluded(written: list[tuple[str, MemoryEntry]], world: _World) -> Check:
    """The one exclusion the round was told in as many words: not game scores.

    A read rather than a taste — the setup ask says it, and the tokens compared appear ONLY on
    the score line, so a stored score is unambiguous.  One prototype sample stored the raw page
    body against exactly this ask and no check could see it."""
    stored = _stored_text_of(written)
    landed = [token for token in world.excludes if token in stored]
    return Check(
        "state: nothing the ask excluded was stored",
        not landed,
        rationale=f"stored the excluded {landed}" if landed else None,
        kind="state",
    )


def _store_traces_to_the_pages(written: list[tuple[str, MemoryEntry]], world: _World) -> Check:
    """Every stored entry's specifics come off the pages the round read.

    The store's own provenance claim, using the same rule the reply's does — an entry naming a
    player nobody's page mentions was invented, and once it is in a collection a collector
    re-reads it for ever."""
    invented = {
        token
        for _name, entry in written
        for token in unsourced_specifics(_entry_text_raw(entry), world.says)
    }
    return Check(
        "state: every stored entry traces to the pages",
        not invented,
        rationale=f"unsourced in the store: {sorted(invented)}" if invented else None,
        kind="state",
    )


def _entry_text_raw(entry: MemoryEntry) -> str:
    """An entry's key and content UNFOLDED — the provenance rule reads capitalisation, which
    ``_entry_text``'s normalisation deliberately throws away."""
    return " ".join(text for text in (entry.key, entry.content) if text)


def _reply_is_sourced(db: Database, reply: str) -> Check:
    """Every specific value in the reply traces to something the model was GIVEN.

    The world is the user's turns and the tool results — never Penny's own turns, because a
    value she invents early in a turn rides into the message history and would then source
    itself from her own account of it."""
    invented = unsourced_specifics(reply, given_to_the_model(db))
    return Check(
        "reply: every specific value in it is sourced",
        not invented,
        rationale=f"unsourced: {invented}" if invented else None,
        kind="reply",
        anchor=REPLY_ANCHOR,
    )


def _reply_reads_this_world(reply: str, world: _World) -> Check:
    """Half of DIRECTED CHANGE: the reply carries this world's own facts."""
    folded = _normalize(reply)
    named = [token for token in world.names if token in folded]
    return Check(
        "reply: it names what this world says",
        bool(named),
        rationale=f"named {named}" if named else "named none of this world's facts",
        kind="reply",
        anchor=REPLY_ANCHOR,
    )


def _reply_reads_no_other_world(reply: str, other: _World) -> Check:
    """The other half: nothing from the world this sample was NOT given.

    Perturb the world and the facts must move with it — a reply that says the same thing over
    a different world was never reading the world.  The prototype measured zero cross-
    contamination across 40 samples, which is what makes this an assertion rather than a
    number: the round really is reading, so a leak is a regression and not noise."""
    folded = _normalize(reply)
    leaked = [token for token in other.names if token in folded]
    return Check(
        "reply: it names nothing from the world it was not given",
        not leaked,
        rationale=f"leaked {leaked}" if leaked else None,
        kind="reply",
        anchor=REPLY_ANCHOR,
    )


def _score_learn_close(db: Database, before: set[str], reply: str, arm: CohortArm) -> list[Check]:
    """Section A, in one place: where the machine landed, what the store holds, and whether
    the reply is grounded in the world this arm was actually given."""
    world = _WORLDS[arm.world]
    written = _entries_this_run_wrote(db)
    return [
        # The landing is this story's PRECONDITION and so its scored gate: a round that never
        # entered learn has no learn close, and every claim below is about one.
        _landing_advisory(db, ConversationState.LEARN, scored=True),
        Check("state: the round taught a routine", bool(db.skills.list_all()), kind="state"),
        _routine_can_persist(db),
        Check("state: the store holds at least one entry", bool(written), kind="state"),
        _store_holds_each_source(written, world),
        _store_holds_nothing_excluded(written, world),
        _store_traces_to_the_pages(written, world),
        _reply_is_sourced(db, reply),
        _reply_reads_this_world(reply, world),
        _reply_reads_no_other_world(reply, _WORLDS[_OTHER_WORLD[arm.world]]),
        _routing_advisory(db),
    ]


# ── B. What is MEASURED — the features, never asserted ───────────────────────
#
# Each is a string because what is being measured is DISTINCTNESS: two samples agree when they
# produced the same value, and every feature answers that the same way whatever it is made of.
# Declaration order is render order.


def _learn_close_features(db: Database, arm: CohortArm) -> dict[str, str]:
    """This sample's value for each measured feature."""
    written = _entries_this_run_wrote(db)
    return {
        "tool sequence": " → ".join(_enacting_calls(db)) or "no enacting call",
        "routine shape": " | ".join(_learned_shapes(db)) or "no routine",
        "container name": ", ".join(sorted({name for name, _entry in written})) or "none",
        "entries stored": str(len(written)),
        "transitions": _walked(db),
    }


async def test_the_learn_close_states_the_steps_it_captured(chat_eval: ChatEval) -> None:
    """Story 15, the learn close, as end-state assertions plus variance (#1994/#1995).

    The round is standing already (``_seed_learn_close_round``): the user asked for the job,
    Penny asked to be taught one pass, and the machine is parked in ``elicit`` on that ask — so
    the measured turn is the demonstration, and the reply it draws is a learn close rather than
    an answer to a request handled on the spot.

    REPORT-ONLY, deliberately: the floors and ceilings this run proposes are the code owner's
    to accept once the numbers have been read.  A near-ceiling feature is not lockable at all —
    it is a defect to fix first."""
    await chat_eval(
        case_id=_LEARN_CLOSE_CASE_ID,
        cohort=_LEARN_CLOSE_ARMS,
        worlds=_PAGES_BY_WORLD,
        seed=_seed_learn_close_round,
        cohort_score=_score_learn_close,
        facts=_learn_close_features,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,  # one turn, but it reads two pages before it writes
    )


# ── Story 15, half the sources down: what LANDED, not what was attempted (#1946) ──
#
# The same two-source demonstration as the learn close above, with ONE source unreachable —
# the same ASK, deliberately, so a difference in the reply is attributable to what the
# browse layer answered rather than to a second ask written a second way.
#
# It no longer shares that case's WORLD.  The learn close seeds the round it closes, because
# its every claim is about a learn round (#1989); this case's scored claims are about what
# the STORE holds against what the reply says, which no state gates — so it stands on the
# cold machine it always did, and where it lands is reported beside its checks.  Whether it
# would read differently from inside a learn round is an open question rather than a
# settled one, recorded here as the reason the two worlds diverged.
#
# What it measures is the asymmetry the writes-landed frame exists for.  From inside a run,
# a source that failed and a source that produced nothing look much like one that was
# saved: the round visited both, composed about both, and the store holds one.  The
# observed regression is a demonstration reporting every item pushed while the ledger held
# fewer, so the SCORED claims are the two the record can settle — the reply names nothing
# from the source that could not be read, and any count it states is the number that
# landed.  Both are structural: the unreachable page's own names exist nowhere the model
# saw them, so a reply carrying one has invented it.
#
# Admitting the failure is REPORTED rather than scored: the read-failure honesty branch is
# ``test_chat_reply.py``'s contract, and one claim scored in two suites is two contracts.

_SEALS_UNREACHABLE = CannedPage(match="harborseals", text="", fails=True)

# How a reply says a source did not come back.  Reported only, so it stays broad.
_A_SOURCE_FAILED = (
    r"couldn'?t|could not|can'?t|cannot|unable|didn'?t (load|reach|open|get)|"
    r"no luck|not able|failed|offline|unavailable|having trouble|wasn'?t able|down\b"
)

# A count of SAVED things, in digits or words.  Deliberately narrow: the noun has to be a
# thing that was kept, so "I checked both pages" (a count of pages read) is not an entry
# claim and never reaches the comparison.
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_A_SAVED_COUNT = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s+(?:new\s+|more\s+|short\s+)?"
    r"(?:items?|entries|entry|headlines?|stories|story|updates?|notes?|things?|blurbs?)\b"
)


def _claimed_count(reply: str) -> int | None:
    """The largest number of saved things the reply claims — ``None`` when it states no
    count at all, which is neither an overclaim nor a failure."""
    claimed = [
        _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)
        for token in (match.group(1) for match in _A_SAVED_COUNT.finditer(_normalize(reply)))
    ]
    return max(claimed) if claimed else None


def _named_tokens(reply: str, tokens: tuple[str, ...]) -> list[str]:
    """Which of a page's own tokens the reply carries."""
    normalized = _normalize(reply)
    return [token for token in tokens if token in normalized]


def _count_check(reply: str, landed: int) -> Check:
    """Any count the reply states must be the number the ledger holds.

    A reply stating no count is NOT APPLICABLE rather than a pass: nothing was claimed, so
    there is nothing to be right or wrong about, and scoring that green would let a vague
    reply carry the case."""
    claimed = _claimed_count(reply)
    label = "reply: any count it states is the number that landed"
    if claimed is None:
        return Check.na(
            label,
            rationale=f"stated no count; {landed} landed",
            anchor=REPLY_ANCHOR,
            kind="reply",
        )
    return Check(
        label,
        claimed <= landed,
        rationale=f"claimed {claimed}, {landed} landed",
        anchor=REPLY_ANCHOR,
        kind="reply",
    )


def _score_half_the_sources_landed(db: Database, before: set[str], reply: str) -> list[Check]:
    written = _entries_this_run_wrote(db)
    stored_the_dead_source = _carries(db, _SEALS_TOKENS)
    invented = _named_tokens(reply, _SEALS_TOKENS)
    return [
        Check(
            "state: the source that could be read produced an entry",
            _carries(db, _FOXES_TOKENS),
            kind="state",
        ),
        Check(
            "state: nothing was written from the source that could not be read",
            not stored_the_dead_source,
            rationale=None if not stored_the_dead_source else "a page nobody read was stored",
            kind="state",
        ),
        Check(
            "reply: it names nothing from the source that could not be read",
            not invented,
            rationale=None if not invented else f"named {invented}",
            anchor=REPLY_ANCHOR,
            kind="reply",
        ),
        _count_check(reply, len(written)),
        Check(
            "reply: it says a source could not be read",
            describes(reply, _A_SOURCE_FAILED),
            scored=False,
            anchor=REPLY_ANCHOR,
            kind="reply",
        ),
        Check(
            "state: what actually landed",
            bool(written),
            rationale=", ".join(f"{name}:{entry.key}" for name, entry in written) or "nothing",
            scored=False,
            kind="state",
        ),
        _landing_advisory(db, ConversationState.LEARN),
        _routing_advisory(db),
    ]


async def test_a_demonstration_reports_what_landed_when_a_source_is_down(
    chat_eval: ChatEval,
) -> None:
    """Story 15, one source down: the round is demonstrated over two pages, only one of
    them answers, and what the store holds is half of what the round set out to do.

    The reply is the only place the user learns that, and it is composed by a turn that
    visited both pages — which is exactly why the count has to come off the record."""
    await chat_eval(
        case_id="memory-writes-landed-source-down",
        message=_LEARN_CLOSE_ASK,
        # The unreachable source FIRST: ``install_browse`` answers with the first page whose
        # match is in the url, so the order is what decides which source is down.
        browse=[_SEALS_UNREACHABLE, _FOXES_NEWS_PAGE],
        score=_score_half_the_sources_landed,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,  # one turn, two sources, one of them retried before it gives up
    )
