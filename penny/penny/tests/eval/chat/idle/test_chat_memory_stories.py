"""The chat memory stories — what a user asks Penny's memory to do, end to end.

Each case drives the REAL chat path against the live model with the user's own words
and scores PERSISTED state: the collections, the entries and their run stamps, the
pages the browse log holds, the machine's own move, and the replies Penny sent.  Five
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

Every story here is a turn that begins and ends in ``idle`` — the whole of it happens
inside the one turn, and the ``state_transition`` rows read ``idle → idle`` for every
case in the file.  That includes the ones whose message carries an instruction
("remember it", "save it to my games list"): a message with no standing or scheduling
component is handled in the conversation, so it never leaves idle.  Story 15's two
teach cases DID leave it — they walk idle → elicit → learn — so they are the learn
microcontext's and live in ``chat/learn/test_two_source_teach.py``.  What stayed from
that world is its third case, the demonstration whose source is down, which is an
ordinary idle turn like the rest.

The worlds are the CURRENT runtime's.  Nothing is pre-seeded any more (migration
0108), so every collection a case starts with is one this file creates the way the
user's own earlier turn would have — through ``create_collection`` and the
collection's own ``write``, stamped with a seeded run id so "what did THIS sample
write" stays answerable (``is_seeded_run``).  The old ``likes`` / ``dislikes``
catch-alls are gone and no case leans on them.

WHERE a write lands is deliberately open where the story does not fix it: the durable
checks read what THIS RUN wrote wherever it landed, and the destination is reported
beside them, so a story that never named a destination is not scored on one.  Story 11
is the exception, because there the destination IS the story: two facts, two fitting
places.

Check labels carry one of three prefixes — ``state:`` (end DB facts), ``reply:``
(what Penny said, against what she did), ``calls:`` (call provenance).  State and
reply checks are SCORED; the call spine, the state the machine landed in, and the
loop-health verdict are ADVISORY (``scored=False``) unless that row IS the story's own
claim — a lookup before a write is one, since a write with no browse is a fact that
came from nowhere.  Where a scored claim only exists in a state a sample never
reached, it reads ``Check.na(...)`` rather than ❌: a precondition nobody met is not a
contract anybody failed.

Every case is REPORT-ONLY (``min_pass_rate=None``): the thresholds are the code
owner's to set once the numbers are read.  All content is synthetic — invented games,
invented teams, invented markets — because the repo is public.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session

from penny.agents.self_state import SelfStateHeader
from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, PromptLog
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    chat_run_tool_sequences,
    collection_entries,
    describes,
    is_ordered_subsequence,
    new_collections,
    outgoing_replies,
    seeded_run_id,
    tool_call_arg_values,
    tool_call_sequence,
    tool_was_called,
)

# The enacting-tool set is read from the suite's shared fixtures, not restated here: the
# state machine's elicitation edge asks the same question of a turn (nothing acted on
# before it was taught), and one policy in two copies is two contracts.
from penny.tests.eval.utils.fixtures import (
    ENACTING_TOOLS,
    MULTIHOP_PAGES,
    CannedPage,
    SynthCollection,
)
from penny.tests.eval.utils.memory_world import (
    _FAMILY,
    _FOXES_TOKENS,
    _SEALS_TOKENS,
    LEARN_CLOSE_ASK,
    _carries,
    _entries_this_run_wrote,
    _entry_text,
    _landing_advisory,
    _normalize,
    _pages_fetched,
    _routing_advisory,
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
from penny.tests.eval.utils.worlds import (
    FOXES_NEWS,
)

pytestmark = pytest.mark.eval

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


def _stored_text(db: Database, name: str) -> str:
    """Every key and content a collection currently holds, normalized and joined."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _landed_in(db: Database, fact: str) -> set[str]:
    """The collections where THIS run's own writes carry ``fact``."""
    needle = _normalize(fact)
    return {name for name, entry in _entries_this_run_wrote(db) if needle in _entry_text(entry)}


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
        message=LEARN_CLOSE_ASK,
        # The unreachable source FIRST: ``install_browse`` answers with the first page whose
        # match is in the url, so the order is what decides which source is down.
        browse=[_SEALS_UNREACHABLE, FOXES_NEWS],
        score=_score_half_the_sources_landed,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,  # one turn, two sources, one of them retried before it gives up
    )
