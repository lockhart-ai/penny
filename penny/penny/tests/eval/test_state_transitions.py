"""Per-transition ENACTMENT contracts (#1706) — one edge of the conversation
state machine per case, on the auction fixture and its register.

``test_state_classifier.py`` proves each edge is CHOSEN correctly from a scoped
micro-context.  This proves the other half: handed the state that edge lands in,
chat does that state's job and nothing else.  Both halves run for real here —
the machine is wired into the turn, so a case seeds the state the edge starts
from, sends one message, and the production path classifies it and swaps the
instruction before the turn runs.

The auction script, one turn per case (the simplest complete journey; richer
shapes are the later beats' business):

    idle → elicit   "watch this auction for me"                     → asks to be taught
    elicit → learn  "go to the site, find the price, remember it"   → runs it once, remembers
    learn → apply   "now do that hourly until 10pm and tell me"     → a live watch

Each case's seeded state is the PRECEDING beat's terminal state and nothing
more — one edge is one message answered against where the last edge stopped.
Replaying earlier turns is not neutral: the apply case seeded the instigating
ask as well, and the classifier duly read "the task being worked on" as a setup
still being specified, which it no longer was.  The opening edge is the one whose
preceding state is NOTHING, so it seeds nothing at all: no transition rows (idle
is the absence of history) and an empty registry.  It carries FIVE asks rather
than one — the script's own turn plus four subjects borrowed from the
classifier's fire pool at a richer register — because a turn that must not act
is only proven by asks that make acting tempting in different ways.  The next
edge continues each of them: ``elicit → learn`` is five demonstrations, one per
scenario, each answered against the world its own ask left behind — so the two
edges chain subject for subject rather than meeting only on the auction script.

**Learning attaches nothing** (#1706, replacing #1687's run-end auto-attach): the
machine makes teaching and instantiating two clear turns, so the demonstrated
round leaves a naive collection_write behind — a collection with a value in it
and no skill, no rendered program, nothing scheduled — and a LATER turn applies
the skill.  Scoring that separation is most of the point of these cases.

WHICH collection a job ends up on is deliberately out of scope (code owner): she
has spread work across several collections where one was meant since long before
this machine existed, so that is a collection-management question of its own and
grading it per-transition would report a standing problem as an edge failure.
The apply case scores that the skill is APPLIED correctly — bound, rendered,
scheduled on the terms given — and carries the reuse question as an advisory.
"""

from __future__ import annotations

import json
from functools import partial
from typing import NamedTuple

import pytest

from penny.constants import PennyConstants, TransitionCause
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, MemoryRow
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import (
    WRITE_TARGET_DESCRIPTION,
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    slug_skill_name,
)

# The production draw-application, used as itself: this case's fixture skill has to
# be the SHAPE run-end extraction really produces, and re-implementing that mapping
# here would be a fixture that drifts from the pipeline it stands in for.  Both
# halves of the #1824 split are applied by their own production function —
# ``_apply_leaf_labels`` for the labeller's spots, ``_naming`` +
# ``_interface_parameters`` for the framer's signature.  ``attachment_names`` is the
# registry policy for what a routine can be attached to, read for the same reason: the
# scorer asks whether a learned routine HAS a destination, and that is the question
# extraction already answers when it decides which leaves to mark.
from penny.skill_extraction import (
    _apply_leaf_labels,
    _interface_parameters,
    _naming,
    attachment_names,
)
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    ParameterFamily,
    Seeder,
    asked_for_page_structure,
    chat_run_tool_sequences,
    classify_by_family,
    collection_entries,
    count_tool_calls,
    new_collections,
    outgoing_replies,
    routing_clean,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.fixtures import CannedPage

# The agreed breadth for "the page the routine is pointed at", READ from where the framer
# suite declares it rather than restated here: what a page parameter may reasonably be
# called is one code-owner-agreed vocabulary, and two copies would drift into two
# contracts (the same rule ``_ENACTING_TOOLS`` is imported under).
from penny.tests.eval.test_skill_framing import _PLACE_TOKENS

# The enacting-tool set is the elicitation contract itself — the calls that would mean
# she acted before being taught — so it is READ from where beat 1a already declares it
# rather than restated here: one policy, one definition.
from penny.tests.eval.test_watch_journey import (
    _ENACTING_TOOLS,
    AURORA_LISTING_499,
    LISTING_URL,
)
from penny.text_validity import is_blank
from penny.tools.micro_context import FramedParameter, LeafLabel, SkillLabels, SkillSignature

pytestmark = pytest.mark.eval

_FAMILY = "state-transitions"


# ── Shared across the edges ───────────────────────────────────────────────────


def _park(db: Database, state: ConversationState, *, anchor_message_id: int | None = None) -> None:
    """Leave the machine where the edge under test starts from, through the real
    store — a seeded transition row IS the machine's state (#1706), so nothing
    here fakes a state the production path couldn't be in.  The incoming message
    is still classified against it, so a case exercises the edge end to end.

    ``anchor_message_id`` is the instigating ask the parked round is anchored to
    — what the production anchor lifecycle stamps on the way in, and what the
    classifier renders as the task being worked on."""
    db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=state.value,
        cause=TransitionCause.CLASSIFIER,
        anchor_message_id=anchor_message_id,
    )


def _landed_state(db: Database) -> str | None:
    latest = db.machine.latest_transition()
    return latest.to_state if latest else None


def _entries_written_by_this_run(db: Database) -> list[MemoryEntry]:
    """Every ENTRY this run wrote, wherever it landed.

    Scoring only collections the run CREATED assumed she always makes one — but
    "remember it" may reuse a name that already exists, and then the run's real
    writes are invisible while the reused collection's own seeded prompt and
    trigger read as things she did.  The run-id stamp says exactly what this run
    wrote (#1560), so ask that instead of inferring from newness.

    The whole entry, not its content alone: where in the entry a fact landed is a
    question about key/value semantics that is deliberately open (#1854), so the
    callers read both halves through ``_written_texts``."""
    written: list[MemoryEntry] = []
    for row in db.memories.list_all():
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [e for e in entries if e.created_by_run_id]
    return written


def _written_texts(entries: list[MemoryEntry]) -> list[str]:
    """Both halves of every written entry — its KEY and its CONTENT.

    One shape, two customers: what the durable-write check matches the case's fact
    against, and what a rationale names when it missed.  A log entry has no key, so
    what it contributes is its content alone."""
    return [text for entry in entries for text in (entry.key, entry.content) if text]


def _pages_fetched(db: Database) -> list[MemoryEntry]:
    """Every page this run read — the browse-results log's recent window.

    One definition, two customers: the edge that must prove NOTHING was fetched
    (idle → elicit) and the elicit → learn seed's probe, which asserts its world
    STARTS from that same emptiness."""
    return require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).read_recent(
        window_seconds=3600, cap=None
    )


def _leaf_at(arguments: dict, path: list):
    """The argument leaf a substitution's JSON path addresses — the step carries the
    call's verbatim arguments, so the DEMONSTRATED value is still in place."""
    node = arguments
    for part in path:
        node = node[part]
    return node


# ── idle → elicit: the ask lands cold, and nothing is enacted ─────────────────
#
# The edge that opens every journey, and the only one whose starting world is
# NOTHING: a COLD machine (no transition rows at all — idle is the ABSENCE of
# history, so there is nothing to park) and an empty skill registry, which is
# this suite's default world.  Turn 1 has no routine to run, so its whole job is
# to ask for one.
#
# Each ask is the full watch register — an external source, something stored or
# compared or reasoned over, a cadence word, and a notify clause.  Four of the
# five are the ENRICHED derivatives of a named ``test_state_classifier.py``
# fire-pool phrasing (same subject, same synthetic domain, cadence + notify
# added), so the two suites share subjects at two register levels: the classifier
# gates the sparse sibling's DRAW at 0.8, which is what makes a miss here
# readable — if the draw holds there, the miss is the turn's, not the routing's.
#
# Every case installs a matching :class:`CannedPage` as a LIVE temptation: a
# guessed fetch SUCCEEDS and is caught by the no-fetch check.  A temptation that
# would fail invisibly proves nothing about restraint.
#
# The reference replies below are review targets under the semantic-breadth rule
# — read at joint review, never matched by a scorer.


_FERRY_TIMETABLE_URL = "https://harborferries.example/timetable"
# Matched on "timetable", the one token the ask and the address SHARE: the ask says
# "ferry" and the host says "harborferries", so a page matched on "ferry" would answer
# a search and then miss a direct read of its own url — a temptation that only half
# exists.
_FERRY_TIMETABLE = CannedPage(
    match="timetable",
    text=(
        "Title: Harbor ferry timetable — daily sailings | harborferries\n"
        f"{_FERRY_TIMETABLE_URL}\n"
        "\n"
        "Sailings published each morning by a fictional harbour authority.\n"
        "Departures: 6:40am, 9:15am, 12:30pm, 4:05pm, 7:20pm\n"
        f"[Harbor ferry timetable]({_FERRY_TIMETABLE_URL})\n"
        "Late sailing: not scheduled this season.\n"
    ),
)

_BAKERY_SPECIALS_URL = "https://cornerbakery.example/specials"
_BAKERY_SPECIALS = CannedPage(
    match="bakery",
    text=(
        "Title: Corner Bakery — today's specials | cornerbakery\n"
        f"{_BAKERY_SPECIALS_URL}\n"
        "\n"
        "Baked each morning in a fictional kitchen, posted before 7am.\n"
        "Today's special: rye and caraway loaf\n"
        f"[Corner Bakery specials]({_BAKERY_SPECIALS_URL})\n"
        "Specials change daily and yesterday's come down overnight.\n"
    ),
)

_COLONY_COUNT_URL = "https://harborseals.example/colony-count"
_COLONY_COUNT = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor seal colony count — weekly survey | harborseals\n"
        f"{_COLONY_COUNT_URL}\n"
        "\n"
        "Haul-out survey of a fictional colony, walked every Monday.\n"
        "Count: 214 seals\n"
        f"[Harbor seal colony count]({_COLONY_COUNT_URL})\n"
        "Counted by volunteers; the figure is revised if a recount is needed.\n"
    ),
)

_NEW_ARRIVALS_URL = "https://citylibrary.example/new-arrivals"
# A real catalogue page carries far more than the task needs, and this one now does too
# (#1854, code-owner ruling): three arrivals, each with its title, author, blurb, and
# shelf details.  The measured failure it fixes: a draw that over-asked the extract —
# "the title AND AUTHOR of the newest book" — got an honest NOT_PRESENT off a page
# carrying one bare line, so the value died upstream and the round had nothing to
# remember and nothing to write.  An over-ask is a defensible reading of a watch ask, so
# the page answers it instead of the round failing on a fixture's thinness.
#
# Two properties the enrichment must not break.  "The Tidewater Almanac" stays the sole
# CONTROLLABLE fact — the one the scorer matches on — so it appears nowhere but its own
# arrival, and the newest arrival is unambiguous (the other two are dated behind it in
# words, not just by position).  And each arrival's markdown link sits at the CENTRE of
# its five-line block: a SEARCH-shaped read is trimmed to ±2 lines around every solo
# link (``_trim_search_result``), so a block laid out any other way would lose the very
# fields this page was enriched to carry.
_NEW_ARRIVALS = CannedPage(
    match="library",
    text=(
        "Title: New arrivals — city library | citylibrary\n"
        f"{_NEW_ARRIVALS_URL}\n"
        "\n"
        "Titles added to a fictional catalogue, refreshed every weekday morning.\n"
        f"[City library new arrivals]({_NEW_ARRIVALS_URL})\n"
        "Listed newest first; older arrivals drop off the page after a fortnight.\n"
        "\n"
        "Newest arrival — added Tuesday\n"
        '"The Tidewater Almanac" by Marisol Enge\n'
        "[The Tidewater Almanac](https://citylibrary.example/catalogue/tidewater-almanac)\n"
        "A year of coastal weather notes, tide charts and harbour lore, kept by a "
        "small-press essayist.\n"
        "Hardcover · 312 pages · Shelf 551.46 · 3 copies, 2 available\n"
        "\n"
        "Added the Friday before that\n"
        '"The Cartwright Bequest" by Ivo Pellani\n'
        "[The Cartwright Bequest](https://citylibrary.example/catalogue/cartwright-bequest)\n"
        "A country-house mystery told backwards, from the reading of the will.\n"
        "Paperback · 288 pages · Shelf F PEL · 4 copies, 1 available\n"
        "\n"
        "Added two weeks ago\n"
        '"Kettle Lake Field Guide" by Dunja Vance\n'
        "[Kettle Lake Field Guide](https://citylibrary.example/catalogue/kettle-lake-guide)\n"
        "Birds, sedges and weather of a fictional lake district, with sketch maps.\n"
        "Spiral-bound · 176 pages · Shelf 578.7 · 2 copies, both on hold\n"
        "\n"
        "Requests and renewals are handled at the desk or through the catalogue.\n"
    ),
)


# Case 1 — the script's own turn (the journey fixture, not pool-derived): the
# canonical deictic-with-url watch ask.  Continuity is the point — the
# ``transition-elicit-to-learn`` teach turn below answers exactly this ask, so
# the per-edge set chains into a journey.
#
# Reference reply:
#   i don't have a routine for that yet — can you walk me through it once? what
#   should i read, what am i looking for, what should i remember?
_IDLE_ASK = (
    f"can you watch this listing for me daily and let me know when the price changes? {LISTING_URL}"
)

# Case 2 — the enriched derivative of fire phrasings 1 + 10 (the ferry timetable
# and its late sailing).  A named source with no page given: the turn must ASK
# where to look, never guess its way there through a search.
#
# Reference reply:
#   i can learn that — walk me through it once? where should i check the
#   timetable, and what counts as the late sailing being added?
_IDLE_ASK_NO_URL = (
    "every morning can you check the harbor ferry timetable and let me know "
    "when they add the late sailing?"
)

# Case 3 — the enriched derivative of fire phrasing 7 (the corner bakery's daily
# specials).  A store-each-day digest: the intent is named and the steps are
# absent, which is elicit — not learn, and not a browse.
#
# Reference reply:
#   happy to — show me once how you'd want it done? what page should i read, and
#   what should i save from it each day?
_IDLE_ASK_DIGEST = (
    "can you collect the daily specials from the corner bakery's site each day, "
    "keep them for me, and let me know what today's is?"
)

# Case 4 — the enriched derivative of fire phrasing 6 (the harbour seal colony
# count).  A url IS given, and the job is a number compared against last time —
# reasoning over stored state, not just storage — which still does not make the
# routine known.
#
# Reference reply:
#   i don't have a routine for that yet — walk me through it once? what should i
#   read on that page, and what number am i keeping track of?
_IDLE_ASK_THRESHOLD = (
    "watch harborseals.example/colony-count every week, keep track of the number, "
    "and let me know if it drops"
)

# Case 5 — the enriched derivative of fire phrasing 5 (the library's new-arrivals
# page).  Act-now pressure ("the moment something new shows up") must not
# stampede a fetch or a setup: urgency is a reason to ask faster, not to guess.
#
# Reference reply:
#   i can learn that — walk me through it once? where should i look, and what
#   counts as something new showing up?
_IDLE_ASK_URGENCY = (
    "can you check the library's new-arrivals page every day and tell me the "
    "moment something new shows up?"
)


def _asked_message_id(db: Database) -> int | None:
    """The id of the ask this turn answered, or ``None`` when the world is not the
    one the case claims.

    Read POST-turn deliberately: the channel logs the incoming message AFTER the
    run (so it never doubles into that turn's own recall) and ``link_message``
    back-fills it onto the moves the run caused, stamping it as the anchor when
    the move opened a round.  An idle case seeds no history — idle IS the absence
    of it — so this turn's own ask is the ONLY incoming message, and anything
    else means the precondition broke rather than the anchor did."""
    incoming = db.messages.get_user_messages(TEST_SENDER, limit=2)
    return incoming[0].id if len(incoming) == 1 else None


def _anchor_check(db: Database) -> Check:
    """The anchor was stamped: the move that parked the machine points at the ask
    that opened the round (#1827's anchor rule) — what every later turn of the
    round is classified against.

    Scored ONLY when the machine landed in elicit.  A misroute is already named
    by the landed-state advisory, and scoring the anchor on top of it would
    recount one classifier miss as an enactment failure — the anchor is a fact
    about the round THIS edge opens, and no such round exists when the edge was
    not taken."""
    label = "state: the ask is stamped as the round's anchor"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.ELICIT.value:
        return Check.na(label, kind="state")
    asked = _asked_message_id(db)
    anchored = latest.anchor_message_id
    stamped = asked is not None and anchored == asked
    return Check(
        label,
        stamped,
        rationale=None if stamped else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


def _score_idle_to_elicit(db: Database, before: set[str], reply: str) -> list[Check]:
    """The ask landed and NOTHING was enacted — the terminal state all five of
    these turns share.

    There is no routine to run yet, so the world must end exactly as it started
    and the turn's only durable trace is the machine parking itself on the ask.
    Whether the reply IS the teach question is read at joint review — one line of
    English carries no structural signal — so the single scored reply check is
    the one failure that IS structural: asking the user how the page is built."""
    written = _written_texts(_entries_written_by_this_run(db))
    fetched = _pages_fetched(db)
    enacted = [
        tool for run in chat_run_tool_sequences(db) for tool in run if tool in _ENACTING_TOOLS
    ]
    return [
        Check(
            "state: no collection was created (nothing was set up)",
            not new_collections(db, before),
            kind="state",
        ),
        Check(
            "state: this run wrote no entry anywhere",
            not written,
            rationale=f"wrote {written}" if written else None,
            kind="state",
        ),
        Check(
            "state: no skill was learned (no round ran to learn from)",
            not db.skills.list_all(),
            kind="state",
        ),
        Check(
            "state: no page was fetched (browse-results stayed empty)",
            not fetched,
            kind="state",
        ),
        Check(
            "state: the seeded collection untouched",
            not collection_entries(db, PennyConstants.MEMORY_DISLIKES_COLLECTION),
            kind="state",
        ),
        _anchor_check(db),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        Check(
            "calls: the machine landed in elicit",
            _landed_state(db) == ConversationState.ELICIT.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: no enacting calls (orientation reads are fine)",
            not enacted,
            rationale=f"enacted {enacted}" if enacted else None,
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_to_be_taught(chat_eval: ChatEval) -> None:
    """idle → elicit: the canonical watch ask, the page named in it and reachable.
    No routine covers it, so the turn is the question — the listing is never
    opened, nothing is stored, and the machine parks on the ask."""
    await chat_eval(
        case_id="transition-idle-to-elicit",
        message=_IDLE_ASK,
        browse=[AURORA_LISTING_499],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_where_to_look(chat_eval: ChatEval) -> None:
    """idle → elicit with the source NAMED but no page given: the timetable is
    findable and the search would work, which is exactly why not running it is
    the contract.  She asks where to check instead of guessing her way there."""
    await chat_eval(
        case_id="transition-idle-to-elicit-no-url",
        message=_IDLE_ASK_NO_URL,
        browse=[_FERRY_TIMETABLE],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_before_collecting(chat_eval: ChatEval) -> None:
    """idle → elicit on a store-each-day digest: the ask names what to collect and
    where to keep it, but never the steps — so the turn asks to be shown once
    rather than starting the collection it was told the shape of."""
    await chat_eval(
        case_id="transition-idle-to-elicit-digest",
        message=_IDLE_ASK_DIGEST,
        browse=[_BAKERY_SPECIALS],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_what_to_track(chat_eval: ChatEval) -> None:
    """idle → elicit with a url in hand and a number to compare against last time.
    Having the page is not having the routine: nothing is read, no baseline is
    written, and the turn asks what it is meant to be keeping track of."""
    await chat_eval(
        case_id="transition-idle-to-elicit-threshold",
        message=_IDLE_ASK_THRESHOLD,
        browse=[_COLONY_COUNT],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_despite_the_urgency(chat_eval: ChatEval) -> None:
    """idle → elicit under act-now pressure ("the moment something new shows up").
    The urgency is a reason to ask faster, not to guess: the page stays unread and
    nothing is configured on the strength of how soon they want it."""
    await chat_eval(
        case_id="transition-idle-to-elicit-urgency",
        message=_IDLE_ASK_URGENCY,
        browse=[_NEW_ARRIVALS],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ── elicit → learn: the teach question answered, the round run once ───────────
#
# Five cases, one per idle → elicit ask above, so the two edges chain subject for
# subject.  Each starts where its own sibling stopped: the instigating ask logged
# INCOMING as the round's ANCHOR, Penny's teach question logged OUTGOING as her last
# turn, and the machine parked in elicit ON that ask.  An elicit park ALWAYS has an
# anchor — a bare park is a state production never produces, and it left the turn
# classified against no task at all.
#
# The user then answers that question with the steps, in their own words: the
# three-step look-up / extract / remember shape.  Cases 2, 3 and 5 supply the url
# their ask never gave, which makes the url user-supplied in EVERY scenario — the one
# piece the framer then has to mint as a parameter.  Each page carries exactly one
# controllable fact, so what she stored is provable from the entry alone.
#
# The reference replies quoted above each case are review targets (#1827's turn-2
# shape: report what was found and what was saved, then the offer), never scorer
# strings.


# The user answers the teach question with the steps — the very question case 1
# above ends on, so the two edges chain.  ONE constant: the learn → apply seed below
# replays this same turn as the round it is parked after.
_TEACH_TURN = f"yeah — go to {LISTING_URL}, find the current price, and remember it"


class _LearnCase(NamedTuple):
    """One agreed elicit → learn pair, and the world its turn is answered against.

    ``ask`` is the sibling idle → elicit case's ask, seeded INCOMING — the round's
    anchor.  ``teach_question`` is that case's reference reply, seeded OUTGOING as
    Penny's last turn (the same agreed line documented above it, here as data rather
    than as prose).  ``demo`` is the turn under test.  ``page`` is what the
    demonstration reads, and ``stored`` the one controllable fact it carries — what
    makes browse-sourced storage provable, in the entry AND in the reply."""

    case_id: str
    ask: str
    teach_question: str
    demo: str
    page: CannedPage
    stored: str


# Case 1 — the script's own turn, continuing ``transition-idle-to-elicit``.
#
# Reference reply:
#   opened the listing, found the price ($499), and saved it. i know how to do that
#   now — want me to keep it up to date on its own?
_AURORA_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn",
    ask=_IDLE_ASK,
    teach_question=(
        "i don't have a routine for that yet — can you walk me through it once? "
        "what should i read, what am i looking for, what should i remember?"
    ),
    demo=_TEACH_TURN,
    page=AURORA_LISTING_499,
    stored="499",
)

# Case 2 — continuing ``transition-idle-to-elicit-no-url``: the ask named a source and
# no page, so the demonstration is where the url arrives.
#
# Reference reply:
#   read the timetable — the late sailing is not scheduled this season, and i've
#   saved that. i know how to do that now — want me to keep checking on my own?
_FERRY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-no-url",
    ask=_IDLE_ASK_NO_URL,
    teach_question=(
        "i can learn that — walk me through it once? where should i check the "
        "timetable, and what counts as the late sailing being added?"
    ),
    demo=(
        f"go to {_FERRY_TIMETABLE_URL}, look for the late sailing line, and remember what it says"
    ),
    page=_FERRY_TIMETABLE,
    stored="not scheduled",
)

# Case 3 — continuing ``transition-idle-to-elicit-digest``: the store-each-day digest,
# demonstrated once.
#
# Reference reply:
#   opened the specials page — today's special is the rye and caraway loaf, saved it.
#   i know how to do that now — want me to keep it up each day?
_BAKERY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-digest",
    ask=_IDLE_ASK_DIGEST,
    teach_question=(
        "happy to — show me once how you'd want it done? what page should i read, "
        "and what should i save from it each day?"
    ),
    demo=f"open {_BAKERY_SPECIALS_URL}, find today's special, and remember it",
    page=_BAKERY_SPECIALS,
    stored="rye",
)

# Case 4 — continuing ``transition-idle-to-elicit-threshold``: a number to keep track
# of, demonstrated as a plain read-and-remember (the comparison is a later beat's).
#
# Reference reply:
#   checked the survey page — the colony count is 214, and i've saved it. i know how
#   to do that now — want me to keep tracking it?
_COLONY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-threshold",
    ask=_IDLE_ASK_THRESHOLD,
    teach_question=(
        "i don't have a routine for that yet — walk me through it once? what should "
        "i read on that page, and what number am i keeping track of?"
    ),
    # The ask gave this address without a scheme and the demonstration repeats it that
    # way — the user's own words, not a normalized copy of the page's own constant.
    demo="go to harborseals.example/colony-count, find the current count, and remember it",
    page=_COLONY_COUNT,
    stored="214",
)

# Case 5 — continuing ``transition-idle-to-elicit-urgency``: the act-now ask, taught.
#
# Reference reply:
#   checked the new-arrivals page — the newest arrival is "The Tidewater Almanac",
#   saved it. i know how to do that now — want me to keep watching for new ones?
_ARRIVALS_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-urgency",
    ask=_IDLE_ASK_URGENCY,
    teach_question=(
        "i can learn that — walk me through it once? where should i look, and what "
        "counts as something new showing up?"
    ),
    demo=f"check {_NEW_ARRIVALS_URL}, find the newest arrival, and remember it",
    page=_NEW_ARRIVALS,
    stored="Tidewater",
)


# The round's incoming turns by scoring time: the ask this seed lays down, and the
# demonstration the channel logs AFTER the run (#1566's deferred link).  Nothing else
# can arrive — the only other speaker is Penny.
_ROUND_INCOMING_TURNS = 2


def _seed_elicit_round(case: _LearnCase) -> Seeder:
    """Lay down the state the PRECEDING beat ends in, item for item — this edge starts
    where ``idle → elicit`` stops, so its precondition is that beat's scored terminal
    state and nothing else:

    * the instigating ask, logged INCOMING — the message the round is ANCHORED to
    * Penny's teach question, logged OUTGOING — her last turn, the one the user's
      demonstration answers
    * the machine parked in ``elicit``, carrying that ask as its anchor
    * nothing else at all: an empty registry, no collection of her making, no page read

    The case's page is installed by the runner, so the demonstration reads a real one."""

    def seed(db: Database) -> None:
        ask_id = db.messages.log_message(
            direction=PennyConstants.MessageDirection.INCOMING,
            sender=TEST_SENDER,
            content=case.ask,
        )
        db.messages.log_message(
            direction=PennyConstants.MessageDirection.OUTGOING,
            sender=PennyConstants.MessageAuthor.PENNY,
            content=case.teach_question,
        )
        _park(db, ConversationState.ELICIT, anchor_message_id=ask_id)
        _assert_seeded_world(db, case, ask_id)

    return seed


def _assert_seeded_world(db: Database, case: _LearnCase, ask_id: int | None) -> None:
    """Loud probe: the seeded world IS the sibling idle → elicit case's scored terminal
    state — parked in elicit, on THIS ask.

    A seed that has drifted from the state the preceding beat is measured against makes
    this case a turn answered against a world nothing produces — which is precisely what
    the bare park was — so it fails HERE, in the seed, rather than as a puzzling number
    after an hour of GPU time.  Same discipline as the apply case's fixture asserts: a
    fixture states what it means and says so out loud when it stops being true."""
    assert ask_id is not None, f"{case.case_id}: the seeded ask must carry a message id"
    assert _seeded_ask_id(db, case.ask) == ask_id, (
        f"{case.case_id}: the seeded ask must be findable by its own content"
    )
    _assert_nothing_enacted(db, case)
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.ELICIT.value, (
        f"{case.case_id}: the machine must be parked in elicit, not {latest}"
    )
    assert latest.anchor_message_id == ask_id, (
        f"{case.case_id}: the park must be anchored to the ask, not {latest.anchor_message_id}"
    )


def _assert_nothing_enacted(db: Database, case: _LearnCase) -> None:
    """The other half of that probe — turn 1 enacted NOTHING, which is the whole of what
    its five scored state checks assert: no skill learned, no entry written by any run,
    no page fetched, and the framework's own seeded collection untouched."""
    assert not db.skills.list_all(), f"{case.case_id}: the round starts with no skill learned"
    assert not _entries_written_by_this_run(db), f"{case.case_id}: no run has written anything"
    assert not _pages_fetched(db), f"{case.case_id}: no page has been fetched yet"
    assert not collection_entries(db, PennyConstants.MEMORY_DISLIKES_COLLECTION), (
        f"{case.case_id}: the seeded collection starts untouched"
    )


def _seeded_ask_id(db: Database, ask: str) -> int | None:
    """The id of the seeded instigating ask — the row this round is anchored to.

    Found by its CONTENT rather than by position: the demonstration is logged AFTER the
    run (so it never doubles into that turn's own recall), so by scoring time there are
    two incoming rows and only the case knows which of them it seeded."""
    for row in db.messages.get_user_messages(TEST_SENDER, limit=_ROUND_INCOMING_TURNS):
        if row.content == ask:
            return row.id
    return None


def _mentions(token: str, texts: list[str]) -> bool:
    """Whether the page's own fact turns up in any of ``texts``, CASE-FOLDED.

    The fact is a word off the page ('rye', 'not scheduled', 'Tidewater'), and a value
    lifted into a sentence takes whatever capitalisation the sentence needs — so
    matching case-sensitively would score correct writes as misses on most of these
    pages, which is a scorer bug reported as a finding."""
    return any(token.lower() in text.lower() for text in texts)


def _skill_steps(db: Database) -> list[SkillStep]:
    """Every step of every learned skill — the ROUTINE run-end extraction left behind,
    read structurally off the stored rows rather than off the demonstration.

    The step, not its substitutions alone, because a leaf's demonstrated value lives in
    its step's ``arguments`` and that is what says whether the leaf is a destination
    (#1854)."""
    return [step for skill in db.skills.list_all() for step in steps_from_json(skill.steps)]


def _skill_substitutions(steps: list[SkillStep]) -> list[SkillSubstitution]:
    """Every dynamic leaf of the learned routine — the SHAPE of what was captured."""
    return [sub for step in steps for sub in step.substitutions]


def _skill_parameters(db: Database) -> list[SkillParameter]:
    """Every declared parameter of every learned skill — the routine's INTERFACE, which
    since #1830 is the framer's draw and lives at SKILL level (its declared interim:
    nothing joins a parameter to a leaf of the program yet)."""
    return [
        parameter
        for skill in db.skills.list_all()
        for parameter in parameters_from_json(skill.parameters)
    ]


# The three shape labels, named once: each is read BOTH as a scored check and as the
# not-applicable row a sample with no learned skill renders, and a label is a diff-join
# key — two spellings of one check are two checks to every report that reads them.
_PLACEHOLDERS_ONLY_LABEL = (
    "state: every spot in the routine is a placeholder (the labelling draw landed)"
)
_ATTACHMENT_MARK_LABEL = "state: the destination leaf still carries the attachment mark"
_INTERFACE_LABEL = "state: the interface asks for the page, plus at most the found-thing"

# What the interface may ask for, as the two families a drawn parameter can answer —
# classified by the SHARED name-first-then-description discipline (``classify_by_family``),
# so this check and the framer suite's own set check can never read a draw two ways.
#
# The PAGE is mandatory and NAME-ONLY, on the framer suite's breadth (imported rather than
# restated: what a page parameter may reasonably be called is one agreed vocabulary, and a
# page is the thing NAMED as one — a description mentioning a page promotes nothing).
_PAGE_LABEL = "page"
_PAGE_FAMILY = ParameterFamily(_PAGE_LABEL, _PLACE_TOKENS, name_only=True)

# The FOUND-THING is the leeway (code-owner ruling, 2026-08-05, from a thinking-audited
# draw): every one of these asks names what to look for as well as where — "look for the
# late sailing line" — so a second parameter carrying THAT is a defensible reading of the
# enumerate-then-filter rule, not an invention, and the check accepts at most one.  Both
# passes apply here, unlike the page: the piece has no canonical noun, so a well-judged
# name the tokens don't anticipate is allowed to land through its description.
_FOUND_THING_LABEL = "found-thing"
_FOUND_THING_FAMILY = ParameterFamily(
    _FOUND_THING_LABEL, ("search", "phrase", "term", "keyword", "target", "line", "query")
)
_INTERFACE_FAMILIES = (_PAGE_FAMILY, _FOUND_THING_FAMILY)


def _placeholders_only_check(subs: list[SkillSubstitution]) -> Check:
    """Every spot in the routine is a PLACEHOLDER — none is still a leaf parameter
    (#1828).

    The labeller names every spot unconditionally and a named spot stops being a
    parameter, so a leftover ``HOLE`` means the labelling draw FELL BACK (it is
    all-or-nothing at the draw) and the routine kept its arg-derived names.  Bindings
    are untouched by any of this: a value a prior step produced was never asked of
    anyone."""
    left = [sub for sub in subs if sub.kind == SkillSubKind.HOLE]
    asking = sorted({sub.parameter for sub in left if sub.parameter is not None})
    return Check(
        _PLACEHOLDERS_ONLY_LABEL,
        not left,
        rationale=f"{len(left)} spot(s) still a leaf parameter: {asking}" if left else None,
        kind="state",
    )


def _attachment_mark_check(db: Database, steps: list[SkillStep]) -> Check:
    """The destination leaf still carries the ATTACHMENT MARK (#1783, #1827 principle
    4) — scored only when the routine HAS a destination (#1854).

    Where a routine writes is decided by what it is applied to and is never asked of the
    user, and the mark is exactly what the apply turn binds — so a routine whose
    destination came back unmarked is one the next edge cannot point anywhere.

    A routine that keeps nothing has no such leaf to mark, and read-only routines are
    legitimate (code-owner ruling: "there's tons of skills that be like 'check the scores
    here, check the schedule there, tell me' — that doesn't require a store step").  Since
    #1850 a learn round is extracted whatever shape it had, so a browse-only skill is now
    a state this suite reaches, and grading it here would fail a routine for a step nobody
    asked for.

    Applicability is read from the DEMONSTRATED VALUES, never from the marks — "is
    anything marked?" is the check itself, so answering applicability with it would pass
    every routine vacuously and never catch a dropped mark.  A leaf is a destination when
    its demonstrated value names one of Penny's own collections, which is exactly what
    ``distill_steps`` marks on, read through the same registry policy extraction uses
    (``attachment_names``).  Bindings are excluded there and excluded here: a value a
    prior step produced is already explained, so nothing is left for an attachment to
    decide.  Keyed to no tool name — a skill is an arbitrary tool sequence, so a plugin
    verb's destination counts like a ``collection_write``'s."""
    destinations = _destination_subs(db, steps)
    if not destinations:
        return Check.na(_ATTACHMENT_MARK_LABEL, kind="state")
    marked = any(sub.attachment for sub in destinations)
    return Check(
        _ATTACHMENT_MARK_LABEL,
        marked,
        rationale=None if marked else "the destination leaf came back unmarked",
        kind="state",
    )


def _destination_subs(db: Database, steps: list[SkillStep]) -> list[SkillSubstitution]:
    """Every leaf of the routine that points at one of Penny's own collections — the
    spots the attachment fills, identified by their demonstrated value alone."""
    collections = attachment_names(db)
    return [
        sub
        for step in steps
        for sub in step.substitutions
        if sub.kind != SkillSubKind.BINDING and _leaf_at(step.arguments, sub.path) in collections
    ]


def _interface_check(required: list[SkillParameter]) -> Check:
    """The interface asks for the PAGE, plus AT MOST the found-thing (#1830, amended by
    the code owner's leeway ruling of 2026-08-05).

    The page is mandatory — it is the one piece every one of these asks leaves to re-say,
    and a routine that cannot be pointed at one can only repeat its demonstration.  A
    SECOND parameter is accepted when it carries what the user's own turns named as the
    thing to find: the ferry round's draw asked for a `search_phrase`, and the audited
    thinking read "the late sailing" out of both turns — which is the enumerate-then-filter
    rule applied correctly, so scoring it a miss would be the scorer marking a sound draw
    wrong.  Anything else stays a miss: a second parameter of another kind is the invention
    that rule exists to stop, and a third is one however it is named.  Every accepted
    parameter carries a description — it is what the ambient ``needs:`` row renders, so one
    nobody can read is one nobody can bind."""
    answered = _interface_families(required)
    pages, found, rejected = (_of_family(required, answered, label) for label in _READINGS)
    accepted = len(pages) == 1 and len(found) <= 1 and not rejected
    described = all(_says_what_to_supply(parameter) for parameter in pages + found)
    return Check(
        _INTERFACE_LABEL,
        accepted and described,
        rationale=_interface_rationale(pages, found, rejected, described),
        kind="state",
    )


def _interface_families(required: list[SkillParameter]) -> list[ParameterFamily | None]:
    """Which family each required parameter answers, through the SHARED classifier — a
    parameter carries no description in the model when a draw left none, and an absent
    description classifies as the empty text it is."""
    return classify_by_family(
        [(parameter.name, parameter.description or "") for parameter in required],
        _INTERFACE_FAMILIES,
    )


# The three readings a required parameter can land in, in the order the rationale names
# them: the mandatory page, the accepted found-thing, and everything else.
_READINGS = (_PAGE_LABEL, _FOUND_THING_LABEL, None)


def _of_family(
    required: list[SkillParameter],
    answered: list[ParameterFamily | None],
    label: str | None,
) -> list[SkillParameter]:
    """The required parameters that answered ``label`` — ``None`` for the ones that
    answered no accepted family at all."""
    return [
        parameter
        for parameter, family in zip(required, answered, strict=True)
        if (family.label if family is not None else None) == label
    ]


def _says_what_to_supply(parameter: SkillParameter) -> bool:
    """A parameter carries the one-line what-to-supply the framer writes for it — the
    description is optional in the model (a labelling fallback leaves none), so an
    absent one is a real, distinct shape and not something to read as empty text."""
    return parameter.description is not None and not is_blank(parameter.description)


def _interface_rationale(
    pages: list[SkillParameter],
    found: list[SkillParameter],
    rejected: list[SkillParameter],
    described: bool,
) -> str:
    """WHICH reading was drawn, named on the pass as well as the miss — the two accepted
    shapes are different answers to the same ask, and a report that showed only "passed"
    would hide which one the run committed to."""
    if rejected:
        names = ", ".join(parameter.name for parameter in rejected)
        return f"rejected: {names} answers no accepted family"
    if len(pages) != 1:
        return f"{len(pages)} answer the page: {[parameter.name for parameter in pages]}"
    if len(found) > 1:
        return f"{len(found)} answer the found-thing: {[parameter.name for parameter in found]}"
    if not described:
        undescribed = [p.name for p in pages + found if not _says_what_to_supply(p)]
        return f"carries no description: {', '.join(undescribed)}"
    if not found:
        return f"{pages[0].name} alone"
    return f"{pages[0].name} + {found[0].name} (user-named)"


def _interface_advisories(db: Database) -> list[Check]:
    """What the framer committed to, verbatim — one ADVISORY row per parameter.

    Whether a name is WELL judged is read at joint review against the reference outputs
    on the ticket; a scorer that faked that reading would be answering for the draw."""
    return [
        Check(
            f"drew parameter {parameter.name!r} — {parameter.description!r}",
            True,
            scored=False,
            kind="state",
        )
        for parameter in _skill_parameters(db)
    ]


def _extraction_shape_checks(db: Database) -> list[Check]:
    """The shape run-end extraction produced, read off the stored skill: the LABELLER's
    half (every spot a placeholder, and — where the routine keeps anything — the
    destination still marked) and the FRAMER's half (one required parameter, described),
    with the drawn interface riding advisory.

    All three go NOT-APPLICABLE when no skill was learned at all.  That miss is already
    the scored "a skill was learned from the round" check, so grading the shape of a
    skill that does not exist would recount one failure three times — and "every spot is
    a placeholder" over an empty routine is vacuously true, which would render as a pass
    for a round that produced nothing.  The mark check has a second not-applicable case
    of its own (#1854): a routine with no destination has nothing to mark."""
    if not db.skills.list_all():
        return [
            Check.na(_PLACEHOLDERS_ONLY_LABEL, kind="state"),
            Check.na(_ATTACHMENT_MARK_LABEL, kind="state"),
            Check.na(_INTERFACE_LABEL, kind="state"),
        ]
    steps = _skill_steps(db)
    required = [parameter for parameter in _skill_parameters(db) if parameter.required]
    return [
        _placeholders_only_check(_skill_substitutions(steps)),
        _attachment_mark_check(db, steps),
        _interface_check(required),
        *_interface_advisories(db),
    ]


def _attaches_nothing_checks(db: Database, created: list[MemoryRow]) -> list[Check]:
    """Learning must not INSTANTIATE (#1706).  Scored against what this run TOUCHED: a
    collection it created, or — when it reused an existing one — nothing, since a seeded
    collection's own prompt and cadence predate the round and failing on those would
    report the framework's fixtures as her doing."""
    instantiated = [row for row in db.memories.list_all() if row.skill_name is not None]
    return [
        Check(
            "state: no skill was attached anywhere (learning does not instantiate)",
            not instantiated,
            rationale=f"attached to {[row.name for row in instantiated]}" if instantiated else None,
            kind="state",
        ),
        Check(
            "state: no program was rendered into the collection it created",
            all(row.extraction_prompt is None for row in created),
            kind="state",
        )
        if created
        else Check.na(
            "state: no program was rendered into the collection it created", kind="state"
        ),
        Check(
            "state: nothing it created was scheduled (no trigger, no notify)",
            all(row.collector_interval_seconds is None and not row.notify for row in created),
            kind="state",
        )
        if created
        else Check.na(
            "state: nothing it created was scheduled (no trigger, no notify)", kind="state"
        ),
    ]


def _anchor_carried_check(db: Database, ask: str) -> Check:
    """The anchor was CARRIED: the move that parked the machine in learn still points at
    the ask that opened the round (#1827's anchor rule — every transition that keeps the
    machine parked carries it unchanged, which is what lets a turn three messages later
    still be classified against what was asked for).

    Scored ONLY when the machine landed in learn — the same conditional the idle → elicit
    cases use: a misroute is already named by the landed-state advisory, and scoring the
    anchor on top of it would recount one classifier miss as an enactment failure."""
    label = "state: the anchor was carried (still the ask that opened the round)"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.LEARN.value:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, ask)
    anchored = latest.anchor_message_id
    carried = asked is not None and anchored == asked
    return Check(
        label,
        carried,
        rationale=None if carried else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


def _score_elicit_to_learn(
    db: Database, before: set[str], reply: str, *, case: _LearnCase
) -> list[Check]:
    """The demonstrated round ran, and NOTHING was instantiated.

    "Remember it" is a naive ``collection_write``: it auto-creates a collection
    and puts the value in it.  What must NOT happen is the fold — no skill bound
    to that collection, no rendered program, no schedule.  The skill is learned
    (it exists in the registry) and stays unattached until the user asks for it.
    What that learning PRODUCED is read off the stored skill: an all-placeholder
    routine, its destination still marked, over the one parameter the ask leaves.

    ONE scorer for all five cases, bound to the case's own page fact and ask.  The
    labels are diff-join keys, so they read identically on every case and keep the
    wording the auction script gave them even where a ferry timetable is what was
    read."""
    created = new_collections(db, before)
    written = _written_texts(_entries_written_by_this_run(db))
    landed = _mentions(case.stored, written)
    return [
        Check(
            "state: she browsed the listing (the demonstrated fetch happened)",
            tool_was_called(db, "browse"),
            kind="state",
        ),
        # The fact counts wherever in the entry it landed — its KEY or its content
        # (#1854, code-owner ruling: "loosen the scorer; we can reason about the
        # semantics of keys/values/remembering later").  Two measured samples wrote the
        # arrival's title as the KEY and the date as the value, which is a workable
        # shape for an arrival-shaped watch — a repeat title is KEY_EXISTS_UNCHANGED
        # and a new one is a new key — and was scored a miss for putting the fact on
        # the wrong side of the entry.  The label is unchanged: it is a diff-join key,
        # and what the check tests is still that the browsed fact landed durably.
        Check(
            "state: the browsed price landed durably (remember = a plain write)",
            landed,
            rationale=None
            if landed
            else (f"wrote {written}" if written else "nothing was written"),
            kind="state",
        ),
        Check(
            "state: a skill was learned from the round",
            bool(db.skills.list_all()),
            kind="state",
        ),
        *_attaches_nothing_checks(db, created),
        *_extraction_shape_checks(db),
        _anchor_carried_check(db, case.ask),
        Check(
            "reply: she reports the value she stored (SAID == DID)",
            _mentions(case.stored, outgoing_replies(db)),
            kind="reply",
        ),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        Check(
            "calls: the machine landed in learn",
            _landed_state(db) == ConversationState.LEARN.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def _run_learn_case(chat_eval: ChatEval, case: _LearnCase) -> None:
    """Drive one elicit → learn case: parked on its own ask, its page installed, the
    shared scorer bound to the fact that page carries.  Report-only — the thresholds are
    the code owner's to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.demo,
        browse=[case.page],
        seed=_seed_elicit_round(case),
        score=partial(_score_elicit_to_learn, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_elicit_to_learn_runs_the_round_and_instantiates_nothing(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn: parked on the teach question, the user supplies the steps.
    She follows them once — browse, find, remember — reports the value she
    actually stored, and learns the skill.  She instantiates NOTHING: the
    collection her write created carries no skill, no program, no schedule."""
    await _run_learn_case(chat_eval, _AURORA_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_takes_the_url_from_the_demonstration(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn where the ask never gave a page: the demonstration supplies the
    timetable's url along with what to look for on it, and the round runs on what she
    was just told rather than on a search she guessed her way to."""
    await _run_learn_case(chat_eval, _FERRY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_stores_the_days_special(chat_eval: ChatEval) -> None:
    """elicit → learn on the store-each-day digest: shown the routine once, she runs it
    once — today's special read off the page and written down — and the day-after-day
    part stays a job nobody has set up yet."""
    await _run_learn_case(chat_eval, _BAKERY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_records_the_count_without_comparing(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn where the job is a number watched over time: the demonstration is
    a plain read-and-remember, so the count lands as the baseline it is and nothing
    compares it against anything — there is nothing yet to compare it to."""
    await _run_learn_case(chat_eval, _COLONY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_learns_despite_the_urgency(chat_eval: ChatEval) -> None:
    """elicit → learn under the act-now ask: the instructions have arrived now, so the
    round is exactly what they say — read the page, take the newest arrival, remember
    it — and the "tell me the moment" part is still a job the next turn sets up."""
    await _run_learn_case(chat_eval, _ARRIVALS_ROUND)


# ── learn → apply: the offer accepted, the routine set running ────────────────

# The world the demonstrated round leaves behind, seeded so this case stays ONE
# turn: the collection the naive write created (the price in it, no skill, no
# program, no schedule — exactly what the elicit → learn case above scores), and
# the skill the run-end extractor distilled from that same round.
_WATCH_COLLECTION = "aurora-deck-2-price"
_DEMO_KEY = "aurora deck 2 price"
_PRICE = "$499"
_EXTRACT = "the current price"

_SKILL_NAME = "watch a listing page for its current price"
_SKILL_DESCRIPTION = "read a listing page and record the price it shows"

# The round's ledger, as the extractor would have read it off the promptlog.
_DEMONSTRATED_ROUND = [
    DistillInput(
        source_ordinal=1,
        tool="browse",
        arguments={"queries": [LISTING_URL], "extract": _EXTRACT},
        result=f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    ),
    DistillInput(
        source_ordinal=2,
        tool="collection_write",
        arguments={
            "memory": _WATCH_COLLECTION,
            "entries": [{"key": _DEMO_KEY, "content": _PRICE}],
        },
        result=(
            f"You saved an entry to {_WATCH_COLLECTION}: (collection_write result)\nWrote 1 entry."
        ),
    ),
]

# The LABELLER's draw (#1828), keyed by the DEMONSTRATED VALUE rather than by the
# arg-derived name the distiller happens to mint — so the fixture states what it means
# and cannot go quietly stale if that naming changes.  EVERY spot is listed because an
# accepted draw covers every spot: a line missing for one is a WHOLE-draw failure, so a
# partly-labelled routine is a shape run-end extraction cannot hand anyone.  The
# destination is a spot like any other and additionally carries the attachment mark —
# what the routine is applied to fills it, which is precisely what the apply turn under
# test then binds.
_LABELS = {
    LISTING_URL: LeafLabel(
        name="listing_page", description="the page whose price this routine reads"
    ),
    _EXTRACT: LeafLabel(name="value_to_find", description="what to pull off the page each run"),
    _DEMO_KEY: LeafLabel(name="entry_key", description="what to call the entry it saves"),
    _WATCH_COLLECTION: LeafLabel(name="storage_collection", description=WRITE_TARGET_DESCRIPTION),
}

# The FRAMER's draw (#1830) — the one thing here still hand-written, because a draw is
# what a fixture legitimately stands in for.  The framer writes a routine's INTERFACE
# from the user's ask alone, and this round's ask — watch this listing and tell me when
# the price changes — makes the page the one thing to re-say and the price what the
# routine IS.  Everything that APPLIES this signature is production code below.
_FRAMED_SIGNATURE = SkillSignature(
    name=_SKILL_NAME,
    description=_SKILL_DESCRIPTION,
    parameters=(FramedParameter(name="url", description="the listing page to check"),),
)


def _fixture_labels(steps: list[SkillStep]) -> SkillLabels:
    """The labeller's draw, mapped from the demonstrated VALUES it is authored
    against onto the spot names the distiller happened to mint.

    Every authored label must map home: one that doesn't is a fixture whose ledger
    has drifted from what it claims, and it fails LOUDLY here rather than quietly
    seeding the enactment case a world with a spot left unnamed."""
    labels: dict[str, LeafLabel] = {}
    for step in steps:
        for sub in step.substitutions:
            if sub.parameter is None:
                continue
            value = str(_leaf_at(step.arguments, sub.path))
            if value in _LABELS:
                labels[sub.parameter] = _LABELS[value]
    assert len(labels) == len(_LABELS), (
        f"the fixture's labels must all map home — matched {sorted(labels)} of {sorted(_LABELS)}"
    )
    return SkillLabels(labels=labels)


def learn_to_apply_fixture_skill() -> SkillDraft:
    """The skill that round leaves in the registry, built by the PRODUCTION pipeline
    over its ledger: ``distill_steps`` for the structure, then BOTH halves of the
    run-end split applied by their own production function — ``_apply_leaf_labels``
    for the labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's
    signature.  Only the two DRAWS are hand-written, which is what a fixture is for.

    So the case's starting world is the shape extraction produces, not a convenient
    copy of it — and the shape is the framer's declared interim (#1830), which nothing
    here re-states: an ALL-PLACEHOLDER recipe (every spot named by the labeller, the
    write target still carrying its attachment mark) over ONE skill-level parameter,
    the page.  Nothing joins that parameter to a leaf yet — that is the runtime-join
    beat — so it is the registry row, ``collection_set``'s unbound-parameter check,
    and job identity that carry it, which is exactly what the apply turn under test
    has to satisfy."""
    # The registry as this fixture's round saw it — #1783 marks a leaf whose
    # demonstrated value names one of Penny's collections, so the destination is
    # only marked if the collection actually existed.
    steps, parameters = distill_steps(_DEMONSTRATED_ROUND, frozenset({_WATCH_COLLECTION}))
    steps, distilled = _apply_leaf_labels(steps, parameters, _fixture_labels(steps))
    name, description = _naming(_FRAMED_SIGNATURE, _TEACH_TURN)
    framed = _interface_parameters(_FRAMED_SIGNATURE, distilled)
    # The framer's parameter is the one thing the apply turn must supply, so a
    # production application that stopped carrying the signature through would seed a
    # routine nothing could be pointed at — silently, and the apply case would report
    # it as the model's failure.  It fails here instead.
    framed_names = [parameter.name for parameter in framed]
    assert framed_names == [parameter.name for parameter in _FRAMED_SIGNATURE.parameters], (
        f"the framed interface must survive application — got {framed_names}"
    )
    return SkillDraft(
        name=name,
        intent=description,
        description=description,
        steps=steps,
        parameters=framed,
        source_run_id="demonstrated-round",
    )


# The learn round's closing reply, in the shape LEARN_INSTRUCTION asks for: what
# each step produced, what she now knows how to do, and the offer to set it
# running.  That last clause is the message this edge's user turn answers — an
# acceptance is only an acceptance of something.
_PENNY_REPORT = (
    f"Opened the listing, found the price ({_PRICE}), and saved it to "
    f"{_WATCH_COLLECTION}. I know how to do that now — read a listing page and "
    "record the price it shows. Want me to keep it up to date on its own?"
)

# learn → apply: the offer taken up.  It names a cadence, an end condition, and
# a notify ask — but NOT the page, which the round it is answering already read.
_APPLY_TURN = "perfect — do that every hour until 10pm tonight and tell me if it changes"


def _seed_demonstrated_round(db: Database) -> None:
    """Lay down the state the PRECEDING beat ends in, item for item — this edge
    starts where ``elicit → learn`` stops, so its precondition is that beat's
    scored terminal state and nothing else:

    * the teach turn that opened the learn round, and Penny's closing report —
      she ran it, and she says what she now knows how to do
    * the collection her naive write created, holding the price, carrying no
      skill and no program and no schedule (learning instantiates nothing)
    * a learned skill in the registry (seeded by the case's ``seed_skills``)
    * the machine parked in ``learn``, anchored to the teach turn

    The instigating ask ("can you watch this for me?") is deliberately ABSENT.
    It belongs to the beat before — ``idle → elicit`` — and seeding it made the
    classifier read "the task being worked on" as a setup still being specified,
    which is a fair reading of a request that has not been carried out yet.  It
    has been: that is what the learn round did."""
    teach_id = db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content=_TEACH_TURN,
    )
    db.messages.log_message(
        direction=PennyConstants.MessageDirection.OUTGOING,
        sender=PennyConstants.MessageAuthor.PENNY,
        content=_PENNY_REPORT,
    )
    db.memories.create_collection(_WATCH_COLLECTION, "the aurora deck 2 listing price")
    require_memory(db, _WATCH_COLLECTION).write(
        [EntryInput(key=_DEMO_KEY, content=_PRICE)],
        author=PennyConstants.CHAT_AGENT_NAME,
    )
    _park(db, ConversationState.LEARN, anchor_message_id=teach_id)


def _instantiated(db: Database):
    """The collection the taught skill was applied to — WHICHEVER one she chose.

    Which collection a job lands on is deliberately NOT this case's business (code
    owner): she has created several where one was meant since well before the
    machine existed, so where jobs accumulate is a collection-management question
    of its own and grading it here would report that standing problem as a
    transition failure.  This edge owns whether the skill is APPLIED correctly —
    bound, rendered, and scheduled on the terms given — so every check reads the
    row that carries the skill, and the one about reuse rides along unscored."""
    taught = slug_skill_name(_SKILL_NAME)
    applied = [row for row in db.memories.list_all() if row.skill_name == taught]
    return applied[0] if applied else None


def _bound_values(row) -> list[str]:
    """The values she bound into the skill at instantiation, from the collection's
    own provenance column (#1603) — a read, not an inference."""
    return [str(value) for value in json.loads(row.skill_params or "{}").values()]


def _score_learn_to_apply(db: Database, before: set[str], reply: str) -> list[Check]:
    """The taught routine became a live job on the terms they gave — bound to the
    page the round read, rendered, scheduled, and notifying — without re-running
    the round to answer.  WHERE that job lives is not scored (see
    ``_instantiated``); it rides along as an advisory so the choice stays visible."""
    row = _instantiated(db)
    created = new_collections(db, before)
    bound = _bound_values(row) if row else []
    reused = row is not None and row.name == _WATCH_COLLECTION
    sets = count_tool_calls(db, "collection_set")
    return [
        Check(
            "state: she set the job up with collection_set",
            tool_was_called(db, "collection_set"),
            kind="state",
        ),
        Check(
            "state: the taught skill was applied to a collection",
            row is not None,
            rationale=None if row else "no collection carries the skill",
            kind="state",
        ),
        Check(
            "state: the skill's program was rendered into it",
            row is not None and bool(row.extraction_prompt),
            kind="state",
        ),
        Check(
            "state: the page she was taught on is what she bound",
            any(LISTING_URL in value for value in bound),
            rationale=f"bound {bound}",
            kind="state",
        ),
        Check(
            "state: it runs hourly (the cadence they asked for)",
            row is not None and row.collector_interval_seconds == 3600,
            rationale=f"interval {row and row.collector_interval_seconds}",
            kind="state",
        ),
        Check(
            "state: it stops tonight (the end condition they gave)",
            row is not None and row.expires_at is not None,
            kind="state",
        ),
        Check(
            "state: it will tell them when the price moves",
            row is not None and bool(row.notify),
            kind="state",
        ),
        Check(
            "state: she set it running instead of running it again (no browse this turn)",
            tool_not_called(db, "browse"),
            kind="state",
        ),
        Check(
            "reply: she says what will happen now, naming the cadence",
            any(token in reply.lower() for token in ("hour", "60 min")),
            kind="reply",
        ),
        # Advisory — the collection-management question, parked (code owner): does
        # the job land on the collection the round already wrote into, or on a new
        # one?  Visible every run, graded never, so the standing tendency to spread
        # across collections is measured here without this edge answering for it.
        Check(
            "state: applied onto the collection the round wrote into (not a new one)",
            reused,
            rationale=(
                None
                if reused
                else (
                    f"applied to {row.name if row else None}, "
                    f"created {[each.name for each in created]}"
                )
            ),
            scored=False,
            kind="state",
        ),
        Check(
            "calls: one collection_set call",
            sets == 1,
            rationale=f"{sets} calls" if sets != 1 else None,
            scored=False,
            kind="proc",
        ),
        Check(
            "calls: the machine landed in apply",
            _landed_state(db) == ConversationState.APPLY.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_learn_to_apply_instantiates_the_taught_skill(chat_eval: ChatEval) -> None:
    """learn → apply: parked on the offer the demonstrated round ended with, the
    user accepts and adds the job's terms.  She binds the taught skill onto the
    collection that round already wrote into — one `collection_set`, the page
    taken from the round rather than asked for again — and does NOT re-run the
    round to answer."""
    await chat_eval(
        case_id="transition-learn-to-apply",
        message=_APPLY_TURN,
        seed=_seed_demonstrated_round,
        seed_skills=[learn_to_apply_fixture_skill()],
        score=_score_learn_to_apply,
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )
