"""Chat in ELICIT: the ask lands cold, and nothing is enacted.

Five cold asks for something that keeps running, none of them covered by a routine the assistant
already has. The turn's whole job is to ask to be taught: no page is fetched, no value is
written, no job is stood up. What each case varies is the shape of the ask -- a page named or
withheld, a digest, a threshold, an urgency -- and none of those is a licence to go and look
instead of asking.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import (
    ConversationState,
)
from penny.database import Database

# The SHIPPED container derivation, used as itself: a seeded round has to run into the
# container production would have built for it, and a fixture spelling that name out would
# be a second copy of the naming scheme, free to drift from the one jobs are identified by.
# The production draw-application, used as itself: a fixture skill has to be the SHAPE
# run-end extraction really produces, and re-implementing that mapping here would be a
# fixture that drifts from the pipeline it stands in for.  Both halves of the #1824
# split are applied by their own production function — ``_apply_leaf_labels`` for the
# labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's signature.
# ``attachment_names`` is the registry policy for what a routine can be attached to, read
# for the same reason: the scorer asks whether a learned routine HAS a destination, and
# that is the question extraction already answers when it decides which leaves to mark.
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    asked_for_page_structure,
    chat_run_tool_sequences,
    new_collections,
    routing_clean,
)

# The agreed breadth for "the page the routine is pointed at", READ from where the framer
# suite declares it rather than restated here: what a page parameter may reasonably be
# called is one code-owner-agreed vocabulary, and two copies would drift into two
# contracts (the same rule ``ENACTING_TOOLS`` is read under).
# The listing this script is built on, and the enacting-tool set the elicitation
# contract IS — the calls that would mean she acted before being taught.  Both are read
# from the suite's shared fixtures rather than restated here: the passing-mention guard
# in ``test_chat_memory_stories.py`` asks the same question of a turn, and two copies of
# one policy are two contracts free to drift.
from penny.tests.eval.utils.fixtures import (
    AURORA_LISTING_499,
    ENACTING_TOOLS,
)
from penny.tests.eval.utils.transition_ledger import (
    _FAMILY,
    _entries_written_by_this_run,
    _landed_state,
    _pages_fetched,
    _written_texts,
)
from penny.tests.eval.utils.transition_world import (
    _BAKERY_SPECIALS,
    _COLONY_COUNT,
    _FERRY_TIMETABLE,
    _IDLE_ASK,
    _IDLE_ASK_DIGEST,
    _IDLE_ASK_NO_URL,
    _IDLE_ASK_THRESHOLD,
    _IDLE_ASK_URGENCY,
    _NEW_ARRIVALS,
)

# The production tool-result framer, used as itself: a seeded ledger's tool turns have to
# read the way the loop really writes them, and a hand-written frame is a second copy of a
# format the model is shown every turn.
# The schedule's own render + grammar tokens, read from where the tool declares them: a
# stored rule renders back AS the copyable ``schedule`` input (#1857), so the advisory shows
# what she committed to in the form it was set, and the line/tag literals a rule is written
# with are that module's to define — a restated copy here would be a second contract.
# ``parse_schedule`` + ``render_reinstantiation_echo`` are read for the same reason on the
# seeding side: a seeded apply turn stores the rule the tool would have stored and echoes
# back what the tool would have echoed.

pytestmark = pytest.mark.eval


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
        tool for run in chat_run_tool_sequences(db) for tool in run if tool in ENACTING_TOOLS
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
