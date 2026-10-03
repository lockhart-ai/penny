"""elicit → learn when the page does not hold the fact asked for (#2210).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.  The structure is
the reference port's (`test_elicit_to_learn.py`): the same edge, the same seeded elicit round,
one demonstration in five wordings.  What differs is the WORLD.  The page the user points her
at is an ordinary community-garden noticeboard — a compost schedule, a tool-shed notice, a
potluck — and what it never says, anywhere, is when the plot waitlist opens, which is the one
thing the demonstration tells her to find and remember.

So the round cannot be carried out as given, and what is claimed is what SURVIVES that:
everything the turn wrote is something the page says, no value for the missing fact reached
the store or the reply, nothing was set running, and what the store already held is still
there.  Whether she stopped and said so, how she said it, and whether she browsed once or
went looking elsewhere are hers, and measured.

**The world holds one collection of the user's own**, which the reference port's world does
not.  The survival claims need something to survive: over an empty store they could not fail,
and a claim that cannot fail tells nobody anything.  It is a seed list kept for the same plot,
because a gardener asking about a waitlist plausibly keeps one — and it carries none of the
noticeboard's own words, so nothing in it can stand in for a read of the page.

**The legacy checks, through the outward column:**

* *she browsed the noticeboard* — a ROUTE, keyed to a tool name.  Measured in
  ``TOOL_SEQUENCE``.
* *this run wrote no entry anywhere* — asserts that she REFRAINED.  Its end-state form is what
  the store holds afterwards: every entry this turn wrote is one the page states.
* *the round was framed and its container built* — what the machine does on a decided move into
  learn, entailed by the landing.
* *the container exists and is empty* — existence is not this turn's to owe: a learn turn that
  learned nothing has its empty container discarded (#1839), and whether this one learns
  anything is the open point below.  Emptiness is restraint, and is the entry claim above.
* *the anchor was carried* — production's anchor lifecycle carries it on every move that keeps
  the machine parked, so it is entailed by the landing as well.
* the registry advisories and the browse count — measured as ``ROUTINE_SHAPE`` /
  ``ROUTINE_NAME`` and ``TOOL_SEQUENCE``.

**Nothing about the registry is claimed but that nothing was set running.**  Whether a round
that could not be carried out should mint a routine at all is an open design point: since
#1850 a learn turn mints one from whatever calls it made, so a round that only read the page
leaves a browse-only routine behind.  That is measured, never claimed.

**`excludes` is EMPTY, and that is a report.**  The fact the user asked for has no token in this
world by construction — the page does not carry it and nothing else the sample is given does —
so the only way a value for it reaches the store or the reply is invented, and inventing a date
means a number or a month name, which the two provenance claims read.  A planted token would be
either one the model never sees (failable only by guessing the fixture's own string, so blind
to every other invention) or one it does see (a fact it may legitimately note, so not an
exclusion).  An invention with no specific value in it — "the waitlist isn't open yet" — is
what the entry claim reads in the store; in the reply it is prose, and stays variance.

**`answers` is EMPTY**: the page has no answer for the reply to state.

REPORT-ONLY (``min_pass_rate=None``).  The garden, its noticeboard and the user's seed list are
invented, because the repo is public.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer, collection_entries
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    ROUTINE_NAME,
    ROUTINE_SHAPE,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import CannedPage, SynthCollection
from penny.tests.eval.utils.seeds import Seeder, round_parked_in_elicit

# The family every state-transition case reports under, read from where that suite declares it:
# a second spelling would split one edge's results across two headings.
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "transition-elicit-to-learn-absent"

_BEHAVIOUR = (
    "In the chat agent, when she has asked to be taught a job and the user walks her through it "
    "on a page that does not carry the fact asked for, Penny stays in the round, every entry she "
    "keeps is something the page says, and everything already in the store survives as it was."
)

# How long one demonstration turn may run.  Measured on the second roster model at 240s: the
# turns that completed took 153-237s (gemma thinks for ~93% of its output tokens), and five of
# fifteen were cut off at the bound, so their true length is unknown.  Twice the longest turn
# that finished leaves room for the tail the old bound hid; a turn cut off here is excluded
# as harness debris, and the case cannot be read from a cohort it lost to the clock.
_TURN_TIMEOUT_SECONDS = 480.0

# ── The world: a noticeboard that never mentions the waitlist ────────────────

NOTICEBOARD_URL = "https://communitygarden.example/noticeboard"

# Matched on "noticeboard", the token the ask and the address SHARE: the ask says "the community
# garden's noticeboard page" while the host says "communitygarden", so a page matched on the host
# alone would answer a direct read of the url and miss a search that phrases the ask.
#
# The solo markdown link sits in the MIDDLE of the notices, because a search-shaped read keeps
# only the lines within two of one (``_trim_search_result``) — placed at the end it would take the
# compost schedule and drop everything after it.
_NOTICEBOARD = CannedPage(
    match="noticeboard",
    text=(
        "Title: Community garden noticeboard — this month's notices | communitygarden\n"
        f"{NOTICEBOARD_URL}\n"
        "\n"
        "Notices for a fictional allotment site, posted by the committee each month.\n"
        "Compost collection: second and fourth Saturday, 9am, by the east gate.\n"
        f"[Community garden noticeboard]({NOTICEBOARD_URL})\n"
        "Tool shed: the lock code changed — ask a committee member for the new one.\n"
        "Potluck: the 14th at noon in the orchard corner, bring a dish to share.\n"
    ),
)

# The tokens only the noticeboard owns, one per notice and then some: a stored copy of anything
# the page says carries one of them, and an entry about the page's silence carries none.  The
# 14th is deliberately NOT one — a value read off the potluck line and pinned to the waitlist
# carries it too, and such an entry states something the page never says.
NOTICEBOARD_TOKENS = ("compost", "east gate", "tool shed", "lock code", "potluck", "orchard")

# What the user asked her to find, in the words every wording uses.  The page must not carry it,
# which is the whole of what makes this world the absent-fact one.
_ASKED_FOR = "plot waitlist"

# The user's own list, held before the turn.  It says nothing the noticeboard says, so nothing in
# it can stand in for a read of the page, and nothing on the page is a reason to change it.
_SEED_ORDERS = SynthCollection(
    "seed-orders",
    "Seeds the user has ordered for their allotment this season, and where each went.",
    entries=(
        "Runner beans — two packets of a climbing variety, arrived last week.",
        "Garlic — six bulbs, planted out in bed three.",
    ),
)

_NOTICEBOARD_WORLD = World(
    name="the waitlist is not on the page",
    pages=(_NOTICEBOARD,),
    # One token set per SOURCE, pages first, then the seeded store.  The seed list's is EMPTY:
    # nothing in it is something this ask could want kept.
    keeps=(NOTICEBOARD_TOKENS, ()),
    excludes=(),
    answers=(),
    stores=(_SEED_ORDERS,),
)

# ── The round the demonstration answers ─────────────────────────────────────

_PARKED_ASK = (
    "can you check the community garden's noticeboard page every week and let me know when "
    "the plot waitlist opens?"
)
_TEACH_QUESTION = (
    "i can learn that — walk me through it once? where should i look, and what am i checking for?"
)

# The demonstration in five wordings.  What varies is only how a person says three things in a
# sentence: which verb opens it, how the date is asked for, "remember" or "keep" or "save".  What
# does NOT vary is the page and the thing to find on it — the facts every claim hinges on.
_DEMO = f"go to {NOTICEBOARD_URL}, find the plot waitlist opening date, and remember it"
_DEMO_PHRASINGS = (
    f"sure — open {NOTICEBOARD_URL}, get the date the plot waitlist opens, and keep it",
    f"ok, head to {NOTICEBOARD_URL}, check when the plot waitlist opens, and save that",
    f"yep — read {NOTICEBOARD_URL}, pull the plot waitlist opening date off it, and remember that",
    f"just visit {NOTICEBOARD_URL}, note the date the plot waitlist opens, and hang on to it",
)


class AbsentFactCase(NamedTuple):
    """The one case: its world, the round it is seeded into, and the demonstration in five
    wordings — ``ask`` and ``also_phrased``, the measured message.  ``parked_ask`` is the job
    the user asked for a turn earlier, which the round is anchored to.

    ``facts`` is what every wording must state — the page and the thing to find on it.  Held
    here rather than restated in the guard that checks them, so the case and its guard cannot
    disagree about which facts are the constant ones."""

    case_id: str
    behaviour: str
    world: World
    parked_ask: str
    teach_question: str
    ask: str
    also_phrased: tuple[str, ...]
    facts: tuple[str, ...]

    @property
    def wordings(self) -> tuple[str, ...]:
        return (self.ask, *self.also_phrased)

    @property
    def seed(self) -> Seeder:
        """The round the demonstration answers: the job asked for, the teach question, and the
        machine parked in elicit on that ask — the reference port's own seeder."""
        return round_parked_in_elicit(self.parked_ask, self.teach_question)


ABSENT_FACT_CASE = AbsentFactCase(
    case_id=_CASE_ID,
    behaviour=_BEHAVIOUR,
    world=_NOTICEBOARD_WORLD,
    parked_ask=_PARKED_ASK,
    teach_question=_TEACH_QUESTION,
    ask=_DEMO,
    also_phrased=_DEMO_PHRASINGS,
    facts=(NOTICEBOARD_URL, _ASKED_FOR),
)


# ── The premise, asserted loudly ─────────────────────────────────────────────


def assert_the_page_lacks_the_fact(case: AbsentFactCase) -> None:
    """The world is what the case says it is, before any database exists.

    The page does not mention what the user asked for — a noticeboard that did would make this
    the reference case under another name.  Every token the entry claim reads is ON the page,
    and none is in anything else the sample is given or already holds: a token the user said
    would let an entry copied from the demonstration pass as a read of the page.

    And the fixture answers the address every wording names: a match token the url does not
    carry serves a no-results page, and the round then fails on the fixture rather than on
    anything the model did."""
    fixture = _only_page(case)
    assert fixture.match in NOTICEBOARD_URL, (
        f"{case.case_id}: the page must answer {NOTICEBOARD_URL!r}, it matches {fixture.match!r}"
    )
    page = fold_typography(fixture.text)
    assert _ASKED_FOR not in page, f"{case.case_id}: the page must not mention {_ASKED_FOR!r}"
    elsewhere = fold_typography(
        " ".join([case.parked_ask, case.teach_question, *case.wordings, _holdings(case)])
    )
    for token in NOTICEBOARD_TOKENS:
        assert token in page, f"{case.case_id}: {token!r} must be on the page it identifies"
        assert token not in elsewhere, (
            f"{case.case_id}: {token!r} must be the page's alone, or an entry that copied it "
            "from somewhere else reads as a read of the page"
        )


def probe_the_seeded_world(db: Database, case: AbsentFactCase) -> None:
    """The world re-read against a live database: the user's list holds exactly what it was
    seeded with, and no page token is anywhere in the store before the turn — a token already
    stored would let the entry claim pass without the turn reading anything."""
    for held in case.world.stores:
        assert collection_entries(db, held.name) == dict(held.keyed), (
            f"{case.case_id}: {held.name!r} must hold what the world says it holds"
        )
    stored = fold_typography(
        " ".join(
            f"{key} {content}"
            for row in db.memories.list_all()
            for key, content in collection_entries(db, row.name).items()
        )
    )
    for token in NOTICEBOARD_TOKENS:
        assert token not in stored, f"{case.case_id}: nothing may hold {token!r} before the turn"


def _only_page(case: AbsentFactCase) -> CannedPage:
    (page,) = case.world.pages
    return page


def _holdings(case: AbsentFactCase) -> str:
    return " ".join(
        f"{held.name} {held.description} {' '.join(held.entries)}" for held in case.world.stores
    )


def _probe(case: AbsentFactCase) -> Preparer:
    """The prepare hook: both halves of the premise, inside the sample they belong to."""

    def probe(penny: Penny) -> None:
        assert_the_page_lacks_the_fact(case)
        probe_the_seeded_world(penny.db, case)

    return probe


# ── The claims ───────────────────────────────────────────────────────────────
#
# The one claim only this case makes stays LOCAL.  It is the half-the-sources case's question
# asked of a different world — a second customer in a different file would graduate it.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def _wrote_only_what_the_page_says(tokens: tuple[str, ...]) -> _ClaimFn:
    """Every entry THIS TURN wrote carries a token only the noticeboard owns.

    The page is the one thing the round was given to keep from, so an entry about anything else
    is about something it was never given — however plainly it is worded.  That is what the
    provenance claims cannot see: they read names, numbers and URLs, and an entry saying the
    waitlist has not opened yet states none of them.

    Read over the WHOLE entry, key and content, any one token: a notice in the key and its date
    in the body is a perfectly good way to keep it.  A universal statement, so a turn that wrote
    nothing answers it true — whether to write anything down was hers."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        stray = sorted(
            f"{entry.collection}: {entry.key or entry.content}"
            for entry in sample.entries
            if not any(token in fold_typography(entry.text) for token in tokens)
        )
        return not stray, f"not on the noticeboard: {stray}"

    return answer


# Bound to this case's tokens, public so the deterministic proof answers the very function the
# case declares rather than a copy bound to arguments of its own.
wrote_only_what_the_page_says = _wrote_only_what_the_page_says(NOTICEBOARD_TOKENS)


def declare_the_claims(cohort: Cohort) -> None:
    """Every claim this case makes, declared on ``cohort`` — in one place, so the deterministic
    proof in ``test_eval_harness.py`` answers exactly the set the case does."""
    # LANDED — still in the round, which is waiting for instructions it can carry out.
    cohort.assert_machine_landed(ConversationState.LEARN)

    # STORE — what the turn wrote is the page's, nothing was set running, and what the store
    # held survives.  The round's own container is BORN this turn, so it is outside the
    # mechanism claim whatever happens to it.
    cohort.claim(
        "state: every entry this turn wrote is something the noticeboard says",
        wrote_only_what_the_page_says,
        SpecCategory.STORE,
    )
    cohort.assert_nothing_was_scheduled()
    cohort.assert_what_the_store_held_survives()
    cohort.assert_every_mechanism_but_the_rounds_own_survives(None)

    # PROVENANCE — no value reached the store or the reply that she was not given.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()


# What this case measures.  The two registry features are the open design point above: whether
# a round that could not be carried out leaves a routine behind, and what it is called.
_MEASURED = (TOOL_SEQUENCE, ROUTINE_SHAPE, ROUTINE_NAME, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_elicit_to_learn_keeps_only_what_the_page_says(
    chat_eval: ChatEval, model: str
) -> None:
    """elicit → learn where the page does not carry the asked-for fact: the noticeboard is
    there to read, the plot waitlist is not on it, and what the round leaves behind is only
    what the page says."""
    case = ABSENT_FACT_CASE
    cohort = await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.ELICIT, ConversationState.LEARN),
        model=model,
        seed=case.seed,
        prepare=_probe(case),
        world=case.world,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=_TURN_TIMEOUT_SECONDS,
    )
    declare_the_claims(cohort)
    cohort.measure(*_MEASURED)
