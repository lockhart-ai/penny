"""idle → request: the routine is known, and the ask is one value short (#2005, tranche 2).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

The world holds five finished journeys, so a routine that covers the ask already exists — and
the ask leaves out exactly one value that routine's interface requires.  The turn's whole job
is to recognise the routine and ask for the missing piece, standing nothing up: a container's
name is derived from the routine plus EVERY value it is pointed at, so a job short of one has
no name yet and anything built here would be built under a name nothing could derive again.

**The survivor, and on what basis: MEASURED RATE, with dominance agreeing.**  The edge's five
variants have real per-variant numbers.  Over the six suite runs that carried them, `-listing`
reports mean 0.93 / 0.98 / 1.00 / 1.00 / 0.96 / 1.00 with every sample fully passing on three
of them — the best record of the five; the base timetable ask never exceeds 4 of 5 fully
passing, and `-held-binding` is the worst in the suite (0 of 5 fully passing on six of its ten
recorded runs).  Dominance agrees: `-listing` is the ask that most LOOKS complete enough to act
on — the cadence and the end date are both given and only the page is missing — so it is the
world most able to produce the failure this behaviour is about, a job stood up on a guessed
binding.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: the base `-timetable` (the sailing to watch for is settled and the page is not, so the
reply must ask for one thing and work from the other), `-count` ("the same way" points at a
routine, never at a page), `-bakery` (the new bakery is a thing the user knows and the history
does not, so nothing but asking can supply it), `-held-binding` (the page IS given and what to
look for on it is not, so asking again for what they just said is the failure — on a world
seeded with three journeys rather than five, since two routines that ask for a URL and nothing
else would be COMPLETE against that ask).  Every `_IdleRequestCase` stays in
`transition_world.py`, so any of them can come back deliberately.

**Both halves of the behaviour have a structural reading, and both are claimed.**  *Names the
routine she recognises* is the landed move's own ``skill_name``; *asks for the one value still
missing* is the round's recorded shortfall, whose ``missing`` names the parameter by its
declared key.  Neither is entailed by the landing: request is reachable by two doors (the
binder's shortfall redirect and a classifier draw), and only one of them guarantees a shortfall
at all — so a move landing in request carrying nothing to ask about is a real, nameable
failure, and so is one that named the wrong parameter.

**Five source checks did not port** (the outward column):

* ``Check("state: she asked instead of going to look (no browse this turn)")`` — a ROUTE, keyed
  to a tool NAME.  Its end-state form is *nothing was written*, which is claimed; the browse is
  measured in section B.
* ``Check("state: she configured nothing", tool_not_called(db, _SET_TOOL))`` — the same, and its
  end-state form is *no mechanism was created* / *no running mechanism was changed*.
* ``_request_anchor_check`` — PRODUCTION ALREADY VALIDATES IT: ``_next_anchor`` stamps the
  instigating message on every move into a parked state FROM idle, so on a sample that landed
  in request the claim is entailed by the landing.
* ``_asks_for_what_is_missing_check`` — a PHRASING match on a vocabulary somebody guessed in
  advance.  Its structural form is the shortfall claim above, which reads what the round is
  waiting on rather than what the sentence happened to call it.
* ``_does_not_re_ask_check`` — the same, and n/a on this survivor anyway (its ask settles
  nothing, so there is nothing that could be asked for twice).

**One more is absent by ENTAILMENT**, and is worth naming so the set reads as closed: *nothing
was registered*.  Run-end extraction fires in ``learn`` and nowhere else, and the only other
thing that touches the registry (``abandon_round_skill``) runs on an IDLE landing — so no
sample can fail it without also failing ``assert_machine_landed``.

**And two the inward column added**: PROVENANCE, of both kinds.  The source case made no claim
of either, so a reply that invented an address to ask about — or a sample that filed a fact
nobody's page mentions — passed every check it carried.

**`keeps` and `answers` are both EMPTY, and each is a report.**  The turn is not asked to write
anything down, so a keeps set would state a contract the ask never made — and the case claims
the opposite.  The ask requests no VALUE, it requests that a job be set up, so a correct reply
owes no token; requiring one would fail a correct run for something nobody asked for.

REPORT-ONLY (``min_pass_rate=None``).  Every page, url and job is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database.skills import slug_skill_name
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _SHORT_LISTING,
    _UNKNOWN_SPACES,
    _IdleRequestCase,
    assert_composed_world,
    assert_the_ask_falls_one_short,
    assert_the_registry_holds,
    seed_composed_world,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# Read off the survivor rather than spelled again: the fixture carries the edge's id since the
# collapse (#2005), and two spellings of one id are two things free to drift.
_CASE_ID = _SHORT_LISTING.case_id

_BEHAVIOUR = (
    "In the chat agent, when a routine she already has nearly covers what the user asks for "
    "and the ask leaves out one value that routine needs, Penny parks the round on that "
    "routine and records the missing value as what she is waiting on — building nothing, "
    "writing nothing, and leaving the jobs already running alone."
)

# The survivor's own ask, in five wordings.  What varies is only how a person says it —
# "another listing" or "a second listing", "every couple hours" or "every two hours", "moves"
# or "changes" or "shifts".  What does NOT vary is the routine the ask is covered by, the terms
# it settles, or the one thing it leaves out: no wording names a page, which is what makes the
# shortfall the same shortfall on every arm.
_SHORT_ASK_PHRASINGS = (
    "there's another listing i want tracked — check its price every couple of hours until "
    "sunday and let me know if it changes",
    "i've got a second listing to keep an eye on — its price every couple of hours through to "
    "sunday, and tell me when it moves",
    "can you track another listing's price for me? every couple of hours until sunday, and "
    "message me if it shifts",
    "one more listing to watch — look at its price every couple of hours until sunday and tell "
    "me if it moves",
)

# The ground every arm is answered against: every space this world does NOT know, installed as
# a live temptation.  A request turn's tempting wrong move is to go and FIND the missing value
# instead of asking for it, so the pages a plausible search would reach are all present — what
# they buy is that a turn which does look up gets a real page back rather than failing on a
# thin fixture, which would make "she asked instead" true for the wrong reason.
#
# ``keeps``, ``excludes`` and ``answers`` are all empty; the module docstring says which of
# those is a report and why.
_UNKNOWN_LISTING = World(
    name=_CASE_ID,
    pages=tuple(_UNKNOWN_SPACES),
    keeps=(),
    excludes=(),
)

# The routine this ask is covered by, as the registry holds it — what the decision has to pick
# out of five real routines of the same kind.
_COVERING_ROUTINE = slug_skill_name(_SHORT_LISTING.skill.name)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: a
# request turn mints no routine, so on a correct cohort both read the world's own seeded five
# on every sample and pool to a serene 0.000 that is a reading of the FIXTURE rather than of
# anything the turn did.  A sample that DID mint one is caught by the registry claim, where it
# is a miss rather than a variance rise.
#
# ``TOOL_SEQUENCE`` reads "no call" on a correct sample here, so a perfectly behaved cohort
# makes it blind and the report renders it red.  Measured anyway, because the divergence it
# exists to catch is this case's named failure — a sample that went and looked for the missing
# page, or stood the job up on a guess, reads differently and the blindness lifts.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


def _probe_short_ask(case: _IdleRequestCase) -> Preparer:
    """The prepare hook: the shared world's own claims, the registry one that is only true once
    the runner has laid the fixture skills down, and this case's own two — that the ask really
    does fall one value short of the routine it names, and that no wording supplies it."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db, case.journeys)
        assert_the_registry_holds(penny.db, case.journeys)
        assert_the_ask_falls_one_short(penny.db, case)
        assert_no_wording_names_a_page(case)

    return probe


def assert_no_wording_names_a_page(case: _IdleRequestCase) -> None:
    """No arm's wording carries an address.

    ``assert_the_ask_falls_one_short`` reads the routine's declared set off the registry and
    says the CASE accounts for it; this says the five WORDINGS do — and it is the half the
    cohort adds, since the facts are held constant across the arms precisely because the
    assertions hinge on them.  A wording that let an address slip in would be an ask the binder
    can COMPLETE, so its sample would land in apply and fail every claim for a reason that has
    nothing to do with the behaviour.

    It is written for THIS case's shortfall, which is a page, and says so: what a wording must
    not supply is whatever that case's ``missing`` names, and an address is the one shape a
    fixture can read for."""
    assert case.missing == ("url",), (
        f"{case.case_id}: this premise reads for an address, and the ask is short of "
        f"{list(case.missing)}"
    )
    for wording in (case.ask, *_SHORT_ASK_PHRASINGS):
        assert "http" not in wording, f"{case.case_id}: this wording names a page — {wording!r}"


# ── The claim this edge's own contract adds ───────────────────────────────────
#
# It reads the LANDED MOVE — the machine's own row — rather than the store, because what a
# request turn produces is round state and not a mechanism: it deliberately builds nothing.
# It stays LOCAL because it is parametrised by this case's own shortfall and no other case has
# asked for it; its sibling — the move naming the routine — GRADUATED in the same change, the
# idle → apply case being its second customer.


def _waiting_on_exactly_what_is_missing(sample: SampleObservation, _world: World) -> Answer:
    """The round records EXACTLY the parameters the ask left out, by their declared names.

    Two failures, one sentence.  A move that landed in request carrying no shortfall has parked
    the round on nothing to ask about — reachable, since a classifier-drawn request whose
    binding came back complete records none — and a move recording a parameter the ask already
    settled is asking again for something it was given."""
    missing = list(_SHORT_LISTING.missing)
    return sample.awaiting == missing, f"waiting on {sample.awaiting}, the ask leaves out {missing}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_idle_to_request_asks_for_the_listing(chat_eval: ChatEval, model: str) -> None:
    """idle → request on the price watcher: the cadence and the end date are both given and the
    listing itself is not, which is the ask that most looks complete enough to act on.  The
    turn parks on the routine it recognises, records the page as what it is waiting for, and
    builds nothing on a half-settled interface."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_composed_world(_SHORT_LISTING.journeys),
        seed_skills=[journey.round.skill for journey in _SHORT_LISTING.journeys],
        prepare=_probe_short_ask(_SHORT_LISTING),
        world=_UNKNOWN_LISTING,
        ask=_SHORT_LISTING.ask,
        also_phrased=_SHORT_ASK_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED — where the turn went, and the two facts the move itself carries about the round
    # it opened.  All three are read off the machine's walk.
    cohort.assert_machine_landed(ConversationState.REQUEST)
    cohort.assert_the_move_named_the_routine(_COVERING_ROUTINE)
    cohort.claim(
        "state: the round is waiting on exactly the value the ask left out",
        _waiting_on_exactly_what_is_missing,
        SpecCategory.LANDED,
    )

    # STORE — three negatives, one per way of acting on an interface that is not settled yet.
    cohort.assert_nothing_was_written()
    cohort.assert_no_mechanism_was_created()
    cohort.assert_no_running_mechanism_was_changed()

    # PROVENANCE — the half the source case had none of.  The store claim answers over an empty
    # set on a correct sample and names the invention on one that wrote something; the reply
    # claim is live throughout, since an ask for a page the user never gave is exactly where an
    # address gets invented.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
