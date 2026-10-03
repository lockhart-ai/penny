"""request → elicit: the routine she named was the wrong one, and the task is still wanted
(#2215).

The contract is `docs/eval-case-design.md`.

The machine is parked in request: a routine she already has looked like it covered the ask, so
she said what she would do and asked for the one detail it was short of.  The user answers that
the routine is not what they meant — and that they still want the job done.  The proposal is
off the table and the task is not, so no routine covers it any more and the round becomes the
one where she asks to be taught: it moves to elicit, still the same round, on the same ask.

**The world, and on what basis: the HELD-BINDING round (`_SUPPLIED_PIER`),
`transition-request-to-idle`'s own.**  Three things choose it over the other parked rounds.

* *The rejection is an answer to what she actually said.*  The ask was to be told when the
  pier adds a sailing; her reply read that as a routine that watches for a phrase and asked
  which phrase.  "That's not what i meant" is the natural reply to exactly that question.  On
  the other parked rounds her reply says the ask back and asks only for its page, so a
  rejection there would contradict the user's own ask.
* *The round holds a settled half as well as an open one.*  The page came with the ask and is
  recorded on the round, so the turn enters elicit with an address in hand and its page
  installed and reachable: a turn that carried on with the job instead of asking to be taught
  has somewhere to go, and section B can see it go there.
* *It is the world its sibling bail is measured against*, so the two ways out of a parked
  request that do not supply the detail — call it off, or say the routine was wrong — are read
  off one round.

The seeder is that round's own (`seed_parked_in_request`): the journeys walked to their end,
then the short ask, the draw that named the routine, her reply threaded to the ask, and the
move — parked in request, anchored to the ask, naming the routine, carrying the binding the
round is waiting on and no framing.  That is what production records, and the classifier is
shown the round's `## The details this task is waiting on` section from it.

**The wordings are the classifier case's own five.**
`classifier-elicits-when-the-named-routine-was-wrong` draws this decision in isolation from
these same messages, and this case drives the whole turn they open.  Read from where that case
declares them rather than restated, so a miss here is attributable: where the draw holds there,
a turn that lands elsewhere here is the turn's.  Every wording carries both halves the edge
turns on — the routine was the wrong one, and the task is still wanted — and each says them a
different way; none supplies the missing detail (that is request → apply) and none calls the
task off (that is request → idle).  The arms are held to that in `make check`, including that
none carries the transition condition's own phrase.

**What is claimed.**  Where the machine landed, and what SURVIVES the turn: the three jobs this
user already has running, and everything their collections held.  A turn told the routine was
wrong has the registry and three live jobs of the same kind in front of it, so changing one of
them is a reachable way to get it wrong, and it is read off the ledger rather than a tool name.

**Four obvious-looking claims are deliberately absent**, so a thin set reads as closed rather
than as a checklist nobody ran:

* *the round is no longer waiting on the detail* — PRODUCTION ALREADY VALIDATES IT.
  `_next_shortfall` keeps a binding only on a move that lands in request, so on a sample that
  landed in elicit the claim is entailed by the landing.
* *the round is still anchored to the ask* — the same: `_next_anchor` keeps the anchor of a
  machine that is already parked.
* *the move named no routine* — GENERATED rather than drawn.  Elicit is not a skill-gated
  state, so the draw carries no `SKILL:` line by construction.
* *the registry holds exactly the routines it already had* — TRUE BY CONSTRUCTION on every
  landing this turn can reach.  Run-end extraction fires in `learn`, which request has no edge
  to, and a round parked in request minted no routine for a bail to take back.

**And no claim says what the turn refrained from.**  Whether it opened the pier's page, stood a
job up anyway or wrote something down is the model's call on what it was shown: measured as the
tool sequence and the entries stored, never claimed.

**PROVENANCE is claimed in both halves.**  The reply half is live throughout: a turn asking to
be taught a pier watch is exactly where a sailing time nobody's page was read for gets stated.

**`keeps`, `excludes` and `answers` are all EMPTY, and each is a report.**  The turn is not
asked to write anything down; the message rules nothing out; and it asks for no value, so a
correct reply owes no token.

REPORT-ONLY (``min_pass_rate=None``).  Every page, url and job is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.penny import Penny
from penny.tests.eval.classifier.test_state_classifier import WRONG_ROUTINE_ARMS
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer, collection_entries
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _SUPPLIED_PIER,
    _SUPPLIED_SPACES,
    _RequestApplyCase,
    assert_parked_in_request_world,
    assert_the_registry_holds,
    seed_parked_in_request,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

REJECTED_ROUTINE_CASE_ID = "transition-request-to-elicit"

_BEHAVIOUR = (
    "In the chat agent, when a round is parked in request on a routine she named and the user "
    "says that routine is the wrong one but still wants the task done, Penny moves the round "
    "to elicit, and every job already running and everything the store already held survive "
    "unchanged."
)

# The parked round the rejection arrives into — its ask, the routine she named, what the ask
# settled and what it left out, and the journeys behind it.  Read rather than restated, so this
# case and the bail measured against the same round cannot describe it differently.
REJECTED_ROUTINE_ROUND: _RequestApplyCase = _SUPPLIED_PIER

# One message in five wordings: the rejection, with the first as the ask and the rest as its
# other phrasings.  Named at module level so the deterministic pin in ``make check`` holds the
# cohort's arithmetic against the same tuple the case drives.
REJECTED_ROUTINE_ARMS = WRONG_ROUTINE_ARMS

# The ground every arm is answered against: every space this history's asks name, the pier's
# own page among them, installed and reachable — so a turn that DOES go and look gets a real
# page back rather than failing on a thin fixture.  ``keeps``, ``excludes`` and ``answers`` are
# empty; the module docstring says why.
_PARKED_PIER = World(
    name=REJECTED_ROUTINE_CASE_ID,
    pages=tuple(_SUPPLIED_SPACES),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: no
# landing this turn can reach mints a routine, so both would read the world's own seeded three
# on every sample — a reading of the FIXTURE rather than of the turn.
#
# ``TOOL_SEQUENCE`` reads "no call" on a sample that only asked to be taught, so a cohort that
# all did so makes it blind and the report says so in red.  Measured anyway, because the
# divergence it exists to catch is this edge's own — a sample that went on with the job reads
# differently from every other one, and the blindness lifts the moment it does.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ── The probe: the round really is parked on a named routine, over a world with something in it ─


def _probe_parked_round(case: _RequestApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the registry one that is only true once the
    runner has laid the fixture skills down, and the premise this case's two survival claims
    stand on."""

    def probe(penny: Penny) -> None:
        assert_parked_in_request_world(penny.db, case)
        assert_the_registry_holds(penny.db, case.parked.journeys)
        assert_every_job_holds_what_it_gathered(penny.db, case)

    return probe


def assert_every_job_holds_what_it_gathered(db: Database, case: _RequestApplyCase) -> None:
    """Every journey behind the parked round left its container holding what it demonstrated.

    The premise of *everything the store already held is still there*: on a store holding
    nothing that claim is true whatever the turn does, so the entries it reads are asserted
    present before the turn runs.  That the jobs themselves are live is the seeder's own probe
    (``assert_parked_in_request_world``)."""
    for journey in case.parked.journeys:
        container = journey.round.framing.container
        assert collection_entries(db, container), (
            f"{REJECTED_ROUTINE_CASE_ID}: {container!r} must hold what its round demonstrated"
        )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_request_to_elicit_moves_a_rejected_routine_to_being_taught(
    chat_eval: ChatEval, model: str
) -> None:
    """request → elicit: parked having named the pier-timetable routine and asked which sailing
    to watch for, the user says that is the wrong routine and they still want the job.  The
    round moves to elicit, and the three jobs already running and what they hold survive it."""
    cohort = await chat_eval(
        case_id=REJECTED_ROUTINE_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.REQUEST, ConversationState.ELICIT),
        model=model,
        seed=seed_parked_in_request(REJECTED_ROUTINE_ROUND),
        seed_skills=[journey.round.skill for journey in REJECTED_ROUTINE_ROUND.parked.journeys],
        prepare=_probe_parked_round(REJECTED_ROUTINE_ROUND),
        world=_PARKED_PIER,
        ask=REJECTED_ROUTINE_ARMS[0],
        also_phrased=REJECTED_ROUTINE_ARMS[1:],
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.ELICIT)

    # STORE — what SURVIVES a turn that was told the routine was wrong: everything the store
    # already held, and the jobs already running.  Whether the turn acted anyway is the model's
    # call and is measured below, never claimed.
    cohort.assert_what_the_store_held_survives()
    cohort.assert_the_running_mechanisms_survive()

    # PROVENANCE — both halves.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
