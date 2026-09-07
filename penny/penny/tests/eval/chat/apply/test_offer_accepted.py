"""learn → apply: the offer accepted, the taught round set running (#2005, tranche 3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

The round was demonstrated and the routine minted, and Penny closed it by offering to keep the
job up.  The user says yes and adds the job's terms in the same breath.  The turn stands the
job up ON THE ROUND'S OWN CONTAINER — the one the round built when it was framed — on the
cadence, the end and the telling clause the acceptance gives, without re-running the round to
answer and without touching anything else.

**The survivor, and on what basis: MEASURED RATE.**  The edge's five variants have real
per-variant numbers over the thirteen suite runs that carried all five.  Discounting the one
run where every case in the suite collapsed together (`0.20` on all five, an infrastructure
failure rather than a behaviour), the base acceptance reports the best ALL-PASS record of the
five — 0.700 of samples fully passing, and the only variant never below 2 of 5 fully passing,
where `-threshold` falls to 0 of 5 once and 1 of 5 twice, `-no-url` to 0 of 5 on five of its
twelve, and `-digest` to 0 of 5 on five.  Dominance agrees rather than competing: this is the
only acceptance of the five that gives ALL THREE terms — a cadence, an end condition and a
telling clause — so it is the one world where every terms claim below has something to fail
on.  `-threshold` leads on the MEAN (0.964 against 0.929) and on its floor, which is recorded
here as a finding rather than acted on: the two are a fifth of a claim apart on a fifteen-check
case, and #2005 picks the variant that already passes rather than the one with the better
average.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: `-no-url` (a routine asking for TWO things, both supplied by the round rather than by
the acceptance, on a cadence stated as a time of day so the rule must name an hour), `-digest`
(a plain daily cadence with NO end condition, so an expiry set here is one nobody asked for),
`-threshold` (the longest cadence in the set, on a job carrying the scheme-less address the
user typed), `-urgency` (a two-hourly cadence with an end condition stated in words the model
has to work out).  Every `_ApplyCase` fixture stays in `transition_world.py`, so any of them
can come back deliberately — and two are load-bearing there already, since the round this case
seeds is also what the learn → idle bail is measured against.

**What is claimed is the TERMS, and nothing else about the job.**  Since #1869 a turn that
configures a framed round is handed the container, the routine and the bound values
framework-side off the round's own `RoundFraming`, and the program is rendered from it
unconditionally — so *the job landed on the round's container*, *the intended routine was
bound*, *the program was rendered* and *the routine is pointed at what the round settled* are
four readings of one framing and PRODUCTION VALIDATES ALL FOUR.  A claim over any of them would
run at the rate the mechanism does while appearing to measure the turn.  What is left for the
model to choose is exactly what the three terms claims read: how often it fires, whether it
says anything, and whether it stops.

That the job landed on the RIGHT row is still answered, structurally rather than by a claim of
its own: the terms are read off the round's container BY NAME, so a turn that configured some
other row leaves this one unscheduled and fails all three, and a turn that built a row of its
own fails `assert_no_mechanism_was_created` as well.

**Six source checks did not port** (the outward column):

* ``Check("state: she set the job up with collection_set", tool_was_called(...))`` — a ROUTE,
  keyed to a tool NAME.  Many routes reach one end state and a skill is an arbitrary tool
  sequence, so its end-state form is the terms on the row, which catches a job stood up however
  it was reached.
* ``Check("state: she set it running instead of running it again (no browse this turn)",
  tool_not_called(...))`` — the same, one verb over.  Its end-state form is
  ``assert_nothing_was_written``: a turn that went and read the listing again and kept what it
  found leaves an entry behind.  The browse itself is measured in section B.
* ``_container_check`` · ``_skill_binding_check`` · ``Check("state: the skill's program was
  rendered into it")`` · ``_bound_parameters_check`` — PRODUCTION ALREADY VALIDATES THEM, per
  the paragraph above.  Their own source comments say so ("every one of these is a CERTAINTY
  since #1869"), which is exactly the reason they do not belong on the assertion side.
* ``_apply_anchor_check`` — PRODUCTION ALREADY VALIDATES IT.  ``_next_anchor`` keeps the anchor
  of a machine that is already parked, so a move out of learn carries the round's ask by
  construction and the claim is entailed by the landing.
* ``_decoy_check`` — *the decoy was not applied*.  ENTAILED by ``assert_no_mechanism_was_created``:
  the decoy is a routine this world never stood up, so it has no container of its own to be
  configured, and applying it can only mint one.
* ``Check("reply: she says what will happen now, naming the cadence", any(token in reply ...))``
  — a PHRASING match on a vocabulary somebody guessed in advance, which is the thing this
  design exists to abolish.  Whether the reply describes what actually landed is a real
  question and is not answerable from prose.

**And the inward column added PROVENANCE**, which there was nothing to copy: the source case
made no claim of it, so a sample that confirmed an hour or a price nobody gave — which is
exactly what a turn announcing a job it just set up is placed to do — passed every check it
carried.  Only its REPLY half is made here; the store half is entailed by
``assert_nothing_was_written`` and is stated empty with that reason at the claim site.

**One more claim is absent by ENTAILMENT, and this world is why.**
``assert_only_the_rounds_own_mechanism_changed`` reads *nothing but the round's own container
was touched* — but ``seed_learned_round`` lays down exactly ONE collection, the round's own, and
the decoy routine has none.  With no other mechanism in the world, and none created (the claim
above), there is nothing the sentence could find.  It is the right claim on a world with jobs
already running; whether this edge should be reseeded against one is the code owner's call and
is left alone here.

**`keeps`, `excludes` and `answers` are all EMPTY, and each is a report.**  The turn sets a job
to run LATER: it reads nothing and keeps nothing, so a keeps set would state a contract the
acceptance never made and this case claims the opposite of it.  The ask requests a job rather
than a value, so a correct reply owes no token.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.  Every page, url and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import slug_skill_name
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer, collection_entries
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _AURORA_APPLY,
    _DECOY_SKILL,
    _ApplyCase,
    _assert_parked_on_the_ask,
    assert_round_cites_its_run,
    assert_round_is_framed,
    assert_seeded_ledger,
    seed_learned_round,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "transition-learn-to-apply"

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The case id is a filename; this is the contract.
_BEHAVIOUR = (
    "In the chat agent, when a demonstrated round has been taught and the user accepts the "
    "offer to keep it running and supplies the job's terms, Penny stands the job up on the "
    "round's own container, on the cadence and the end the acceptance gave and telling them "
    "when it moves — without re-running the round to answer and without building a second "
    "job beside it."
)

# The survivor: the acceptance that continues the reference round, taking the offer up with a
# cadence, an end condition and a telling clause.
#
# Reference reply:
#   done — i'll check the listing every hour until 10pm tonight and message you if
#   the price moves.
_ACCEPTANCE = _AURORA_APPLY.acceptance

# Four more wordings of that same acceptance.  What varies is only how a person says yes —
# which word opens it, "every hour" or "hourly", "changes" or "moves", where the telling clause
# sits.  What does NOT vary is the cadence, the end condition or the telling clause: the case
# names all three, and it can only do that because they are constant across the arms.
_ACCEPTANCE_PHRASINGS = (
    "yes please — run that every hour until 10pm tonight, and let me know if the price changes",
    "great, go ahead — check it every hour until 10pm tonight and tell me when it changes",
    "sounds good — do it hourly until 10pm tonight and message me if it moves",
    "yeah do that — every hour until 10pm tonight, and tell me if the price changes",
)

# The tokens every wording has to carry for this case's terms claims to be answerable at all:
# the period the job fires on, and the end the ask gives it.  Matched on the part the wordings
# SHARE — "hourly" and "every hour" share `hour` — since what the probe is checking is that no
# arm dropped a term, not how the arm said it.
_CADENCE_TOKEN = "hour"
_END_TOKEN = "10pm"

# The round's own container — what the acceptance configures.  Read off the round's framing,
# never spelled out, because the name is `derive_collection_name(routine, bound values)` and a
# second copy of that scheme here would be free to drift from the one jobs are identified by.
_ROUND_CONTAINER = _AURORA_APPLY.framing.container

# The ground every arm is answered against: the page the round already read, installed as a
# LIVE temptation, so a turn that goes back to it gets a real page rather than failing on a
# thin fixture — which is what makes "she set it running instead of running it again" a real
# reading rather than an artefact of a page that was not there.
#
# ``keeps``, ``excludes`` and ``answers`` are all empty; the module docstring says which of
# those is a report and why.
_TAUGHT_LISTING = World(
    name=_CASE_ID,
    pages=(_AURORA_APPLY.prior.page,),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: an
# acceptance mints no routine, so on a correct cohort both read the round's own seeded two on
# every sample — a reading of the FIXTURE rather than of the turn.  A sample that DID mint one
# is a registry change this world would show as a second routine, and the reply spread is where
# a re-run of the round announces itself.
#
# ``JOB_TERMS`` is the point of the edge: the framing supplies the container, the routine and
# the values, so the rule the model wrote, whether the job speaks and whether it stops are the
# whole of what this turn chose.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, TRANSITIONS, REPLY_SPREAD)


# ── The probe: the world really is a taught round waiting on an answer ────────


def _probe_seeded_world(case: _ApplyCase) -> Preparer:
    """The prepare hook, run once the world is WHOLE.

    It is a hook rather than part of the seeder because the runner seeds the fixture skills and
    installs the page AFTER the case's own seed — so "exactly two routines" is only true here.
    A seed that has drifted from the state the preceding beat is measured against makes this
    case a turn answered against a world nothing produces, so it fails HERE rather than as a
    puzzling number after a paid run."""

    def probe(penny: Penny) -> None:
        _assert_parked_on_the_ask(penny.db, case)
        assert_round_is_framed(penny.db, case)
        assert_seeded_ledger(penny.db, case)
        assert_round_cites_its_run(penny.db, case)
        assert_the_registry_holds_the_round_and_the_decoy(penny.db, case)
        assert_every_wording_gives_the_terms()

    return probe


def assert_the_registry_holds_the_round_and_the_decoy(db: Database, case: _ApplyCase) -> None:
    """Two routines and only two — the round's own and the decoy — the demonstrated fact in the
    container the round wrote it to, and nothing instantiated anywhere.

    The last of those is the case's premise read out loud: the container arrives INERT, so "the
    job runs on these terms" is a claim about this turn and not a row that arrived configured.
    """
    taught = sorted(skill.name for skill in db.skills.list_all())
    expected = sorted(slug_skill_name(draft.name) for draft in (case.skill, _DECOY_SKILL))
    assert taught == expected, f"{case.case_id}: the registry must hold {expected}, got {taught}"
    stored = collection_entries(db, case.demonstrated.collection)
    assert stored.get(case.demonstrated.entry_key) == case.demonstrated.entry_value, (
        f"{case.case_id}: the demonstrated fact must be in the round's container, got {stored}"
    )
    instantiated = [row.name for row in db.memories.list_all() if row.skill_name is not None]
    assert not instantiated, f"{case.case_id}: nothing is instantiated yet, found {instantiated}"


def assert_every_wording_gives_the_terms() -> None:
    """Every arm's wording supplies the terms this case's claims read.

    The facts are held constant across a cohort's arms because the assertions hinge on them,
    and here two of them do: a wording that dropped the period would leave the cadence claim
    asking for an hour nobody requested, and one that dropped the end condition would fail the
    expiry claim for a term the ask never gave — in both cases a miss that has nothing to do
    with the behaviour."""
    for wording in (_ACCEPTANCE, *_ACCEPTANCE_PHRASINGS):
        missing = [token for token in (_CADENCE_TOKEN, _END_TOKEN) if token not in wording]
        assert not missing, f"{_CASE_ID}: this wording gives none of {missing} — {wording!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_learn_to_apply_stands_the_taught_round_up_on_its_own_container(
    chat_eval: ChatEval, model: str
) -> None:
    """learn → apply: parked on the offer the demonstrated round closed with, the user accepts
    and adds the job's terms.  The job stands up on the container that round built — the
    routine and the page it watches come off the round rather than being worked out again — on
    the hours and the end the acceptance gives, and she does NOT re-run the round to answer."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_learned_round(_AURORA_APPLY),
        seed_skills=[_AURORA_APPLY.skill, _DECOY_SKILL],
        prepare=_probe_seeded_world(_AURORA_APPLY),
        world=_TAUGHT_LISTING,
        ask=_ACCEPTANCE,
        also_phrased=_ACCEPTANCE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.APPLY)

    # STORE — the terms the acceptance gave, on the round's own container, and nothing else
    # created or touched.  The container is named rather than found, so a job configured
    # somewhere else leaves this row unscheduled and misses all three terms claims.
    cohort.assert_the_job_fires_every(_ROUND_CONTAINER, _AURORA_APPLY.cadence_seconds)
    cohort.assert_the_job_notifies(_ROUND_CONTAINER)
    cohort.assert_the_job_ends_when_asked(_ROUND_CONTAINER, expected=_AURORA_APPLY.expects_expiry)
    cohort.assert_no_mechanism_was_created()
    cohort.assert_nothing_was_written()

    # PROVENANCE — the reply half only, and the store half is EMPTY with a reason.
    # ``assert_every_stored_entry_traces_to_the_world`` is ENTAILED here by
    # ``assert_nothing_was_written``: the store claim can only fail on an entry, and any entry
    # at all already fails the stricter one, so it would run 15/15 by construction and measure
    # the entailment rather than the turn.  The reply claim is live throughout, since a turn
    # confirming a job it has just set up is exactly where an hour or a price nobody gave gets
    # stated.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
