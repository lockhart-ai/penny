"""request → apply: the value she was waiting on arrives, and the job stands up (#2005, t3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

The machine is parked on one question — the value the previous ask was short of — and the user
answers it.  The turn composes that answer with what the original ask already settled, rather
than reading the reply as a fresh instruction or asking again for what it already holds, and
the completed signature is what the job is built under.

**THE COMPLETION DRAW, in situ.**  The binder runs with the round's settled half in hand
(``bind_entry(…, settled=…)``) and has only the arriving value to place; the microcontext case
for that mechanism is #2057's, and this is the same mechanism measured through a whole chat
turn.  Both are wanted, and this is the one that can say the composed values reached a job.

**The survivor, and on what basis: MEASURED RATE, AMONG THE VARIANTS THAT CAN EXPRESS THE
BEHAVIOUR.**  The edge's behaviour is composition across two turns, and only two of the five
variants have anything to compose: the routine behind `-listing`, `-count` and `-digest`
declares exactly ONE parameter, so their ask settles nothing (``settled={}``) and the whole
binding arrives in the supply.  A case built on one of them could state the sentence this case
exists to check and never test it — the derived name would be a function of the one value that
just arrived, and a turn that lost the ask's half entirely would pass.

That excludes the best raw numbers in the edge, which is worth recording rather than hiding:
`-count` is the strongest case in the whole transitions family (mean 0.994, 0.900 of samples
fully passing, never below 4 of 5 over six runs), and it cannot carry this edge's sentence.

Of the two that can, the base timetable ask wins on every reading: mean 0.932 against 0.908,
0.333 of samples fully passing against 0.000, and one fully-passing run against none.
`-held-binding` has never fully passed a single sample on any of the six runs that carried it —
a finding recorded on the way past, not a case to tune.  Dominance agrees: the value that
arrives here is the PAGE, which is the half a turn is most likely to treat as a fresh
instruction, while `-held-binding`'s arriving keyword reads as an answer whatever happens.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: `-listing` (a bare address answering an ask whose terms were complete from the start),
`-count` (the address arriving INSIDE a sentence, which this case absorbs as one of its five
wordings), `-digest` (the answer bringing a NEW term with it, so the terms themselves compose
across the turns), `-held-binding` (the reverse order — the page came with the ask and the
thing to look for arrives now).  Every `_RequestApplyCase` fixture stays in
`transition_world.py`, so any of them can come back deliberately.

**Where the composition claim lives.**  The container's name is
``derive_collection_name(routine, values in DECLARED order)`` — a pure shipped function with no
discretion — so *exactly one mechanism was created and it is the name both turns' values
derive* says all of it at once: that a job exists, that it is the right job, and that BOTH
halves of the signature reached it.  A job built from the arriving value alone derives a
different name and fails; so does one that re-asked and built nothing.

**Which means two source checks are ENTAILED by it and are not written.**
``_bound_parameters_check`` reads the values injected into the row, which are the values the
name is a function of; ``_one_container_check`` reads that exactly one collection was built,
which is the claim's own first half.  Both would run at exactly the rate it does.

**Six more did not port** (the outward column):

* ``Check("state: she set the job up with collection_set", tool_was_called(...))`` and
  ``Check("state: she set it running instead of running it now (no browse this turn)",
  tool_not_called(...))`` — ROUTES, keyed to tool NAMES.  Their end-state forms are the terms
  on the row and ``assert_nothing_was_written``; the calls themselves are measured in section
  B, where a cohort that started browsing shows as a variance rise.
* ``Check("state: the routine's program was rendered into it")`` — PRODUCTION ALREADY
  VALIDATES IT: the program is rendered from the routine unconditionally at instantiation, so
  a row carrying the routine carries a program.
* ``_supplied_anchor_check`` — PRODUCTION ALREADY VALIDATES IT.  ``_next_anchor`` keeps the
  anchor of a machine that is already parked, and the from-state is what this world seeds, so
  the claim is entailed by the landing.
* ``Check("reply: asked for no page structure", asked_for_page_structure(reply) is None)`` and
  ``_names_the_cadence_check`` — PHRASING matches on vocabularies somebody guessed in advance.
  What the second was reaching for — whether the reply describes what actually landed — is not
  answerable from prose.

**And two the inward column added**: PROVENANCE, of both kinds.  The source case made no claim
of either, so a sample that confirmed a sailing time or an address nobody gave passed
everything it carried.

**`keeps`, `excludes` and `answers` are all EMPTY, and each is a report.**  The turn sets a job
to run LATER; it reads nothing and keeps nothing, so a keeps set would state a contract the
supply never made and this case claims the opposite.  The supply answers a question rather than
asking one, so a correct reply owes no token.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.  Every page, url and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import derive_collection_name, slug_skill_name
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    HOUR_PART,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    rule_parts,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _NORTH_PIER_URL,
    _SUPPLIED_SPACES,
    _SUPPLIED_TIMETABLE,
    _declared_order,
    _RequestApplyCase,
    assert_parked_in_request_world,
    assert_the_ask_falls_one_short,
    assert_the_registry_holds,
    assert_values_are_new,
    seed_parked_in_request,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "transition-request-to-apply"

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The case id is a filename; this is the contract.
_BEHAVIOUR = (
    "In the chat agent, when the value a round was parked waiting on arrives, Penny composes "
    "it with what the ask already settled and stands the job up under the name both turns' "
    "values derive, on the terms the ask gave — without asking again for what she already "
    "holds and without touching the jobs already running."
)

_CASE = _SUPPLIED_TIMETABLE

# The survivor's supply: the page for the second timetable, arriving on its own.  The sailing
# to watch for is still back in the ask, which is what makes this the composing edge.
#
# Reference reply:
#   done — i'll check the north pier timetable every morning and message you when
#   the dawn sailing shows up.
_SUPPLY = _CASE.supply

# Four more wordings of that same answer.  What varies is only how a person hands over an
# address — bare, introduced, or inside a sentence.  What does NOT vary is the address itself:
# the case names the container both turns' values derive, and it can only do that because the
# value arriving is byte-identical on every arm.
_SUPPLY_PHRASINGS = (
    f"it's {_NORTH_PIER_URL}",
    f"here you go — {_NORTH_PIER_URL}",
    f"this one: {_NORTH_PIER_URL}",
    f"they post it at {_NORTH_PIER_URL}",
)

# The routine's declared parameter order, read off the FIXTURE DRAFT the runner upserts — which
# is what lets the expected name below be computed without a database, so the claim stays a
# pure function over the sample.  That the REGISTRY agrees with the draft is not assumed: the
# probe reads the order back off the registry row once it exists.
_DECLARED_ORDER = [parameter.name for parameter in _CASE.parked.skill.parameters]

# The container this job runs into — the SHIPPED derivation over the routine and BOTH turns'
# values, computed here rather than written down: a name spelled out would be a second copy of
# the naming scheme, free to drift from the one production identifies jobs by, and silently.
_EXPECTED_CONTAINER = derive_collection_name(
    slug_skill_name(_CASE.parked.skill.name),
    [_CASE.bound[name] for name in _DECLARED_ORDER],
)

# The routine the round was negotiating, as the registry holds it — what the move has to name
# out of the five real routines this world's history taught.
_NEGOTIATED_ROUTINE = slug_skill_name(_CASE.parked.skill.name)

# The ground every arm is answered against: every space these supplies name, installed as LIVE
# temptations, so a turn that DOES go and open the page it was handed gets a real page back
# rather than failing on a thin fixture — which is what makes "she set it running instead of
# running it now" a real reading.
#
# ``keeps``, ``excludes`` and ``answers`` are all empty; the module docstring says which of
# those is a report and why.
_SUPPLIED_WORLD = World(
    name=_CASE_ID,
    pages=tuple(_SUPPLIED_SPACES),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: a
# completed binding mints no routine, so on a correct cohort both read the world's own seeded
# five on every sample — a reading of the FIXTURE rather than of the turn.
#
# ``JOB_TERMS`` is what the model chose here: the binder supplies the routine and the values,
# so the rule it wrote, whether the job speaks and whether it stops are the whole of the turn's
# discretion.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, TRANSITIONS, REPLY_SPREAD)


# ── The probe: the round really is one value short, and this supply completes it ──


def _probe_parked_round(case: _RequestApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the registry one that is only true once the
    runner has laid the fixture skills down, and this case's own four — the ask really did fall
    short, the supply completes it, the registry declares the order this case's expected name
    is built from, and the job it completes is one this world has never stood up."""

    def probe(penny: Penny) -> None:
        assert_parked_in_request_world(penny.db, case)
        assert_the_registry_holds(penny.db, case.parked.journeys)
        assert_the_ask_falls_one_short(penny.db, case.parked)
        assert_the_supply_completes_the_routine(penny.db, case)
        assert_the_job_has_no_container_yet(penny.db, case)
        assert_values_are_new(penny.db, case.case_id, case.supplies.values())
        assert_every_wording_carries_the_value(case)

    return probe


def assert_the_supply_completes_the_routine(db: Database, case: _RequestApplyCase) -> None:
    """The supply answers exactly what the ask left out, the two together answer the routine's
    declared parameters, and the registry declares them in the order this case's expected name
    is derived from.

    All three matter and none is checkable from another.  A supply answering something the ask
    had already settled would leave the signature short whatever the model did, so every claim
    here would read as a failure the turn never made.  A supply answering a parameter the
    routine dropped would point the job at a value nothing binds.  And a registry whose
    declared ORDER differs from the draft's would make the expected container name plausible,
    different from every seeded job, and wrong — so the derived-name claim would go on passing
    while measuring a job nobody asked for."""
    declared = _declared_order(db, case.parked.skill)
    assert declared == _DECLARED_ORDER, (
        f"{case.case_id}: the registry declares {declared}, this case derives its name from "
        f"{_DECLARED_ORDER}"
    )
    assert sorted(declared) == sorted(case.bound), (
        f"{case.case_id}: the routine declares {sorted(declared)}, the two turns settle "
        f"{sorted(case.bound)}"
    )
    assert sorted(case.supplies) == sorted(case.parked.missing), (
        f"{case.case_id}: the ask fell short of {sorted(case.parked.missing)}, the supply "
        f"answers {sorted(case.supplies)}"
    )


def assert_the_job_has_no_container_yet(db: Database, case: _RequestApplyCase) -> None:
    """Nothing carries this job yet: the container its completed values derive is a name no
    collection in this world holds.

    The premise of the whole edge — the request turn built nothing, because a job short of a
    value has no name to build under — and what would otherwise let a FIND score as a mint."""
    assert db.memories.get(_EXPECTED_CONTAINER) is None, (
        f"{case.case_id}: {_EXPECTED_CONTAINER!r} must not exist until this turn builds it"
    )


def assert_every_wording_carries_the_value(case: _RequestApplyCase) -> None:
    """Every arm's wording supplies the value the round is waiting on.

    The facts are held constant across a cohort's arms because the assertions hinge on them,
    and here the container's derived name does: a wording that dropped the address would be an
    answer that answers nothing, so its sample would stay parked in request and fail every
    claim for a reason that has nothing to do with the behaviour."""
    for wording in (_SUPPLY, *_SUPPLY_PHRASINGS):
        missing = [value for value in case.supplies.values() if value not in wording]
        assert not missing, f"{case.case_id}: this wording supplies none of {missing} — {wording!r}"


# ── The one claim this case makes alone ───────────────────────────────────────


def _states_an_hour_to_run_at(sample: SampleObservation, _world: World) -> Answer:
    """The stored rule names an hour to fire at, because the terms name a time of DAY.

    "Every morning" is not a period, it is a moment: a rule stating only its frequency fires at
    whatever instant the collection happened to be created, so a job set up at four in the
    afternoon checks the timetable every afternoon for ever.  WHICH hour is not claimed — the
    code owner's leeway ruling — only that one was chosen at all, read as a PART the rule
    states rather than off the parsed object, since dateutil defaults an unstated hour to the
    start's and cannot tell a chosen hour from an inherited one.

    Local, and it stays local: it is the only ported case whose terms name a time of day, so
    there is no second customer for it to graduate to.  A violating sample stores something
    like ``FREQ=DAILY`` with no ``BYHOUR`` part and is named by the rationale."""
    job = next((one for one in sample.mechanisms if one.name == _EXPECTED_CONTAINER), None)
    if job is None or job.schedule is None:
        return False, f"{_EXPECTED_CONTAINER!r} carries no schedule"
    return HOUR_PART in rule_parts(job.schedule), f"states {sorted(rule_parts(job.schedule))}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_request_to_apply_composes_both_turns_and_stands_the_job_up(
    chat_eval: ChatEval, model: str
) -> None:
    """request → apply: the page for the second timetable arrives on its own, and the sailing
    to watch for is still back in the ask — so the job that stands up carries one value from
    each turn, under the name those two values derive, on the cadence the ask gave."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_parked_in_request(_CASE),
        seed_skills=[journey.round.skill for journey in _CASE.parked.journeys],
        prepare=_probe_parked_round(_CASE),
        world=_SUPPLIED_WORLD,
        ask=_SUPPLY,
        also_phrased=_SUPPLY_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED — where the turn went, and which routine the move it recorded was about.  The
    # routine is a real decision here and not a carry: a round parked in request has no
    # framing, so nothing is supplied framework-side and the binder has to pick it again.
    cohort.assert_machine_landed(ConversationState.APPLY)
    cohort.assert_the_move_named_the_routine(_NEGOTIATED_ROUTINE)

    # STORE — the composed identity first, then the terms the ask gave it.
    cohort.assert_the_job_is_the_container_its_values_derive(_EXPECTED_CONTAINER)
    cohort.assert_the_job_fires_every(_EXPECTED_CONTAINER, _CASE.cadence_seconds)
    cohort.claim(
        "state: the job's rule names an hour to run at",
        _states_an_hour_to_run_at,
        SpecCategory.STORE,
    )
    cohort.assert_the_job_notifies(_EXPECTED_CONTAINER)
    cohort.assert_the_job_ends_when_asked(_EXPECTED_CONTAINER, expected=_CASE.expects_expiry)
    # And what it did NOT do: run the round now, or reach into the five jobs already going.
    cohort.assert_nothing_was_written()
    cohort.assert_no_running_mechanism_was_changed()

    # PROVENANCE — the half the source case had none of.  The store claim answers over an empty
    # set on a correct sample and names the invention on one that opened the page and kept what
    # it found; the reply claim is live throughout, since a turn confirming a job it has just
    # set up is exactly where a sailing time or an address nobody gave gets stated.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
