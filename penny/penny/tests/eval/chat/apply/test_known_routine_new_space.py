"""idle → apply: a known routine pointed at a new space (#2005, tranche 2).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

The world holds five finished journeys, so the routine already exists and the ask teaches
nothing — it names the new place to run it and the job's terms, in ONE cold message, with
everything the interface needs.  The turn recognises the routine, binds it to the page the ask
names, and stands the job up on those terms, without opening the page and without disturbing
any of the five jobs already running.

**The survivor, and on what basis: MEASURED RATE.**  The edge's five variants have real
per-variant numbers and they separate sharply.  Across the eight suite runs that carried them
the base cold price ask reports mean 0.99 / 0.99 / 1.00 / 0.99 / 1.00 / 1.00 / 0.99 / 0.89, and
is the ONLY variant never below 3 of 5 samples fully passing.  `-urgency` is the worst in the
suite (0 of 5 fully passing on three of its seven runs, means down to 0.25); `-two-params` is
1 of 5 or worse on six of its eight; `-threshold` bottoms out at 0.63 and `-digest` at 0.76.

Dominance argues for `-two-params` — it binds the page AND what to look for on it out of one
message, which is strictly more than one binding, and a world that can produce a failure the
others cannot.  It loses to the measured rate deliberately: this epic picks the variant that
already passes consistently, and tuning a weaker one into the canonical slot is what #2005 says
not to do.  The finding is recorded rather than acted on — the two-parameter cold bind is the
suite's least stable stand-up, and it is quarantined with its numbers rather than lost.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: `-two-params` (both values out of one cold message, with a time-of-DAY cadence so the
rule must state an hour), `-digest` (a plain daily cadence with NO end condition, so an expiry
set here is one nobody asked for), `-threshold` (the longest cadence in the set, against the
closest look-alike the registry holds), `-urgency` (act-now pressure plus an end condition the
model has to work out).  Every `_IdleApplyCase` fixture stays here, so any of them can come
back deliberately.

**What is claimed is what got CREATED and CONFIGURED, read off the registry and the ledger.**
The job's identity is its container's NAME, which is `derive_collection_name(routine, values in
declared order)` — a pure shipped function with no discretion — so a single claim covering
*exactly one mechanism was born, and it is the one this routine and this listing derive* says
both that a job exists and that it is the right job, in a key that is strictly identifiable.
The terms are then record fields on that row: how often it fires, whether it tells the user,
whether it stops.  None is read off the reply, and none is read off a tool name.

**Which routine the JOB runs, and whether it carries a program, are NOT claimed** — the two
obvious neighbours of the name claim, and both PRODUCTION-VALIDATED.  On a turn that configures
a framed round, `CollectionSetTool._with_the_rounds_routine` supplies the collection, the
routine and the bound values framework-side from the round's own `RoundFraming`, and
`round_framing.container_name` derives that container from the same signature — so the row's
name and its `skill_name` are two reads of one framing, and the program is rendered from it
unconditionally.  A claim over either would run at exactly the rate the name claim does while
appearing to measure something else.  What survives the override is the model's: the schedule,
the notification flag and the end condition, which is what the three terms claims read.

**How often it fires is a COUNT, not a rule.**  ``cadence_seconds`` walks the stored rule and
returns the gap between its first two occurrences, so ``FREQ=HOURLY`` and
``FREQ=MINUTELY;INTERVAL=60`` are the same answer — the value, not the notation the draw
happened to choose.

**Six source checks did not port** (the outward column):

* ``Check("state: she set the job up with collection_set", tool_was_called(...))`` — a ROUTE,
  keyed to a tool NAME.  Its end-state form is the born-mechanism claim, which catches a job
  stood up however it was reached.
* ``Check("state: she set it running instead of running it now (no browse this turn)")`` — the
  same, and its end-state form is *nothing was written*: a turn that went and read the listing
  and kept what it found leaves an entry behind.  The browse itself is measured in section B.
* ``_cold_anchor_check`` — PRODUCTION ALREADY VALIDATES IT: ``_next_anchor`` stamps the
  instigating message on every move into a parked state FROM idle, so the claim is entailed by
  the landing.
* ``_no_teach_question_check`` — *she did not ask to be taught*.  ENTAILED by
  ``assert_machine_landed(APPLY)``: a turn cannot land in apply and in elicit, so the check
  would run at exactly the rate the landing claim does.
* ``_expiry_check``'s drawn-argument leg — it reads ``last_tool_args(db, _SET_TOOL)``, which is
  a route keyed to a tool name.  What survives is the ROW's own ``expires_at``, which is the
  right reading in this direction: the ask GIVES an end condition, so a row carrying none
  failed, including when a far-future sentinel normalised it away.
* ``_seeded_jobs_untouched_check`` — the right question, read off the ledger against five
  enumerated names.  It ports as ``assert_no_running_mechanism_was_changed``, which asks it of
  every mechanism rather than of a list somebody wrote down.

**One more is absent by ENTAILMENT**, and is worth naming so the set reads as closed: *nothing
was registered*, the claim that a cold apply teaches nothing.  Run-end extraction fires in
``learn`` and nowhere else, and the only other thing that touches the registry
(``abandon_round_skill``) runs on an IDLE landing — so no sample can fail it without also
failing ``assert_machine_landed``.

**And one the inward column added**: PROVENANCE, of the REPLY kind.  The source case made no
claim of it.  Its store half is absent by ENTAILMENT — this turn writes nothing, which
*nothing was written* already claims, so no stored entry exists to trace and the claim could
only pass vacuously on every sample.

**`keeps` and `answers` are both EMPTY, and each is a report.**  The turn sets a job to run
LATER; it reads nothing and keeps nothing, so a keeps set would state a contract the ask never
made and the case claims the opposite.  The ask requests a job, not a value, so a correct reply
owes no token.

REPORT-ONLY (``min_pass_rate=None``).  Every page, url and job is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import SkillDraft, derive_collection_name, slug_skill_name
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _AURORA_SKILL,
    _JOURNEYS,
    _KEEL_LANTERN_LISTING,
    _KEEL_LANTERN_URL,
    _declared_order,
    assert_composed_world,
    assert_the_registry_holds,
    assert_values_are_new,
    seed_composed_world,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "transition-idle-to-apply"

_BEHAVIOUR = (
    "In the chat agent, when a cold ask names a new place to run a routine she already has "
    "and supplies everything that routine needs, Penny binds it to what the ask names and "
    "stands the job up on the terms it gave — without reading the page, teaching anything, or "
    "touching the jobs already running."
)


class _IdleApplyCase(NamedTuple):
    """One agreed cold ask, and what the job it should stand up has to look like.

    ``skill`` is the routine the ask is covered by — the one of five the decision has to pick.
    ``page`` is the new space, installed as a live temptation.  The rest is what the ask's own
    terms give: ``cadence_seconds`` is how far apart the job should fire whatever rule spelling
    says so, ``expects_expiry`` whether they gave an end condition at all (inventing one is a
    failure), and ``bound`` every value the MESSAGE supplies that the routine has to be pointed
    at.

    ``bound`` is KEYED BY PARAMETER NAME since #1870, because the container's name is derived
    from those values in the routine's DECLARED order — so the fixture states which value
    answers which parameter and the order is read off the registry, rather than being an order
    this tuple has to be kept in and nothing could check."""

    case_id: str
    ask: str
    page: CannedPage
    skill: SkillDraft
    cadence_seconds: int
    expects_expiry: bool
    bound: dict[str, str]


# The survivor: a second listing for the price watcher, long after it was taught on the first.
# Nothing in the ask refers back, so the routine has to be recognised from what it asks for.
#
# Reference reply:
#   done — i'll check the keel lantern listing every hour until sunday night and
#   message you if the price moves.
_COLD_ASK = (
    "can you watch this listing for me and let me know when the price changes? "
    f"{_KEEL_LANTERN_URL} — every hour until sunday night is fine"
)

# Four more wordings of that same ask.  What varies is only how a person says it — which verb
# opens it, where the terms sit relative to the address, "changes" or "moves".  What does NOT
# vary is the page, the cadence, the end condition or the telling clause: the case names a
# cadence and a derived container, and it can only do that because they are constant.
_COLD_ASK_PHRASINGS = (
    f"keep an eye on {_KEEL_LANTERN_URL} and tell me when the price changes — every hour "
    "until sunday night",
    f"could you check {_KEEL_LANTERN_URL} every hour until sunday night and let me know if "
    "the price moves?",
    f"watch this one and tell me when the price changes please: {_KEEL_LANTERN_URL} — hourly, "
    "up to sunday night",
    f"i'd like {_KEEL_LANTERN_URL} checked every hour until sunday night, with a message when "
    "the price changes",
)

# How a person asks to be TOLD, as this case's own five wordings say it.  An explicit set
# rather than a derived one because what it reads is the FIXTURE, authored here: a sixth
# wording that asks another way adds its opening to this line, which is the point — the
# alternative is a wording that quietly asks for nothing and is scored as though it had.
_TELLING_CLAUSES = ("tell me", "let me know", "message when")

# The claim those clauses feed, named once so the probe that guards its input and the claim
# itself cannot drift apart.
_NOTIFIES_CLAIM = "state: the job tells them when it changes"

_COLD_PRICE = _IdleApplyCase(
    case_id=_CASE_ID,
    ask=_COLD_ASK,
    page=_KEEL_LANTERN_LISTING,
    skill=_AURORA_SKILL,
    cadence_seconds=3600,
    expects_expiry=True,
    bound={"url": _KEEL_LANTERN_URL},
)

# The container this job runs into — the SHIPPED derivation over the routine it runs and the
# value it is pointed at, computed here rather than written down for the reason the seeded
# rounds' containers are: a name spelled out would be a second copy of the naming scheme, free
# to drift from the one production identifies jobs by, and silently.
#
# The declared ORDER the derivation needs comes off the REGISTRY, which the claim cannot read —
# so the probe below asserts the registry declares exactly what this fixture supplies, in this
# order, and the claim is then a pure function over the sample.
_EXPECTED_CONTAINER = derive_collection_name(
    slug_skill_name(_COLD_PRICE.skill.name), list(_COLD_PRICE.bound.values())
)

# The routine this ask is covered by, as the registry holds it — what the decision has to pick
# out of five real routines of the same kind.
_COVERING_ROUTINE = slug_skill_name(_COLD_PRICE.skill.name)

# The ground every arm is answered against: the new space, installed as a live temptation, so a
# turn that DOES open it gets a real page back rather than failing on a thin fixture — which is
# what makes "she set it running instead of running it now" a real reading.
#
# ``keeps``, ``excludes`` and ``answers`` are all empty; the module docstring says which of
# those is a report and why.
_NEW_LISTING = World(
    name=_CASE_ID,
    pages=(_COLD_PRICE.page,),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: a
# cold apply mints no routine, so on a correct cohort both read the world's own seeded five on
# every sample — a reading of the FIXTURE rather than of the turn.  A sample that DID mint one
# is caught by the registry claim, where it is a miss rather than a variance rise.
#
# ``JOB_TERMS`` is the one thing left for the model to choose once the framing has supplied the
# container, the routine and the values — the rule it wrote, whether the job speaks, whether it
# stops.  Added with tranche 3's two stand-up edges (#2005), which made three customers for it.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, TRANSITIONS, REPLY_SPREAD)

# Every cold ask, in one place — so the deterministic pin in ``test_eval_harness.py`` can check
# the case's claim about the world without a GPU.
IDLE_APPLY_CASES = (_COLD_PRICE,)


def _probe_composed_world(case: _IdleApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the registry one that is only true once the
    runner has laid the fixture skills down, and this case's own three — the routine declares
    exactly what the ask supplies, the space is new to the world, and every wording names it."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db)
        assert_the_registry_holds(penny.db, _JOURNEYS)
        assert_the_ask_fills_the_routine(penny.db, case)
        assert_new_space_is_unknown(penny.db, case)
        assert_every_wording_names_the_space(case)

    return probe


def assert_the_ask_fills_the_routine(db: Database, case: _IdleApplyCase) -> None:
    """The case's ``bound`` values answer the routine's declared parameters — every one of
    them, in DECLARED ORDER, and nothing the routine does not declare.

    That mapping is what the derived container's name is built from (#1870), so a fixture
    naming a parameter the routine dropped, or missing one it added, would derive a name that
    still looks plausible and still differs from every seeded job: the born-mechanism claim
    would go on passing while measuring a job nobody asked for.  Read off the REGISTRY row,
    since the registry is what the binder is handed — and asserted as an ORDER rather than a
    set, because the claim computes the expected name from this fixture's own ordering and has
    no database of its own to check it against."""
    declared = _declared_order(db, case.skill)
    assert declared == list(case.bound), (
        f"{case.case_id}: the routine declares {declared}, the ask supplies {list(case.bound)}"
    )


def assert_new_space_is_unknown(db: Database, case: _IdleApplyCase) -> None:
    """Every value this ask supplies is NEW to the world it is answered against.

    This is what makes "bound from the message" a real claim: a value the history also carries
    could have been copied out of the world instead of read out of the ask."""
    assert_values_are_new(db, case.case_id, case.bound.values())


def assert_every_wording_names_the_space(case: _IdleApplyCase) -> None:
    """Every arm's wording supplies every value the routine has to be pointed at, and asks to
    be TOLD — every fact a claim over this cohort reads.

    The facts are held constant across a cohort's arms because the assertions hinge on them,
    and here two do: the container's derived name hangs on the address, so a wording that
    dropped it would be an ask the binder falls SHORT of — its sample would land in request and
    fail every claim for a reason that has nothing to do with the behaviour.  The telling
    clause is the same shape of defect one step quieter: ``the job tells them when it changes``
    reads ``notifies`` on EVERY sample, so an arm that never asked to be told scores that claim
    against a job correctly configured not to notify, and the miss reads as the model's."""
    for wording in (case.ask, *_COLD_ASK_PHRASINGS):
        missing = [value for value in case.bound.values() if value not in wording]
        assert not missing, f"{case.case_id}: this wording supplies none of {missing} — {wording!r}"
        assert any(clause in wording for clause in _TELLING_CLAUSES), (
            f"{case.case_id}: this wording never asks to be told, so "
            f"{_NOTIFIES_CLAIM!r} has nothing to read — {wording!r}"
        )


# ── The claims: what got created, and what it was configured to do ────────────
#
# Every one reads the MECHANISM RECORD — the collection row as the sample left it — because
# this is the positive direction of the reads a bail makes: what a turn that stands a job up
# leaves behind is a row, and its terms are that row's own fields.  Nothing here reads a tool
# name, and nothing reads the reply.  (The move naming the routine is the case's other kind of
# claim and reads the machine's own row instead.)
#
# All four are in ``assertions.py``: tranche 3's two stand-up edges arrived as their second and
# third customers (#2005), and each label went with its claim so one claim's history stays one
# row.
#
# ONE READING CHANGED when they graduated, and it is worth stating because it changes what these
# claims answer on a WRONG sample.  They used to read the single row BORN this run, and ``None``
# where the turn minted none or several — so a turn that built nothing failed them for want of a
# job.  They now read the container BY NAME, born or not, because the accepted offer settles the
# terms of a container its round built EARLIER and the born-this-run reading holds in only two
# of the three edges.  Here the two coincide on a correct sample, since the derived container is
# the row this turn mints; where they part is a turn that configured a row it did not create,
# which the name reading answers on its own terms and the born reading could only report as an
# absence.


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_idle_to_apply_points_a_known_routine_at_a_new_listing(
    chat_eval: ChatEval, model: str
) -> None:
    """idle → apply, cold: a second listing, long after the price watcher was taught on the
    first.  Nothing in the ask refers back, so she has to recognise the job from what it asks
    for, bind the page it names, and set it running on the hours and the end it gives —
    without opening the listing to check."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_composed_world(),
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        prepare=_probe_composed_world(_COLD_PRICE),
        world=_NEW_LISTING,
        ask=_COLD_PRICE.ask,
        also_phrased=_COLD_ASK_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED — where the turn went, and which routine the move it recorded was about.
    cohort.assert_machine_landed(ConversationState.APPLY)
    cohort.assert_the_move_named_the_routine(_COVERING_ROUTINE)

    # STORE — what got created, and the terms it runs on.
    cohort.assert_the_job_is_the_container_its_values_derive(_EXPECTED_CONTAINER)
    cohort.assert_the_job_fires_every(_EXPECTED_CONTAINER, _COLD_PRICE.cadence_seconds)
    cohort.assert_the_job_notifies(_EXPECTED_CONTAINER)
    cohort.assert_the_job_ends_when_asked(_EXPECTED_CONTAINER, expected=_COLD_PRICE.expects_expiry)
    # And what it did NOT do: run the round now, teach anything, or reach into the five jobs
    # already going.
    cohort.assert_nothing_was_written()
    cohort.assert_no_running_mechanism_was_changed()

    # PROVENANCE — the half the source case had none of.  The STORE half is absent by
    # entailment: this turn writes nothing, which ``assert_nothing_was_written`` above already
    # claims, so no stored entry exists to trace and the claim could only pass vacuously —
    # fifteen guaranteed-green checks inflating the headline rate.  The reply claim carries the
    # category, and is live throughout: a turn confirming a job it just set up is exactly where
    # an hour or a price nobody gave gets stated.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
