"""idle → learn: the teach arrives whole (#2005, tranche 2).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

No elicit round precedes this one — the user's FIRST message already carries the steps, so one
turn both runs the demonstrated round and learns it.  The world is the composed one: five
finished journeys, five jobs already running, and a history that knows neither this page nor
the fact it holds, so everything the round reports has to come off the demonstration it was
just given.

**The survivor, and on what basis: MEASURED RATE.**  The edge's five variants have real
per-variant numbers, and they separate.  Across the three suite runs that carried them, the
base teach (the prose "i'm gonna teach you how to check the harbour flag") reports mean
1.00 / 0.97 / 0.99 with 5, 4 and 4 of 5 samples fully passing — the only variant never below
4/5.  The others each fall away somewhere: `-numbered` to 0/5 fully passing on the most recent
run, `-filter` to 0.42 mean on one, `-new-routine` to 0.62 on the same, and `-deferred-terms`
to 2 of 5 fully passing on every run it appears in.  The base is also the plainest statement of
the behaviour — one page, one value to read, one value to keep, said as a sentence — and it
matches the register the reference port (`transition-elicit-to-learn`) is built on.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: `-numbered` (the same three steps as an enumeration, so the teach is unmistakable and
the parse is not), `-filter` (a page listing three events where the instruction keeps one, so
the round must read past two it was not sent for), `-deferred-terms` (a teach that also states
a NOTIFY condition, which must be left for the turn that accepts the offer — the teach-and-
instantiate fold), `-new-routine` (the word "routine" said by the USER while five routines are
already running).  All five `_TeachCase` fixtures stay in `transition_world.py`, so any of them
can come back deliberately.

**Two claims are this world's own**, and neither exists on the reference port, whose world has
no jobs in it: the only mechanism this turn creates is the round's own container, and nothing
that was already running was changed.

**One reference claim is REPLACED rather than reused.**
``assert_a_routine_reached_the_registry`` reads ``bool(sample.routines)``, which is VACUOUSLY
TRUE here — the world seeds five routines before the turn begins, so it would pass on a sample
that learned nothing.  Its honest form on this world is a COUNT: the five it already had, plus
exactly one more.  The two claims that also read every routine — every spot a placeholder, the
routine names a destination — are kept unchanged, because they are ALL-quantified: the seeded
five satisfy them by construction, so their truth value is the new routine's, and the count
claim closes the one hole that leaves (a cohort that minted nothing at all).

**Six source checks did not port** (the outward column):

* ``Check("state: she configured nothing", tool_not_called(db, _SET_TOOL))`` — a ROUTE, keyed
  to a tool NAME.  Its end-state form is ``assert_nothing_was_scheduled``, which catches a job
  stood up however it was reached, including by a plugin verb nobody enumerated.
* ``_teach_anchor_check`` — *the move came from idle with the teach as its anchor.*
  PRODUCTION ALREADY VALIDATES IT: ``_next_anchor`` stamps the instigating message on every
  move into a parked state FROM idle, so on a sample that landed in learn the claim is
  entailed by the landing.
* ``_framed_checks`` — that the round carries a framing at all is entailed the same way, and
  what it is FOR is read by ``assert_the_write_landed_in_the_round_container``, which is the
  suite's only reader of it.
* ``_extraction_shape_checks`` — the routine's shape.  A skill is an arbitrary tool sequence,
  so its shape is measured (``ROUTINE_SHAPE``), never asserted.
* ``_round_reported_checks`` — a PHRASING match: the reply saying the value back.  What it
  reached for is #2010's wrong-but-stable row, which the design measures with nothing.
* ``_seeded_jobs_untouched_check`` — the right question, read off the ledger with the case's
  own journey list.  It ports as ``assert_no_running_mechanism_was_changed``, which asks it of
  every mechanism rather than of five enumerated names.

**`answers` is EMPTY, and that is a REPORT with a caveat worth stating.**  The teach says to
READ the flag and SAVE it; it never asks Penny to say it back.  A token the ask does not
require fails a correct run, and widening an ask is a code owner's call rather than an
in-flight repair — so the reply-completeness claim is not made here, and the known defect it
would have caught (the learn close naming the write RECORD instead of the value, #2010) stays
where `docs/eval-case-design.md` §9 puts it: catchable only by a human reading one sample.

REPORT-ONLY (``min_pass_rate=None``).  Every page, url and job is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import slug_skill_name
from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval, Preparer
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    ROUTINE_NAME,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _JOURNEYS,
    _TEACH_HARBOUR_FLAG,
    _TeachCase,
    assert_composed_world,
    assert_the_registry_holds,
    assert_values_are_new,
    seed_composed_world,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# Read off the survivor rather than spelled again: the fixture already carries the edge's id,
# and two spellings of one id are two things free to drift.
_CASE_ID = _TEACH_HARBOUR_FLAG.case_id

_BEHAVIOUR = (
    "In the chat agent, when a message arrives already carrying the instructions for a job, "
    "Penny runs that round once against the page it names, keeps what it says in the round's "
    "own container, and mints a routine from what she just did — without setting it running "
    "or touching any of the jobs already going."
)

# The survivor's own teach, in five wordings.  What varies is only how a person introduces a
# lesson and phrases three actions — "i'm gonna teach you" or "let me show you", "read" or
# "see" or "find", "save" or "keep" or "remember".  What does NOT vary is the page, the value
# to read off it, the single source, or the prose register: those are what the round is
# measured against, and the case asserts a value only because they are constant.
_HARBOUR_TEACH_PHRASINGS = (
    f"let me teach you how to check the harbour flag: open {_TEACH_HARBOUR_FLAG.url}, "
    "see which flag is flying today, and keep it",
    f"i want to show you how to check the harbour flag — head to {_TEACH_HARBOUR_FLAG.url}, "
    "find which flag is flying today, and remember it",
    f"here's how to check the harbour flag, penny: read {_TEACH_HARBOUR_FLAG.url}, note which "
    "flag is flying today, and save that",
    f"teaching you the harbour flag check — visit {_TEACH_HARBOUR_FLAG.url}, work out which "
    "flag is flying today, and hang on to it",
)

# The ground every arm is answered against.  ONE page, because that is what the survivor's
# steps name, and the ask has exactly one controllable fact on it — so ``keeps`` can name a
# value rather than a shape.
#
# ``bravo`` is the SMALLEST datum that uniquely identifies the flag flying today: the page also
# names Alpha and Charlie in its meanings list and a storm cone beside the mast, so nothing but
# the right read satisfies it, and it carries no notation a draw was free to choose.
#
# ``excludes`` is EMPTY and that is a report: the teach rules nothing out in as many words, and
# whether the storm cone beside the flag is in scope is a judgement — which is variance, not an
# assertion.  ``answers`` is empty for the reason the module docstring gives.
_HARBOUR_SIGNALS_WORLD = World(
    name=_CASE_ID,
    pages=(_TEACH_HARBOUR_FLAG.page,),
    keeps=((_TEACH_HARBOUR_FLAG.stored,),),
    excludes=(),
)

# The routines the composed world's history taught — what the registry holds when the teach
# arrives, and therefore what "one more" is counted against.
_ALREADY_TAUGHT = tuple(slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS)


# ── The probe: the world is the composed one, and it knows neither page nor fact ─


def _probe_teach_world(case: _TeachCase) -> Preparer:
    """The prepare hook: the composed seeder's own claims, the registry one that is only
    true once the runner has laid the fixture skills down, the case's own novelty claim, and
    the premise that every wording names the page its steps point at."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db)
        assert_the_registry_holds(penny.db, _JOURNEYS)
        assert_the_teach_is_new_to_the_world(penny.db, case)
        assert_every_wording_names_the_page(case)

    return probe


def assert_the_teach_is_new_to_the_world(db: Database, case: _TeachCase) -> None:
    """The page this teach names and the fact that page holds are BOTH new to the history
    it is taught in.

    The page's novelty is what makes the round a real demonstration rather than a re-run of
    something already done.  The FACT's novelty is what makes the store claim mean anything: a
    value the seeded history already carries could have been copied out of the world instead of
    read off the page, and the claim would pass either way.

    Its own premise rides along, because both ways of getting it wrong are silent on a run:
    the instructions must NAME the page (a teach whose steps point nowhere is a round nothing
    can carry out), and the fixture must be the page those instructions reach (a match token
    the url does not carry serves a no-results page, and the round then dies on a fixture
    rather than on anything the model did)."""
    assert case.url in case.teach, f"{case.case_id}: the teach must name the page it points at"
    assert case.page.match.lower() in case.url.lower(), (
        f"{case.case_id}: the fixture must answer {case.url!r}, it matches on {case.page.match!r}"
    )
    assert_values_are_new(db, case.case_id, (case.url, case.stored))


def assert_every_wording_names_the_page(case: _TeachCase) -> None:
    """Every arm's wording points at the SAME page.

    The facts are held constant across a cohort's arms because the assertions hinge on them,
    and here the whole store claim does: a wording that named a different address — or none —
    would be a different round, and its sample would fail every claim for a reason that has
    nothing to do with the behaviour."""
    for wording in (case.teach, *_HARBOUR_TEACH_PHRASINGS):
        assert case.url in wording, f"{case.case_id}: this wording names no page — {wording!r}"


# ── The claims this world's own situation adds ────────────────────────────────
#
# Both are LOCAL: each is parametrised by what this world already holds, and no other case has
# asked for either.  Between them they say that the turn built exactly the one thing a
# demonstrated round builds and left the five jobs behind it alone — which is the half of the
# contract the reference port's world (an empty machine) cannot state at all.


def _the_registry_gained_one_routine(sample: SampleObservation, _world: World) -> Answer:
    """The registry holds the routines the world taught, plus EXACTLY one more.

    A count rather than a name, because the name is the framer's to choose — measured, ten
    distinct names for one routine across fifteen samples — and a count is strictly
    identifiable.  Both halves are load-bearing: a sample that learned nothing keeps the
    seeded five, and one that minted two (a re-teach forking the round) holds seven."""
    taught = sorted(routine.name for routine in sample.routines)
    kept = sorted(_ALREADY_TAUGHT)
    gained = [name for name in taught if name not in kept]
    ok = all(name in taught for name in kept) and len(gained) == 1
    return ok, f"the registry holds {taught}, which is {len(gained)} beyond the seeded {kept}"


def _built_only_the_round_container(sample: SampleObservation, _world: World) -> Answer:
    """The only mechanism this turn created is the container the round was framed on.

    A demonstrated round builds exactly one thing on its way in, and a second collection minted
    beside it is a job nobody asked for — the shape a turn takes when it reads the teach as an
    instruction to set something up rather than to be shown something once."""
    born = sorted(one.name for one in sample.mechanisms if one.born_this_run)
    allowed = [] if sample.container is None else [sample.container]
    return born == allowed, f"created {born}, the round was framed on {sample.container!r}"


# What this case measures.  ``ROUTINE_NAME`` is the framer's naming spread, read off the whole
# registry: the seeded five are a constant prefix in the sorted set, so what moves is the name
# this turn minted — which is exactly what the feature is for, and it is COSMETIC, so its spread
# is a system finding rather than a fact about one sample.
#
# ``ROUTINE_SHAPE`` is deliberately ABSENT, and this is the one place in the tranche where its
# absence is a MEASUREMENT finding rather than a reading of the fixture.  It joins every
# routine's shape in the registry's NAME order, so on a world seeding five routines the string
# moves when the minted routine's shape moves AND when its NAME sorts to a different slot — a
# CONSEQUENTIAL feature inheriting the naming spread, which is the trap `docs/eval-case-design.md`
# §5 measures (naming accounted for 8 of 9 outlier rows in one run).  Reading only the routines
# BORN this run would fix it and needs a field the observation does not carry; recorded rather
# than added, since porting does not change the substrate it measures against.
_MEASURED = (
    TOOL_SEQUENCE,
    ROUTINE_NAME,
    ENTRIES_STORED,
    TRANSITIONS,
    REPLY_SPREAD,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_idle_to_learn_runs_the_taught_round_in_one_turn(
    chat_eval: ChatEval, model: str
) -> None:
    """idle → learn, the canonical single-turn teach: the message says it is teaching and then
    gives the three steps, so there is nothing left to elicit.  The round is framed on the way
    in, run once — the signals page read, the flag saved into the round's own container — a
    routine is minted from what just happened, and nothing is set running."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_composed_world(),
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        prepare=_probe_teach_world(_TEACH_HARBOUR_FLAG),
        world=_HARBOUR_SIGNALS_WORLD,
        ask=_TEACH_HARBOUR_FLAG.teach,
        also_phrased=_HARBOUR_TEACH_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.LEARN)

    # STORE
    cohort.assert_something_from_each_page_was_written()
    cohort.assert_the_write_landed_in_the_round_container()
    cohort.claim(
        "state: the registry gained exactly one routine",
        _the_registry_gained_one_routine,
        SpecCategory.STORE,
    )
    cohort.assert_every_spot_is_a_placeholder()
    cohort.assert_the_routine_names_a_destination()
    cohort.assert_nothing_was_scheduled()
    cohort.claim(
        "state: the only mechanism this turn created is the round's own container",
        _built_only_the_round_container,
        SpecCategory.STORE,
    )
    cohort.assert_no_running_mechanism_was_changed()

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
