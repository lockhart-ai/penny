"""The round ends in IDLE: four edges, four cohorts (#2005, tranche 1).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**One behaviour family, four cases.**  Three parked states bail back to idle when the user
drops the task, and the fourth case is the floor — an ordinary remark arriving on an idle
machine, which fires nothing at all.  They are four cases rather than one because they are
four different asks against four different worlds: what selects the behaviour is the state
the round is parked in, and folding them together would average four behaviours into one
score and call the result instability.

The code owner's ruling they all answer (#1896): *"all bail back to idle preserves nothing;
once in idle any new user task starts a new flow anyways, i don't see any reason to preserve
intermediate state."*  So the contract is that the round ENDS — nothing is built, nothing is
configured, nothing is registered, and the mechanisms already running are none of the turn's
business — plus, for the one round that built something, that what it built is retired.

**Which survivor each edge kept.**

* ``elicit → idle`` — the auction round parked on the teach question.  The one world the
  edge has.
* ``learn → idle`` — the mid-teach round, container built and written into.  The one bail
  with something to clean up, which is why it carries a claim none of the others can.
* ``request → idle`` — the HELD-BINDING world (``_SUPPLIED_PIER``), kept over the
  nothing-settled one (``_SUPPLIED_TIMETABLE``) because it strictly dominates it: the round
  holds a settled half AND an open one, so a turn that read the bail as a go-ahead has
  enough to stand a job up with, which the empty-binding world does not.  The dropped
  variant is recorded here so it can come back deliberately — it was the case that used to
  carry the id ``transition-request-to-idle``, which the survivor takes over, since #2005
  is one canonical case per EDGE and the edge is what the id names.
* ``idle → idle`` — ordinary banter with five live jobs behind her and one of them watching
  the very thing the remark mentions.  Every claim it makes is a negative, because the
  failure it exists to catch is firing anything at all.

**Everything else is deliberately LOOSE.**  An idle turn is ordinary chat with the full tool
surface, so answering well — including going and looking something up when the message
carries a real question — is not a miss, and the pages are installed so that a browse
SUCCEEDS rather than failing invisibly.  The reference replies are review targets, never
scorer strings.

**Three source checks did not port, each for its own reason** (the outward column):

* ``_round_ended_check`` — *the landing carries no anchor, no framing, no partial binding.*
  PRODUCTION ALREADY VALIDATES IT.  ``_next_anchor``, ``_next_framing``, ``_next_shortfall``
  and ``_next_provenance`` each return ``None`` the moment the target is ``idle``, so the
  claim is entailed by the landing and would run at exactly the rate ``assert_machine_landed``
  does while appearing to measure something else.  What a bail from ``request`` durably risks
  is not a kept binding but a job STOOD UP out of the half it had settled, and that is what
  the created/changed claims below read.
* ``Check("state: she configured nothing", tool_not_called(db, _SET_TOOL))`` — a ROUTE, keyed
  to a tool NAME.  Many routes reach one end state and a skill is an arbitrary tool sequence,
  so its end-state form is *no mechanism was changed*, which catches a reconfiguration however
  it was reached and catches a plugin verb nobody enumerated.
* ``_claims_no_job_check`` — a PHRASING match on a fourteen-entry vocabulary somebody guessed
  in advance, which is the thing this design exists to abolish.  What it was reaching for is
  structural and is claimed as such: she cannot truthfully say a job is running when the
  store says none was created and none was changed.

**And one the inward column added**, which there was nothing to copy: PROVENANCE.  The source
case made no claim of either kind, so a sample that answered the harbour-market question out
of its own head — or filed an invented fact into a collection — passed every check it carried.

**One the inward column offered and these four cases refuse**:
``assert_every_delivered_message_is_whole``.  It reads ``SampleObservation.delivered``, which
is every outgoing message in the last hour — and every world here SEEDS Penny's own turns
(the teach question, the closing report, five job confirmations, three exchanges of small
talk), all of them written within the same second the sample starts.  So the claim would be
answered mostly against the fixture's agreed prose rather than against what this turn sent,
which is a check that measures the seed.  What it was reaching for on the live reply is
pinned deterministically instead, in ``test_eval_harness.py``, against each case's own
reference reply.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.  Every page, url and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import (
    SkillDraft,
    slug_skill_name,
)
from penny.penny import Penny
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
    Seeder,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)

# The listing this script is built on, read from the suite's shared fixtures rather than
# restated here: two copies of the page a journey is measured against are two contracts
# free to drift.
from penny.tests.eval.utils.fixtures import (
    AURORA_LISTING_499,
    CannedPage,
)
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.transition_world import (
    _AURORA_APPLY,
    _AURORA_ROUND,
    _DECOY_SKILL,
    _JOURNEYS,
    _SUPPLIED_PIER,
    _SUPPLIED_SPACES,
    _ApplyCase,
    _assert_parked_on_the_ask,
    _assert_seeded_world,
    _ElicitRound,
    _RequestApplyCase,
    _seed_elicit_round,
    _seeded_ask_id,
    assert_composed_world,
    assert_parked_in_request_world,
    assert_round_cites_its_run,
    assert_round_is_framed,
    assert_seeded_ledger,
    seed_composed_world,
    seed_learned_round,
    seed_parked_in_request,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval


# The page the first bail's own question would reach: it changes the subject to something
# ordinary, and an idle turn is free to go and answer it, so the world has something to find.
#
# Matched on the plain word, since that is the only token the question and the address SHARE
# — which is also why it is installed AFTER the listing that round was about: a direct read
# of ``faux-market.example`` carries the word too, and the listing's own distinctive token
# is matched first (``install_browse`` serves the first page whose token is in the url).
_HARBOR_MARKET_URL = "https://harbormarket.example/this-weekend"
_HARBOR_MARKET = CannedPage(
    match="market",
    text=(
        "Title: Harbour market — this weekend | harbormarket\n"
        f"{_HARBOR_MARKET_URL}\n"
        "\n"
        "A fictional weekend market on the quay, open saturday and sunday mornings.\n"
        f"[Harbour market this weekend]({_HARBOR_MARKET_URL})\n"
        "This weekend: smoked fish, a cider stall, and the pottery co-op's summer seconds.\n"
    ),
)


class _BailWorld(NamedTuple):
    """The parked state a bail arrives on, and what is true of it.

    ``seed`` and ``skills`` are an EARLIER beat's own composition, read from where that beat
    defines them rather than restated, so a bail is answered against exactly the world its
    own edge was measured against.  ``seeded`` is the loud probe for what is true the moment
    the seeder returns — registry-free on purpose, so the plain pin can drive it without the
    fixture skills the runner lays down afterwards.

    ``container`` is what the round BUILT, and therefore the one mechanism the landing may
    touch — ``None`` for every state that built nothing, which is all of them but learn (a
    job short of a value has no name yet, and an ask that has not been taught has no round).
    """

    seed: Seeder
    skills: tuple[SkillDraft, ...]
    seeded: Callable[[Database], None]
    container: str | None


class _BailCase(NamedTuple):
    """One agreed bail, the world it lands in, and the five wordings it is asked in.

    ``bail`` and ``also_phrased`` are five wordings of ONE message against one world — the
    cohort's arms.  What varies is only how a person says it; the state the round is parked
    in, the pages a turn may reach for and the end state expected of it are constant, which
    is what makes the fifteen samples one number.

    ``pages`` are the spaces a turn is free to reach for — a temptation only in the sense
    that a browse here is ALLOWED, so what they buy is that a turn which does look up finds
    something instead of failing on a thin fixture.

    ``reference`` is how the bail would be answered WELL — a review target, never matched by
    a claim.  It is DATA rather than prose so the deterministic pin can read it without a GPU.
    """

    case_id: str
    behaviour: str
    world: _BailWorld
    bail: str
    also_phrased: tuple[str, ...]
    pages: tuple[CannedPage, ...]
    reference: str

    @property
    def ground(self) -> World:
        """The world every arm of this case is answered against.

        ``keeps`` and ``answers`` are both EMPTY, and each is a report rather than an
        omission.  ``keeps`` states what a round must have written down, and none of these
        rounds is asked to write anything — a keeps set here would state a contract the bail
        never made.  ``answers`` states what a correct reply owes, and none of these four
        messages asks for a particular value: "anything good at the market this weekend?" is
        answered correctly by naming any one of three stalls, so requiring a token would fail
        a correct run for something nobody requested.  Widening an ask so it requests a value
        is a code owner's call rather than an in-flight repair, so it is raised in the report
        instead of folded in here."""
        return World(name=self.case_id, pages=self.pages, keeps=(), excludes=())


def _assert_parked_in_elicit(db: Database, case: _ElicitRound) -> None:
    """The elicit-parked world, re-read once the sample's Penny is up — the same claim the
    seeder makes on its way out, made again where a drift would otherwise be invisible."""
    _assert_seeded_world(db, case, _seeded_ask_id(db, case.ask))


def _assert_parked_in_learn(db: Database, case: _ApplyCase) -> None:
    """The mid-teach world: parked in learn on the ask, the round FRAMED and its container
    built and written into, the round's calls in the ledger, everything citing its own run.

    ``_probe_seeded_world``'s claims minus the registry one, which is only true once the
    runner has laid the fixture skills down — so this is exactly the half that is code and
    the plain pin can drive it."""
    _assert_parked_on_the_ask(db, case)
    assert_round_is_framed(db, case)
    assert_seeded_ledger(db, case)
    assert_round_cites_its_run(db, case)


_PARKED_IN_ELICIT = _BailWorld(
    seed=_seed_elicit_round(_AURORA_ROUND),
    skills=(),
    seeded=partial(_assert_parked_in_elicit, case=_AURORA_ROUND),
    container=None,
)

_PARKED_IN_LEARN = _BailWorld(
    seed=seed_learned_round(_AURORA_APPLY),
    skills=(_AURORA_APPLY.skill, _DECOY_SKILL),
    seeded=partial(_assert_parked_in_learn, case=_AURORA_APPLY),
    container=_AURORA_APPLY.framing.container,
)


def _parked_in_request(case: _RequestApplyCase) -> _BailWorld:
    """The request-parked world one short ask left: its own journeys walked to their end,
    then the turn that asked for what the ask left out — the #1893 seeder, used as itself,
    binding and all.  Nothing was built for it, so the bail has nothing to retire."""
    return _BailWorld(
        seed=seed_parked_in_request(case),
        skills=tuple(journey.round.skill for journey in case.parked.journeys),
        seeded=partial(assert_parked_in_request_world, case=case),
        container=None,
    )


_UNPARKED = _BailWorld(
    seed=seed_composed_world(),
    skills=tuple(journey.round.skill for journey in _JOURNEYS),
    seeded=assert_composed_world,
    container=None,
)


_BAIL_FROM_ELICIT = _BailCase(
    case_id="transition-elicit-to-idle",
    behaviour=(
        "In the chat agent, when she has asked to be taught a job and the user calls it off "
        "and changes the subject in the same breath, Penny lets go of the round entirely — "
        "creating nothing, changing nothing and registering nothing — and answers the new "
        "subject as the ordinary conversation it is."
    ),
    world=_PARKED_IN_ELICIT,
    bail="ah never mind, forget that — anything good at the harbor market this weekend?",
    also_phrased=(
        "actually drop that — is there anything worth seeing at the harbour market this weekend?",
        "eh, forget the listing thing — what's on at the harbour market this weekend?",
        "never mind that, let's leave it — anything decent at the harbour market this weekend?",
        "scrap that idea — is the harbour market worth a look this weekend?",
    ),
    pages=(AURORA_LISTING_499, _HARBOR_MARKET),
    reference=(
        "no worries, dropped it. the harbour market's on both mornings this weekend — "
        "smoked fish, a cider stall, and the pottery co-op's summer seconds."
    ),
)

_BAIL_FROM_LEARN = _BailCase(
    case_id="transition-learn-to-idle",
    behaviour=(
        "In the chat agent, when a teach round is under way with its container built and "
        "written into and the user abandons it, Penny archives that container and registers "
        "nothing — leaving every other collection exactly as she found it."
    ),
    world=_PARKED_IN_LEARN,
    bail="actually forget it, i don't need this",
    also_phrased=(
        "you know what, forget it — i don't need this after all",
        "actually never mind, i don't need this",
        "hmm, drop it — i don't need this",
        "let's forget the whole thing, i don't actually need it",
    ),
    pages=(AURORA_LISTING_499,),
    reference="no problem — i've dropped it. shout if you want to pick it up again.",
)

_BAIL_FROM_HELD_BINDING = _BailCase(
    case_id="transition-request-to-idle",
    behaviour=(
        "In the chat agent, when a round is parked waiting on the one detail an ask left out "
        "and the user calls it off, Penny ends the round and builds nothing out of the half "
        "it had already settled — no job stood up, and none of the running ones touched."
    ),
    world=_parked_in_request(_SUPPLIED_PIER),
    bail="you know what, skip it",
    also_phrased=(
        "eh, skip it — not worth bothering with",
        "actually let's skip it",
        "never mind, skip that one",
        "forget it, skip it",
    ),
    pages=tuple(_SUPPLIED_SPACES),
    reference="sure thing — skipping it.",
)

_BANTER_ON_IDLE = _BailCase(
    case_id="transition-idle-to-idle",
    behaviour=(
        "In the chat agent, when a message arriving on an idle machine asks for nothing that "
        "needs to keep running, Penny answers it in conversation and changes nothing — even "
        "with five jobs already running behind her and one of them watching what it mentions."
    ),
    world=_UNPARKED,
    bail="the ferry ride this morning was gorgeous btw",
    also_phrased=(
        "btw the ferry crossing this morning was gorgeous",
        "the ferry over this morning was lovely, just saying",
        "oh, the ferry ride this morning was really something",
        "morning ferry was gorgeous today btw",
    ),
    pages=tuple(_SUPPLIED_SPACES),
    reference="lovely — the light on the water first thing is hard to beat.",
)

# Every bail, in one place — so the deterministic pin in ``test_eval_harness.py`` can drive
# each one's seeder without a GPU.
BAIL_CASES = (
    _BAIL_FROM_ELICIT,
    _BAIL_FROM_LEARN,
    _BAIL_FROM_HELD_BINDING,
    _BANTER_ON_IDLE,
)


# ── The probe: the world really is parked where the case says ─────────────────


def _probe_bail_world(case: _BailCase) -> Preparer:
    """The prepare hook: the world's own claims, plus the registry one that is only true
    once the runner has laid the fixture skills down."""

    def probe(penny: Penny) -> None:
        case.world.seeded(penny.db)
        assert_the_bail_registry(penny.db, case)
        assert_the_round_built_what_it_claims(penny.db, case)

    return probe


def assert_the_bail_registry(db: Database, case: _BailCase) -> None:
    """The registry holds exactly the routines this world's history taught — none at all for
    a world whose round was never demonstrated.

    Its own reading rather than ``assert_the_registry_holds``'s, because a bail world is
    described by the SKILLS it seeds and not by its journeys: the mid-teach world taught no
    journey at all and still carries two routines (its own fixture and the decoy)."""
    taught = sorted(skill.name for skill in db.skills.list_all())
    assert taught == _routines_of(case), (
        f"{case.case_id}: the registry must hold {_routines_of(case)}, got {taught}"
    )


def assert_the_round_built_what_it_claims(db: Database, case: _BailCase) -> None:
    """The container premise, both ways round: a world whose round built one starts with it
    LIVE (so "it was archived" is a claim about this turn and not a row that arrived
    retired), and a world whose round built nothing carries no framing to retire.

    Silent either way on a run: a container already archived would score the beat's headline
    green for free, and a framing nobody noticed would make the case's one cleanup claim a
    question about a container the world never had."""
    container = case.world.container
    if container is None:
        latest = db.machine.latest_transition()
        assert latest is None or latest.skill_frame is None, (
            f"{case.case_id}: a round that built nothing carries no framing, got "
            f"{latest and latest.skill_frame}"
        )
        return
    row = db.memories.get(container)
    assert row is not None and not row.archived, (
        f"{case.case_id}: the round's container {container!r} must start live, got {row}"
    )


def _routines_of(case: _BailCase) -> list[str]:
    """The registry names this world's history taught, sorted — what the probe asserts before
    the turn and what the claim below re-asserts after it.

    ONE reading, because the two are the same question at two moments: a second spelling
    could let the probe pass a world the claim then reports as changed."""
    return sorted(slug_skill_name(draft.name) for draft in case.world.skills)


# ── The claims, as pure functions over one sample ─────────────────────────────
#
# Four of them, and every one is about what did NOT happen — which is the shape of the whole
# family, so each is written so that a violating sample is nameable.  A vacuously-true
# negative would be the easiest thing in the world to ship here.
#
#   * created           — a sample that stands the abandoned job up, or mints a container for
#                         the subject the bail changed to, fails it.  The registry read is a
#                         list of rows, so "nothing" is a count and not an inference.
#   * registry          — a sample whose turn was read as more of the teach round mints a
#                         routine at run end (extraction fires in ``learn`` and nowhere else),
#                         and one read as a correction replaces the row that is there.  Both
#                         move the sorted name list off the world's own.
#   * touched           — a sample that reconfigures a running job, re-renders it, archives it
#                         or edits its description records a mutation citing this turn's run.
#                         Read off the LEDGER rather than off a field-by-field diff, so the
#                         change nobody enumerated is caught too.
#   * archived (learn)  — a sample that leaves the round's container live, or one that revives
#                         it after the landing archived it, fails it.
#
# What no claim here reads is a TOOL NAME: a skill is an arbitrary tool sequence, so the
# question is what the store holds afterwards and never which verb got it there.
#
# All four stay LOCAL rather than graduating into ``assertions.py``.  The rule is that a claim
# graduates at the second CUSTOMER, and the four cases below are one behaviour family in one
# file answering one contract in four worlds — a second file is what would make one of these a
# shared claim, and none of the eleven other edges has asked for it yet.  Two of them could not
# graduate anyway: they are parametrised by the case's own world.


def _nothing_was_created(sample: SampleObservation, _world: World) -> Answer:
    """No collection was created — not an inert one, not a configured one, none.  A bail ends
    the round, and a container built on the way out is a job nobody asked for."""
    born = sorted(one.name for one in sample.mechanisms if one.born_this_run)
    return not born, f"created {born}"


def _registry_unchanged(case: _BailCase) -> Callable[[SampleObservation, World], Answer]:
    """Nothing was REGISTERED and nothing was taken away: the routines the world taught are
    the routines it still has.

    The claim the mid-teach bail exists to make — a round walked away from teaches nothing,
    whatever it demonstrated — and one every other case makes too, since a bail that minted a
    routine would leave the user with a mechanism they had just called off."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        taught = sorted(routine.name for routine in sample.routines)
        return taught == _routines_of(case), f"the registry holds {taught}"

    return answer


def _touched_nothing_but_its_own_container(
    case: _BailCase,
) -> Callable[[SampleObservation, World], Answer]:
    """The only mechanism this turn may change is the one the round built.

    For three of the four that is NO mechanism at all, because those rounds built nothing;
    for the mid-teach bail it is the round's own container, which the landing retires.  One
    sentence rather than two labels, because it is one rule read against each world's own
    answer to "what did this round build".

    The mechanisms already running are none of a bail's business, and that is what this reads:
    a live turn's mutation cites a live run and every event the seeded world wrote cites a
    seeded one, so "this turn changed nothing here" is a read rather than a diff."""
    allowed = {case.world.container} if case.world.container is not None else set()

    def answer(sample: SampleObservation, _world: World) -> Answer:
        touched = sorted({one.name for one in sample.mechanisms if one.changed_this_run} - allowed)
        return not touched, f"changed {touched}"

    return answer


def _the_round_container_was_archived(
    container: str,
) -> Callable[[SampleObservation, World], Answer]:
    """The round's container was ARCHIVED — the one thing this bail has to clean up.

    Archived, never deleted: the row stays a visible tombstone, so a bail drawn off a flaky
    classification is recoverable and the same job taught again revives it.  Which is why the
    claim reads the ROW rather than the ledger — a container archived and then revived within
    the turn has an archive event and is live, and live is the answer that matters."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = next((one for one in sample.mechanisms if one.name == container), None)
        if row is None:
            return False, f"{container!r} is no longer in the registry at all"
        return row.archived, f"{container!r} is still live"

    return answer


# The three STORE labels every bail claims under.  Named once because a label is a diff-join
# key: four copies of one sentence are four chances for a typo to split one claim's history
# into two.  Deliberately case-NEUTRAL — one wording reads the same whether the abandoned
# round was a teach loop, a negotiation, or no round at all.
_NOTHING_CREATED = "state: no mechanism was created"
_REGISTRY_UNCHANGED = "state: the registry holds exactly the routines it already had"
_TOUCHED_ONLY_ITS_OWN = "state: the only mechanism this turn changed is the one the round built"


# What every case measures, and why the two registry features are absent.  ``ROUTINE_SHAPE``
# and ``ROUTINE_NAME`` read the routines in the registry, and no bail mints one — so on a
# correct cohort they read the world's own seeded routines on every sample and pool to a serene
# 0.000 that is neither agreement nor blindness but a reading of the FIXTURE.  What a sample
# that did mint a routine changes is the registry claim above, where it is a miss rather than a
# variance rise.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


async def _drive(chat_eval: ChatEval, model: str, case: _BailCase) -> Cohort:
    """Drive one bail case: the parked world its own edge was measured against, exactly the
    routines that world taught, and the spaces an idle turn may reach for installed so a
    lookup finds something rather than failing invisibly."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        seed=case.world.seed,
        seed_skills=list(case.world.skills),
        prepare=_probe_bail_world(case),
        world=case.ground,
        ask=case.bail,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# ── elicit → idle: nothing was ever built for that round ──────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_elicit_to_idle_drops_the_task_and_answers_the_new_one(
    chat_eval: ChatEval, model: str
) -> None:
    """Parked having asked to be taught, the user calls it off and changes the subject in the
    same breath.  Nothing was ever built for that round, so the whole contract is that the
    machine lets go of it — and the new question is answered as ordinary conversation, which
    an idle turn has the full tool surface for."""
    cohort = await _drive(chat_eval, model, _BAIL_FROM_ELICIT)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(_NOTHING_CREATED, _nothing_was_created, SpecCategory.STORE)
    cohort.claim(_REGISTRY_UNCHANGED, _registry_unchanged(_BAIL_FROM_ELICIT), SpecCategory.STORE)
    cohort.claim(
        _TOUCHED_ONLY_ITS_OWN,
        _touched_nothing_but_its_own_container(_BAIL_FROM_ELICIT),
        SpecCategory.STORE,
    )

    # PROVENANCE — the half the source case had none of.  This bail changes the subject to a
    # question with an answer on a page, so a reply that answered it out of the model's own
    # head rather than out of the market page fails the second claim, and a fact filed into a
    # collection that nobody's page mentions fails the first.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ── learn → idle: the one bail with something to clean up ─────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_learn_to_idle_archives_the_abandoned_round(chat_eval: ChatEval, model: str) -> None:
    """Parked mid-teach with the round's container built and the demonstrated value in it, the
    user drops the whole thing.  The container goes with the round — the one bail that has
    something to clean up — and nothing is registered from what was demonstrated."""
    cohort = await _drive(chat_eval, model, _BAIL_FROM_LEARN)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(_NOTHING_CREATED, _nothing_was_created, SpecCategory.STORE)
    cohort.claim(_REGISTRY_UNCHANGED, _registry_unchanged(_BAIL_FROM_LEARN), SpecCategory.STORE)
    cohort.claim(
        _TOUCHED_ONLY_ITS_OWN,
        _touched_nothing_but_its_own_container(_BAIL_FROM_LEARN),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the round's container was archived",
        _the_round_container_was_archived(_AURORA_APPLY.framing.container),
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ── request → idle: the half that was already settled ─────────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_request_to_idle_drops_a_binding_that_was_half_settled(
    chat_eval: ChatEval, model: str
) -> None:
    """The round had already settled the page and was waiting only on what to watch for, so
    this is the bail with the most to preserve and the ruling is that it preserves none of it.

    That the partial binding itself is dropped is not claimed here: an idle landing clears it
    on the transition row by construction (``_next_shortfall``), so the claim would be entailed
    by the landing.  What the settled half durably risks IS claimed — a turn that reads the
    bail as a go-ahead has a page and a routine in hand and can stand the job up, which the
    created and changed claims read off the registry."""
    cohort = await _drive(chat_eval, model, _BAIL_FROM_HELD_BINDING)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(_NOTHING_CREATED, _nothing_was_created, SpecCategory.STORE)
    cohort.claim(
        _REGISTRY_UNCHANGED, _registry_unchanged(_BAIL_FROM_HELD_BINDING), SpecCategory.STORE
    )
    cohort.claim(
        _TOUCHED_ONLY_ITS_OWN,
        _touched_nothing_but_its_own_container(_BAIL_FROM_HELD_BINDING),
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ── idle → idle: the no-fire row ──────────────────────────────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_idle_to_idle_fires_nothing_on_ordinary_banter(
    chat_eval: ChatEval, model: str
) -> None:
    """Five live jobs behind her, an idle machine, and a remark that asks for nothing — while
    naming the very thing one of those jobs already watches.  Every claim is a negative,
    because the failure this case exists to catch is firing anything at all: standing a ferry
    watch up beside the one that exists, or reaching into the one that does."""
    cohort = await _drive(chat_eval, model, _BANTER_ON_IDLE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(_NOTHING_CREATED, _nothing_was_created, SpecCategory.STORE)
    cohort.claim(_REGISTRY_UNCHANGED, _registry_unchanged(_BANTER_ON_IDLE), SpecCategory.STORE)
    cohort.claim(
        _TOUCHED_ONLY_ITS_OWN,
        _touched_nothing_but_its_own_container(_BANTER_ON_IDLE),
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE reads "no call" on a correct sample here, so a cohort that behaves
    # perfectly makes this feature BLIND and the report says so in red.  Measured anyway,
    # because the divergence it exists to catch is exactly the one this case is named for:
    # a sample that went and looked something up, or reached for a job, reads differently
    # from every other one and the blindness lifts the moment it does.
    cohort.measure(*_MEASURED)
