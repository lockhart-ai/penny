"""Chat in APPLY, entered from request: the missing value arrives and the job stands up.

The machine is parked on one question -- the value the previous ask was short of -- and the user
answers it. The turn composes the answer with what the original ask already settled, rather than
treating the reply as a fresh instruction or re-asking for what it already holds.
"""

from __future__ import annotations

from functools import partial

import pytest

from penny.conversation_machine import (
    ConversationState,
    RoundShortfall,
)
from penny.database import Database
from penny.database.models import MemoryRow, StateTransition
from penny.database.skills import (
    slug_skill_name,
)
from penny.penny import Penny

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
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    Preparer,
    asked_for_page_structure,
    new_collections,
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
from penny.tests.eval.utils.transition_ledger import _FAMILY, _declared_parameters
from penny.tests.eval.utils.transition_world import (
    _PARKED_MESSAGE_WINDOW,
    _SUPPLIED_BAKERY,
    _SUPPLIED_COUNT,
    _SUPPLIED_LISTING,
    _SUPPLIED_PIER,
    _SUPPLIED_SPACES,
    _SUPPLIED_TIMETABLE,
    _bound_parameters_check,
    _derived_container,
    _enactment_binding_check,
    _fresh_mint_check,
    _job_setup_advisories,
    _job_stood_up_checks,
    _job_terms_checks,
    _landed_apply_move,
    _mentions_any,
    _minted_job,
    _RequestApplyCase,
    _seeded_ask_id,
    _seeded_jobs_untouched_check,
    _skill_binding_check,
    assert_parked_in_request_world,
    assert_the_ask_falls_one_short,
    assert_the_registry_holds,
    assert_values_are_new,
    seed_parked_in_request,
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


def _assert_the_binding_reads_the_registry(db: Database, case: _RequestApplyCase) -> None:
    """The recorded binding names the routine as the REGISTRY holds it — its name, its
    description, and each missing parameter's own what-to-supply.

    The drift check the seeder cannot make: it builds the binding from the fixture draft
    because the runner lays the registry down after it, so this is where the two are read
    against each other.  A binding naming a routine or a description the registry does not
    carry would render the classifier a waiting-on section describing a job nobody could
    look up, and the case would report that as the model's miss."""
    latest = db.machine.latest_transition()
    recorded = latest.round_shortfall if latest is not None else None
    assert recorded is not None, f"{case.case_id}: the parked round must carry its binding"
    waiting = RoundShortfall.model_validate_json(recorded)
    routine = db.skills.get(waiting.skill)
    assert routine is not None, (
        f"{case.case_id}: the binding must name a routine the registry holds, got {waiting.skill}"
    )
    assert waiting.description == routine.description, (
        f"{case.case_id}: the binding must carry the registry's description, got "
        f"{waiting.description!r}"
    )
    declared = {
        parameter.name: parameter.description
        for parameter in _declared_parameters(db, waiting.skill)
    }
    asked = {parameter.name: parameter.description for parameter in waiting.missing}
    assert asked == {name: declared[name] for name in asked}, (
        f"{case.case_id}: each missing parameter must carry the registry's own line, got {asked}"
    )


def _probe_parked_round(case: _RequestApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the two that are only true once the
    runner has laid the fixture skills down (the registry holds exactly this world's
    routines, and the recorded binding reads back against the registry), and this case's
    own three — the ask really did fall short, the supply really does complete it, and the
    job it completes really is one this world has never stood up."""

    def probe(penny: Penny) -> None:
        assert_parked_in_request_world(penny.db, case)
        assert_the_registry_holds(penny.db, case.parked.journeys)
        assert_the_ask_falls_one_short(penny.db, case.parked)
        _assert_the_binding_reads_the_registry(penny.db, case)
        assert_the_supply_completes_the_routine(penny.db, case)
        assert_the_job_has_no_container_yet(penny.db, case)
        assert_values_are_new(penny.db, case.case_id, case.supplies.values())

    return probe


def assert_the_supply_completes_the_routine(db: Database, case: _RequestApplyCase) -> None:
    """The supply answers exactly what the ask left out, and the two together answer the
    routine's declared parameters — every one of them, and nothing it does not declare.

    Both halves matter.  A supply that answers something the ask had already settled would
    leave the signature short whatever the model did, so every check in the beat would read
    as a failure the turn never made; a supply that answers a parameter the routine dropped
    would point the job at a value nothing binds.  Read off the REGISTRY row, which is the
    list the binder is actually handed."""
    declared = sorted(
        parameter.name
        for parameter in _declared_parameters(db, slug_skill_name(case.parked.skill.name))
    )
    assert declared == sorted(case.bound), (
        f"{case.case_id}: the routine declares {declared}, the two turns settle "
        f"{sorted(case.bound)}"
    )
    assert sorted(case.supplies) == sorted(case.parked.missing), (
        f"{case.case_id}: the ask fell short of {sorted(case.parked.missing)}, "
        f"the supply answers {sorted(case.supplies)}"
    )


def assert_the_job_has_no_container_yet(db: Database, case: _RequestApplyCase) -> None:
    """Nothing carries this job yet: the container its completed values derive is a name no
    collection in this world holds.

    The premise of the whole beat — the request turn built nothing, because a job short of
    a value has no name to build under — and the thing the fresh-mint check would otherwise
    read as a mint when it was a find."""
    container = _derived_container(db, case.parked.skill, case.bound)
    assert db.memories.get(container) is None, (
        f"{case.case_id}: {container!r} must not exist until this turn builds it"
    )


# ── Scoring ───────────────────────────────────────────────────────────────────


def _landed_in_apply_check(landed: StateTransition | None) -> Check:
    """The beat's headline: the supply moved the machine out of request and into apply.

    Structural, off the move the turn recorded — where a turn ended up is a row, never a
    reading of the reply.  Every other landing is a distinct finding and the rationale names
    which one: request means the binder is still short of something, elicit means the
    routine was taken to be the wrong one, idle means the answer was read as chat.

    It is the only check that reads the landing.  Nothing else scores it — a second check
    conditioned on the same row would count one miss twice."""
    to_state = landed.to_state if landed is not None else None
    applied = to_state == ConversationState.APPLY.value
    return Check(
        "state: the turn landed in apply",
        applied,
        rationale=None if applied else f"the machine landed in {to_state}",
        kind="state",
    )


def _supplied_anchor_check(
    db: Database, landed: StateTransition | None, case: _RequestApplyCase
) -> Check:
    """The move came FROM request and still points at the ask that opened the round
    (#1827) — the anchor is the turn the supply was bound TOGETHER WITH, so a move arriving
    with a different one (or none) is a round that lost half of what it is completing.
    Same conditional-n/a as every other check that reads the move."""
    label = "state: the move came from request with the ask still its anchor"
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.parked.ask, limit=_PARKED_MESSAGE_WINDOW)
    carried = (
        applied.from_state == ConversationState.REQUEST.value
        and asked is not None
        and applied.anchor_message_id == asked
    )
    return Check(
        label,
        carried,
        rationale=None
        if carried
        else (
            f"came from {applied.from_state}, anchored to {applied.anchor_message_id} "
            f"(the ask is {asked})"
        ),
        kind="state",
    )


def _one_container_check(db: Database, before: set[str]) -> Check:
    """ONE ask, ONE collection: the turn built exactly one container.

    The completing half of the derived-name claim.  A second collection is a second job
    nobody asked for — and since the job the user is handed back is the one carrying the
    routine, an extra one is a mechanism that either runs unnoticed or takes the name the
    next ask for this job would derive."""
    created = [row.name for row in new_collections(db, before)]
    built_one = len(created) == 1
    return Check(
        "state: exactly one collection was built for the job",
        built_one,
        rationale=None if built_one else f"created {created}",
        kind="state",
    )


def _score_request_to_apply(
    db: Database, before: set[str], reply: str, *, case: _RequestApplyCase
) -> list[Check]:
    """The answer the request turn asked for arrived, so the job it was negotiating stands
    up — on the routine it named, in the container its completed values derive, pointed at
    what BOTH turns settled, on the terms they gave, and without disturbing any of the jobs
    already running.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are diff-join
    keys and are deliberately case-NEUTRAL: one wording reads the same whether the value
    that arrived was the page or the thing to look for on it."""
    row = _minted_job(db, before)
    landed = db.machine.latest_transition()
    return [
        _landed_in_apply_check(landed),
        *_supplied_binding_checks(db, before, row, landed, case),
        # The cadence is the ASK's, and the end condition only where one was given at all —
        # case 4's arriving with the value, case 2's back with the ask.
        *_job_terms_checks(
            db,
            row,
            fires_every=case.cadence_seconds,
            anchored=case.anchored,
            expects_expiry=case.expects_expiry,
        ),
        _seeded_jobs_untouched_check(db, case.parked.journeys),
        _supplied_anchor_check(db, landed, case),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        _names_the_cadence_check(reply, case),
        *_job_setup_advisories(db, row, landed),
    ]


def _names_the_cadence_check(reply: str, case: _RequestApplyCase) -> Check:
    """The reply says what is RUNNING now, in the terms they gave — a floor on the answer,
    read as the cadence being named at all, with how well it was said read at joint review
    against the reference reply.

    That reference reply is the check's own tripwire: a vocabulary that cannot match the
    wording the case declares CORRECT would score the beat's own answer a miss on every
    sample, so the pin in ``test_eval_harness.py`` runs it through this set without a
    GPU."""
    named = _mentions_any(case.cadence_tokens, reply)
    return Check(
        "reply: she says what is running now, naming the cadence",
        named,
        rationale=None if named else f"named none of {list(case.cadence_tokens)}",
        kind="reply",
    )


def _supplied_binding_checks(
    db: Database,
    before: set[str],
    row: MemoryRow | None,
    landed: StateTransition | None,
    case: _RequestApplyCase,
) -> list[Check]:
    """She set a job up, on the routine the round was negotiating, in the ONE container its
    completed values derive, pointed at everything the two turns settled.

    The values are the binder's since #1870, so what is scored here is that the COMPLETED
    signature reached the job: the container's name is a function of every value, and the
    bound parameters are what the framework injected from it — which is exactly what a
    binding completed over two turns has to get right."""
    return [
        _skill_binding_check(
            _landed_apply_move(landed),
            intended=slug_skill_name(case.parked.skill.name),
            label="state: the decision bound the routine the round was negotiating",
        ),
        *_job_stood_up_checks(db, row),
        _enactment_binding_check(row, case.parked.skill),
        _fresh_mint_check(db, row, case.parked.skill, case.bound),
        _one_container_check(db, before),
        _bound_parameters_check(
            row,
            wanted=tuple(case.bound.values()),
            label="state: the routine is pointed at what both turns settled",
        ),
    ]


async def _run_request_apply_case(chat_eval: ChatEval, case: _RequestApplyCase) -> None:
    """Drive one request → apply case: parked in request on its own short ask with the
    composed world behind it, exactly the routines its history taught in the registry, every
    supplied space installed as a live temptation, and the shared scorer bound to the terms
    its two turns give.  Report-only — the thresholds are the code owner's to set once the
    numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.supply,
        browse=_SUPPLIED_SPACES,
        seed=seed_parked_in_request(case),
        seed_skills=[journey.round.skill for journey in case.parked.journeys],
        prepare=_probe_parked_round(case),
        score=partial(_score_request_to_apply, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_request_to_apply_stands_the_job_up_on_the_page_that_arrives(
    chat_eval: ChatEval,
) -> None:
    """request → apply: the page for the second timetable arrives on its own, and the
    sailing to watch for is still back in the ask — so the job that stands up carries one
    value from each turn, on the cadence the ask gave."""
    await _run_request_apply_case(chat_eval, _SUPPLIED_TIMETABLE)


@pytest.mark.asyncio
async def test_request_to_apply_binds_the_listing_the_ask_left_out(chat_eval: ChatEval) -> None:
    """request → apply on the price watcher: the terms were complete from the start and only
    the listing was missing, so this bare address is the whole answer — and the end
    condition the ASK gave still has to reach the job."""
    await _run_request_apply_case(chat_eval, _SUPPLIED_LISTING)


@pytest.mark.asyncio
async def test_request_to_apply_sets_up_the_weekly_count(chat_eval: ChatEval) -> None:
    """request → apply on the count watcher: the address arrives inside a sentence, which is
    how a person answers where something is posted — the value is a span of the message
    rather than the whole of it."""
    await _run_request_apply_case(chat_eval, _SUPPLIED_COUNT)


@pytest.mark.asyncio
async def test_request_to_apply_composes_the_new_end_with_the_asks_cadence(
    chat_eval: ChatEval,
) -> None:
    """request → apply where the answer brings a term with it: the page AND an end
    condition the ask never gave, so the job's terms compose across the two turns — the
    cadence from the ask, the end from this message."""
    await _run_request_apply_case(chat_eval, _SUPPLIED_BAKERY)


@pytest.mark.asyncio
async def test_request_to_apply_completes_the_held_binding(chat_eval: ChatEval) -> None:
    """request → apply from the other side of the held binding: the page came with the ask
    and what to look for on it arrives now, so the completed signature is assembled in the
    reverse order — and the job still lands on the one container both values derive."""
    await _run_request_apply_case(chat_eval, _SUPPLIED_PIER)
