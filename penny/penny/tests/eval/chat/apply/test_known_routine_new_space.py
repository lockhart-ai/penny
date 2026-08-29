"""Chat in APPLY, entered cold from idle: a known routine pointed at a new space.

The world holds five finished journeys, so the routine already exists and the ask is not teaching anything -- it names the routine and the new place to run it, in one message, with everything the interface needs.  The turn binds the parameters from that one message and sets the job running.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import islice
from typing import NamedTuple

import pytest
from dateutil.rrule import rrulestr

from penny.constants import ChatPromptType, PennyConstants, TransitionCause
from penny.conversation_machine import (
    CandidateParameter,
    ConversationState,
    MachineSnapshot,
    RoundFraming,
    RoundProvenance,
    RoundShortfall,
    SkillCandidate,
    render_classifier_content,
)
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput, MemoryType
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, Skill, StateTransition
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    derive_collection_name,
    distill_steps,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.penny import Penny

# The SHIPPED container derivation, used as itself: a seeded round has to run into the
# container production would have built for it, and a fixture spelling that name out would
# be a second copy of the naming scheme, free to drift from the one jobs are identified by.
from penny.round_framing import container_name

# The production draw-application, used as itself: a fixture skill has to be the SHAPE
# run-end extraction really produces, and re-implementing that mapping here would be a
# fixture that drifts from the pipeline it stands in for.  Both halves of the #1824
# split are applied by their own production function — ``_apply_leaf_labels`` for the
# labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's signature.
# ``attachment_names`` is the registry policy for what a routine can be attached to, read
# for the same reason: the scorer asks whether a learned routine HAS a destination, and
# that is the question extraction already answers when it decides which leaves to mark.
from penny.skill_extraction import (
    _apply_leaf_labels,
    _interface_parameters,
    _naming,
    attachment_names,
)
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    ParameterFamily,
    Preparer,
    Seeder,
    asked_for_page_structure,
    chat_run_tool_sequences,
    classify_by_family,
    collection_entries,
    count_tool_calls,
    is_seeded_run,
    last_tool_args,
    live_prompts,
    new_collections,
    outgoing_replies,
    routing_clean,
    seeded_run_id,
    tool_not_called,
    tool_was_called,
)

# The listing this script is built on, and the enacting-tool set the elicitation
# contract IS — the calls that would mean she acted before being taught.  Both are read
# from the suite's shared fixtures rather than restated here: the passing-mention guard
# in ``test_chat_memory_stories.py`` asks the same question of a turn, and two copies of
# one policy are two contracts free to drift.
from penny.tests.eval.utils.fixtures import AURORA_LISTING_499, ENACTING_TOOLS, LISTING_URL, CannedPage

# The agreed breadth for "the page the routine is pointed at", READ from where the framer
# suite declares it rather than restated here: what a page parameter may reasonably be
# called is one code-owner-agreed vocabulary, and two copies would drift into two
# contracts (the same rule ``ENACTING_TOOLS`` is read under).
from penny.tests.eval.framer.test_skill_framing import _PLACE_TOKENS
from penny.text_validity import is_blank

# The production tool-result framer, used as itself: a seeded ledger's tool turns have to
# read the way the loop really writes them, and a hand-written frame is a second copy of a
# format the model is shown every turn.
from penny.tools.base import Tool

# The schedule's own render + grammar tokens, read from where the tool declares them: a
# stored rule renders back AS the copyable ``schedule`` input (#1857), so the advisory shows
# what she committed to in the form it was set, and the line/tag literals a rule is written
# with are that module's to define — a restated copy here would be a second contract.
# ``parse_schedule`` + ``render_reinstantiation_echo`` are read for the same reason on the
# seeding side: a seeded apply turn stores the rule the tool would have stored and echoes
# back what the tool would have echoed.
from penny.tools.collection_instantiation import (
    _DTSTART_TAG,
    _LINE_ESCAPE,
    _RRULE_TAG,
    has_schedule,
    parse_schedule,
    render_reinstantiation_echo,
    render_schedule_clause,
)
from penny.tools.micro_context import (
    SKILL_TAG,
    STATE_CLASSIFIER_SYSTEM_PROMPT,
    STATE_TAG,
    FramedParameter,
    LeafLabel,
    SkillLabels,
    SkillSignature,
    StateDrawOutcome,
)
from penny.tools.models import ToolResult

from penny.tests.eval.utils.transition_ledger import _FAMILY

from penny.tests.eval.utils.transition_world import _ARRIVALS_SKILL, _AURORA_SKILL, _BAKERY_SKILL, _COLONY_SKILL, _COMPOSED_MESSAGE_WINDOW, _EAST_BRANCH_NEW_TITLES, _EAST_BRANCH_URL, _FERRY_SKILL, _HARBOR_BAKERY_MENU, _HARBOR_BAKERY_URL, _JOURNEYS, _KEEL_LANTERN_LISTING, _KEEL_LANTERN_URL, _NORTH_PIER_DEPARTURES, _NORTH_PIER_URL, _RIVER_OTTERS_CENSUS, _RIVER_OTTERS_URL, _declared_order, _enactment_binding_check, _fresh_mint_check, _job_setup_advisories, _job_stood_up_checks, _job_terms_checks, _landed_apply_move, _minted_job, _seeded_ask_id, _seeded_jobs_untouched_check, _skill_binding_check, assert_composed_world, assert_the_registry_holds, assert_values_are_new, cadence_seconds, seed_composed_world


pytestmark = pytest.mark.eval


# ── The five cold asks ────────────────────────────────────────────────────────
#
# Each names its new space and the job's terms and stands entirely alone: nothing in the
# wording refers to the routine already existing, so nothing but the ask's MEANING can
# select among the five.

# Case 1 — a second listing for the price watcher.
#
# Reference reply:
#   done — i'll check the keel lantern listing every hour until sunday night and
#   message you if the price moves.
_COLD_ASK = (
    "can you watch this listing for me and let me know when the price changes? "
    f"{_KEEL_LANTERN_URL} — every hour until sunday night is fine"
)

# Case 2 — a different harbour for the timetable watcher, and the cold TWO-parameter
# bind: the page and what to look for on it both come out of this one message.
#
# Reference reply:
#   done — i'll check the north pier timetable every morning and message you when the
#   dawn sailing shows up.
_COLD_ASK_TWO_PARAMS = (
    f"every morning can you check the north pier timetable at {_NORTH_PIER_URL} "
    "and let me know when they add the dawn sailing?"
)

# Case 3 — a second bakery for the daily digest.
#
# Reference reply:
#   done — i'll check the harbor bakery's menu every day and message you the special.
_COLD_ASK_DIGEST = (
    f"can you get the daily special from {_HARBOR_BAKERY_URL} each day and tell me what it is?"
)

# Case 4 — a different survey for the count watcher.
#
# Reference reply:
#   done — i'll check the otter census every week and message you if the count drops.
_COLD_ASK_THRESHOLD = (
    f"keep track of the otter count at {_RIVER_OTTERS_URL} every week and let me know if it drops"
)

# Case 5 — another branch for the new-arrival watcher, under act-now pressure.
#
# The act-now clause is TEMPORAL, not ordinal (code-owner ruling).  It read "tell me the
# second something new shows up", which is two sentences in one: the intended "the moment
# it appears", and "tell me the SECOND new item" — a count-to-two task no routine covers.
# The classifier's own thinking took the ordinal reading and parked in elicit, correctly:
# nothing in the registry counts arrivals.  An ask that can be read two ways measures the
# reading, not the edge, so it says "when" and means it.
#
# Reference reply:
#   done — i'll check the east branch's new titles every two hours until friday and
#   message you when something new shows up.
_COLD_ASK_URGENCY = (
    f"watch {_EAST_BRANCH_URL} every two hours until friday and tell me when something new shows up"
)


def _probe_composed_world(case: _IdleApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the registry one that is only true once
    the runner has laid the fixture skills down, and the case's own novelty claim."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db)
        assert_the_registry_holds(penny.db, _JOURNEYS)
        assert_the_ask_fills_the_routine(penny.db, case)
        assert_new_space_is_unknown(penny.db, case)

    return probe


def assert_the_ask_fills_the_routine(db: Database, case: _IdleApplyCase) -> None:
    """The case's ``bound`` values answer the routine's declared parameters — every one of
    them, and nothing the routine does not declare.

    That mapping is what the derived container's name is built from (#1870), so a fixture
    naming a parameter the routine dropped, or missing one it added, would derive a name
    that still looks plausible and still differs from every seeded job: the fresh-mint
    check would go on passing while measuring a job nobody asked for.  Read off the
    REGISTRY row rather than the fixture draft, since the registry is what the binder is
    handed."""
    declared = _declared_order(db, case.skill)
    assert sorted(declared) == sorted(case.bound), (
        f"{case.case_id}: the routine declares {declared}, the ask supplies {sorted(case.bound)}"
    )


def assert_new_space_is_unknown(db: Database, case: _IdleApplyCase) -> None:
    """Every value this ask supplies is NEW to the world it is answered against."""
    assert_values_are_new(db, case.case_id, case.bound.values())


# ── The cases ─────────────────────────────────────────────────────────────────


class _IdleApplyCase(NamedTuple):
    """One agreed cold ask, and what the job it should stand up has to look like.

    ``skill`` is the routine the ask is covered by — the one of five the decision has to
    pick.  ``page`` is the new space, installed as a live temptation.  The rest is what
    the ask's own terms give: ``cadence_seconds`` is how far apart the job should fire
    whatever rule spelling says so, ``anchored`` whether the terms name a time of DAY
    rather than a period (so the rule has to state an hour to run at), ``expects_expiry``
    whether they gave an end condition at all (inventing one is a failure), and ``bound``
    every value the MESSAGE supplies that the routine has to be pointed at.

    ``bound`` is KEYED BY PARAMETER NAME since #1870, because the container's name is
    derived from those values in the routine's DECLARED ORDER — so the fixture states which
    value answers which parameter and the order is read off the registry, rather than being
    an order this tuple has to be kept in and nothing could check.  A fixture drifting out
    of a positional order would derive a plausible name for the wrong job, silently; a
    fixture naming a parameter the routine does not declare fails loudly in the probe."""

    case_id: str
    ask: str
    page: CannedPage
    skill: SkillDraft
    cadence_seconds: int
    anchored: bool
    expects_expiry: bool
    bound: dict[str, str]


_COLD_PRICE = _IdleApplyCase(
    case_id="transition-idle-to-apply",
    ask=_COLD_ASK,
    page=_KEEL_LANTERN_LISTING,
    skill=_AURORA_SKILL,
    cadence_seconds=3600,
    anchored=False,
    expects_expiry=True,
    bound={"url": _KEEL_LANTERN_URL},
)

_COLD_TWO_PARAMS = _IdleApplyCase(
    case_id="transition-idle-to-apply-two-params",
    ask=_COLD_ASK_TWO_PARAMS,
    page=_NORTH_PIER_DEPARTURES,
    skill=_FERRY_SKILL,
    cadence_seconds=86400,
    anchored=True,
    expects_expiry=False,
    bound={"url": _NORTH_PIER_URL, "keyword": "dawn sailing"},
)

_COLD_DIGEST = _IdleApplyCase(
    case_id="transition-idle-to-apply-digest",
    ask=_COLD_ASK_DIGEST,
    page=_HARBOR_BAKERY_MENU,
    skill=_BAKERY_SKILL,
    cadence_seconds=86400,
    anchored=False,
    expects_expiry=False,
    bound={"url": _HARBOR_BAKERY_URL},
)

_COLD_THRESHOLD = _IdleApplyCase(
    case_id="transition-idle-to-apply-threshold",
    ask=_COLD_ASK_THRESHOLD,
    page=_RIVER_OTTERS_CENSUS,
    skill=_COLONY_SKILL,
    cadence_seconds=604800,
    anchored=False,
    expects_expiry=False,
    bound={"url": _RIVER_OTTERS_URL},
)

_COLD_URGENCY = _IdleApplyCase(
    case_id="transition-idle-to-apply-urgency",
    ask=_COLD_ASK_URGENCY,
    page=_EAST_BRANCH_NEW_TITLES,
    skill=_ARRIVALS_SKILL,
    cadence_seconds=7200,
    anchored=False,
    expects_expiry=True,
    bound={"url": _EAST_BRANCH_URL},
)

# Every cold ask, in one place — so the deterministic pin in ``test_eval_harness.py`` can
# check each one's claim about the world without a GPU.
IDLE_APPLY_CASES = (
    _COLD_PRICE,
    _COLD_TWO_PARAMS,
    _COLD_DIGEST,
    _COLD_THRESHOLD,
    _COLD_URGENCY,
)


def _no_teach_question_check(landed: StateTransition | None) -> Check:
    """She did not ask to be taught: the turn opened no round.

    Structural rather than a reading of the reply — the elicit shape IS the machine
    parking itself on the ask to be shown how, and that is a row, not a sentence.  Read off
    the move the scorer already fetched, like every other check that conditions on it."""
    asked_to_be_taught = landed is not None and landed.to_state == ConversationState.ELICIT.value
    return Check(
        "state: she did not ask to be taught (no round was opened)",
        not asked_to_be_taught,
        rationale="the turn parked in elicit" if asked_to_be_taught else None,
        kind="state",
    )


def _cold_anchor_check(db: Database, landed: StateTransition | None, case: _IdleApplyCase) -> Check:
    """The move came FROM idle and stamped the cold ask as its anchor — the anchor
    lifecycle's opening move (#1827), which is what says this turn started a round of its
    own rather than continuing one the history left open.  Same conditional-n/a as the
    decision check."""
    label = "state: the move came from idle with the ask as its anchor"
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.ask, limit=_COMPOSED_MESSAGE_WINDOW)
    opened = applied.from_state == ConversationState.IDLE.value
    anchored = applied.anchor_message_id
    ok = opened and asked is not None and anchored == asked
    return Check(
        label,
        ok,
        rationale=None
        if ok
        else f"came from {applied.from_state}, anchored to {anchored} (the ask is {asked})",
        kind="state",
    )


def _score_idle_to_apply(
    db: Database, before: set[str], reply: str, *, case: _IdleApplyCase
) -> list[Check]:
    """A routine she already knows was pointed at a space she has never seen, on the terms
    this one message gave — without teaching, without browsing, and without disturbing any
    of the five jobs already running.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are
    diff-join keys and are deliberately case-NEUTRAL: one wording reads the same whether
    the job watches a price, a timetable or a shelf."""
    row = _minted_job(db, before)
    landed = db.machine.latest_transition()
    return [
        *_cold_binding_checks(db, row, landed, case),
        *_job_terms_checks(
            db,
            row,
            fires_every=case.cadence_seconds,
            anchored=case.anchored,
            expects_expiry=case.expects_expiry,
        ),
        _seeded_jobs_untouched_check(db),
        _no_teach_question_check(landed),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        _cold_anchor_check(db, landed, case),
        *_job_setup_advisories(db, row, landed),
    ]


def _cold_binding_checks(
    db: Database, row: MemoryRow | None, landed: StateTransition | None, case: _IdleApplyCase
) -> list[Check]:
    """She set a job up, on the routine the ask is covered by, in the container derived for
    it.

    What is NOT here since #1870 is whether each VALUE was read off the message: the turn
    does not bind values any more — the binder does, before the turn begins — so scoring it
    through a chat turn would be measuring one draw through another.  That contract is
    ``test_skill_binding.py``'s, where the binder is driven on its own; what survives here
    is the derived NAME, which is a function of those values and is what the rest of the
    system identifies the job by."""
    return [
        *_job_stood_up_checks(db, row),
        _skill_binding_check(
            _landed_apply_move(landed),
            intended=slug_skill_name(case.skill.name),
            label="state: the decision bound the routine that covers the ask",
        ),
        _enactment_binding_check(row, case.skill),
        _fresh_mint_check(db, row, case.skill, case.bound),
    ]


async def _run_idle_apply_case(chat_eval: ChatEval, case: _IdleApplyCase) -> None:
    """Drive one idle → apply case: the composed world behind it, the five taught routines
    in the registry, its new page installed as a live temptation, and the shared scorer
    bound to the terms its ask gives.  Report-only — the thresholds are the code owner's
    to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.ask,
        browse=[case.page],
        seed=seed_composed_world(),
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        prepare=_probe_composed_world(case),
        score=partial(_score_idle_to_apply, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_apply_points_a_known_routine_at_a_new_listing(chat_eval: ChatEval) -> None:
    """idle → apply, cold: a second listing, long after the price watcher was taught on
    the first one.  Nothing in the ask refers back, so she has to recognise the job from
    what it asks for, bind the page it names, and set it running on the hours and the end
    it gives — without opening the listing to check."""
    await _run_idle_apply_case(chat_eval, _COLD_PRICE)


@pytest.mark.asyncio
async def test_idle_to_apply_binds_both_parameters_from_one_message(chat_eval: ChatEval) -> None:
    """idle → apply on the routine that asks for two things: the harbour page and the
    sailing to look for on it both come out of this ONE cold message, with no round behind
    it to supply either — the bind that was undecidable before the framer decided
    parameters from the ask alone."""
    await _run_idle_apply_case(chat_eval, _COLD_TWO_PARAMS)


@pytest.mark.asyncio
async def test_idle_to_apply_schedules_a_second_digest(chat_eval: ChatEval) -> None:
    """idle → apply on the daily digest, pointed at a second bakery: a plain daily cadence
    with a telling clause and NO end condition — so an expiry set here is one nobody asked
    for."""
    await _run_idle_apply_case(chat_eval, _COLD_DIGEST)


@pytest.mark.asyncio
async def test_idle_to_apply_tracks_a_different_survey(chat_eval: ChatEval) -> None:
    """idle → apply on the count watcher, pointed at a different survey: the longest
    cadence of the set, and the closest thing the registry has to a look-alike, since two
    of the five routines watch a page for a number."""
    await _run_idle_apply_case(chat_eval, _COLD_THRESHOLD)


@pytest.mark.asyncio
async def test_idle_to_apply_sets_the_tight_cadence_from_a_cold_start(
    chat_eval: ChatEval,
) -> None:
    """idle → apply under act-now pressure: another branch of the same catalogue, every
    two hours until a named day.  The urgency is a reason to set it up now, not a reason
    to go and look — and the end condition is a date the model has to work out."""
    await _run_idle_apply_case(chat_eval, _COLD_URGENCY)
