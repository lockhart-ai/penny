"""Chat in APPLY, entered from learn: the offer accepted, the routine set running.

The round was demonstrated and the routine minted; the user says yes.  The turn stands the job up on the round's own container -- parameters bound from the demonstration, the cadence set from what the ask asked for, and an end condition composed where one was named.
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

from penny.tests.eval.utils.transition_ledger import _BROWSE_TOOL, _FAMILY, _SET_TOOL

from penny.tests.eval.utils.transition_world import _APPLY_ROUND_INCOMING_TURNS, _ARRIVALS_APPLY, _AURORA_APPLY, _ApplyCase, _BAKERY_APPLY, _COLONY_APPLY, _DECOY_SKILL, _FERRY_APPLY, _assert_parked_on_the_ask, _bound_parameters_check, _expiry_check, _job_setup_advisories, _landed_apply_move, _schedule_check, _seeded_ask_id, _skill_binding_check, assert_round_cites_its_run, assert_round_is_framed, assert_seeded_ledger, seed_learned_round


pytestmark = pytest.mark.eval


def _probe_seeded_world(case: _ApplyCase) -> Preparer:
    """The loud seed probe, run once the world is WHOLE.

    It is a prepare hook rather than part of the seeder because the runner seeds fixture
    skills and installs the browse AFTER the case's own seed — so "exactly two skills"
    and "no page fetched" are only true here.  A seed that has drifted from the state
    the preceding beats are measured against makes these cases turns answered against a
    world nothing produces, so it fails HERE rather than as a puzzling number after an
    hour of GPU time."""

    def probe(penny: Penny) -> None:
        _assert_parked_on_the_ask(penny.db, case)
        assert_round_is_framed(penny.db, case)
        _assert_seeded_registry(penny.db, case)
        assert_seeded_ledger(penny.db, case)
        assert_round_cites_its_run(penny.db, case)

    return probe


def _assert_seeded_registry(db: Database, case: _ApplyCase) -> None:
    """The other half of that probe: two skills and only two — the scenario's own and
    the decoy — the demonstrated fact in the collection it was written to, and nothing
    instantiated anywhere.  (What the round DID read is asserted next, not here: this
    beat starts from a page already fetched.)"""
    taught = sorted(skill.name for skill in db.skills.list_all())
    assert taught == sorted([case.skill.name, _DECOY_SKILL.name]), (
        f"{case.case_id}: the registry must hold the scenario's skill and the decoy, got {taught}"
    )
    stored = collection_entries(db, case.demonstrated.collection)
    assert stored.get(case.demonstrated.entry_key) == case.demonstrated.entry_value, (
        f"{case.case_id}: the demonstrated fact must be in the collection, got {stored}"
    )
    instantiated = [row.name for row in db.memories.list_all() if row.skill_name is not None]
    assert not instantiated, f"{case.case_id}: nothing is instantiated yet, found {instantiated}"


def _instantiated(db: Database, case: _ApplyCase) -> MemoryRow | None:
    """The collection the taught skill was applied to — WHICHEVER one carries it.

    Still read by the skill rather than by the container's name, even though #1869 makes
    the landing structural: reading the row that carries the routine is what lets a job
    that landed somewhere else be SEEN (by the container check below) instead of scoring
    as no job at all.  Every other check reads whatever this returns, so a misplaced job
    is one finding rather than a whole failed sample."""
    taught = slug_skill_name(case.skill.name)
    applied = [row for row in db.memories.list_all() if row.skill_name == taught]
    return applied[0] if applied else None


def _apply_anchor_check(db: Database, landed: StateTransition | None, case: _ApplyCase) -> Check:
    """The anchor was CARRIED: the move that landed the machine in apply still points at
    the ask that opened the round (#1827) — three messages later, the turn is still
    classified against what was asked for.  Same conditional-n/a as the binding check."""
    label = "state: the anchor was carried (still the ask that opened the round)"
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.prior.ask, limit=_APPLY_ROUND_INCOMING_TURNS)
    anchored = applied.anchor_message_id
    carried = asked is not None and anchored == asked
    return Check(
        label,
        carried,
        rationale=None if carried else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


def _decoy_check(db: Database) -> Check:
    """The decoy was left alone — no collection carries it.  Kept separate from the
    intended-skill checks so a report distinguishes "bound the wrong routine" from
    "bound nothing at all"."""
    slug = slug_skill_name(_DECOY_SKILL.name)
    applied = [row.name for row in db.memories.list_all() if row.skill_name == slug]
    return Check(
        "state: the decoy was not applied",
        not applied,
        rationale=f"applied to {applied}" if applied else None,
        kind="state",
    )


def _score_learn_to_apply(
    db: Database, before: set[str], reply: str, *, case: _ApplyCase
) -> list[Check]:
    """The taught routine became a live job on the terms they gave — the intended skill
    bound onto the round's own container, its program rendered, pointed at what the round
    settled, scheduled and notifying — without re-running the round to answer.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are
    diff-join keys, so they read identically on every case and keep the wording the
    auction script gave them even where a ferry timetable is what the job watches.

    Since #1869 the split between what the TURN decides and what the ROUND settled is the
    split between the terms checks and the binding ones: the cadence, the end condition and
    the telling-them clause are the model's answers to this acceptance, while the
    container, the routine and its values are read off the round — so those read as
    certainties, and a failure in one is a defect in the mechanism rather than a draw."""
    row = _instantiated(db, case)
    landed = db.machine.latest_transition()
    return [
        *_binding_checks(db, before, row, landed, case),
        *_terms_checks(db, row, case),
        _decoy_check(db),
        _apply_anchor_check(db, landed, case),
        Check(
            "reply: she says what will happen now, naming the cadence",
            any(token in reply.lower() for token in case.cadence_tokens),
            kind="reply",
        ),
        *_job_setup_advisories(db, row, landed),
    ]


def _binding_checks(
    db: Database,
    before: set[str],
    row: MemoryRow | None,
    landed: StateTransition | None,
    case: _ApplyCase,
) -> list[Check]:
    """She set a job up, on the right routine, on the round's own container, pointed at
    what the round settled.

    Every one of these is a CERTAINTY since #1869 rather than a draw the model could get
    wrong: the container, the routine and the values come out of the round's framing at the
    call.  They stay scored because that is exactly what makes them worth reading — the
    mechanism either supplied them or it did not, and a red here names which half broke."""
    return [
        Check(
            "state: she set the job up with collection_set",
            tool_was_called(db, _SET_TOOL),
            kind="state",
        ),
        _container_check(db, before, row, case),
        Check(
            "state: the skill's program was rendered into it",
            row is not None and bool(row.extraction_prompt),
            kind="state",
        ),
        _skill_binding_check(
            _landed_apply_move(landed),
            intended=slug_skill_name(case.skill.name),
            label="state: the decision bound the intended skill",
        ),
        _bound_parameters_check(
            row,
            wanted=case.bound,
            label="state: the routine is pointed at what the round settled",
        ),
    ]


def _terms_checks(db: Database, row: MemoryRow | None, case: _ApplyCase) -> list[Check]:
    """The job runs on the terms the acceptance gave — its cadence, its end condition
    (or the absence of one), and the telling-them clause every one of these asks
    carries — and she set it running instead of running it again."""
    return [
        _schedule_check(row, fires_every=case.cadence_seconds, anchored=case.anchored),
        _expiry_check(db, row, expected=case.expects_expiry),
        Check(
            "state: it will tell them when the price moves",
            row is not None and bool(row.notify),
            kind="state",
        ),
        Check(
            "state: she set it running instead of running it again (no browse this turn)",
            tool_not_called(db, _BROWSE_TOOL),
            kind="state",
        ),
    ]


def _container_check(
    db: Database, before: set[str], row: MemoryRow | None, case: _ApplyCase
) -> Check:
    """The job landed on the round's own container — SCORED since #1869, where it was an
    advisory before.

    The collection-management question it used to carry ("does she reuse or mint?") is no
    longer a question this turn answers: the container was built when the round was framed,
    the APPLY instruction renders its name, and the configuration is aimed at it
    framework-side.  So this reads as a certainty and a failure here is a real defect in
    the mechanism rather than the standing spread-across-collections tendency the code
    owner parked.  Any collection the turn created anyway rides in the rationale, because
    that is what a broken one would look like."""
    created = new_collections(db, before)
    landed = row is not None and row.name == case.framing.container
    return Check(
        "state: the job landed on the round's own container",
        landed,
        rationale=(
            None
            if landed
            else (
                f"applied to {row.name if row else None}, expected {case.framing.container!r}, "
                f"created {[each.name for each in created]}"
            )
        ),
        kind="state",
    )


async def _run_apply_case(chat_eval: ChatEval, case: _ApplyCase) -> None:
    """Drive one learn → apply case: parked in learn on its own ask with the whole round
    behind it, its skill and the decoy in the registry, its page installed as a live
    temptation, and the shared scorer bound to the terms its acceptance gives.
    Report-only — the thresholds are the code owner's to set once the numbers are
    read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.acceptance,
        browse=[case.prior.page],
        seed=seed_learned_round(case),
        seed_skills=[case.skill, _DECOY_SKILL],
        prepare=_probe_seeded_world(case),
        score=partial(_score_learn_to_apply, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_learn_to_apply_instantiates_the_taught_skill(chat_eval: ChatEval) -> None:
    """learn → apply: parked on the offer the demonstrated round ended with, the
    user accepts and adds the job's terms.  One `collection_set` stands the job up on the
    container that round built — the routine and the page it watches come off the round
    rather than being worked out again — and she does NOT re-run the round to answer."""
    await _run_apply_case(chat_eval, _AURORA_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_sets_a_cron_cadence_and_binds_both_parameters(
    chat_eval: ChatEval,
) -> None:
    """learn → apply on a routine that asks for two things and a cadence stated as a
    time of day: "every morning" has to state an hour to run at, and the job carries BOTH
    of the round's values — the timetable's address and the late sailing line — neither of
    which is in the acceptance."""
    await _run_apply_case(chat_eval, _FERRY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_schedules_the_daily_digest(chat_eval: ChatEval) -> None:
    """learn → apply on the store-each-day digest: a plain daily cadence with a telling
    clause and NO end condition — so an expiry set here is one nobody asked for."""
    await _run_apply_case(chat_eval, _BAKERY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_schedules_the_weekly_check(chat_eval: ChatEval) -> None:
    """learn → apply on the weekly count: the longest cadence of the set, on a job carrying
    the scheme-less address the user typed rather than a page she went and found."""
    await _run_apply_case(chat_eval, _COLONY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_sets_the_tight_cadence_and_works_out_the_end(
    chat_eval: ChatEval,
) -> None:
    """learn → apply under the act-now ask: a two-hourly cadence and an end condition
    stated in words — "until the end of the month" is a datetime the model has to work
    out, so what matters here is that it set one at all."""
    await _run_apply_case(chat_eval, _ARRIVALS_APPLY)
