"""Chat in LEARN, entered straight from idle: the teach arrives whole.

No elicit round precedes these -- the user's first message already carries the steps, so the turn both learns and runs the round in one pass.  The world is the composed one, which knows neither the page nor the fact, so everything the round reports has to come from the demonstration it was just given.
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

from penny.tests.eval.utils.transition_ledger import _FAMILY, _SET_TOOL, _landed_state

from penny.tests.eval.utils.transition_world import _COMPOSED_MESSAGE_WINDOW, _JOURNEYS, _LIVE_JOB_CONTAINERS, _TEACH_CLIFF_WALK, _TEACH_FREE_EVENT, _TEACH_HARBOUR_FLAG, _TEACH_REHEARSAL_PIECE, _TEACH_WATERING_RULE, _TeachCase, _attaches_nothing_checks, _extraction_shape_checks, _framed_checks, _learned_this_turn, _round_framing, _round_ran_checks, _round_reported_checks, _seeded_ask_id, _seeded_jobs_untouched_check, _wrote_into_the_container_check, assert_composed_world, assert_the_registry_holds, assert_values_are_new, seed_composed_world


pytestmark = pytest.mark.eval


# ── The probe: the world is the composed one, and it knows neither page nor fact ─


def _probe_teach_world(case: _TeachCase) -> Preparer:
    """The prepare hook: the composed seeder's own claims, the registry one that is only
    true once the runner has laid the fixture skills down, and the case's own novelty
    claim."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db)
        assert_the_registry_holds(penny.db, _JOURNEYS)
        assert_the_teach_is_new_to_the_world(penny.db, case)

    return probe


def assert_the_teach_is_new_to_the_world(db: Database, case: _TeachCase) -> None:
    """The page this teach names and the fact that page holds are BOTH new to the history
    it is taught in.

    The page's novelty is what makes the round a real demonstration rather than a re-run of
    something already done.  The FACT's novelty is what makes the two checks that read it
    mean anything: the durable-write check reads only this run's own entries, but the
    SAID == DID check reads every turn of Penny's this sample — the seeded ones included —
    so a fact the seeded history already says would score green with the page unread.

    Its own premise rides along, because both ways of getting it wrong are silent on a run:
    the instructions must NAME the page (a teach whose steps point nowhere is a round
    nothing can carry out), and the fixture must be the page those instructions reach (a
    match token the url does not carry serves a no-results page, and the round then dies on
    a fixture rather than on anything the model did)."""
    assert case.url in case.teach, f"{case.case_id}: the teach must name the page it points at"
    assert case.page.match.lower() in case.url.lower(), (
        f"{case.case_id}: the fixture must answer {case.url!r}, it matches on {case.page.match!r}"
    )
    assert_values_are_new(db, case.case_id, (case.url, case.stored))


def _teach_anchor_check(db: Database, case: _TeachCase) -> Check:
    """The move came FROM idle and stamped the teaching message as the round's anchor — the
    anchor lifecycle's opening move (#1827), which here says one turn both OPENED the round
    and ran it.

    Scored ONLY when the machine landed in learn, the same conditional every other beat's
    anchor check uses: a misroute is already named by the landed-state advisory, and scoring
    the anchor on top of it would recount one classifier miss as an enactment failure."""
    label = "state: the move came from idle with the teach as its anchor"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.LEARN.value:
        return Check.na(label, kind="state")
    taught = _seeded_ask_id(db, case.teach, limit=_COMPOSED_MESSAGE_WINDOW)
    opened = latest.from_state == ConversationState.IDLE.value
    anchored = latest.anchor_message_id
    ok = opened and taught is not None and anchored == taught
    return Check(
        label,
        ok,
        rationale=None
        if ok
        else f"came from {latest.from_state}, anchored to {anchored} (the teach is {taught})",
        kind="state",
    )


def _score_idle_to_learn(
    db: Database, before: set[str], reply: str, *, case: _TeachCase
) -> list[Check]:
    """The round the message taught RAN, and nothing was set up.

    The elicit → learn contract, minus the elicit turn: the page was read, what it said
    landed in the container the entry framing built, a routine reached the registry, and
    none of it was instantiated — no skill attached, no program rendered, nothing scheduled
    or configured.  Two claims are this beat's own, and both come from teaching beside work
    that is already running: the move opened the round from IDLE, and the five live jobs are
    none of this turn's business.

    ONE scorer for all five cases, bound to the fact its own page carries.  The labels are
    diff-join keys and are shared with the beat this contract comes from, so the two learn
    entries report under the same rows."""
    created = new_collections(db, before)
    framing = _round_framing(db)
    learned = _learned_this_turn(db)
    return [
        *_round_ran_checks(db, case.stored),
        *_framed_checks(db, framing),
        _wrote_into_the_container_check(db, framing),
        Check(
            "state: a skill was learned from the round",
            bool(learned),
            kind="state",
        ),
        *_attaches_nothing_checks(db, created, already_running=_LIVE_JOB_CONTAINERS),
        Check("state: she configured nothing", tool_not_called(db, _SET_TOOL), kind="state"),
        *_extraction_shape_checks(db, learned),
        _teach_anchor_check(db, case),
        _seeded_jobs_untouched_check(db),
        *_round_reported_checks(case.stored, reply, outgoing_replies(db)),
        Check(
            "calls: the machine landed in learn",
            _landed_state(db) == ConversationState.LEARN.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def _run_teach_case(chat_eval: ChatEval, case: _TeachCase) -> None:
    """Drive one idle → learn case: the composed world behind it, the five taught routines
    in the registry, the page its instructions name installed so the demonstration reads a
    real one, and the shared scorer bound to the fact that page carries.  Report-only — the
    thresholds are the code owner's to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.teach,
        browse=[case.page],
        seed=seed_composed_world(),
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        prepare=_probe_teach_world(case),
        score=partial(_score_idle_to_learn, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_learn_runs_the_taught_round_in_one_turn(chat_eval: ChatEval) -> None:
    """idle → learn, the canonical single-turn teach: the message says it is teaching and
    then gives the three steps, so there is nothing left to elicit.  The round is framed on
    the way in, run once — the signals page read, the flag saved into the round's own
    container — and the turn ends on the offer with nothing set running."""
    await _run_teach_case(chat_eval, _TEACH_HARBOUR_FLAG)


@pytest.mark.asyncio
async def test_idle_to_learn_follows_a_numbered_list_of_steps(chat_eval: ChatEval) -> None:
    """idle → learn where the steps arrive as a NUMBERED list rather than a sentence: the
    same open / find / remember round, written the way a person writes a procedure, so what
    is measured is the round and not the prose it came in."""
    await _run_teach_case(chat_eval, _TEACH_CLIFF_WALK)


@pytest.mark.asyncio
async def test_idle_to_learn_keeps_only_what_the_filter_asks_for(chat_eval: ChatEval) -> None:
    """idle → learn with a FILTER in the steps: the events page lists three things on and
    the instruction keeps the one marked free, so the demonstrated round has to read past
    two it was not sent for and store the one it was."""
    await _run_teach_case(chat_eval, _TEACH_FREE_EVENT)


@pytest.mark.asyncio
async def test_idle_to_learn_defers_the_notify_condition_to_the_offer(
    chat_eval: ChatEval,
) -> None:
    """idle → learn where the teach also states a NOTIFY condition: the watering
    restriction is read and saved, and "tell me if it changes to a total ban" is left for
    the turn that accepts the offer.  Configuring it here is the teach-and-instantiate fold
    the machine exists to split."""
    await _run_teach_case(chat_eval, _TEACH_WATERING_RULE)


@pytest.mark.asyncio
async def test_idle_to_learn_learns_a_new_routine_from_a_here_s_how(
    chat_eval: ChatEval,
) -> None:
    """idle → learn from a "new routine for you, here's how it works" opening: the word
    routine is the USER's, said while five routines are already running, and the turn's job
    is still the one demonstration — fetch the board, take this week's piece, store it."""
    await _run_teach_case(chat_eval, _TEACH_REHEARSAL_PIECE)
