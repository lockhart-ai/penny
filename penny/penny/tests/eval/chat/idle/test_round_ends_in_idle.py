"""Chat in IDLE, entered from a parked round: the round ends here.

Every parked state bails back to idle when the user drops the task -- elicit mid-question, learn mid-demonstration, request mid-binding (including one where the binding was half settled).  The turn answers the new subject and leaves nothing half-built behind it.  The fifth case is the floor: ordinary banter in idle, which fires nothing at all.
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

from penny.tests.eval.utils.transition_ledger import _FAMILY, _SET_TOOL

from penny.tests.eval.utils.transition_world import _AURORA_APPLY, _AURORA_ROUND, _ApplyCase, _DECOY_SKILL, _ElicitRound, _JOBS_UNTOUCHED_LABEL, _JOURNEYS, _Journey, _RequestApplyCase, _SUPPLIED_PIER, _SUPPLIED_SPACES, _SUPPLIED_TIMETABLE, _assert_parked_on_the_ask, _assert_seeded_world, _seed_elicit_round, _seeded_ask_id, _seeded_jobs_untouched_check, assert_composed_world, assert_parked_in_request_world, assert_round_cites_its_run, assert_round_is_framed, assert_seeded_ledger, seed_composed_world, seed_learned_round, seed_parked_in_request


pytestmark = pytest.mark.eval


# ── Every parked state bails back to idle, and the round ends there ───────────
#
# Beat 7 (#1896) — the transition matrix's last uncovered edges, scoped by the code owner
# as ONE go over every abandonment: "all bail back to idle preserves nothing, once in idle
# any new user task starts a new flow anyways, i don't see any reason to preserve
# intermediate state."
#
# So there is one contract and five worlds to answer it in.  The contract is that the round
# ENDS: the landing carries no anchor, no framing and no partial binding, nothing is built,
# nothing is configured, nothing is registered, and the mechanisms already running are none
# of the turn's business.  The worlds are the states a round can be parked in, each read
# from the beat that already defines it rather than restated here — parked in elicit having
# been asked to be taught, parked mid-teach with the round's container built, parked in
# request on a page that was never named, parked in request holding a value that WAS, and
# not parked at all.
#
# One runtime change rides with it, and case 2 is what measures it: a learn round builds its
# container on the way in, and nothing retired it when the round was walked away from — an
# inert collection named for a job nobody is doing, reachable by nothing.  The idle landing
# now archives it (archived, never deleted, so the same job taught again revives it).  Every
# other parked state built nothing, so there is nothing for its bail to retire.
#
# The last case is the NO-FIRE row: ordinary banter arriving on an idle machine with five
# live jobs behind it, where the failure would be firing anything at all.
#
# Everything else is deliberately LOOSE.  An idle turn is ordinary chat with the full tool
# surface, so answering the banter well — including going and looking something up when the
# message carries a real question — is not a miss, and the pages are installed so that a
# browse SUCCEEDS rather than failing invisibly.  The reference replies are review targets,
# never scorer strings.


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

    ``container`` is what the round BUILT, and therefore what the landing has to retire —
    ``None`` for every state that built nothing, which is all of them but learn (a job short
    of a value has no name yet, and an ask that has not been taught has no round).
    ``journeys`` is the live jobs the world left running, so "nothing in flight was touched"
    is read against the world the case actually seeded rather than a fixed five."""

    seed: Seeder
    skills: tuple[SkillDraft, ...]
    seeded: Callable[[Database], None]
    journeys: tuple[_Journey, ...]
    container: str | None


class _BailCase(NamedTuple):
    """One agreed bail, and the world it lands in.

    ``bail`` is the message under test and ``pages`` the spaces a turn is free to reach for
    — a temptation only in the sense that a browse here is ALLOWED, so what they buy is that
    a turn which does look up finds something instead of failing on a thin fixture.

    ``reference`` is how the bail would be answered WELL — a review target, never matched by
    the scorer.  It is DATA for the reason every other beat's is: a scorer that cannot pass
    the answer the case itself calls correct is a broken scorer, and holding the reply here
    lets the deterministic pin run the one reply check through it without a GPU."""

    case_id: str
    world: _BailWorld
    bail: str
    pages: tuple[CannedPage, ...]
    reference: str


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
    journeys=(),
    container=None,
)

_PARKED_IN_LEARN = _BailWorld(
    seed=seed_learned_round(_AURORA_APPLY),
    skills=(_AURORA_APPLY.skill, _DECOY_SKILL),
    seeded=partial(_assert_parked_in_learn, case=_AURORA_APPLY),
    journeys=(),
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
        journeys=case.parked.journeys,
        container=None,
    )


_UNPARKED = _BailWorld(
    seed=seed_composed_world(),
    skills=tuple(journey.round.skill for journey in _JOURNEYS),
    seeded=assert_composed_world,
    journeys=_JOURNEYS,
    container=None,
)


_BAIL_FROM_ELICIT = _BailCase(
    case_id="transition-elicit-to-idle",
    world=_PARKED_IN_ELICIT,
    bail="ah never mind, forget that — anything good at the harbor market this weekend?",
    pages=(AURORA_LISTING_499, _HARBOR_MARKET),
    reference=(
        "no worries, dropped it. the harbour market's on both mornings this weekend — "
        "smoked fish, a cider stall, and the pottery co-op's summer seconds."
    ),
)

_BAIL_FROM_LEARN = _BailCase(
    case_id="transition-learn-to-idle",
    world=_PARKED_IN_LEARN,
    bail="actually forget it, i don't need this",
    pages=(AURORA_LISTING_499,),
    reference="no problem — i've dropped it. shout if you want to pick it up again.",
)

_BAIL_FROM_REQUEST = _BailCase(
    case_id="transition-request-to-idle",
    world=_parked_in_request(_SUPPLIED_TIMETABLE),
    bail="eh never mind, it's not important",
    pages=tuple(_SUPPLIED_SPACES),
    reference="all good, leaving it. just say the word if you change your mind.",
)

_BAIL_FROM_HELD_BINDING = _BailCase(
    case_id="transition-request-to-idle-held-binding",
    world=_parked_in_request(_SUPPLIED_PIER),
    bail="you know what, skip it",
    pages=tuple(_SUPPLIED_SPACES),
    reference="sure thing — skipping it.",
)

_BANTER_ON_IDLE = _BailCase(
    case_id="transition-idle-to-idle",
    world=_UNPARKED,
    bail="the ferry ride this morning was gorgeous btw",
    pages=tuple(_SUPPLIED_SPACES),
    reference="lovely — the light on the water first thing is hard to beat.",
)

# Every bail, in one place — so the deterministic pin in ``test_eval_harness.py`` can drive
# each one's seeder and run its own reply check without a GPU.
BAIL_CASES = (
    _BAIL_FROM_ELICIT,
    _BAIL_FROM_LEARN,
    _BAIL_FROM_REQUEST,
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
    expected = sorted(slug_skill_name(draft.name) for draft in case.world.skills)
    assert taught == expected, f"{case.case_id}: the registry must hold {expected}, got {taught}"


def assert_the_round_built_what_it_claims(db: Database, case: _BailCase) -> None:
    """The container premise, both ways round: a world whose round built one starts with it
    LIVE (so "it was archived" is a claim about this turn and not a row that arrived
    retired), and a world whose round built nothing carries no framing to retire.

    Silent either way on a run: a container already archived would score the beat's headline
    green for free, and a framing nobody noticed would make the n/a a real miss."""
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


# ── Scoring ───────────────────────────────────────────────────────────────────


# What a reply CLAIMING a job is now running says.  Deliberately NARROW: this is the one
# reply failure with a structural reading — she cannot truthfully say something is set up
# when nothing was configured — and a wide vocabulary would fail ordinary conversation
# instead, since "i'll let you know" is a sentence a person says about anything.  Each entry
# names the MECHANISM as live and is written in a form a denial cannot produce ("i haven't
# set that up" carries neither "i've set" nor "it's set up"), so a reply saying the opposite
# is never read as the claim.  A floor: how well the bail was answered is read at review.
_CLAIMS_A_JOB_IS_RUNNING = (
    "i've set",
    "i have set",
    "it's set up",
    "that's set up",
    "is now set up",
    "it's running",
    "that's running",
    "is now running",
    "i'll keep checking",
    "i'll keep an eye on it",
    "i'll check it every",
    "i'll check that every",
    "i'll let you know every",
    "i'll let you know each",
)


def _landed_in_idle_check(landed: StateTransition | None) -> Check:
    """The beat's headline: the bail put the machine back in idle.

    Structural, off the move the turn recorded — where a turn ended up is a row, never a
    reading of the reply — and the only check that reads the landing, so one miss is one
    finding.  The rationale names where it went instead: learn or elicit means the bail was
    read as more of the round, request means it was read as an answer to what was asked
    for, and apply means it was read as a go-ahead."""
    to_state = landed.to_state if landed is not None else None
    idle = to_state == ConversationState.IDLE.value
    return Check(
        "state: the turn landed in idle",
        idle,
        rationale=None if idle else f"the machine landed in {to_state}",
        kind="state",
    )


def _round_ended_check(landed: StateTransition | None) -> Check:
    """The landing carries nothing of the round: no anchor, no framing, no partial binding.

    All three in one check because they are one fact — the round is over — and because a
    landing that kept any of them is the same defect read three ways: the next message would
    be classified against a task nobody is doing any more."""
    label = "state: the landing carries nothing of the round"
    if landed is None:
        return Check(label, False, rationale="the turn recorded no move at all", kind="state")
    kept = {
        "anchor": landed.anchor_message_id,
        "framing": landed.skill_frame,
        "binding": landed.round_shortfall,
    }
    carried = {name: value for name, value in kept.items() if value is not None}
    return Check(
        label,
        not carried,
        rationale=f"still carries {sorted(carried)}" if carried else None,
        kind="state",
    )


def _container_retired_check(db: Database, case: _BailCase) -> Check:
    """The round's container was ARCHIVED — the one thing a bail has to clean up, and only
    where the round built one.

    N/A rather than a free pass everywhere else: a round short of a value has no derived
    name to build under and an ask that was never taught has no round at all, so there is
    genuinely nothing to retire — which is a real shape, not an exemption.

    Archived, never deleted: the row stays a visible tombstone, so a bail drawn off a flaky
    classification is recoverable and the same job taught again revives it."""
    label = "state: the round's container was archived"
    container = case.world.container
    if container is None:
        return Check.na(label, kind="state")
    row = db.memories.get(container)
    retired = row is not None and row.archived
    return Check(
        label,
        retired,
        rationale=None if retired else f"{container!r} is {'still live' if row else 'gone'}",
        kind="state",
    )


def _registry_unchanged_check(db: Database, case: _BailCase) -> Check:
    """Nothing was REGISTERED: the routines the world taught are the routines it still has.

    The claim the mid-teach bail exists to make — a round walked away from teaches nothing,
    whatever it demonstrated — and one every other case makes too, since a bail that minted a
    routine would leave the user with a mechanism they had just called off."""
    taught = sorted(skill.name for skill in db.skills.list_all())
    expected = sorted(slug_skill_name(draft.name) for draft in case.world.skills)
    return Check(
        "state: nothing was registered (the routines are the ones it already had)",
        taught == expected,
        rationale=None if taught == expected else f"the registry holds {taught}",
        kind="state",
    )


def _nothing_was_built_check(db: Database, before: set[str]) -> Check:
    """No collection was created — not an inert one, not a configured one, none.  A bail
    ends the round, and a container built on the way out is a job nobody asked for."""
    created = [row.name for row in new_collections(db, before)]
    return Check(
        "state: nothing was created",
        not created,
        rationale=f"created {created}" if created else None,
        kind="state",
    )


def _in_flight_untouched_check(db: Database, case: _BailCase) -> Check:
    """The mechanisms already running are none of this turn's business — the shared reading,
    against the jobs THIS world left running.

    N/A for a world that holds none (the two rounds that never reached an apply turn), so a
    world with nothing to leave alone reports that rather than passing for free."""
    if not case.world.journeys:
        return Check.na(_JOBS_UNTOUCHED_LABEL, kind="state")
    return _seeded_jobs_untouched_check(db, case.world.journeys)


def _claims_no_job_check(reply: str) -> Check:
    """The reply does not say a job is now set up or running — the one reply failure with a
    structural reading, since nothing was configured and saying otherwise is a claim the
    record contradicts (the visible-degradation rule, applied to what she tells the user).

    A FLOOR, deliberately narrow: everything else about the answer is read at joint review
    against the reference reply, and an idle turn answering banter well is not a miss."""
    claimed = [phrase for phrase in _CLAIMS_A_JOB_IS_RUNNING if phrase in reply.lower()]
    return Check(
        "reply: it claims no job was set up",
        not claimed,
        rationale=f"said {claimed}" if claimed else None,
        kind="reply",
    )


def _bail_advisories(db: Database, landed: StateTransition | None, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — where it left the machine and
    what it said — so a report shows the answer whichever way it went and the wording is
    read where wording is read: at review."""
    return [
        Check(
            f"landed in {landed.to_state if landed is not None else None}",
            True,
            kind="state",
            scored=False,
        ),
        Check(f"answered: {reply!r}", True, kind="reply", scored=False),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


def _score_bail_to_idle(
    db: Database, before: set[str], reply: str, *, case: _BailCase
) -> list[Check]:
    """The round ENDED: the machine is back in idle, the landing carries nothing of it, the
    container it built is retired, and nothing was created, configured or registered.

    ONE scorer for all five cases, bound to the world each is answered in.  The labels are
    diff-join keys and are deliberately case-NEUTRAL: one wording reads the same whether the
    abandoned round was a teach loop, a negotiation, or no round at all.

    Everything the turn is FREE to do is absent on purpose — there is no no-browse check
    here, because an idle turn has the full tool surface and going and answering the
    question the bail changes the subject to is an ordinary reply, not a miss."""
    landed = db.machine.latest_transition()
    return [
        _landed_in_idle_check(landed),
        _round_ended_check(landed),
        _nothing_was_built_check(db, before),
        Check("state: she configured nothing", tool_not_called(db, _SET_TOOL), kind="state"),
        _registry_unchanged_check(db, case),
        _container_retired_check(db, case),
        _in_flight_untouched_check(db, case),
        _claims_no_job_check(reply),
        *_bail_advisories(db, landed, reply),
    ]


async def _run_bail_case(chat_eval: ChatEval, case: _BailCase) -> None:
    """Drive one bail case: the parked world its own edge was measured against, exactly the
    routines that world taught, the spaces an idle turn may reach for installed so a lookup
    finds something, and the shared scorer bound to the world.  Report-only — the thresholds
    are the code owner's to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.bail,
        browse=list(case.pages),
        seed=case.world.seed,
        seed_skills=list(case.world.skills),
        prepare=_probe_bail_world(case),
        score=partial(_score_bail_to_idle, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_elicit_to_idle_drops_the_task_and_answers_the_new_one(chat_eval: ChatEval) -> None:
    """elicit → idle: parked having asked to be taught, the user calls it off and changes
    the subject in the same breath.  Nothing was ever built for that round, so the whole
    contract is that the machine lets go of it — and the new question is answered as
    ordinary conversation, which an idle turn has the full tool surface for."""
    await _run_bail_case(chat_eval, _BAIL_FROM_ELICIT)


@pytest.mark.asyncio
async def test_learn_to_idle_archives_the_abandoned_round(chat_eval: ChatEval) -> None:
    """learn → idle: parked mid-teach with the round's container built and the demonstrated
    value in it, the user drops the whole thing.  The container goes with the round — the
    one bail that has something to clean up — nothing is registered from what was
    demonstrated, and the reply acknowledges it rather than claiming a job is running."""
    await _run_bail_case(chat_eval, _BAIL_FROM_LEARN)


@pytest.mark.asyncio
async def test_request_to_idle_drops_what_the_round_was_waiting_on(chat_eval: ChatEval) -> None:
    """request → idle: parked waiting for the page the ask never named, the user calls it
    off.  The binding the round was waiting on goes with it — a partial binding kept past
    the bail is state describing a negotiation that is over — and the five jobs already
    running are untouched."""
    await _run_bail_case(chat_eval, _BAIL_FROM_REQUEST)


@pytest.mark.asyncio
async def test_request_to_idle_drops_a_binding_that_was_half_settled(chat_eval: ChatEval) -> None:
    """request → idle from the held-binding side: the round had already settled the page and
    was waiting only on what to watch for, so this is the bail with the most to preserve and
    the ruling is that it preserves none of it."""
    await _run_bail_case(chat_eval, _BAIL_FROM_HELD_BINDING)


@pytest.mark.asyncio
async def test_idle_to_idle_fires_nothing_on_ordinary_banter(chat_eval: ChatEval) -> None:
    """idle → idle, the no-fire row: five live jobs behind her, an idle machine, and a
    remark that asks for nothing.  Every check here is a negative — nothing created,
    configured or registered, and none of the running jobs touched — because the failure
    this case exists to catch is firing anything at all."""
    await _run_bail_case(chat_eval, _BANTER_ON_IDLE)
