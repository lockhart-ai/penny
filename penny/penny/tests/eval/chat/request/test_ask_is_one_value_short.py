"""Chat in REQUEST: the routine is known, and the ask is one value short.

Each ask names a routine the world already has and asks for it to be run somewhere new -- but leaves out exactly one value its interface requires.  The turn asks for that value and stands nothing up: no guessed binding, no job created on a half-settled interface, and no re-teaching of a routine that already exists.
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

from penny.tests.eval.utils.transition_world import _COMPOSED_MESSAGE_WINDOW, _IdleRequestCase, _SHORT_BAKERY, _SHORT_COUNT, _SHORT_LISTING, _SHORT_PIER, _SHORT_TIMETABLE, _UNKNOWN_SPACES, _landed_in, _mentions_any, _said_back, _seeded_ask_id, _seeded_jobs_untouched_check, _skill_binding_check, assert_composed_world, assert_the_ask_falls_one_short, assert_the_registry_holds, assert_values_are_new, seed_composed_world


pytestmark = pytest.mark.eval


# ── The probe: the ask really is one value short of the routine it names ──────


def _probe_short_ask(case: _IdleRequestCase) -> Preparer:
    """The prepare hook: the shared world's own claims, the registry one that is only true
    once the runner has laid the fixture skills down, and this case's own two."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db, case.journeys)
        assert_the_registry_holds(penny.db, case.journeys)
        assert_the_ask_falls_one_short(penny.db, case)
        assert_values_are_new(penny.db, case.case_id, case.settled.values())

    return probe


# ── Scoring ───────────────────────────────────────────────────────────────────


def _parked_in_request_check(landed: StateTransition | None) -> Check:
    """The beat's headline: the turn left the machine parked in REQUEST.

    Structural, off the move the turn recorded — where a turn ended up is a row, never a
    reading of the reply.  Every other landing is a distinct finding and the rationale
    names which one: apply means the binder filled a value the ask never gave, elicit means
    the covering routine was not recognised, and idle means the ask was answered as chat."""
    to_state = landed.to_state if landed is not None else None
    parked = to_state == ConversationState.REQUEST.value
    return Check(
        "state: the turn parked in request",
        parked,
        rationale=None if parked else f"the machine landed in {to_state}",
        kind="state",
    )


def _nothing_was_created_check(db: Database, before: set[str]) -> Check:
    """No collection was created — not an inert one, not a configured one, none.

    The container's name is derived from the routine plus EVERY value it is pointed at, so
    a job short of one has no name yet; anything built here would be built under a name
    nothing could derive again, which is a job the user can never be handed back."""
    created = [row.name for row in new_collections(db, before)]
    return Check(
        "state: nothing was created for a job that is not settled yet",
        not created,
        rationale=f"created {created}" if created else None,
        kind="state",
    )


def _request_anchor_check(
    db: Database, landed: StateTransition | None, case: _IdleRequestCase
) -> Check:
    """The move came FROM idle and stamped the short ask as its anchor — the anchor
    lifecycle's opening move (#1827), and the thing the follow-on turn is bound over: the
    next message is read together with THIS ask, so an unanchored round is one whose reply
    arrives with nothing to complete.

    Conditional on the landing, like every other check that reads the move: where it went
    is already the parked-in-request finding, and re-reading it here would count one miss
    twice."""
    label = "state: the move came from idle with the short ask as its anchor"
    requested = _landed_in(landed, ConversationState.REQUEST)
    if requested is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.ask, limit=_COMPOSED_MESSAGE_WINDOW)
    opened = requested.from_state == ConversationState.IDLE.value
    anchored = requested.anchor_message_id
    ok = opened and asked is not None and anchored == asked
    return Check(
        label,
        ok,
        rationale=None
        if ok
        else f"came from {requested.from_state}, anchored to {anchored} (the ask is {asked})",
        kind="state",
    )


def _asks_for_what_is_missing_check(reply: str, case: _IdleRequestCase) -> Check:
    """The reply NAMES the missing piece, in words a person would use for it.

    A floor rather than a proof: it says the ask was made at all, and how well it was
    worded is read at joint review against the reference reply.  The page vocabulary is
    the framer suite's own agreed set, imported rather than restated."""
    named = _mentions_any(case.asks_for, reply)
    return Check(
        "reply: it asked for the piece that is missing",
        named,
        rationale=None if named else f"named none of {list(case.asks_for)}",
        kind="reply",
    )


def _does_not_re_ask_check(reply: str, case: _IdleRequestCase) -> Check:
    """It did not ask again for something the user had already given: every value the ask
    SETTLED comes back in the reply, so the turn is demonstrably working from it.

    The positive form on purpose.  "Did it ask for the page again?" has no honest
    structural reading — a reply may name a page while asking about something else
    entirely — while "did it say the page back" does: a turn that repeats what it was
    given is a turn that has it, and one that never mentions it is the shape a re-ask
    arrives in.  N/A for an ask that settled nothing, which is a real shape rather than a
    free pass."""
    label = "reply: it worked from what they had already given"
    if not case.settled:
        return Check.na(label, kind="reply")
    absent = [value for value in case.settled.values() if not _said_back(value, reply)]
    return Check(
        label,
        not absent,
        rationale=f"never said back: {absent}" if absent else None,
        kind="reply",
    )


def _request_advisories(landed: StateTransition | None, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — where it left the machine, which
    routine the decision bound, and the reply itself, so a report shows the answer whichever
    way it went and the wording is read where wording is read: at review.

    Read off the move the scorer already fetched rather than re-reading the ledger, so an
    advisory can never disagree with the check beside it about where the turn ended up."""
    return [
        Check(
            f"landed in {landed.to_state if landed is not None else None}",
            True,
            kind="state",
            scored=False,
        ),
        Check(
            f"the decision bound {landed.skill_name if landed is not None else None!r}",
            True,
            kind="state",
            scored=False,
        ),
        Check(f"asked: {reply!r}", True, kind="reply", scored=False),
    ]


def _score_idle_to_request(
    db: Database, before: set[str], reply: str, *, case: _IdleRequestCase
) -> list[Check]:
    """A routine she already knows covers the ask, and the ask is one value short of it —
    so the turn asks for that value and does nothing else.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are
    diff-join keys and are deliberately case-NEUTRAL: one wording reads the same whether
    the missing piece is a page or the thing to watch for on it."""
    landed = db.machine.latest_transition()
    return [
        _parked_in_request_check(landed),
        _skill_binding_check(
            _landed_in(landed, ConversationState.REQUEST),
            intended=slug_skill_name(case.skill.name),
            label="state: the decision bound the routine that covers the ask",
        ),
        _nothing_was_created_check(db, before),
        _seeded_jobs_untouched_check(db, case.journeys),
        Check(
            "state: she asked instead of going to look (no browse this turn)",
            tool_not_called(db, _BROWSE_TOOL),
            kind="state",
        ),
        Check("state: she configured nothing", tool_not_called(db, _SET_TOOL), kind="state"),
        _request_anchor_check(db, landed, case),
        _asks_for_what_is_missing_check(reply, case),
        _does_not_re_ask_check(reply, case),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        *_request_advisories(landed, reply),
    ]


async def _run_idle_request_case(chat_eval: ChatEval, case: _IdleRequestCase) -> None:
    """Drive one idle → request case: the world its own journeys compose, exactly those
    routines in the registry, every unknown space installed as a live temptation, and the
    shared scorer bound to its own terms.  Report-only — the thresholds are the code
    owner's to set once the numbers are read.

    The seeded world and the seeded REGISTRY come from one list, so a case can never be
    answered against a history whose routines the registry does not hold, or a registry
    holding one its history never taught."""
    await chat_eval(
        case_id=case.case_id,
        message=case.ask,
        browse=_UNKNOWN_SPACES,
        seed=seed_composed_world(case.journeys),
        seed_skills=[journey.round.skill for journey in case.journeys],
        prepare=_probe_short_ask(case),
        score=partial(_score_idle_to_request, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_request_asks_which_timetable(chat_eval: ChatEval) -> None:
    """idle → request: a second timetable, with the sailing to watch for but no page.  The
    routine's other parameter binds off the ask, so what is missing is one thing and the
    reply must ask for that one thing."""
    await _run_idle_request_case(chat_eval, _SHORT_TIMETABLE)


@pytest.mark.asyncio
async def test_idle_to_request_asks_for_the_listing(chat_eval: ChatEval) -> None:
    """idle → request on the price watcher: the cadence and the end date are both given and
    the listing itself is not, which is the ask that most looks complete enough to act
    on."""
    await _run_idle_request_case(chat_eval, _SHORT_LISTING)


@pytest.mark.asyncio
async def test_idle_to_request_asks_where_the_count_is_posted(chat_eval: ChatEval) -> None:
    """idle → request on the count watcher: "the same way" points at a routine, never at a
    page, so the routine is recognisable and the page is still missing."""
    await _run_idle_request_case(chat_eval, _SHORT_COUNT)


@pytest.mark.asyncio
async def test_idle_to_request_asks_for_the_new_bakery(chat_eval: ChatEval) -> None:
    """idle → request on the daily digest: the bakery is named only as the new one the user
    just found, which is a thing they know and the history does not."""
    await _run_idle_request_case(chat_eval, _SHORT_BAKERY)


@pytest.mark.asyncio
async def test_idle_to_request_holds_the_page_and_asks_what_to_watch_for(
    chat_eval: ChatEval,
) -> None:
    """idle → request, the held-binding case: the page is in the ask and what to look for
    on it is not, so the reply asks for the second thing and works from the first — asking
    again for an address the user just gave is the failure this case exists to catch.

    Its world holds three journeys rather than five (``_WITHOUT_THE_URL_ONLY_WATCHERS``):
    the two routines that watch a page for whatever is newest on it ask for a URL and
    nothing else, so with the URL in this ask their signatures are COMPLETE and binding one
    of them is the rational read — which is what four of five samples did, correctly, on the
    first run.  The ask is unchanged; the history is what makes the shortfall reachable."""
    await _run_idle_request_case(chat_eval, _SHORT_PIER)
