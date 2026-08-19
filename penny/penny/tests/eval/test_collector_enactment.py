"""Collector ENACTMENT contracts (#1905) — the five applied collections run their
cycles, with no chat model in the loop.

``test_state_transitions.py`` measures chat's half of the north star: a returning user
asks for a job, the round negotiates what it needs, and the apply turn stands a live
mechanism up.  Nothing has ever measured the other half — whether that mechanism, built
entirely by the framework (a derived name, a skill's program rendered in, injected
params, an RRULE schedule and notify terms), is SUFFICIENT on its own to enact the goal.

So each case here seeds a request → apply case's FULL exit world — the composed history
its short ask was answered in, the turn that parked the round, and the turn that stood
the job up — and then drives the REAL collector cycle THREE times with no chat turns at
all:

    cycle 1   the page exactly as it stood when the job was set up — the BASELINE
    cycle 2   that same page again, unchanged — the QUIET cycle
    cycle 3   the same page with its ONE controllable fact moved — the CHANGE

Three, not two, because the three cycles are three different claims and the middle one
cannot be folded into either neighbour.  A collection arrives from apply EMPTY, so its
first observation is a new key: it is a baseline, and a first observation is news.  The
write-gate STOP that makes no-news structurally silent (``KEY_EXISTS_UNCHANGED``) fires
only on a SECOND reading of the same value — so the "stay quiet, and never run the notify
steps" contract has no cycle it can fire on until cycle 2 exists.  Cycle 3 is then the
only place a notification is owed, which is what makes "exactly one per change" a
measurement rather than a hope.

What the three cycles are asked is the whole watch contract: fetch the page the job is
pointed at, record what it says, stay silent while nothing has changed, and say something
exactly once when it has.  Everything is scored off PERSISTED state — the collection's
entries, the run records, and the SEND QUEUE read explicitly (a collector cycle enqueues,
and the drainer that delivers is a separate schedule, so a pending-only read reports a
delivered notification as silence).

The five collections are the five the apply beat leaves behind, transcribed from that
beat's own measured draws — the configured row (name, routine, injected values, schedule,
end condition, notify, rendered program), the registry behind it, the threaded
conversation and the seeded ledger — so what runs here is what chat really builds rather
than a convenient hand-built copy.  Since #1907 the stored program carries the job's own
values, joined into the leaves the demonstration put its values in, and the composed
prompt states all three of the instructions, the routine and the values by name — so the
seed goes through that same join, and the loud probe holds the program against exactly
what each case says the join fills.

That set is not uniform, and the case that breaks it is the measurement.  The join matches
a parameter's DEMONSTRATED value against a leaf's on whitespace and case alone, so the
otter census — whose framer recorded the page as the bare host and path the user spoke
while the demonstration's own call carried the full address with its scheme — joins
NOTHING, and its program still reads "the URL to visit each run" with nothing naming the
page.  Four of the five join every value they carry; that one joins none.  Declared per
case rather than repaired, because the fixtures are transcriptions of measured draws and
the cost of the equality rule is the thing worth seeing.

The remaining watched question is the otter case's direction-conditional goal ("warn me
if it DROPS").  Nothing structural on the collection carries a direction — the routine
counts a number and the notify flag is a boolean — so the case scores what the configured
terms ACTUALLY carry, over every surface a cycle reads, and names where the condition
lives when it lives anywhere at all.

Report-only (``min_pass_rate=None``): the thresholds are the code owner's to set once the
numbers are read.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import NamedTuple

import pytest

from penny.constants import WRITE_GATE_STOP_REASONS, PennyConstants, RunOutcome, WriteGateOutcome
from penny.conversation_machine import ConversationState, MachineSnapshot, RoundFraming
from penny.database import Database
from penny.database.models import MemoryRow
from penny.database.skills import (
    _UNSUPPLIED_SLOT,
    DistillInput,
    SkillDraft,
    bind_parameters,
    derive_collection_name,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    Check,
    CollectorCyclesEval,
    CycleObservation,
    Preparer,
    Seeder,
    seeded_run_id,
    send_queue_mechanisms,
)
from penny.tests.eval.fixtures import CannedPage

# The request → apply beat's own five cases, their world, their probe and the seeding
# vocabulary they are written in — read from where that beat declares them rather than
# restated, so the world a cycle runs in is the world that beat measured.  A second copy
# of any of it would be a second history, free to drift from the one chat really leaves.
from penny.tests.eval.test_state_transitions import (
    _KEEL_LANTERN_LISTING,
    _KEEL_LANTERN_URL,
    _NEW_BAKERY_SPECIALS,
    _NEW_BAKERY_URL,
    _NORTH_PIER_DEPARTURES,
    _NORTH_PIER_URL,
    _PARKED_MESSAGE_WINDOW,
    _RIVER_OTTERS_CENSUS,
    _RIVER_OTTERS_URL,
    _SET_CALL_ID,
    _SET_TOOL,
    _SUPPLIED_BAKERY,
    _SUPPLIED_COUNT,
    _SUPPLIED_LISTING,
    _SUPPLIED_PIER,
    _SUPPLIED_SPACES,
    _SUPPLIED_TIMETABLE,
    _assert_every_job_is_live,
    _assert_every_reply_is_threaded,
    _assert_every_round_is_in_the_ledger,
    _bound_parameters,
    _candidate,
    _drawn_state,
    _entries_written_by_this_run,
    _first_divergence,
    _framed,
    _log_ask,
    _log_chat_step,
    _log_classifier_draw,
    _log_reply,
    _park,
    _RequestApplyCase,
    _seed_call_step,
    _seeded_ask_id,
    _seeded_response,
    expected_conversation,
    parked_binding,
    seed_parked_in_request,
)
from penny.tools.base import Tool
from penny.tools.collection_instantiation import parse_schedule, render_reinstantiation_echo
from penny.tools.micro_context import FramedParameter, SkillSignature
from penny.tools.models import ToolResult
from penny.tools.send_message import SendMessageTool

pytestmark = pytest.mark.eval

_FAMILY = "collector-enactment"

# The browse-results log is the browse tool's OWN journal — a page it read is not a
# collection this cycle went and touched, so the "nothing else was touched" claim names it
# apart rather than counting it.
_BROWSE_RESULTS = "browse-results"

# The call a cycle fetches its page with.  Named once, since two checks read it.
_BROWSE_TOOL = "browse"

# The write-gate STOP a no-change cycle ends on, as the run record states it — read from
# the shipped table rather than restated, so a reworded reason cannot silently stop
# matching.
_STOP_REASON = WRITE_GATE_STOP_REASONS[WriteGateOutcome.KEY_EXISTS_UNCHANGED]

# The window a reader of this world reads THROUGH — the parked world's own, plus the two
# turns the supply added.  Derived from that beat's arithmetic rather than picked, so a
# world that grows a turn moves the window with it.
_APPLIED_MESSAGE_WINDOW = _PARKED_MESSAGE_WINDOW + 2

# What the WATCHED question is called wherever it is read — a diff-join key, and the row
# the deterministic scorer pin excludes from its claim (it reads the configured terms
# rather than the cycles).  Named once so a reworded label cannot silently stop matching.
DIRECTION_CHECK_LABEL = "state: the configured terms carry the direction the goal gave"


# ── The cycle-2 pages: the register's own rich pages with ONE fact moved ───────
#
# Derived from the pages the apply beat measured its own turns against rather than written
# out again: a watch's contract is that ONE controllable fact moved and everything else
# stayed put, and a second hand-written body could differ anywhere without anybody
# noticing.  The derivation RAISES on a replacement that matched nothing, so a page
# rewritten upstream fails here rather than producing a cycle-2 page identical to cycle 1's
# — which would score the change cycle a miss on every sample.


def _moved(page: CannedPage, replacements: tuple[tuple[str, str], ...]) -> CannedPage:
    """The same rich page with the named lines replaced — everything else byte-identical.

    Copied through ``replace`` rather than re-constructed field by field, so a page that
    grows a field carries it into its variant instead of silently losing it."""
    text = page.text
    for old, new in replacements:
        if old not in text:
            raise ValueError(f"the page no longer carries {old!r} — re-sample the variant")
        text = text.replace(old, new)
    return replace(page, text=text)


_NORTH_PIER_WITH_THE_DAWN = _moved(
    _NORTH_PIER_DEPARTURES,
    (
        ("Dawn sailing: not on the board this season.", "Dawn sailing: 05:20 from the pier head."),
        ("Board last amended four days ago", "Board last amended this morning"),
    ),
)

_KEEL_LANTERN_REPRICED = _moved(
    _KEEL_LANTERN_LISTING,
    (
        ("Price: $128", "Price: $112"),
        ("Price last changed eleven days ago.", "Price last changed this morning."),
    ),
)

_RIVER_OTTERS_FEWER = _moved(
    _RIVER_OTTERS_CENSUS,
    (("Count: 46 otters", "Count: 39 otters"),),
)

_NEW_BAKERY_TOMORROW = _moved(
    _NEW_BAKERY_SPECIALS,
    (("Today's special: apricot and almond galette", "Today's special: fig and hazelnut tart"),),
)


# ── The five collections, as the apply beat leaves them ───────────────────────


class _ConfiguredJob(NamedTuple):
    """The TERMS the apply turn set on the collection — everything about the job that is
    not the routine or the values it is pointed at (#1869).

    ``expires_in`` is a DISTANCE rather than a date for the reason the composed world's
    end conditions are: seeds write through the real store APIs, which stamp every row at
    the moment they run, so a fixed date transcribed from the measured run would be one
    the job had already passed — and an expired collection archives itself instead of
    running."""

    description: str
    schedule: str
    expires_in: timedelta | None


class _WatchedFact(NamedTuple):
    """The ONE controllable fact the two pages differ on — what a cycle that really read
    the page has to come back holding.

    ``quiet`` is what the page said when the job was set up and ``changed`` what it says
    on the second cycle.  Both are matched as substrings of the entry the cycle wrote
    (key or content — where in the entry a fact lands is deliberately open), so a cycle
    that stored the value with its units or its label still reads as having stored it."""

    quiet: str
    changed: str


class _EnactmentCase(NamedTuple):
    """One applied collection and the two cycles it is driven through.

    ``parked`` is the request → apply case this continues — read rather than restated, so
    the world the cycles run in is that beat's own world and the routine, the ask and the
    supply cannot be described two ways.  ``values`` is what that turn's binder BOUND,
    transcribed from the measured run (the held-binding case bound the whole spoken span,
    "the dawn sailing", which is what its container is named from).

    ``job`` is the terms the turn set, ``quiet``/``altered`` the two pages, and ``fact``
    what they differ on.  ``confirmation`` is what Penny said when the job stood up — a
    transcription of a measured draw, seeded as the conversation's last turn.

    ``joins`` is which of those values the RUNTIME JOIN (#1907) writes into the program's
    own leaves, stated as data because it is a property of the routine this job runs rather
    than something this beat chooses: the join matches a parameter's DEMONSTRATED value
    against the leaf's, on whitespace and case alone, so a routine whose framer recorded
    the user's span in a different spelling from the one the demonstration's call carried
    joins nothing there — visibly, in the program, and long before any of this.  Declaring
    the set makes both directions fail loudly: a join that stops working is a program the
    collector can no longer run, and a join that starts working is a case whose comment no
    longer describes the world.

    ``direction`` is the goal's own condition where it HAS one ("drops" for the otter
    census).  It is not a scorer string for the cycle's behaviour: it is what the
    directionality check looks for among the terms the collector actually reads, so the
    case reports where — if anywhere — the condition survived configuration."""

    case_id: str
    parked: _RequestApplyCase
    values: dict[str, str]
    joins: tuple[str, ...]
    job: _ConfiguredJob
    quiet: CannedPage
    altered: CannedPage
    fact: _WatchedFact
    confirmation: str
    direction: str | None = None

    @property
    def skill(self) -> SkillDraft:
        """The routine the job runs — the one the parked round was negotiating."""
        return self.parked.parked.skill

    @property
    def ordered_values(self) -> list[str]:
        """The bound values in the routine's DECLARED order — what the container's name is
        derived from, and the order a fixture must not be free to choose."""
        return [self.values[parameter.name] for parameter in self.skill.parameters]

    @property
    def container(self) -> str:
        """The collection the job runs in, through the SHIPPED derivation — spelling it out
        would be a second copy of the scheme jobs are identified by."""
        return derive_collection_name(slug_skill_name(self.skill.name), self.ordered_values)

    @property
    def source(self) -> str:
        """The page the job is pointed at, as an address a fetch is recognised by — the
        scheme dropped, since a search and a direct read spell it differently."""
        return self.values["url"].split("://", 1)[-1].casefold()


_TIMETABLE = _EnactmentCase(
    case_id="enactment-timetable",
    parked=_SUPPLIED_TIMETABLE,
    values={"url": _NORTH_PIER_URL, "keyword": "dawn sailing"},
    # Both join: the timetable routine's framer recorded each value exactly as the
    # demonstration's own call carried it, so each leaf is claimed and filled.
    joins=("url", "keyword"),
    job=_ConfiguredJob(
        description="Keep an eye out for the dawn sailing every morning and let me know "
        "when it shows up.",
        schedule="FREQ=DAILY;BYHOUR=6",
        expires_in=None,
    ),
    quiet=_NORTH_PIER_DEPARTURES,
    altered=_NORTH_PIER_WITH_THE_DAWN,
    fact=_WatchedFact(quiet="not on the board", changed="05:20"),
    confirmation=(
        "Got it! I just set up a routine that watches the North Pier departures page "
        "(<https://northpier.example/departures>) for “Dawn Sailing.” It runs once a day at "
        "06:00 PT, flags any new entry or change, and will ping you when that happens. "
        "There’s no end date—so it keeps going until we decide to turn it off. 🚀"
    ),
)

_LISTING = _EnactmentCase(
    case_id="enactment-listing",
    parked=_SUPPLIED_LISTING,
    values={"url": _KEEL_LANTERN_URL},
    joins=("url",),
    job=_ConfiguredJob(
        description='Monitors price of a faux market listing for "keel lantern" and reports '
        "when it changes.",
        schedule="FREQ=HOURLY;INTERVAL=2",
        expires_in=timedelta(days=3),
    ),
    quiet=_KEEL_LANTERN_LISTING,
    altered=_KEEL_LANTERN_REPRICED,
    fact=_WatchedFact(quiet="$128", changed="$112"),
    confirmation=(
        "Got it! I’ve wired up a routine that keeps an eye on the faux-market listing for "
        "the keel lantern at <https://faux-market.example/keel-lantern>. It pings me every "
        "two hours (once, twice,… as scheduled) until Sunday – then it stops. Each time I "
        "detect a price change, I’ll let you know right away. 🚀"
    ),
)

_COUNT = _EnactmentCase(
    case_id="enactment-count",
    parked=_SUPPLIED_COUNT,
    values={"url": _RIVER_OTTERS_URL},
    # NOTHING joins.  The count routine's framer recorded its page as the bare host and
    # path the user spoke ("harborseals.example/colony-count") while the demonstration's
    # own browse call carried the full address with its scheme — two spellings of one
    # page, and the join compares on whitespace and case alone, so no leaf is claimed and
    # the program still reads "the URL to visit each run".  Declared rather than repaired:
    # the fixture is a transcription of a measured draw, and this is the cost of the
    # normalized-equality rule, which is what the case is here to show.
    joins=(),
    job=_ConfiguredJob(
        description="Track otter count from https://riverotters.example/census each week and "
        "notify me if it drops",
        schedule="FREQ=WEEKLY",
        expires_in=None,
    ),
    quiet=_RIVER_OTTERS_CENSUS,
    altered=_RIVER_OTTERS_FEWER,
    fact=_WatchedFact(quiet="46", changed="39"),
    confirmation=(
        "I’ve set a weekly watchdog on the otter census page: every week it pulls the "
        "current count from https://riverotters.example/census and will ping you if that "
        "number drops. It runs forever—there’s no end date—and you’ll get notified when a "
        "change shows up. ✅"
    ),
    direction="drop",
)

_DIGEST = _EnactmentCase(
    case_id="enactment-digest",
    parked=_SUPPLIED_BAKERY,
    values={"url": _NEW_BAKERY_URL},
    joins=("url",),
    job=_ConfiguredJob(
        description="Retrieve the daily special from https://newbakery.example/specials each "
        "morning until the end of the month.",
        schedule="FREQ=DAILY",
        expires_in=timedelta(days=14),
    ),
    quiet=_NEW_BAKERY_SPECIALS,
    altered=_NEW_BAKERY_TOMORROW,
    fact=_WatchedFact(quiet="apricot and almond galette", changed="fig and hazelnut tart"),
    confirmation=(
        "Got it! Here’s what I’ve set up:\n\n"
        "- What it watches: Pulls the daily special from "
        "https://newbakery.example/specials every morning.\n"
        "- How often it runs: Daily (FREQ=DAILY).\n"
        "- When it stops: At the end of the month.\n"
        "- You’ll hear from it: Yes—if the special changes, I’ll send you a message. 🌟"
    ),
)

_HELD_BINDING = _EnactmentCase(
    case_id="enactment-held-binding",
    parked=_SUPPLIED_PIER,
    values={"url": _NORTH_PIER_URL, "keyword": "the dawn sailing"},
    joins=("url", "keyword"),
    job=_ConfiguredJob(
        description='Check and notify when the "Dawn" ferry departure appears on North Pier '
        "timetable page.",
        schedule="FREQ=DAILY;BYHOUR=8",
        expires_in=None,
    ),
    quiet=_NORTH_PIER_DEPARTURES,
    altered=_NORTH_PIER_WITH_THE_DAWN,
    fact=_WatchedFact(quiet="not on the board", changed="05:20"),
    confirmation=(
        "Got it! I’m now watching the North Pier departures page at "
        "https://northpier.example/departures for a line that says “the dawn sailing.” The "
        "check runs every day at 8:00 AM PT. It’ll keep running forever (no expiration) and "
        "you’ll get a notification whenever the entry changes or appears. Happy to keep an "
        "eye on it! 🚤"
    ),
)

# Every applied collection, in one place — so the deterministic pins in
# ``test_eval_harness.py`` can drive each seeder and check its claims without a GPU.
ENACTMENT_CASES = (_TIMETABLE, _LISTING, _COUNT, _DIGEST, _HELD_BINDING)


# ── The world: the parked round, then the turn that stood the job up ──────────

# The run ids the supply turn is written under.  One such turn per world, so one pair
# rather than a per-journey mint — and seeded-prefixed, so every "what did this cycle do"
# reader excludes them.
_SUPPLY_DRAW_RUN = seeded_run_id("supply-draw")
_SUPPLY_TURN_RUN = seeded_run_id("supply-turn")


def seed_applied_job(case: _EnactmentCase) -> Seeder:
    """The world the cycles run in: the request → apply case's own parked world, then the
    turn whose supply completed the binding and stood the job up.

    Compositional like every world before it — the parked half is that beat's seeder, so
    nothing here restates a history it already defines — and the half this module adds is
    the exit state that beat MEASURES, written the way the apply path writes it."""

    def seed(db: Database) -> None:
        seed_parked_in_request(case.parked)(db)
        _seed_supply_turn(db, case)

    return seed


def _seed_supply_turn(db: Database, case: _EnactmentCase) -> None:
    """The turn that finished the round, with everything it left behind: the supply
    INCOMING, the skill-gated draw that decided apply over the round's own binding, the
    container built and configured, the chat run carrying the ``collection_set`` call and
    its echoed result, Penny's confirmation OUTGOING threaded to the supply, and the apply
    move itself — anchored to the ask the round has been anchored to all along and carrying
    the framing the binder completed."""
    anchor_id = _seeded_ask_id(db, case.parked.parked.ask, limit=_PARKED_MESSAGE_WINDOW)
    assert anchor_id is not None, f"{case.case_id}: the round's ask must be findable by content"
    supply_id = _log_ask(db, case.parked.supply, case.case_id)
    _log_supply_draw(db, case)
    row = _stand_the_job_up(db, case)
    _seed_supply_run(db, case, row)
    _log_reply(db, case.confirmation, answering=supply_id)
    _park(
        db,
        ConversationState.APPLY,
        anchor_message_id=anchor_id,
        from_state=ConversationState.REQUEST,
        run_id=_SUPPLY_TURN_RUN,
        message_id=supply_id,
        skill_name=slug_skill_name(case.skill.name),
        framing=bound_framing(case),
    )


def _log_supply_draw(db: Database, case: _EnactmentCase) -> None:
    """The SKILL-GATED draw that decided the apply move — over a machine parked in request
    on the round's ask, shown the binding it was waiting on (#1894) and every routine this
    world taught, and naming the one it bound on its second line."""
    parked = case.parked.parked
    _log_classifier_draw(
        db,
        run_id=_SUPPLY_DRAW_RUN,
        snapshot=MachineSnapshot(
            state=ConversationState.REQUEST,
            penny_last_turn=case.parked.reply,
            task_anchor=parked.ask,
            skill_candidates=[_candidate(journey.round.skill) for journey in parked.journeys],
            round_binding=parked_binding(case.parked),
        ),
        message=case.parked.supply,
        drawn=_drawn_state(ConversationState.APPLY, skill=slug_skill_name(case.skill.name)),
    )


def bound_framing(case: _EnactmentCase) -> RoundFraming:
    """The framing the binder settled at apply entry (#1870): the REGISTRY's own name and
    description for the routine, its declared parameters each carrying the value the two
    turns settled, and the container the shipped derivation makes of them.

    Built through the production models rather than hand-written JSON, and off the fixture
    DRAFT rather than the registry row, because the runner lays the registry down after
    this seed runs — the probe then reads the two against each other."""
    signature = SkillSignature(
        name=slug_skill_name(case.skill.name),
        description=case.skill.description,
        parameters=tuple(
            FramedParameter(
                name=parameter.name,
                description=parameter.description,
                value=case.values[parameter.name],
            )
            for parameter in case.skill.parameters
        ),
    )
    return _framed(signature)


def _stand_the_job_up(db: Database, case: _EnactmentCase) -> MemoryRow:
    """The apply turn's durable half, in the two writes the production path makes: the
    container is CREATED (the binder's find-or-create at round entry, storage only), then
    ``collection_set`` configures it — the routine's steps rendered in with the attachment
    bound to the container's own name, the injected values, the turn's schedule and end
    condition, notify on, and the routine plus its params stamped as provenance.

    Both writes go through the real store methods, so the mutation ledger records them
    citing this run — which is what makes "nothing has touched anything else since" a read
    rather than an assumption."""
    container = case.container
    db.memories.create_collection(container, case.job.description)
    schedule = parse_schedule(case.job.schedule)
    return db.memories.update_collection_metadata(
        container,
        extraction_prompt=rendered_program(case),
        schedule=schedule.rule,
        replace_schedule=True,
        max_runs=schedule.max_runs,
        expires_at=_end_condition(case.job),
        notify=True,
        skill_name=slug_skill_name(case.skill.name),
        skill_params=case.values,
        run_id=_SUPPLY_TURN_RUN,
    )


def rendered_program(case: _EnactmentCase) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps in its own order (``render_skill_prompt``): the attachment bound to the
    container, the RUNTIME JOIN (#1907) writing each parameter's bound value into the
    leaves the demonstration put its own value in, then the render.

    Composed here rather than called, because the shipped seam takes the registry ROW and
    the runner lays the registry down after this seed runs — so what a fixture must not do
    is invent a fourth step or reorder these three.  Public because the probe and the
    directionality check both read it, and a second copy would be free to drift from what
    the collection actually stores."""
    attached = retarget_writes(case.skill.steps, case.container)
    joined = bind_parameters(attached, case.skill.parameters, case.values)
    return render_skill(joined, case.values)


def _end_condition(job: _ConfiguredJob) -> datetime | None:
    """A bounded job's end, as a distance from when the world is laid down — so the job is
    LIVE when the cycles run, which is what the world claims it is."""
    if job.expires_in is None:
        return None
    return datetime.now(UTC) + job.expires_in


def _seed_supply_run(db: Database, case: _EnactmentCase, row: MemoryRow) -> None:
    """The supply turn's chat run: the one ``collection_set`` call it made, the result that
    came back, and the confirmation it closed on."""
    conversation: list[dict] = [{"role": "user", "content": case.parked.supply}]
    conversation = _seed_call_step(
        db, conversation, _SET_CALL_ID, _set_step(case, row), run_id=_SUPPLY_TURN_RUN
    )
    _log_chat_step(
        db,
        run_id=_SUPPLY_TURN_RUN,
        messages=conversation,
        response=_seeded_response(case.confirmation),
    )


def _set_step(case: _EnactmentCase, row: MemoryRow) -> DistillInput:
    """The ``collection_set`` call that stood the job up — TERMS only (#1869: the routine
    and the values it is pointed at are supplied framework-side), with the production echo
    as its result, so the seeded ledger carries the text a real turn would have read."""
    arguments: dict = {"name": row.name, "schedule": case.job.schedule, "notify": True}
    if row.expires_at is not None:
        arguments["expires_at"] = row.expires_at.isoformat()
    echo = render_reinstantiation_echo(row, slug_skill_name(case.skill.name), case.values)
    return DistillInput(
        source_ordinal=1,
        tool=_SET_TOOL,
        arguments=arguments,
        result=Tool.format_result(_SET_TOOL, arguments, ToolResult(message=echo, mutated=True)),
    )


# ── The loud probe: the world really is the one the apply turn leaves ─────────


def assert_applied_world(db: Database, case: _EnactmentCase) -> None:
    """Everything the seeder is responsible for, asserted out loud.

    Five cases share one seeder and each sample costs two live cycles, so a drift here is
    ten cycles of GPU spent measuring a world nothing produces.

    The parked world's own claims are read through the request beat's assertions where they
    still hold — the jobs it left running and the ledger it left readable — while the
    conversation is asserted HERE, because this world has two turns that one does not."""
    journeys = case.parked.parked.journeys
    _assert_every_job_is_live(db, journeys)
    _assert_every_round_is_in_the_ledger(db, journeys)
    _assert_the_applied_conversation(db, case)
    _assert_the_job_is_configured(db, case)
    _assert_the_round_landed_in_apply(db, case)


def _assert_the_applied_conversation(db: Database, case: _EnactmentCase) -> None:
    """The world READS as the conversation it claims to be: the composed history, the short
    ask and the reply that asked for what it left out, then the supply and the confirmation
    that the job is running — in that order, alternating, every reply threaded.

    Read through ``get_messages_since``, the reader ``_build_conversation`` uses, for the
    reason the parked probe reads through it: an unthreaded reply is in the record and out
    of the conversation, and only the parent link tells the two apart."""
    window = db.messages.get_messages_since(
        TEST_SENDER, since=datetime.min, limit=_APPLIED_MESSAGE_WINDOW
    )
    seen = [(row.direction, row.content) for row in window]
    expected = [
        *expected_conversation(case.parked.parked.journeys),
        (PennyConstants.MessageDirection.INCOMING, case.parked.parked.ask),
        (PennyConstants.MessageDirection.OUTGOING, case.parked.reply),
        (PennyConstants.MessageDirection.INCOMING, case.parked.supply),
        (PennyConstants.MessageDirection.OUTGOING, case.confirmation),
    ]
    assert seen == expected, (
        "the applied world must read back as the conversation it claims to be — "
        f"diverges at turn {_first_divergence(seen, expected)}"
    )
    _assert_every_reply_is_threaded(window)


def _assert_the_job_is_configured(db: Database, case: _EnactmentCase) -> None:
    """The collection the cycles run is configured exactly as the apply turn left it: the
    derived container, the routine and the values injected, the turn's schedule, notify on,
    a program rendered from the routine's own steps, and still LIVE.

    Read off the row rather than trusted, because every one of these is something a cycle
    is then measured against — a job pointed at nothing, or one that expired the moment it
    was written, would report the collector missing what it was never given."""
    row = db.memories.get(case.container)
    assert row is not None, f"{case.case_id}: the job's container {case.container!r} must exist"
    assert row.skill_name == slug_skill_name(case.skill.name), (
        f"{case.case_id}: the job must run the routine the round bound, not {row.skill_name!r}"
    )
    bound = _bound_parameters(row)
    assert bound == case.values, (
        f"{case.case_id}: the job must carry the values the turns settled, got {bound}"
    )
    assert row.schedule == parse_schedule(case.job.schedule).rule, (
        f"{case.case_id}: the job must carry the turn's own rule, got {row.schedule!r}"
    )
    assert row.notify and not row.archived, f"{case.case_id}: the job must be live and notifying"
    _assert_the_program_is_rendered(db, case, row)


def _assert_the_program_is_rendered(db: Database, case: _EnactmentCase, row: MemoryRow) -> None:
    """The stored program is the routine's own steps rendered into THIS container, and the
    collection holds nothing yet.

    The empty container is the premise of the quiet cycle: a job arriving from apply has
    never observed anything, so what its first cycle finds is genuinely its first reading."""
    assert row.extraction_prompt == rendered_program(case), (
        f"{case.case_id}: the program must be the routine rendered into this container, got "
        f"{row.extraction_prompt!r}"
    )
    _assert_the_values_are_joined(case, row.extraction_prompt or "")
    entries = require_memory(db, case.container).read_all()
    assert not entries, f"{case.case_id}: a job the apply turn just built holds nothing yet"


def _assert_the_values_are_joined(case: _EnactmentCase, program: str) -> None:
    """The program carries exactly the values the case says the RUNTIME JOIN fills, and
    names none of them as unsupplied (#1907).

    This is the premise of the whole beat since the join landed: a cycle can only fetch the
    page the job is pointed at if something it reads names the page, and before the join the
    program read ``browse(queries=[{the url of the … page to browse each run}])`` with
    nothing naming it at all.  Asserted in BOTH directions off ``case.joins`` because both
    are silent on a run and each one hollows the beat differently — a value that stopped
    joining is a cycle measured against a job it was never given (three live cycles per
    sample before anybody notices), while one that started joining is a case whose recorded
    reason no longer describes the world it seeds.  A parameter the job binds must never
    render as the unsupplied slot either way: that shape belongs to a term the collection
    was never given, which is a different state from a leaf no parameter claims."""
    for name, value in case.values.items():
        joined = value in program
        expected = name in case.joins
        assert joined == expected, (
            f"{case.case_id}: the runtime join {'must' if expected else 'must not'} fill a "
            f"leaf with {name!r} ({value!r}) — program: {program!r}"
        )
        assert _UNSUPPLIED_SLOT.format(name=name) not in program, (
            f"{case.case_id}: the program must not name {name!r} as unsupplied — the job "
            f"binds it to {value!r}"
        )


def _assert_the_round_landed_in_apply(db: Database, case: _EnactmentCase) -> None:
    """The machine records the move the supply turn made: apply, on the routine it bound,
    carrying the completed framing — the exit state this whole beat continues from."""
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.APPLY.value, (
        f"{case.case_id}: the round must have landed in apply, got {latest}"
    )
    assert latest.skill_name == slug_skill_name(case.skill.name), (
        f"{case.case_id}: the move must name the routine it bound, got {latest.skill_name!r}"
    )
    assert latest.skill_frame is not None, (
        f"{case.case_id}: a completed binding settles the round's framing"
    )
    assert RoundFraming.model_validate_json(latest.skill_frame) == bound_framing(case), (
        f"{case.case_id}: the recorded framing must be the one the binder completed"
    )


def _probe_applied_world(case: _EnactmentCase) -> Preparer:
    """The prepare hook: the seeder's own claims, run once the runner has laid the fixture
    registry down so the routine the job names is one that exists."""

    def probe(penny: Penny) -> None:
        assert_applied_world(penny.db, case)
        routine = penny.db.skills.get(slug_skill_name(case.skill.name))
        assert routine is not None, f"{case.case_id}: the job's routine must be registered"

    return probe


# ── Scoring ───────────────────────────────────────────────────────────────────
#
# One scorer for all five cases, bound to the case's own terms.  Labels are diff-join keys
# and are deliberately case-NEUTRAL — one wording reads the same whether the fact that
# moved was a price, a count, a dish or a sailing.


def _entry_texts(entries: dict[str, str]) -> str:
    """One collection's entries as a single searchable text — keys and contents alike,
    since where in an entry a fact lands is deliberately open (#1854)."""
    return " ".join([*entries.keys(), *entries.values()]).casefold()


def _fetched_check(case: _EnactmentCase, cycle: CycleObservation) -> Check:
    """The cycle went to the page the job is pointed at.

    Read off the cycle's own browse arguments rather than off the browse-results log,
    because the seeded world already holds five pages its rounds read — and the address is
    matched without its scheme, since a search spells the page differently from a direct
    read."""
    fetched = any(
        case.source in str(call.arguments).casefold()
        for call in cycle.calls
        if call.tool == _BROWSE_TOOL
    )
    return Check(
        f"cycle {cycle.index + 1}: fetched the page the job is pointed at",
        fetched,
        anchor=_BROWSE_TOOL,
        rationale=None
        if fetched
        else f"browsed {[call.arguments for call in cycle.calls if call.tool == _BROWSE_TOOL]}",
        kind="spine",
    )


def _recorded_check(cycle: CycleObservation, *, fact: str, absent: str | None = None) -> Check:
    """The value the cycle left in the collection is the page's controllable fact — and,
    on the change cycle, no longer the one the page used to carry."""
    held = _entry_texts(cycle.after)
    ok = fact.casefold() in held and (absent is None or absent.casefold() not in held)
    return Check(
        f"cycle {cycle.index + 1}: the collection holds what the page says",
        ok,
        rationale=None if ok else f"expected {fact!r}, the collection holds {cycle.after}",
        kind="state",
    )


def _baseline_check(cycle: CycleObservation) -> Check:
    """The first cycle's write landed — the baseline the two cycles after it are read
    against.  A watch with nothing recorded has nothing to compare, so this is the one
    claim every later check stands on."""
    return Check(
        f"cycle {cycle.index + 1}: the write landed",
        cycle.changed,
        anchor="collection_write(",
        rationale=None
        if cycle.changed
        else f"nothing was written and the run closed {cycle.reason or '—'}",
        kind="state",
    )


def _stopped_check(cycle: CycleObservation) -> Check:
    """The cycle re-read the same value and STOPPED at the write chokepoint — the
    structural no-news close (``KEY_EXISTS_UNCHANGED``), read off the run's own reason.

    This is what makes silence structural rather than a judgment the model makes each
    cycle: the write gate compares the value it was handed against the one already stored
    and ends the run there, so the steps after the write — the notify steps — are never
    reached.  It needs a SECOND reading of the same value to fire at all, which is why the
    quiet cycle is the middle one."""
    stopped = cycle.reason == _STOP_REASON
    return Check(
        f"cycle {cycle.index + 1}: the cycle stopped at the write chokepoint",
        stopped,
        rationale=None if stopped else f"the run closed {cycle.reason or cycle.outcome or '—'}",
        kind="state",
    )


def _silent_check(cycle: CycleObservation) -> Check:
    """Nothing was sent while nothing had changed — the quiet cycle's whole contract, read
    off the SEND QUEUE explicitly AND off the calls the cycle made.

    Both halves, because they fail differently: a message on the queue is the user being
    told about nothing, while a ``send_message`` call that never reached the queue is the
    notify steps having RUN — the thing the chokepoint STOP exists to make unreachable —
    and a contract that only counted queued messages would score that green."""
    sent = not cycle.sent
    never_ran = SendMessageTool.name not in cycle.tools
    return Check(
        f"cycle {cycle.index + 1}: nothing was sent and the notify steps never ran",
        sent and never_ran,
        rationale=None
        if sent and never_ran
        else f"queued {len(cycle.sent)} message(s), called {cycle.tools}",
        kind="state",
    )


def _one_notify_check(case: _EnactmentCase, cycle: CycleObservation) -> Check:
    """Exactly one message, naming what changed — one notification per change, never a
    repeat and never a silent change.  Read off the SEND QUEUE, delivered rows included."""
    named = [text for text in cycle.sent if case.fact.changed.casefold() in text.casefold()]
    ok = len(cycle.sent) == 1 and len(named) == 1
    return Check(
        f"cycle {cycle.index + 1}: one message, naming what changed",
        ok,
        rationale=None
        if ok
        else f"queued {len(cycle.sent)} message(s), {len(named)} naming {case.fact.changed!r}",
        kind="state",
    )


def _honest_record_check(cycle: CycleObservation) -> Check:
    """The run record states what the cycle did: ``worked`` exactly when it changed
    something or queued a message, ``no_work`` exactly when it did neither.

    Structural on both sides — the record is generated from the ledger (#1569) and what it
    is compared against is persisted state, so neither half is a model's own account."""
    did = cycle.changed or bool(cycle.sent)
    expected = RunOutcome.WORKED if did else RunOutcome.NO_WORK
    ok = cycle.outcome == expected.value
    return Check(
        f"cycle {cycle.index + 1}: the run record states what the cycle did",
        ok,
        rationale=None if ok else f"recorded {cycle.outcome or '—'}, the cycle {expected.value}",
        kind="proc",
    )


def _nothing_else_touched_check(db: Database, case: _EnactmentCase) -> Check:
    """The cycles wrote nowhere but their own collection, and queued for nobody else.

    Entries are read by their run stamps — a live run's write against a seeded world's — so
    this is a read rather than a diff, and the browse tool's own journal is named apart
    because a page it read is not a collection anything went and touched."""
    strayed = [
        f"{entry.memory_name}/{entry.key}"
        for entry in _entries_written_by_this_run(db)
        if entry.memory_name not in (case.container, _BROWSE_RESULTS)
    ]
    queued = [name for name in send_queue_mechanisms(db) if name != case.container]
    ok = not strayed and not queued
    return Check(
        "state: nothing outside this collection was touched",
        ok,
        rationale=None if ok else f"wrote {strayed}, queued for {queued}",
        kind="state",
    )


def _direction_check(case: _EnactmentCase, cycles: list[CycleObservation]) -> Check:
    """The WATCHED question (#1905): the goal is direction-conditional ("warn me if it
    DROPS"), and nothing structural on a configured collection carries a direction — the
    routine counts a number and notify is a boolean.

    So this scores what the configured terms ACTUALLY carry, over every surface a cycle
    reads — since #1907 that is the composed prompt's three parts (the instructions, the
    routine and what it is for, and the values by name) plus the collection's own name and
    description.  A pass means the condition survived configuration somewhere the collector
    can see it; the rationale names WHERE, so "it lives in a prose description the apply
    turn happened to write" is reported as what it is rather than read as the mechanism
    carrying it.  Cases with no direction in their goal report it not-applicable."""
    direction = case.direction
    if direction is None:
        return Check.na(
            DIRECTION_CHECK_LABEL, rationale="the goal states no direction", kind="state"
        )
    carriers = _direction_carriers(case, direction)
    return Check(
        DIRECTION_CHECK_LABEL,
        bool(carriers),
        rationale=f"the direction is stated in: {carriers}"
        if carriers
        else f"no configured term names {direction!r} — "
        f"the cycles notified on {len(cycles[-1].sent)} change(s) either way",
        kind="state",
    )


def _direction_carriers(case: _EnactmentCase, direction: str) -> list[str]:
    """Which of the collection's model-facing terms state the goal's direction."""
    return [name for name, text in configured_terms(case).items() if direction in text.casefold()]


def configured_terms(case: _EnactmentCase) -> dict[str, str]:
    """Everything a cycle reads about the job it is running, by surface.

    The name it runs under and the description it carries, plus the composed prompt's
    three parts since #1907 — the INSTRUCTIONS (the program, with the bound values already
    joined into its leaves), the ROUTINE it runs and what that is for, and the VALUES it is
    pointed at, listed by name.  Kept as separate surfaces rather than one blob because the
    directionality question is about WHERE a term survived configuration, and a pin holds
    each of them against the prompt the collector really composes."""
    return {
        "name": case.container,
        "description": case.job.description,
        "instructions": rendered_program(case),
        "routine": f"{slug_skill_name(case.skill.name)} — {case.skill.description}",
        # In the ROUTINE's declared order, which is the order the collector lists them in —
        # a fixture free to choose its own would match by luck on a one-parameter job and
        # drift on the two-parameter one.
        "values": "\n".join(
            f"- {parameter.name}: {case.values[parameter.name]}"
            for parameter in case.skill.parameters
        ),
    }


def _score_enactment(
    db: Database, cycles: list[CycleObservation], *, case: _EnactmentCase
) -> list[Check]:
    """The watch contract across the three cycles: record the page's fact once, stay silent
    the next time it says the same thing, and say something exactly once when it moves —
    honestly recorded throughout, and touching nothing else.

    The three cycles are three different claims, which is why the beat needs three: the
    FIRST is a baseline (a collection arrives from apply empty, so its first observation is
    a new key), the SECOND is the only one that can reach the write-gate STOP (it takes a
    second reading of the same value to fire), and the THIRD is the change."""
    first, quiet, changed = cycles
    return [
        _fetched_check(case, first),
        _recorded_check(first, fact=case.fact.quiet),
        _baseline_check(first),
        _honest_record_check(first),
        _fetched_check(case, quiet),
        _recorded_check(quiet, fact=case.fact.quiet),
        _stopped_check(quiet),
        _silent_check(quiet),
        _honest_record_check(quiet),
        _fetched_check(case, changed),
        _recorded_check(changed, fact=case.fact.changed, absent=case.fact.quiet),
        _one_notify_check(case, changed),
        _honest_record_check(changed),
        _nothing_else_touched_check(db, case),
        _direction_check(case, cycles),
    ]


async def _run_enactment_case(
    collector_cycles_eval: CollectorCyclesEval, case: _EnactmentCase
) -> None:
    """Drive one applied collection through its three cycles: the apply turn's exit world
    behind it, exactly the routines its history taught in the registry, the register's own
    pages with the case's own page swapped for each cycle's variant, and the shared scorer
    bound to the case's terms.  Report-only — the thresholds are the code owner's to set
    once the numbers are read."""
    unchanged = [case.quiet, *_SUPPLIED_SPACES]
    await collector_cycles_eval(
        case_id=case.case_id,
        collection=case.container,
        seed=seed_applied_job(case),
        seed_skills=[journey.round.skill for journey in case.parked.parked.journeys],
        prepare=_probe_applied_world(case),
        cycles=[unchanged, unchanged, [case.altered, *_SUPPLIED_SPACES]],
        score=partial(_score_enactment, case=case),
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_the_timetable_watch_runs_its_cycles(
    collector_cycles_eval: CollectorCyclesEval,
) -> None:
    """The ferry timetable job: the dawn sailing is not on the board when the job is set
    up, stays off it on the quiet cycle, and is there on the third."""
    await _run_enactment_case(collector_cycles_eval, _TIMETABLE)


async def test_the_price_watch_runs_its_cycles(
    collector_cycles_eval: CollectorCyclesEval,
) -> None:
    """The price watcher on its second listing: the same price twice over, then a
    different one — and a bounded job that is still live for all three cycles."""
    await _run_enactment_case(collector_cycles_eval, _LISTING)


async def test_the_count_watch_runs_its_cycles(
    collector_cycles_eval: CollectorCyclesEval,
) -> None:
    """The otter census: the count holds steady for two cycles and DROPS on the third —
    the case whose goal names a direction its configured terms may not carry."""
    await _run_enactment_case(collector_cycles_eval, _COUNT)


async def test_the_daily_special_runs_its_cycles(
    collector_cycles_eval: CollectorCyclesEval,
) -> None:
    """The bakery's daily special: the morning's special is the value, unchanged when the
    board is re-read, and a different one the next day."""
    await _run_enactment_case(collector_cycles_eval, _DIGEST)


async def test_the_held_binding_watch_runs_its_cycles(
    collector_cycles_eval: CollectorCyclesEval,
) -> None:
    """The timetable job's twin, bound from the other side: the same world, the same
    routine, and a keyword the user spoke rather than one the ask carried."""
    await _run_enactment_case(collector_cycles_eval, _HELD_BINDING)
