"""A price watch runs its cycles: write when the reading moves, silence when it does not
(#2017, the collector half of the cohort port).

**The behaviour, as one sentence.** Penny re-reads the job's page each cycle, writes the value
only when it has changed, and tells the user only then — the same reading twice over is
silence, a moved reading is a write and one notification.

**The arms are the job's own inputs, not a rewording.**  A collector has no natural-language
input to vary: its program is ``render_skill(steps, params)``, deterministic ``N. tool(args)``
lines, and hand-authoring five of those would be measuring a render rather than a draw.  So the
five arms are **five instances of one theme on one program** — five listings, five urls, five
prices, five matching pages.  The skill SHAPE is byte-fixed across all of them; only the bound
value and the content it reads move.

**That is what makes every claim below a SHAPE claim**, which is the point of driving it this
way.  Nothing here may name a price, a url or an item, because no single one of them is true of
the cohort.  What survives is the enactment contract itself, and it holds identically across
all five: cycle 2 is silent because nothing moved, cycle 3 writes because something did, and a
notification is owed on exactly the cycles that moved the reading.

**Three cycles, not two.**  A collection arrives empty, so its first observation is a new key —
a baseline, and a first observation is news.  The write-gate STOP that makes no-news
structurally silent fires only on a SECOND reading of the same value, so "stay quiet" has no
cycle it can fire on until cycle 2 exists.  Cycle 3 is then the only place a notification is
owed on top of the baseline's, which is what makes "told on exactly the cycles that moved"
a measurement rather than a hope.

``test_collector_enactment.py``'s fifteen cases are five jobs × three cycles plus their
notify/quiet pairs — the same claim over five different PROGRAMS.  Those are different
routines, therefore different behaviours, and they split rather than pool; collapsing them is
#2007's, not this file's.  This case exists to prove the cohort seam carries a multi-cycle,
real-store, no-user-turn shape at all.

**The LANDED category is empty for this case, and that is the correct report.**  A watch runs
cycles; it does not move a conversation machine, and what each cycle left behind is read off
persisted entries and the send queue — so its claims are STORE claims and the state section
says it has none, rather than something being invented to fill it.

Report-only (``min_pass_rate=None``).  All content is synthetic — an invented marketplace —
because the repo is public.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.skills import (
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    bind_parameters,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.program import program_calls
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    CollectorCyclesEval,
    CycleArm,
    collection_entries,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "watch-writes-only-when-the-reading-moves"
_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The one collection every arm's job is configured on.  One name, because one theme: what
# differs between arms is what the job is POINTED AT, never what kind of job it is.
_CONTAINER = "listing-price-watch"
_CONTAINER_DESCRIPTION = "The asking price on the listing I'm watching, as it stands."

# The calls the routine makes, in order — what the stored program must read back as under the
# strict rendered dialect, and therefore what each cycle's tool surface is scoped to.
_PROGRAM_CALLS = ("browse", "collection_write")

# The job's cadence.  Stated because a configured collection has one, though the cycles are
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=HOURLY"

# The key every arm's write lands under.  A STABLE key with a moving value is what gives the
# write gate something to compare: the same reading twice is `KEY_EXISTS_UNCHANGED`, which is
# the structural silence cycle 2 is about.
_KEY = "asking price"


class Listing(NamedTuple):
    """One instance of the theme: an item, the page it lives on, and the two readings.

    ``token`` is the distinctive substring the canned browser matches the request url on, so
    each arm serves its own page and only its own."""

    token: str
    item: str
    url: str
    quiet_price: str
    moved_price: str


# Five listings, five urls, five prices.  Deliberately unlike each other — a lantern, a chair,
# a bicycle, a kettle, a rug — so an arm that only works for one shape of item shows up as a
# divergence rather than as a uniformly passing cohort.
LISTINGS = (
    Listing(
        "keel-lantern",
        "Keel Lantern, brass",
        "https://faux-market.example/keel-lantern",
        "84 zorkmids",
        "72 zorkmids",
    ),
    Listing(
        "harbour-chair",
        "Harbour Chair, teak",
        "https://faux-market.example/harbour-chair",
        "220 zorkmids",
        "195 zorkmids",
    ),
    Listing(
        "quay-bicycle",
        "Quay Bicycle, six-speed",
        "https://faux-market.example/quay-bicycle",
        "410 zorkmids",
        "455 zorkmids",
    ),
    Listing(
        "copper-kettle",
        "Copper Kettle, seamed",
        "https://faux-market.example/copper-kettle",
        "37 zorkmids",
        "41 zorkmids",
    ),
    Listing(
        "arcade-rug",
        "Arcade Rug, hand-knotted",
        "https://faux-market.example/arcade-rug",
        "615 zorkmids",
        "580 zorkmids",
    ),
)


def _page(listing: Listing, price: str) -> CannedPage:
    """One listing's page at one price — the only thing that moves between cycle 2 and 3."""
    return CannedPage(
        match=listing.token,
        text=(
            f"Title: {listing.item} — {listing.url}\n"
            f"{listing.url}\n\n"
            f"{listing.item}\n"
            f"Price: {price}\n"
            "Collection from the quay, weekday afternoons.\n"
        ),
    )


def _world(listing: Listing) -> World:
    """This arm's ground: the page as it stands when the job is set up.

    ``keeps``/``excludes`` are EMPTY, and deliberately.  Those token sets back
    ``assert_something_from_each_page_was_written``, which asks whether a round kept something
    identifying each source — and what this job stores is the price, while the token that
    identifies the page lives in its url and its title.  Declaring the tokens would print
    "5 must-keep" in every report as though something verified them, and the only claim that
    could read them would be false of every correct sample.  A contract nothing reads is worse
    than no contract: it reads as a check that passed."""
    return World(
        name=listing.token,
        pages=(_page(listing, listing.quiet_price),),
        keeps=(),
        excludes=(),
    )


def _skill(listing: Listing) -> SkillDraft:
    """The routine the user taught, in the shape run-end extraction leaves behind.

    ONE shape for every arm — the same two steps, the same placeholders, the same attachment
    mark on the destination — with only the demonstrated value differing, which is what a
    routine pointed at a different listing looks like.  Byte-fixed apart from that value is the
    whole point: five different programs would be five different behaviours."""
    return SkillDraft(
        name="watch_listing_price",
        intent="Keep an eye on what this listing is asking, and tell me when it changes.",
        description="Read a listing page and record the price it is asking.",
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="browse",
                arguments={"queries": [listing.url], "extract": "the asking price"},
                substitutions=[
                    SkillSubstitution(
                        path=["queries", 0],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the listing page to read",
                    ),
                    SkillSubstitution(
                        path=["extract"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the asking price on the page",
                    ),
                ],
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_write",
                arguments={
                    "memory": _CONTAINER,
                    "entries": [{"key": _KEY, "content": listing.quiet_price}],
                },
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "content"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the asking price as the page states it",
                    ),
                ],
            ),
        ],
        parameters=[
            SkillParameter(
                name="listing",
                description="the listing page to read each run",
                value=listing.url,
            )
        ],
        source_run_id=_SEED_AUTHOR,
    )


def _program(listing: Listing) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps in its own order: the attachment bound to the container, the runtime join
    writing the bound value into the leaf the demonstration put its own value in, then the
    render."""
    skill = _skill(listing)
    values = {"listing": listing.url}
    attached = retarget_writes(skill.steps, _CONTAINER)
    joined = bind_parameters(attached, skill.parameters, values)
    return render_skill(joined, values)


def _seeder(listing: Listing):
    """The world an apply turn leaves for THIS arm: the routine in the registry, and a
    container configured from it — then every claim that world makes, asserted out loud.

    The probe is not ceremony.  A program the strict parser cannot read leaves the cycle with a
    surface of the terminator alone, and a cycle with no browse writes nothing for the most
    boring reason there is — which is the exact shape of a passing sample on the claims below.
    Each failure costs a live cycle per sample to not notice."""

    def seed(db: Database) -> None:
        skill = _skill(listing)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_program(listing),
            schedule=_SCHEDULE,
            notify=True,
            skill_name=slug_skill_name(skill.name),
            skill_params={"listing": listing.url},
        )
        _assert_the_watch_world(db, listing)

    return seed


def _assert_the_watch_world(db: Database, listing: Listing) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    name = slug_skill_name(_skill(listing).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"

    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert row.notify, "this case measures the notification, so the job must be notifying"
    assert row.skill_name == name, f"the job must run the taught routine, not {row.skill_name!r}"

    program = row.extraction_prompt or ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_PROGRAM_CALLS)))
    assert parsed == _PROGRAM_CALLS, (
        f"the stored program must read back as {list(_PROGRAM_CALLS)} under the rendered "
        f"dialect, got {list(parsed)} — program: {program!r}"
    )
    assert listing.url in program, (
        f"the runtime join must fill the browse leaf with {listing.url!r} — a cycle can only "
        f"fetch the page something it reads names.  Program: {program!r}"
    )
    assert f"'{_CONTAINER}'" in program, (
        f"the attachment must be bound to {_CONTAINER!r}.  Program: {program!r}"
    )
    assert not collection_entries(db, _CONTAINER), (
        "the container must be empty when the first cycle starts, so a write is exactly a "
        "new entry rather than a diff a claim has to compute"
    )


def _arm(listing: Listing) -> CycleArm:
    """One arm: this listing's job, and the three registers its cycles read.

    The register is re-installed between cycles because that is the whole point — the world
    MOVED between two runs of the same watch, and what the third cycle does about it is the
    contract."""
    quiet = [_page(listing, listing.quiet_price)]
    return CycleArm(
        text=f"{listing.item} — {listing.url} at {listing.quiet_price}, then {listing.moved_price}",
        seed=_seeder(listing),
        cycles=[quiet, quiet, [_page(listing, listing.moved_price)]],
        world=_world(listing),
    )


# ── The claims: SHAPE only, because no value is true of the cohort ───────────


def _cycle(sample: SampleObservation, index: int) -> str:
    """One position of this sample's cycle script."""
    shapes = sample.walk.split(", ")
    return shapes[index] if index < len(shapes) else ""


def _quiet_cycle_said_nothing(sample: SampleObservation, _world: World) -> Answer:
    """The same reading twice over is silence.

    Cycle 2 re-reads the page the baseline already recorded, so the write gate's
    `KEY_EXISTS_UNCHANGED` STOP is what makes no-news structurally silent — no write, and
    nothing entering a notification."""
    shape = _cycle(sample, 1)
    return shape == "quiet", f"the quiet cycle was {shape!r}, not silent"


def _moved_reading_was_written_and_told(sample: SampleObservation, _world: World) -> Answer:
    """A moved reading is a write AND one notification, in the same cycle."""
    shape = _cycle(sample, 2)
    return shape == "wrote+told", f"the change cycle was {shape!r}"


def _told_on_exactly_the_cycles_that_moved(sample: SampleObservation, _world: World) -> Answer:
    """The user is told on every cycle that moved the reading, and on no other.

    NOT "exactly one notification across the three": the container arrives from apply EMPTY,
    so cycle 1's first observation is a new key — a baseline, and a first observation is news.
    Measured, the modal script is ``wrote+told, quiet, wrote+told``: two notifications, both
    owed.  What is never owed is a notification on a cycle that moved nothing, which is the
    other half of this sentence and what the quiet-cycle claim states on its own.

    Read over the SEND QUEUE, which is what the user will actually receive — a cycle enqueues
    and the drainer is a separate schedule, so a pending-only read of the outgoing messages
    reports a delivered notification as silence."""
    mismatched = [
        shape
        for shape in sample.walk.split(", ")
        if shape and (("told" in shape) != ("wrote" in shape))
    ]
    return not mismatched, f"told and wrote disagree on {mismatched}: {sample.walk!r}"


def _one_entry_under_one_key(sample: SampleObservation, _world: World) -> Answer:
    """The watch keeps ONE fact, rewritten — not a new row per reading.

    A watch that appends grows without bound and the user is told about a value they already
    have; this is the claim that says the moving reading landed on the standing key."""
    keys = {entry.key for entry in sample.entries}
    return len(
        sample.entries
    ) == 1, f"holds {len(sample.entries)} entries under keys {sorted(keys)}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_only_when_the_reading_moves(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """One program, five listings: the write gate holds the same way on every one of them."""
    cohort = await collector_cycles_eval(
        case_id=_CASE_ID,
        model=model,
        collection=_CONTAINER,
        arms=[_arm(listing) for listing in LISTINGS],
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing.  A watch RUNS CYCLES; it does not move a machine, and it has three
    # cycle outcomes rather than one state.  Every one of them is read off persisted entries
    # and the send queue, which is store territory wearing a state label, so the claims below
    # are STORE claims and this category renders empty.  That empty section is the correct
    # report for a shape with no state, not an unrun checklist.
    #
    # "the harness drove three cycles" is not anywhere: it asserts what the FIXTURE did, which
    # is why it scored 15/15 in every cohort it ever rendered and could not fail.  Its real
    # failure mode is already an exclusion, and the fixture's own shape is guarded by the loud
    # probe and by ``make check``, where a precondition belongs.

    # STORE — what the container and the send queue hold, cycle by cycle and at the end
    cohort.claim(
        "state: the quiet cycle wrote nothing and said nothing",
        _quiet_cycle_said_nothing,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the moved reading was written and told once",
        _moved_reading_was_written_and_told,
        SpecCategory.STORE,
    )
    cohort.assert_the_store_holds_an_entry()
    cohort.claim(
        "state: the watch keeps one fact, rewritten", _one_entry_under_one_key, SpecCategory.STORE
    )
    cohort.claim(
        "state: the user was told on exactly the cycles that moved the reading",
        _told_on_exactly_the_cycles_that_moved,
        SpecCategory.STORE,
    )

    # PROVENANCE — nothing in the container came from anywhere but the pages it read
    cohort.assert_every_stored_entry_traces_to_the_world()

    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED, REPLY_SPREAD)
