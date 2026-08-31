"""A price watch runs its cycles: write when the reading moves, silence when it does not
(#2017, the collector half of the cohort port).

**The behaviour, as one sentence.** Penny re-reads the job's page each cycle, writes the value
only when it has changed, and tells the user only then — the same reading twice over is
silence, a moved reading is a write and one notification.

**The arms are five wordings of one instruction, over one set of facts.**  A collector has no
user turn, so its natural-language surface is the ``extract`` instruction its rendered program
carries and the prose of the page that answers it.  That instruction is written by the
``SkillSubstitution`` on the ``extract`` path and reaches the model through the shipped
instantiation seam — ``retarget_writes`` → ``bind_parameters`` → ``render_skill`` — so varying
it varies a draw rather than a hand-authored render.  Which page each arm reads is the other
half: same url, same product, same price, five catalogue-grade prose variants around a
byte-identical datum line.

**The FACTS are constant, and the claims hinge on them.**  One listing, one url, one pair of
prices: ``$499`` before the change and ``$449`` after.  So this case can say what the store
holds by name — after the cycle that moved the reading the watch holds ``$449`` and no longer
holds ``$499`` — which is the claim a cohort of five different listings could never make.  The
two prices are mutually exclusive (neither is a substring of the other), the rule
``test_collector_enactment.py``'s ``_WatchedFact`` states, because the claim asserts one is
present and the other gone.

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

Report-only (``min_pass_rate=None``).  All content is synthetic — the house listing fixture on
an invented marketplace — because the repo is public.
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
from penny.tests.eval.utils.fixtures import LISTING_URL, CannedPage, datum
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "watch-writes-only-when-the-reading-moves"
_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The one collection every arm's job is configured on.  One name, because one job.
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

# ── The facts, held constant across every arm ────────────────────────────────
#
# One listing, one url, one pair of prices — the house listing every other case in the suite
# is built on, so a reader who knows `LISTING_URL` knows this world already.  Constant is what
# lets the claims below name a value: five listings would force every one of them back to a
# shape claim, and a shape claim cannot tell a watch that recorded the right price from one
# that recorded a plausible number.
_ITEM = "Aurora Deck 2"
_MATCH = "aurora-deck-2"

# The two readings.  MUTUALLY EXCLUSIVE — neither is a substring of the other, in the bare form
# or in the instruction-labelled pair a cycle may store since #1918 — because the change claim
# asserts one is present and the other gone.  ``$499`` and ``$4499`` would not have been.
_BASELINE_PRICE = "$499"
_MOVED_PRICE = "$449"

# The watched line itself, byte-identical on all five pages, and labelled as THE current price
# so the neighbouring items' prices further down each page cannot be read as it (round 5 of
# the enactment beat measured a wrong-price grab against an unlabelled one).
_DATUM = f"Current price: {_BASELINE_PRICE}"
_MOVED_DATUM = f"Current price: {_MOVED_PRICE}"


class Reading(NamedTuple):
    """One arm: how the job's ``extract`` step words what to look for, and the page prose that
    answers it.

    ``name`` is what this arm is called wherever it is identified — its world's name, and the
    row a reader opens when the report says a phrasing diverged.  ``extract`` is the
    substitution description that becomes ``extract={…}`` in the rendered program, which is the
    only natural language a collector cycle is handed.  ``body`` is this arm's whole page."""

    name: str
    extract: str
    body: str


_SPEC_LINK = f"[{_ITEM} specification sheet]({LISTING_URL}/spec)"
_LISTING_LINK = f"[{_ITEM} listing]({LISTING_URL})"
_HEAD = f"Title: {_ITEM} — handheld console | faux-market\n{LISTING_URL}\n\n"


# Five catalogue-grade pages: same url, same product, same price, five voices.  Each carries
# far more than the job needs — a seller blurb, a specification block, neighbouring items with
# their OWN prices, housekeeping notes — because a real page does, and because a page thin
# enough to answer only the asked question cannot tell restraint from luck.  The neighbours are
# the load-bearing part: a cycle that grabs a price is only demonstrably reading the RIGHT one
# where wrong ones are on the page to grab.  Every markdown link sits at the CENTRE of its
# block, since a search-shaped read is trimmed to ±2 lines around each solo link.
READINGS = (
    Reading(
        "current-price",
        "the current price",
        _HEAD + f"{_ITEM} (open box, tested). Sold by a fictional reseller; stock moves weekly.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: nebula_resale (4.9 stars). The asking price last changed nine days ago.\n"
        "Condition: open box, tested. Dispatched within two working days.\n"
        "\n"
        "Specification\n"
        "7-inch 1200p display with 512 GB of storage\n"
        f"{_SPEC_LINK}\n"
        "About four hours of battery per charge; the cell is replaceable.\n"
        "Weight 640g · Original box and charger · One controller included\n"
        "\n"
        "Others from this seller\n"
        "Aurora Deck 1, refurbished — $329\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Nebula Dock, charging stand — $59\n"
        "Both ship from the same fictional warehouse.\n"
        "\n"
        "Returns are accepted within thirty days; return postage is the buyer's.\n",
    ),
    Reading(
        "asking-price",
        "the asking price on the page",
        _HEAD
        + "A fictional marketplace listing. The shop reprices this one whenever the shelf moves.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Offered by driftwood_games (4.8 stars). The asking price is reviewed every Monday.\n"
        "Condition: open box, tested. Two working days to dispatch.\n"
        "\n"
        "What you get\n"
        f"{_ITEM} handheld · 7-inch 1200p display · 512 GB\n"
        f"{_SPEC_LINK}\n"
        "The battery runs about four hours; charger and one controller are in the box.\n"
        "Weight 640g · Original packaging · No trade-ins accepted on this listing\n"
        "\n"
        "Also listed by this shop\n"
        "Aurora Deck Lite, boxed — $279\n"
        "[Aurora Deck Lite](https://faux-market.example/aurora-deck-lite)\n"
        "Spare stylus, twin pack — $19\n"
        "Everything ships from one fictional depot.\n"
        "\n"
        "Thirty-day returns. The buyer pays return postage.\n",
    ),
    Reading(
        "asking-right-now",
        "what the listing is asking right now",
        _HEAD + "Listed by a fictional trade-in counter that re-prices its shelf every few days.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: quayside_swap (4.7 stars). What the counter asks moves with what it pays.\n"
        "Condition: open box, tested. Ready to dispatch inside two working days.\n"
        "\n"
        "Hardware\n"
        "A 7-inch 1200p screen and 512 GB of onboard storage\n"
        f"{_SPEC_LINK}\n"
        "Around four hours of play per charge; a workshop can swap the battery.\n"
        "Weight 640g · Boxed as it came · Charger included\n"
        "\n"
        "On the same shelf\n"
        "Aurora Deck 1, tested — $315\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Moulded travel case — $44\n"
        "The counter posts its shelf every morning.\n"
        "\n"
        "Returns within thirty days, postage paid by the buyer.\n",
    ),
    Reading(
        "listed-today",
        "the price it is listed at today",
        _HEAD + "A fictional clearance warehouse. Today's shelf, republished each morning.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: pier_clearance (4.6 stars). The asking price is set each day at opening.\n"
        "Condition: open box, tested. Two working days from order to dispatch.\n"
        "\n"
        "Specification\n"
        "Display 7-inch 1200p · Storage 512 GB\n"
        f"{_SPEC_LINK}\n"
        "Roughly four hours of battery; the pack is a serviceable part.\n"
        "Weight 640g · Original box and charger · One controller\n"
        "\n"
        "Today's other handhelds\n"
        "Aurora Deck Lite, open box — $265\n"
        "[Aurora Deck Lite](https://faux-market.example/aurora-deck-lite)\n"
        "Screen guard, twin pack — $12\n"
        "The whole shelf ships from one fictional warehouse.\n"
        "\n"
        "Returns accepted for thirty days; the buyer covers return postage.\n",
    ),
    Reading(
        "listing-shows",
        "the price this listing shows",
        _HEAD
        + "A fictional consignment listing. The page shows whatever the owner last agreed to.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: lantern_consign (4.9 stars). The asking price is whatever this page shows.\n"
        "Condition: open box, tested. Dispatch within two working days.\n"
        "\n"
        "Specification sheet\n"
        "7-inch 1200p display · 512 GB storage · 640g\n"
        f"{_SPEC_LINK}\n"
        "The battery lasts about four hours and is replaceable at a workshop.\n"
        "Charger, one controller and the original box are included.\n"
        "\n"
        "Elsewhere in this consignment\n"
        "Aurora Deck 1, boxed — $340\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Dock and cable set — $69\n"
        "All consigned stock ships from one fictional store.\n"
        "\n"
        "Thirty days to return; return postage is the buyer's.\n",
    ),
)


def _page(reading: Reading) -> CannedPage:
    """This arm's page as it stands before the change — the baseline and the quiet cycle."""
    return CannedPage(match=_MATCH, text=reading.body)


def _moved_page(reading: Reading) -> CannedPage:
    """The same page with the watched datum — and only the watched datum — moved.

    Derived by the shared ``datum`` edit rather than rebuilt from a template, so the two sides
    of the pair are one text and one span: the edit RAISES unless the old line appears exactly
    once, which is what a rebuilt twin cannot check."""
    return datum(_page(reading), _DATUM, _MOVED_DATUM)


def _world(reading: Reading) -> World:
    """This arm's ground: the page as it stands when the job is set up.

    ``keeps``/``excludes`` are EMPTY, and deliberately.  Those token sets back
    ``assert_something_from_each_page_was_written``, which this case never calls — and there is
    nothing they could usefully hold: a ``keeps`` token has to appear on ONE page so a stored
    copy says which page it came from, and every arm here reads the same url, the same product
    and the same price.  Declaring tokens anyway would print "5 must-keep" in every report as
    though something had verified them.  A contract nothing reads is worse than no contract: it
    reads as a check that passed."""
    return World(
        name=reading.name,
        pages=(_page(reading),),
        keeps=(),
        excludes=(),
    )


def _skill(reading: Reading) -> SkillDraft:
    """The routine the user taught, in the shape run-end extraction leaves behind.

    ONE shape for every arm — the same two steps, the same placeholders, the same bound url,
    the same attachment mark on the destination.  The one thing that differs is the
    ``extract`` substitution's DESCRIPTION, which is what ``render_skill`` prints into the
    program and therefore the only natural language a cycle reads.  The demonstrated
    ``arguments["extract"]`` stays constant across the arms because it never renders: the
    labeller's description replaces it at the seam."""
    return SkillDraft(
        name="watch_listing_price",
        intent="Keep an eye on what this listing is asking, and tell me when it changes.",
        description="Read a listing page and record the price it is asking.",
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="browse",
                arguments={"queries": [LISTING_URL], "extract": "the asking price"},
                substitutions=[
                    SkillSubstitution(
                        path=["queries", 0],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the listing page to read",
                    ),
                    SkillSubstitution(
                        path=["extract"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description=reading.extract,
                    ),
                ],
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_write",
                arguments={
                    "memory": _CONTAINER,
                    "entries": [{"key": _KEY, "content": _BASELINE_PRICE}],
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
                value=LISTING_URL,
            )
        ],
        source_run_id=_SEED_AUTHOR,
    )


def _program(reading: Reading) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps in its own order: the attachment bound to the container, the runtime join
    writing the bound value into the leaf the demonstration put its own value in, then the
    render."""
    skill = _skill(reading)
    values = {"listing": LISTING_URL}
    attached = retarget_writes(skill.steps, _CONTAINER)
    joined = bind_parameters(attached, skill.parameters, values)
    return render_skill(joined, values)


def _extract_slot(reading: Reading) -> str:
    """How this arm's instruction renders inside the program — the one span that moves."""
    return f"extract={{{reading.extract}}}"


def _seeder(reading: Reading):
    """The world an apply turn leaves for THIS arm: the routine in the registry, and a
    container configured from it — then every claim that world makes, asserted out loud.

    The probe is not ceremony.  A program the strict parser cannot read leaves the cycle with a
    surface of the terminator alone, and a cycle with no browse writes nothing for the most
    boring reason there is — which is the exact shape of a passing sample on the claims below.
    Each failure costs a live cycle per sample to not notice."""

    def seed(db: Database) -> None:
        skill = _skill(reading)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_program(reading),
            schedule=_SCHEDULE,
            notify=True,
            skill_name=slug_skill_name(skill.name),
            skill_params={"listing": LISTING_URL},
        )
        _assert_the_watch_world(db, reading)

    return seed


def _assert_the_watch_world(db: Database, reading: Reading) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    name = slug_skill_name(_skill(reading).name)
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
    assert LISTING_URL in program, (
        f"the runtime join must fill the browse leaf with {LISTING_URL!r} — a cycle can only "
        f"fetch the page something it reads names.  Program: {program!r}"
    )
    assert f"'{_CONTAINER}'" in program, (
        f"the attachment must be bound to {_CONTAINER!r}.  Program: {program!r}"
    )
    assert _extract_slot(reading) in program, (
        f"this arm's instruction must reach the model as {_extract_slot(reading)!r} — the "
        f"extract description is the whole arm axis.  Program: {program!r}"
    )
    assert not collection_entries(db, _CONTAINER), (
        "the container must be empty when the first cycle starts, so a write is exactly a "
        "new entry rather than a diff a claim has to compute"
    )


def _arm(reading: Reading) -> CycleArm:
    """One arm: the one listing read under this arm's instruction, and the three registers its
    cycles read.

    The register is re-installed between cycles because that is the whole point — the world
    MOVED between two runs of the same watch, and what the third cycle does about it is the
    contract.  ``text`` is the instruction, because that is what makes this arm this arm."""
    quiet = [_page(reading)]
    return CycleArm(
        text=reading.extract,
        seed=_seeder(reading),
        cycles=[quiet, quiet, [_moved_page(reading)]],
        world=_world(reading),
    )


# ── The claims ───────────────────────────────────────────────────────────────


def _cycle(sample: SampleObservation, index: int) -> str:
    """One position of this sample's cycle script."""
    shapes = sample.walk.split(", ")
    return shapes[index] if index < len(shapes) else ""


def _quiet_cycle_said_nothing(sample: SampleObservation, _world: World) -> Answer:
    """The same reading twice over is silence.

    Cycle 2 re-reads the page the baseline already recorded, so the write gate's
    `KEY_EXISTS_UNCHANGED` STOP is what makes no-news structurally silent — no write, and
    nothing entering a notification.  It is also what says the baseline recorded the RIGHT
    value: the STOP fires only where the stored value equals the one just read, so a silent
    cycle 2 is a cycle 1 that stored what the page said."""
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


def _holds_the_price_the_page_moved_to(sample: SampleObservation, _world: World) -> Answer:
    """After the last cycle the watch holds the price the page moved TO, and no longer the one
    it moved FROM.

    The facts are the same on every arm, so this claim can name them: ``$449`` present,
    ``$499`` absent.  Both halves are needed and neither is sufficient — a watch that appended
    the new price beside the old one holds ``$449`` while still telling the user something that
    is no longer true, and a watch that never re-read the page holds neither.  Matched as
    substrings of the whole entry, key and content together, so a cycle that stored the value
    with its label (the instruction-labelled pair a browse result hands back since #1918) still
    reads as having stored it."""
    held = sample.stored_text
    ok = _MOVED_PRICE in held and _BASELINE_PRICE not in held
    return ok, f"expected {_MOVED_PRICE} and not {_BASELINE_PRICE}; the watch holds {held!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_only_when_the_reading_moves(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """One job, five wordings of its instruction: the write gate holds the same way on all of
    them, and the value it holds at the end is the one the page moved to."""
    cohort = await collector_cycles_eval(
        case_id=_CASE_ID,
        model=model,
        collection=_CONTAINER,
        arms=[_arm(reading) for reading in READINGS],
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
        f"state: the watch holds {_MOVED_PRICE} and no longer {_BASELINE_PRICE}",
        _holds_the_price_the_page_moved_to,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the user was told on exactly the cycles that moved the reading",
        _told_on_exactly_the_cycles_that_moved,
        SpecCategory.STORE,
    )

    # PROVENANCE — nothing in the container came from anywhere but the pages it read
    cohort.assert_every_stored_entry_traces_to_the_world()

    # What is MEASURED.  Variance is ORTHOGONAL to correctness — it surfaces samples unlike
    # the pack, never whether a value is right — so `ENTRIES_STORED` belongs here even though
    # `_one_entry_under_one_key` asserts the same count.  The two answer different questions:
    # the claim says THIS RUN WAS WRONG, the feature says THIS SAMPLE IS UNLIKE THE OTHERS, and
    # a sample storing zero or two entries is an outlier worth opening whichever way the claim
    # went.  Being covered by a claim is not a reason to drop an axis.
    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED, REPLY_SPREAD)
