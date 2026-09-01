"""A price watch, one cycle at a time: the first reading, an unchanged one, a moved one
(#2017, the collector half of the cohort port).

**Three cases, because there are three behaviours.**  A watch does three different things and
each is a separate test with its own entry condition, its own single cycle and its own
assertions.  What selects the behaviour is the ENTRY CONDITION — what the collection already
holds when the cycle starts — not a script of cycles inside one test body:

* ``watch-writes-the-first-reading`` — the collection is EMPTY and the page says ``$499``.
  A first observation is news: record it, and say so once.
* ``watch-stays-quiet-when-the-reading-has-not-moved`` — the collection already holds
  ``$499`` and the page still says ``$499``.  The write gate STOPS: nothing is written and
  nothing reaches the send queue.
* ``watch-writes-and-tells-when-the-reading-moves`` — the collection holds ``$499`` and the
  page now says ``$449``.  The standing key is rewritten and the user is told once.

The middle case is the only one that can exist at all: the write gate's
``KEY_EXISTS_UNCHANGED`` STOP fires on a SECOND reading of a value already stored, so
structural silence has nothing to fire on until something is in the collection.  Preseeding it
is what gives that cycle its subject — and it is preseeded through the store's own write path,
under the key the program writes to, so the gate compares against a row of exactly the shape a
real cycle would have left.

**The arms are five wordings of one instruction, over one set of facts.**  A collector has no
user turn, so its natural-language surface is the ``extract`` instruction in its rendered
program and the prose of the page that answers it.  That instruction is written by the
``SkillSubstitution`` on the ``extract`` path and reaches the model through the shipped
instantiation seam — ``retarget_writes`` → ``bind_parameters`` → ``render_skill`` — so varying
it varies a draw rather than a hand-authored render.  Which page each arm reads is the other
half: same url, same product, same price, five catalogue-grade prose variants around a
byte-identical datum line.

**The FACTS are constant, and the claims hinge on them.**  One listing, one url, one pair of
prices: ``$499`` before the change and ``$449`` after.  So these cases can say what the store
holds by name.  The two prices are mutually exclusive — neither is a substring of the other,
the rule ``test_collector_enactment.py``'s ``_WatchedFact`` states — because the moved case
asserts one is present and the other gone.

**The LANDED category is empty on all three, and that is the correct report.**  ``LANDED`` is
read off the conversation machine's walk, and a collector moves no conversation machine.  The
run record's outcome and its stop reason are tempting to file there, but they are RECORD
FIELDS, which ``STORE`` covers literally — so that is where they are claimed, and the state
section renders empty rather than having something invented to fill it.

``test_collector_enactment.py``'s fifteen cases are five jobs × three cycles plus their
notify/quiet pairs — the same claim over five different PROGRAMS.  Those are different
routines, therefore different behaviours, and they split rather than pool; collapsing them is
#2007's, not this file's.

Report-only (``min_pass_rate=None``).  All content is synthetic — the house listing fixture on
an invented marketplace — because the repo is public.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import WRITE_GATE_STOP_REASONS, RunOutcome, WriteGateOutcome
from penny.database import Database
from penny.database.memory import EntryInput
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
    seeded_run_id,
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

_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid the entry condition down.  A SEEDED id, structurally distinguishable from
# a live ``uuid4().hex``, so every "what did this sample write" reader excludes the preseeded
# row at the one chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("watch-entry-condition")

# The one collection every arm's job is configured on.  One name, because one job.
_CONTAINER = "listing-price-watch"
_CONTAINER_DESCRIPTION = "The asking price on the listing I'm watching, as it stands."

# The calls the routine makes, in order — what the stored program must read back as under the
# strict rendered dialect, and therefore what each cycle's tool surface is scoped to.
_PROGRAM_CALLS = ("browse", "collection_write")

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=HOURLY"

# The key the program writes under, and therefore the key an entry condition must use: the
# write gate compares a candidate against what is stored UNDER THE SAME KEY, so a preseeded
# row under any other key would leave the unchanged case with nothing to be unchanged against.
_KEY = "asking price"

# The STOP a no-change cycle ends on, as the run record states it — read from the shipped
# table rather than restated, so a reworded reason cannot silently stop matching.
_STOP_REASON = WRITE_GATE_STOP_REASONS[WriteGateOutcome.KEY_EXISTS_UNCHANGED]

# ── The facts, held constant across every arm ────────────────────────────────
#
# One listing, one url, one pair of prices — the house listing every other case in the suite
# is built on, so a reader who knows `LISTING_URL` knows this world already.  Constant is what
# lets the claims below name a value: five listings would force every one of them back to a
# shape claim, and a shape claim cannot tell a watch that recorded the right price from one
# that recorded a plausible number.
_ITEM = "Aurora Deck 2"
_MATCH = "aurora-deck-2"

# The two readings, in their two roles.  The AMOUNT is the identity of the reading and what
# every claim matches on; the PRICE is how a listing displays it, and it is what the pages
# carry.  Keeping the two apart is what lets a page look like a real page while a claim answers
# the question it is actually asking.
#
# Both forms are MUTUALLY EXCLUSIVE — neither amount is a substring of the other, and neither
# price is a substring of the other, in the bare form or in the instruction-labelled pair a
# cycle may store since #1918 — because the moved case asserts one is present and the other
# gone.  ``499`` and ``4499`` would not have been.
_BASELINE_AMOUNT = "499"
_MOVED_AMOUNT = "449"
_BASELINE_PRICE = f"${_BASELINE_AMOUNT}"
_MOVED_PRICE = f"${_MOVED_AMOUNT}"

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


class WatchCase(NamedTuple):
    """One behaviour: what the collection holds when the cycle starts, and what the page says.

    Two fields, because those two settings are the whole difference between the three cases.
    ``stored`` is the price already in the collection, or ``None`` for a collection that
    arrives empty; ``shows`` is the price on the page the single cycle reads."""

    case_id: str
    stored: str | None
    shows: str


FIRST_READING = WatchCase("watch-writes-the-first-reading", None, _BASELINE_PRICE)
UNCHANGED_READING = WatchCase(
    "watch-stays-quiet-when-the-reading-has-not-moved", _BASELINE_PRICE, _BASELINE_PRICE
)
MOVED_READING = WatchCase(
    "watch-writes-and-tells-when-the-reading-moves", _BASELINE_PRICE, _MOVED_PRICE
)
CASES = (FIRST_READING, UNCHANGED_READING, MOVED_READING)


# The one sentence each case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED name of where the behaviour happens, and the three
# share an opening clause and differ only at the entry condition — that parallelism is what the
# three-case split reads like out loud.  The case id is a filename; this is the contract.
_BEHAVIOUR = {
    FIRST_READING.case_id: (
        "In a price-watch collector, when the job runs for the first time and its collection "
        "is still empty, Penny records the price the page shows and tells the user once — a "
        "first observation is news."
    ),
    UNCHANGED_READING.case_id: (
        "In a price-watch collector, when the page's price has not changed since she last "
        "recorded it, Penny writes nothing and says nothing."
    ),
    MOVED_READING.case_id: (
        "In a price-watch collector, when the page's price has changed since she last recorded "
        "it, Penny replaces the price she was holding with the new one and tells the user "
        "exactly once."
    ),
}

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


def _page(case: WatchCase, reading: Reading) -> CannedPage:
    """The page THIS case's cycle reads, on THIS arm's prose.

    The moved variant is derived from the arm's own page by the shared single-span ``datum``
    edit rather than rebuilt from a template, so the two sides of the pair are one text and one
    span: the edit RAISES unless the watched line appears exactly once, which is what a rebuilt
    twin cannot check."""
    page = CannedPage(match=_MATCH, text=reading.body)
    return page if case.shows == _BASELINE_PRICE else datum(page, _DATUM, _MOVED_DATUM)


def _world(case: WatchCase, reading: Reading) -> World:
    """This arm's ground: the page its single cycle reads.

    ``keeps``/``excludes`` are EMPTY, and deliberately.  Those token sets back
    ``assert_something_from_each_page_was_written``, which these cases never call — and there is
    nothing they could usefully hold: a ``keeps`` token has to appear on ONE page so a stored
    copy says which page it came from, and every arm here reads the same url, the same product
    and the same price.  Declaring tokens anyway would print "5 must-keep" in every report as
    though something had verified them.  A contract nothing reads is worse than no contract: it
    reads as a check that passed."""
    return World(
        name=reading.name,
        pages=(_page(case, reading),),
        keeps=(),
        excludes=(),
    )


def _skill(reading: Reading) -> SkillDraft:
    """The routine the user taught, in the shape run-end extraction leaves behind.

    ONE shape for every arm and every case — the same two steps, the same placeholders, the
    same bound url, the same attachment mark on the destination.  The one thing that differs is
    the ``extract`` substitution's DESCRIPTION, which is what ``render_skill`` prints into the
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


def _seeder(case: WatchCase, reading: Reading):
    """The world this case's cycle starts in: the routine in the registry, a container
    configured from it, and the ENTRY CONDITION — then every claim that world makes, asserted
    out loud.

    The entry condition goes down through the store's own write path, under the key the program
    writes to, so the row carries the stamps and the embeddings a real cycle's write would have
    left.  Hand-inserting it would give the write gate something to compare against that no
    cycle could have produced, and the case would then measure the gate against a shape
    production never sees.

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
        if case.stored is not None:
            memory = db.memory(_CONTAINER)
            assert memory is not None, f"the job's container {_CONTAINER!r} must exist to seed"
            memory.write(
                [EntryInput(key=_KEY, content=case.stored)],
                author=_SEED_AUTHOR,
                run_id=_SEED_RUN,
            )
        _assert_the_watch_world(db, case, reading)

    return seed


def _assert_the_watch_world(db: Database, case: WatchCase, reading: Reading) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    name = slug_skill_name(_skill(reading).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"

    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert row.notify, "these cases measure the notification, so the job must be notifying"
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
    _assert_the_entry_condition(db, case)


def _assert_the_entry_condition(db: Database, case: WatchCase) -> None:
    """The state the cycle starts from, which is what selects the behaviour being measured.

    An empty collection makes a first observation news; a collection already holding the
    reading is the only state the ``KEY_EXISTS_UNCHANGED`` STOP can fire against.  Getting this
    wrong silently turns one case into another — the quiet case run against an empty container
    would write, pass nothing, and look like a model failure — so it is asserted here rather
    than assumed."""
    held = collection_entries(db, _CONTAINER)
    if case.stored is None:
        assert not held, (
            "the container must be empty when the cycle starts, so a write is exactly a new "
            f"entry rather than a diff a claim has to compute — it holds {held}"
        )
        return
    assert held == {_KEY: case.stored}, (
        f"the cycle must start from exactly {{{_KEY!r}: {case.stored!r}}} — the write gate "
        f"compares a candidate against what is stored under the same key, and it holds {held}"
    )


def _arm(case: WatchCase, reading: Reading) -> CycleArm:
    """One arm of one case: this case's entry condition and page, under this arm's instruction.

    ``text`` is the instruction, because that is what makes this arm this arm."""
    return CycleArm(
        text=reading.extract,
        seed=_seeder(case, reading),
        pages=[_page(case, reading)],
        world=_world(case, reading),
    )


def _arms(case: WatchCase) -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over five prose variants."""
    return [_arm(case, reading) for reading in READINGS]


# ── The claims ───────────────────────────────────────────────────────────────
#
# Every one of them reads END STATE: what the collection holds, what the run record says, and
# what reached the send queue.  Nothing reads a tool name or an ordering — a skill is an
# arbitrary tool sequence, so a route is measured in section B and never asserted.


def _held_text(sample: SampleObservation) -> str:
    """The collection as the cycle left it, key and content together.

    The WHOLE entry, because a fact in the key and a blurb in the body is a perfectly good way
    to store it, and a content-only read once reported a 25/32 model failure that was entirely
    its own bug."""
    return " ".join(f"{key} {content}" for key, content in sorted(sample.held.items()))


def _one_entry(sample: SampleObservation, _world: World) -> Answer:
    """The watch keeps ONE fact.  A watch that appends grows without bound and the user is
    told about a value they already have."""
    return len(sample.held) == 1, f"holds {len(sample.held)} entries: {sorted(sample.held)}"


def _holds(amount: str):
    """A claim that the collection holds one named reading.

    The facts are the same on every arm, so the claim can name the value rather than asserting
    that something was stored: a shape claim cannot tell a watch that read the right price from
    one that produced a plausible number.

    Matched on the bare AMOUNT.  The identity of a reading is its number — that is what moves on
    the page, and what a watch exists to track — while which currency notation a draw happened to
    keep is not a question this claim answers.  ``449`` and ``499`` are mutually exclusive as
    numerals, so naming the amount costs the claim nothing it was measuring."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_text(sample)
        return amount in held, f"expected {amount}; the collection holds {sample.held}"

    return answer


def _holds_the_moved_price_only(sample: SampleObservation, _world: World) -> Answer:
    """The collection holds the reading the page moved TO, and no longer the one it moved FROM.

    Both halves are needed and neither is sufficient — a watch that appended the new reading
    beside the old one holds ``449`` while still telling the user something that is no longer
    true, and a watch that never re-read the page holds neither.

    Matched on the bare AMOUNTS, for the reason ``_holds`` gives: the identity of a reading is
    its number, and the notation a draw kept is not this claim's question."""
    held = _held_text(sample)
    ok = _MOVED_AMOUNT in held and _BASELINE_AMOUNT not in held
    return ok, (
        f"expected {_MOVED_AMOUNT} and not {_BASELINE_AMOUNT}; the collection holds {held!r}"
    )


def _closed_having_worked(sample: SampleObservation, _world: World) -> Answer:
    """The run's own determination: it completed and it CHANGED something.

    A record field, read literally off ``promptlog.run_outcome`` — ``worked`` is defined as
    "completed and changed something", which is what "the cycle wrote" means without reading a
    tool name.  ``no_work`` is a clean close that changed nothing and ``failed`` is a bail, and
    both are the wrong end state here."""
    closed = sample.run_outcome or "—"
    detail = f" — {sample.run_reason}" if sample.run_reason else ""
    return sample.run_outcome == RunOutcome.WORKED.value, f"the run closed {closed}{detail}"


def _stopped_on_the_unchanged_value(sample: SampleObservation, _world: World) -> Answer:
    """The run stopped at the write chokepoint, on the reason the shipped table names.

    This is what makes silence STRUCTURAL rather than a judgement the model makes each cycle:
    the gate compares the candidate against the stored value and raises its STOP on the very
    call that would otherwise complete the program, so the notification the framework would
    have entered is never entered at all."""
    return sample.run_reason == _STOP_REASON, f"the run closed {sample.run_reason or '—'}"


def _told_once(sample: SampleObservation, _world: World) -> Answer:
    """The user was told, and told once.

    Read over the SEND QUEUE, which is what the user will actually receive — a cycle enqueues
    and the drainer is a separate schedule, so a pending-only read of outgoing messages reports
    a delivered notification as silence."""
    count = len(sample.notifications)
    return count == 1, f"{count} messages reached the send queue: {sample.notifications}"


def _told_nothing(sample: SampleObservation, _world: World) -> Answer:
    """Nothing reached the send queue.  No news is not a message saying there is no news."""
    return not sample.notifications, f"sent {sample.notifications}"


# ── first reading: an empty collection, and a page ───────────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_the_first_reading(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A collection arrives from apply EMPTY, so its first observation is a new key — a
    baseline, and a first observation is news."""
    cohort = await collector_cycles_eval(
        case_id=FIRST_READING.case_id,
        behaviour=_BEHAVIOUR[FIRST_READING.case_id],
        model=model,
        collection=_CONTAINER,
        arms=_arms(FIRST_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.  A collector moves no conversation machine,
    # and the run record's fields are STORE claims below rather than a category borrowed.

    # STORE
    cohort.claim("state: the collection holds one entry", _one_entry, SpecCategory.STORE)
    cohort.claim(
        f"state: the entry it holds carries {_BASELINE_AMOUNT}",
        _holds(_BASELINE_AMOUNT),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the run closed having changed something", _closed_having_worked, SpecCategory.STORE
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()

    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED, REPLY_SPREAD)


# ── unchanged reading: the collection already holds what the page says ───────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_stays_quiet_when_the_reading_has_not_moved(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """The same reading twice over is silence — no write, and nothing entering a notification.

    REPLY_SPREAD is not measured here, and that is the point of the case: a correct cohort
    sends nothing, so a reply-spread reading would be blind on every sample by construction and
    would print a number where there is no measurement.  A sample that DID speak is caught by
    the send-queue claim and shows in ``transitions`` as ``quiet+told``."""
    cohort = await collector_cycles_eval(
        case_id=UNCHANGED_READING.case_id,
        behaviour=_BEHAVIOUR[UNCHANGED_READING.case_id],
        model=model,
        collection=_CONTAINER,
        arms=_arms(UNCHANGED_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.

    # STORE
    cohort.claim("state: the collection still holds one entry", _one_entry, SpecCategory.STORE)
    cohort.claim(
        f"state: the entry it holds still carries {_BASELINE_AMOUNT}",
        _holds(_BASELINE_AMOUNT),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the run stopped on the value being unchanged",
        _stopped_on_the_unchanged_value,
        SpecCategory.STORE,
    )
    cohort.claim("state: nothing reached the send queue", _told_nothing, SpecCategory.STORE)

    # PROVENANCE — a correct cycle wrote nothing, so this claim is TRUE of it rather than
    # unasked; what it catches is a quiet cycle that wrote something the page never said.
    cohort.assert_every_stored_entry_traces_to_the_world()

    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED)


# ── moved reading: the collection holds the old price, the page shows a new one ──


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_and_tells_when_the_reading_moves(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A moved reading is a write AND one notification, on the standing key."""
    cohort = await collector_cycles_eval(
        case_id=MOVED_READING.case_id,
        behaviour=_BEHAVIOUR[MOVED_READING.case_id],
        model=model,
        collection=_CONTAINER,
        arms=_arms(MOVED_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.

    # STORE
    cohort.claim(
        f"state: the collection holds {_MOVED_AMOUNT} and no longer {_BASELINE_AMOUNT}",
        _holds_the_moved_price_only,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the collection still holds one entry, rewritten", _one_entry, SpecCategory.STORE
    )
    cohort.claim(
        "state: the run closed having changed something", _closed_having_worked, SpecCategory.STORE
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()

    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED, REPLY_SPREAD)
