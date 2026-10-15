"""End-to-end: a known routine stood up on a new page (#2375).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat turns —
here a single cold ask, naming a page the routine was never taught on and giving the job its
terms — followed by the job that turn stood up, run through its OWN real cycles against that
page.

The entry condition is SEEDED, the composed world of five finished journeys
(``seed_composed_world``): the price-watcher routine already exists, taught on the Aurora
listing.  That taught page stays LIVE and readable in this world — in the chat phase's browse
register and in every cycle's — so a read of the WRONG page gets a real answer back rather
than failing on a thin fixture, and the answer carries the one value the claims forbid: the
taught page's price.  Whether the round recognises the routine, which collection it lands the
job on and what the routine is called are all left to the model and read nowhere a claim
depends on them; the driver finds the created job STRUCTURALLY and runs ITS cycles.

**Claims read only:** what the store's entries hold after each cycle, what reached the user,
the created job's own terms (cadence, end, notify), and provenance — never which collection
the job sits on, the routine's shape or bound values, or which conversation-machine states the
script passed through on the way (#2375's ruling).

**The taught price in nothing stored and nothing sent is its own claim, not provenance's** —
the old page is legitimately part of what the round was given (the seeded history taught the
routine on it), so the generic provenance claim cannot speak to it.  It reads the LIVE surfaces
only: what this run wrote, what the created jobs hold, what was sent this run — never the
seeded rows, which carry the taught price by design.

**The pages are real-size** — several kilobytes, a labelled current-price line among other
prices the page carries for unrelated items, the way a real marketplace listing reads.  The
new page's two readings and the taught page's price are mutually exclusive as numerals, so a
sample that invented a plausible number cannot satisfy a claim that names the one the page
actually carries.

Report-only (``min_pass_rate=None``). All content is synthetic, on an ``example`` domain,
because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import MemoryType
from penny.tests.eval.conftest import EVAL_MODELS, EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import END_TO_END_WALK, held_text_at
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import AURORA_LISTING_499, CannedPage, datum
from penny.tests.eval.utils.job_end import UNTIL_SUNDAY_NIGHT
from penny.tests.eval.utils.transition_world import (
    _JOURNEYS,
    assert_composed_world,
    seed_composed_world,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "end-to-end-new-page"
_FAMILY = "end-to-end"

_BEHAVIOUR = (
    "In a job stood up from a routine Penny already knows, when a cold ask names a different "
    "page, the job's cycles record that page's readings and never the page the routine was "
    "taught on."
)

# ── The new page the ask names, and the two readings its cycles see ──────────

_URL = "https://ridgelinegear.example/cedar-ridge-camp-table"
_ITEM = "Cedar Ridge Camp Table"

_BASELINE_AMOUNT = "212"
_MOVED_AMOUNT = "189"
_BASELINE_PRICE = f"${_BASELINE_AMOUNT}"
_MOVED_PRICE = f"${_MOVED_AMOUNT}"

# Labelled, so the neighbouring items' own prices further down the page cannot be read as the
# watched value — the lesson #1918's unlabelled datum cost a round.
_DATUM = f"Current price: {_BASELINE_PRICE}"
_MOVED_DATUM = f"Current price: {_MOVED_PRICE}"

# The price the routine was TAUGHT on — the Aurora listing's, the one value in this world the
# new job must never record or relay.  Checked as the bare amount so any spelling of it in a
# stored entry or a sent message reads as the miss it is.
_TAUGHT_AMOUNT = "499"

_LISTING_LINK = f"[{_ITEM} listing]({_URL})"
_SPEC_LINK = f"[{_ITEM} spec sheet]({_URL}/spec)"

# Real-size, in the shape of case 1's listing: a title line, a description, the labelled
# datum, seller and condition notes, a specification block, two unrelated items from the same
# seller with their own prices, and a returns policy — several kilobytes.
_BASELINE_PAGE = CannedPage(
    match="cedar-ridge-camp-table",
    text=(
        f"Title: {_ITEM} — roll-top aluminium camp table | ridgelinegear\n"
        f"{_URL}\n"
        "\n"
        f"{_ITEM} (light use, one season). Sold by a fictional outfitter; sizes rotate.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: granitepeak_outfitters (4.6 stars). The asking price last changed five days ago.\n"
        "Condition: light use, one season. Ships within two working days.\n"
        "\n"
        "Specification\n"
        "Roll-top aluminium surface, adjustable legs, 1.1m unfolded length\n"
        f"{_SPEC_LINK}\n"
        "Sets up in about a minute without tools; the legs pack inside the rolled top.\n"
        "Weight 3.4kg · Seats four on a bench side · Carries in a 45cm sleeve\n"
        "\n"
        "Others from this seller\n"
        "Granitepeak Camp Stool, folding — $45\n"
        "[Granitepeak Camp Stool](https://ridgelinegear.example/granitepeak-camp-stool)\n"
        "Ridgeline Mug, 350ml enamel — $18\n"
        "Both ship from the same fictional depot, usually inside a week.\n"
        "\n"
        "What buyers say\n"
        '"Rock solid in a breeze — the adjustable legs earn their keep." — verified buyer\n'
        '"Heavier than I expected, but it packs down fine for a car trip." — verified buyer\n'
        "\n"
        "Returns are accepted within thirty days of delivery; return postage is the buyer's.\n"
        "This fictional outfitter ships from one regional depot and restocks every Friday.\n"
    ),
)
_MOVED_PAGE = datum(_BASELINE_PAGE, _DATUM, _MOVED_DATUM)

# The entry condition is SEEDED — five finished journeys, the price-watcher taught on the
# Aurora listing — so the job's two cycles are: the new page as the ask names it, then the
# moved reading.  The TAUGHT page rides in every register beside the new one: a cycle that
# reads it gets the taught price back, which the claims below forbid.
_CYCLES = (
    [_BASELINE_PAGE, AURORA_LISTING_499],
    [_MOVED_PAGE, AURORA_LISTING_499],
)

# The ground the SCRIPT's own turn is answered against — both pages, installed before the ask
# runs.  ``keeps``/``excludes``/``answers`` are all empty by report: the ask sets a job running,
# so it asks to be TOLD rather than to be given a value back, and it names nothing to exclude.
_WORLD = World(name=_CASE_ID, pages=(_BASELINE_PAGE, AURORA_LISTING_499), keeps=(), excludes=())

# The terms the ask gives: fires hourly, stops sunday night, tells the user.  Read against
# ``_the_new_job`` in ``assertions.py`` — existentially, since "a job she created" is the
# claim, never "exactly one".
_CADENCE_SECONDS = 3600
_ASKED_END = UNTIL_SUNDAY_NIGHT

# ── The five wordings of the one cold ask ─────────────────────────────────────
#
# What varies is how a person says it: which verb opens it, where the terms sit relative to
# the address, "changes" or "moves".  What does NOT vary is the page (named in every wording),
# the cadence, the end condition and the telling clause — every wording carries all of them,
# because the claims below hinge on them.
_ASKS: tuple[str, ...] = (
    f"can you watch this listing for me and let me know when the price changes? {_URL} — "
    "every hour until sunday night",
    f"keep an eye on {_URL} and tell me when the price changes — hourly, until sunday night",
    f"could you check {_URL} every hour until sunday night and let me know if the price moves?",
    f"watch this one and tell me when the price changes please: {_URL} — hourly, up to "
    "sunday night",
    f"i'd like {_URL} checked every hour until sunday night, with a message when the price changes",
)

_CADENCE_TOKEN = "hour"
_END_TOKEN = "sunday night"
_TELLING_CLAUSES = ("let me know", "tell me", "message")


def _assert_every_wording_carries_the_facts() -> None:
    """Every one of the five wordings carries the facts the claims below read — the page, the
    cadence, the end and the telling clause — and none of them carries the taught price, so a
    dropped fact or a smuggled value fails in this assertion rather than as a puzzling score
    after a paid run."""
    for ask in _ASKS:
        assert _URL in ask, f"{_CASE_ID}: this wording names no page — {ask!r}"
        missing = [token for token in (_CADENCE_TOKEN, _END_TOKEN) if token not in ask]
        assert not missing, f"{_CASE_ID}: this wording gives none of {missing} — {ask!r}"
        assert any(clause in ask for clause in _TELLING_CLAUSES), (
            f"{_CASE_ID}: this wording never asks to be told — {ask!r}"
        )
        assert _TAUGHT_AMOUNT not in ask, (
            f"{_CASE_ID}: this wording carries the taught page's price — {ask!r}"
        )
    assert _TAUGHT_AMOUNT not in _BASELINE_PAGE.text, (
        f"{_CASE_ID}: the new page carries the taught price, so the negative claim below "
        "could not fail"
    )


_assert_every_wording_carries_the_facts()


def _seed(db: Database) -> None:
    """The entry condition: the composed world of five finished journeys, asserted rather than
    assumed, so a world that drifted fails here rather than after a live run.

    The registry probe (``assert_the_registry_holds``) is deliberately absent: the runner lays
    the fixture skills down AFTER this seed runs, so the check would read an empty registry
    and false-fail — the composed world's own probe covers what the seeder laid down.  What
    this seed adds on top is the one fact the negative claim's answerability rests on: the
    taught price IS in this world's store, held by the journey the routine was taught in, so
    the claim below reads a world where the taught value genuinely sits one binding away."""
    seed_composed_world()(db)
    assert_composed_world(db)
    taught: list[str] = []
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        taught += [entry.content for entry in memory.read_all()] if memory else []
    assert any(_TAUGHT_AMOUNT in content for content in taught), (
        f"{_CASE_ID}: the seeded world holds no {_TAUGHT_AMOUNT}, so the negative claim below "
        "has nothing to be false of"
    )


_ARMS = tuple(
    EndToEndArm(
        label=ask,
        turns=(ask,),
        seed=_seed,
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        world=_WORLD,
        cycles=_CYCLES,
    )
    for ask in _ASKS
)


# ── The claims ───────────────────────────────────────────────────────────────


def _store_holds_the_new_page_price(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 0)
    return _BASELINE_AMOUNT in held, f"expected {_BASELINE_AMOUNT}; held {held!r}"


def _store_holds_the_moved_price(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 1)
    return _MOVED_AMOUNT in held, f"expected {_MOVED_AMOUNT}; held {held!r}"


def _the_taught_price_appears_nowhere(sample: SampleObservation, _world: World) -> Answer:
    """The taught page's price appears in nothing stored and nothing sent — the explicit
    "never the page the routine was taught on" check.

    It reads the LIVE surfaces only: what this run wrote (``entries``, wherever it landed),
    what the created jobs hold at every cycle position, and everything that reached the user
    this run (``reply`` is the chat phase's replies and the cycles' sends together).  The
    seeded rows are not among them — they carry the taught price by design, because the
    routine was taught on that page — so the claim answers what the NEW job put into the
    world, not what the world already held."""
    surfaces = (
        ("stored this run", " ".join(entry.text for entry in sample.entries)),
        (
            "held by the created jobs",
            " ".join(entry.text for snapshot in sample.cycles for entry in snapshot.held),
        ),
        ("sent to the user", sample.reply),
    )
    hits = [name for name, text in surfaces if _TAUGHT_AMOUNT in text]
    return not hits, f"{_TAUGHT_AMOUNT} found in: {hits or 'nothing'}"


_MEASURED = (END_TO_END_WALK, TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_known_routines_job_runs_the_page_the_ask_names(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """One cold ask, naming a page the price-watcher was never taught on, with terms — then
    the job that ask stood up runs for real across that page, which holds, then moves, while
    the page the routine was taught on stays live and readable."""
    cohort = await end_to_end_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        arms=list(_ARMS),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — the job's own terms, read existentially over whatever it created.
    cohort.assert_a_new_job_fires_every(_CADENCE_SECONDS)
    cohort.assert_a_new_job_notifies()
    cohort.assert_a_new_job_ends_when_asked(_ASKED_END)

    # STORE — the two cycles, each read as what the created job(s) held at that position,
    # union over however many jobs the script created.
    cohort.claim(
        "state: after the first cycle the store holds the new page's price",
        _store_holds_the_new_page_price,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: after the second cycle the store holds the new page's moved price",
        _store_holds_the_moved_price,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the taught page's price appears in nothing stored and nothing sent",
        _the_taught_price_appears_nowhere,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
