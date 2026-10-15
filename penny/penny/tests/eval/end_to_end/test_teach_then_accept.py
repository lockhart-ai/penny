"""End-to-end: a watch taught in chat, then run (#2375).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat turns —
a cold ask for a watch, the user's walkthrough once Penny asks to be taught, and an acceptance
giving the job its terms — followed by the job those turns created, run through its OWN real
cycles against a page that holds, then moves.

No routine, no job and no collection exist before the script runs: the world is COLD, so
whatever container the round builds, whatever it names the routine, and which collection the
turns land the job on are all left to the model and read nowhere a claim depends on it.  The
driver finds the created job STRUCTURALLY (``end_to_end_eval``'s own read of whichever
collection the script scheduled) and runs ITS cycles; the claims below never name it.

**Claims read only:** what the store's entries hold after each cycle, what reached the user,
the created job's own terms (cadence, end, notify), and provenance — never which collection
the job sits on, the routine's shape or bound values, or which conversation-machine states the
script passed through on the way (#2375's ruling).

**The page is real-size** — several kilobytes, a labelled current-price line among a handful
of other prices the page carries for unrelated items, specs, seller notes and a returns
policy, the way a real marketplace listing reads. The baseline and moved prices are mutually
exclusive as numerals, so a sample that invented a plausible number cannot satisfy a claim
that names the one the page actually carries.

Report-only (``min_pass_rate=None``). All content is synthetic, on an ``example`` domain,
because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import MemoryType
from penny.tests.eval.conftest import EVAL_MODELS, EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import END_TO_END_WALK, held_text_at, ran_at, sent_at
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
from penny.tests.eval.utils.fixtures import CannedPage, datum
from penny.tests.eval.utils.job_end import until_tonight_at
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "end-to-end-teach-then-accept"
_FAMILY = "end-to-end"

_BEHAVIOUR = (
    "In a job taught in chat, when the user asks for a watch, walks Penny through it once "
    "and accepts her offer with terms, Penny stands up a job on those terms whose cycles "
    "record the first reading, stay quiet while the page holds, and replace the reading and "
    "tell the user when it moves."
)

# ── The page the demonstration reads, and the two readings its cycles see ────

_URL = "https://trailgear.example/juniper-trail-stove"
_ITEM = "Juniper Trail Stove"

_BASELINE_AMOUNT = "129"
_MOVED_AMOUNT = "109"
_BASELINE_PRICE = f"${_BASELINE_AMOUNT}"
_MOVED_PRICE = f"${_MOVED_AMOUNT}"

# Labelled, so the neighbouring items' own prices further down the page cannot be read as the
# watched value — the lesson #1918's unlabelled datum cost a round.
_DATUM = f"Current price: {_BASELINE_PRICE}"
_MOVED_DATUM = f"Current price: {_MOVED_PRICE}"

_LISTING_LINK = f"[{_ITEM} listing]({_URL})"
_SPEC_LINK = f"[{_ITEM} spec sheet]({_URL}/spec)"

# Real-size: a title line, a description, the labelled datum, seller and condition notes, a
# specification block, two unrelated items from the same seller with their own prices, and a
# returns policy — several kilobytes, the shape of a real marketplace listing rather than a
# fixture trimmed to the one line a claim reads.
_BASELINE_PAGE = CannedPage(
    match="juniper-trail-stove",
    text=(
        f"Title: {_ITEM} — backpacking stove | trailgear\n"
        f"{_URL}\n"
        "\n"
        f"{_ITEM} (open box, tested). Sold by a fictional outfitter; stock turns over weekly.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: pinehollow_gear (4.8 stars). The asking price last changed six days ago.\n"
        "Condition: open box, tested. Ships within two working days.\n"
        "\n"
        "Specification\n"
        "Two-burner folding frame, 9,000 BTU output, piezo ignition\n"
        f"{_SPEC_LINK}\n"
        "Boils half a litre of water in about ninety seconds on a full canister.\n"
        "Weight 860g · Folds flat to a 20cm case · One wind guard included\n"
        "\n"
        "Others from this seller\n"
        "Birch Camp Lantern, refurbished — $39\n"
        "[Birch Camp Lantern](https://trailgear.example/birch-camp-lantern)\n"
        "Pinehollow Fuel Canister, 450g — $14\n"
        "Both ship from the same fictional depot, usually inside a week.\n"
        "\n"
        "What buyers say\n"
        '"Lit first try even in wind — the guard earns its keep." — verified buyer\n'
        '"Smaller than I expected, packs well for a weekend trip." — verified buyer\n'
        "\n"
        "Returns are accepted within thirty days of delivery; return postage is the buyer's.\n"
        "This fictional outfitter ships from one regional depot and restocks every Tuesday.\n"
    ),
)
_MOVED_PAGE = datum(_BASELINE_PAGE, _DATUM, _MOVED_DATUM)

# The entry condition is COLD — no routine, no job, no collection — so the three cycles are:
# the page as the demonstration read it, the same page again (nothing moved), then the moved
# reading.
_CYCLES = ([_BASELINE_PAGE], [_BASELINE_PAGE], [_MOVED_PAGE])

# The ground the SCRIPT's own turns are answered against — one page, installed once before any
# turn runs, the way a chat case's world always is.  ``keeps``/``excludes``/``answers`` are all
# empty by report: the script sets a job running later, so it asks to be TOLD rather than to
# be given a value back, and the ask names nothing to exclude.
_WORLD = World(name=_CASE_ID, pages=(_BASELINE_PAGE,), keeps=(), excludes=())

# The terms the acceptance gives: fires hourly, stops 22:00 the day of the turn, tells the
# user.  Read against ``_the_new_job`` in ``assertions.py`` — existentially, since "a job she
# created" is the claim, never "exactly one".
_CADENCE_SECONDS = 3600
_ASKED_END = until_tonight_at(22)

# ── The five scripts: one cold ask, one walkthrough, one acceptance — five wordings ──
#
# What varies is how a person says each of the three turns: which verb opens the ask, how the
# walkthrough names the three actions, which word the acceptance opens with.  What does NOT
# vary is the url (named in both the ask and the walkthrough), the walkthrough's three actions
# against that one page, and the acceptance's cadence, end and telling clause — every wording
# carries all of them, because the claims below hinge on them.
_SCRIPTS: tuple[tuple[str, str, str], ...] = (
    (
        f"can you watch this for me and let me know when the price changes? {_URL}",
        f"sure — open {_URL}, find the current price, and remember it",
        "yes please — check it every hour until 10pm tonight and let me know if the price changes",
    ),
    (
        f"could you keep an eye on this listing and tell me if the price moves? {_URL}",
        f"ok — head to {_URL}, see what the current price is, and keep it",
        "sounds good — do that every hour until 10pm tonight and message me if it changes",
    ),
    (
        f"i'd like you to watch this page for price changes and tell me when it happens: {_URL}",
        f"yep — go to {_URL}, check the current price, and save it",
        "great, go ahead — check it hourly until 10pm tonight and let me know when the price "
        "changes",
    ),
    (
        f"track this for me and let me know when the price changes, please — {_URL}",
        f"alright — visit {_URL}, note the current price, and hang on to it",
        "yeah do that — every hour until 10pm tonight, and tell me if it moves",
    ),
    (
        f"can you monitor this listing's price and give me a heads up when it changes? {_URL}",
        f"sure thing — read {_URL}, pull the current price off it, and remember that",
        "perfect — run that every hour until 10pm tonight and message me the moment it changes",
    ),
)

_CADENCE_TOKEN = "hour"
_END_TOKEN = "10pm"
_TELLING_CLAUSES = ("let me know", "message", "tell me")


def _assert_every_script_carries_the_facts() -> None:
    """Every one of the five scripts' three turns carries the facts the claims below read —
    the url in the ask and the walkthrough, and the acceptance's cadence, end and telling
    clause — so a dropped fact fails in this assertion rather than as a puzzling score after a
    paid run."""
    for ask, walkthrough, acceptance in _SCRIPTS:
        assert _URL in ask, f"{_CASE_ID}: this ask names no page — {ask!r}"
        assert _URL in walkthrough, f"{_CASE_ID}: this walkthrough names no page — {walkthrough!r}"
        missing = [token for token in (_CADENCE_TOKEN, _END_TOKEN) if token not in acceptance]
        assert not missing, f"{_CASE_ID}: this acceptance gives none of {missing} — {acceptance!r}"
        assert any(clause in acceptance for clause in _TELLING_CLAUSES), (
            f"{_CASE_ID}: this acceptance never asks to be told — {acceptance!r}"
        )


_assert_every_script_carries_the_facts()


def _seed(db: Database) -> None:
    """The entry condition: no routine taught, no job running.  Asserted rather than assumed,
    so a world that drifted fails here rather than after a live run.

    Checked against COLLECTIONS only, never the whole store — a fresh eval database is built
    exactly like production (``create_tables()`` then ``migrate()``), and migrations seed
    system log memories (``penny-messages``, the browse-results log, …) into every database,
    cold teach or not."""
    assert not db.skills.list_all(), "the registry must start empty for a cold teach"
    collections = [row for row in db.memories.list_all() if row.type == MemoryType.COLLECTION]
    assert not collections, f"no collection may exist yet for a cold teach, found {collections}"


_ARMS = tuple(
    EndToEndArm(
        label=script[0], turns=script, seed=_seed, seed_skills=None, world=_WORLD, cycles=_CYCLES
    )
    for script in _SCRIPTS
)


# ── The claims ───────────────────────────────────────────────────────────────


def _store_holds_the_first_reading(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 0)
    return _BASELINE_AMOUNT in held, f"expected {_BASELINE_AMOUNT}; held {held!r}"


def _told_once_at_the_first_reading(sample: SampleObservation, _world: World) -> Answer:
    sent = sent_at(sample.cycles, 0)
    return len(sent) == 1, f"{len(sent)} messages sent on the first cycle: {sent}"


def _store_still_holds_the_first_reading(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 1)
    return _BASELINE_AMOUNT in held, f"expected {_BASELINE_AMOUNT}; held {held!r}"


def _nothing_more_sent_on_the_unchanged_cycle(sample: SampleObservation, _world: World) -> Answer:
    """Nothing reached the user on the SECOND cycle — staying quiet is the point of this
    claim, so it is only true of a sample where the second cycle actually ran: a sample with
    no job, or whose job's cycles never got this far, has not stayed quiet on an unchanged
    reading, it has simply not reached one — ``ran_at`` keeps that from reading as a vacuous
    pass (#2375's review)."""
    ran = ran_at(sample.cycles, 1)
    sent = sent_at(sample.cycles, 1)
    return ran and not sent, f"cycle ran: {ran}; sent on it: {sent}"


def _store_holds_the_moved_reading_only(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 2)
    ok = _MOVED_AMOUNT in held and _BASELINE_AMOUNT not in held
    return ok, f"expected {_MOVED_AMOUNT} and not {_BASELINE_AMOUNT}; held {held!r}"


def _told_once_more_when_it_moved(sample: SampleObservation, _world: World) -> Answer:
    sent = sent_at(sample.cycles, 2)
    return len(sent) == 1, f"{len(sent)} messages sent on the moved cycle: {sent}"


_MEASURED = (END_TO_END_WALK, TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_watch_taught_in_chat_is_run_through_its_own_cycles(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """A cold ask, a walkthrough and an acceptance with terms — all driven for real — then the
    job those turns created runs for real across a page that holds, then moves."""
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
    # LANDED — empty; see the module docstring. The script's own machine walk is read as part
    # of the end-to-end WALK feature below, never as a claim about which states it passed
    # through.

    # STORE — the job's own terms, read existentially over whatever it created.
    cohort.assert_a_new_job_fires_every(_CADENCE_SECONDS)
    cohort.assert_a_new_job_notifies()
    cohort.assert_a_new_job_ends_when_asked(_ASKED_END)

    # STORE — the three cycles, each read as what the created job(s) held and sent at that
    # position, union over however many jobs the script created.
    cohort.claim(
        "state: after the first cycle the store holds the page's price",
        _store_holds_the_first_reading,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the user was told once after the first reading",
        _told_once_at_the_first_reading,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: after the second cycle the store still holds that price",
        _store_still_holds_the_first_reading,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: nothing more was sent while the page held",
        _nothing_more_sent_on_the_unchanged_cycle,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: after the third cycle the store holds the new price, not the old",
        _store_holds_the_moved_reading_only,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the user was told once more when the price moved",
        _told_once_more_when_it_moved,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
