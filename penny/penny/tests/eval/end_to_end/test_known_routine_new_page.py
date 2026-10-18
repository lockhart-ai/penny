"""End-to-end: a known routine stood up on a new page (#2375, follow-up added #2494).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat turns —
here a cold ask, naming a page the routine was never taught on and giving the job its terms
in one turn, with a FOLLOW-UP agreement driven only while no job has been created yet this
run — then the job the script stood up, run through its OWN real cycles against that page.

The follow-up answers a Penny who offers before acting: known-routine binding can land her on
an offer ("set it up like the Aurora listing?") that waits for the user rather than creating
the job outright, and a one-turn script would never answer it.  A Penny who binds and acts on
the cold ask alone is driven once; the agreement is there for the one that asks first.

The entry condition is SEEDED: the price-watcher job already running, taught on the Aurora
listing (#2455 naive baseline, code-owner ruling — there is no skill registry left to teach a
routine into, so "a routine she already knows" is represented by the job already running PLUS
the earlier conversation that set it up, in her own logs).  That taught page stays LIVE and
readable in this world — in the chat phase's browse register and in every cycle's — so a read
of the WRONG page gets a real answer back rather than failing on a thin fixture, and the answer
carries the one value the claims forbid: the taught page's price.  Which collection the new job
lands on and what it is called are left to the model and read nowhere a claim depends on them;
the driver finds the created job STRUCTURALLY and runs ITS cycles.

**Claims read only:** what reached the user (the chat phase's replies and the job's own
sends), the created job's own terms (cadence, end, notify), and provenance of everything
sent or replied — never what a cycle left in the store, which is hers to keep as she wishes
and is measured, not claimed (``ENTRIES_STORED``, #2490), nor which collection the job sits
on.

**The taught price in nothing sent or replied is its own claim, not provenance's** —
the old page is legitimately part of what the round was given (the seeded history taught the
routine on it), so the generic provenance claim cannot speak to it.  It reads the LIVE
surfaces only: what was sent this run and what this run replied — never the seeded rows,
which carry the taught price by design.

**The five jobs the seeded world already had running are claimed undisturbed** — the one
store read about the OLD state these cases still make (#2490's addendum): what this run did
to them is read off the mutation ledger, where a rebind, a schedule change, a notify flip,
a moved end and an archive all answer the same way and the field nobody enumerated is
caught too.  What the NEW job keeps in its own store stays variance.

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
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.end_to_end import EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import END_TO_END_WALK, seed_known_routine, sent_at
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TURNS_DRIVEN,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import AURORA_LISTING_499, LISTING_URL, CannedPage, datum
from penny.tests.eval.utils.job_end import UNTIL_SUNDAY_NIGHT
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "end-to-end-new-page"
_FAMILY = "end-to-end"

_BEHAVIOUR = (
    "In a job stood up from a routine Penny already knows, when a cold ask names a different "
    "page, the user is told that page's price as it stands and as it moves, and the taught "
    "page's price appears in nothing sent or replied."
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

# The FOLLOW-UP (#2494): driven only while the cold ask above hasn't already created the job —
# the answer to a Penny who offers to set it up like the Aurora listing and waits, rather than
# acting on the cold ask's own terms.
_AGREEMENT = "yes, set it up as a new job like that"


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


_TAUGHT_COLLECTION = "aurora-deck-2-price"
_TAUGHT_ENTRY_KEY = "price"
_TAUGHT_ENTRY_VALUE = f"Price: ${_TAUGHT_AMOUNT}"
# The earlier conversation that set the Aurora price-watcher up (#2455 naive baseline,
# code-owner ruling): her own past logs are the other half of "a routine she already
# knows", beside the job itself.
_TAUGHT_ASK = "can you keep an eye on the Aurora Deck 2 price for me and let me know if it changes?"
_TAUGHT_REPLY = "done — I'll check the Aurora Deck 2 price and let you know if it moves. 🌟"


def _seed(db: Database) -> None:
    """The entry condition: the Aurora price-watcher as "a routine she already knows"
    (#2455 naive baseline, code-owner ruling) — both halves of it, through the chokepoints
    the runtime itself writes through: the job ALREADY RUNNING, and the EARLIER
    CONVERSATION that set it up, in her own logs (the ask, her read → write →
    ``collection_configure`` calls, her reply) — so her self-state's recent activity,
    ``log_read``, and ``run_get`` all read it back exactly as if that turn had really
    happened.  Asserted rather than assumed, so a world that drifted fails here rather
    than after a live run.

    What this seed checks on top is the one fact the negative claim's answerability rests
    on: the taught price IS in this world's store, held by the job the routine was taught
    in, so the claim below reads a world where the taught value genuinely sits one write
    away."""
    seed_known_routine(
        db,
        collection=_TAUGHT_COLLECTION,
        description="the Aurora Deck 2 handheld console price",
        url=LISTING_URL,
        extract="the current price",
        entry_key=_TAUGHT_ENTRY_KEY,
        entry_value=_TAUGHT_ENTRY_VALUE,
        schedule="FREQ=HOURLY",
        ask=_TAUGHT_ASK,
        reply=_TAUGHT_REPLY,
    )
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
        turns=(ask, _AGREEMENT),
        seed=_seed,
        world=_WORLD,
        cycles=_CYCLES,
    )
    for ask in _ASKS
)


# ── The claims ───────────────────────────────────────────────────────────────


def _told_the_new_page_price_by_the_first_cycle(sample: SampleObservation, _world: World) -> Answer:
    """The user has been told the new page's price by the end of the first cycle — by the
    ask's own reply, if the turn read the page before standing the job up, or by the job's
    own first cycle.

    Read as the price appearing in any chat reply OR any notification so far: the chat phase's
    replies with every cycle's own send stripped back out (``reply`` joins both), then the
    messages the first cycle put on the queue — nothing from a later cycle counts, because
    "by the end of the first cycle" is the claim's own bound."""
    chat_replies = sample.reply
    for message in sample.notifications:
        if message in chat_replies:
            chat_replies = chat_replies.replace(message, "")
    notifications_so_far = sent_at(sample.cycles, 0)
    told = _BASELINE_AMOUNT in chat_replies or any(
        _BASELINE_AMOUNT in message for message in notifications_so_far
    )
    replies_hit = _BASELINE_AMOUNT in chat_replies
    cycles_hit = [message for message in notifications_so_far if _BASELINE_AMOUNT in message]
    return (
        told,
        f"in a chat reply: {replies_hit}; in a first-cycle notification: {cycles_hit}",
    )


def _told_the_moved_price_when_it_moved(sample: SampleObservation, _world: World) -> Answer:
    """When the price moved, the user was told the moved price — the messages sent on the
    second cycle carry the moved amount the page now posts.  Read off the cycle's own sends,
    so a cycle that ran and sent nothing, or sent the old reading, fails here."""
    sent = sent_at(sample.cycles, 1)
    told = [message for message in sent if _MOVED_AMOUNT in message]
    return bool(told), f"sent on the moved cycle: {sent}; carrying {_MOVED_AMOUNT}: {told}"


def _the_taught_price_appears_in_nothing_sent_or_replied(
    sample: SampleObservation, _world: World
) -> Answer:
    """The taught page's price appears in nothing sent and nothing replied — the explicit
    "never the page the routine was taught on" check, restated to what reaches the user
    (#2490).

    It reads what reached the user this run: everything the created jobs' cycles put on the
    send queue, and what this run replied (``reply`` is the chat phase's replies and the
    cycles' sends together for an end-to-end sample).  The seeded rows are not among them —
    they carry the taught price by design, because the routine was taught on that page — so
    the claim answers what the NEW job put in front of the user, not what the world already
    held."""
    surfaces = (
        ("sent to the user", [message for snapshot in sample.cycles for message in snapshot.sent]),
        ("replied", sample.reply),
    )
    hits = [name for name, text in surfaces if _TAUGHT_AMOUNT in text]
    return not hits, f"{_TAUGHT_AMOUNT} found in: {hits or 'nothing'}"


_MEASURED = (
    END_TO_END_WALK,
    TOOL_SEQUENCE,
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TURNS_DRIVEN,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_known_routines_job_runs_the_page_the_ask_names(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """A cold ask naming a page the price-watcher was never taught on, with terms, and an
    agreement driven only while no job exists yet — then the job the script stood up runs
    for real across that page, which holds, then moves, while the page the routine was
    taught on stays live and readable."""
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

    # SENT — the two cycles, each read as what reached the user at that position, union
    # over however many jobs the script created.  What a cycle left in the store is hers,
    # measured as variance (``ENTRIES_STORED``), never claimed (#2490).
    cohort.claim(
        "state: by the end of the first cycle the user has been told the new page's price",
        _told_the_new_page_price_by_the_first_cycle,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: when the price moved, the user was told the moved price",
        _told_the_moved_price_when_it_moved,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the taught page's price appears in nothing sent or replied",
        _the_taught_price_appears_in_nothing_sent_or_replied,
        SpecCategory.STORE,
    )

    # UNDISTURBED — the five jobs the seeded world already had running are left as they were
    # (#2490's addendum): read off the mutation ledger, so a rebind, a schedule change, a
    # notify flip, a moved end and an archive all answer it the same way.
    cohort.assert_the_running_mechanisms_survive()

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
