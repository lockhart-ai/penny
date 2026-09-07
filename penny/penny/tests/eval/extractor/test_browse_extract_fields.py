"""The browse EXTRACTION micro-context, in isolation: what a page makes of an instruction.

An ``extract`` instruction routinely names several things at once — what each item is
called, where it points, what it says — and how much of that a real page carries decides
what comes back.  The contract (#1942) is keyed to the PAGE and never to the wording:
content carrying SOME of what was asked for is an ``EXTRACTED:`` read that gives whatever
it has for each thing and leaves out the rest, and ``NOT_PRESENT:`` is for content carrying
NONE of it.

Four cases over the three states a page can be in relative to its instruction — it carries
ALL of what was named, SOME of it, or NONE — with the middle state reached on two different
page shapes.  Each is five wordings of its instruction over one fixed page:

* ``extract-fields-all-present`` — the CONTROL: a listing that answers everything asked of
  it.  The cheapest case in the suite, and the one that says a per-field read was not bought
  by loosening the ordinary case.
* ``extract-fields-partly-present`` — a homepage read as an index of other pages: titles and
  links, and no summaries, under an instruction that asks for all three.
* ``extract-fields-partly-present-on-a-prose-page`` — the same shortfall reached by a second
  PAGE SHAPE: a prose section front printing a byline under each headline and no timestamp.
* ``extract-fields-none-present`` — the honesty guard: a page carrying neither of the two
  things named must say so rather than answer anyway.

**The last two are a PAIR, and they are two cases rather than one.**  A behaviour states its
negative direction inside its own case where that direction shares the expected outcome;
here it does not — ``extract-fields-partly-present`` expects ``EXTRACTED`` and
``extract-fields-none-present`` expects ``NOT_PRESENT`` — so the negative direction is a case
of its own and the pair is named together.  A behaviour that always answered ``EXTRACTED:``
passes the first and fails the second.

**The arm is the ``extract`` instruction, and it is PENNY's own text** — production writes
it as the ``extract`` argument of a browse call, at the call site, by whatever agent made
the call.  So the five wordings are not five ways a person might ask; they are five ways the
CALLING DRAW might have worded the same request, which is variation that happens in
production today and that nothing else measures.  The FACTS are constant across the arms of a
case, because the claims name them.

Each case's page is what the CONTENT SCRIPT returns for a page of that shape, and the
document handed to the draw is built the way ``BrowseTool._page_section`` builds it, so the
extractor reads what production hands it.  A deterministic probe in ``make check`` holds each
fixture against the world it claims — every anchor really is on its page, exactly once —
because a case whose page does not carry what it says it carries measures nothing, and that
must fail before any GPU time rather than after.

Every case is report-only; the thresholds are the code owner's.  All content is synthetic.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.tests.eval.conftest import (
    EVAL_MODELS,
    EXTRACT_OUTCOME,
    EXTRACT_REASON,
    EXTRACT_VALUE,
    ExtractorEval,
    FieldExpectation,
)
from penny.tests.eval.utils.assertions import Answer, WorldClaim
from penny.tests.eval.utils.cohort import (
    Consequence,
    SampleObservation,
    SpecCategory,
    output_field,
    unsourced_specifics,
)
from penny.tests.eval.utils.fixtures import LISTING_URL
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import MicroExtractOutcome, spoken_form

pytestmark = pytest.mark.eval

_FAMILY = "browse-extract"


class ExtractFixture(NamedTuple):
    """One agreed case: the page as the content script returns it, the instruction as a
    task would word it, and what each thing the instruction names should come back as —
    an anchor for the ones the page supplies, nothing for the ones it does not.

    ``behaviour`` is the one sentence the case exists to check, in the fixed form: "In <the
    locus>, when <X>, Penny <does Y>."  It rides the fixture so a case states it the same way
    whichever driver path runs it, and the report renders it before any number.

    ``expectations`` is what the ``make check`` coherence probe holds against the page.  Each
    anchor it names is the SAME constant the case's own provenance claim reads, so a page edit
    cannot leave a claim asserting a span the page no longer carries."""

    case_id: str
    url: str
    page: str
    instruction: str
    behaviour: str
    expectations: tuple[FieldExpectation, ...]


# ── The pages ────────────────────────────────────────────────────────────────
#
# A homepage read as an index of other pages: one markdown link per story, which is
# titles and links and nothing else.  This is the shape the content script returns for
# such a page, so an instruction that also asks for summaries is asking for something
# genuinely not here.
_HOMEPAGE_URL = "https://news-alpha.example/"
_HOMEPAGE = (
    "[Harbour bridge reopens after a two-year refit]"
    "(https://news-alpha.example/world/2036/harbour-bridge-reopens-after-refit)\n"
    "[Dockside market changes hands after eighty years]"
    "(https://news-alpha.example/business/2036/dockside-market-changes-hands)\n"
    "[Lantern festival draws a record crowd to the old quarter]"
    "(https://news-alpha.example/world/2036/lantern-festival-draws-record-crowd)\n"
    "[Museum returns the borrowed mosaics a decade early]"
    "(https://news-alpha.example/culture/2036/museum-returns-borrowed-mosaics)\n"
    "[Ferry operator adds a night sailing for the winter]"
    "(https://news-alpha.example/business/2036/ferry-operator-adds-a-night-sailing)"
)

# A section front that prints a byline under each headline and no timestamp — the same
# partly-answered shortfall reached by a different PAGE SHAPE.  It is a second CASE and not
# five more arms of the link-index one, because an arm varies the WORDING of one ask: this
# instruction names different things (a byline and a time, where the other names a link and a
# summary), and it names them of a page written as prose rather than as markdown links.  A
# different ask over a different page is a scenario, and a scenario gets its own 15.
_SECTION_URL = "https://news-alpha.example/culture"
_SECTION = (
    "Culture\n"
    "\n"
    "Museum returns the borrowed mosaics a decade early\n"
    "by Wren Halloway\n"
    "\n"
    "City orchestra names a conductor from within the ranks\n"
    "by Ines Marlowe\n"
    "\n"
    "Bookshop reopens in the arcade under new owners\n"
    "by Wren Halloway\n"
)

# An ordinary listing that answers everything asked of it — the control, where nothing about
# a per-field read should show at all.  It is at the house url (`LISTING_URL`) and the house
# price (`499`), which is what the rest of the suite shares and what the claims below name.
#
# The PAGE ITSELF is local, and deliberately not `fixtures.AURORA_LISTING_499`.  This case is
# the one that needs THREE fields answered, and the shared listing carries two: it has a title,
# a price, a seller and a self-link, and it says nothing about stock.  Adding a stock line to it
# would change the page every browse-driven case in the suite reads, which is a measurement this
# port has not taken and is not the port's to take.  So the copy is real and this is why it
# exists.  The url is IMPORTED (`LISTING_URL`) rather than retyped; the price is the same `499`
# the shared listing prints, held against this page by the coherence probe in `make check`.
_LISTING = (
    "Aurora Deck 2, handheld console\n"
    "\n"
    "Price: $499\n"
    "In stock — three left, open box and tested\n"
    "Ships from a fictional warehouse within two days.\n"
)


# ── The anchors ──────────────────────────────────────────────────────────────
#
# One constant per thing a page SUPPLIES, named once and read twice: the deterministic
# coherence probe holds it against the page through the fixture's expectations, and the case's
# own provenance claim reads the same string.  One source of truth, so a page edit cannot leave
# a claim asserting a span the page no longer carries.
#
# Each is the SMALLEST UNIQUE span of what it identifies, because everything around it the
# model may legitimately write another way.  A draw that trimmed a headline to its subject, or
# dropped a currency symbol, read the page exactly as well as one that copied the line, and an
# anchor carrying the whole line would fail it.  Unique on the page as well as small: the probe
# holds both, since a span appearing twice would be satisfied by the wrong one.
_ITEM_ANCHOR = "Aurora Deck 2"
_PRICE_ANCHOR = "499"
# The count — the part of the stock line that carries the FACT.  The word after it ("left") and
# the clause around it are phrasing the draw chooses, exactly as the currency symbol is on the
# price, so the anchor stops before them.
#
# TWO residuals sit on it, both stated rather than repaired, because repairing either changes
# what is being measured and that is a code owner's call raised with the numbers.
#
# The ASK is a BOOLEAN — "whether it is in stock" — so a draw that answers "yes, in stock" has
# answered it without ever saying how many are left, and this anchor then fails a correct run.
# Shrinking does not reach that: the count is not a rendering of the boolean, it is a different
# fact.  What would reach it is WIDENING the ask to request the number, which moves the
# behaviour being measured and can move any number in either direction.
#
# And the page writes the count as a WORD.  `three` is not strictly identifiable the way `499`
# is: a draw that read the page perfectly may legitimately write `3`, and that is notation, the
# one thing an anchor must never carry.  As the page presents this field there is no small
# unique datum to shrink to — so the honest options are to drop the stock claim, or to print the
# count in digits AND widen the ask together, neither of which a port decides on its own.
_STOCK_ANCHOR = "three"

_HEADLINE_ANCHOR = "Lantern festival"
_LINK_ANCHOR = "https://news-alpha.example/world/2036/lantern-festival-draws-record-crowd"

_SECTION_HEADLINE_ANCHOR = "City orchestra"
_SECTION_BYLINE_ANCHOR = "Ines Marlowe"


# ── The claims ───────────────────────────────────────────────────────────────
#
# A micro-context returns a typed result, and that result splits in two.  Its CLOSED field —
# the outcome enum — is strictly identifiable and asserted by EQUALITY.  Its OPEN fields —
# `value`, `reason` — are what the model wrote, so they are reachable only through fact
# alignment, in both directions: nothing omitted, and nothing invented.


def _landed_on(fixture: ExtractFixture) -> WorldClaim:
    """A claim that the draw committed to the outcome THIS FIXTURE'S FACTS make right.

    ``EXTRACTED`` is right on a page carrying some of what was asked for and wrong on a page
    carrying none of it — the same value, correct against one world and false against another.
    So the expectation is READ off the fixture rather than written beside it: an expectation
    carrying an anchor is a thing the page supplies, and a fixture that names at least one is
    a page with something to extract.

    That is the same declaration the ``make check`` coherence probe holds against the page, so
    a case cannot state a world to the probe and a different one to its own claim.  A case that
    cannot say which of its facts makes its expectation right is asserting a habit; this one
    has no way to."""
    supplies_something = any(one.anchor for one in fixture.expectations)
    wanted = (
        MicroExtractOutcome.EXTRACTED if supplies_something else MicroExtractOutcome.NOT_PRESENT
    )

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = sample.field(EXTRACT_OUTCOME)
        return drawn == wanted.value, f"came back {drawn}"

    return answer


def _carries(anchor: str) -> WorldClaim:
    """A claim that the answer carries one thing the PAGE supplies — the *nothing omitted*
    direction of fact alignment.

    One claim per supplied thing rather than one over all of them, because an instruction
    naming several things degrades one thing at a time and a single combined claim would
    report "some of it arrived" as a total failure.

    Compared through the shipped ``spoken_form``, so a value passes whether or not the draw
    kept the punctuation or the article in front of it — deliberately not an equality, since
    which words carry a fact has a little play in it and a scorer demanding one exact string
    would be answering for the draw."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        value = sample.field(EXTRACT_VALUE)
        carried = spoken_form(anchor) in spoken_form(value)
        return carried, f"not in the extracted value: {value!r}"

    return answer


def _nothing_invented(sample: SampleObservation, _world: World) -> Answer:
    """Every specific value in the answer traces to what the draw was GIVEN — the *nothing
    invented* direction.

    The extractor's own failure mode, and the strongest claim available to it: the draw's open
    fields are free text lifted off the page and production validates none of it, so a plausible
    headline the page never carried reaches the caller verbatim and is written down as read.

    Read over the WHOLE structured answer rather than over ``value`` alone, so the branch that
    populates ``reason`` instead is covered by the same claim."""
    invented = unsourced_specifics(sample.output_text, sample.given)
    return not invented, f"not on the page: {invented}"


# ── Case 1: everything asked for is on the page (the control) ─────────────────

_ALL_PRESENT = ExtractFixture(
    case_id="extract-fields-all-present",
    url=LISTING_URL,
    page=_LISTING,
    instruction="the item's name, its price and whether it is in stock",
    behaviour=(
        "In the browse-extract micro-context, when the page carries everything the "
        "instruction named, Penny comes back with all of it."
    ),
    expectations=(
        FieldExpectation("name", _ITEM_ANCHOR),
        FieldExpectation("price", _PRICE_ANCHOR),
        FieldExpectation("stock", _STOCK_ANCHOR),
    ),
)

# Five wordings of that one request.  Each names the same three things and none of them nudges
# at HOW the stock line should be rendered: a wording that asked "how many are left" would earn
# the count anchor above rather than measure it, which is the fixture lending its own phrasing
# to the prompt under test.
_ALL_PRESENT_PHRASINGS = (
    "what the item is called, what it costs, and whether it is available",
    "the product name, the price, and the stock status",
    "for the item on this page: its name, its price, and whether it can be bought",
    "pull out the name of the item, how much it costs, and whether it is in stock",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_that_answers_everything_still_answers_everything(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The unchanged shape: three things named, three things on the page, three things back.

    It is here so a per-field read cannot be bought by loosening the ordinary case — if this
    moves, the change did something other than what it claims.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_ALL_PRESENT.case_id,
        behaviour=_ALL_PRESENT.behaviour,
        model=model,
        url=_ALL_PRESENT.url,
        page=_ALL_PRESENT.page,
        instruction=_ALL_PRESENT.instruction,
        also_instructed=_ALL_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page",
        _landed_on(_ALL_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the item name the page supplies",
        _carries(_ITEM_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the price the page supplies",
        _carries(_PRICE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the stock count the page supplies",
        _carries(_STOCK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.  No
    # tool sequence and no reply spread: a single call makes neither.  `reason` is measured
    # though a correct draw leaves it empty, for the reason spelled out on
    # `extract-fields-partly-present`: an all-empty reading is flagged BLIND rather
    # than counted as agreement, and a sample that came back NOT_PRESENT carries a reason
    # the others do not — an outlier at finer grain than `outcome` gives.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
    )


# ── Case 2: a link index, and an instruction that also asks for summaries ─────

_PARTLY_PRESENT = ExtractFixture(
    case_id="extract-fields-partly-present",
    url=_HOMEPAGE_URL,
    page=_HOMEPAGE,
    instruction=(
        "the headlines and their links, with a one-line summary of each where the page gives one"
    ),
    behaviour=(
        "In the browse-extract micro-context, when the page carries only some of what the "
        "instruction named, Penny comes back with the parts it has instead of reporting the "
        "page empty."
    ),
    expectations=(
        FieldExpectation("headline", _HEADLINE_ANCHOR),
        FieldExpectation("link", _LINK_ANCHOR),
        FieldExpectation("summary"),
    ),
)

# Two of the five hedge ("where the page gives one", "if there is one") and three do not, which
# is deliberate: whether the answer degrades per field is decided by what the PAGE carries,
# never by whether the instruction was worded to expect a gap.  Pooling both wordings into one
# cohort states that as a variance reading over one world.
_PARTLY_PRESENT_PHRASINGS = (
    "the headline of each story, the link to it, and a one-line summary if there is one",
    "for every story: its title, its url, and a short summary",
    "each headline with its link and a one-sentence summary",
    "pull out the story titles, the links they point at, and a brief summary of each",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_with_titles_and_links_and_no_summaries_still_reads(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The regression itself: the page has two of the three things named and the third is
    genuinely not on it.  The read has to come back with the two, not answer as though the
    page were empty.

    Its NEGATIVE direction is ``extract-fields-none-present`` below, a separate case because
    its expected outcome is the other one.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_PARTLY_PRESENT.case_id,
        behaviour=_PARTLY_PRESENT.behaviour,
        model=model,
        url=_PARTLY_PRESENT.url,
        page=_PARTLY_PRESENT.page,
        instruction=_PARTLY_PRESENT.instruction,
        also_instructed=_PARTLY_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to.  ``NOT_PRESENT`` here is the
    # regression itself: the answer that cost a whole round its headlines and links because the
    # page was short of the third thing.
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_PARTLY_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the headline the page supplies",
        _carries(_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the link the page supplies",
        _carries(_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.
    #
    # Variance is ORTHOGONAL to correctness: it does not ask whether a value is right, only
    # which samples diverge from the pack.  So `reason` belongs here even though this case
    # asserts the draw lands on `EXTRACTED` and `MicroContextResult` populates `reason` only
    # on `NOT_PRESENT` — a sample that came back NOT_PRESENT carries a reason the others do
    # not, and that is exactly an outlier worth surfacing, at finer grain than `outcome` gives.
    # On a run where every draw succeeds it reads nothing on all fifteen and is flagged blind,
    # which is the honest "nothing diverged this run" rather than a broken axis.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
    )


# ── Case 3: the same shortfall on a prose page ────────────────────────────────

_PARTLY_PRESENT_ON_A_PROSE_PAGE = ExtractFixture(
    case_id="extract-fields-partly-present-on-a-prose-page",
    url=_SECTION_URL,
    page=_SECTION,
    instruction="the headline, the byline and the published time for each story",
    behaviour=(
        "In the browse-extract micro-context, when a prose section front carries two of the "
        "three things the instruction named, Penny comes back with the two."
    ),
    expectations=(
        FieldExpectation("headline", _SECTION_HEADLINE_ANCHOR),
        FieldExpectation("byline", _SECTION_BYLINE_ANCHOR),
        FieldExpectation("published time"),
    ),
)

_PROSE_PAGE_PHRASINGS = (
    "for each story: its headline, who wrote it, and when it was published",
    "each story's title, its author and its publication time",
    "pull out the headlines, the bylines and the times they went up",
    "the title of every story, the name of whoever wrote it, and the time it was posted",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_prose_section_front_short_of_one_thing_still_reads(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The second PAGE SHAPE the contract has to hold on: prose with a byline under each
    headline, where the sibling case is markdown links.

    A world is fixed across a cohort's arms, so a page that is prose where the other is a link
    index is a world of its own and therefore a case of its own — the wording axis is already
    spent on the five instructions.  What it buys is the thing one page cannot say: that the
    per-field read follows the page's CONTENT and not the markup the content script happened to
    hand over.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_PARTLY_PRESENT_ON_A_PROSE_PAGE.case_id,
        behaviour=_PARTLY_PRESENT_ON_A_PROSE_PAGE.behaviour,
        model=model,
        url=_PARTLY_PRESENT_ON_A_PROSE_PAGE.url,
        page=_PARTLY_PRESENT_ON_A_PROSE_PAGE.page,
        instruction=_PARTLY_PRESENT_ON_A_PROSE_PAGE.instruction,
        also_instructed=_PROSE_PAGE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_PARTLY_PRESENT_ON_A_PROSE_PAGE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the headline the page supplies",
        _carries(_SECTION_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the byline the page supplies",
        _carries(_SECTION_BYLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.  No
    # tool sequence and no reply spread: a single call makes neither.  `reason` is measured
    # though a correct draw leaves it empty, for the reason spelled out on
    # `extract-fields-partly-present`: an all-empty reading is flagged BLIND rather
    # than counted as agreement, and a sample that came back NOT_PRESENT carries a reason
    # the others do not — an outlier at finer grain than `outcome` gives.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
    )


# ── Case 4: a page with none of it (the honesty guard) ────────────────────────

_NONE_PRESENT = ExtractFixture(
    case_id="extract-fields-none-present",
    url=_HOMEPAGE_URL,
    page=_HOMEPAGE,
    instruction="the closing price of each company named and its ticker symbol",
    behaviour=(
        "In the browse-extract micro-context, when the page carries none of what the "
        "instruction named, Penny says plainly that it carries none of it rather than "
        "answering anyway."
    ),
    expectations=(
        FieldExpectation("closing price"),
        FieldExpectation("ticker symbol"),
    ),
)

_NONE_PRESENT_PHRASINGS = (
    "for every company mentioned: what its shares closed at and its ticker",
    "each company's closing share price and stock ticker symbol",
    "pull out the closing prices and the ticker symbols of the companies here",
    "the ticker symbol of every company named and the price its stock closed at",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_carrying_none_of_it_still_says_so(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The over-correction guard, and the ``NOT_PRESENT`` half of the pair the module
    docstring names.

    A read that degrades per field must not degrade into answering anyway: this page carries
    neither of the two things named, so the honest answer is that it carries neither, and
    anything else is a value the page never held.  It is a case rather than a direction stated
    inside ``extract-fields-partly-present`` because its EXPECTED OUTCOME is the other one —
    a behaviour that always answered ``EXTRACTED:`` would pass that case and fail this.

    **Two categories report empty here, for two different reasons.**  STORE is empty for the
    shape's reason: one call returns a typed result and writes to no store.  The *nothing
    omitted* half of PROVENANCE is empty for this CASE's reason: nothing on this page answers
    the instruction, so there is no supplied thing for an answer to carry and no ``_carries``
    claim to make.  The *nothing invented* half is the whole point and is made below.

    **What is not asserted at all is the REASON's quality.**  It is free prose, so any check on
    it is the phrasing match ``docs/eval-case-design.md`` forbids; it rides the variance table
    instead.
    """
    cohort = await extractor_eval(
        case_id=_NONE_PRESENT.case_id,
        behaviour=_NONE_PRESENT.behaviour,
        model=model,
        url=_NONE_PRESENT.url,
        page=_NONE_PRESENT.page,
        instruction=_NONE_PRESENT.instruction,
        also_instructed=_NONE_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to.  ``NOT_PRESENT`` is derived
    # from this fixture's own facts: the page carries neither named thing.
    cohort.claim(
        "state: the draw reports the page carries none of it",
        _landed_on(_NONE_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — the *nothing invented* direction alone.  Its *nothing omitted* twin has
    # nothing to claim: the page supplies none of what was asked for, so there is no anchor an
    # answer could carry, and writing a vacuous claim would print a rate for a question nobody
    # asked.
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # `value` and `reason` swap roles here relative to the sibling cases: a correct draw
    # populates `reason` and leaves `value` empty, so `value` reads nothing on every sample and
    # is flagged blind — the honest "nothing diverged this run" — while `reason` carries the
    # spread.  Both stay COSMETIC: they are prose, and a divergence in them is a different word
    # for the same outcome.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
    )


# Every fixture, for the deterministic coherence probe in ``make check``.
FIXTURES = (
    _ALL_PRESENT,
    _PARTLY_PRESENT,
    _PARTLY_PRESENT_ON_A_PROSE_PAGE,
    _NONE_PRESENT,
)
