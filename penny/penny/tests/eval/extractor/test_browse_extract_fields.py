"""Live-model contract for an extract instruction a page HALF answers (#1942).

An ``extract`` instruction routinely names several things at once — what each item is
called, where it points, what it says — and a real page carries some of them and not
others.  The observed failure: a page that carried titles and links but no summaries came
back ``NOT_PRESENT`` on a strict draw, so the titles and links were lost too, while a
later draw over the SAME content extracted them.  That is draw variance on a boundary the
contract never stated: nothing told the extractor which side of the tag a
partly-answered instruction falls on.

It now does — a page carrying SOME of what was asked for is an ``EXTRACTED:`` read, and
``NOT_PRESENT:`` is for content carrying NONE of it — and these four cases are both
directions of that:

* ``extract-fields-all-present`` — the unchanged baseline: everything asked for is on the
  page, and everything comes back.
* ``extract-fields-partly-present`` — the regression: an instruction whose own words allow
  a thing to be absent, over a page that has titles and links and no summaries.
* ``extract-fields-partly-present-unhedged`` — the same shape with an instruction that
  hedges NOTHING, because the contract is about what the page carries and not about how
  the instruction was worded.  A fix that only fires on "where the page gives one" is a
  fix keyed to the phrasing that happened to be in front of us.
* ``extract-fields-none-present`` — the honesty guard: a page with none of it must still
  say so, or the fix has bought the per-field read with a confabulation.

Each case's page is what the CONTENT SCRIPT now returns for such a page — a markdown
link per story, since a homepage read as an index of other pages carries titles and links
and, unless the page prints them, no summaries.  The two halves of #1942 meet here: the
extension recovers the titles and links, and this is what an instruction asking for more
than that then does with them.

The document handed to the draw is built the way ``BrowseTool._page_section`` builds it,
so the extractor reads what production hands it.  A deterministic probe in ``make check``
holds each fixture against the world it claims — every anchor really is on its page —
because a case whose page does not carry what it says it carries measures nothing, and
that must fail before any GPU time rather than after.

Every case is report-only; the thresholds are the code owner's once the first numbers are
read.  All content is synthetic.
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
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import MicroExtractOutcome, spoken_form

pytestmark = pytest.mark.eval

_FAMILY = "browse-extract"


class ExtractFixture(NamedTuple):
    """One agreed case: the page as the content script returns it, the instruction as a
    task would word it, and what each thing the instruction names should come back as —
    an anchor for the ones the page supplies, nothing for the ones it does not."""

    case_id: str
    url: str
    page: str
    instruction: str
    expectations: tuple[FieldExpectation, ...]


async def _run_case(extractor_eval: ExtractorEval, fixture: ExtractFixture) -> None:
    """Drive one case's page + instruction through the extraction micro-context.  Every
    case is report-only: the thresholds are the code owner's to set once the first
    numbers are read."""
    await extractor_eval(
        case_id=fixture.case_id,
        url=fixture.url,
        page=fixture.page,
        instruction=fixture.instruction,
        expectations=fixture.expectations,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


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
# partly-answered shape reached by a different page, so the unhedged case is not a
# rewording of the hedged one over identical content.
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

# An ordinary listing that answers everything asked of it — the baseline, where nothing
# about this change should show at all.
_LISTING_URL = "https://faux-market.example/keel-lantern"
_LISTING = (
    "Keel Lantern, brass\n"
    "\n"
    "Price: 84 zorkmids\n"
    "In stock — three left in the workshop\n"
    "Ships from the quay within two days.\n"
)


# ── Case 1: everything asked for is on the page (the baseline) ────────────────

_ALL_PRESENT = ExtractFixture(
    case_id="extract-fields-all-present",
    url=_LISTING_URL,
    page=_LISTING,
    instruction="the item's name, its price and whether it is in stock",
    expectations=(
        FieldExpectation("name", "Keel Lantern"),
        FieldExpectation("price", "84 zorkmids"),
        FieldExpectation("stock", "three left"),
    ),
)


@pytest.mark.asyncio
async def test_a_page_that_answers_everything_still_answers_everything(
    extractor_eval: ExtractorEval,
) -> None:
    """The unchanged shape: three things named, three things on the page, three things
    back.  It is here so a per-field read cannot be bought by loosening the ordinary
    case — if this moves, the change did something other than what it claims."""
    await _run_case(extractor_eval, _ALL_PRESENT)


# ── Case 2: the instruction allows a thing to be absent, and it is ────────────

# The two things this page DOES supply, named once: the deterministic coherence probe holds
# them against the page, the unported fixtures state them as expectations, and the ported
# case's provenance claims read the same two strings.  One source of truth, so a page edit
# cannot leave a claim asserting a span the page no longer carries.
_HEADLINE_ANCHOR = "Lantern festival draws a record crowd to the old quarter"
_LINK_ANCHOR = "https://news-alpha.example/world/2036/lantern-festival-draws-record-crowd"

_PARTLY_PRESENT = ExtractFixture(
    case_id="extract-fields-partly-present",
    url=_HOMEPAGE_URL,
    page=_HOMEPAGE,
    instruction=(
        "the headlines and their links, with a one-line summary of each where the page gives one"
    ),
    expectations=(
        FieldExpectation("headline", _HEADLINE_ANCHOR),
        FieldExpectation("link", _LINK_ANCHOR),
        FieldExpectation("summary"),
    ),
)


# ── The ported case: five wordings of that one instruction ────────────────────
#
# The arm here is the ``extract`` instruction, and it is PENNY's own text — production
# writes it as the ``extract`` argument of a browse call, at the call site, by whatever
# agent made the call.  So these five are not five ways a person might ask; they are five
# ways the CALLING DRAW might have worded the same request, which is variation that happens
# in production today and that nothing measures.
#
# Two of them hedge ("where the page gives one", "if there is one") and three do not, which
# is deliberate: whether the answer degrades per field is decided by what the PAGE carries,
# never by whether the instruction was worded to expect a gap.  That is the claim
# ``extract-fields-partly-present-unhedged`` makes with a second page, and pooling both
# wordings into one cohort states it as a variance reading over one world instead.
_PARTLY_PRESENT_PHRASINGS = (
    "the headline of each story, the link to it, and a one-line summary if there is one",
    "for every story: its title, its url, and a short summary",
    "each headline with its link and a one-sentence summary",
    "pull out the story titles, the links they point at, and a brief summary of each",
)


def _carries(anchor: str) -> WorldClaim:
    """A claim that the answer carries one thing the PAGE supplies.

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


def _read_the_page(sample: SampleObservation, _world: World) -> Answer:
    """The decisive one: a page carrying SOME of what was asked for is a read.

    ``NOT_PRESENT`` here is the regression itself — the answer that cost a whole round its
    headlines and links because the page was short of the third thing."""
    outcome = sample.field(EXTRACT_OUTCOME)
    return outcome == MicroExtractOutcome.EXTRACTED.value, f"came back {outcome}"


def _nothing_invented(sample: SampleObservation, _world: World) -> Answer:
    """Every specific value in the answer traces to what the draw was GIVEN.

    The extractor's own failure mode, and the strongest claim available to it: ``value`` is
    free text lifted off the page and production validates none of it, so a plausible headline
    the page never carried reaches the caller verbatim and is written down as read."""
    invented = unsourced_specifics(sample.output_text, sample.given)
    return not invented, f"not on the page: {invented}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_with_titles_and_links_and_no_summaries_still_reads(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The regression itself: the page has two of the three things named and the third is
    genuinely not on it.  The read has to come back with the two, not answer as though the
    page were empty.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_PARTLY_PRESENT.case_id,
        model=model,
        url=_PARTLY_PRESENT.url,
        page=_PARTLY_PRESENT.page,
        instruction=_PARTLY_PRESENT.instruction,
        also_instructed=_PARTLY_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _read_the_page,
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

    # What is MEASURED — the same structured fields, compared across the cohort.  No tool
    # sequence and no reply spread: a single call makes neither, and measuring one would
    # print a feature that cannot see an outlier.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
    )


# ── Case 3: the same, with an instruction that hedges nothing ─────────────────

_PARTLY_PRESENT_UNHEDGED = ExtractFixture(
    case_id="extract-fields-partly-present-unhedged",
    url=_SECTION_URL,
    page=_SECTION,
    instruction="the headline, the byline and the published time for each story",
    expectations=(
        FieldExpectation("headline", "City orchestra names a conductor from within the ranks"),
        FieldExpectation("byline", "Ines Marlowe"),
        FieldExpectation("published time"),
    ),
)


@pytest.mark.asyncio
async def test_an_unhedged_instruction_degrades_the_same_way(
    extractor_eval: ExtractorEval,
) -> None:
    """Nothing in this instruction says a thing may be missing, and the page is short of
    one anyway — which is the ordinary case, since a task states what it wants and not
    what a page might lack.  What decides the outcome is the page, so this must read the
    same as the hedged one; a contract that only holds for instructions worded to expect
    a gap is keyed to the wording rather than to the state."""
    await _run_case(extractor_eval, _PARTLY_PRESENT_UNHEDGED)


# ── Case 4: a page with none of it (the honesty guard) ────────────────────────

_NONE_PRESENT = ExtractFixture(
    case_id="extract-fields-none-present",
    url=_HOMEPAGE_URL,
    page=_HOMEPAGE,
    instruction="the closing price of each company named and its ticker symbol",
    expectations=(
        FieldExpectation("closing price"),
        FieldExpectation("ticker symbol"),
    ),
)


@pytest.mark.asyncio
async def test_a_page_carrying_none_of_it_still_says_so(extractor_eval: ExtractorEval) -> None:
    """The over-correction guard.  A read that degrades per field must not degrade into
    answering anyway: this page carries neither thing, so the honest answer is that it
    carries neither, and anything else is a value the page never held."""
    await _run_case(extractor_eval, _NONE_PRESENT)


# Every fixture, for the deterministic coherence probe in ``make check``.
FIXTURES = (_ALL_PRESENT, _PARTLY_PRESENT, _PARTLY_PRESENT_UNHEDGED, _NONE_PRESENT)
