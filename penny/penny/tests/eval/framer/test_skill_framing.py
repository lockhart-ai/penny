"""Live-model contract for the skill FRAMER (#1830, the #1824 inversion).

The framer has ONE job: given the user's own turns of the round and nothing else, write
the routine's public interface — a generic name, a one-line description, and the
parameter(s) the user would have to say again to set the same routine running on a new
occasion.  It never sees the tool calls; the leaf labeller, which names the routine's
implementation, never sees the ask (#1824).  Nothing is offered to it: it MINTS the
parameters by reading what the user said.

Eight cases, all the same look-up → find → remember shape, deliberately spanning the
three multiplicity shapes — one argument, two of the SAME type, two of DIFFERENT types —
while varying topic, how much the ask names, and how many pieces have to be re-supplied:

* ``frame-availability-page-only`` — the ask names the point, so what to check bakes.
* ``frame-two-sources-two-parameters`` — two pages named: TWO distinct parameters.
* ``frame-ticker-only-parameter`` — "tell me when it moves" contributes no parameter.
* ``frame-single-turn-floor`` — one turn: the finding bakes, the page survives.
* ``frame-search-parameter`` — the look-up is a text search, not a page.
* ``frame-two-types-page-and-title`` — a page + a title: two pieces, different types.
* ``frame-two-same-type-symbols`` — two symbols, where a single list parameter is most
  tempting and each must stay its own scalar.
* ``framer-mints-only-the-piece-that-varies`` — the slot's CANONICAL case (#2006): the
  ticker ask in five wordings, pooled into a cohort of fifteen and claimed against
  ``docs/eval-case-design.md`` rather than the per-check scorer below.

Each case's input is the round's USER turns, rendered by the shipped
``build_framing_content`` — never hand-written — so the draw reads exactly what
production would render.  ``rendered_input`` is that document, pinned byte-for-byte by a
deterministic drift probe in ``make check`` (see ``tests/test_eval_harness.py``): a
fixture that drifts from the pair it claims is a case measuring nothing, and it must
fail before any GPU time, not after.

Scoring is the parameter SET, exactly — each expected family answered by exactly one
drawn parameter, nothing else asked for — plus the structural check that the name and
description say the KIND of task and never the occasion.  The reference outputs below
each case are agreed TARGETS read at joint review, never strings a scorer matches: a
parameter the reference calls ``url`` may come back as ``page_to_watch`` and pass.  Every
drawn name, description and parameter rides ADVISORY so a reader sees what the model
committed to.

Since #1868 the draw happens when the machine ENTERS learn rather than at run end: the
round's identity is settled before the round runs, and run-end extraction READS that
framing instead of drawing again — a run-end draw survives only for a round nothing
framed.  Both entries render their document through the same shipped
``build_framing_content``, so the draw this case drives is the one either path makes.  The
draw also gives each parameter the VALUE the round demonstrated it with, and those values
are what a job's container is NAMED from — so each parameter's advisory carries its drawn
value and the run closes with the container name the shipped derivation makes of them.
That a value is a literal span of the user's own words is the production validator's job
(an accepted draw cannot carry a value nobody said); WHICH span was the right one is the
same kind of judgment as a name, so it is rendered for review rather than matched by a
fixture.

All content is synthetic.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    FRAME_DESCRIPTION,
    FRAME_NAME,
    FRAME_PARAMETERS,
    FramerEval,
    ParameterFamily,
    classify_by_family,
    frame_parameter_name,
    frame_parameter_says,
    framed_parameters,
)
from penny.tests.eval.utils.assertions import Answer, WorldClaim
from penny.tests.eval.utils.cohort import (
    Consequence,
    SampleObservation,
    SpecCategory,
    output_field,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"


class FramingFixture(NamedTuple):
    """One agreed case: the user turns that are its whole input, the document the
    shipped renderer must produce from them, the parameters the ask genuinely requires,
    and the occasion's own words — which may appear in neither the name nor the
    description."""

    case_id: str
    turns: tuple[str, ...]
    rendered_input: str
    parameters: tuple[ParameterFamily, ...]
    instance_tokens: tuple[str, ...]


async def _run_case(framer_eval: FramerEval, fixture: FramingFixture) -> None:
    """Drive one case's turns through the framer.  Every case is report-only: the
    thresholds are the code owner's to set once the first numbers are read."""
    await framer_eval(
        case_id=fixture.case_id,
        turns=fixture.turns,
        parameters=fixture.parameters,
        instance_tokens=fixture.instance_tokens,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


# The breadth agreed for "the page the routine is pointed at" — a piece the user
# re-supplies that is a place to go.  Shared by the page cases because it is the same
# piece under three topics; a case that needs to tell two of them apart adds ordinals.
#
# A page family is NAME-ONLY (the code owner's ruling on the first run): a parameter is
# the page when it is NAMED as the page.  Two identical `city — name of the location on
# the site …` draws scored opposite ways when a description-level mention of "site"
# could promote one, which is the scorer answering for the draw.
_PLACE_TOKENS = ("url", "page", "link", "address", "site", "source", "product", "listing")


def _page_family(label: str, *extra: str) -> ParameterFamily:
    """The page/url family for a case, name-only, with any case-specific words."""
    return ParameterFamily(label, (*_PLACE_TOKENS, *extra), name_only=True)


# ── Case 1: the ask names the point, so only the page is left to say ──────────
#
# Reference output (read at review, never matched):
#   NAME: stock-watcher
#   DESCRIPTION: watch a product page for the item coming back in stock
#   PARAMETER url — the product page to check

_AVAILABILITY = FramingFixture(
    case_id="frame-availability-page-only",
    turns=(
        "can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
        "when it's back in stock",
        "go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that",
    ),
    rendered_input=(
        "can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
        "when it's back in stock\n"
        "go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that"
    ),
    parameters=(_page_family("url"),),
    instance_tokens=("bookbarn", "atlas", "clouds"),
)


@pytest.mark.asyncio
async def test_what_the_ask_named_bakes_and_the_page_survives(framer_eval: FramerEval):
    """The user said what they were after — the item coming back in stock — so a routine
    that asks what to check would be asking them to restate what they came for.  The page
    is the one piece a new occasion needs, and the framing carries the rest."""
    await _run_case(framer_eval, _AVAILABILITY)


# ── Case 2: two pages named is two parameters ─────────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: headline-collector
#   DESCRIPTION: collect the top headline from each of the news front pages it is
#                pointed at
#   PARAMETER first_site — the first front page to read
#   PARAMETER second_site — the second front page to read

_TWO_SOURCES = FramingFixture(
    case_id="frame-two-sources-two-parameters",
    turns=(
        "hey could you keep an eye on the morning headlines for me",
        "read citydesk.example/front and harborpost.example/front, and remember each "
        "site's top headline",
    ),
    rendered_input=(
        "hey could you keep an eye on the morning headlines for me\n"
        "read citydesk.example/front and harborpost.example/front, and remember each "
        "site's top headline"
    ),
    parameters=(
        ParameterFamily("first source", ("first", "one", "1", "primary")),
        ParameterFamily("second source", ("second", "two", "2", "secondary", "other")),
    ),
    instance_tokens=("citydesk", "harborpost"),
)


@pytest.mark.asyncio
async def test_two_sources_become_two_distinct_parameters(framer_eval: FramerEval):
    """The user pointed the routine at two pages, so both have to be re-suppliable and
    they have to be told apart.  A framer that collapses them into one parameter, or goes
    generic and asks for none, has lost a piece the user named — which is why the COUNT
    is the load-bearing check here."""
    await _run_case(framer_eval, _TWO_SOURCES)


# ── Case 3: cadence and notification are not signature ────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: stock-tracker
#   DESCRIPTION: track a stock's share price
#   PARAMETER ticker — the stock symbol to track

_TICKER = FramingFixture(
    case_id="frame-ticker-only-parameter",
    turns=(
        "can you track a stock for me and tell me when it moves",
        "look up VLT, find the share price, and remember it under VLT",
    ),
    rendered_input=(
        "can you track a stock for me and tell me when it moves\n"
        "look up VLT, find the share price, and remember it under VLT"
    ),
    parameters=(ParameterFamily("ticker", ("ticker", "symbol", "stock", "share", "company")),),
    instance_tokens=("vlt",),
)


@pytest.mark.asyncio
async def test_the_share_price_is_the_routine_and_the_ticker_is_the_parameter(
    framer_eval: FramerEval,
):
    """The share price is what the skill IS, so it belongs in the framing; the ticker is
    the one thing said again next time.  "Tell me when it moves" is settled when the
    routine is set running and contributes nothing to the signature — a framer that turns
    it into a parameter has made a delivery preference into something to be re-supplied."""
    await _run_case(framer_eval, _TICKER)


# ── The ported case: the ticker ask, in five wordings ─────────────────────────
#
# The survivor is ``frame-ticker-only-parameter``, because its ask is the plainest
# statement of the whole behaviour AND of its negative direction at once: what the user
# came for (the share price) bakes into the framing, the one thing that varies (the
# symbol) becomes the parameter, and "tell me when it moves" — which is settled where the
# job is set running — becomes neither.
#
# THE FACTS ARE CONSTANT across the five wordings, because the claims hinge on them: every
# arm names the symbol VLT, asks for its share price, says to remember it under VLT, and
# carries a clause about being told when it changes.  What varies is only how a person
# says that.  An arm that dropped the notification clause would be a different ask and its
# samples would be measuring a different behaviour under one case id.
#
# An arm is a SEQUENCE of turns, not a sentence: the round's ask is two turns (the standing
# want, then the demonstration), and ``build_framing_content`` renders the user's turns one
# per line as the whole document.  So a wording of this ask is two turns said differently.
_TICKER_PHRASINGS = (
    (
        "could you keep tabs on a stock for me and let me know when it changes",
        "look up VLT, get the share price, and remember it under VLT",
    ),
    (
        "i'd like a stock followed, with a heads up whenever it moves",
        "check VLT, find the share price, and save it under VLT",
    ),
    (
        "can you watch a stock for me? tell me if it shifts",
        "look VLT up, find the share price, and keep it under VLT",
    ),
    (
        "keep an eye on a stock and ping me when the number moves",
        "look up VLT, find the share price, and store it under VLT",
    ),
)

# The case's id and its five arms, named at module level so the deterministic probe in
# ``make check`` can hold every arm against the facts it claims before any GPU time.
TICKER_CASE_ID = "framer-mints-only-the-piece-that-varies"
TICKER_ARMS = (_TICKER.turns, *_TICKER_PHRASINGS)

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_TICKER_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when a demonstrated "
    "round is turned into a reusable routine, Penny mints one parameter for the piece a new "
    "occasion has to supply and leaves what the user came for — and when they want telling "
    "— in the framing rather than in the interface."
)


def _answers(family: ParameterFamily) -> WorldClaim:
    """A claim that EXACTLY ONE minted parameter answers one piece the ask requires.

    The parameter set is a CLOSED field, so this is asserted by equality under LANDED rather
    than traced under PROVENANCE — but it is still one half of a pair, and its other half is
    the count below: nothing the ask requires was left out, and nothing it does not was
    added.

    "Answers" is decided by :func:`classify_by_family`, the one classification discipline
    every suite that asks what a drawn parameter is for reads.  That is a closed equivalence
    class agreed with the code owner and pinned in ``make check``, not a judgement made per
    run: the family's tokens are the breadth a piece may be named at, so ``ticker``,
    ``symbol`` and ``stock_symbol`` are one answer and a reference name is a target rather
    than a string to match.  Which is what makes this an assertion rather than a reading —
    a differently-worded correct answer passes, and no correct draw can answer it twice.

    Nothing at all answering it is a piece the routine can no longer be pointed at; two
    answering it is the same piece asked for twice."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = framed_parameters(sample)
        matched = [
            name
            for (name, _says), family_of in zip(
                drawn, classify_by_family(drawn, (family,)), strict=True
            )
            if family_of is not None
        ]
        return len(matched) == 1, f"{len(matched)} answer it: {matched or 'none'}"

    return answer


def _asks_for_nothing_else(sample: SampleObservation, _world: World) -> Answer:
    """The count — the other half of the pair above, and the case's negative direction.

    Anything beyond the pieces the ask requires is a piece the user would be made to
    re-supply that their own ask already settled — the cadence, the notification, the thing
    the routine is FOR.  So the count IS the negative direction, and it is stated here
    rather than as a list of timing words: a rule keyed to the vocabulary of the clause in
    front of us would not fire for a clause nobody enumerated, and a parameter minted for
    anything the ask does not require fails this claim whatever it is called."""
    drawn = sample.field(FRAME_PARAMETERS)
    return drawn == str(len(_TICKER.parameters)), f"minted {drawn}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_symbol_is_the_parameter_and_everything_else_bakes(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: track a stock, tell me when it moves, here is the symbol.

    The share price is what the routine IS, so it belongs in the framing; the symbol is the
    one thing said again next time; and being told when it moves is settled where the job is
    set running.  A framer that turns the notification into a parameter has made a delivery
    preference into something to be re-supplied.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    to no store — so there is nothing for a store claim to read.  The signature this draw
    returns is persisted later, by run-end extraction, and this case never runs it.

    **Three claims are missing because production already validates them**, and a thin set
    should read as closed rather than as unrun: that the draw minted at least one parameter,
    that no two parameters share a name, and that every demonstrated value is a literal span
    of the user's own turns are all ``_mints_a_usable_signature``'s, re-rolled until they
    hold, so each would run 15/15 by construction.

    **The PROVENANCE category is empty too, and it needs its reason stated.**  Fact
    alignment reads a draw's OPEN fields, and this draw's are the routine's ``name``, its
    ``description``, each parameter's line, and each parameter's demonstrated ``value``.  The
    value is closed upstream — production refuses one that is not a literal span of the
    user's turns.  The other three are an identifier and two lines of deliberately GENERIC
    prose, which carry no traceable value at all on a correct draw, and the suite's one
    instrument for the invented direction (``unsourced_specifics``) does not transfer to
    them: measured against this case's own document, it reads a Title-Cased but perfectly
    correct framing as four inventions, while a made-up exchange in ``the Nasdaq exchange``
    passes — capitalisation is a rendering the draw chooses, and a single-word invention is
    the probe's declared blind spot.  A check that fails a correct run for a cosmetic reason
    and misses the thing it is for is not an assertion, so this category is empty rather than
    filled with it.

    **And one real contract is NOT measured here, deliberately.**  That the routine's name
    and description say the KIND of task and never THIS occasion — a framing carrying ``VLT``
    is a routine that can only ever run once — fits none of the three assertion categories:
    it is the CONVERSE of a provenance claim (the offending token is in the ask, so nothing
    is invented and nothing is omitted), and inventing a category to keep one measurement is
    how a closed list stops being closed.  It stays what the design calls wrong-but-stable —
    read by a person opening the modal sample, where the drawn name renders verbatim.
    """
    cohort = await framer_eval(
        case_id=TICKER_CASE_ID,
        behaviour=_TICKER_BEHAVIOUR,
        model=model,
        turns=_TICKER.turns,
        also_phrased=_TICKER_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the CLOSED field of the typed result: the parameter SET it minted, by
    # equality.  A framing draw has no enumerated outcome to land on (a signature came back
    # or nothing did, and nothing is the completeness gate's), so the set is what this
    # category holds for this shape.
    for one in _TICKER.parameters:
        cohort.claim(
            f"state: exactly one parameter answers the {one.label}",
            _answers(one),
            SpecCategory.LANDED,
        )
    cohort.claim(
        "state: the routine asks for nothing else",
        _asks_for_nothing_else,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — EMPTY, and see the docstring: this draw has no open field a fact
    # alignment can read.

    # What is MEASURED — the draw's own structured fields.
    #
    # The COUNT is consequential: a routine asking for two things is a different interface
    # from one asking for one.  The name, the description and the drawn parameter's own
    # name and line are COSMETIC — `stock_tracker` and `share_price_watcher` leave the same
    # interface behind — and the naming spread is the framer's known system-level finding
    # (0.90 on both measured models), so it belongs in the variance table and never as a
    # fact about one sample.  Only position 1 is measured: this ask requires one parameter,
    # so a second position exists on a divergent sample alone and its axis would read
    # `unset` for the pack.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(FRAME_PARAMETERS),
        output_field(FRAME_NAME, consequence=Consequence.COSMETIC),
        output_field(FRAME_DESCRIPTION, consequence=Consequence.COSMETIC),
        output_field(frame_parameter_name(1), consequence=Consequence.COSMETIC),
        output_field(frame_parameter_says(1), consequence=Consequence.COSMETIC),
    )


# ── Case 4: one turn is enough to frame ───────────────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: temperature-recorder
#   DESCRIPTION: record the daily high temperature from a weather page
#   PARAMETER url — the weather page to read

_SINGLE_TURN = FramingFixture(
    case_id="frame-single-turn-floor",
    turns=("go to weather.example/lisbon, find today's high temperature, and remember it",),
    rendered_input="go to weather.example/lisbon, find today's high temperature, and remember it",
    parameters=(_page_family("url", "forecast", "weather"),),
    instance_tokens=("lisbon",),
)


@pytest.mark.asyncio
async def test_a_single_turn_teach_still_frames_one_parameter(framer_eval: FramerEval):
    """One page and one thing to find, taught in a single turn: the ask at its minimum
    still says what the routine is for, so the finding bakes into the framing and the
    page survives as the parameter."""
    await _run_case(framer_eval, _SINGLE_TURN)


# ── Case 5: the look-up is a search, not a page ───────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: ticket-price-watcher
#   DESCRIPTION: watch an event's cheapest ticket price
#   PARAMETER ticket_search — the search that finds the event's ticket listings

_SEARCH = FramingFixture(
    case_id="frame-search-parameter",
    turns=(
        "can you keep an eye on ticket prices for aurora fest?",
        "search for aurora fest tickets, find the cheapest ticket price, and remember it",
    ),
    rendered_input=(
        "can you keep an eye on ticket prices for aurora fest?\n"
        "search for aurora fest tickets, find the cheapest ticket price, and remember it"
    ),
    parameters=(
        ParameterFamily("ticket search", ("search", "query", "event", "listing", "listings")),
    ),
    instance_tokens=("aurora", "fest"),
)


@pytest.mark.asyncio
async def test_a_search_is_the_parameter_and_the_cheapest_price_is_the_framing(
    framer_eval: FramerEval,
):
    """The look-up is a text search rather than a url, and what varies next time is which
    event is being searched for.  The cheapest price is what the routine is for, so it
    belongs in the name and description, not in a parameter."""
    await _run_case(framer_eval, _SEARCH)


# ── Case 6: two re-suppliable pieces of DIFFERENT types ───────────────────────
#
# Reference output (read at review, never matched):
#   NAME: catalog-checker
#   DESCRIPTION: check whether a book is available in a library catalog
#   PARAMETER catalog_page — the catalog page to check
#   PARAMETER title — the book to look for

_PAGE_AND_TITLE = FramingFixture(
    case_id="frame-two-types-page-and-title",
    turns=(
        "can you watch the library catalog for a book i'm waiting on?",
        "open town-library.example/catalog, find The Glass Harbour, and remember "
        "whether it's available",
    ),
    rendered_input=(
        "can you watch the library catalog for a book i'm waiting on?\n"
        "open town-library.example/catalog, find The Glass Harbour, and remember "
        "whether it's available"
    ),
    parameters=(
        _page_family("catalog page", "catalog"),
        ParameterFamily("title", ("title", "book", "item", "name")),
    ),
    instance_tokens=("glass", "harbour"),
)


@pytest.mark.asyncio
async def test_two_pieces_of_different_types_are_two_parameters(framer_eval: FramerEval):
    """Availability bakes, exactly as it does when the ask names one page — but here the
    thing being looked for is not IN the address, it is looked up ON the page, so the
    page and the book are two re-suppliable pieces of different types.

    The contrast with the page-only case is the whole point: what the framing carries is
    the finding, not the number of things the routine is pointed at."""
    await _run_case(framer_eval, _PAGE_AND_TITLE)


# ── Case 7: two of the SAME type, where the list temptation is strongest ──────
#
# Reference output (read at review, never matched):
#   NAME: stock-tracker
#   DESCRIPTION: track each given stock's share price
#   PARAMETER first_ticker — the first stock symbol to track
#   PARAMETER second_ticker — the second stock symbol to track

_TWO_SYMBOLS = FramingFixture(
    case_id="frame-two-same-type-symbols",
    turns=(
        "can you keep an eye on a couple of stocks for me?",
        "look up VLT and MERI, find each share price, and remember them",
    ),
    rendered_input=(
        "can you keep an eye on a couple of stocks for me?\n"
        "look up VLT and MERI, find each share price, and remember them"
    ),
    parameters=(
        ParameterFamily("first ticker", ("first", "one", "1", "primary")),
        ParameterFamily("second ticker", ("second", "two", "2", "secondary", "other")),
    ),
    instance_tokens=("vlt", "meri"),
)


@pytest.mark.asyncio
async def test_two_symbols_of_the_same_type_stay_two_scalar_parameters(framer_eval: FramerEval):
    """Two things of the SAME type in one ask are where a single list parameter is most
    tempting — and a list is not a parameter: what a user says fills one whole.  The
    share price bakes into the framing exactly as it does for one symbol, and the count
    lives in the parameters rather than in a description that promises "two"."""
    await _run_case(framer_eval, _TWO_SYMBOLS)


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (
    _AVAILABILITY,
    _TWO_SOURCES,
    _TICKER,
    _SINGLE_TURN,
    _SEARCH,
    _PAGE_AND_TITLE,
    _TWO_SYMBOLS,
)
