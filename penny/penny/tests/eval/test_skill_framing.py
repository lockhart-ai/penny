"""Live-model contract for the run-end skill FRAMER (#1830, the #1824 inversion).

The framer has ONE job: given the user's own turns of the round and nothing else, write
the routine's public interface — a generic name, a one-line description, and the
parameter(s) the user would have to say again to set the same routine running on a new
occasion.  It never sees the tool calls; the leaf labeller, which names the routine's
implementation, never sees the ask (#1824).  Nothing is offered to it: it MINTS the
parameters by reading what the user said.

Five cases, all the same look-up → find → remember shape, varying topic, how much the
ask names, and how many pieces genuinely have to be re-supplied:

* ``frame-availability-page-only`` — the ask names the point, so what to check bakes.
* ``frame-two-sources-two-parameters`` — two pages named: TWO distinct parameters.
* ``frame-ticker-only-parameter`` — "tell me when it moves" contributes no parameter.
* ``frame-single-turn-floor`` — one turn: the finding bakes, the page survives.
* ``frame-search-parameter`` — the look-up is a text search, not a page.

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

All content is synthetic.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.tests.eval.conftest import FramerEval, ParameterFamily

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


# The breadth agreed for "the page/search the routine is pointed at" — a piece the user
# re-supplies that is a place to go.  Shared by the two page cases because it is the same
# piece under two topics; a case that needs to tell two of them apart adds ordinals.
_PLACE_TOKENS = ("url", "page", "link", "address", "site", "source", "product", "listing")


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
    parameters=(ParameterFamily("url", _PLACE_TOKENS),),
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
    parameters=(ParameterFamily("url", (*_PLACE_TOKENS, "forecast", "weather")),),
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


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (_AVAILABILITY, _TWO_SOURCES, _TICKER, _SINGLE_TURN, _SEARCH)
