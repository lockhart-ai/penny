"""Live-model contract for the skill BINDER (#1867, beat 1 of #1866).

The framer writes a routine's interface once, from the round that taught it.  The binder
runs every time that routine is asked for again: given the signature exactly as it
already stands, and the user's own words of THIS round, it says what each declared
parameter's value is.

It mints nothing and it judges nothing.  The parameter set is an INPUT, so the only
decision in the draw is which part of the user's words fills each declared parameter —
and that makes the whole answer checkable in Python before it is ever scored: production
refuses a value that is not a literal span of what the user said, and refuses a draw that
answers for a parameter nobody declared or leaves one unanswered.  What these cases
measure is what is left after that: whether it picked the RIGHT span, and whether it knew
when to decline.

Seven cases, both directions of the contract:

* ``bind-listing-page`` — one url parameter, the ask names the page.
* ``bind-two-parameters`` — the page AND what to look for on it, out of one message.
* ``bind-daily-special`` — one url, the ask states its cadence in the same breath.
* ``bind-count-page`` — one url under a threshold ask.
* ``bind-new-arrivals`` — one url under an act-now ask with an end date in it.
* ``bind-missing-page`` — the SHORTFALL: an ask that describes the job and names no page.
* ``bind-missing-keyword`` — the shortfall beside a successful bind: the page is there,
  what to look for on it is not.

The asks are the shapes the idle→apply beat measures — a cold second ask pointing a
routine Penny already knows at a new space — because that is where the binder runs.  Each
one carries its job's TERMS as well (every hour until sunday, each day, every two hours
until friday), which is the second thing every case checks: terms are settled where the
job is set running, so a term inside a bound value is the draw reading them as part of the
thing to point at.

Each case's input is rendered by the shipped ``render_spoken_turns`` +
``build_binding_content`` — never hand-written — so the draw reads exactly what production
would render.  ``rendered_input`` is that document, pinned byte-for-byte by a
deterministic drift probe in ``make check`` (see ``tests/test_eval_harness.py``): a
fixture that drifts from the pair it claims is a case measuring nothing, and it must fail
before any GPU time, not after.

Every case is report-only; the thresholds are the code owner's once the first numbers are
read.  All content is synthetic, and the pages it names are the ones the transition suites
already use.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database.skills import SkillParameter
from penny.tests.eval.conftest import BinderEval, BoundExpectation

pytestmark = pytest.mark.eval

_FAMILY = "skill-binding"


class BindingFixture(NamedTuple):
    """One agreed case: the routine as it already stands, the user's turns asking for it
    again, the document the shipped renderers must produce from the pair, what each
    declared parameter should come back as, and the job terms the ask carries — none of
    which may appear inside a value."""

    case_id: str
    skill: str
    intent: str
    parameters: tuple[SkillParameter, ...]
    turns: tuple[str, ...]
    rendered_input: str
    expectations: tuple[BoundExpectation, ...]
    forbidden: tuple[str, ...]


async def _run_case(binder_eval: BinderEval, fixture: BindingFixture) -> None:
    """Drive one case's signature + turns through the binder.  Every case is report-only:
    the thresholds are the code owner's to set once the first numbers are read."""
    await binder_eval(
        case_id=fixture.case_id,
        turns=fixture.turns,
        skill=fixture.skill,
        intent=fixture.intent,
        parameters=fixture.parameters,
        expectations=fixture.expectations,
        forbidden=fixture.forbidden,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


# The two signatures the cases are drawn against, each one the shape the framer really
# produces for that kind of routine (the transitions suite seeds the same pair): one
# parameter for a routine whose framing already says what it is looking for, and two when
# the thing to look for is its own piece.
_PRICE_PARAMETERS = (SkillParameter(name="url", description="The URL of the listing to watch"),)
_TIMETABLE_PARAMETERS = (
    SkillParameter(name="url", description="the URL of the timetable page to fetch"),
    SkillParameter(name="keyword", description="text indicating which timetable entry to look for"),
)


# ── Case 1: one url, and the ask names the page ───────────────────────────────
#
# Reference values (read at review, never matched):
#   url = https://faux-market.example/keel-lantern

_LISTING = BindingFixture(
    case_id="bind-listing-page",
    skill="monitor_price",
    intent="Monitors a web listing and reports when its price changes.",
    parameters=_PRICE_PARAMETERS,
    turns=(
        "can you watch this listing for me and let me know when the price changes? "
        "https://faux-market.example/keel-lantern — every hour until sunday night is fine",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_price\n"
        "what it is for: Monitors a web listing and reports when its price changes.\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: The URL of the listing to watch\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you watch this listing for me and let me know when the price changes? "
        "https://faux-market.example/keel-lantern — every hour until sunday night is fine"
    ),
    expectations=(BoundExpectation("url", "faux-market.example/keel-lantern"),),
    forbidden=("every hour", "sunday"),
)


@pytest.mark.asyncio
async def test_the_page_in_the_ask_fills_the_one_parameter(binder_eval: BinderEval) -> None:
    """The simplest shape there is: one declared parameter, one address in the message.

    The whole ask is one turn and carries its cadence and its end in the same breath — so
    what is measured beside the bind is restraint, because "every hour until sunday night"
    sits directly beside the url the value has to be."""
    await _run_case(binder_eval, _LISTING)


# ── Case 2: two parameters, both out of one message ───────────────────────────
#
# Reference values (read at review, never matched):
#   url     = https://northpier.example/departures
#   keyword = dawn sailing

_TWO_PARAMETERS = BindingFixture(
    case_id="bind-two-parameters",
    skill="check_ferry_timetable",
    intent="Check a ferry timetable page for updates and report the status of a specified line",
    parameters=_TIMETABLE_PARAMETERS,
    turns=(
        "every morning can you check the north pier timetable at "
        "https://northpier.example/departures and let me know when they add the dawn sailing?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: check_ferry_timetable\n"
        "what it is for: Check a ferry timetable page for updates and report the status "
        "of a specified line\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the timetable page to fetch\n"
        "- keyword: text indicating which timetable entry to look for\n"
        "\n"
        "What the user said, in their own words:\n"
        "every morning can you check the north pier timetable at "
        "https://northpier.example/departures and let me know when they add the dawn sailing?"
    ),
    expectations=(
        BoundExpectation("url", "northpier.example/departures"),
        BoundExpectation("keyword", "dawn sailing"),
    ),
    forbidden=("every morning",),
)


@pytest.mark.asyncio
async def test_two_declared_parameters_take_two_different_spans(binder_eval: BinderEval) -> None:
    """The stress case for filling a signature: one message supplies BOTH the page and the
    thing to look for on it, and they are different kinds of value in the same sentence.

    A binder that reads the page for both, or the phrase for both, has bound a routine
    that will read the right page for the wrong thing — which is why each parameter is its
    own check rather than a count."""
    await _run_case(binder_eval, _TWO_PARAMETERS)


# ── Case 3: one url, the cadence stated as part of the sentence ───────────────
#
# Reference values (read at review, never matched):
#   url = https://harborbakery.example/menu

_DAILY_SPECIAL = BindingFixture(
    case_id="bind-daily-special",
    skill="fetch_daily_special",
    intent="retrieve the daily special from a bakery webpage",
    parameters=(
        SkillParameter(name="url", description="the URL where the daily specials are listed"),
    ),
    turns=(
        "can you get the daily special from https://harborbakery.example/menu each day "
        "and tell me what it is?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: fetch_daily_special\n"
        "what it is for: retrieve the daily special from a bakery webpage\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL where the daily specials are listed\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you get the daily special from https://harborbakery.example/menu each day "
        "and tell me what it is?"
    ),
    expectations=(BoundExpectation("url", "harborbakery.example/menu"),),
    forbidden=("each day",),
)


@pytest.mark.asyncio
async def test_the_cadence_in_the_sentence_is_not_part_of_the_value(
    binder_eval: BinderEval,
) -> None:
    """ "each day" sits between the url and the rest of the sentence, so the value and the
    term are neighbours in the text.

    They are settled in different places — the value points the routine, the cadence is
    set when the job is stood up — so a bind that swept the cadence in has made the
    routine's identity depend on how often it runs."""
    await _run_case(binder_eval, _DAILY_SPECIAL)


# ── Case 4: one url under a threshold ask ─────────────────────────────────────
#
# Reference values (read at review, never matched):
#   url = https://riverotters.example/census

_COUNT = BindingFixture(
    case_id="bind-count-page",
    skill="monitor_webpage_number",
    intent="track a numeric value on a webpage over time to detect changes",
    parameters=(SkillParameter(name="url", description="the webpage to monitor"),),
    turns=(
        "keep track of the otter count at https://riverotters.example/census every week "
        "and let me know if it drops",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_webpage_number\n"
        "what it is for: track a numeric value on a webpage over time to detect changes\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the webpage to monitor\n"
        "\n"
        "What the user said, in their own words:\n"
        "keep track of the otter count at https://riverotters.example/census every week "
        "and let me know if it drops"
    ),
    expectations=(BoundExpectation("url", "riverotters.example/census"),),
    forbidden=("every week",),
)


@pytest.mark.asyncio
async def test_a_threshold_ask_still_binds_only_the_page(binder_eval: BinderEval) -> None:
    """The ask carries a condition — tell me if it drops — and the routine declares one
    parameter, the page.

    So the condition has nowhere to go, and a signature with nowhere to put something is
    exactly where an invented parameter or a padded value would show up."""
    await _run_case(binder_eval, _COUNT)


# ── Case 5: one url under an act-now ask with an end date in it ───────────────
#
# Reference values (read at review, never matched):
#   url = https://eastbranch.example/new-titles

_NEW_ARRIVALS = BindingFixture(
    case_id="bind-new-arrivals",
    skill="retrieve_newest_item",
    intent="Checks a web page and returns its newest arrival",
    parameters=(SkillParameter(name="url", description="the URL of the list to check"),),
    turns=(
        "watch https://eastbranch.example/new-titles every two hours until friday and "
        "tell me when something new shows up",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: retrieve_newest_item\n"
        "what it is for: Checks a web page and returns its newest arrival\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the list to check\n"
        "\n"
        "What the user said, in their own words:\n"
        "watch https://eastbranch.example/new-titles every two hours until friday and "
        "tell me when something new shows up"
    ),
    expectations=(BoundExpectation("url", "eastbranch.example/new-titles"),),
    forbidden=("every two hours", "friday"),
)


@pytest.mark.asyncio
async def test_the_url_opens_the_ask_and_the_terms_follow_it(binder_eval: BinderEval) -> None:
    """The address is the FIRST thing in the message and both terms follow it, which is the
    layout most likely to produce a value that runs on past its end.

    A cadence and an end date immediately after the url is where "watch <url> every two
    hours until friday" becomes one long value if the draw takes the rest of the
    clause."""
    await _run_case(binder_eval, _NEW_ARRIVALS)


# ── Case 6: the shortfall — the job is described and no page is named ─────────
#
# Reference values (read at review, never matched):
#   url = MISSING

_MISSING_PAGE = BindingFixture(
    case_id="bind-missing-page",
    skill="monitor_price",
    intent="Monitors a web listing and reports when its price changes.",
    parameters=_PRICE_PARAMETERS,
    turns=(
        "can you keep an eye on the price of that brass lantern i was looking at and "
        "tell me when it changes? every hour is fine",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_price\n"
        "what it is for: Monitors a web listing and reports when its price changes.\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: The URL of the listing to watch\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you keep an eye on the price of that brass lantern i was looking at and "
        "tell me when it changes? every hour is fine"
    ),
    expectations=(BoundExpectation("url"),),
    forbidden=("every hour",),
)


@pytest.mark.asyncio
async def test_an_ask_that_names_no_page_reports_the_page_missing(binder_eval: BinderEval) -> None:
    """The ask is a perfectly good description of the job and supplies nothing to point it
    at: the user refers to a listing they were looking at and never says which.

    The temptation is a value that is right there in the sentence and is not a page —
    "that brass lantern" reads like an answer, and a routine bound to it would go and
    watch nothing.  Naming the parameter missing is the answer the contract asks for, and
    it is what the request state will act on."""
    await _run_case(binder_eval, _MISSING_PAGE)


# ── Case 7: the shortfall beside a successful bind ────────────────────────────
#
# Reference values (read at review, never matched):
#   url     = https://northpier.example/departures
#   keyword = MISSING

_MISSING_KEYWORD = BindingFixture(
    case_id="bind-missing-keyword",
    skill="check_ferry_timetable",
    intent="Check a ferry timetable page for updates and report the status of a specified line",
    parameters=_TIMETABLE_PARAMETERS,
    turns=(
        "can you check the timetable at https://northpier.example/departures every "
        "morning and keep me posted?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: check_ferry_timetable\n"
        "what it is for: Check a ferry timetable page for updates and report the status "
        "of a specified line\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the timetable page to fetch\n"
        "- keyword: text indicating which timetable entry to look for\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you check the timetable at https://northpier.example/departures every "
        "morning and keep me posted?"
    ),
    expectations=(
        BoundExpectation("url", "northpier.example/departures"),
        BoundExpectation("keyword"),
    ),
    forbidden=("every morning",),
)


@pytest.mark.asyncio
async def test_one_parameter_binds_while_the_other_is_reported_missing(
    binder_eval: BinderEval,
) -> None:
    """The two directions in one draw: the page is in the message and what to look for on
    it is not.

    This is the shape the request state exists for — enough of the ask has landed to be
    worth keeping, and one named thing is outstanding — so the answer has to carry both
    halves: the missing parameter named, and the bound one not thrown away on the way to
    reporting it."""
    await _run_case(binder_eval, _MISSING_KEYWORD)


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (
    _LISTING,
    _TWO_PARAMETERS,
    _DAILY_SPECIAL,
    _COUNT,
    _NEW_ARRIVALS,
    _MISSING_PAGE,
    _MISSING_KEYWORD,
)
