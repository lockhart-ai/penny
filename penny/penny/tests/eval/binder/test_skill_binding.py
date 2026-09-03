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

Eight cases, both directions of the contract:

* ``bind-listing-page`` — one url parameter, the ask names the page.
* ``bind-two-parameters`` — the page AND what to look for on it, out of one message.
* ``bind-daily-special`` — one url, the ask states its cadence in the same breath.
* ``bind-count-page`` — one url under a threshold ask.
* ``bind-new-arrivals`` — one url under an act-now ask with an end date in it.
* ``bind-missing-page`` — the SHORTFALL: an ask that describes the job and names no page.
* ``bind-missing-keyword`` — the shortfall beside a successful bind: the page is there,
  what to look for on it is not.
* ``binder-fills-one-and-names-the-other-missing`` — the slot's CANONICAL case (#2006):
  that same ask in five wordings, pooled into a cohort of fifteen and claimed against
  ``docs/eval-case-design.md`` rather than the per-parameter scorer below.

Since #1894 the binder is the ONE door for every entry against a routine the registry
already holds — a cold apply and a request the classifier drew directly both come through
it — and a round coming back for a missing detail hands over what it already SETTLED, so
only the still-open parameters are drawn.  These cases drive the COLD shape: the whole
declared set, nothing settled, which is the ask the idle→apply and idle→request beats
measure — a second ask pointing a routine Penny already knows at a new space.  Each one
carries its job's TERMS as well (every hour until sunday, each day, every two hours until
friday), which is the second thing every case checks: terms are settled where the job is
set running, so a term inside a bound value is the draw reading them as part of the thing
to point at.

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

from penny.constants import PennyConstants
from penny.database.skills import SkillParameter
from penny.tests.eval.conftest import (
    BIND_MISSING,
    BIND_OUTCOME,
    EVAL_MODELS,
    BinderEval,
    BindOutcome,
    BoundExpectation,
    bound_value_field,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    FIELD_UNSET,
    Consequence,
    SampleObservation,
    SpecCategory,
    output_field,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import spoken_form

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
    since #1885 it is what ROUTES the turn into request — an enumerated outcome the
    machine acts on, never a failed draw."""
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
    reporting it.  Since #1894 that surviving half becomes the round's own state, handed
    back to the next draw as its settled values, so a page given now is never asked for
    again."""
    await _run_case(binder_eval, _MISSING_KEYWORD)


# ── The ported case: one parameter filled, one reported missing ───────────────
#
# The survivor is ``bind-missing-keyword``, because it is the one ask that states the whole
# behaviour in a single draw: the page is in the message and what to look for on it is not,
# so the answer has to carry both halves — the value it could read, and the parameter
# nothing supplied — while the cadence sitting beside the address stays out of either.
#
# THE FACTS ARE CONSTANT across the five wordings, because every claim hinges on them.  The
# address appears verbatim in each; the cadence appears verbatim as "every morning"; and no
# wording names a timetable entry, so the shortfall is a property of the ask rather than of
# how it was phrased.  Varying the cadence's own words would be varying the world a claim
# reads, not the words a person used.
#
# An arm is a SEQUENCE of turns, matching the framer's shape: an ask is however many turns
# the user took to make it, and ``render_spoken_turns`` renders them as the haystack the
# span check reads.
_MISSING_KEYWORD_PHRASINGS = (
    (
        "every morning could you look at https://northpier.example/departures for me "
        "and let me know?",
    ),
    (
        "would you mind checking https://northpier.example/departures every morning and "
        "telling me what you find?",
    ),
    ("please look at https://northpier.example/departures every morning and keep me in the loop",),
    (
        "i'd like https://northpier.example/departures checked every morning — just let "
        "me know how it looks",
    ),
)

# The case's id and its five arms, named at module level so the deterministic probe in
# ``make check`` can hold every arm against the facts it claims — the address, the cadence,
# and the absence of anything answering the keyword — before any GPU time is spent.
MISSING_KEYWORD_CASE_ID = "binder-fills-one-and-names-the-other-missing"
MISSING_KEYWORD_ARMS = (_MISSING_KEYWORD.turns, *_MISSING_KEYWORD_PHRASINGS)

# The two declared parameters, by the side of the contract each one is: the ask supplies an
# address for the first and nothing at all for the second.  Named rather than indexed at each
# claim, so a signature edit that reordered them breaks here rather than silently swapping
# which parameter every claim is about.  The drift probe already holds this pair against the
# signature's own declared order.
_SUPPLIED = _MISSING_KEYWORD.expectations[0]
_UNSUPPLIED = _MISSING_KEYWORD.expectations[1]

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_MISSING_KEYWORD_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_BIND_AGENT_NAME} micro-context, when a routine Penny "
    "already knows is pointed at something new and the ask supplies only part of what it "
    "declares, Penny fills each parameter from the span that supplies it and names the one "
    "nothing supplies — taking the job's cadence into neither."
)


def _reported_a_shortfall(sample: SampleObservation, _world: World) -> Answer:
    """The draw came back a SHORTFALL rather than a complete binding.

    An enumerated outcome, asserted by equality: production answers with ``BoundValues``
    when every declared parameter got a value and ``MissingParameters`` when one did not,
    and which of the two it wrote is the decision this case is about.  It is not validated
    anywhere upstream — the draw chooses, per parameter, whether to write a value line or a
    missing line — so this is the open question and not the validator's.

    The wrong answer is the interesting one: a complete binding here means the draw filled
    the keyword from something in the sentence that is not a timetable entry, which is a
    routine that will read the right page for the wrong thing."""
    outcome = sample.field(BIND_OUTCOME)
    return outcome == BindOutcome.SHORTFALL.value, f"came back {outcome}"


def _names_the_keyword_missing(sample: SampleObservation, _world: World) -> Answer:
    """It names the keyword, and only the keyword, as the thing nothing supplies.

    Equality over a closed set — the parameters the signature DECLARES — so this is the
    smallest datum that identifies the answer: naming the url as well would be a routine
    that cannot be pointed anywhere despite being handed an address, and naming nothing at
    all is the outcome claim above said a second way."""
    reported = sample.field(BIND_MISSING)
    return reported == _UNSUPPLIED.parameter, f"reported {reported!r}"


def _the_url_is_the_span_that_supplies_it(sample: SampleObservation, _world: World) -> Answer:
    """The bound url carries the address the ask supplies — the *nothing omitted* half.

    Production validates that a value is A span of the user's words (``_is_a_spoken_span``)
    and re-rolls until it is, so what is left to measure is whether it is THE span: an ask
    carrying one address and one cadence has two spans a draw could have taken, and only
    one of them is a page.

    The anchor is the SMALLEST datum unique in this world — the host and path, without the
    scheme — because a draw that bound the address with ``https://`` and one that bound it
    without read the ask exactly as well, and the scheme is a rendering the draw was free to
    choose.  Compared through the shipped ``spoken_form``, so what a case calls a match and
    what production calls a span are one definition."""
    bound = sample.field(bound_value_field(_SUPPLIED.parameter))
    if bound == FIELD_UNSET:
        return False, "no value came back for it"
    carried = spoken_form(_SUPPLIED.anchor) in spoken_form(bound)
    return carried, f"bound {bound!r}, not the page the ask names"


def _no_value_carries_the_cadence(sample: SampleObservation, _world: World) -> Answer:
    """No bound value carries the job's cadence — the *nothing invented* half.

    How often a routine runs is settled where the job is set running, never by the binder,
    so a cadence INSIDE a value means the draw read the terms as part of the thing to point
    the routine at, and the routine's identity then depends on how often it runs.  It is a
    fact of this world rather than a vocabulary rule: the case declares the words its own
    ask carries, exactly as a world declares what must not be kept from a page, so nothing
    here is keyed to a list of timing words in general."""
    bound = {
        one.parameter: sample.field(bound_value_field(one.parameter))
        for one in _MISSING_KEYWORD.expectations
    }
    offenders = [
        f"{name} ({term})"
        for name, value in bound.items()
        if value != FIELD_UNSET
        for term in _MISSING_KEYWORD.forbidden
        if spoken_form(term) in spoken_form(value)
    ]
    return not offenders, f"carried the terms: {'; '.join(offenders)}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_page_binds_and_the_entry_is_reported_missing(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: check this timetable page every morning and keep me posted.

    The page is in the message and what to look for on it is not, so the answer has to carry
    both halves — the missing parameter named, and the bound one not thrown away on the way
    to reporting it — with the cadence in neither.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    to no store — so there is nothing for a store claim to read.  What this binding becomes
    is round state on a later transition, and this case never runs that.

    **Three claims are missing because production already validates them**, and a thin set
    should read as closed rather than as unrun.  Two are ``_fills_the_declared_signature``'s:
    that the drawn names equal the declared names exactly (so no parameter is answered
    twice, left unanswered or invented), and that every value is a literal span of what the
    user said.  The third is a GENERATED record rather than a validator — that nothing was
    bound for the parameter reported missing.  ``_skill_binding`` builds ``values`` from the
    value lines and ``names`` from the rest, so a name in one can never be in the other: the
    partition is the framework's arithmetic, not the draw's answer.
    """
    cohort = await binder_eval(
        case_id=MISSING_KEYWORD_CASE_ID,
        behaviour=_MISSING_KEYWORD_BEHAVIOUR,
        model=model,
        turns=_MISSING_KEYWORD.turns,
        also_phrased=_MISSING_KEYWORD_PHRASINGS,
        skill=_MISSING_KEYWORD.skill,
        intent=_MISSING_KEYWORD.intent,
        parameters=_MISSING_KEYWORD.parameters,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the CLOSED fields of the typed result, asserted by equality: which of the two
    # enumerated answers the draw wrote, and which declared parameter it named.
    cohort.claim(
        "state: the draw reported a shortfall rather than a complete binding",
        _reported_a_shortfall,
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: it names the keyword as the one thing nothing supplies",
        _names_the_keyword_missing,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — the OPEN field, which for this shape is each bound value, read in both
    # directions: the value carries the span of the ask that supplies it, and it carries
    # nothing the parameter does not own.
    cohort.claim(
        "state: the url is bound to the address the ask supplies",
        _the_url_is_the_span_that_supplies_it,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence",
        _no_value_carries_the_cadence,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields.
    #
    # The OUTCOME and what it reported missing are CONSEQUENTIAL: a complete binding and a
    # shortfall are two different end states, and which parameter is named decides what the
    # turn goes on to ask for.  The bound url is COSMETIC — a draw that kept the scheme and
    # one that dropped it point the routine at the same page.
    #
    # The KEYWORD's own value axis is absent, deliberately: a correct draw binds nothing for
    # it, so the field is absent from every sample and an axis over it would read `unset`
    # throughout, which the pooler reports as blind rather than as agreement.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(BIND_OUTCOME),
        output_field(BIND_MISSING),
        output_field(bound_value_field(_SUPPLIED.parameter), consequence=Consequence.COSMETIC),
    )


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
