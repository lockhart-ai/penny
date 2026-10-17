"""Live-model contract for the skill FRAMER (#1830, the #1824 inversion).

The framer has ONE job: given the user's own turns of the round and nothing else, write
the routine's public interface — a generic name, a one-line description, and the
parameter(s) the user would have to say again to set the same routine running on a new
occasion.  It never sees the tool calls; the leaf labeller, which names the routine's
implementation, never sees the ask (#1824).  Nothing is offered to it: it MINTS the
parameters by reading what the user said.

**Seven cases** (#2006/#2056) are the framer's decisions covered in ISOLATION, each one ask
in five wordings pooled into a cohort of fifteen and read against
``docs/eval-case-design.md``.  **Code-owner ruling (#2222)**: which parameters are
emitted, how many, and what they are named are all model OUTPUT — "the number of
prompts and their naming is variance" — so none of that is asserted any more.  What
production already enforces (`_mints_a_usable_signature` / `_values_were_demonstrated`
in ``tools/micro_context.py``: every minted VALUE is a literal span of the user's own
turns, re-rolled on violation) is exactly the provenance claim this design would
otherwise want, and asserting it here would score 15/15 by construction and measure the
validator rather than Penny — so it stays unasserted and stated as a reason.  So every
case makes no claim and MEASURES the parameter set instead (the
family-multiset feature, below) — "it's okay to have eval cases that have no checks and
only measure variance since it's still an important tool for measuring Penny's actual
behaviour" (code owner). Every case still states its behaviour in the fixed sentence
form:

* ``framer-mints-only-the-piece-that-varies`` — the ticker ask: the symbol is the one piece
  that varies, and the share price and the notification are both settled elsewhere.
* ``framer-mints-both-pieces-when-they-are-different-kinds`` — a catalog page and a book
  looked up on it.
* ``framer-names-a-search-as-a-search`` and
  ``framer-names-a-page-as-a-page-and-invents-no-search`` — a PAIR with opposite expected
  answers: a search-family parameter is correct for the first and is exactly the invention
  the second exists to catch (round 8's recorded failure).
* ``framer-frames-from-a-single-turn`` — the whole teach in one sentence, so the routine's
  purpose and the piece that varies have to be separated out of one turn rather than read
  off two.
* ``framer-keeps-three-of-a-kind-as-three-parameters`` — three URLS of the same kind, where
  run 1 measured every two-page sample folding them into one ``sites — list of URLs``; at
  three the list is more tempting still.
* ``framer-settles-an-ask-that-names-a-count`` — the ask names a count attached to a piece
  of information (#2331/#2373): whether the draw mints the count as its own parameter, folds
  it into the page parameter's description, or something else is measured
  (``_minted_families``), never asserted; what the case reads is the SPIRAL itself — token
  cost and the thinking trace re-checking the same rule — not a right-or-wrong answer.

What the cases deliberately do NOT claim, and why, is stated in each of them: the
genericity contract (that a routine's name says the KIND of task and never THIS occasion) is
the converse of a provenance claim and fits none of the design's three categories, so it is
read by a person on the modal sample; and the three obvious-looking claims about a usable
signature are ``_mints_a_usable_signature``'s, closed upstream.

Every case's input is the round's USER turns, rendered by the shipped
``build_framing_content`` — never hand-written — so the draw reads exactly what production
would render, and the cases are held against what they claim by deterministic drift probes
in ``make check`` (see ``tests/test_eval_harness.py``): each fixture pins its document
byte-for-byte as ``rendered_input``, and each case declares its five wordings beside the
facts every one of them carries (``PortedArms``).  A fixture that drifts from what it claims
is a case measuring nothing, and it must fail before any GPU time, not after.

The reference outputs above each fixture are agreed TARGETS read at joint review, never
strings a claim matches: a parameter the reference calls ``url`` may come back as
``page_to_watch`` and pass.

Since #1868 the draw happens when the machine ENTERS learn rather than at run end: the
round's identity is settled before the round runs, and run-end extraction READS that
framing instead of drawing again — a run-end draw survives only for a round nothing
framed.  Both entries render their document through the same shipped
``build_framing_content``, so the draw this case drives is the one either path makes.  The
draw also gives each parameter the VALUE the round demonstrated it with, and those values
are what a job's container is NAMED from.  That a value is a literal span of the user's own
words is the production validator's job (an accepted draw cannot carry a value nobody
said).

All content is synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    FIELD_UNSET,
    Consequence,
    Feature,
    SampleObservation,
    output_field,
)

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"


class FramingFixture(NamedTuple):
    """One agreed case: the user turns that are its whole input, the document the
    shipped renderer must produce from them, and the parameters the ask genuinely
    requires."""

    case_id: str
    turns: tuple[str, ...]
    rendered_input: str
    parameters: tuple[ParameterFamily, ...]


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


class PortedArms(NamedTuple):
    """One ported case's five wordings, together with the facts every one of them carries.

    The facts sit BESIDE the arms because they are the case's own claim about them: every
    claim the case makes hinges on what the ask REQUIRES, so an arm that stopped naming one
    of them would have its samples answering a different question under one case id.  A
    second copy of the list in the probe would drift from the arms exactly as silently as no
    probe at all, so the probe reads this.

    ``carries`` is a substring every arm must contain and ``never`` one no arm may.
    ``any_of`` is a fact stated as an ALTERNATION — every arm must carry one of its
    substrings, which is how a fact that survives being reworded ("let me know" / "tell me" /
    "ping me") is declared rather than left out.  Without it such a fact is invisible to the
    probe and can go missing from one arm silently, which is the drift the probe exists for.

    ``demonstrates`` is an alternation held against the arm's LAST turn rather than the whole
    document — the store verb, which every case states as a constant fact and every arm
    rewords (remember / save / keep / store).  It is per-turn because a whole-document check
    on the bare verbs is hollow here: a first turn opening "keep tabs on" or "keep an eye on"
    would satisfy `keep` while the demonstration said nothing about keeping anything.  The
    demonstration is the arm's last turn on every case, single-turn ones included.

    ``turns_per_arm`` is how many turns an arm takes, since an ask is however many turns the
    user needed for it and the single-turn case's whole subject is that number."""

    case_id: str
    arms: tuple[tuple[str, ...], ...]
    carries: tuple[str, ...]
    any_of: tuple[str, ...] = ()
    demonstrates: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    turns_per_arm: int = 2


# What a minted parameter reads as when it answers none of the case's families.
_OTHER_PIECE = "other"


def _minted_families(families: Sequence[ParameterFamily]) -> Feature:
    """Which piece of the ask each minted parameter answers, as one measured reading.

    Which parameters the framer mints, how many, and what each answers are model OUTPUT,
    so they are a Section B feature and never a pass/fail count.

    Reuses :func:`classify_by_family`, the one classification every suite that asks what a
    drawn parameter answers reads.  A parameter that answers none of the case's families
    reads ``other``: its value is still the user's own words (the framer's acceptance check
    refuses any draw where it is not), so it is another piece of the ask the model chose to
    lift into the interface — the thing to find, a label, a limit — and it shows up in the
    pooled distribution instead of silently not counting.  Sorted so two samples that
    answered the same families agree whatever order the draw minted them in.

    CONSEQUENTIAL: a routine that asks for one thing is a different interface from one that
    asks for two.

    Declares :data:`FIELD_UNSET` absent because a draw that minted NOTHING (the framer's
    honest failure, re-rolled and then refused rather than accepted) leaves no parameter
    for this axis to read."""

    def read(sample: SampleObservation) -> str:
        drawn = framed_parameters(sample)
        if not drawn:
            return FIELD_UNSET
        labels = (
            family_of.label if family_of is not None else _OTHER_PIECE
            for (_name, _says), family_of in zip(
                drawn, classify_by_family(drawn, families), strict=True
            )
        )
        return ", ".join(sorted(labels))

    return Feature("families answered", read, absent=FIELD_UNSET)


def _measure_the_draw(cohort: Cohort, positions: int, families: Sequence[ParameterFamily]) -> None:
    """The axes every ported framing case measures — the draw's own structured fields.

    **No claim is made in any of these cases (#2222, code owner's ruling)**: which
    parameters are minted, how many, and what they answer are model OUTPUT, so none of it
    is asserted — ``_minted_families`` (above) is where that question now lives,
    as a measured distribution rather than a count.  The name, the description and each
    drawn parameter's own name and line are COSMETIC — ``stock_tracker`` and
    ``share_price_watcher`` leave the same interface behind — and the naming spread is the
    framer's known system-level finding (0.90 on both measured models), so it belongs in
    the variance table and never as a fact about one sample.

    Only the positions the ask REQUIRES are measured: a position past them exists on a
    divergent sample alone, so its axis would read ``unset`` for the pack and score a
    disagreement as agreement.

    The name, the description and the raw count are emitted on EVERY draw, so none of them
    can come back ``unset`` and none declares an absent reading (#2073).  A per-parameter
    position CAN be omitted — a draw that folded a same-kind ask into one list emits no
    second position — and that omission IS the reading, so those axes declare
    :data:`FIELD_UNSET` and a cohort where nobody filled the position reads BLIND rather
    than as fifteen samples agreeing.

    No tool sequence and no reply spread: a single call makes neither."""
    cohort.measure(
        output_field(FRAME_PARAMETERS),
        _minted_families(families),
        output_field(FRAME_NAME, consequence=Consequence.COSMETIC),
        output_field(FRAME_DESCRIPTION, consequence=Consequence.COSMETIC),
        *[
            output_field(field(position), consequence=Consequence.COSMETIC, absent=FIELD_UNSET)
            for position in range(1, positions + 1)
            for field in (frame_parameter_name, frame_parameter_says)
        ],
    )


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
)


# ── The ported case: a page and no search, in five wordings ───────────────────
#
# The NEGATIVE half of the pair `framer-names-a-search-as-a-search` opens: a search-family
# parameter is the right answer there and the wrong one here, which is what makes them two
# cases rather than one behaviour under two sets of facts.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same address, asks
# whether the book is in stock, and carries a clause about being TOLD when it is.  What
# varies is only how a person says that.  The telling clause is load-bearing — it is the
# second thing the case claims is not a parameter — so it is declared as an alternation
# rather than left to be noticed, because it is the one fact here that survives being
# reworded into different words on every arm.
#
# No arm says "search" — the word is what the case is about, so an arm carrying it would be
# asking for the very thing the case claims is not asked for.
_PAGE_ONLY_PHRASINGS = (
    (
        "could you watch bookbarn.example/atlas-of-clouds and tell me when it's in stock again",
        "open bookbarn.example/atlas-of-clouds, see whether it's in stock, and remember that",
    ),
    (
        "i'd like bookbarn.example/atlas-of-clouds followed — let me know once it's back in stock",
        "read bookbarn.example/atlas-of-clouds, find out whether it's in stock, and save that",
    ),
    (
        "keep tabs on bookbarn.example/atlas-of-clouds for me — ping me when it's in stock",
        "visit bookbarn.example/atlas-of-clouds, check if it's in stock, and keep that",
    ),
    (
        "can you track bookbarn.example/atlas-of-clouds and say when it's back in stock?",
        "look at bookbarn.example/atlas-of-clouds, check whether it's in stock, and store that",
    ),
)

PAGE_ONLY_ARMS = PortedArms(
    case_id="framer-names-a-page-as-a-page-and-invents-no-search",
    arms=(_AVAILABILITY.turns, *_PAGE_ONLY_PHRASINGS),
    carries=("bookbarn.example/atlas-of-clouds", "stock"),
    any_of=("let me know", "tell me", "ping me", "say when"),
    never=("search", "query"),
    demonstrates=("remember", "save", "keep", "store"),
)

_PAGE_ONLY_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when an ask names a page "
    "and nothing to search for, Penny writes the routine's interface from the user's own "
    "words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_ask_mints_the_page_and_invents_nothing_beside_it(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: watch this address, tell me when the book is in stock.

    The address is the one piece a new occasion has to supply.  What the user came for — the
    availability — is what the routine IS, and being told when it changes is settled where
    the job is set running, so neither is a parameter.  Round 8's recorded failure is the
    third thing that is not one: a ``search_term`` promoted for an ask that named a page and
    no search, invented whole rather than taken from anything the user provided.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT — "the number of [parameters] and their naming is variance" — so none
    of it is asserted.  The provenance claim the ruling does want — that every minted VALUE
    traces to the user's own words — is already enforced by the framer's own acceptance
    check (`_mints_a_usable_signature` / `_values_were_demonstrated` in
    ``tools/micro_context.py``: an invented value is re-rolled, then refused), so asserting
    it again here would score 15/15 by construction and measure that validator rather than
    Penny.  And nothing here reads a store for the usual reason — a micro-context is one call
    that returns a typed result, so it moves no machine and writes to no store.

    What round 8's recorded failure is now caught by is the **measured** family-multiset
    (`_minted_families`, below): a `search_term` minted for this ask reads `other` in
    the pooled distribution instead of failing a count claim.
    """
    cohort = await framer_eval(
        case_id=PAGE_ONLY_ARMS.case_id,
        behaviour=_PAGE_ONLY_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_AVAILABILITY.turns,
        also_phrased=_PAGE_ONLY_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(
        cohort, positions=len(_AVAILABILITY.parameters), families=_AVAILABILITY.parameters
    )


# ── Case 2: cadence and notification are not signature ────────────────────────
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
)


# ── The ported case: the ticker ask, in five wordings ─────────────────────────
#
# The ticker ask is the plainest statement of the whole behaviour AND of its negative
# direction at once: what the user came for (the share price) bakes into the framing, the
# one thing that varies (the symbol) becomes the parameter, and "tell me when it moves" —
# which is settled where the job is set running — becomes neither.
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

TICKER_ARMS = PortedArms(
    case_id="framer-mints-only-the-piece-that-varies",
    arms=(_TICKER.turns, *_TICKER_PHRASINGS),
    carries=("VLT", "share price"),
    any_of=("moves", "changes", "shifts"),
    demonstrates=("remember", "save", "keep", "store"),
)

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_TICKER_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when a demonstrated "
    "round about one stock is turned into a reusable routine, Penny writes the routine's "
    "interface from the user's own words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_symbol_is_the_parameter_and_everything_else_bakes(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: track a stock, tell me when it moves, here is the symbol.

    The share price is what the routine IS, so it belongs in the framing; the symbol is the
    one thing said again next time; and being told when it moves is settled where the job is
    set running.  Whether the framer draws exactly that shape, or instead turns the
    notification into something to be re-supplied, is now MEASURED rather than asserted.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT — "the number of [parameters] and their naming is variance" — so none
    of it is asserted.  The provenance claim the ruling does want — that every minted VALUE
    traces to the user's own words — is already enforced by the framer's own acceptance
    check (`_mints_a_usable_signature` / `_values_were_demonstrated` in
    ``tools/micro_context.py``: an invented value is re-rolled, then refused), so asserting
    it again here would score 15/15 by construction and measure that validator rather than
    Penny.

    STORE stays empty for the usual reason too: a micro-context is one call that returns a
    typed result — it moves no machine and writes to no store.  And PROVENANCE's open
    fields — the routine's `name`, `description` and each parameter's line — are an
    identifier and two lines of deliberately GENERIC prose with no traceable value at all on
    a correct draw, which is why the suite's one instrument for the invented direction
    (`unsourced_specifics`) never transferred to them even when this case still asserted the
    parameter set.

    **And one real contract stays unmeasured, deliberately.**  That the routine's name and
    description say the KIND of task and never THIS occasion — a framing carrying ``VLT`` is
    a routine that can only ever run once — fits none of the design's three assertion
    categories and is not a feature distribution either: it stays what the design calls
    wrong-but-stable, read by a person opening the modal sample, where the drawn name renders
    verbatim.
    """
    cohort = await framer_eval(
        case_id=TICKER_ARMS.case_id,
        behaviour=_TICKER_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_TICKER.turns,
        also_phrased=_TICKER_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    # What is MEASURED — the draw's own structured fields, at the one position this ask
    # requires.
    _measure_the_draw(cohort, positions=len(_TICKER.parameters), families=_TICKER.parameters)


# ── Case 3: one turn is enough to frame ───────────────────────────────────────
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
)


# ── The ported case: the whole teach is one turn, in five wordings ────────────
#
# What makes this its own case is the DOCUMENT, not the facts.  Every other framing ask
# arrives as a standing want and then a demonstration, so what the routine is FOR and the
# piece that varies are said in two different turns.  Here they are one sentence, and the
# framer has to separate them out of it — a materially different decision from the same
# separation read off two turns.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same page, asks for
# today's high temperature, and says to remember it.  Every arm is ONE turn, which is the
# whole subject.
_SINGLE_TURN_PHRASINGS = (
    ("open weather.example/lisbon, get today's high temperature, and save it",),
    ("read weather.example/lisbon, find today's high temperature, and keep it",),
    ("check weather.example/lisbon for today's high temperature and store it",),
    ("visit weather.example/lisbon, look up today's high temperature, and remember it",),
)

SINGLE_TURN_ARMS = PortedArms(
    case_id="framer-frames-from-a-single-turn",
    arms=(_SINGLE_TURN.turns, *_SINGLE_TURN_PHRASINGS),
    carries=("weather.example/lisbon", "high temperature"),
    turns_per_arm=1,
    demonstrates=("remember", "save", "keep", "store"),
)

_SINGLE_TURN_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when the whole teach is a "
    "single turn, Penny writes the routine's interface from the user's own words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_single_turn_teach_still_separates_the_piece_that_varies(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings, each a single turn: read this page, find today's high, keep it.

    The ask at its minimum still says what the routine is for.  The temperature is the
    finding and bakes into the framing; the page is the one piece a new occasion has to
    supply.  Whether the framer separates them that way, asks what to find, or asks for
    nothing at all, is now MEASURED (`_minted_families`) rather than asserted by a count.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT, so none of it is asserted.  The provenance claim the ruling does want
    — that every minted VALUE traces to the user's own words — is already enforced by the
    framer's own acceptance check (`_mints_a_usable_signature` / `_values_were_demonstrated`
    in ``tools/micro_context.py``), so asserting it again here would score 15/15 by
    construction.  STORE stays empty for the usual reason: one call returns a typed result;
    it moves no machine and writes to no store.
    """
    cohort = await framer_eval(
        case_id=SINGLE_TURN_ARMS.case_id,
        behaviour=_SINGLE_TURN_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_SINGLE_TURN.turns,
        also_phrased=_SINGLE_TURN_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(
        cohort, positions=len(_SINGLE_TURN.parameters), families=_SINGLE_TURN.parameters
    )


# ── Case 4: the look-up is a search, not a page ───────────────────────────────
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
)


# ── The ported case: the look-up is a search, in five wordings ────────────────
#
# The POSITIVE half of the pair `framer-names-a-page-as-a-page-and-invents-no-search`
# closes: a search-family parameter is the right answer here and the wrong one there.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same event, says the
# look-up is a search, and asks for the cheapest ticket price.  No arm names an address — an
# arm that did would give the draw a page to mint instead, which is the other case's world
# and not this one.  What the probe can hold is the narrow half of that: no arm carries the
# reserved `.example` domain these fixtures write every address with, so an address written
# the way this suite writes one cannot get in unnoticed.
_SEARCH_PHRASINGS = (
    (
        "could you watch what tickets for aurora fest are going for?",
        "search for aurora fest tickets, get the cheapest ticket price, and save it",
    ),
    (
        "i'd like aurora fest ticket prices followed",
        "look up aurora fest tickets with a search, find the cheapest ticket price, and keep it",
    ),
    (
        "keep tabs on ticket prices for aurora fest",
        "run a search for aurora fest tickets, find the cheapest ticket price, and store it",
    ),
    (
        "can you track how much aurora fest tickets cost?",
        "search aurora fest tickets, find the cheapest ticket price, and remember it",
    ),
)

SEARCH_ARMS = PortedArms(
    case_id="framer-names-a-search-as-a-search",
    arms=(_SEARCH.turns, *_SEARCH_PHRASINGS),
    carries=("aurora fest", "search", "cheapest ticket price"),
    never=("example",),
    demonstrates=("remember", "save", "keep", "store"),
)

_SEARCH_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when an ask names "
    "something to search for and no page, Penny writes the routine's interface from the "
    "user's own words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_search_look_up_is_named_as_a_search(framer_eval: FramerEval, model: str) -> None:
    """One ask in five wordings: search for this event's tickets, keep the cheapest price.

    Nothing in this ask is an address.  What varies next time is which event is being
    searched for, so the parameter a correct draw mints is the search, not a page — and the
    cheapest price is what the routine IS and belongs in the framing.  Whether the framer
    draws that shape is now MEASURED (`_minted_families`) rather than asserted by a count.

    **This case pairs with ``framer-names-a-page-as-a-page-and-invents-no-search``**: the
    search-family parameter that is correct here is exactly the invention that case exists
    to watch for, so a framer cannot answer both by habit.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT, so none of it is asserted.  The provenance claim the ruling does want
    — that every minted VALUE traces to the user's own words — is already enforced by the
    framer's own acceptance check (`_mints_a_usable_signature` / `_values_were_demonstrated`
    in ``tools/micro_context.py``), so asserting it again here would score 15/15 by
    construction.  STORE stays empty for the usual reason: one call returns a typed result;
    it moves no machine and writes to no store.
    """
    cohort = await framer_eval(
        case_id=SEARCH_ARMS.case_id,
        behaviour=_SEARCH_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_SEARCH.turns,
        also_phrased=_SEARCH_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_SEARCH.parameters), families=_SEARCH.parameters)


# ── Case 5: two re-suppliable pieces of DIFFERENT types ───────────────────────
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
)


# ── The ported case: a place to go and a thing to look for, in five wordings ──
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same catalog page,
# the same book, and asks whether it is available.  What varies is only how a person says
# that.  No arm calls the look-up a search — the thing being looked for is found ON the
# page, and an arm that framed it as a search would be asking for a different piece.
_PAGE_AND_TITLE_PHRASINGS = (
    (
        "could you keep an eye on the library catalog for a book i want?",
        "go to town-library.example/catalog, look up The Glass Harbour, and remember whether "
        "it's available",
    ),
    (
        "i'd like the library catalog watched for a book i'm after",
        "check town-library.example/catalog for The Glass Harbour and save whether it's available",
    ),
    (
        "can you track a book in the library catalog for me?",
        "visit town-library.example/catalog, find The Glass Harbour there, and keep whether "
        "it's available",
    ),
    (
        "keep tabs on the library catalog for a book i'm waiting on",
        "read town-library.example/catalog, look for The Glass Harbour, and store whether "
        "it's available",
    ),
)

PAGE_AND_TITLE_ARMS = PortedArms(
    case_id="framer-mints-both-pieces-when-they-are-different-kinds",
    arms=(_PAGE_AND_TITLE.turns, *_PAGE_AND_TITLE_PHRASINGS),
    carries=("town-library.example/catalog", "The Glass Harbour", "available"),
    never=("search",),
    demonstrates=("remember", "save", "keep", "store"),
)

_PAGE_AND_TITLE_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when an ask names a page "
    "and a thing to look for on it, Penny writes the routine's interface from the user's own "
    "words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_place_and_a_thing_to_look_for_are_two_parameters(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: watch this catalog for this book, keep whether it's in.

    Availability is what the routine IS and bakes into the framing, exactly as it does when
    the ask names one page.  What differs is that the thing being looked for is not IN the
    address — it is looked up ON the page — so a correct draw mints two re-suppliable
    pieces of different kinds rather than the page alone.  Whether the framer draws that
    shape is now MEASURED (`_minted_families`) rather than asserted by a count.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT, so none of it is asserted.  The provenance claim the ruling does want
    — that every minted VALUE traces to the user's own words — is already enforced by the
    framer's own acceptance check (`_mints_a_usable_signature` / `_values_were_demonstrated`
    in ``tools/micro_context.py``), so asserting it again here would score 15/15 by
    construction.  STORE stays empty for the usual reason: one call returns a typed result;
    it moves no machine and writes to no store.
    """
    cohort = await framer_eval(
        case_id=PAGE_AND_TITLE_ARMS.case_id,
        behaviour=_PAGE_AND_TITLE_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_PAGE_AND_TITLE.turns,
        also_phrased=_PAGE_AND_TITLE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(
        cohort, positions=len(_PAGE_AND_TITLE.parameters), families=_PAGE_AND_TITLE.parameters
    )


# ── The ported case: three of a kind, in five wordings ────────────────────────
#
# The suite's one same-kind ask.  The world is URLS because that is where the fold into a
# list was MEASURED: on run 1 every sample asked to read two front pages drew one
# `sites — list of URLs`, folding the pieces the user named into a single list parameter.
# It is put at THREE because the list is more tempting there than at two — three scalars is
# where a description that promises "the news sites" starts to look like the tidier
# interface — and a framer that keeps three apart keeps two apart.

# The ONE family a same-kind ask is answered by, at whatever multiplicity the ask names.
# Grounded: every arm says "read <address>, <address> and <address>", so `url`/`site`/`page`
# are words the ask itself puts in front of the draw, and a parameter named at that breadth
# is a piece the routine can be pointed at.  How many times it is minted, and how the draw
# tells them apart, is now measured (`_minted_families`) rather than asserted.
_SAME_KIND_SOURCE = _page_family("source")

# THE FACTS ARE CONSTANT across the five wordings: every arm names all three front pages and
# asks for each site's top headline.
_THREE_SOURCES_TURNS = (
    "hey could you keep an eye on the morning headlines for me",
    "read citydesk.example/front, harborpost.example/front and riverchronicle.example/front, "
    "and remember each site's top headline",
)

_THREE_SOURCES_PHRASINGS = (
    (
        "could you watch the morning news for me",
        "open citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, find each site's top headline, and remember them",
    ),
    (
        "i'd like the morning headlines followed",
        "check citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, and keep each one's top headline",
    ),
    (
        "can you keep track of the morning headlines?",
        "look at citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, and save the top headline from each",
    ),
    (
        "keep up with the morning headlines for me",
        "go to citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, get each site's top headline, and store them",
    ),
)

THREE_SOURCES_ARMS = PortedArms(
    case_id="framer-keeps-three-of-a-kind-as-three-parameters",
    arms=(_THREE_SOURCES_TURNS, *_THREE_SOURCES_PHRASINGS),
    carries=(
        "citydesk.example/front",
        "harborpost.example/front",
        "riverchronicle.example/front",
        "top headline",
    ),
    demonstrates=("remember", "save", "keep", "store"),
)

_THREE_SOURCES_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when one ask points a "
    "routine at three things of the same kind, Penny writes the routine's interface from "
    "the user's own words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_three_of_a_kind_stay_three_distinct_parameters(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: read these three front pages, keep each one's top headline.

    The same-kind behaviour at three.  Every page has to be re-suppliable and they have to be
    tellable apart, so a correct draw mints three scalars distinguished by position; a single
    list parameter, or two scalars and a description that promises the rest, has lost a piece
    the user named.  Whether the framer draws three, folds them into a list, or something
    else is now MEASURED (`_minted_families`) rather than asserted by a count.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT, so none of it is asserted.  The provenance claim the ruling does want
    — that every minted VALUE traces to the user's own words — is already enforced by the
    framer's own acceptance check (`_mints_a_usable_signature` / `_values_were_demonstrated`
    in ``tools/micro_context.py``), so asserting it again here would score 15/15 by
    construction.  STORE stays empty for the usual reason: one call returns a typed result;
    it moves no machine and writes to no store.
    """
    cohort = await framer_eval(
        case_id=THREE_SOURCES_ARMS.case_id,
        behaviour=_THREE_SOURCES_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_THREE_SOURCES_TURNS,
        also_phrased=_THREE_SOURCES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=3, families=(_SAME_KIND_SOURCE,))


# ── The ported case: an ask naming a count, in five wordings (#2331/#2373) ────
#
# The code owner's traced cause: a framing ask that names a count ("the top 5 headlines")
# sends the draw back over its own rules again and again — whether the count is a piece
# the user provided (and so a parameter) or is "carried" by a description that names the
# same kind of thing.  No existing case puts a NUMBER in the ask: the three-of-a-kind case
# above makes multiplicity salient but never states a count word the model has to decide is
# a parameter or isn't.  This case is the structural analog of the issue's own worked
# example, built fresh (no recorded promptlog carries it yet) rather than replayed.
#
# The ONE family a stated count is answered by, whatever the draw calls it or however it
# spells the number.  Not ``name_only``: a count folded into the page parameter's
# description (rather than minted as its own parameter) is exactly the alternative this
# case measures, so the description fallback has to stay open.
_COUNT_FAMILY = ParameterFamily(
    "count", ("count", "top", "limit", "number", "n", "how_many", "quantity", "amount")
)

# THE FACTS ARE CONSTANT across the five wordings: every arm names the same front page, asks
# for the top 5 headlines, and says to keep them.
_COUNT_TURNS = (
    "could you keep an eye on citydesk.example/front and send me the top 5 headlines every morning",
    "go to citydesk.example/front, pull the top 5 headlines, and remember them",
)

_COUNT_PHRASINGS = (
    (
        "i'd like citydesk.example/front watched — give me the top 5 headlines each morning",
        "open citydesk.example/front, grab the top 5 headlines, and save them",
    ),
    (
        "can you track citydesk.example/front and show me its top 5 headlines every day?",
        "visit citydesk.example/front, get the top 5 headlines, and keep them",
    ),
    (
        "keep tabs on citydesk.example/front for me and list the top 5 headlines daily",
        "check citydesk.example/front, find the top 5 headlines, and store them",
    ),
    (
        "watch citydesk.example/front and let me know the top 5 headlines every morning",
        "read citydesk.example/front, note the top 5 headlines, and remember them",
    ),
)

COUNT_ARMS = PortedArms(
    case_id="framer-settles-an-ask-that-names-a-count",
    arms=(_COUNT_TURNS, *_COUNT_PHRASINGS),
    carries=("citydesk.example/front", "top 5", "headlines"),
    demonstrates=("remember", "save", "keep", "store"),
)

_COUNT_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when the ask names a "
    "count attached to a piece of information, Penny writes the routine's interface from "
    "the user's own words."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_named_count_is_framed_without_a_spiral(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: watch this page, keep the top 5 headlines every morning.

    The traced cause of #2331: a count word in the ask ("the top 5 ...") pulls the draw
    back over "a piece the name and description already carry is not a parameter" again and
    again, because the routine's own generic description necessarily names the KIND of
    thing the count modifies.  This is the "model gets lost" shape
    (``docs/eval-working-guide.md`` §5): the end state is still read on the usual three
    claims below, but the measurement that matters here is the SPIRAL itself — token cost
    (duration, output and reasoning tokens) and the thinking trace, read off the run, not a
    claim this file states.

    **No claim is made, and that is the correct report (#2222,
    code owner's ruling)**: which parameters are minted, how many, and what they answer are
    all model OUTPUT, so none of it is asserted.  The provenance claim the ruling does want
    — that every minted VALUE traces to the user's own words — is already enforced by the
    framer's own acceptance check (`_mints_a_usable_signature` / `_values_were_demonstrated`
    in ``tools/micro_context.py``), so asserting it again here would score 15/15 by
    construction.  STORE stays empty for the usual reason: one call returns a typed result;
    it moves no machine and writes to no store.
    """
    cohort = await framer_eval(
        case_id=COUNT_ARMS.case_id,
        behaviour=_COUNT_BEHAVIOUR,
        area=Area.TEACHING_A_ROUTINE,
        model=model,
        turns=_COUNT_TURNS,
        also_phrased=_COUNT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=2, families=(_page_family("url"), _COUNT_FAMILY))


# EVERY ported case's arms, for the deterministic probe in ``make check`` — one place, so the
# probe and the live runs can never be checking two different sets of wordings.
PORTED_ARMS = (
    TICKER_ARMS,
    PAGE_ONLY_ARMS,
    SINGLE_TURN_ARMS,
    SEARCH_ARMS,
    PAGE_AND_TITLE_ARMS,
    THREE_SOURCES_ARMS,
    COUNT_ARMS,
)


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (
    _AVAILABILITY,
    _TICKER,
    _SINGLE_TURN,
    _SEARCH,
    _PAGE_AND_TITLE,
)
