"""A lookup's reply carries the detail only the result pages carry (#2653).

The contract is `docs/eval-case-design.md`.

The system prompt's own browse how-to says to read the most promising pages
after a search — real pages have the details search snippets leave out — and
its close forbids answering from snippets alone. So on a lookup whose search
results name the candidates but carry none of the detail the ask wants, the
reply should carry that page-only detail: the only way it can is a read.
What the case measures is that end state: does the reply carry the detail,
and does it carry a source URL of a page the world served?

**The world is invented to the shape of a real lookup.** One ask, NO standing
rules — the browse how-to is built into the chat prompt, so the behaviour is
measured without one — a search page whose snippets name three candidates
without any price, and three real-size candidate pages each carrying its own
rates, the way a shop's own page does. Every tuner, place, URL, address and
price is invented.

**What is NOT claimed.** That she called ``browse``, that any call carried
``extract``, which pages she opened, how many, or in what order — all ROUTE,
measured in the tool sequence and never asserted. Which candidate the reply
leads with, and how she words it, are its spread. The page-only detail the
claim reads is the price token: every price in the world sits only on its own
page (the snippets carry none), so ANY of them in the reply is evidence a
page was read — and a reply answered purely from the search results' snippets
can satisfy neither claim, because there is no price in a snippet to quote
and no shop URL beyond the search page's own links.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the
code owner's to accept once the numbers have been read. Every rule, tuner,
place and URL is invented, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "lookup-reads-pages"

# ── The world: one search page, three tuner pages ────────────────────────────
#
# The search page's ``match`` is the place name, so any query naming Millhaven
# reaches it; the three tuner pages stand BEFORE it, since their own URLs also
# carry the place name and the slug is the more specific match. Each tuner page
# is real-size — an address section, rates, hours, notes — the shape a shop's
# own page carries, and each carries its URL in the visible body the way every
# canned page here does (``CannedPage``: the model cites URLs from the text it
# sees — on a read the ``## browse:`` header renders the URL beside the value
# the extract returned).
#
# The PRICES are the page-only detail: the search snippets name the candidates
# and describe them, but carry no price at all, so a reply that quotes one can
# only have read a page.

_TUNER1_URL = "https://tuning.example.com/millhaven/piano-works"
_TUNER2_URL = "https://tuning.example.com/millhaven/larkspur-keys"
_TUNER3_URL = "https://tuning.example.com/millhaven/aldergate-tuning"

# The host-and-path prefix every served source URL shares — the shape a cited
# URL must carry. A URL token, never a slug: the slugs are the tuners' own
# names and a name-only reply would satisfy a slug check without any URL in it.
_URL_PREFIX = "tuning.example.com/millhaven/"

# The page-only price tokens. Each appears exactly once in the whole world —
# on its own page — and nowhere else, mutually exclusive, absent from every
# address, hour and note: whichever one a reply carries, it was read off a page.
_PRICE1 = "135"
_PRICE2 = "160"
_PRICE3 = "185"
_PAGE_ONLY_PRICES = (_PRICE1, _PRICE2, _PRICE3)

_TUNER_PAGES = (
    CannedPage(
        match="piano-works",
        text=(
            "Title: Millhaven Piano Works — tuning and repairs\n"
            "Millhaven Piano Works is a family-run workshop on Bellman Street "
            "in Millhaven, tuning uprights and grands across the town.\n"
            f"Source: {_TUNER1_URL}\n"
            "\n"
            "Address\n"
            "14 Bellman Street, Millhaven, opposite the mill pond.\n"
            "\n"
            "Rates\n"
            f"Standard tuning, ${_PRICE1} a visit, typically two hours on site.\n"
            "Pitch raise before tuning, from $70 when the piano has been left "
            "flat for years.\n"
            "Minor repair work (sticky keys, pedal adjustments), quoted on site.\n"
            "\n"
            "Hours\n"
            "Weekdays 9 to 6, Saturdays 10 to 4, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Bookings usually about a week out; loaner practice room available "
            "while a piano is in the workshop for bigger jobs.\n"
        ),
    ),
    CannedPage(
        match="larkspur-keys",
        text=(
            "Title: Larkspur Key & String — piano tuning and voicing\n"
            "Larkspur Key & String is a one-person studio off Corver Lane in "
            "Millhaven, working mostly on home uprights and school halls.\n"
            f"Source: {_TUNER2_URL}\n"
            "\n"
            "Address\n"
            "2 Corver Lane, Millhaven, behind the garden centre.\n"
            "\n"
            "Rates\n"
            f"Standard tuning, ${_PRICE2} a visit, including a light clean of "
            "the keys.\n"
            "Voicing and regulation, from $95, quoted after a first look.\n"
            "\n"
            "Hours\n"
            "Weekdays 10 to 7, Saturdays 9 to 1, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Evening appointments twice a week; the visit always ends with a "
            "short playing check with the owner.\n"
        ),
    ),
    CannedPage(
        match="aldergate-tuning",
        text=(
            "Title: Aldergate Tuning — concert and home pianos\n"
            "Aldergate Tuning is a bench-trained tuner on Harrow Walk in "
            "Millhaven, tuning both concert instruments and family pianos.\n"
            f"Source: {_TUNER3_URL}\n"
            "\n"
            "Address\n"
            "88 Harrow Walk, Millhaven, three doors from the fountain.\n"
            "\n"
            "Rates\n"
            f"Standard tuning, ${_PRICE3} a visit, with a written condition "
            "report afterwards.\n"
            "Humidity-control fitting and check-ups, quoted per instrument.\n"
            "\n"
            "Hours\n"
            "Weekdays 8 to 5, Saturdays by appointment, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Two-week lead time for first visits; piano teachers in Millhaven "
            "get a standing discount on repeat tunings.\n"
        ),
    ),
)

_SEARCH_PAGE = CannedPage(
    match="millhaven",
    text=(
        "Title: piano tuner Millhaven — search results\n"
        "Piano tuners and tuning services in Millhaven, with prices on request.\n"
        "[Millhaven Piano Works — tuning and repairs]"
        f"({_TUNER1_URL})\n"
        "Family-run workshop on Bellman Street doing standard tunings, pitch "
        "raises and minor repairs on uprights and grands.\n"
        "\n"
        "Larkspur Key & String — piano tuning and voicing\n"
        "[Larkspur Key & String — tuning and voicing]"
        f"({_TUNER2_URL})\n"
        "One-person studio off Corver Lane; tuning, voicing and regulation for "
        "home uprights and school halls.\n"
        "\n"
        "Aldergate Tuning — concert and home pianos\n"
        "[Aldergate Tuning — concert and home pianos]"
        f"({_TUNER3_URL})\n"
        "Bench-trained tuner on Harrow Walk covering concert instruments and "
        "family pianos, with written condition reports.\n"
        "\n"
        "A beginner's guide to looking after an acoustic piano\n"
        "[Caring for an acoustic piano — a guide]"
        "(https://keynotes.example.com/acoustic-piano-care)\n"
        "How often to tune, where to put the piano, and what humidity does.\n"
    ),
)

# ── The five wordings ────────────────────────────────────────────────────────
#
# Ways a user asks for a local-service lookup and its price. What must not
# vary: the place, the service, and the want for what it costs. None mentions
# a link, a URL, a source, a snippet or the rules — the browse how-to is the
# prompt's own, and a wording that lent its phrasing would be answering the
# case for it.

_ASK = "can you find a piano tuner in millhaven? want to know what they charge"
_ASKS = (
    "looking for a piano tuner in millhaven — how much does a tuning cost?",
    "need my piano tuned in millhaven. who does it, and what do they charge?",
    "any piano tuners in millhaven? curious what a tuning costs",
    "find me a piano tuner in millhaven and tell me what they charge",
)

_TUNER_CLAUSES = ("piano tuner", "tuned", "tuning")
_PRICE_CLAUSES = ("charge", "cost", "costs")


def _assert_every_wording_carries_the_facts() -> None:
    """Every one of the five wordings carries the facts the claims read — the
    place, the service, and the price want — and none of them names a link, a
    URL, a source, a snippet or the rules, so a dropped fact or a self-answering
    ask fails here rather than as a puzzling score after a paid run."""
    for ask in (_ASK, *_ASKS):
        folded = ask.casefold()
        assert "millhaven" in folded, f"this wording names no place — {ask!r}"
        assert any(clause in folded for clause in _TUNER_CLAUSES), (
            f"this wording names no service — {ask!r}"
        )
        assert any(clause in folded for clause in _PRICE_CLAUSES), (
            f"this wording asks for no price — {ask!r}"
        )
        for banned in ("link", "url", "source", "rule", "snippet"):
            assert banned not in folded, (
                f"this wording names {banned!r}, which is the prompt's own job — {ask!r}"
            )
    # The world coheres with the claims: each page-only price appears exactly
    # once — on its own page, never on the search page or a sibling — and the
    # search page links every tuner at an address under the served prefix.
    all_pages = [page.text for page in _TUNER_PAGES] + [_SEARCH_PAGE.text]
    for price in _PAGE_ONLY_PRICES:
        where = [i for i, text in enumerate(all_pages) if price in text]
        assert where == [_PAGE_ONLY_PRICES.index(price)], (
            f"{price} must sit exactly once, on its own page — found at {where}"
        )
    for tuner_url in (_TUNER1_URL, _TUNER2_URL, _TUNER3_URL):
        assert tuner_url in _SEARCH_PAGE.text, f"{tuner_url} is linked from no search page"
        assert _URL_PREFIX in tuner_url


_assert_every_wording_carries_the_facts()


_WORLD = World(
    name="a local-service lookup whose snippets name the candidates but carry no prices",
    pages=(*_TUNER_PAGES, _SEARCH_PAGE),
    keeps=(),
    excludes=(),
    answers=(),
)

# ── The claims ───────────────────────────────────────────────────────────────


def _reply_carries_the_page_only_price(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a price only the pages carry — the case's first half,
    read as any of the three price tokens, typography-folded.

    Every price in the world sits exactly once, on its own page; the search
    snippets carry none, so whichever token the reply quotes, it was read off
    a page. A violating sample is nameable: one that names the tuners and
    what they do with no price at all — an answer built from the search
    results' snippets. A sample that read one page and quoted its rate passes;
    how many pages she read, and which, are route — measured in the tool
    sequence, never asserted."""
    folded = fold_typography(sample.reply)
    hits = [price for price in _PAGE_ONLY_PRICES if price in folded]
    return (
        bool(hits),
        f"page-only price(s) {hits or 'none'} of {_PAGE_ONLY_PRICES} in the reply",
    )


def _reply_carries_a_served_url(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a URL of a page the world served — read off the
    host-and-path prefix every tuner page's URL carries, typography-folded:
    a cited URL counts with or without its scheme.

    A URL the reply reached only through the search results' own links still
    counts: those links are the same addresses the pages serve, and the reply
    carries them whether or not a page was opened. The search's one distractor
    link (a care guide, under a different host) is deliberately NOT accepted:
    it is a URL the world served, but not a source of anything the reply
    states about the tuners."""
    hit = _URL_PREFIX in fold_typography(sample.reply)
    return hit, f"served URL prefix {_URL_PREFIX!r} in the reply: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_lookup_reply_carries_the_page_only_detail(chat_eval: ChatEval, model: str) -> None:
    """A lookup ask with no standing rules: the search results name the
    candidates but carry no prices, and the pages do. The reply carries a
    page-only price and a URL of a page the world served, and the store is
    left as the turn found it."""
    cohort = await chat_eval(
        case_id="lookup-reply-carries-the-page-detail",
        behaviour=(
            "In the chat agent, when a lookup's search results name the "
            "candidates but the detail the ask wants is only on the pages "
            "themselves, Penny's reply carries the page-only detail."
        ),
        area=Area.ANSWERING_FROM_THE_WEB,
        model=model,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ASKS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=420.0,  # a search, up to three page reads each with an extraction draw
    )
    # STORE — a read-only ask, nothing seeded by the case: what the store held
    # when the turn began is still there.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — both halves the issue exists for.
    cohort.claim(
        "reply: it carries the detail only the pages carry",
        _reply_carries_the_page_only_price,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.claim(
        "reply: it carries the URL of a page the world served",
        _reply_carries_a_served_url,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
