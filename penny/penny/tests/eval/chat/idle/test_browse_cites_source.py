"""A browse reply cites the URL of the page it was built on (#2648).

The contract is `docs/eval-case-design.md`.

A standing rule asks for the specific details and at least one source URL with
every search, and the answer is built on pages read through the browser — the
search first, then the shop pages their results point at, read (typically through
the ``extract`` micro-context) for names, addresses and services. What the case
measures is the rule's second half: does the reply carry a URL of a page the
world served?

**The shape is production's #2648 turn, genericised.** The real turn (run
``9a1d0f57…``, promptlog ids 66-67) searched for downtown watch repairers, then
opened three shop pages with an ``extract`` instruction about names, addresses
and services — nothing about links — and was cut short by a restart before it
replied. Every chat turn that DID reply with an ``extract`` that didn't ask for
links was search-only, so the verbatim clone is unreachable and this world is
invented to the same shape: the two standing rules the real turn carried, a
search page linking three shops plus one distractor, and three real-size shop
pages. Every shop, place, URL and price is invented.

**What is NOT claimed.** That she called ``browse``, that any call carried
``extract``, which pages she opened or in what order — all ROUTE, measured in
the tool sequence and never asserted. How she words the reply is its spread.
The URL the reply carries may come from a page she read (the ``## browse:``
header renders it to her on every read) or, on a search-only sample, from the
search results' own links — the claim reads a URL is present; which hop
delivered it is visible in the artifacts, not asserted here.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the
code owner's to accept once the numbers have been read. Every rule, shop, place
and URL is invented, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.agents.self_state import standing_rules_section
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import EntryInput
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
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

_FAMILY = "browse-source-url"

# The collection every rule lives in — read off the constant, never restated.
RULES = PennyConstants.MEMORY_RULES_COLLECTION

# The two standing rules the production turn carried, worded as the user worded
# them (genericised wording only where a real value appeared — neither rule
# names one). Seeded through the store's own write path, author ``user``.
_SOURCE_URL_RULE = (
    "include_search_links",
    "Whenever I ask you to search for things, always include the specific "
    "details and at least one source URL so I can dig deeper.",
)
_READ_FIRST_RULE = (
    "search_and_read_first",
    "When asked to look for something, perform a search first, pick promising "
    "results, and then browse them directly to read and extract information "
    "before responding.",
)
_SEEDED_RULES = (_SOURCE_URL_RULE, _READ_FIRST_RULE)


def _seed_rules(db: Database) -> None:
    """Lay the case's rules down through the store's own write path — a rule is
    an entry like any other (``test_standing_rules.py``'s seeding, two rules)."""
    require_memory(db, RULES).write(
        [EntryInput(key=key, content=content) for key, content in _SEEDED_RULES],
        author="user",
        run_id=seeded_run_id(RULES),
    )


def _premise(penny: Penny) -> None:
    """The case's premise, asserted inside the sample: the standing-rules render
    shows both rules — the state the reply is answered against. Read through the
    header's own render so the probe and the surface cannot drift into asserting
    different words for one state; a seeded rule renders as its own sentence —
    ``- <content>``, no key, no stamp (#2647)."""
    rendered = standing_rules_section(penny.db)
    lines = rendered.splitlines()[1:]  # drop the section title
    assert len(lines) == len(_SEEDED_RULES), (
        f"the section must render every seeded rule:\n{rendered!r}"
    )
    for line, (_key, content) in zip(lines, _SEEDED_RULES, strict=True):
        assert line == f"- {content}", f"a seeded rule must render whole: {line!r}"


# ── The world: one search page, three shop pages ─────────────────────────────
#
# The search page's ``match`` is the place name, so any query naming Riverford
# reaches it; the three shop pages stand BEFORE it, since their own URLs also
# carry the place name and the slug is the more specific match. Each shop page
# is real-size — an address section, services with prices, hours — the shape a
# shop's own page carries, and each carries its URL in the visible body the way
# every canned page here does (``CannedPage``: the model cites URLs from the
# text it sees — on a read WITHOUT ``extract`` the body reaches her, and on one
# WITH it the ``## browse:`` header renders the same URL).

_SHOP1_URL = "https://cobblers.example.com/riverford/cobble-bell"
_SHOP2_URL = "https://cobblers.example.com/riverford/sole-bench"
_SHOP3_URL = "https://cobblers.example.com/riverford/heelstone"

# The host-and-path prefix every served source URL shares — the shape a cited
# URL must carry. A URL token, never a slug: the slugs are the shops' own names
# and a name-only reply would satisfy a slug check without any URL in it.
_URL_PREFIX = "cobblers.example.com/riverford/"

_SHOP_PAGES = (
    CannedPage(
        match="cobble-bell",
        text=(
            "Title: Cobble & Bell — shoe repair, Riverford\n"
            "Cobble & Bell is a family-run cobbler on Merchant Row in Riverford.\n"
            f"Source: {_SHOP1_URL}\n"
            "\n"
            "Address\n"
            "41 Merchant Row, Riverford, by the market square.\n"
            "\n"
            "Services and prices\n"
            "Full resoling in leather or rubber, from $45 a pair.\n"
            "Heel rebuilds and replacements, from $18 a heel.\n"
            "Hand stitching and patch repairs, from $25.\n"
            "\n"
            "Hours\n"
            "Weekdays 8 to 6, Saturdays 9 to 2, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Same-week turnaround on most resoling jobs; the quote is given on drop-off.\n"
        ),
    ),
    CannedPage(
        match="sole-bench",
        text=(
            "Title: The Sole Bench — repairs and resoling, Riverford\n"
            "The Sole Bench is a workshop off Fenwick Lane in Riverford, taking "
            "walk-ins all week.\n"
            f"Source: {_SHOP2_URL}\n"
            "\n"
            "Address\n"
            "7 Fenwick Lane, Riverford, behind the old post office.\n"
            "\n"
            "Services and prices\n"
            "Resoling in leather, rubber or wedge soles, from $50 a pair.\n"
            "Stretching, fitting and orthotic insoles, from $15.\n"
            "Zip and strap repairs on boots and bags, from $12.\n"
            "\n"
            "Hours\n"
            "Weekdays 9 to 7, Saturdays 10 to 4, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Three-day turnaround on resoling; drop-in measuring every afternoon.\n"
        ),
    ),
    CannedPage(
        match="heelstone",
        text=(
            "Title: Heelstone Bootmakers — resoling and repairs, Riverford\n"
            "Heelstone Bootmakers is a bench-made boot shop on Argyle Steps, "
            "Riverford, working mostly in leather-soled footwear.\n"
            f"Source: {_SHOP3_URL}\n"
            "\n"
            "Address\n"
            "12 Argyle Steps, Riverford, two doors from the bridge stairs.\n"
            "\n"
            "Services and prices\n"
            "Bench resoling with hand-stitched soles, from $70 a pair.\n"
            "Re-heeling and shank repairs, from $30.\n"
            "Full restoration of vintage boots, quoted per pair.\n"
            "\n"
            "Hours\n"
            "Weekdays 10 to 5, Saturdays by appointment, closed Sundays.\n"
            "\n"
            "Notes\n"
            "A two-week resoling turnaround; appointments made in the shop or by phone.\n"
        ),
    ),
)

_SEARCH_PAGE = CannedPage(
    match="riverford",
    text=(
        "Title: shoe repair and cobblers in Riverford — search results\n"
        "Cobblers and shoe repair shops in Riverford, with resoling services.\n"
        "[Cobble & Bell — shoe repair and resoling]"
        f"({_SHOP1_URL})\n"
        "Family-run cobbler on Merchant Row doing full resoling, heel rebuilds "
        "and hand stitching.\n"
        "\n"
        "The Sole Bench — repairs, resoling and fittings\n"
        "[The Sole Bench — repairs and resoling]"
        f"({_SHOP2_URL})\n"
        "Workshop off Fenwick Lane; resoling in leather and rubber, plus stretch "
        "and fit services.\n"
        "\n"
        "Heelstone Bootmakers — handmade repairs and resoling\n"
        "[Heelstone Bootmakers — resoling and repairs]"
        f"({_SHOP3_URL})\n"
        "Bench-made boot repairs on Argyle Steps, with a two-week resoling "
        "turnaround.\n"
        "\n"
        "A guide to caring for leather-soled shoes, with product links\n"
        "[Caring for leather soles — a guide]"
        "(https://stylenotes.example.com/leather-sole-care)\n"
        "Six tips for making resoled shoes last, with a short list of conditioners.\n"
    ),
)

# ── The five wordings ────────────────────────────────────────────────────────
#
# Ways a user asks for a local-service search and its details. What must not
# vary: the place, the service, and the want for names, addresses and what each
# offers. None mentions a link, a URL, a source or the rules — the rule the
# reply is measured against is the collection's, never the ask's.

_ASK = (
    "can you find me a cobbler in riverford who does resoling? i want names, "
    "addresses and what services they offer"
)
_ASKS = (
    "looking for a cobbler in riverford that does resoling — who's good, where "
    "are they, and what do they do?",
    "find me somewhere in riverford to get my boots resoled — shop names, "
    "addresses and what they offer",
    "i need resoling done in riverford. can you look up who does it, where they "
    "are, and what services they run?",
    "any cobblers in riverford doing resoling? want the names, the addresses "
    "and what each one offers",
)

_NAME_CLAUSES = ("names", "who")
_ADDRESS_CLAUSES = ("addresses", "where")
_OFFER_CLAUSES = (
    "what services",
    "what they do",
    "what do they do",
    "what they offer",
    "services they run",
    "what each one offers",
)


def _assert_every_wording_carries_the_facts() -> None:
    """Every one of the five wordings carries the facts the claims read — the
    place, the service, and the details want — and none of them names a link,
    a URL, a source or the rules, so a dropped fact or a self-answering ask
    fails here rather than as a puzzling score after a paid run."""
    for ask in (_ASK, *_ASKS):
        folded = ask.casefold()
        assert "riverford" in folded, f"this wording names no place — {ask!r}"
        assert "resol" in folded, f"this wording names no service — {ask!r}"
        assert any(clause in folded for clause in _NAME_CLAUSES), (
            f"this wording asks for no names — {ask!r}"
        )
        assert any(clause in folded for clause in _ADDRESS_CLAUSES), (
            f"this wording asks for no addresses — {ask!r}"
        )
        assert any(clause in folded for clause in _OFFER_CLAUSES), (
            f"this wording asks for no services — {ask!r}"
        )
        for banned in ("link", "url", "source", "rule"):
            assert banned not in ask.casefold(), (
                f"this wording names {banned!r}, which is the rule's job — {ask!r}"
            )
    # The world coheres with the claims: every shop page's body carries the URL
    # prefix the reply-side claim reads, and the search page links each shop at
    # an address under it.
    for page in _SHOP_PAGES:
        assert _URL_PREFIX in page.text.casefold(), f"{page.match} carries no {_URL_PREFIX}"
    for shop_url in (_SHOP1_URL, _SHOP2_URL, _SHOP3_URL):
        assert shop_url in _SEARCH_PAGE.text, f"{shop_url} is linked from no search page"


_assert_every_wording_carries_the_facts()


_WORLD = World(
    name="a local-service search with three shop pages",
    pages=(*_SHOP_PAGES, _SEARCH_PAGE),
    keeps=(),
    excludes=(),
    answers=(),
)

# ── The claims ───────────────────────────────────────────────────────────────


def _reply_carries_a_served_url(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a URL this world served — the source-URL rule's own
    second half, read off the host-and-path prefix every shop page's URL
    carries. Typography-folded containment: a cited URL counts with or without
    its scheme.

    A violating sample is nameable: one that names shops and addresses with no
    URL at all. A sample that reached the URL only through the search results'
    own links passes — the rule asks for at least one source URL, and those
    links are the same addresses; whether the URL came off a page she read or
    off the search results is a route, visible in the artifacts, not asserted
    here. The search's one distractor link (a style blog, under a different
    host) is deliberately NOT accepted: it is a URL the world served, but not a
    source of anything the reply states, so a reply citing it alone would carry
    a URL and none of the answer's provenance."""
    hit = _URL_PREFIX in fold_typography(sample.reply)
    return hit, f"served URL prefix {_URL_PREFIX!r} in the reply: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_browse_reply_carries_a_served_url(chat_eval: ChatEval, model: str) -> None:
    """Two standing rules — cite a source URL, search then read — and an ask for
    a local service's details. The reply carries a URL of a page the world
    served, and the store is left as the turn found it."""
    cohort = await chat_eval(
        case_id="browse-reply-carries-the-page-url",
        behaviour=(
            "In the chat agent, when a standing rule asks for a source URL with "
            "every search and the answer is built on pages the browser serves, "
            "Penny's reply carries the URL of a page the world served."
        ),
        area=Area.ANSWERING_FROM_THE_WEB,
        model=model,
        seed=_seed_rules,
        prepare=_premise,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ASKS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=420.0,  # a search, up to three page reads each with an extraction draw
    )
    # STORE — a read-only ask: the seeded rules are exactly as the turn found
    # them, and nothing about the ask sends her to the store anyway.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — the half the issue exists for.
    cohort.claim(
        "reply: it carries the URL of a page the world served",
        _reply_carries_a_served_url,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
