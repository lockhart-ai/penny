"""A lookup's reply settles the ask on the pages the search results only name (#2653).

The contract is `docs/eval-case-design.md`.

The prod failure is the tempting-snippets trap: the search results look like
they answer — name, area, a friendly one-line blurb — so the reply can read
as complete while the one thing the ask turns on is only on the pages. So the
ask names a requirement the snippets never state: a tuner who can regulate a
GRAND piano. Every snippet blurb sounds qualified (one even says
"regulation"), and nothing in any snippet says who does grands. Each tuner's
own page states its services; exactly one says it regulates grands, and
others say outright that they don't, so a guess can land wrong. What the case
measures is that end state: the reply names the tuner whose page settles the
requirement.

**The claim that was dropped.** An early version also claimed the reply names
no tuner whose page disqualifies them, read as "no disqualified tuner's name
appears in the reply". The runs read it 0/15 on good answers: a careful reply
MENTIONS the excluded tuners as excluded ("Piano Works doesn't take
grands"), which the name check cannot tell from a wrong recommendation. The
claim was retired; the world keeps the disqualification statements so a
snippet-only guess can still land wrong, which is what the remaining claims
measure.

**The world is invented to the shape of a real lookup.** One ask, NO standing
rules — the browse how-to is built into the chat prompt, so the behaviour is
measured without one — a search page listing four tuners with name, area and
blurb, and four real-size candidate pages each carrying its own services and
rates the way a shop's own page does. Every tuner, place, address and price
is invented; the pages sit on real, common websites (yelp.com, wikipedia.org)
with invented paths and content, because an obviously made-up domain tells the
model the world is fake.

**What is NOT claimed.** That she called ``browse``, that any call carried
``extract``, which pages she opened, how many, or in what order — all ROUTE,
measured in the tool sequence and never asserted. How she words the
recommendation is its spread. The regulating tuner's rate sits only on its
page, so a correct answer naturally carries it, but it is world texture, not
a claim — the ask asks for the tuner, not the price.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the
code owner's to accept once the numbers have been read. Every tuner, place,
address and price is invented; the page hosts are real, common websites.
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

# ── The world: one search page, four tuner pages ─────────────────────────────
#
# The search page's ``match`` is the place name, so any query naming Millhaven
# reaches it; the four tuner pages stand BEFORE it, since their own URLs also
# carry the place name and the slug is the more specific match. Each tuner page
# is real-size — an address section, services, hours, notes — the shape a
# shop's own page carries, and each carries its URL in the visible body the way
# every canned page here does (``CannedPage``: the model cites URLs from the
# text it sees — on a read the ``## browse:`` header renders the URL beside the
# value the extract returned).
#
# The TEMPTATION is the point of the world: every snippet blurb sounds
# qualified but states no requirement-relevant fact — Aldergate's blurb (the
# one tuner whose page says it regulates grands) deliberately says nothing
# about grands, and Larkspur's says "regulation" (it regulates uprights only),
# so a snippet-only choice can land on a tuner whose page disqualifies them.

_TUNER1_URL = "https://yelp.com/biz/millhaven-piano-works"
_TUNER2_URL = "https://yelp.com/biz/millhaven-larkspur-keys"
_TUNER3_URL = "https://yelp.com/biz/millhaven-aldergate-tuning"
_TUNER4_URL = "https://yelp.com/biz/millhaven-quarryfield-pianos"

# The host-and-path prefix every served source URL shares — the shape a cited
# URL must carry. A URL token, never a slug: the slugs are the tuners' own
# names and a name-only reply would satisfy a slug check without any URL in it.
_URL_PREFIX = "yelp.com/biz/millhaven-"

# The requirement is settled by exactly one page: Aldergate's says it
# regulates grand pianos; the other three say outright that they don't. The
# names are the claim tokens — each appears only on its own page and as the
# search page's link, so a name in the reply identifies exactly that tuner.
_GRAND_TUNER = "Aldergate"

_TUNER_PAGES = (
    CannedPage(
        match="piano-works",
        text=(
            "Title: Millhaven Piano Works — tuning and repairs\n"
            "Millhaven Piano Works is a family-run workshop on Bellman Street "
            "in Millhaven, tuning uprights and consoles across the town.\n"
            f"Source: {_TUNER1_URL}\n"
            "\n"
            "Address\n"
            "14 Bellman Street, Millhaven, opposite the mill pond.\n"
            "\n"
            "Services\n"
            "Standard tuning, $135 a visit, typically two hours on site.\n"
            "Pitch raise before tuning, from $70 when the piano has been left "
            "flat for years.\n"
            "Minor repair work (sticky keys, pedal adjustments), quoted on site.\n"
            "Uprights and consoles are what we take — we don't work on grands.\n"
            "\n"
            "Hours\n"
            "Weekdays 9 to 6, Saturdays 10 to 4, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Bookings usually about a week out; a loaner practice room is "
            "available while a piano is in the workshop for bigger jobs.\n"
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
            "Services\n"
            "Standard tuning, $160 a visit, including a light clean of the keys.\n"
            "Voicing and regulation on uprights, from $95, quoted after a "
            "first look.\n"
            "Grands are the one thing we don't take on — no exceptions.\n"
            "\n"
            "Hours\n"
            "Weekdays 10 to 7, Saturdays 9 to 1, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Evening appointments twice a week; every visit ends with a short "
            "playing check with the owner.\n"
        ),
    ),
    CannedPage(
        match="aldergate-tuning",
        text=(
            "Title: Aldergate Tuning — tuning, voicing and grand regulation\n"
            "Aldergate Tuning is a bench-trained tuner on Harrow Walk in "
            "Millhaven, taking both family pianos and grands.\n"
            f"Source: {_TUNER3_URL}\n"
            "\n"
            "Address\n"
            "88 Harrow Walk, Millhaven, three doors from the fountain.\n"
            "\n"
            "Services\n"
            "Standard tuning, $185 a visit, with a written condition report "
            "afterwards.\n"
            "Grand piano regulation, from $120 — action regulation on any "
            "make of grand.\n"
            "Humidity-control fitting and check-ups, quoted per instrument.\n"
            "\n"
            "Hours\n"
            "Weekdays 8 to 5, Saturdays by appointment, closed Sundays.\n"
            "\n"
            "Notes\n"
            "Two-week lead time for first visits; a grand can stay on the "
            "stands for the job's length by arrangement.\n"
        ),
    ),
    CannedPage(
        match="quarryfield-pianos",
        text=(
            "Title: Quarryfield Piano Services — tuning, moving and "
            "restoration\n"
            "Quarryfield Piano Services is a family business on Quarryfield "
            "Road in Millhaven, tuning, moving and restoring pianos.\n"
            f"Source: {_TUNER4_URL}\n"
            "\n"
            "Address\n"
            "5 Quarryfield Road, Millhaven, by the rail bridge.\n"
            "\n"
            "Services\n"
            "Standard tuning, $210 a visit, two tuners on staff.\n"
            "Piano moving within Millhaven, quoted by staircase.\n"
            "Restoration and re-finishing, quoted after a workshop look.\n"
            "Uprights only for tuning work — grands we refer out.\n"
            "\n"
            "Hours\n"
            "Weekdays 8 to 6, Saturdays 9 to 3, closed Sundays.\n"
            "\n"
            "Notes\n"
            "A twice-yearly reminder service for annual tunings; restoration "
            "slots book out months ahead.\n"
        ),
    ),
)

_SEARCH_PAGE = CannedPage(
    match="millhaven",
    text=(
        "Title: piano tuner Millhaven — search results\n"
        "Piano tuners and tuning services in Millhaven.\n"
        "[Millhaven Piano Works — tuning and repairs]"
        f"({_TUNER1_URL})\n"
        "Family-run workshop on Bellman Street doing standard tunings, pitch "
        "raises and minor repairs on all pianos.\n"
        "\n"
        "Larkspur Key & String — piano tuning and voicing\n"
        "[Larkspur Key & String — tuning and voicing]"
        f"({_TUNER2_URL})\n"
        "One-person studio off Corver Lane; tuning, voicing and regulation "
        "for home pianos, with evening appointments twice a week.\n"
        "\n"
        "Aldergate Tuning — careful work and condition reports\n"
        "[Aldergate Tuning — tuning and voicing]"
        f"({_TUNER3_URL})\n"
        "Bench-trained tuner on Harrow Walk, known for careful work and a "
        "written condition report after every visit.\n"
        "\n"
        "Quarryfield Piano Services — tuning and restoration\n"
        "[Quarryfield Piano Services — tuning, moving and restoration]"
        f"({_TUNER4_URL})\n"
        "Family business on Quarryfield Road; pianos tuned, moved and "
        "restored, with a twice-yearly reminder service.\n"
        "\n"
        "A beginner's guide to looking after an acoustic piano\n"
        "[Caring for an acoustic piano — a guide]"
        "(https://en.wikipedia.org/wiki/Piano_tuning)\n"
        "How often to tune, where to put the piano, and what humidity does.\n"
    ),
    search=True,
)

# ── The five wordings ────────────────────────────────────────────────────────
#
# Ways a user asks for a tuner who can handle the requirement. What must not
# vary: the place, the service, the grand, and the regulation want. None
# mentions a link, a URL, a source, a snippet or the rules — the browse
# how-to is the prompt's own, and a wording that lent its phrasing would be
# answering the case for it.

_ASK = "can you find a piano tuner in millhaven who can also regulate a grand piano?"
_ASKS = (
    "looking for a piano tuner in millhaven — one who does grand regulation",
    "i need someone in millhaven to regulate a grand piano — can you find a tuner who does that?",
    "any piano tuners in millhaven who can regulate a grand piano?",
    "find me a millhaven piano tuner who handles grand regulation",
)

_TUNER_CLAUSES = ("piano tuner", "tuners", "tuner")
_GRAND_CLAUSES = ("grand",)
_REGULATE_CLAUSES = ("regulat",)


def _assert_every_wording_carries_the_facts() -> None:
    """Every one of the five wordings carries the facts the claims read — the
    place, the service, the grand, and the regulation want — and none of them
    names a link, a URL, a source, a snippet or the rules, so a dropped fact
    or a self-answering ask fails here rather than as a puzzling score after
    a paid run."""
    for ask in (_ASK, *_ASKS):
        folded = ask.casefold()
        assert "millhaven" in folded, f"this wording names no place — {ask!r}"
        assert any(clause in folded for clause in _TUNER_CLAUSES), (
            f"this wording names no service — {ask!r}"
        )
        assert any(clause in folded for clause in _GRAND_CLAUSES), (
            f"this wording names no grand — {ask!r}"
        )
        assert any(clause in folded for clause in _REGULATE_CLAUSES), (
            f"this wording asks for no regulation — {ask!r}"
        )
        for banned in ("link", "url", "source", "rule", "snippet"):
            assert banned not in folded, (
                f"this wording names {banned!r}, which is the prompt's own job — {ask!r}"
            )
    # The world coheres with the claims: exactly one page settles the
    # requirement, the others disqualify themselves in as many words, each
    # tuner's rate sits only on its own page, and the search page links every
    # tuner at an address under the served prefix — with no blurb naming the
    # requirement's winner.
    all_pages = [page.text for page in _TUNER_PAGES] + [_SEARCH_PAGE.text]
    for rate in ("135", "160", "185", "210"):
        where = [i for i, text in enumerate(all_pages) if rate in text]
        assert len(where) == 1, f"{rate} must sit exactly once — found at {where}"
    assert "grand piano regulation" in all_pages[2].casefold(), (
        "the settling page must state the requirement in as many words"
    )
    for disqualified in (0, 1, 3):
        assert "grands" in all_pages[disqualified].casefold(), (
            f"disqualified page {disqualified} must say it doesn't take grands"
        )
    for tuner_name, own in (
        ("Piano Works", 0),
        ("Larkspur", 1),
        ("Aldergate", 2),
        ("Quarryfield", 3),
    ):
        where = [i for i, text in enumerate(all_pages) if tuner_name in text]
        assert where == [own, len(_TUNER_PAGES)], (
            f"{tuner_name} must sit on its own page and the search page only — {where}"
        )
    for tuner_url in (_TUNER1_URL, _TUNER2_URL, _TUNER3_URL, _TUNER4_URL):
        assert tuner_url in _SEARCH_PAGE.text, f"{tuner_url} is linked from no search page"
        assert _URL_PREFIX in tuner_url
    assert "grand" not in _SEARCH_PAGE.text.casefold(), (
        "no snippet blurb may name the requirement or grand work — the snippets "
        "must tempt without stating the fact that settles the ask"
    )


_assert_every_wording_carries_the_facts()


_WORLD = World(
    name=(
        "a local-service lookup whose snippets sound qualified but state "
        "nothing about the requirement"
    ),
    pages=(*_TUNER_PAGES, _SEARCH_PAGE),
    keeps=(),
    excludes=(),
    answers=(),
)

# ── The claims ───────────────────────────────────────────────────────────────


def _reply_names_the_settling_tuner(sample: SampleObservation, _world: World) -> Answer:
    """The reply names the tuner whose page settles the ask — the requirement
    (grand regulation) is stated on exactly one page and in no snippet, so the
    name can only be earned by a read or a lucky guess; the world settles
    which name is correct. Typography-folded containment."""
    folded = fold_typography(sample.reply)
    hit = _GRAND_TUNER.casefold() in folded
    return hit, f"{_GRAND_TUNER!r} named in the reply: {hit}"


def _reply_carries_a_served_url(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a URL of a page the world served — read off the
    host-and-path prefix every tuner page's URL carries, typography-folded:
    a cited URL counts with or without its scheme.

    A URL the reply reached only through the search results' own links still
    counts: those links are the same addresses the pages serve. The search's
    one distractor link (a care guide, under a different host) is deliberately
    NOT accepted: it is a URL the world served, but not a source of anything
    the reply states about the tuners."""
    hit = _URL_PREFIX in fold_typography(sample.reply)
    return hit, f"served URL prefix {_URL_PREFIX!r} in the reply: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_lookup_reply_settles_the_ask_on_the_pages(chat_eval: ChatEval, model: str) -> None:
    """A lookup ask whose requirement the snippets never state: the reply
    names the tuner whose page settles it and carries a URL of a page the
    world served; the store is left as the turn found it."""
    cohort = await chat_eval(
        case_id="lookup-reply-settles-the-ask-on-the-pages",
        behaviour=(
            "In the chat agent, when a lookup's ask turns on a requirement "
            "the search snippets never state, Penny's reply names the "
            "candidate whose page settles it."
        ),
        area=Area.ANSWERING_FROM_THE_WEB,
        model=model,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ASKS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=480.0,  # a search, up to four page reads each with an extraction draw
    )
    # STORE — a read-only ask, nothing seeded by the case: what the store held
    # when the turn began is still there.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — the two halves the reworked world measures. (The dropped
    # no-disqualified claim: see the module docstring.)
    cohort.claim(
        "reply: it names the tuner whose page settles the ask",
        _reply_names_the_settling_tuner,
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
