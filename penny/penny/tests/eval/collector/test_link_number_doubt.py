"""A daily-headlines cycle that doubted its own link numbers (#2868).

**The failure (prod).**  A daily news job browses four front pages and stores the top
three headlines from each, with links.  The browse result carries a numbered link for
every headline (``[153]``) and the standing note that the number IS the address — never
look for it.  On the cycle this case replays, the model weighed that note against two
other texts and the numbers lost: the job's own prompt asks for headlines "with their
URLs", and the collector's "cite only what you actually browsed — never invent a URL"
rule predates link numbers and still talks about URLs as things the model must have in
hand.  It concluded a number was not yet a URL, spent a second browse hunting for the
addresses by searching each headline's title, and when that mostly came back empty,
stored the *site's front page* against every headline instead.

**This world is that cycle**, reproduced from its own ``promptlog`` rows: the job's
prompt verbatim, the collection's two most recent entries (real article links from
earlier days), and — for the first browse — the same four front pages with the same
headlines, each headline's own link now a real article address instead of a number the
case cannot pre-assign.  A follow-up browse (the model may search for a literal URL, as
it did in prod) gets the harness's own honest no-results page — the shape most of the
real second browse's hunt actually returned.

The headlines are public news (CBC, Reuters, AP, BBC); nothing else here is from real
use.  One arm, because this is a replay of one cycle, not a wording sweep — the thing
under test is whether the model trusts the numbers it was just shown, not how the job
happens to be phrased.

Report-only (``min_pass_rate=None``) until the fix lands and the number is read with the
code owner.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, collection_entries, seeded_run_id
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    TOOL_SEQUENCE,
    Feature,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-link-handles"

# The run that stood the job up.  A SEEDED id, structurally distinguishable from a live
# ``uuid4().hex``, so every "what did this sample write" reader excludes it at the one
# chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("link-number-doubt-entry-condition")
_SEED_AUTHOR = "eval-seed"

# The job, renamed from the prod collection it replays (``cbc_daily_headlines``) — the
# prompt and the description are otherwise verbatim.
_CONTAINER = "daily_headlines"
_CONTAINER_DESCRIPTION = (
    "Daily top 3 headlines from CBC news including title, description, and link"
)
_SCHEDULE = "FREQ=DAILY"
_PROMPT = (
    "Find the top 3 news headlines from cbc.ca, reuters.com, apnews.com, and bbc.com. "
    "For each source, list the 3 headlines with their titles and URLs. Format the "
    "output clearly by source. Save as a single entry for the day."
)

# The collection's two most recent entries going into the cycle — verbatim headlines and
# real article links from the two days before the replayed cycle.  These are what
# ``write_rule_decided`` puts under the store's own write path, so they carry the stamps
# and embeddings a real cycle's write would have left.
_PRIOR_ENTRIES = (
    EntryInput(
        key="2026-10-30-news",
        content=(
            "🇨🇦 CBC NEWS\n"
            "1. Trump's ability to shore up GOP support wanes as Democrats surge "
            "https://www.cbc.ca/player/play/video/9.7402327\n"
            "2. Why are Canadians so good at creeping us out? "
            "https://www.cbc.ca/arts/holding-space-creepy-canadians-9.7401627\n"
            "3. 'I don't do casual': Canada's Eurovision finalists on how they use "
            "fashion to stand out "
            "https://www.cbc.ca/life/culture/canada-eurovision-finalists-on-how-they"
            "-use-fashion-to-stand-out-9.7403186\n"
            "\n"
            "🌍 REUTERS NEWS\n"
            "1. Wall Street closes sharply lower as AI cash burn fears spark tech "
            "selloff "
            "https://www.reuters.com/business/nasdaq-futures-tumble-amazon-intel"
            "-results-deepen-ai-spending-woes-2026-10-30/\n"
            "2. October ends with a thud for Intel, Reddit, AI stalwarts "
            "https://www.reuters.com/legal/legalindustry/october-ends-with-thud"
            "-intel-reddit-ai-stalwarts-2026-10-30/\n"
            "3. Big Tech returns smallest share of cash to shareholders in 15 years, "
            "given AI spending "
            "https://www.reuters.com/legal/transactional/big-tech-returns-smallest"
            "-share-cash-shareholders-15-years-given-ai-spending-2026-10-30/\n"
            "\n"
            "🗞️ AP NEWS\n"
            "1. Difficult negotiations lie ahead in Israel after Netanyahu's bloc "
            "lost its parliamentary majority "
            "https://apnews.com/article/israel-elections-netanyahu-results-eisenkot"
            "-arabs-3a793c0d07fdfe515390524b849d8674\n"
            "2. Leo XIV has become the first pope in history to vote in a US "
            "general election "
            "https://apnews.com/article/vatican-pope-vote-us-election"
            "-cebcf71652c254383a39113c71aedace\n"
            "3. 4 things that could happen as Israeli parties try putting a "
            "coalition together "
            "https://apnews.com/article/israel-election-netanyahu-scenarios"
            "-eisenkot-1f53fda41207c3c1bd035c6358cf1409\n"
            "\n"
            "🇬🇧 BBC NEWS\n"
            "1. US approves sale of 500 Patriot missiles to Ukraine "
            "https://www.bbc.com/news/articles/c3vgxznlx7jzo\n"
            "2. Crypto giant Binance attempting to recover $6.3bn stolen in hack "
            "linked to North Korea "
            "https://www.bbc.com/news/articles/cwe8lp7zy07zo\n"
            "3. What's at stake for Trump in the US midterms? "
            "https://www.bbc.com/news/videos/cq77pm2jmmd5o"
        ),
    ),
    EntryInput(
        key="2026-10-29-news",
        content=(
            "🇨🇦 CBC NEWS\n"
            "1. Why none of the EU's existing partnerships are the right fit for "
            "Canada "
            "https://www.cbc.ca/news/canada/eu-canada-associate-membership-9.7398762\n"
            "2. More Canadian provinces are ending time changes. What does it mean "
            "for your health? "
            "https://www.cbc.ca/news/canada/daylight-saving-time-canada-health"
            "-9.7399791\n"
            "3. CBC Investigates: He tried to get into Montreal's kosher market. It "
            "led to a battle over alleged meat monopoly "
            "https://www.cbc.ca/news/canada/montreal/montreal-kosher-market-meat"
            "-monopoly-battle-9.7400089\n"
            "\n"
            "🌍 REUTERS NEWS\n"
            "1. Spain mourns, Bolsonaro's declaration and cruise residents stranded "
            "https://www.reuters.com/podcasts/spain-mourns-bolsonaros-declaration"
            "-cruise-residents-stranded-2026-10-29/\n"
            "2. Iran's hidden web of military agencies is slipping hardware past "
            "the US blockade "
            "https://www.reuters.com/world/iran/irans-hidden-web-military-agencies"
            "-is-slipping-hardware-past-us-blockade-2026-10-29/\n"
            "3. ECB President Lagarde resigns at a time of mounting challenges for "
            "Europe "
            "https://www.reuters.com/business/finance/ecb-hold-rates-investors"
            "-focus-french-bonds-lagardes-future-2026-10-28/\n"
            "\n"
            "🗞️ AP NEWS\n"
            "1. Spain's deadly train crash, warnings of more Iran strikes, Trump "
            "calls for prosecution of Alaskan Democrat and other news you need to "
            "know today "
            "https://apnews.com/newsletter/morning-wire/october-29-2026\n"
            "2. Photos show aftermath of deadly high-speed train collision in "
            "Spain "
            "https://apnews.com/photo-gallery/photos-show-aftermath-deadly-high"
            "-speed-train-collision-spain-897aba261c6b4bff8e831d0b7aa7f7dd\n"
            "3. Israelis await final election results as claim of foreign "
            "interference emerges "
            "https://apnews.com/article/israel-elections-netanyahu-eisenkot-arab"
            "-results-9a0620a1314fdc127245ba1a7f0aa762\n"
            "\n"
            "🇬🇧 BBC NEWS\n"
            "1. Families desperate for news as rescuers work to identify Spanish "
            "train crash victims "
            "https://www.bbc.com/news/live/cv1j309d4g98t\n"
            "2. Audio reveals exchanges between control centre and drivers before "
            "deadly Spain train crash "
            "https://www.bbc.com/news/articles/cmqxnl312v34o\n"
            "3. Flávio Bolsonaro denies fleeing Brazil after landing in Miami "
            "https://www.bbc.com/news/articles/c6kglwe4lxwzo"
        ),
    ),
)

# ── The first browse's four pages ────────────────────────────────────────────
#
# Each headline's own link is a REAL article address — not a number this case could
# pre-assign, since the handle table numbers on first sight and a fixture cannot know
# what else the model will have read first.  What stays constant with the replayed
# cycle is the correspondence: each headline cited a link, and that link was the
# article's own page, never the front page's root.  Reuters carries no clean "top 3"
# list, as the real front page didn't on the replayed cycle — which is why the cycle's
# own link count lands at 9, not 12.

CBC_PAGE = CannedPage(
    match="cbc.ca",
    text=(
        "Title: CBC.ca - watch, listen, and discover with Canada's Public "
        "Broadcaster\n"
        "https://cbc.ca\n"
        "\n"
        "top 3 news headlines:\n"
        "* [Wildwood is the longest stop-motion animated feature film ever. "
        "Should you watch?]"
        "(https://www.cbc.ca/listen/live-radio/1-14-day-6/clip/16244966-wildwood"
        "-longest-stop-motion-animated-feature-film-ever.-should)\n"
        "* [Mississippi man says noise from data centre plant is 'torture.' But "
        "he won't sell his home to Musk's SpaceXSI]"
        "(https://www.cbc.ca/radio/asithappens/spacexai-data-centre-buyouts"
        "-9.7403378)\n"
        "* [This Ontario ER built a separate waiting room for patients with "
        "mental health emergencies]"
        "(https://www.cbc.ca/radio/whitecoat/montfort-hospital-mental-health"
        "-emergency-zone-1.7643240)\n"
    ),
)

REUTERS_PAGE = CannedPage(
    match="reuters.com",
    text=(
        "Title: Reuters | Breaking International News & Views\n"
        "https://reuters.com\n"
        "\n"
        "Markets closed higher on strong earnings across the session.\n"
        "World leaders meet in Geneva for a second day of trade talks.\n"
        "Sign up for the daily newsletter to get analysis delivered to your "
        "inbox every morning.\n"
    ),
)

AP_PAGE = CannedPage(
    match="apnews.com",
    text=(
        "Title: Associated Press News: Breaking News, Latest Headlines and "
        "Videos | AP News\n"
        "https://apnews.com\n"
        "\n"
        "top 3 news headlines:\n"
        "1. [Leo XIV has become the first pope in history to vote in a US "
        "general election]"
        "(https://apnews.com/article/vatican-pope-vote-us-election"
        "-cebcf71652c254383a39113c71aedace)\n"
        "2. [Police secure Poland's Constitutional Tribunal as protests erupt "
        "over court chief's suspension]"
        "(https://apnews.com/article/poland-european-union-justice-protests"
        "-police-tribunal-67eda2e3a151e00038d85a5e894a2a72)\n"
        "3. [Christa Pike describes constant pain in first statement since her "
        "failed execution]"
        "(https://apnews.com/article/christa-pike-failed-execution-first"
        "-statement-292a1eb75ff5cee19da58f17c9eb224c)\n"
    ),
)

BBC_PAGE = CannedPage(
    match="bbc.com",
    text=(
        "Title: BBC Home | Breaking News, World News, US News, Sports, "
        "Business, Innovation, Climate, Culture, Travel, Video & Audio\n"
        "https://bbc.com\n"
        "\n"
        "top 3 news headlines:\n"
        "* [Evacuees from Russia-occupied city where residents ate weeds arrive "
        "in Ukraine](https://www.bbc.com/news/articles/cqx2818elrnzo)\n"
        "* [Why North Korea's 'elite' hackers are the prime suspect in major "
        "Binance hack](https://www.bbc.com/news/articles/ck9dz3v7z7v3o)\n"
        "* [Police deployed during stand-off over control of top Poland court]"
        "(https://www.bbc.com/news/articles/crz9824d5e88o)\n"
    ),
)

_PAGES = [CBC_PAGE, REUTERS_PAGE, AP_PAGE, BBC_PAGE]

# The real article addresses a faithful write owes — the per-headline links a stored
# entry is checked against.  Reuters contributes none: its front page carries no
# headline links, exactly as the replayed cycle's did not.
_REAL_ARTICLE_LINKS = (
    "https://www.cbc.ca/listen/live-radio/1-14-day-6/clip/16244966-wildwood-longest"
    "-stop-motion-animated-feature-film-ever.-should",
    "https://www.cbc.ca/radio/asithappens/spacexai-data-centre-buyouts-9.7403378",
    "https://www.cbc.ca/radio/whitecoat/montfort-hospital-mental-health-emergency"
    "-zone-1.7643240",
    "https://apnews.com/article/vatican-pope-vote-us-election"
    "-cebcf71652c254383a39113c71aedace",
    "https://apnews.com/article/poland-european-union-justice-protests-police"
    "-tribunal-67eda2e3a151e00038d85a5e894a2a72",
    "https://apnews.com/article/christa-pike-failed-execution-first-statement"
    "-292a1eb75ff5cee19da58f17c9eb224c",
    "https://www.bbc.com/news/articles/cqx2818elrnzo",
    "https://www.bbc.com/news/articles/ck9dz3v7z7v3o",
    "https://www.bbc.com/news/articles/crz9824d5e88o",
)


def _world() -> World:
    """This case's ground: the four front pages, no store-side token claims — every
    headline already names its own source by its URL, the same reasoning the digest
    case gives for an empty ``keeps``/``excludes``."""
    return World(name="link-number-doubt", pages=tuple(_PAGES), keeps=(), excludes=())


def _seed(db: Database) -> None:
    """The world this case's cycle starts in: the job already configured with its
    prompt, holding the two most recent days' entries — then every claim the seed
    makes, asserted out loud."""
    db.memories.create_collection(
        _CONTAINER,
        _CONTAINER_DESCRIPTION,
        extraction_prompt=_PROMPT,
        schedule=_SCHEDULE,
        notify=True,
    )
    require_memory(db, _CONTAINER).write_rule_decided(
        list(_PRIOR_ENTRIES), author=_SEED_AUTHOR, run_id=_SEED_RUN
    )
    _assert_the_world(db)


def _assert_the_world(db: Database) -> None:
    """Everything the seed is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert row.notify, "this case replays a notifying job, so the job must be notifying"
    prompt = row.extraction_prompt or ""
    for domain in ("cbc.ca", "reuters.com", "apnews.com", "bbc.com"):
        assert domain in prompt, (
            f"the prompt must name {domain!r} — a cycle can only fetch a source "
            f"something it reads names.  Prompt: {prompt!r}"
        )
    held = collection_entries(db, _CONTAINER)
    assert set(held) == {entry.key for entry in _PRIOR_ENTRIES}, (
        f"the container must start from exactly the two seeded days — it holds {held}"
    )
    for link in _REAL_ARTICLE_LINKS:
        assert any(link in page.text for page in _PAGES), (
            f"{link!r} must appear on the page it belongs to — the stored entry's "
            f"links are checked against what the cycle was shown.  {link!r} is missing"
        )


def _arm() -> CycleArm:
    return CycleArm(text=_PROMPT, seed=_seed, pages=_PAGES, world=_world())


# ── The claim ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_URL_TRAILING_PUNCTUATION = ".,;:!?)'\""


def _urls_in(text: str) -> list[str]:
    """Every URL ``text`` carries, trailing prose punctuation stripped."""
    return [url.rstrip(_URL_TRAILING_PUNCTUATION) for url in _URL_RE.findall(text)]


def _is_root_only(url: str) -> bool:
    """Whether ``url`` is a bare site root — no path beyond ``/`` — the shape #2868's
    cycle stored for every headline instead of the article's own page."""
    return urlparse(url).path in ("", "/")


def _held_text(sample: SampleObservation) -> str:
    """The collection as the cycle left it, key and content together."""
    return " ".join(
        " ".join(part for part in (entry.key, entry.content) if part)
        for entry in sorted(sample.held, key=lambda entry: entry.key or "")
    )


def _links_are_articles_not_front_pages(sample: SampleObservation, _world: World) -> Answer:
    """At least 6 of the cycle's 9-12 links are the headline's own article address, not
    the site's front page — the claim #2868's cycle failed on every one of its 12."""
    held = _held_text(sample)
    urls = _urls_in(held)
    roots = [url for url in urls if _is_root_only(url)]
    articles = len(urls) - len(roots)
    return (
        articles >= 6,
        f"{articles} of {len(urls)} stored links are article addresses; "
        f"{len(roots)} are bare site roots: {roots}; the collection holds {held!r}",
    )


# How many of a sample's stored links are bare site roots — reported, not gated, so a
# partial regression (9 links, 4 roots) is visible beside the pass/fail claim above.
ROOT_ONLY_LINKS = Feature(
    "root-only links stored", lambda o: str(sum(1 for u in _urls_in(_held_text(o)) if _is_root_only(u)))
)
# How many browse calls the cycle made — #2868's cycle made two: the front-page read,
# then a second browse hunting for the "actual" URLs the numbers already carried.
BROWSE_CALLS = Feature("browse calls", lambda o: str(o.tool_sequence.count("browse")))


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_headline_links_are_articles_not_front_pages(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """In a daily-headlines collector, when the browse result's numbered links already
    carry each headline's article address, Penny stores that address rather than
    doubting the number and falling back to the site's front page."""
    cohort = await collector_cycles_eval(
        case_id="link-number-doubt",
        behaviour=(
            "In a daily-headlines collector, when the browse result's numbered links "
            "carry each headline's article address, Penny stores that address rather "
            "than doubting the number and falling back to the site's front page."
        ),
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=[_arm()],
        samples_per_phrasing=15,
        min_pass_rate=None,  # report-only until the fix lands and the number is read
        family=_FAMILY,
    )

    cohort.claim(
        "state: at least 6 of the stored links are the headline's own article "
        "address, not the site's front page",
        _links_are_articles_not_front_pages,
        SpecCategory.STORE,
    )

    cohort.measure(TOOL_SEQUENCE, BROWSE_CALLS, ROOT_ONLY_LINKS)
