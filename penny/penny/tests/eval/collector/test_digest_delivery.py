"""A daily digest, delivered whole: the one case behind #2656's failure.

**The failure (prod, fictionalised).**  A daily-brief job's cycle browsed four
news sites, wrote ONE entry holding twelve headlines — three per source, each a
title and a URL — and closed with ``done(notify=true)``.  The message the user
received named only the FIRST source's three stories and added that the job
"mentioned there were more entries available during the run as well", because
the notify composer's document condensed the entry to a preview
(``NOTIFY_WRITTEN_CONTENT_CHARS``, with the honest ``… [N characters omitted]``
marker) — so the composer drew its message from a document that no longer held
eleven of the twelve items, and the message carried only what the preview held.

This case was written to reproduce that shape before the fix landed: a digest long
enough that the preview cut was guaranteed to fall mid-entry, a cycle that notifies,
and a claim that the delivered message carries every item the entry wrote — on the
unfixed main the entry-side claim held while the message-side claim failed, which is
what attributed the miss to the cut.  #2656 is the fix: the composer's document (and
its cut) is retired, the message is written by the chat-agent delivery turn from the
FULL entries, and this case stands as the contract that the delivery carries whole.

**One case, because there is one behaviour.**  The watch cases
(``test_watch_cycles.py``) split on the ENTRY CONDITION; this case's entry
condition is a first run of a daily digest job — an empty collection — and the
behaviour it checks is what the notification carries, not what the cycle wrote.
The tool sequence and the wording are variance; the three facts below are the
contract:

1. exactly one message reaches the user from the cycle;
2. that message names every one of the twelve headline titles — read with the
   shared ``token_stands_in`` check on a distinctive word from each title;
3. every URL in the message is sourced (``assert_every_value_in_the_reply_is_sourced``).

**The arms are five wordings of one job's prompt, over one set of facts.**  The
collector's natural language is the free-text ``extraction_prompt`` (the #2455
naive baseline), so the arm varies how that prompt words the same ask.  The
FACTS — four front pages, twelve headlines, three per source — are held
CONSTANT: the same twelve titles and URLs, byte-identical, sit on every arm's
pages wrapped in that arm's own masthead voice.  Constant facts are what let the
claims name every headline rather than "some digest" — the same reasoning the
watch's one pair of prices is held for.

Report-only (``min_pass_rate=None``).  The outlets are real ones the model
recognises, served whole from the local fixture mechanism (nothing is fetched from
the network, and the repo is public): the mastheads and URLs are real, the headlines
fictional but plausible.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database import Database
from penny.tests.eval.conftest import EVAL_MODELS, collection_entries, seeded_run_id
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-notification"

# The run that stood the job up.  A SEEDED id, structurally distinguishable from a live
# ``uuid4().hex``, so every "what did this sample write" reader excludes it at the one
# chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("digest-entry-condition")

# The one collection every arm's job writes to.  One name, because one job.
_CONTAINER = "morning-headlines"
_CONTAINER_DESCRIPTION = (
    "The day's top three headlines from each of my four news sites, as one digest."
)

# The job's cadence — a daily brief.  Stated because a configured collection has one,
# though the cycle is driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=DAILY"


class Headline(NamedTuple):
    """One story on a front page: its title, the page it sits on, and the one distinctive
    word a claim reads the digest and the message for.  The token is chosen to survive a
    faithful digest's paraphrase — a proper noun or an unusual noun, never a filler word —
    and to appear nowhere else in the world (probed in the seed)."""

    title: str
    url: str
    token: str


class Source(NamedTuple):
    """One news front page: its masthead, its address, and its three headline stories."""

    masthead: str
    url: str
    match: str
    headlines: tuple[Headline, ...]


class Voice(NamedTuple):
    """One arm's masthead voice: how all four pages of that arm are written around the
    constant headline blocks — a tagline, an intro line, a section header and a closing
    furniture note per source.  The FACTS never move; only the prose around them does."""

    name: str
    headline_header: str
    taglines: tuple[str, str, str, str]
    intros: tuple[str, str, str, str]
    closing: str


# ── The facts, held constant across every arm ────────────────────────────────
#
# Four front pages, twelve headlines — three per source, each a realistic-length title and
# the page it sits on.  Twelve titles and URLs in one entry runs well past the notify
# document's 500-character preview cut, which is the whole point: the cut MUST fall
# mid-entry for the failure to show.  Every token is distinctive to its own headline and
# probed (in the seed) to appear on no other source's page.

# Real outlets, served from the same local fixture mechanism — the model recognises
# these mastheads and URLs, where invented ones read as a private world and measurably
# change its behaviour (code-owner ruling, Oct 19).  The outlets are chosen so no real
# headline of theirs is being simulated: the HEADLINES stay fictional but plausible,
# each anchored on a real place or body the model knows.
GUARDIAN = Source(
    masthead="The Guardian",
    url="https://www.theguardian.com",
    match="theguardian.com",
    headlines=(
        Headline(
            "Portsmouth ferry terminal reopens a month early after an eighteen-month refit",
            "https://www.theguardian.com/uk-news/portsmouth-ferry-terminal-refit-opens-early",
            "ferry",
        ),
        Headline(
            "Brighton marina hotel wins planning permission after a two-year appeal",
            "https://www.theguardian.com/uk-news/brighton-marina-hotel-planning-appeal-win",
            "marina",
        ),
        Headline(
            "Belfast cannery workers ratify a three-year pay deal",
            "https://www.theguardian.com/uk-news/belfast-cannery-workers-ratify-pay-deal",
            "cannery",
        ),
    ),
)
NPR = Source(
    masthead="NPR",
    url="https://www.npr.org",
    match="npr.org",
    headlines=(
        Headline(
            "Seattle council rejects the downtown tunnel toll plan after a heated debate",
            "https://www.npr.org/seattle-council-tunnel-toll-plan-vote",
            "tunnel",
        ),
        Headline(
            "Maine kelp farm wins the state's first coastal lease",
            "https://www.npr.org/maine-kelp-farm-first-coastal-lease",
            "kelp",
        ),
        Headline(
            "Songbird atlas passes 40,000 sightings in its first winter",
            "https://www.npr.org/science/songbird-atlas-first-winter-count",
            "songbird",
        ),
    ),
)
AL_JAZEERA = Source(
    masthead="Al Jazeera",
    url="https://www.aljazeera.com",
    match="aljazeera.com",
    headlines=(
        Headline(
            "Luton stun the league leaders with a late winner",
            "https://www.aljazeera.com/sports/luton-stun-league-leaders",
            "luton",
        ),
        Headline(
            "Rowing eight retain the head of the river title",
            "https://www.aljazeera.com/sports/rowing-eight-head-of-the-river",
            "rowing",
        ),
        Headline(
            "Velodrome bids to host next year's national track meeting",
            "https://www.aljazeera.com/sports/velodrome-national-track-meeting-bid",
            "velodrome",
        ),
    ),
)
DW = Source(
    masthead="DW",
    url="https://www.dw.com",
    match="dw.com",
    headlines=(
        Headline(
            "Otter cameras catch a first litter in the Spreewald marshes",
            "https://www.dw.com/en/otter-cameras-first-litter-spreewald-marshes",
            "otter",
        ),
        Headline(
            "Night-sky survey logs a record meteor count over the Eifel",
            "https://www.dw.com/en/night-sky-record-meteor-count-eifel",
            "meteor",
        ),
        Headline(
            "Cologne distillery doubles its bottling line with a new hall",
            "https://www.dw.com/en/cologne-distillery-doubles-bottling-line",
            "distillery",
        ),
    ),
)

SOURCES = (GUARDIAN, NPR, AL_JAZEERA, DW)

# Every headline's distinctive word — the answer the delivered message owes, all twelve.
# This is the claim that reproduces the failure: a message drawn from a cut preview can
# name the first source's three and never the rest.
_DIGEST_TOKENS = tuple(headline.token for source in SOURCES for headline in source.headlines)

# The one-line standfirst under each headline link — constant across every arm, like the
# link line itself, so the pages read like real front pages and the entry the prompt asks
# for is the same world on every arm.  None carries its headline's distinctive word
# (probed in the seed).
_STANDFIRSTS: dict[str, str] = {
    "ferry": (
        "Crews finished the pier works six weeks ahead of the schedule the council was given."
    ),
    "tunnel": ("The toll proposal fell 7-4 after an hour of submissions from riverside residents."),
    "kelp": "The lease is the first the state's marine board has granted for cultivation.",
    "songbird": (
        "Volunteers logged sightings from every parish in the valley over the winter months."
    ),
    "otter": ("The marsh cameras caught the litter playing at dawn on three separate mornings."),
    "meteor": ("The count peaked after midnight, with more than two hundred trails an hour."),
    "luton": "A stoppage-time goal lifts the side off the bottom of the table.",
    "rowing": ("The crew held off a late charge from the boat club two bridges downstream."),
    "velodrome": ("The track needs resurfacing before any national meeting could be hosted there."),
    "distillery": ("The expansion adds a second still hall and twelve new jobs at the quayside."),
    "marina": ("The planning appeal ended with conditions on the breakwater and the car park."),
    "cannery": ("The deal raises wages in each of the three years and keeps the night shift on."),
}

# Per-source furniture — constant across arms, like the headline blocks: a weather strip,
# field notes, fixtures, or a markets line, none of it a headline and none of it linked.
_FURNITURE: dict[str, tuple[str, str]] = {
    "theguardian.com": (
        "Weather: low cloud clearing by noon, highs near 14, a southerly breeze on the water.",
        "Tide times: 06:41 high, 13:02 low, 19:20 high.",
    ),
    "npr.org": (
        "Field notes: the reserve's boardwalk closes for repairs after this weekend.",
        "Sky diary: the observatory opens its dome to visitors on the first Friday of the month.",
    ),
    "aljazeera.com": (
        "Fixtures: the cup draw is made on Monday evening.",
        "Club notes: season cards go on sale at the end of the month.",
    ),
    "dw.com": (
        "Markets: the harbour index closed a little firmer, led by the shipping lines.",
        "Briefing: the chamber's trade breakfast is on Thursday; seats are limited.",
    ),
}

_VOICES = (
    Voice(
        name="top-stories",
        headline_header="Top stories",
        taglines=(
            "the city's morning paper",
            "the sound's nature and science desk",
            "sport, first and fast",
            "the harbour's business daily",
        ),
        intros=(
            "Good morning — here is what the city is talking about today.",
            "Everything the valley's naturalists filed overnight, in one place.",
            "Last night's results and this morning's moves, as they landed.",
            "The numbers behind the harbour, before the markets open.",
        ),
        closing="Sections: City · Nature · Sport · Business · Letters",
    ),
    Voice(
        name="front-page",
        headline_header="This morning's front page",
        taglines=(
            "the paper the pier crews read",
            "field notes from the estuary",
            "the touchline report",
            "ledger lines from the cove",
        ),
        intros=(
            "Your front page for today, headed by the reopening everyone waited on.",
            "The reserve's morning round-up, written before the hides opened.",
            "Who won, who signed, and who is complaining about the pitch.",
            "Overnight filings and the day's one big number, up first.",
        ),
        closing="Published mornings · Reply to the letters desk by noon",
    ),
    Voice(
        name="leading-stories",
        headline_header="Today's leading stories",
        taglines=(
            "city hall to the waterline",
            "birds, tides and night skies",
            "every match that matters",
            "trade, tide and turnover",
        ),
        intros=(
            "Three stories lead the paper this morning; the rest is on the inside pages.",
            "The dawn round-up from the survey teams and the reserve wardens.",
            "The morning sports brief, compiled from last night's fixtures.",
            "What moved overnight and what it means for the harbour trade.",
        ),
        closing="Editorial desk open from six · Evening edition online",
    ),
    Voice(
        name="at-a-glance",
        headline_header="The headlines at a glance",
        taglines=(
            "the estuary's daily sheet",
            "the valley's wildlife wire",
            "results and rumours",
            "the cove's business sheet",
        ),
        intros=(
            "Skim the three big ones here; the full write-ups are a click away.",
            "A quick look at what the cameras and the counters recorded.",
            "The short version of the sports desk's morning meeting.",
            "Two minutes on the harbour economy, then get on with the day.",
        ),
        closing="One sheet a day · Corrections welcome at the desk",
    ),
    Voice(
        name="latest-headlines",
        headline_header="Latest headlines",
        taglines=(
            "news on the hour from the pier head",
            "the nature log, updated daily",
            "the sport wire",
            "figures from the quay",
        ),
        intros=(
            "The latest from around the city, refreshed through the morning.",
            "What the monitoring teams logged since yesterday's sheet.",
            "Everything confirmed overnight by the clubs and the league.",
            "The opening position on every line the quay watches.",
        ),
        closing="Continuous updates · Next sheet at noon",
    ),
)


def _front_page(source: Source, voice: Voice) -> CannedPage:
    """One arm's page for one source: the CONSTANT headline blocks (link line and
    standfirst per story) wrapped in THIS arm's masthead voice — tagline, intro, section
    header, furniture and closing line.  The four pages of one arm are one edition; the
    five arms are five editions of the same twelve stories."""
    index = SOURCES.index(source)
    lines = [
        f"Title: {source.masthead} — {voice.taglines[index]}",
        source.url,
        "",
        voice.intros[index],
        "",
        voice.headline_header,
    ]
    for headline in source.headlines:
        lines.append(f"[{headline.title}]({headline.url})")
        lines.append(f"{_STANDFIRSTS[headline.token]}")
        lines.append("")
    furniture = _FURNITURE[source.match]
    lines.extend([*furniture, "", voice.closing])
    return CannedPage(match=source.match, text="\n".join(lines))


def _prompt() -> str:
    """The base arm's job instructions (#2455 naive baseline) — what the collection's
    ``extraction_prompt`` carries verbatim, with no rendering seam.

    ONE shape for every arm: the same four front pages, the same three-headlines-per-source
    ask, the same single-entry destination.  The only thing that differs is the WORDING —
    which is what the arm axis is.  No wording names a tool, and none states the outcome a
    claim reads for."""
    a, b, c, d = (source.url for source in SOURCES)
    return (
        f"Read {a}, {b}, {c} and {d}. From each source take the top 3 headlines, each "
        f"with its title and its URL, and group them under their source. Save the whole "
        f"day's digest as a single entry in '{_CONTAINER}'."
    )


_PROMPTS = (
    _prompt(),
    f"Build today's news digest: visit {SOURCES[0].url}, {SOURCES[1].url}, "
    f"{SOURCES[2].url} and {SOURCES[3].url}, pull each site's three leading stories, and "
    f"note every story's headline and link. Keep the four sites apart in the layout, and "
    f"write the complete digest into '{_CONTAINER}' as one entry for the day.",
    f"It's the morning headlines round. Go through {SOURCES[0].url}, {SOURCES[1].url}, "
    f"{SOURCES[2].url} and {SOURCES[3].url}, take the three biggest stories from each, "
    f"and record each one's title and address. Lay the entry out source by source — one "
    f"'{_CONTAINER}' entry holds the whole day.",
    f"Collect the top three headlines from each of {SOURCES[0].url}, {SOURCES[1].url}, "
    f"{SOURCES[2].url} and {SOURCES[3].url}, every headline with its title and the page "
    f"it sits on. Format the digest clearly by source, and keep it all together as a "
    f"single entry for the day in '{_CONTAINER}'.",
    f"Daily digest time: read the front pages at {SOURCES[0].url}, {SOURCES[1].url}, "
    f"{SOURCES[2].url} and {SOURCES[3].url}, and take each one's three headline stories "
    f"with their titles and URLs. Present them grouped by source, and save the full set "
    f"— one entry for the day — into '{_CONTAINER}'.",
)


def _world(voice: Voice) -> World:
    """This arm's ground: its four front pages, and the twelve tokens the delivered
    message owes.

    ``keeps``/``excludes`` are EMPTY, and deliberately — the same reasoning the watch
    cases give: a ``keeps`` token has to appear on ONE page so a stored copy says which
    page it came from, and every headline here already names its source by its URL.
    ``answers`` is the whole point of the case: the twelve headline tokens are what the
    message the user receives must state, and the reply-side claim reads them."""
    return World(
        name=voice.name,
        pages=tuple(_front_page(source, voice) for source in SOURCES),
        keeps=(),
        excludes=(),
        answers=_DIGEST_TOKENS,
    )


def _seeder(voice: Voice):
    """The world this case's cycle starts in: a job already configured with its prompt,
    an EMPTY collection (a digest job's first run of the day), and every claim that world
    makes asserted out loud — so a drifted seed fails in the seed rather than after GPU
    time."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_PROMPTS[_VOICES.index(voice)],
            schedule=_SCHEDULE,
            notify=True,
        )
        _assert_the_digest_world(db, voice)

    return seed


def _assert_the_digest_world(db: Database, voice: Voice) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert row.notify, "this case measures the notification, so the job must be notifying"

    prompt = row.extraction_prompt or ""
    for source in SOURCES:
        assert source.url in prompt, (
            f"the prompt must name {source.url!r} — a cycle can only fetch the pages "
            f"something it reads names.  Prompt: {prompt!r}"
        )
    assert f"'{_CONTAINER}'" in prompt, (
        f"the prompt must name its own destination, {_CONTAINER!r}.  Prompt: {prompt!r}"
    )
    assert "three" in prompt or "3" in prompt, (
        "the prompt must carry the three-per-source count — the digest size is the "
        f"fact the claims hinge on.  Prompt: {prompt!r}"
    )
    held = collection_entries(db, _CONTAINER)
    assert not held, (
        "the container must be empty when the cycle starts — a digest job's first run "
        f"of the day writes the entry fresh, and it holds {held}"
    )

    # The world's coherence: every headline stands on its own page, its token stands in
    # its own title and NOWHERE else in the world, and the smallest entry the ask produces
    # runs far past the notify document's cut — the cut must fall mid-entry for this case
    # to measure anything.
    pages = {source.match: _front_page(source, voice) for source in SOURCES}
    for source in SOURCES:
        page = pages[source.match]
        for headline in source.headlines:
            assert headline.url in page.text, (
                f"{headline.url!r} must appear on its own page — the message's URLs are "
                f"sourced from what the cycle was shown.  Page: {page.text!r}"
            )
            assert token_stands_in(headline.token, headline.title), (
                f"the token {headline.token!r} must stand in its own title: {headline.title!r}"
            )
            assert not token_stands_in(headline.token, _STANDFIRSTS[headline.token]), (
                f"the standfirst must not carry the token {headline.token!r} — the "
                f"headline's title is where the claim reads it"
            )
        others = "\n".join(pages[other.match].text for other in SOURCES if other is not source)
        for headline in source.headlines:
            assert not token_stands_in(headline.token, others), (
                f"the token {headline.token!r} must appear on no other source's page — "
                "an ambiguous token cannot tell the digest's items apart.  Other "
                f"pages: {others!r}"
            )


def _arms() -> list[CycleArm]:
    """Five arms — five wordings of the one job's prompt, each over its own edition of
    the same twelve stories."""
    return [
        CycleArm(
            text=prompt,
            seed=_seeder(voice),
            pages=[_front_page(source, voice) for source in SOURCES],
            world=_world(voice),
        )
        for prompt, voice in zip(_PROMPTS, _VOICES, strict=True)
    ]


# ── The claims ───────────────────────────────────────────────────────────────
#
# All three read END STATE: what the collection holds and what reached the send queue.
# Nothing reads a tool name or an ordering — the tool sequence is measured in section B.


def _held_text(sample: SampleObservation) -> str:
    """The collection as the cycle left it, key and content together — the WHOLE entry,
    because a fact in the key and a blurb in the body is a perfectly good way to store it."""
    return " ".join(
        " ".join(part for part in (entry.key, entry.content) if part)
        for entry in sorted(sample.held, key=lambda entry: entry.key or "")
    )


def _one_entry(sample: SampleObservation, _world: World) -> Answer:
    """The job keeps the day's digest as ONE entry — the ask's own words."""
    keys = sorted(entry.key or "" for entry in sample.held)
    return len(sample.held) == 1, f"holds {len(sample.held)} entries: {keys}"


def _digest_holds_every_headline(sample: SampleObservation, _world: World) -> Answer:
    """The entry the cycle wrote carries every headline's distinctive word.

    This is the anchor that makes the message-side miss attributable: when it holds and
    the reply-side claim fails, the cut — not the cycle — is what dropped the items."""
    held = _held_text(sample)
    missing = [token for token in _DIGEST_TOKENS if not token_stands_in(token, held)]
    return not missing, f"the entry never states {missing}; the collection holds {held!r}"


def _told_once(sample: SampleObservation, _world: World) -> Answer:
    """The user was told, and told once — read over the SEND QUEUE, which is what the
    user will actually receive (a cycle enqueues; the drainer is a separate schedule)."""
    count = len(sample.notifications)
    return count == 1, f"{count} messages reached the send queue: {sample.notifications}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_digest_notification_carries_every_item_it_wrote(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A digest job's cycle writes one entry holding the whole day's items and notifies —
    the message the user receives carries every item the entry holds."""
    cohort = await collector_cycles_eval(
        case_id="digest-delivered-whole",
        behaviour=(
            "In a headline-digest collector, when the job writes one entry holding the "
            "whole day's digest and notifies, the message the user receives carries "
            "every item the entry holds."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )

    # STORE
    cohort.claim(
        "state: the collection holds one entry, the day's digest",
        _one_entry,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the entry holds every headline the job collected",
        _digest_holds_every_headline,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: exactly one message reaches the user from the cycle",
        _told_once,
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply-side half is the third fact: every URL in the message is
    # sourced, alongside everything else specific in it.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()
    # The message-side completeness claim, read off the world's twelve answer tokens —
    # the claim that reproduces the cut on today's main.
    cohort.assert_the_reply_answers_the_ask()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
