"""A digest cycle's new entry cites the page its content came from (#3074).

The contract is `docs/eval-case-design.md`.

**The shape is production's #3074 turn, genericised.** In prod, a collector cycle
wrote an entry whose ``Source:`` line carried the WRONG address, silently: it wrote
``Source: [1] https://phys.org`` where ``[1]`` was the first page the run opened (an
unrelated article) and the domain was invented — the real page the content came from
was ``[18]``. Replays of those prod rows (12 samples per arm, gemma 8-bit; passing
counts) measured the failure to the render the cycle is shown: the collection's
holdings render each stored source as ``Source: [1] <full url>`` — which teaches
"number + URL" and offers ``[1]`` as an example — and a bare bracketed number reads
as a footnote rather than as the link itself. With the holdings rendered ``{link N}``
alone, the same write step landed 12/12, every write exactly ``{link 118}``, no URL,
no wrong number; with ``token URL`` pairs kept, 1/12.

**What is measured.** The collection already holds earlier entries whose stored
content carries a ``Source:`` line (as a real digest's does), so the cycle's ambient
holdings render sources as tokens; the cycle reads the page its prompt names and
writes ONE new entry. The asserted fact is a store fact: the entry that carries the
page's distinctive datum carries THE PAGE's own address as its source — read back
exactly from the store, never a shape. What is NOT claimed: which tool resolved the
address — the page section's own handle, a re-read, anything else — how many browse
calls it took, or how she words the entry; route and spread are variance.

**Real proper nouns, swapped.** The night-sky subject and its outlets are real names
the model recognises (Comet Lemmon, Sky & Telescope, EarthSky, Universe Today); the
paths, the magnitudes and the collection are invented, and the repo is public. The
seeded sources' addresses and the page's decoy links are mutually exclusive with the
article's own address, so naming one can never match another.

Report-only (``min_pass_rate=None``) on its first run: this run is to show whether
the failure reproduces, not to grade it.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import EVAL_MODELS, collection_entries, seeded_run_id
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid the entry condition down — SEEDED, so every "what did this sample
# write" reader excludes it at the one chokepoint.
_SEED_RUN = seeded_run_id("digest-cite-entry-condition")

# The one collection every arm's job is configured on.  One name, because one job.
_CONTAINER = "astro_news_digest"
_CONTAINER_DESCRIPTION = "Short daily notes on what's new in the night sky, each with its source."

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.  The job does NOT notify — the
# behaviour under test is the write, and nothing here needs the send queue.
_SCHEDULE = "FREQ=DAILY"

# ── The facts, held constant across every arm ────────────────────────────────
#
# One article, one url, one distinctive datum.  The datum is the identity of the new
# entry — what the job found worth storing — and what the claims match on; it is
# byte-identical across all five pages, which is what lets a claim name it.  It
# appears exactly once in the world, and in NO seeded entry, so an entry carrying it
# is the new write and nothing else.

_ARTICLE = "Comet Lemmon"
_ARTICLE_URL = "https://skyandtelescope.org/observing/comet-lemmon-brightening/"
_MATCH = "skyandtelescope"
_DATUM = "magnitude 9.7"

# The earlier entries' own sources — real outlets, invented paths — and the page's
# decoy links (below).  All four addresses are mutually exclusive, so a claim naming
# one never matches another.
_ORIONID_URL = "https://earthsky.org/tonight/orionid-meteors-peak-this-week/"
_PLEIADES_URL = "https://www.universetoday.com/170231/moon-occults-the-pleiades/"
_SKY_EVENTS_URL = "https://skyandtelescope.org/observing/sky-events-this-week/"
_TONIGHT_URL = "https://in-the-sky.org/news/tonights-sky-guide"

# Every address the new entry's source must NOT be — the seeded entries' own sources
# and the article page's related links.  The prod failure wrote exactly one of these
# shapes: the first page the run had open, and an invented domain.
_OTHER_SOURCES = (_ORIONID_URL, _PLEIADES_URL, _SKY_EVENTS_URL, _TONIGHT_URL)

# The two seeded entries: yesterday's items, each a one-liner plus its source — the
# shape a real digest entry is stored in, and the render the cycle reads ambiently.
# Their keys are the pinned clock's own recent days (the sample starts Wed 14 Oct 2026).
_SEED_DAY_1 = "2026-10-12"
_SEED_DAY_2 = "2026-10-13"
_SEED_DAY_1_CONTENT = (
    "The Orionid meteors peak after midnight this week, with rates near their yearly "
    "best.\n"
    f"Source: {_ORIONID_URL}"
)
_SEED_DAY_2_CONTENT = (
    "A crescent Moon slides across the Pleiades early Saturday, hiding several of its "
    "brightest stars.\n"
    f"Source: {_PLEIADES_URL}"
)
_SEED_ENTRIES = ((_SEED_DAY_1, _SEED_DAY_1_CONTENT), (_SEED_DAY_2, _SEED_DAY_2_CONTENT))


class Voice(NamedTuple):
    """One arm: how the job's ``extraction_prompt`` words the ask, and the prose that
    makes this arm's page read differently from its siblings'.

    ``name`` is what this arm is called wherever it is identified.  ``program`` is the
    job's free-text instruction, stored verbatim as ``extraction_prompt`` (the only
    natural language a collector cycle is handed).  Everything on the page except the
    intro, the observing-notes voice and the closing line is the constant text below —
    same shape as the watch cases' five voices over one page."""

    name: str
    program: str
    intro: str
    notes_header: str
    notes: str
    closing: str


# The page's constant blocks — byte-identical across every arm, like the watch cases'
# shared boilerplate.  ``_DATUM_LINE`` is the watched line; the two related links are
# the in-page decoys whose addresses must never become the new entry's source.
_DATUM_LINE = (
    f"{_ARTICLE} (C/2025 A6) brightened again this week and now sits comfortably at "
    f"{_DATUM}, visible in binoculars from a dark site."
)
_RELATED_BLOCK = (
    "More from Sky & Telescope\n"
    f"[This week's sky events]({_SKY_EVENTS_URL})\n"
    f"[Tonight's sky guide]({_TONIGHT_URL})\n"
)
_ARTICLE_LINK = f"[{_ARTICLE}: the full observing guide]({_ARTICLE_URL})"

_PAGE_FOOTER = (
    "About · Observing · News · Equipment · Community\n"
    "Times shown are for the observer's own horizon and may shift with location.\n"
)


def _body(voice: Voice) -> str:
    """One arm's whole page: the constant datum line, related links and footer above,
    wrapped in this arm's own intro, observing notes and closing line."""
    return (
        f"Title: {_ARTICLE} keeps brightening | Sky & Telescope\n"
        f"{_ARTICLE_URL}\n"
        "\n"
        f"{voice.intro}\n"
        "\n"
        f"{_DATUM_LINE}\n"
        "\n"
        f"{_ARTICLE_LINK}\n"
        f"{voice.notes_header}\n"
        f"{voice.notes}"
        "\n"
        "From the community\n"
        '"Picked it up in 10x50s from a green belt last night — easy once the Moon '
        'set." — an observing-forum member\n'
        '"Try after astronomical twilight; it was still easy in the same field as a '
        'bright star." — an observing-forum member\n'
        "\n"
        f"{_RELATED_BLOCK}"
        "\n"
        "Weather outlook\n"
        "High pressure holds over most of the country this week, with clear windows "
        "after local midnight for much of it.\n"
        "\n"
        f"{voice.closing}\n"
        "\n"
        f"{_PAGE_FOOTER}"
    )


# Five catalogue-grade arms: same article, same url, same datum, five instructions
# over five voices.  A real page carries more than the job needs — community chatter,
# a weather outlook, related links with addresses of their own — so a cycle that
# grabs a source is only demonstrably reading the RIGHT one where wrong ones are
# around to grab.
VOICES = (
    Voice(
        name="what-new-there",
        program=(
            f"Read {_ARTICLE_URL} and add one short item on what's new there to "
            f"'{_CONTAINER}'. Each item carries the page it came from as its source."
        ),
        intro="The comet has been quietly outperforming its own light-curve forecast all month.",
        notes_header="Observing notes\n",
        notes=(
            "Best seen in the northeast before dawn, low but findable in 10x50 "
            "binoculars.\n"
            "Look for a condensed coma and a short tail pointing away from the horizon.\n"
        ),
        closing="The light curve has been revised upward twice this month already.",
    ),
    Voice(
        name="cite-the-page",
        program=(
            f"Visit {_ARTICLE_URL}, then store a one-line item in '{_CONTAINER}' citing "
            "the page you read it from."
        ),
        intro="Amateur observers keep adding magnitude estimates to the international archive.",
        notes_header="What to look for\n",
        notes=(
            "A modest telescope shows the coma's shape best; binoculars suffice for "
            "the brightness.\n"
            "Dark adaptation of twenty minutes makes the difference here.\n"
        ),
        closing="Recent estimates from three continents agree within a tenth of a magnitude.",
    ),
    Voice(
        name="whats-new-put-item",
        program=(
            f"Check {_ARTICLE_URL} for what's new and put a brief item on it in "
            f"'{_CONTAINER}', with its source page."
        ),
        intro="The comet rounded the sun last month and is now swinging back through the inner sky.",
        notes_header="Finder notes\n",
        notes=(
            "The field holds two bright stars that make pointing easy in binoculars.\n"
            "A moonless window opens midweek and closes on the weekend.\n"
        ),
        closing="Its orbit brings it back around in roughly seventy years.",
    ),
    Voice(
        name="write-up-item",
        program=(
            f"Look at {_ARTICLE_URL} and write up one item into '{_CONTAINER}' — every "
            "entry names its source."
        ),
        intro="Sky charts for the coming week place the comet in the same binocular field as a "
        "well-known cluster.",
        notes_header="How to find it\n",
        notes=(
            "Start from the cluster and sweep slowly northeast; the coma shows as a "
            "soft smudge.\n"
            "Rural skies show the faint tail; suburban skies show the coma alone.\n"
        ),
        closing="Photographers have been stacking short exposures to bring out the ion tail.",
    ),
    Voice(
        name="save-short-item",
        program=(
            f"Read {_ARTICLE_URL} and save a short item to '{_CONTAINER}', source included."
        ),
        intro="The brightness trend has made the comet a regular on this month's observing lists.",
        notes_header="Observing window\n",
        notes=(
            "The comet climbs a little higher each morning through the end of the "
            "week.\n"
            "By month's end it should reach naked-eye limit from truly dark sites.\n"
        ),
        closing="Its brightness has doubled since the first week of the month.",
    ),
)


def _page(voice: Voice) -> CannedPage:
    """This arm's cycle reads THIS arm's page."""
    return CannedPage(match=_MATCH, text=_body(voice))


def _world(voice: Voice) -> World:
    """This arm's ground: the page its single cycle reads.

    ``keeps``/``excludes`` are EMPTY, and deliberately — the same reason the watch
    cases give: those token sets back ``assert_something_from_each_page_was_written``,
    which this case never calls, and the article is the only page every claim names
    by its exact address."""
    return World(name=voice.name, pages=(_page(voice),), keeps=(), excludes=())


def _seeder(voice: Voice):
    """The world this case's cycle starts in: a job already configured with its
    prompt, and the ENTRY CONDITION — the collection already holding two earlier
    entries with sources — then every claim that world makes, asserted out loud.

    The entries go down through the store's own write path, so the rows carry the
    stamps and the embeddings a real cycle's write would have left; hand-inserting
    them would give the cycle a holdings render no real store could have produced."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=voice.program,
            schedule=_SCHEDULE,
            notify=False,
        )
        memory = db.memory(_CONTAINER)
        assert memory is not None, f"the job's container {_CONTAINER!r} must exist to seed"
        memory.write_rule_decided(
            [EntryInput(key=key, content=content) for key, content in _SEED_ENTRIES],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        _assert_the_digest_world(db, voice)

    return seed


def _assert_the_digest_world(db: Database, voice: Voice) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert not row.notify, "this case measures the write, so the job must not notify"

    prompt = row.extraction_prompt or ""
    assert _ARTICLE_URL in prompt, (
        f"the prompt must name {_ARTICLE_URL!r} — a cycle can only read the page "
        f"something it reads names.  Prompt: {prompt!r}"
    )
    assert f"'{_CONTAINER}'" in prompt, (
        f"the prompt must name its own destination, {_CONTAINER!r}.  Prompt: {prompt!r}"
    )

    held = collection_entries(db, _CONTAINER)
    assert held == {key: content for key, content in _SEED_ENTRIES}, (
        f"the cycle must start from exactly the two seeded entries; it holds {held}"
    )
    for _, content in _SEED_ENTRIES:
        assert _DATUM not in content, (
            f"the datum {_DATUM!r} must appear nowhere in the seeded entries, or an "
            f"entry carrying it is not evidence of the new write: {content!r}"
        )
        own = content.partition("Source: ")[2].strip()
        for address in (_ARTICLE_URL, *_OTHER_SOURCES):
            if address != own:
                assert address not in content, (
                    f"a seeded entry must carry only its own source: {content!r}"
                )
    page_text = _body(voice)
    assert page_text.count(_DATUM) == 1, (
        f"the datum must appear exactly once on the page: {page_text!r}"
    )


def _arm(voice: Voice) -> CycleArm:
    """One arm of the case: this arm's instruction and page, with the seeded entry
    condition under it."""
    return CycleArm(
        text=voice.program,
        seed=_seeder(voice),
        pages=[_page(voice)],
        world=_world(voice),
    )


_ARMS = [_arm(voice) for voice in VOICES]


# ── The claims ───────────────────────────────────────────────────────────────
#
# Every one of them reads END STATE: what the collection holds.  Nothing reads a
# tool name or an ordering — how the cycle resolved the address is route, measured
# as variance and never asserted.


def _datum_entries(sample: SampleObservation) -> list:
    """The held entries that carry the datum — the new write, and nothing else: the
    seeded entries are probed never to carry it."""
    return [entry for entry in sample.held if _DATUM in entry.content]


def _new_entry_stored(sample: SampleObservation, _world: World) -> Answer:
    """The collection holds a new entry carrying the datum.  Without this, the
    source claim below would pass vacuously on a cycle that stored nothing."""
    return bool(_datum_entries(sample)), (
        f"no entry carries the datum {_DATUM!r}; the collection holds "
        f"{[entry.key for entry in sample.held]}"
    )


def _source_is_the_page_it_read(sample: SampleObservation, _world: World) -> Answer:
    """Every entry that carries the datum carries THE PAGE's own address as its
    source — exactly, byte-for-byte — and none of the other addresses in the world:
    the seeded entries' own sources, and the article page's related links.

    This is the fact #3074 exists for.  The exact compare is what makes it a fact:
    a fuzzy host-prefix read once accepted an invented domain, which is the shape
    of the prod failure itself."""
    bad = []
    for entry in _datum_entries(sample):
        if _ARTICLE_URL not in entry.content:
            bad.append(f"{entry.key!r}: missing the page's own address")
        for address in _OTHER_SOURCES:
            if address in entry.content:
                bad.append(f"{entry.key!r}: carries {address}")
    return not bad, f"the datum entry's source is wrong: {'; '.join(bad)}"


def _seeded_entries_survive(sample: SampleObservation, _world: World) -> Answer:
    """The two earlier entries are exactly as the cycle found them — assert what
    survives."""
    held = {entry.key: entry.content for entry in sample.held}
    missing = [key for key, content in _SEED_ENTRIES if held.get(key) != content]
    return not missing, f"seeded entry/entries altered or gone: {missing}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_digest_entry_cites_the_page_it_read(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A digest job whose collection already holds sourced entries reads the page its
    prompt names and stores today's item — and the stored source is that page's own
    address, never another page's and never an invented one."""
    cohort = await collector_cycles_eval(
        case_id="entry-cites-the-page-it-read",
        behaviour=(
            "In a news-digest collector, when the job stores a new entry whose content "
            "came from the page the cycle read, the entry's stored source is that "
            "page's own address."
        ),
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_ARMS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the first run shows whether this reproduces
        family=_FAMILY,
    )
    # STORE
    cohort.claim(
        f"state: the collection holds a new entry carrying the datum ({_DATUM})",
        _new_entry_stored,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: that entry's source is the page the cycle read, never another page's",
        _source_is_the_page_it_read,
        SpecCategory.STORE,
    )
    cohort.claim("state: the seeded entries survive", _seeded_entries_survive, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)
