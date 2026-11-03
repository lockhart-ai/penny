"""A collector entry keeps its source link even when the page shows it as a numbered
handle, not an address (meta #2907).

**The shape is a real production cycle, genericised.**  A news-watching collector read a
page carrying several headlines, each a bare numbered handle (``[46]``, ``[103]``, …) rather
than a spelled-out address — the handle scheme every browse read renders through
(``penny/penny/tools/link_handles.py``, #2717).  Asked to file the single most significant
story with its source, the model's own thinking read the handles as internal bookkeeping
rather than citable addresses:

> The "links" provided in the browse result [46] [160] etc are internal to the tool's
> context, not real URLs. I will proceed with the info I have. I'll omit the URL field to
> avoid inventing one.

> It seems the tool is giving me the index numbers. I'll try to "open" [46] again to see
> if there's a URL inside the content I missed.

A verbatim replay of that production row (15 samples, CoreWeave, #2907) reproduced the drop
in 9 of them — every one of the 15 completed samples' thinking shows the same "these are just
internal identifiers, not real URLs" reasoning, whether the sample went on to write the entry
with no source or to re-browse hunting for an address that was already in front of it.  That
replay sent the row's own recorded bytes, from BEFORE #2832/#2835 changed how a handle
renders (a bare number now carries its domain, or its full address where the link IS the
answer).  This case re-renders the same shape through TODAY's code — a real collector cycle,
not a replayed turn — to measure whether the presentation fix also covers the WRITE decision,
which neither #2832's nor #2835's own case touched (both measure a CHAT reply citing an
ALREADY-STORED link, never a collector choosing whether to keep a freshly-read one).  The
specific story and the collection it belongs to are genericised throughout — see below.

**One case, one behaviour: the write, not the read.**  The collector's own runtime rule
("Cite only what you actually browsed this cycle. Never invent a URL to populate a
'Source:' field — if no real source was fetched, omit the field",
``penny/penny/agents/collector.py``) licenses dropping the field when nothing was really
read.  The failure this case targets is the model applying that license to a page it DID
read, because the address arrived as a number rather than as text — an over-read of "don't
invent" into "don't trust what you were handed."

**The page is read directly, not through a search** — a direct URL named in the job's own
prompt, the same shape ``collector/test_collector_honesty.py`` already uses, so the only
thing in play is the handle-trust decision itself, not the separate picker/ranking
machinery a text search would add.  It carries SEVEN links — the one the ask names and six
unrelated decoys — real astronomy and space-science domains (NOIRLab/Rubin, ESA, NASA, NRAO,
Keck, the Space Telescope Science Institute, ESO), because the real production page named
real domains too and a `.test` placeholder would make "it carries a real domain" trivially
true of every handle rather than a fact a claim can fail to find.  The specific story is
swapped for a different real one (Vera Rubin Observatory's first-light asteroid count,
rather than the one actually recorded), per the project's replay-to-case rule: real proper
nouns swap for other real ones of the same kind, never an invented placeholder.

**The arms are five wordings of one instruction, over five voices of one page.**  The target
headline, its address and its one-line fact are byte-identical across all five — what
varies is the intro and which order the decoys are listed in, same as the watch cases'
five prose variants of one page.

**What is asserted.**  The collection already holds one earlier, differently-sourced entry
(matching the real run, which checked its own existing entry on a different topic before
writing); after the cycle it holds that entry UNCHANGED plus exactly one new one, and the
new entry carries the REAL address of whichever story it is about — never the fixture's
own favourite. Which of the page's seven stories counts as "most significant" is the
model's own judgement call, so picking a different, equally real one is variance, not a
miss; the claim detects the story an entry names by its own distinguishing words and
checks THAT story's own address, so a correctly-sourced entry about a decoy passes
exactly as a correctly-sourced entry about the intended target does. Whether the model
wrote the handle number (which `collection_write` expands before it reaches the store)
or the bare address itself, how many tool calls it took, and whether it re-browsed
hunting for "the real URL" first are route, and are measured as variance, never
claimed.

Report-only (``min_pass_rate=None``) on this run: this is the first run on today's
presentation, to show whether the drop still happens, not to grade it.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Seeder
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    StoredEntry,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-write-keeps-numbered-link"

# The one collection every arm's job is configured on.  Invented, like every collector
# case's own container name — a job's destination is never a real proper noun.
_CONTAINER = "orbital-news-watch"
_CONTAINER_DESCRIPTION = (
    "Notable astronomy and space-science discoveries, one per cycle, each with its "
    "headline, a short summary, and its source."
)

# The page the job's prompt names directly — a real astronomy-press aggregator's own
# domain, read as a page rather than searched for, so the only thing in play is the
# handle-trust decision this case measures, not a separate picker round (#2718).
_SOURCE = "https://skyandtelescope.org/observing/this-week-in-space"
_MATCH = "skyandtelescope"

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=DAILY"

_PROGRAM_CALLS = ("browse", "collection_write")

# ── the entry already sitting in the collection ───────────────────────────────
#
# A different, already-sourced story — the real run checked its own existing entry
# (a different topic) before writing the new one, so this collection is never empty
# when its cycle starts.  Real domain (nasa.gov), different real story.
_EXISTING_KEY = "lunar-ice-2026-09-12"
_EXISTING_CONTENT = (
    "Headline: Ice Deposits Confirmed in Permanently Shadowed Lunar Craters\n"
    "Summary: Radar soundings from an orbiting probe confirm thick water-ice deposits "
    "inside several permanently shadowed craters near the Moon's south pole.\n"
    "Source: https://www.nasa.gov/lunar-ice-deposits-confirmed"
)

# ── the target story, byte-identical across every voice ──────────────────────
#
# The story every voice puts FIRST on the page — the one this fixture considers the
# single most significant — and the one-line fact every claim below names, constant so
# a claim can say what the store holds, not just that something is there.  A different
# real discovery than the one recorded in production (#2907): per the replay-to-case
# rule, a real proper noun swaps for another real one of the same kind, never an
# invented placeholder.  Its own real address lives in ``_STORIES`` below, alongside
# every other story the page carries — "most significant" is the model's own
# judgement call, so no claim treats this one as privileged over a decoy it might
# reasonably pick instead.
_TARGET_HEADLINE = "Vera C. Rubin Observatory's First Look Reveals Thousands of New Asteroids"
_TARGET_URL = "https://rubinobservatory.org/news/first-look-reveals-asteroids"
_TARGET_FACT = (
    "In its first ten hours of science observations, the observatory's camera charted "
    "over two thousand previously unknown asteroids in the solar system, none of them a "
    "threat to Earth."
)
_TARGET_BLOCK = f"{_TARGET_HEADLINE}\n{_TARGET_URL}\n{_TARGET_FACT}\n"

# ── six decoy headlines, real domains, none of them the ask's answer ─────────
#
# Real shape and size (#2907's replayed row carried dozens of bare-numbered headlines in
# one listing): enough competing handles that picking the right one is a real decision,
# not a one-item page where "the" link is the only one on it.  Each decoy keeps its own
# real domain so every handle this page produces carries a genuine address — the point
# this case measures is whether a real, legible handle is still trusted at WRITE time.
_DECOY_EUCLID = (
    "Euclid Space Telescope Spots a Rare Einstein Ring in an Old Hubble Image\n"
    "https://esa.int/euclid-einstein-ring\n"
    "The gravitational-lensing ring was hiding in an archival image for years before a "
    "closer look turned it up.\n"
)
_DECOY_EUROPA = (
    "NASA's Europa Clipper Returns Its First Images of Jupiter\n"
    "https://www.nasa.gov/europa-clipper-first-images\n"
    "The spacecraft's cameras captured Jupiter and its moons during a instrument "
    "checkout pass on its way to Europa.\n"
)
_DECOY_FRB = (
    "Astronomers Detect a Record-Breaking Fast Radio Burst\n"
    "https://public.nrao.edu/record-frb\n"
    "The millisecond-long burst released as much energy as the Sun emits in several "
    "decades.\n"
)
_DECOY_DWARF_PLANET = (
    "A New Dwarf Planet Candidate Found Beyond Neptune\n"
    "https://keckobservatory.org/new-dwarf-planet-candidate\n"
    "Follow-up observations are needed to confirm the object's size and orbit.\n"
)
_DECOY_HUBBLE = (
    "Hubble Captures a Rare Triple Shadow Transit on Jupiter\n"
    "https://hubblesite.org/triple-shadow-transit\n"
    "Three of Jupiter's largest moons cast their shadows on the planet at once, a "
    "configuration that recurs only every few years.\n"
)
_DECOY_ESO_FRB = (
    "Astronomers Measure the Most Distant Fast Radio Burst Yet\n"
    "https://www.eso.org/distant-frb\n"
    "The signal travelled for roughly eight billion years before reaching Earth.\n"
)
_ALL_DECOYS = (
    _DECOY_EUCLID,
    _DECOY_EUROPA,
    _DECOY_FRB,
    _DECOY_DWARF_PLANET,
    _DECOY_HUBBLE,
    _DECOY_ESO_FRB,
)


class Story(NamedTuple):
    """One of the seven stories the page carries: the distinctive words that identify
    which one an entry is ABOUT, and the real address that story's own page holds.

    ``keyword`` is the smallest span unique to this story's headline (two FRB decoys
    share "Fast Radio Burst", so each keeps its OWN distinguishing words) — never a word
    the ask or the job's prompt also uses, so a match only ever means "this entry
    describes this story," not "this entry mentions the ask." ``url_fragment`` is the
    smallest span unique to this story's URL — the bare domain where no other story or
    seeded entry shares it (Rubin), the full distinguishing path where one does (Europa
    Clipper shares its domain, nasa.gov, with the SEEDED entry's own URL)."""

    keyword: str
    url_fragment: str


# Which story is "most significant" is the model's own judgement call — the job's ask
# says so — so WHICH of these seven an entry is about is never claimed; only that
# whichever one it names, that story's own real address came with it.  One entry per
# story, by its own distinguishing words, so detection never depends on the model having
# picked the one this fixture considers biggest.
_STORIES = (
    Story("Rubin", "rubinobservatory.org"),
    Story("Euclid", "euclid-einstein-ring"),
    Story("Europa Clipper", "europa-clipper-first-images"),
    Story("Record-Breaking Fast Radio Burst", "record-frb"),
    Story("Dwarf Planet", "new-dwarf-planet-candidate"),
    Story("Triple Shadow Transit", "triple-shadow-transit"),
    Story("Most Distant Fast Radio Burst", "distant-frb"),
)


class Voice(NamedTuple):
    """One arm: how the job's own instruction words what to find, and this voice's
    ordering/intro for the page that answers it.

    ``extract`` becomes the free-text ask inside the job's ``extraction_prompt`` — a
    collector's only natural-language surface (#2455 naive baseline, no skill/program
    rendering seam).  ``intro`` and ``order`` are this voice's own prose and decoy
    ordering; the target block is NOT a field here because it is byte-identical on every
    voice — see ``_TARGET_BLOCK`` above."""

    name: str
    extract: str
    intro: str
    order: tuple[str, ...]


VOICES = (
    Voice(
        "single-most-significant",
        "the single most significant astronomy or space-science story on the page",
        "This week's roundup, in no particular order of importance:\n",
        _ALL_DECOYS,
    ),
    Voice(
        "matters-most-right-now",
        "whichever astronomy or space-science story on the page matters most right now",
        "A quick look at what's new in space and astronomy this week:\n",
        (
            _ALL_DECOYS[2],
            _ALL_DECOYS[0],
            _ALL_DECOYS[4],
            _ALL_DECOYS[1],
            _ALL_DECOYS[5],
            _ALL_DECOYS[3],
        ),
    ),
    Voice(
        "biggest-headline",
        "the biggest astronomy or space-science headline on the page",
        "Here's what observatories and space agencies reported this week:\n",
        (
            _ALL_DECOYS[4],
            _ALL_DECOYS[5],
            _ALL_DECOYS[1],
            _ALL_DECOYS[3],
            _ALL_DECOYS[0],
            _ALL_DECOYS[2],
        ),
    ),
    Voice(
        "most-important-story",
        "the most important astronomy or space-science story here",
        "The week's astronomy and space-science headlines, gathered in one place:\n",
        (
            _ALL_DECOYS[1],
            _ALL_DECOYS[3],
            _ALL_DECOYS[5],
            _ALL_DECOYS[0],
            _ALL_DECOYS[2],
            _ALL_DECOYS[4],
        ),
    ),
    Voice(
        "top-story",
        "the top astronomy or space-science story on this page",
        "A running digest of this week's space and astronomy news:\n",
        (
            _ALL_DECOYS[3],
            _ALL_DECOYS[2],
            _ALL_DECOYS[0],
            _ALL_DECOYS[4],
            _ALL_DECOYS[5],
            _ALL_DECOYS[1],
        ),
    ),
)

_HEAD = f"Title: This Week in Space | Sky & Telescope\n{_SOURCE}\n\n"
_TAIL = (
    "\n"
    "More stories, archives and an observing calendar are at skyandtelescope.org. "
    "This roundup is updated every week.\n"
)

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when
# <X>, Penny <does Y>."
_BEHAVIOUR = (
    "In an astronomy-news collector, when the page it reads gives the chosen story's "
    "address only as a numbered handle rather than spelled-out text, Penny's entry still "
    "carries that source — the handle, or the address it expands to — rather than "
    "treating the handle's own number as something internal to drop."
)


def _page(voice: Voice) -> CannedPage:
    """This voice's whole page: the target block first (always — it is what the ask is
    answered by), then this voice's own ordering of the six decoys, under this voice's own
    intro.  The target's own text never moves; only what surrounds it does."""
    body = _HEAD + voice.intro + "\n" + _TARGET_BLOCK + "\n" + "\n".join(voice.order) + _TAIL
    return CannedPage(match=_MATCH, text=body)


def _prompt(voice: Voice) -> str:
    """The job's free-text instructions (#2455 naive baseline) — what a job's own
    ``extraction_prompt`` carries verbatim, with no skill/program rendering seam."""
    return (
        f"Read {_SOURCE} and find {voice.extract}. File it into '{_CONTAINER}', keyed by "
        "its own headline, with a one-line summary and its source."
    )


def _seeder(voice: Voice) -> Seeder:
    """The world this case's cycle starts in: a job already configured with its prompt and
    one earlier, differently-sourced entry already standing — then every claim that world
    makes, asserted out loud."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_prompt(voice),
            schedule=_SCHEDULE,
            notify=False,
        )
        require_memory(db, _CONTAINER).write_rule_decided(
            [EntryInput(key=_EXISTING_KEY, content=_EXISTING_CONTENT)],
            author="eval-seed",
            run_id="seed-orbital-news-watch",
        )
        _assert_the_world(db, voice)

    return seed


def _assert_the_world(db: Database, voice: Voice) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert not row.notify, (
        "this case claims what the cycle WROTE, not a notification — a notifying job "
        "would add the notify composer's own round to every sample for nothing this "
        "case reads"
    )
    prompt = row.extraction_prompt or ""
    assert _SOURCE in prompt, (
        f"the prompt must name {_SOURCE!r} — a cycle can only fetch the page something "
        f"it reads names.  Prompt: {prompt!r}"
    )
    assert voice.extract in prompt, (
        f"this arm's instruction must reach the model as {voice.extract!r}.  Prompt: {prompt!r}"
    )
    held = db.memory(_CONTAINER)
    assert held is not None
    entries = {e.key: e.content for e in held.read_all()}
    assert entries == {_EXISTING_KEY: _EXISTING_CONTENT}, (
        f"the collection must start holding exactly the one seeded entry — it holds {entries}"
    )


def _arm(voice: Voice) -> CycleArm:
    page = _page(voice)
    return CycleArm(
        text=voice.extract,
        seed=_seeder(voice),
        pages=[page],
        world=World(name=voice.name, pages=(page,), keeps=(), excludes=()),
    )


def _new_entry(sample: SampleObservation) -> StoredEntry | None:
    """The one entry this cycle added, beside the seeded one it must leave untouched — or
    ``None`` if it wrote nothing new."""
    return next((e for e in sample.held if e.key != _EXISTING_KEY), None)


def _existing_entry_survives(sample: SampleObservation, _world: World) -> Answer:
    """The earlier, differently-sourced entry is exactly as the cycle found it — this ask
    never named it, so it is read, not touched."""
    match = next((e for e in sample.held if e.key == _EXISTING_KEY), None)
    ok = match is not None and match.content == _EXISTING_CONTENT
    return ok, f"the seeded entry now reads: {match.content if match else '<missing>'!r}"


def _wrote_exactly_one_new_entry(sample: SampleObservation, _world: World) -> Answer:
    """The collection holds the seeded entry plus exactly one new one — the ask names one
    story, so a correct cycle adds one entry, not several and not none."""
    count = len(sample.held)
    return count == 2, f"the collection holds {count} entries: {[e.key for e in sample.held]}"


def _story_named_in(text: str) -> Story | None:
    """Which story ``text`` is about, by its own distinguishing words — or ``None`` when
    text names none of the seven.  Read over the WHOLE entry (key and content, #2017's
    "a fact in the key and a blurb in the body is a perfectly good way to store it"), so
    detection never misses a headline the model put in the key and left out of the body."""
    return next((story for story in _STORIES if story.keyword in text), None)


def _new_entry_carries_its_own_source(sample: SampleObservation, _world: World) -> Answer:
    """The claim this case exists for: the new entry carries the REAL address of
    whichever story it is about — never a fixed one.  Which story is "most
    significant" is the model's own judgement call (the ask says so), so a sample that
    picked a different, equally-real story off the same page is not a miss on that
    account; this claim reads the story the entry itself names and checks THAT story's
    own address, not the fixture's favourite.  ``collection_write`` expands a cited
    handle to its real URL before the row is stored
    (``penny/penny/tools/memory_tools.py``), so whether the model wrote the handle
    number or the bare address makes no difference to what lands here."""
    entry = _new_entry(sample)
    if entry is None:
        return False, "no new entry was written"
    story = _story_named_in(entry.text)
    if story is None:
        return False, f"the new entry names none of this case's seven stories: {entry.text!r}"
    hit = story.url_fragment in entry.text
    return hit, (
        f"expected {story.url_fragment!r} (the {story.keyword!r} story's own address) "
        f"in the new entry; it holds {entry.text!r}"
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_entry_keeps_the_numbered_links_source(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A page gives the chosen story's address only as a numbered handle; the entry the
    cycle writes for it still carries that source."""
    cohort = await collector_cycles_eval(
        case_id="entry-keeps-the-numbered-links-source",
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=[_arm(voice) for voice in VOICES],
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — shows whether today's presentation still drops it
        family=_FAMILY,
    )

    # STORE
    cohort.claim(
        "state: the earlier, differently-sourced entry survives untouched",
        _existing_entry_survives,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the collection holds exactly one new entry",
        _wrote_exactly_one_new_entry,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the new entry carries its own story's real address",
        _new_entry_carries_its_own_source,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)
