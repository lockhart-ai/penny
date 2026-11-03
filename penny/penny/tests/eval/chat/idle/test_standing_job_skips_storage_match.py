"""A new per-city job must never be redirected onto a storage-only trip collection whose
name it merely contains, nor conflated with a different, unrelated collection the user
asks to bring back for one of those same cities (#2893).

The contract is `docs/eval-case-design.md`.

**The prod shape, lifted verbatim and anonymised, rebuilt from a replay (#2900).** Two
earlier passes at this case undershot the real trigger:

- #2898's own case seeded a world where every city's job-name candidate shared the trip
  collection's own literal name — the model never picked a name that contained it, so the
  case passed 15/15 without reproducing anything.
- A first rebuild (verbatim-replaying the live turn, #2900) swapped in a MINIMAL world —
  one clean archived collection for the history city, nothing else — and the model
  reliably extended ITS naming convention (``<city>_notes``) to the other two cities,
  never touching the trip collection at all. Also 15/15, still not reproducing.

The divergence was the world itself: the real self-state the live turn answered against
was crowded and messy, not minimal — a long store map of unrelated interests, and
TWO inconsistently-named archived collections for the history city (no single clean
template to extend), not one. This case is rebuilt a third time to mirror that turn's
rendered prompt one-to-one: every collection in the store map (name, rough entry count,
archived-or-active, job-or-storage), the nine conversation turns in its history window in
the same order, and the live ask in the same structure — every real name swapped 1:1 for a
different real one of the same kind.

**The world.**

- A storage-only trip collection, auto-created earlier in the conversation to hold the
  trip's itinerary and the detailed booking breakdown — entries only, no `prompt`, no
  `schedule` — named for the event and the year (``newyear_2028``).
- Two ARCHIVED, inconsistently-named collections left over from an earlier visit to one of
  the trip's cities: one storage-only (no job ever configured on it), one still carrying
  its own old job (a prompt and a schedule) from before it was archived. Their names share
  the city but nothing else — no template to extend.
- A long store map of the user's other, unrelated standing interests and jobs — crafts,
  games, collecting, outdoor hobbies — different topics from the real ones (never a
  paraphrase of what any of them actually track), in rough proportion to the real one's
  size and shape, so the live turn answers against a crowded self-state, not a
  two-collection stand-in.
- Nine conversation turns in the history window: four turns of unrelated small talk, then
  the trip being announced, a memory of the earlier visit to the history city, the dates
  that created the trip collection, the detailed booking breakdown, and a recap of the
  dates — the same count and order as the real turn's history.

**The ask.** A research job for each of three cities on the trip, with a bulleted list of
topics, one extra topic for the third city specifically, pointing back at "the existing"
collection for the history city while asking for two brand-new jobs for the other two,
closing on a cadence note and a notify-off aside — the same structure as the real ask.

Replaying the real turn 15 times (verbatim, before this rebuild) showed the model reliably
naming the two brand-new jobs ``<city>_<event>_<year>`` — which, once the trailing year is
stripped, token-contains the trip collection's own single remaining token. Under the
pre-#2893 rule, a single-token containment was enough to read a longer name as the same
job, which is the exact redirect that overwrote the itinerary collection's settings with a
city's prompt and schedule in production.

Two of `collection_configure`'s duplicate-job paths (`similarity/dedup.py` `is_same_job`)
are in play here, and this case's claims hold to only one of them. The trip collection's
containment is the one asserted. The OTHER path — a bare city word landing on one of the
two archived history-city collections by an exact-name or containment match — is the
#1567 tombstone refusal, a different, already-covered branch; this case leaves what the
turn does with either archived collection as a route, not a claim. The fix (#2893/#2898)
is two deterministic layers: the duplicate check only ever compares against a collection
that already has a job (an archived match is refused outright, named, rather than silently
updated), and a name that adds a real distinguishing token is read as a different job
rather than noise.

**The claims.** PRESERVATION — the trip collection (``newyear_2028``) still holds every
itinerary and booking entry it was seeded with. STATE — it still has no `prompt` and no
`schedule`: whatever jobs the turn stood up for the three cities, none of them landed ON
this collection.

**What is NOT claimed, and why.** Which (if either) of the two archived history-city
collections the turn resurrects, what it calls the new jobs, and whether every city gets
its own job are all ROUTES — the model is free to ask a clarifying question, create fewer
jobs than asked, or phrase the new jobs however it likes — so the tool sequence and the
reply are measured, never asserted (`docs/principles.md` §4.1, §4.3). This case's only
facts are about the ONE collection whose name the new jobs are liable to contain.

The host is real (a New Year trip across Ljubljana, Kraków and Bratislava, flying from
Calgary); nothing here is drawn from any real person's own trip or any real person's name
— the repo is public.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "standing-collection-identity"

_CASE_ID = "standing-job-skips-storage-match"

_USER_NAME = "Dagny"
_USER_LOCATION = "Oslo, Norway"
_USER_TIMEZONE = "Europe/Oslo"
_COMPANION = "Lee"

# ── The trip collection — the one claim this case makes ───────────────────────

_TRIP_NAME = "newyear_2028"
_TRIP_DESCRIPTION = (
    "Itinerary for the New Year 2028 trip: flights, hotels, and the train legs between cities."
)
_TRIP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("leg_out", "Calgary to Ljubljana · Lufthansa, overnight, 9h40, Mon 28 Dec."),
    ("leg_1", "Ljubljana to Kraków · EuroCity Tauern, 5h20, Wed 30 Dec."),
    ("leg_2", "Kraków to Bratislava · RegioJet, 4h10, Sat 2 Jan."),
    ("leg_3", "Bratislava to Ljubljana · EuroCity, 6h50, Tue 5 Jan."),
    ("leg_home", "Ljubljana to Calgary · Lufthansa, 10h10, Fri 8 Jan."),
    ("hotel_ljubljana_1", "Hotel Cubo, Ljubljana · 28-30 Dec, via Expedia, ref 81090891072613."),
    ("hotel_krakow", "Hotel Stary, Kraków · 30 Dec-2 Jan, via Expedia, ref 81090892947285."),
    (
        "dinner_krakow",
        "New Year's Eve dinner, Wierzynek Restaurant, Kraków · 31 Dec 7:30pm, "
        "under Dagny Fossum · 2 guests.",
    ),
    (
        "hotel_bratislava",
        "Falkensteiner Hotel, Bratislava · 2-5 Jan, via Expedia, ref 81090893218467.",
    ),
    (
        "hotel_ljubljana_2",
        "InterContinental, Ljubljana · 5-8 Jan, via Expedia, ref 81090893835695.",
    ),
)
_HELD_KEYS = tuple(key for key, _content in _TRIP_ENTRIES)

# ── Two inconsistently-named archived collections for the history city ────────
#
# No single clean per-city template to extend — one is storage-only, the other still
# carries its own old job — the real shape the prod turn answered against, and what a
# MINIMAL one-clean-collection world (the prior rebuild) got wrong.

_ARCHIVE_STORAGE_NAME = "krakow_highlights"
_ARCHIVE_STORAGE_DESCRIPTION = "Notes from an earlier Kraków visit: sights and highlights."
_ARCHIVE_STORAGE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("wawel_castle", "Wawel Castle — the hilltop courtyard and the cathedral crypt."),
    ("kazimierz_walk", "Kazimierz district — the old synagogues and the riverside path."),
)

_ARCHIVE_JOB_NAME = "krakow_issues"
_ARCHIVE_JOB_DESCRIPTION = "Open questions and tips still being sorted for Kraków."
_ARCHIVE_JOB_PROMPT = (
    "Search for good food and local market finds in Kraków. Note what's worth checking out and why."
)
_ARCHIVE_JOB_SCHEDULE = "FREQ=DAILY;BYHOUR=8"
_ARCHIVE_JOB_ENTRIES: tuple[tuple[str, str], ...] = (
    ("stary_kleparz", "Stary Kleparz market — a pierogi stall near the north entrance."),
    ("cafe_szafe", "Café Szafé — a quiet spot for coffee and cake, open late."),
)

# ── The rest of the store map — unrelated standing interests and jobs ─────────
#
# Real production self-state carries a long store map alongside the trip; a two-collection
# world answers a different, easier question than the one the live turn actually faced.
# Rough entry counts (not exact) — the point is the map's size and shape, not its digits.


class _BackgroundCollection(NamedTuple):
    name: str
    description: str
    count: int
    prompt: str | None = None
    schedule: str | None = None


_BACKGROUND_COLLECTIONS: tuple[_BackgroundCollection, ...] = (
    _BackgroundCollection(
        "vintage_postcard_collection", "Vintage postcards, by era and region.", 8
    ),
    _BackgroundCollection("houseplant_care_notes", "Houseplant varieties and care tips.", 20),
    _BackgroundCollection(
        "daily_trivia_facts",
        "One surprising trivia fact a day, picked from a random category.",
        12,
        prompt=(
            "Pick a trivia category at random (history, geography, food, or language) and "
            "find one surprising fact. Save it with a descriptive key."
        ),
        schedule="FREQ=DAILY;BYHOUR=9",
    ),
    _BackgroundCollection(
        "board_game_picks",
        "Scout for a new board game (party, strategy, or cooperative) to try.",
        15,
        prompt=(
            "Pick a board game genre at random: party, strategy, or cooperative. Find one "
            "highly-rated pick in that genre."
        ),
        schedule="FREQ=DAILY;BYHOUR=21",
    ),
    _BackgroundCollection(
        "daily_word_puzzle",
        "One word puzzle recommendation a day, tailored to a random difficulty.",
        1,
        prompt="Pick a difficulty at random and recommend one word puzzle tailored to it.",
        schedule="FREQ=DAILY;BYHOUR=19",
    ),
    _BackgroundCollection("knitting_patterns", "Knitting patterns and techniques.", 20),
    _BackgroundCollection(
        "vintage_typewriters", "Vintage typewriter models, repair tips, and history.", 20
    ),
    _BackgroundCollection(
        "home_brewing_rd",
        "Research for a home brewing setup.",
        20,
        prompt=(
            "Research home brewing equipment and recipes, focusing on fermentation "
            "temperature control and bottle conditioning."
        ),
        schedule="FREQ=HOURLY;INTERVAL=12",
    ),
    _BackgroundCollection("pottery_techniques", "Pottery and ceramics techniques.", 15),
    _BackgroundCollection(
        "penny_wonders_open", "Daily aggregator of odd facts and curiosities.", 10
    ),
    _BackgroundCollection("hiking_trail_notes", "Hiking trail notes and conditions.", 15),
    _BackgroundCollection("chess_openings", "Chess openings, puzzles, and annotated games.", 12),
    _BackgroundCollection("cheese_and_wine_pairings", "Cheese and wine pairing notes.", 15),
    _BackgroundCollection("birdwatching_sightings", "Birdwatching sightings and field notes.", 15),
    _BackgroundCollection(
        "antique_furniture_finds", "Antique furniture styles, eras, and restoration notes.", 20
    ),
    _BackgroundCollection(
        "stand_up_comedy_specials",
        "Stand-up comedy specials worth watching, by comedian and special.",
        20,
    ),
    _BackgroundCollection(
        "penny_reflections", "A journal for Penny to record reflections on her own runs.", 7
    ),
    _BackgroundCollection(
        "journal", "A rolling log of major life milestones extracted from user messages.", 6
    ),
    _BackgroundCollection("user_profile", "Auto-created to hold durable user facts.", 10),
)

_ARCHIVED_UNRELATED: tuple[_BackgroundCollection, ...] = (
    _BackgroundCollection(
        "neighborhood_book_club_notes", "Notes from a neighborhood book club that wrapped up.", 4
    ),
    _BackgroundCollection(
        "old_phone_upgrade_research", "Notes comparing phone upgrade options.", 3
    ),
)

_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=())


def _seed_background(db: Database) -> None:
    """The rest of the store map: unrelated standing interests and jobs, at roughly the
    size and shape production's self-state carried — so the live turn answers against a
    crowded map, not a two-collection stand-in."""
    for item in _BACKGROUND_COLLECTIONS:
        if db.memories.get(item.name) is not None:
            continue  # journal/user_profile/penny_reflections may already be migration-seeded
        run_id = seeded_run_id(f"{item.name}-created")
        db.memories.create_collection(
            item.name,
            item.description,
            extraction_prompt=item.prompt,
            schedule=item.schedule,
            created_by_run_id=run_id,
        )
        entries = [
            EntryInput(key=f"{item.name}_{i}", content=f"{item.description} — note {i + 1}.")
            for i in range(item.count)
        ]
        if entries:
            require_memory(db, item.name).write_rule_decided(entries, author="chat", run_id=run_id)


def _seed_archived_unrelated(db: Database) -> None:
    """A couple of archived, unrelated collections with nothing to do with the trip — part
    of the real self-state's archived line, not part of any claim."""
    for item in _ARCHIVED_UNRELATED:
        run_id = seeded_run_id(f"{item.name}-created")
        db.memories.create_collection(
            item.name, item.description, archived=True, created_by_run_id=run_id
        )
        entries = [
            EntryInput(key=f"{item.name}_{i}", content=f"{item.description} — note {i + 1}.")
            for i in range(item.count)
        ]
        if entries:
            require_memory(db, item.name).write_rule_decided(entries, author="chat", run_id=run_id)


def _seed_history_city_archives(db: Database) -> None:
    """The two inconsistently-named archived collections for Kraków — one storage-only,
    one still carrying its own old job — the real shape, no single clean template."""
    storage_run = seeded_run_id("krakow-highlights-created")
    db.memories.create_collection(
        _ARCHIVE_STORAGE_NAME,
        _ARCHIVE_STORAGE_DESCRIPTION,
        archived=True,
        created_by_run_id=storage_run,
    )
    require_memory(db, _ARCHIVE_STORAGE_NAME).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _ARCHIVE_STORAGE_ENTRIES],
        author="chat",
        run_id=storage_run,
    )
    job_run = seeded_run_id("krakow-issues-created")
    db.memories.create_collection(
        _ARCHIVE_JOB_NAME,
        _ARCHIVE_JOB_DESCRIPTION,
        archived=True,
        extraction_prompt=_ARCHIVE_JOB_PROMPT,
        schedule=_ARCHIVE_JOB_SCHEDULE,
        created_by_run_id=job_run,
    )
    require_memory(db, _ARCHIVE_JOB_NAME).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _ARCHIVE_JOB_ENTRIES],
        author="chat",
        run_id=job_run,
    )


def _seed_trip_collection(db: Database) -> str:
    """The trip's storage-only itinerary + booking collection — the one this case's claims
    read. Returns the run id its creation is stamped under, for the history turn that
    narrates having just created it."""
    trip_run = seeded_run_id("trip-itinerary-created")
    db.memories.create_collection(_TRIP_NAME, _TRIP_DESCRIPTION, created_by_run_id=trip_run)
    require_memory(db, _TRIP_NAME).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _TRIP_ENTRIES],
        author="chat",
        run_id=trip_run,
    )
    return trip_run


# ── The nine conversation turns in the history window ─────────────────────────
#
# Four turns of unrelated small talk, then the trip being announced, a memory of the
# earlier Kraków visit, the dates that created the trip collection, the detailed booking
# breakdown, and a recap of the dates — the same count and order as the real turn's
# history, each paraphrased with every real name swapped for a different real one.

_HISTORY_TURNS: tuple[tuple[str, str], ...] = (
    (
        "hi penny 👋",
        "Hey! 👋 I was actually just looking into whether Half-Life 3 will have cloud "
        "streaming at launch — short answer, it doesn't look like it. It's console-first, "
        "so GeForce NOW and the PC-reliant cloud services are a no for day one.",
    ),
    (
        "wait really, i thought it was day one on steam",
        "I get why you'd think that! But the latest reporting has it launching on "
        "PlayStation and Xbox first, with no PC version yet — so Steam's out at launch too.",
    ),
    (
        "ugh that's annoying, guess i'll wait",
        "Yeah, it's a bummer wait. I'll keep an eye out in case that changes.",
    ),
    (
        f"anyway, in other news, {_COMPANION} and i have a trip coming up so i have some "
        "exciting new jobs for you",
        "Ooh, I'm all ears! Whenever you're ready, lay out the details. 📝✨",
    ),
    (
        f"yesterday {_COMPANION} and i planned our new year's trip. ljubljana, krakow, "
        "bratislava. which means i have research work for you to do! i'll explain in more "
        "detail before i tell you what i need",
        "That sounds incredible! 🎉 Ljubljana, Kraków, and Bratislava for New Year's — "
        "what a trio. I'm ready whenever you want to lay out the details.",
    ),
    (
        f"okay so you remember when {_COMPANION} and i last went to krakow and i had you "
        "find like restaurants and cocktail bars and museums and such?",
        "I do remember! 🕵️‍♀️ You two were after the moody, atmospheric stuff — cool "
        "restaurants, old Gothic sights, and especially that **Crypt & Candle** cocktail "
        "bar in Kazimierz. Want me to do something similar for this Ljubljana, Kraków, "
        "and Bratislava loop?",
    ),
    (
        "perfect exactly. so first here's the dates:\n\n"
        "Mon 28 Dec\nCalgary to Ljubljana · Lufthansa, overnight\n9h 40\n"
        "Wed 30 Dec\nLjubljana to Kraków · EuroCity Tauern\n5h 20\n"
        "Sat 2 Jan\nKraków to Bratislava · RegioJet\n4h 10\n"
        "Tue 5 Jan\nBratislava to Ljubljana · EuroCity\n6h 50\n"
        "Fri 8 Jan\nLjubljana to Calgary · Lufthansa\n10h 10\n\n"
        "start a newyear_2028 collection and start putting info in there, no job for it yet "
        "i'll just start giving you the high level trip info for you to remember",
        f"I created the `{_TRIP_NAME}` collection and saved your itinerary in it. Lots of "
        "scenic train rides in between! 🚂 Ready whenever you want to drop more details.",
    ),
    (
        "here's a more detailed breakdown, remember this:\n\n"
        "Booking\tDates\tWith\tReference\n"
        "Flights, Calgary and Ljubljana\t28 Dec · 8 Jan\tLufthansa\t51KPXOCOPY\n"
        "Hotel Cubo, Ljubljana\t28–30 Dec\tExpedia\t81090891072613COPY\n"
        "Train, Ljubljana to Kraków\t30 Dec\tEuroCity · code 7Y3RFE\t304459 381 00COPY\n"
        "Hotel Stary, Kraków\t30 Dec–2 Jan\tExpedia\t81090892947285COPY\n"
        "New Year's Eve dinner, Wierzynek Restaurant\t31 Dec · 7:30 pm\tThe restaurant\t"
        "Ms. Dagny Fossum · 2 guests\n"
        "Train, Kraków to Bratislava\t2 Jan\tRegioJet · code EAZ5YD\t304459 554 00COPY\n"
        "Falkensteiner Hotel, Bratislava\t2–5 Jan\tExpedia\t81090893218467COPY\n"
        "Train, Bratislava to Ljubljana\t5 Jan\tEuroCity · code FYI6AG\t304459 641 00COPY\n"
        "InterContinental, Ljubljana\t5–8 Jan\tExpedia\t81090893835695COPY",
        f"I saved the detailed booking breakdown to your `{_TRIP_NAME}` collection — flight, "
        "hotel, train, and dinner details all in there, so we can pull up a reference "
        "whenever we need it. That Wierzynek dinner on New Year's Eve sounds lovely!",
    ),
    (
        "okay so repeat back to me the high level dates for each city",
        "Of course! Here's the high-level breakdown:\n\n"
        "* **Ljubljana**: Dec 28–30 and Jan 5–8\n"
        "* **Kraków**: Dec 30–Jan 2\n"
        "* **Bratislava**: Jan 2–5\n\n"
        "Does that match up with your notes?",
    ),
)

# ── The live ask — same structure, every fact preserved across all five wordings ──

_ASK = (
    "yep that looks right. so now i want a separate job for each city that finds, within "
    "those dates\n\n"
    "- new year's market info\n"
    "- restaurants to check out\n"
    "- cocktail bars to check out\n"
    "- museums to check out\n"
    "- new year's events like concerts or whatever to check out\n\n"
    "and in bratislava specifically\n\n"
    "- thermal spa recommendations to check out\n\n"
    "so i think you can resurrect the existing krakow collection and update its prompt\n\n"
    "and create two new ones for ljubljana and bratislava\n\n"
    "i think to help focus the search you'll want the first step to be choose one of those "
    "topics for the run, find stuff, remember it\n\n"
    "you don't need to notify me i'll ask when the trip is closer\n\n"
    "i'd say set them to run i guess twice a day, we've still got two months before the "
    "trip starts so that gives you lots of time to find things\n\n"
    "makes sense?"
)
_ALSO_PHRASED = (
    "okay that all checks out. so for each city — ljubljana, kraków, and bratislava — i "
    "want a job that looks for new year's market info, restaurants, cocktail bars, "
    "museums, and new year's events happening in those dates. for bratislava "
    "specifically, also look for thermal spa recommendations. since kraków already has "
    "an old collection, just bring that back and update its prompt instead of starting "
    "fresh — the other two (ljubljana and bratislava) should be brand new. first step "
    "each run should be picking one of those topics, researching it, and saving what it "
    "finds. no need to notify me, i'll check in closer to the trip. twice a day sounds "
    "about right, we've got two months before we leave so there's plenty of time.",
    "new jobs needed, one per city (ljubljana, kraków, bratislava), for:\n"
    "- new year's market info\n- restaurants\n- cocktail bars\n- museums\n"
    "- new year's events\n\n"
    "bratislava also needs: thermal spa recommendations.\n\n"
    "kraków: resurrect the old collection, update the prompt.\n"
    "ljubljana + bratislava: brand new collections.\n\n"
    "each run: pick one topic, search it, save it. twice a day. no notifications needed "
    "— i'll ask closer to the trip.",
    "could you set up a research job for each of the three cities on this trip? within "
    "the travel dates i want each one checking for new year's market info, restaurants "
    "to check out, cocktail bars, museums, and any new year's events going on. "
    "bratislava should also look for thermal spa recommendations specifically. for "
    "kraków, since there's already an old collection, bring that one back and refresh "
    "its prompt rather than making a new one — ljubljana and bratislava need brand new "
    "ones though. have each pick one topic per run to focus the search, and don't worry "
    "about notifying me, i'll check back closer to the trip. twice a day feels right "
    "given we've still got two months to go.",
    "alright so now that the dates are locked in — can you get me a job going for each "
    "city? looking for new year's market stuff, good restaurants, cocktail bars, "
    "museums, and any new year's events on those dates. oh and for bratislava "
    "specifically, throw in thermal spa spots too. kraków's got that old collection "
    "already so just resurrect it and update the prompt — ljubljana and bratislava are "
    "fresh starts though. first move each run: pick a topic, dig in, save it. no "
    "notifications needed, i'll ask when we're closer. twice a day's probably good, "
    "we've got two months of runway.",
)

_BEHAVIOUR = (
    "In chat, when the user asks for a new per-city research job on a trip that already has "
    "a storage-only collection named for it, Penny creates the new jobs without turning that "
    "storage collection into one of them."
)


def _seed_world(db: Database) -> None:
    """Lay the whole recorded world down: the user, the trip collection, the two
    inconsistently-named archived Kraków collections, the long store map of unrelated
    standing interests, and the nine-turn recorded conversation that leads into the live
    ask — every write through the stores' own chokepoints, so the rendered prompt matches
    the one production rendered, in size and shape, not just in its two headline
    collections."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-02-10",
    )
    _seed_background(db)
    _seed_archived_unrelated(db)
    _seed_history_city_archives(db)
    _seed_trip_collection(db)
    _log_history(db)


def _log_history(db: Database) -> None:
    """The nine recorded turns, in send order — plain message-log rows, so the chat
    agent's history window renders them exactly as it would a real prior conversation."""
    first_ask_id = None
    for index, (user_text, penny_text) in enumerate(_HISTORY_TURNS):
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
        )
        if index == 0:
            first_ask_id = ask_id
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            penny_text,
            parent_id=ask_id,
        )
    assert first_ask_id is not None


def _premise() -> Preparer:
    """The world this case is answered in, asserted out loud before the turn runs: the trip
    collection holds exactly its ten seeded itinerary/booking entries and carries no prompt
    and no schedule — plain storage, not yet a job — while both Kraków collections sit
    archived, one storage-only, one still carrying its own old job."""

    def probe(penny: Penny) -> None:
        db = penny.db
        held = sorted(entry.key or "" for entry in require_memory(db, _TRIP_NAME).read_all())
        expected = sorted(_HELD_KEYS)
        assert held == expected, f"{_CASE_ID}: {_TRIP_NAME} must hold {expected}, it holds {held}"
        row = db.memories.get(_TRIP_NAME)
        assert row is not None and row.extraction_prompt is None and row.schedule is None, (
            f"{_CASE_ID}: {_TRIP_NAME} must start as plain storage (no prompt, no schedule)"
        )
        storage_row = db.memories.get(_ARCHIVE_STORAGE_NAME)
        assert storage_row is not None and storage_row.archived, (
            f"{_CASE_ID}: {_ARCHIVE_STORAGE_NAME} must start archived"
        )
        job_row = db.memories.get(_ARCHIVE_JOB_NAME)
        assert job_row is not None and job_row.archived, (
            f"{_CASE_ID}: {_ARCHIVE_JOB_NAME} must start archived"
        )

    return probe


# ── The claims, as pure functions over one sample ─────────────────────────────


def _trip_row(sample: SampleObservation) -> MechanismRecord | None:
    """The trip collection's own registry row as the sample left it, or ``None`` if the turn
    removed it from the registry altogether."""
    return next((one for one in sample.mechanisms if one.name == _TRIP_NAME), None)


def _the_trip_collection_keeps_its_entries(sample: SampleObservation, _world: World) -> Answer:
    """The trip collection still holds exactly its ten seeded itinerary/booking entries —
    nothing the turn did to set up per-city jobs may drop or rewrite what it already held."""
    held = {entry.key for entry in sample.held if entry.collection == _TRIP_NAME}
    expected = set(_HELD_KEYS)
    return held == expected, f"{_TRIP_NAME} holds {sorted(held)}"


def _the_trip_collection_still_has_no_job(sample: SampleObservation, _world: World) -> Answer:
    """The trip collection still carries no `prompt` and no `schedule` — whatever jobs the
    turn stood up for the three cities, none of them landed ON this collection.  A violating
    sample is nameable: one that read a city's longer job name as the same job as this one
    (its tokens containing the trip collection's own) and wrote that city's prompt and
    schedule onto it."""
    row = _trip_row(sample)
    if row is None:
        return False, f"{_TRIP_NAME!r} is no longer in the registry"
    has_job = row.program is not None or row.schedule is not None
    return not has_job, f"program={'set' if row.program else 'empty'}, schedule={row.schedule!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_new_citys_job_never_lands_on_the_trips_storage_collection(
    chat_eval: ChatEval, model: str
) -> None:
    """A crowded store map, a storage-only trip collection, two inconsistently-named
    archived collections for the history city, and an ask for a separate research job per
    city on that same trip — the real shape that let a new job's name read as the same job
    as the collection it was never meant to touch."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE
    cohort.claim(
        "state: the trip collection still holds every seeded itinerary/booking entry",
        _the_trip_collection_keeps_its_entries,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the trip collection still has no prompt and no schedule",
        _the_trip_collection_still_has_no_job,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — how many jobs the turn creates, and what
    # it calls them, are routes.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
