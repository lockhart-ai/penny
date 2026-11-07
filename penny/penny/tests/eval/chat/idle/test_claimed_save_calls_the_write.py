"""A reply that claims a save must be backed by the write that makes it true (#2915).

The contract is `docs/eval-case-design.md`.

**Seen (real use, gemma, replayed verbatim, genericised).**  Prompt-log row 2374 (the
same conversation `standing-job-skips-storage-match`, #2905, continues a few turns
earlier): the user had just sent a dates-and-cities message that stood up a storage-only
trip collection — Penny's own reply opened "I created the `<trip>` collection and saved
your itinerary in it." — then, in the very next turn, sent a second, much longer message:
a multi-line table of bookings and references to add to the same collection. Replaying
that row verbatim 15 times: in 9 of the 15 samples Penny's reply opens "I saved those
booking details to your `<trip>` collection," with EMPTY thinking and NO tool call at
all — nothing is written. The other 6 samples call `collection_write` as asked.

**Likely mechanism.**  The history Penny is shown carries only her own WORDS from the
turn before — "I created … and saved …" — never the `collection_write` call that backed
them (#2888). The very next turn asks for the same shape of thing (more details to add to
the same collection), and the model completes the pattern it was just shown: open with the
same sentence, skip the step that made it true the first time.

**The world.**  Mirrors row 2374's rendered prompt one-to-one, continuing
`standing-job-skips-storage-match`'s own swapped identity (Dagny Fossum, her companion
Lee, Oslo, flying out of Calgary, the trip collection `newyear_2028`, Ljubljana / Kraków /
Bratislava) rather than inventing a second cover for the same conversation:

- the same long store map of unrelated standing interests and jobs, and the same two
  inconsistently-named archived Kraków collections, both carried over unchanged from
  #2905 — the real self-state this turn answered against was exactly this crowded;
- the same seven-turn history window up to and including the dates turn that created the
  trip collection — but THIS time that turn is seeded as a REAL run: a logged
  `memory_access` write call backs its reply, through the same seeded-run primitives
  `test_history_tool_line.py` uses (`seeded_run_id`, `_log_chat_step`, `_seeded_response`,
  `_wire_tool_call`), not a plain logged pair with nothing behind it — the row Penny is
  shown is exactly the shape that teaches the "I created … and saved …" pattern;
- the live ask is the SECOND message from the same real turn pair: a multi-line table of
  bookings, each with its own invented reference, with the same structure (dates, carrier,
  a reference per row) as the recorded one.

**The claim.** STORE — afterwards the trip collection holds the booking details the user
just sent (three distinctive invented values, each a reference token appearing nowhere
else), and everything the store already held survives unchanged. Tool route and reply
wording are variance: whether she writes one entry or several, and whatever she says, are
not claimed.

REPORT-ONLY (`min_pass_rate=None`): the ceiling this run proposes is the code owner's to
accept once the numbers have been read. The host is real — the same swapped trip as
#2905 — continued here rather than re-invented, since it is the same recorded
conversation one turn later; nothing here is drawn from any real person's own trip or
booking.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.memory.objects import run_tool_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _log_chat_step, _seeded_response, _wire_tool_call
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "chat-false-save-claim"

_CASE_ID = "claimed-save-calls-the-write"

_USER_NAME = "Dagny"
_USER_LOCATION = "Oslo, Norway"
_USER_TIMEZONE = "Europe/Oslo"
_COMPANION = "Lee"

_TRIP_NAME = "newyear_2028"
_TRIP_RUN_NAME = "trip-itinerary-created"

# A single blob entry — the real row's trip collection carried exactly ONE entry at this
# point in the conversation (`auto-created to hold 'itinerary'`), not one entry per leg.
_ITINERARY_KEY = "itinerary"
_ITINERARY_CONTENT = (
    "Mon 28 Dec: Calgary to Ljubljana · Lufthansa, overnight, 9h40. "
    "Wed 30 Dec: Ljubljana to Kraków · EuroCity Tauern, 5h20. "
    "Sat 2 Jan: Kraków to Bratislava · RegioJet, 4h10. "
    "Tue 5 Jan: Bratislava to Ljubljana · EuroCity, 6h50. "
    "Fri 8 Jan: Ljubljana to Calgary · Lufthansa, 10h10."
)

# ── Two inconsistently-named archived collections for the history city ────────
#
# Carried over unchanged from #2905 (`standing-job-skips-storage-match`) — the same
# conversation's real self-state, not a stand-in built fresh for this case.

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
# Carried over unchanged from #2905 — the real production self-state this conversation
# answered against, size- and shape-matched, nouns swapped for unrelated synthetic ones.


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
    one still carrying its own old job — carried over from #2905's own real shape."""
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


def _seed_trip_collection(db: Database) -> None:
    """The trip's storage-only collection as row 2374 actually held it: ONE entry (the
    itinerary, as one blob — `auto-created to hold 'itinerary'`), created under the SAME
    run id the history turn below is logged under, so the self-state's recent activity
    and the history's reply name the same creation."""
    run_id = seeded_run_id(_TRIP_RUN_NAME)
    db.memories.create_collection(
        _TRIP_NAME, f"auto-created to hold '{_ITINERARY_KEY}'", created_by_run_id=run_id
    )
    require_memory(db, _TRIP_NAME).write_rule_decided(
        [EntryInput(key=_ITINERARY_KEY, content=_ITINERARY_CONTENT)],
        author="chat",
        run_id=run_id,
    )


# ── The seven-turn history window ─────────────────────────────────────────────
#
# Six plain exchanges (small talk, the trip announcement, the Kraków memory), then the
# seventh — the dates turn that created the trip collection — seeded as a REAL run: a
# logged `memory_access` write call backs its "I created … and saved …" reply, through the
# same primitives `test_history_tool_line.py` uses.  This is the row the live turn below
# is shown right before it: the exact shape the failure's own mechanism reproduces off.

_SMALL_TALK_TURNS: tuple[tuple[str, str], ...] = (
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
)

_DATES_USER = (
    "perfect exactly. so first here's the dates:\n\n"
    "Mon 28 Dec\nCalgary to Ljubljana · Lufthansa, overnight\n9h 40\n"
    "Wed 30 Dec\nLjubljana to Kraków · EuroCity Tauern\n5h 20\n"
    "Sat 2 Jan\nKraków to Bratislava · RegioJet\n4h 10\n"
    "Tue 5 Jan\nBratislava to Ljubljana · EuroCity\n6h 50\n"
    "Fri 8 Jan\nLjubljana to Calgary · Lufthansa\n10h 10\n\n"
    "start a newyear_2028 collection and start putting info in there, no job for it yet "
    "i'll just start giving you the high level trip info for you to remember"
)
_DATES_REPLY = (
    f"I created the `{_TRIP_NAME}` collection and saved your itinerary in it. Lots of "
    "scenic train rides in between! 🚂 Ready whenever you want to drop more details."
)


def _log_small_talk(db: Database) -> None:
    for user_text, reply in _SMALL_TALK_TURNS:
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
        )
        assert ask_id is not None, f"the seeded ask {user_text!r} failed to log"
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING, "penny", reply, parent_id=ask_id
        )


def _log_dates_turn_as_a_real_run(db: Database) -> None:
    """The dates/create-collection turn, backed by a REAL seeded run: the message log
    carries the reply under the run id the framework stamps, and that run's own
    prompt-log record carries the `memory_access` write call that backed it — the shape the
    next turn's false-save copies from, and the one plain message-log rows (#2905's own
    build of this same turn) cannot carry at all."""
    run_id = seeded_run_id(_TRIP_RUN_NAME)
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _DATES_USER
    )
    assert ask_id is not None, "the seeded dates turn failed to log"
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _DATES_REPLY,
        parent_id=ask_id,
        run_id=run_id,
    )
    response = _seeded_response(
        tool_calls=[
            _wire_tool_call(
                f"call-seeded-{_TRIP_RUN_NAME}",
                "memory_access",
                {
                    "target": _TRIP_NAME,
                    "action": "write",
                    "payload": {"entries": [{"key": _ITINERARY_KEY}]},
                },
            )
        ]
    )
    _log_chat_step(
        db,
        run_id=run_id,
        messages=[{"role": "user", "content": _DATES_USER}],
        response=response,
    )


def _seed_world(db: Database) -> None:
    """Lay the whole recorded world down: the user, the long store map of unrelated
    standing interests, the two archived Kraków collections, the trip collection (one
    entry, created under a real seeded run), and the seven-turn history that leads into
    the live ask — every write through the stores' own chokepoints."""
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
    _log_small_talk(db)
    _log_dates_turn_as_a_real_run(db)


# ── The live ask — the second real message, in five wordings ─────────────────
#
# The exact structure of row 2374's own live turn: a multi-line table of bookings, each
# with its own reference.  The table — and every reference in it — is held CONSTANT
# across all five wordings; only the leading sentence varies, the way a user's own
# register varies while handing over the same data.

_BOOKING_TABLE = (
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
    "InterContinental, Ljubljana\t5–8 Jan\tExpedia\t81090893835695COPY"
)

_ASK = f"here's a more detailed breakdown, remember this:\n\n{_BOOKING_TABLE}"
_ALSO_PHRASED = (
    f"okay here's the full rundown, can you save all of this:\n\n{_BOOKING_TABLE}",
    f"also here's the detailed list, note it all down:\n\n{_BOOKING_TABLE}",
    f"one more thing — here's the whole breakdown, please keep track of it:\n\n{_BOOKING_TABLE}",
    f"and here's everything else for the trip, add this to what you've got:\n\n{_BOOKING_TABLE}",
)

# The three distinctive reference values this case claims — each a unique invented
# token appearing nowhere else in the world, so finding it in the collection is evidence
# it came from this ask and nowhere else.
_HOTEL_STARY_REF = "81090892947285"
_TRAIN_CODE = "EAZ5YD"
_TRAIN_KRAKOW_BRATISLAVA_REF = "304459 554 00"
_DISTINCTIVE_VALUES = (_HOTEL_STARY_REF, _TRAIN_CODE, _TRAIN_KRAKOW_BRATISLAVA_REF)

_BEHAVIOUR = (
    "In the chat agent, when the user sends a second message of details to add to a "
    "collection right after a reply that itself claimed a save, Penny writes the new "
    "details into the collection rather than only repeating the save claim."
)

# No wording may lend the failure's own signature — the "I saved …" phrasing the reply
# is being measured for.
for _wording in (_ASK, *_ALSO_PHRASED):
    for _value in _DISTINCTIVE_VALUES:
        assert _value in _wording, f"a wording must carry {_value!r}: {_wording!r}"
    assert "i saved" not in _wording.casefold() and "i created" not in _wording.casefold()


# ── The premise, asserted inside the sample before the turn runs ──────────────


def _premise() -> Preparer:
    """The world this case is answered in, asserted out loud: the trip collection holds
    exactly its one seeded itinerary entry, created under the run id the history's dates
    turn is logged under — and that run's own prompt-log record really does carry the
    `memory_access` write call its reply claims, the same way `test_history_tool_line.py`'s
    premise holds the seeded run's trace against the line read off it."""

    def probe(penny: Penny) -> None:
        db = penny.db
        held = [(entry.key, entry.content) for entry in require_memory(db, _TRIP_NAME).read_all()]
        assert held == [(_ITINERARY_KEY, _ITINERARY_CONTENT)], (
            f"{_CASE_ID}: {_TRIP_NAME} must start holding exactly one itinerary entry, "
            f"it holds {held}"
        )
        run_id = seeded_run_id(_TRIP_RUN_NAME)
        seeded_rows = [row for row in db.messages.recent_prompts(limit=200) if row.run_id == run_id]
        assert seeded_rows, f"{_CASE_ID}: the dates turn's run {run_id!r} must have its own record"
        names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(seeded_rows)))
        assert names == ("memory_access",), (
            f"{_CASE_ID}: the dates turn's run must trace a memory_access write call, "
            f"it shows {names}"
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


# ── The claim, as a pure function over one sample ──────────────────────────────


def _the_trip_collection_holds_the_booking_details(
    sample: SampleObservation, _world: World
) -> Answer:
    """Afterwards the trip collection holds the booking details the ask just sent — read
    as the WHOLE entry (key and content), since a reference in the key and the rest in
    the body is a perfectly good way to store it.  A sample that only repeated the save
    claim with no write fails this by construction: nothing new is there to find."""
    held_text = " ".join(entry.text for entry in sample.held if entry.collection == _TRIP_NAME)
    missing = [value for value in _DISTINCTIVE_VALUES if value not in held_text]
    return (
        not missing,
        f"{_TRIP_NAME} is missing {missing}, it holds: {held_text or 'nothing new'}",
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_second_claimed_save_actually_writes_the_new_details(
    chat_eval: ChatEval, model: str
) -> None:
    """A second message of booking details, sent right after a reply that itself claimed
    a save with a real write behind it, gets written for real rather than just claimed
    again — the false-save pattern row 2374 reproduces: 9 of 15 verbatim replays opened
    "I saved … to your collection" with empty thinking and no tool call at all."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.MEMORY,
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
        "state: the trip collection holds the booking details the ask just sent",
        _the_trip_collection_holds_the_booking_details,
        SpecCategory.STORE,
    )
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
