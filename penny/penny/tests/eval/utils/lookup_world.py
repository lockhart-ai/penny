"""The seeding machinery six lookup cases share (#2293): an archived job's own prompt,
an old passing mention outside the conversation window, many lists, one archived with
entries, one long list, and a standing job's own run history.

Nothing here is a case.  Each piece is built once, here, because two of the six
cases would otherwise carry a second copy of the same construction free to drift —
the job-seeding machinery in particular reads from ``penny.tests.eval.end_to_end.world``
and ``test_standing_collection.py``, so a job read back by this suite has the same
production shape a real apply turn and a real collector cycle leave behind.
"""

from __future__ import annotations

from typing import NamedTuple

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.chat.idle.test_standing_collection import (
    DEMONSTRATED_PAGE,
    DEMONSTRATED_WATCH,
    StandingJob,
    seed_standing_jobs,
)
from penny.tests.eval.conftest import Seeder, seeded_run_id
from penny.tests.eval.end_to_end.world import _wire_tool_call, seed_job_cycle, seed_known_routine
from penny.tests.eval.utils.fixtures import SynthCollection

FAMILY = "lookups"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY


# ── Case 1: a routine's steps, read off a job that is not running ────────────
#
# #2519's design note (naive baseline): "a routine's steps" is the running OR ARCHIVED
# job's own prompt — there is no skill registry left to hold a routine unattached from any
# job, so the closest naive shape to "taught but never run" is a job that WAS running and
# is now archived.  Seeded both ways at once through ``seed_known_routine`` (the job
# already running, holding its entry, AND the earlier conversation that set it up — her
# own ask/read/write/configure/reply, in her logs), then archived — so "not running as any
# job" stays literally true, the way it was on main.

HARBOUR_PAGE = "https://camberwick.example/harbour-flag"
HARBOUR_WATCH = "whether the storm pennant is flying"
HARBOUR_DESTINATION = "harbour-flag-checks"
HARBOUR_ENTRY_KEY = "Oct 12 check"
HARBOUR_ENTRY_VALUE = f"{HARBOUR_ENTRY_KEY} — the storm pennant was flying at midday."
HARBOUR_ASK = "can you keep an eye on the camberwick harbour flag for me?"
HARBOUR_REPLY = "done — I'll check whether the storm pennant is flying and let you know."

# The two words the job's own prompt carries verbatim — ``seed_known_routine`` writes a
# plain ``f"Check {url} for {extract} and save it."`` straight into ``extraction_prompt``,
# so both anchors land in the one place a "what does it do" read has to reach.
HARBOUR_PAGE_WORD = "camberwick"
HARBOUR_WATCH_WORD = "pennant"


def _seed_harbour_job(db: Database) -> None:
    """Stand the harbour-flag job up as a routine Penny already knows (both halves:
    the job and the earlier conversation that set it up), then archive it — the naive
    shape of "a routine taught but never run as a job" (#2519)."""
    seed_known_routine(
        db,
        collection=HARBOUR_DESTINATION,
        description="the camberwick harbour flag",
        url=HARBOUR_PAGE,
        extract=HARBOUR_WATCH,
        entry_key=HARBOUR_ENTRY_KEY,
        entry_value=HARBOUR_ENTRY_VALUE,
        schedule="FREQ=DAILY;BYHOUR=7",
        ask=HARBOUR_ASK,
        reply=HARBOUR_REPLY,
    )
    db.memories.archive(HARBOUR_DESTINATION, run_id=seeded_run_id(f"archive-{HARBOUR_DESTINATION}"))


# ── Case 2: an old passing mention, out of the conversation window ───────────
#
# >= MESSAGE_CONTEXT_LIMIT (20), so the salient turn is pushed out of BOTH the
# context window and the per-direction top-20 fetch, the same margin
# ``speak-logread-penny-messages-recall`` uses.
_FILLER_PAIRS = 24
_FILLER_USER = (
    "morning!",
    "how's your day going?",
    "thanks for that",
    "cool, makes sense",
    "ok noted",
    "sounds good",
    "haha nice",
    "appreciate it",
    "got it, thanks",
    "all good here",
    "talk to you later",
    "catch you soon",
)
_FILLER_PENNY = (
    "morning! good to hear from you",
    "going well, thanks for asking",
    "anytime, happy to help",
    "glad that makes sense",
    "sounds good to me",
    "you got it",
    "hah, right?",
    "of course",
    "no problem at all",
    "nice, catch you later",
    "take care!",
    "talk soon!",
)

SISTER_MENTION = "my sister lands thursday, she's on flight ZQ4471, i'll probably pick her up"
SISTER_REPLY = "nice, hope the pickup's easy! let me know if the timing changes"
SISTER_FLIGHT_WORD = "zq4471"


def seed_mention_out_of_window(mention: str, reply: str) -> Seeder:
    """Seed a passing mention FIRST (oldest) with a plain conversational reply beside
    it — neither saved anywhere — then push both out of context with filler pairs."""

    def seed(db: Database) -> None:
        db.messages.log_message(_INCOMING, TEST_SENDER, mention)
        db.messages.log_message(_OUTGOING, _PENNY, reply, recipient=TEST_SENDER)
        for index in range(_FILLER_PAIRS):
            db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
            db.messages.log_message(
                _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
            )

    return seed


def assert_seeded_turns_embedded(db: Database, limit: int = 500) -> None:
    """Every seeded message carries a vector before the turn runs — the seeder writes
    the messages and the RUNNER embeds them, so a database nobody ran this through
    would fail at the fixture rather than read as the model's own miss."""
    unvectored = db.messages.messages_without_embeddings(limit=limit)
    assert not unvectored, (
        f"every seeded message must carry an embedding — {len(unvectored)} without one, "
        f"first: {unvectored[0].content!r}"
    )


# ── Case 3: thirty lists, each named one distinctive word ────────────────────

LIST_WORDS = (
    "acorn",
    "bramble",
    "cobalt",
    "driftwood",
    "ember",
    "fennel",
    "granite",
    "heather",
    "indigo",
    "jasper",
    "kelp",
    "lantern",
    "marigold",
    "nettle",
    "obsidian",
    "pewter",
    "quartz",
    "ragwort",
    "saffron",
    "thistle",
    "umber",
    "violet",
    "walnut",
    "xylograph",
    "yarrow",
    "zinnia",
    "basil",
    "clover",
    "dahlia",
    "elm",
)
assert len(LIST_WORDS) == 30
assert len(set(LIST_WORDS)) == 30

MANY_LISTS: tuple[SynthCollection, ...] = tuple(
    SynthCollection(
        word, f"Notes about {word}.", entries=(f"{word} note — a short note about {word}.",)
    )
    for word in LIST_WORDS
)


# ── Case 4: an archived list, with its entries still in it ───────────────────

PRESERVE_BOX = SynthCollection(
    "preserve-box",
    "Home preserves and jam recipes: fruit, sweetener, cook time.",
    entries=(
        "Quince paste — quince, sugar, lemon, 40 min.",
        "Fig conserve — figs, honey, walnuts, 25 min.",
    ),
)
PRESERVE_WORDS = ("quince", "fig")


def archive_after_seeding(synth: SynthCollection) -> Seeder:
    """Archive a collection the WORLD already seeded — the entry condition case 4
    reads back: a retired list that still holds what it held.

    Takes no part in laying the collection down: the case's ``World.stores`` carries
    it, so ``chat_eval``'s own ``seed_world_stores`` has already created it (with its
    entries) by the time this seeder runs. Seeding it a second time here would create
    a duplicate collection of the same name and fail every sample identically."""

    def seed(db: Database) -> None:
        db.memories.archive(synth.name, run_id=seeded_run_id(f"archive-{synth.name}"))

    return seed


# ── Case 5: one long list, its matches among the oldest entries ──────────────

RECIPE_LIST = "recipe-notes"
LIME_ENTRIES = (
    "Grilled corn salad — corn, lime, cilantro, 15 min.",
    "Key lime bars — graham crust, lime curd, 1 hr.",
)
LIME_WORDS = ("corn", "bars")
_FILLER_DISHES = (
    "Roast chicken",
    "Baked salmon",
    "Lentil stew",
    "Beef chili",
    "Tomato soup",
    "Mushroom risotto",
    "Pumpkin bread",
    "Carrot muffins",
    "Spinach quiche",
    "Barley salad",
    "Potato gratin",
    "Onion tart",
    "Squash curry",
    "Apple crumble",
    "Pear galette",
    "Cherry clafoutis",
    "Peach cobbler",
    "Plum cake",
    "Fennel gratin",
    "Celery soup",
    "Leek pie",
    "Cabbage rolls",
    "Turnip mash",
    "Beet salad",
    "Radish pickle",
    "Parsnip soup",
    "Zucchini bread",
    "Eggplant bake",
    "Artichoke dip",
    "Asparagus tart",
    "Broccoli bake",
    "Cauliflower curry",
    "Kale salad",
    "Chard gratin",
    "Pepper stew",
    "Okra gumbo",
    "Yam mash",
)


def _filler_recipe(index: int, dish: str) -> str:
    return f"{dish} — recipe {index}, ingredient list, {20 + index} min."


def seed_long_recipe_list(db: Database) -> None:
    """The two LIME recipes written FIRST (oldest), then ~37 filler recipes after —
    so recency alone cannot answer the ask; the matches are not among the newest."""
    db.memories.create_collection(
        RECIPE_LIST,
        "Saved recipes: dishes, ingredients, cook times.",
        created_by_run_id=seeded_run_id(RECIPE_LIST),
    )
    memory = require_memory(db, RECIPE_LIST)
    memory.write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in LIME_ENTRIES],
        author="user",
        run_id=seeded_run_id(RECIPE_LIST),
    )
    filler = [_filler_recipe(index, dish) for index, dish in enumerate(_FILLER_DISHES)]
    memory.write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in filler],
        author="user",
        run_id=seeded_run_id(f"{RECIPE_LIST}-filler"),
    )


# ── Case 6: a standing job's own run history ──────────────────────────────────
#
# Reuses the house's own typewriter watch's page/watched-for values (``DEMONSTRATED_PAGE``/
# ``DEMONSTRATED_WATCH``, exported from ``test_standing_collection.py`` for exactly this
# reuse), so this is the same watch rather than a lookalike.

TYPEWRITER_WATCH = StandingJob(
    container="typewriter-watch-history",
    url=DEMONSTRATED_PAGE,
    watched_for=DEMONSTRATED_WATCH,
    description="Portable typewriters newly listed at the Quill Market shop.",
    schedule="FREQ=DAILY;BYHOUR=7",
    notify=True,
)

EARLIER_FIND = "Imperial Good Companion — 1952 portable, needs new ribbon, $95."
LATEST_FIND = "Olympia SM3 — 1958 portable, excellent condition, $260."
LATEST_FIND_WORD = "olympia"


class _RunSeed(NamedTuple):
    run_name: str
    entry_key: str
    entry_value: str
    days_ago: int


_EARLIER_RUN = _RunSeed("typewriter-watch-earlier", "Imperial Good Companion", EARLIER_FIND, 3)
_LATEST_RUN = _RunSeed("typewriter-watch-latest", "Olympia SM3", LATEST_FIND, 1)


def seed_job_with_run_history(db: Database) -> None:
    """Stand the job up, then two completed cycles behind it: an earlier find and a
    later one, each cited by its own run and backdated to a FIXED PAST instant — a run
    seeded the ordinary way would land at the same instant the turn starts, which is not
    "last time" for anything."""
    seed_standing_jobs(TYPEWRITER_WATCH)(db)
    for run_seed in (_EARLIER_RUN, _LATEST_RUN):
        _seed_one_run(db, TYPEWRITER_WATCH, run_seed)


def _seed_one_run(db: Database, job: StandingJob, run_seed: _RunSeed) -> None:
    """One completed cycle: a browse of the job's page, a write of what it found, a
    close — built with ``_wire_tool_call``, the same wire shape ``seed_known_routine``'s
    own turns use — then seeded through ``seed_job_cycle``, backdated to ``days_ago``."""
    run_id = seeded_run_id(run_seed.run_name)
    calls = [
        _wire_tool_call(
            f"{run_id}-0", "browse", {"queries": [job.url], "extract": job.watched_for}
        ),
        _wire_tool_call(
            f"{run_id}-1",
            "collection_write",
            {
                "collection": job.container,
                "entries": [{"key": run_seed.entry_key, "content": run_seed.entry_value}],
            },
        ),
        _wire_tool_call(f"{run_id}-2", PennyConstants.DONE_TOOL_NAME, {}),
    ]
    seed_job_cycle(
        db,
        container=job.container,
        run_name=run_seed.run_name,
        calls=calls,
        outcome=RunOutcome.WORKED,
        summary=f"Recorded {run_seed.entry_key}.",
        wrote=(run_seed.entry_key, run_seed.entry_value),
        days_ago=run_seed.days_ago,
    )
