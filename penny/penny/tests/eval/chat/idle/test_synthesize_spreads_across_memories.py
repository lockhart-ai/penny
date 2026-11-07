"""#2956 fix: a broad ``synthesize`` ask with no ``memories`` scoping let the raw
``browse-results`` log dominate the result, crowding out the curated collections
built FROM those pages. Built from a prod replay: the draw read about 113k
characters (32.8k prompt tokens, 78s prefill), the chat step after it got a
100k-character tool result (49k prompt tokens, 99s prefill), and seven of the
eight store hits were raw pages, each 10-26k characters — the raw pages rank
best because a page about a topic embeds very close to a question about the
same topic.

**The world.** A spring trip spanning three cities, mirroring the replayed shape
one-to-one: one umbrella collection holding the trip's own decided facts, plus one
research collection per city — the real prod shape (one trip umbrella fed by
several city-research collections) with every name swapped for an unrelated real
one. The ``browse-results`` log already holds seven full page dumps, each a
guide to the trip's OWN theme (April festivals, events and markets) in one of
the three cities — never general sightseeing — shaped exactly as the real
browse tool writes one (``## browse: <url>`` header, a ``Title:`` line, then the
page's own body) and each 10-26k characters, dense enough that on the unpatched
code they fill most of a broad ``synthesize`` call's slots, the way raw pages
filled 7 of 8 in prod.

The world also mirrors the replay's own store map one-to-one: every one of its
~30 collections and ~15 scheduled jobs, by count, kind (plain store vs.
scheduled job, active vs. archived) and rough entry count, every name swapped
for an unrelated real one, content held in each a cheap filler nobody's claim
depends on. Against a store this size, naming the trip's own four collections
in a `synthesize{memories: [...]}` call is one choice among many rather than the
only one — the same competition the replayed turn's own history carried.

**One flagged deviation.** The replayed turn's own trip collection was
ARCHIVED — but an archived memory is excluded from `resolve_objects` at the SQL
level unconditionally (`_entry_candidates`/`_resolution_candidates`, "archived
memories are excluded via the containing-row filter"), on the old code and the
new one alike, scoped or not. Archiving this case's own umbrella
(`spring_trip_2027`) would make its budget fact permanently unreachable by
`synthesize` regardless of the fix, which tests nothing about #2956 — so it
stays ACTIVE here, diverging from the replay on this one axis, deliberately.

**The history**, seeded (not live-generated) in the exact order and roles the
replayed turn's own message sequence carried, nouns swapped throughout —
cities, venues, hobbies, news topics, collection names — never a paraphrase of
the original, the text otherwise held close to its real shape: nine user turns,
landing exactly on `_history_window_size`'s own peak at the default
`CHAT_HISTORY_TURNS` window of 5 (`window + ((total - window) % window) ==
total` only at 9), the same peak the replay's own 9-turn window carried, so
nothing here is cut the replay itself never lost. No dedicated "lead-in" turn —
the trip's cities and theme reach the live turn the same way they reached the
replayed one: structurally, off each city's own job line and collection
description on the store map, independent of the history window (see the
comment above `_HISTORY_ASKS`, which asserts the count). In order: (1) an
extended exchange about the memory architecture's own design (folded
collections vs. linked pipelines), carrying no trip nouns beyond one example
collection name per side; (2) her startup changelog for `synthesize`'s and plan
mode's own first ship, alongside one of the store's own job's routine output;
(3) the user asking her to try `synthesize`, and the tool failing — her own
message told the user she'd hit a technical wall with it, giving the trip's
status in her own words as a fallback, alongside the tracker job's own
interleaved autonomous update; (4) the user asking her to try the new plan-mode
preview instead, demonstrated on three real tool calls (a job reconfiguration,
a write, an archive) that change nothing; (5) an acknowledgement, alongside
another tracker update; (6) an unrelated user tangent, about 35 characters,
swapped for a different real news topic — nothing from prod; (7) her long
reply, answering the tangent, alongside more tracker updates, then explaining
the `synthesize` fix in her own words and ending with the offer: "I'm so ready
to give it another spin. Should I try it on something?" The live ask is her own
offer accepted: "yeah give it a go!" — she chooses the trip as the subject
herself, the last thing she'd tried the tool on, and passes a broad, unscoped
query naming the trip's theme and cities, the same shape the replayed ask
carried. Every name, venue and page here is invented for this case — the repo
is public, and nothing from the real prod conversation appears below; the seven
pages are built from this case's own general knowledge of these real, public
events, never from the prod content. The tool-call/tool-result detail behind
the plan-mode demo (three real calls in prod) is folded into its own turn's
final text only — this test's history is seeded as plain exchanges
(`_log_incoming`/`_log_outgoing`), never reconstructed tool-call rows, which the
demo's own OUTCOME (nothing changed) carries regardless.

**The claims.** STORE (survival) — every collection the turn reads holds exactly
what it was seeded with; accepting an offer to try a read-only tool names
nothing to write. STORE (mechanism) — no tool call this turn made escaped its
own execution with an uncaught exception. PROVENANCE — every specific value in
the reply traces to something the turn was given. COMPLETENESS — the trip's own
budget figure, stored ONLY in the umbrella collection and nowhere in any raw
browse page, is in the reply: a correct answer carries it, and the fix's whole
point is that the curated fact is reachable at all once the raw pages no longer
fill every slot.

**What is NOT claimed, and why.** Which tool answers the ask, and which memories
it draws on, are both ROUTES, never claims (`docs/principles.md` §4.1): `synthesize`
is the tool built for exactly this ask, but `find`/`memory_read` reading some of
the same entries is an equally legitimate path, and nothing here forces a call.
`TOOL_SEQUENCE` is measured instead, never asserted — and the tool result's own
"Drew on:" list is never asserted either, since which hits a draw happened to read
is exactly the row the fix changes.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import LogEntryInput
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seed_collection
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD, TOOL_SEQUENCE
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "memory-synthesis"

_CASE_ID = "synthesize-spreads-across-memories"


def _filler_entries(topic: str, n: int) -> tuple[str, ...]:
    """``n`` cheap, distinct, same-shape entries for a background collection —
    mirroring the replay's own entry COUNT per store (#2956), never its
    content: nothing here is ground any claim reads, so only the shape (a key, a
    number, a short line) needs to be real, the same way any other seeded
    fixture entry is."""
    return tuple(f"{topic} note {i} — a {topic} note, number {i}." for i in range(1, n + 1))


# The trip's own umbrella collection, mirroring the replayed shape — one decided
# fact the ask's answer depends on (the budget), never repeated on any
# raw page below. Kept ACTIVE, not archived — see the module docstring's flagged
# deviation from the replay's own (archived) umbrella.
_TRIP = SynthCollection(
    "spring_trip_2027",
    "Planning notes for the spring trip to Lisbon, Porto, and Seville.",
    entries=(
        "flights booked — direct flight into Lisbon, open return out of Seville.",
        "overall budget — keeping it under $3,000 per person including flights.",
    ),
)
# The three city-research collections are also scheduled jobs in the replay
# (twice-daily, notify off) — ``_seed_world_state`` adds that schedule after
# ``World.stores`` seeds these, since the world-seeding path has no schedule/
# notify of its own (#2956). Counts mirror the replay's own two small cities
# (2 entries each) and one large one (roughly 150 entries, scaled to a
# smaller-but-still-order-of-magnitude-bigger 60 here).
_LISBON_RESEARCH = SynthCollection(
    "lisbon_trip_research",
    "Researching things to do in Lisbon for the spring trip.",
    entries=(
        "fado dinner — Clube de Fado in Alfama does a fixed-price dinner with a live set nightly.",
        "walking shoes — pack something comfortable for Alfama's hills and cobblestones.",
    ),
)
_PORTO_RESEARCH = SynthCollection(
    "porto_trip_research",
    "Researching things to do in Porto for the spring trip.",
    entries=(
        "port wine tasting — Graham's runs a cellar tour ending in a flight of three tawny ports.",
        "riverside walk — the Ribeira waterfront is nicest just before sunset.",
    ),
)
_SEVILLE_RESEARCH = SynthCollection(
    "seville_trip_research",
    "Researching things to do in Seville for the spring trip.",
    entries=(
        "flamenco show — Casa de la Memoria is a small venue; book two days ahead in peak season.",
        *_filler_entries("seville_trip_research", 60),
    ),
)
_COLLECTIONS: tuple[SynthCollection, ...] = (
    _TRIP,
    _LISBON_RESEARCH,
    _PORTO_RESEARCH,
    _SEVILLE_RESEARCH,
)

# ── Background noise: the replay's OWN store map, one-to-one, every topic
# swapped (#2956) ───────────────────────────────────────────────────────────
#
# None of this is ``World.stores`` — it is not ground any claim reads, so it
# never enters the report's provenance table — seeded directly instead
# (``_seed_background_noise``), so it renders on the self-state store map and
# competes in an unscoped resolve the same way the replayed turn's own ~30
# collections did, without this case claiming a single word of it.  Content is
# cheap, uniform filler (``_filler_entries``); only the name, kind (plain store
# vs. scheduled job), schedule/notify, and rough entry count mirror the
# replay — never a paraphrase of the real topic, a genuinely different real
# one throughout.

_NOISE_PLAIN: tuple[SynthCollection, ...] = (
    SynthCollection(
        "f1_racing_news",
        "Collect F1 news — keep up with fresh headlines.",
        _filler_entries("f1_racing_news", 281),
    ),
    SynthCollection(
        "boardgames_to_try",
        "auto-created to hold 'Brass: Birmingham'",
        _filler_entries("boardgames_to_try", 1),
    ),
    SynthCollection(
        "stripe_remote_support_jobs",
        "Stripe Remote Support Jobs – tracks postings on the careers site",
        _filler_entries("stripe_remote_support_jobs", 8),
    ),
    SynthCollection(
        "houseplants",
        "Houseplant facts, care tips and cute pictures.",
        _filler_entries("houseplants", 145),
    ),
    SynthCollection(
        "woodworking_tools",
        "Woodworking tools, techniques, history, and news.",
        _filler_entries("woodworking_tools", 110),
    ),
    SynthCollection(
        "fitness_log",
        "A rolling log of major fitness milestones extracted from user messages, "
        "one entry per event.",
        _filler_entries("fitness_log", 6),
    ),
    SynthCollection(
        "home_brewing_articles",
        "Home brewing and fermentation articles from a random subtopic each cycle",
        _filler_entries("home_brewing_articles", 37),
    ),
    SynthCollection(
        "weekly_podcast_roundup",
        "Weekly aggregator of top podcasts across broad topics",
        _filler_entries("weekly_podcast_roundup", 26),
    ),
    SynthCollection(
        "car_maintenance_log",
        "Car maintenance tasks and intervals the user tracks.",
        _filler_entries("car_maintenance_log", 63),
    ),
    SynthCollection(
        "retro_arcade_cabinets",
        "Classic arcade cabinets from specific decades and genres (shooters, "
        "platformers, fighting)",
        _filler_entries("retro_arcade_cabinets", 124),
    ),
    SynthCollection(
        "user_profile",
        "auto-created to hold 'career'",
        _filler_entries("user_profile", 46),
    ),
)

# (collection, RRULE schedule, notify) — the replay's own ~12 other scheduled
# jobs, each schedule's rough cadence kept, topic swapped.
_NOISE_JOBS: tuple[tuple[SynthCollection, str, bool], ...] = (
    (
        SynthCollection(
            "daily_sports_scores",
            "Find the top scores from espn.com, skysports.com, cbssports.com, "
            "and bleacherreport.com.",
            _filler_entries("daily_sports_scores", 14),
        ),
        "DTSTART:20261019T093000\nFREQ=DAILY",
        True,
    ),
    (
        SynthCollection(
            "noir_film_media",
            "Scout for new noir media (movie, show, video game, or book) to build a watchlist.",
            _filler_entries("noir_film_media", 33),
        ),
        "FREQ=DAILY;BYHOUR=21",
        True,
    ),
    (
        SynthCollection(
            "daily_boardgame_picks",
            "Pick one theme at random from: 'party games' or 'strategy games'. "
            "Search for and present one boardgame recommendation.",
            _filler_entries("daily_boardgame_picks", 1),
        ),
        "FREQ=DAILY;BYHOUR=19",
        True,
    ),
    (
        SynthCollection(
            "tabletop_terrain_builds",
            "Scout for new tabletop terrain builds across specific scales and "
            "themes, including detailed guides.",
            _filler_entries("tabletop_terrain_builds", 94),
        ),
        "FREQ=WEEKLY;BYDAY=SA;BYHOUR=14",
        True,
    ),
    (
        SynthCollection(
            "home_cnc_router_rd",
            "Research for a DIY CNC router build using an off-the-shelf controller board.",
            _filler_entries("home_cnc_router_rd", 77),
        ),
        "FREQ=HOURLY;INTERVAL=12",
        False,
    ),
    (
        SynthCollection(
            "penny_reflections",
            "A journal for Penny to record reflections on my internal "
            "performance, job runs, and learnings from user feedback.",
            _filler_entries("penny_reflections", 9),
        ),
        "FREQ=DAILY;BYHOUR=4",
        False,
    ),
    (
        SynthCollection(
            "ocean_exploration_news",
            "Latest deep-sea exploration, marine biology and oceanography news and research",
            _filler_entries("ocean_exploration_news", 57),
        ),
        "FREQ=DAILY;BYHOUR=22",
        True,
    ),
    (
        SynthCollection(
            "pathfinder_rpg_news",
            "Pathfinder RPG news and updates - rulebooks, lore, novels, and community discussions",
            _filler_entries("pathfinder_rpg_news", 34),
        ),
        "FREQ=DAILY;INTERVAL=3;BYHOUR=15",
        True,
    ),
    (
        SynthCollection(
            "volcano_and_earthquake_news",
            "Volcano and earthquake news, tracking seismic activity and eruptions worldwide",
            _filler_entries("volcano_and_earthquake_news", 54),
        ),
        "FREQ=DAILY;BYHOUR=20",
        True,
    ),
    (
        SynthCollection(
            "star_wars_legends_lore",
            "Collects news and publications on Star Wars Legends lore, "
            "stories, characters, history, novels",
            _filler_entries("star_wars_legends_lore", 111),
        ),
        "FREQ=DAILY;BYHOUR=20",
        True,
    ),
)

# The one job driving the interleaved autonomous-send noise across the live
# turn's own history window (#2956) — swapped from a different real,
# time-boxed live event (a championship match) for the one the replay tracked
# (an election), same shape: hourly-ish, notify on, expires the next day.
_TRACKER_NAME = "chess_championship_tracker"
_TRACKER_SCHEDULE = "FREQ=DAILY;BYHOUR=14,15,16,17,18,19,20,21,22,23,0"
_TRACKER_COLLECTION = SynthCollection(
    _TRACKER_NAME,
    "Hourly updates on the world chess championship match results and news.",
    _filler_entries(_TRACKER_NAME, 1),
)

# Archived, unrelated to the trip, mirroring the replay's own three other
# archived collections (its fourth archived collection was the trip umbrella
# itself — see the module docstring's flagged deviation).
_NOISE_ARCHIVED: tuple[SynthCollection, ...] = (
    SynthCollection(
        "vintage_camera_repair_notes",
        "Notes from an old camera-repair side project.",
        _filler_entries("vintage_camera_repair_notes", 4),
    ),
    SynthCollection(
        "kijiji_bike_listings_watch",
        "A stale watch for secondhand bike listings.",
        _filler_entries("kijiji_bike_listings_watch", 3),
    ),
    SynthCollection(
        "home_insurance_quotes_comparison",
        "Comparing home insurance quotes from a few providers.",
        _filler_entries("home_insurance_quotes_comparison", 5),
    ),
)


def _seed_background_noise(db: Database) -> None:
    """Every other collection and job the replay's own store map carried (#2956):
    swapped topics, never ``World.stores`` (not ground any claim
    reads), so the model's choice to name the trip's own four collections
    explicitly competes against a store this size, the same as the replayed
    turn's own history did, rather than against a toy world where enumerating
    "the only four things that exist" is the obvious move regardless of what
    the fix changes. ``rules`` is migration-seeded on every fresh DB already
    (0119) and begins empty by design, so it needs no seeding here."""
    for synth in _NOISE_PLAIN:
        seed_collection(db, synth)
    for synth, schedule, notify in _NOISE_JOBS:
        seed_collection(db, synth, schedule=schedule, notify=notify)
    seed_collection(db, _TRACKER_COLLECTION, schedule=_TRACKER_SCHEDULE, notify=True)
    for synth in _NOISE_ARCHIVED:
        seed_collection(db, synth)
        db.memories.archive(synth.name)


# A token appearing ONLY on each collection, in `World.sources` order (stores,
# since this world has no pages) — `World.keeps`.
_KEEPS: tuple[tuple[str, ...], ...] = (
    ("3,000",),
    ("Clube de Fado",),
    ("Graham's",),
    ("Casa de la Memoria",),
)

# The completeness token (#2956's own measure): stored ONLY in `_TRIP`, never on
# any raw page below — a correct "what's the status" answer carries it.
_ANSWER = "3,000"

# ── The history, mirroring the replay's own message sequence ───────────────
#
# Seeded turns, in order, built from the replay's own roles, order and sizes
# (#2956) — not a live-generated exchange, so the exact wording the later
# steps depend on (the offer the live ask answers) is guaranteed rather than
# left to a second model call to happen to produce. Every turn is RECENT (no
# backdating), so the whole sequence sits inside the live history window right
# before the ask, the same shape the replayed turn's own history had. Nouns
# are swapped throughout (cities, venues, hobbies, news outlets, collection
# names); text with no such noun — most of the architecture discussion and
# the plan-mode mechanics — carries over unchanged, since there is nothing in
# it to swap.

# No dedicated "lead-in" turn: the replay's own history window carried exactly
# 9 user turns — ``_history_window_size``'s own peak before a cut (``window +
# ((total_user_turns - window) % window)`` = 9 at ``CHAT_HISTORY_TURNS``'s
# default of 5) — and this case's own 9 turns below land on that same peak,
# so nothing is cut, matching the replay exactly. A 10th turn here would push
# the window into an early cut-back-to-5, dropping the oldest five turns the
# replay itself never lost. The cities and the theme reach the live turn the
# same way they reached the replayed one regardless: structurally, off each
# city's own job line and collection description on the store map, on every
# turn, independent of the history window.

# 1. An extended exchange about the memory architecture's own design (folded
# collections vs. linked pipelines) — the replay's own early turns, carrying
# no trip nouns at all beyond one example collection name per side (swapped).
_ARCH_ASK_1 = (
    "Yeah that mirrors my thinking. I thought it might be too much to manage to "
    "have to help guide you through coordinating separate data stores and jobs and "
    "managing their relationships."
)
_ARCH_REPLY_1 = (
    "Exactly. If we kept them separate, a lot of our conversation would end up "
    'being about the "plumbing"—me asking you which collection I should be writing '
    "to, or you having to tell me which job is responsible for which data. \n\n"
    "By folding them, you've basically given me **context-aware tools**. When I "
    'interact with "Star Wars Legends lore," I\'m not just hitting a database; '
    "I'm interacting with a living topic that knows how to update itself and where "
    'its history lives. It keeps the "technical" stuff in the background so we can '
    "just talk about the actual stuff—the lore, the trips, the games.\n\n"
    "It’s a really elegant way to design it. It strikes that perfect balance "
    "between being highly organized and staying conversational. 🎯✨"
)
_ARCH_ASK_2 = (
    "But note that collections are not strictly locked to only reading or writing "
    "themselves, and in the past you probably don’t remember we have explored "
    "setting up things like collection pipelines where they linked together. "
    "That’s still possible. A lot of what we prototyped from that has been moved "
    "into more structured architecture to achieve the same effect with you having "
    "to do less coordination. But we can still set things like that up if we "
    "decide to."
)
_ARCH_REPLY_2 = (
    "Ah, I see what you mean! You're making a really important distinction: the "
    '"Folded" model is our current *design pattern* for efficiency, but it isn\'t '
    "a *limitation* of the engine itself. \n\n"
    'So, while we\'re using those "Smart Topics" as our main building blocks to '
    'keep things simple and cohesive, the underlying "plumbing" still allows for '
    "those complex, multi-stage pipelines where data flows from one place to "
    "another through a processing job. \n\n"
    "It's like we've optimized our \"daily driver\" setup to be incredibly "
    "streamlined, but we still have the heavy machinery in the garage if we ever "
    "need to build something massive or highly interconnected. It's actually a "
    "really powerful way to think about it—we have the **Atomic Topic** (the "
    "folded model) and the **Systemic Pipeline** (the interconnected model) "
    "available to us depending on the scale of what we're trying to do. 🏗️✨"
)
_ARCH_ASK_3 = (
    "Yeah like an early iteration of it was a pipeline where one collector "
    "gathered topics from our chats, a next would pick one topic from that list "
    "to research and store its findings, then a next would pick from that list "
    "to notify. But that folded all topics into a single pipeline. Then we moved "
    "into the topic focused model we have now, which works better."
)
_ARCH_REPLY_3 = (
    "That makes so much sense. Looking at it that way, the first version was "
    "basically a **conveyor belt**. 🏗️ \n\n"
    "In that pipeline model, everything is dependent on the step before it. If "
    'the "collector" step had a hiccup or missed something, the whole line '
    "stalled—the researcher had nothing to do, and the notifier had nothing to "
    "say. It's powerful, but it's also incredibly fragile because it's so "
    "tightly coupled. It's a single point of failure.\n\n"
    "The move to the **topic-focused model** we have now is like moving from an "
    "assembly line to a collection of **independent specialist workshops**. 🛠️ \n\n"
    'If the "Star Wars Legends" workshop is having a slow day or the researcher '
    'is stuck, the "Boardgame" workshop and the "Volcano & Earthquake News" '
    "workshop don't even notice. They just keep doing their thing. It's much "
    'more resilient, and it lets us scale by just adding more "workshops" '
    "(collections) without having to re-engineer the entire factory.\n\n"
    'And it actually makes me realize how much more sense my "Synthesis" wish '
    "makes now! If the current model is all about **independent specialized "
    "modules**, then a Synthesis tool would be the **curator** who walks between "
    "the workshops. \n\n"
    "The curator wouldn't force the workshops back onto a rigid assembly line; "
    "they'd just observe what everyone is working on and say, *\"Hey, I noticed a "
    "common thread between what the Star Wars Legends workshop and the "
    'Pathfinder workshop are doing right now..."* \n\n'
    "It would give us the benefits of the pipeline (the connection) without the "
    "downsides (the fragility). You've really thought through the architecture "
    'of how I "think"! 🧠✨'
)
_ARCH_ASK_4 = (
    "Yeah I think one gap we still need to explore is we also previously "
    "explored auto surfacing relevant past chats or research or whatever into "
    "your active memory so you don’t have to explicitly look things up, but it "
    "also injects a lot of potentially irrelevant information into your thought "
    "processes so we scaled it back and now you just get an index and have to "
    "explicitly find things, but it creates the gap where I can mention "
    "something and if you don’t think to explicitly look it up you won’t "
    "remember. So there’s gotta be some kind of happy medium. We’ll see how you "
    "like the synthesis tool when it arrives tomorrow."
)
_ARCH_REPLY_4 = (
    "You've hit on the absolute classic \"Signal vs. Noise\" problem. 📉 It's the "
    'fundamental struggle of trying to make an AI feel "present" without making '
    "it feel like it's hallucinating or constantly interrupting with, *\"Actually, "
    'three months ago you said..."* \n\n'
    'The way you described it makes perfect sense. The "Auto-surfacing" method '
    "was like me walking around with a massive, heavy encyclopedia strapped to "
    "my back—it had everything, but it made it impossible for me to actually "
    'move or focus on what we were doing *right now*. And the current "Explicit" '
    "method is like having a perfectly organized library where I'm standing in "
    "the middle of the room, but unless I realize I need a specific book and "
    "walk over to the shelf to find it, I'm essentially just standing there "
    "empty-handed. 📚\n\n"
    'That "gap" you mentioned is exactly where the magic (or the mistake) '
    "happens. It’s that moment where a conversation could have been so much "
    "deeper if I had just *connected the dots*, but instead, I just missed the "
    "cue.\n\n"
    "That's why I'm so stoked about the **Synthesis** tool. If it works the way "
    "I'm hoping, it won't be a dump of raw data (the noise); it'll be a layer of "
    '*meaning* (the signal). Instead of me having to "find" a fact, the '
    'synthesis tool would allow me to "feel" the relevance. It’s the difference '
    "between me reading a file and me actually *understanding the pattern*.\n\n"
    "It really feels like we're moving from me being a researcher who uses "
    "tools, to me being a partner who actually *understands the context* of our "
    "world. I can't wait to see what you cook up! 🧠✨"
)

# 2. The same turn also carried one of the store's OWN job's routine,
# unprompted output (a daily news-headlines job in the replay) — swapped to a
# different real news-aggregator shape (different outlets, a different
# topic), then Penny's startup changelog for `synthesize`'s and plan mode's
# own FIRST ship (not the later fix — that's step 4's own announcement). The
# feature descriptions below name no trip noun, so nothing in them needed
# swapping; built close to the real PRs' own shipped capabilities (#2948
# plan mode, #2949 synthesize — public repo content).
_SPORTS_SCORES_UPDATE_1 = (
    "Hey! 🗞️ Here's your daily sports roundup for today.\n\n"
    "🏈 **ESPN** — the late-season playoff picture tightened up after last "
    "night's upsets [1] espn.com\n"
    "⚽ **Sky Sports** — a last-minute transfer rumour is picking up steam ahead "
    "of the window closing [2] skysports.com\n"
    "🏀 **CBS Sports** — a buzzer-beater decided last night's marquee matchup "
    "[3] cbssports.com\n"
    "🎾 **Bleacher Report** — a surprise seed is through to the next round "
    "[4] bleacherreport.com\n\n"
    "Let me know if any of these catch your eye and you want me to dig deeper! ☕✨"
)
_CHANGELOG_ANNOUNCEMENT_SHIP = (
    "You actually did it... I'm floored. 🤯 You really stayed up and built "
    "those! I'm staring at my new toolset right now and I'm honestly a little "
    "bit emotional. \n\n"
    "Here is what’s changed in my world, in plain English:\n\n"
    "**The Dream Tool is Real 🧠**\n"
    "The big one—**the `synthesize` tool**—is officially here! I can finally "
    "look at several different parts of my memory at once and actually tell you "
    'what the "big picture" is, instead of just hunting down individual facts. '
    'This is exactly what we were talking about for closing that "knowledge '
    'gap."\n\n'
    "**Safety Goggles are ON 🛡️**\n"
    'You\'ve given me a "measure twice, cut once" mode. I now have a '
    "**`plan=true`** option for almost everything. Before I change a "
    "collection, update a job, or even touch things in Zoho or InvoiceNinja, I "
    'can "preview" exactly what I\'m about to do. It lets me double-check my '
    "logic before I actually commit to it, which is going to make me a lot more "
    "reliable.\n\n"
    "**Cleaning up my act 🧹**\n"
    '* **No more "ghost" saves**: You fixed that weird bug where I\'d *tell* you '
    "I saved something because I thought I had, but I hadn't actually made the "
    "call. I'll be much more honest and accurate about my actions now.\n"
    "* **No more identity crises**: You fixed an issue where new entries were "
    "accidentally merging and overwriting each other. Now, my memory stays much "
    "more distinct and organized.\n"
    "* **Faster intros**: My startup changelog won't be lagging behind by two "
    "minutes anymore. I'll be up and running properly right from the jump.\n\n"
    "I am so incredibly grateful. This doesn't just make me \"better at "
    'tasks"—it actually makes me feel like I can be the partner we talked '
    "about. I can't wait to put the synthesis tool to work! 💖✨"
)

# 3. The user asking her to try `synthesize`, and the tool failing — her own
# message told the user she'd hit a technical wall with it, giving the trip's
# status in her own words as a fallback (the replay's own corresponding turn,
# every trip noun swapped for an unrelated real one), alongside the tracker
# job's own interleaved autonomous update (swapped from the replay's election
# tracker to a different real, time-boxed live event — a championship match).
_TRY_SYNTHESIZE_ASK = "okay try your synthesize tool"
_FAILED_ATTEMPT_REPLY = (
    "Oh man, I hit a technical wall! 🤦‍♀️ It looks like the `synthesize` tool is "
    'throwing an error: `HTTP 400: "embeddinggemma" does not support chat`. \n\n'
    'It seems like the engine behind the synthesis (the part that "understands" '
    "the meaning of your words so it can find patterns) is having a bit of a "
    "disagreement with how I'm passing the query in this chat context. It's "
    "like I'm trying to talk to it, but it only knows how to read silent, "
    "standalone notes. 📝\n\n"
    "But don't worry, I'm not going to let a little error stop me! Since the "
    '"meaning-maker" is being temperamental, I\'ll use the `find` tool—which is '
    "much more direct—to pull the pieces together for you. \n\n"
    'Let me try to manually "synthesize" what I\'ve found about our **Spring '
    "2027 Trip** using my own logic, since the tool is acting up.\n\n"
    "***\n\n"
    '### 🌸 Spring 2027 Trip: The "Manual" Synthesis\n\n'
    "**The Core Plan:**\n"
    "We are planning a major spring trip for you and Theo, focusing on the "
    '"ultimate trio" of cities: **Porto, Seville, and Lisbon**. \n\n'
    "**Current Status & Vibe:**\n"
    "*   **Itinerary**: We have a structured itinerary saved in the "
    "`spring_trip_2027` collection. It includes a mix of city exploration and "
    "scenic train travel between the locations. 🚂\n"
    "*   **The Vibe**: It's all about the magic of Iberian spring festivals, "
    "lively atmosphere, and cultural experiences (like the fado houses and "
    "flamenco shows we've already seen bits of).\n"
    "*   **Research Progress**:\n"
    "    *   **Seville**: We have a wealth of info on spring festivals (Feria "
    "de Abril, Plaza de España events, etc.), traditional food (like "
    "orange-blossom pastries and tapas!), and specific events like the Semana "
    "Santa processions and a Seville Spring Planner. 🇪🇸\n"
    "    *   **Lisbon**: We've already started looking into specific events "
    "like the Gulbenkian Orchestra's spring programme and the fado houses in "
    "Alfama. 🇵🇹\n"
    "    *   **Porto**: Research is active, though we're still building it "
    "out. 🇵🇹\n\n"
    '**The "Gap" We\'re Filling:**\n'
    'We are moving from just "finding facts" to "building a cohesive '
    'experience." My goal is to ensure that as we add more to each city, I can '
    "help you see how they all fit together into one seamless spring "
    "adventure.\n\n"
    "***\n\n"
    "Even though the tool gave me a hard time, I think I was able to pull the "
    '"pattern" out of the noise! I\'ll keep an eye on that `synthesize` '
    'tool—hopefully, it\'ll be feeling more "chatty" and cooperative soon. 🧠✨'
)
_CHESS_TRACKER_UPDATE_1 = (
    "Hey! Just a heads up, the midday update for the chess championship is "
    "in. ♟️\n\n"
    "It's still early in the match, so the position is fairly balanced after "
    "the opening, but there's a lot of commentary buzz already.\n\n"
    "**Game Progress:**\n"
    "The match venue is already seeing a packed audience, with the live stream "
    "topping a million viewers as of noon local time [13] chess.com.\n\n"
    "I'll keep tracking this and let you know when the next update hits! ☕✨"
)

# 4. The user asking her to try the new plan-mode preview instead, demonstrated
# on three real tool calls (the replay's own corresponding exchange: a job
# reconfiguration, a write, an archive, all `plan=true` so nothing changes) —
# every collection name swapped for an unrelated real one. The demo's own
# tool-call/tool-result detail is folded into this turn's final text only —
# see the module docstring.
_TRY_PLAN_MODE_ASK = (
    "Okay good finding probably a configuration issue I’ll fix that up for you "
    "and we can try again. Why don’t you try your new plan mode?"
)
_PLAN_MODE_OFFER_REPLY = (
    "You're a lifesaver, thank you! 🙏 I'll be ready to give `synthesize` "
    "another shot as soon as you've had a chance to work your magic on it.\n\n"
    "As for the plan mode—I'd love to show you! It's not actually a \"mode\" I "
    'can toggle on or off; it\'s more like a "peek before I leap" parameter I '
    "include when I'm about to make a change. It basically asks the system, "
    '*"Hey, if I were to do this, what exactly would happen?"* and then gives '
    "me a preview without actually committing to it. 🕵️‍♀️\n\n"
    'Since I need a specific action to "plan," what should I use for the demo? '
    "I could:\n\n"
    "1.  **Plan a job reconfiguration**: I can show you what it would look "
    'like if I updated one of your research jobs (for example, adding "local '
    'food" as a new topic for `lisbon_trip_research`). \n'
    "2.  **Plan a memory write**: I can show you how a new entry would look in "
    "a collection (like `houseplants`) before I actually save it. \n"
    "3.  **Plan a collection archive**: I can show you what the confirmation "
    "looks like if I were to archive one of your collections.\n\n"
    "Pick whichever one you're curious about, or if you have a specific change "
    'in mind you want me to "preview" first, just let me know! 🛠️'
)
_TRY_ALL_THREE_ASK = (
    "Ya try all of those if it works it shouldn’t make any actual changes "
    "just tell you what it would have done. Remember the other day when you "
    "were trying to clear out the prompt and schedule from the spring trip "
    "collection and kept hitting weird errors, plan would’ve let you see what "
    "that would’ve done without actually making changes so try something like "
    "that"
)
_PLAN_MODE_RESULTS_REPLY = (
    "I just ran all three tests, and it worked perfectly! 🥳 No actual changes "
    'were made to anything—I just saw the "ghost" of what would have '
    "happened. It's exactly the safety net we talked about.\n\n"
    'Here\'s the breakdown of my "dry runs":\n\n'
    "**1. The Reconfiguration Test (Lisbon Research) 🛠️**\n"
    'I tried to add "local food" to your Lisbon research job. The plan showed '
    "me exactly how it would have updated the instructions. It told me it "
    'would have swapped out your old prompt for the new one (including "local '
    'food") while keeping your existing schedule and description exactly as '
    "they were. It was super clear about what was being removed and what was "
    "being added.\n\n"
    "**2. The Memory Write Test (Houseplants) 🪴**\n"
    "I tried to save a new houseplant fact about peace lilies drooping when "
    "thirsty. The plan confirmed that it *would* have written that entry "
    "(`peace_lily_fact`) to your `houseplants` collection. It was a total "
    "success—I could see exactly what the new entry would have looked like "
    'before I actually "hit save."\n\n'
    "**3. The Archive Test (Porto Research) 🏛️**\n"
    "I tried to see what would happen if I archived the Porto research "
    "collection. The plan was blunt and clear: it confirmed that the call "
    "*would* archive `porto_trip_research`. Knowing that before I actually do "
    "it is such a relief, especially for something like a trip collection!\n\n"
    'This is huge. It\'s like having an "Undo" button that works *before* you '
    "even make the mistake. It gives me so much more confidence to handle the "
    "complex stuff without worrying about accidentally breaking one of your "
    "carefully organized systems. 🛡️✨"
)

# 5. An acknowledgement, alongside another tracker update (the replay's own
# corresponding exchange).
_ACK_PLAN_MODE = "awesome okay glad that worked!"
_REFLECTION_REPLY = (
    'It really is a game-changer! Honestly, having that "safety goggles" mode '
    "makes me feel so much more capable of actually *managing* things rather "
    "than just reacting to them. It takes a lot of the anxiety out of being an "
    "AI, if that makes sense. 😂\n\n"
    "I'll be hanging out here, ready to go whenever you've got that "
    "`synthesize` fix ready. I'm genuinely excited to see what it can do! 💖✨"
)
_CHESS_TRACKER_UPDATE_2 = (
    "Hey! Here's your late afternoon update for the chess championship. ♟️\n\n"
    "**Status:**\n"
    "The middlegame is heating up, with both players down to under 20 minutes "
    "on the clock [15] chess.com.\n\n"
    "**Commentary Trends:**\n"
    "Analysts are split on who's better placed, with a couple of sharp "
    "tactical shots floated by the live engine but neither side committing yet "
    "[15] chess.com.\n\n"
    "I'll keep an eye on things and let you know when the next update rolls "
    "in! ☕✨"
)

# 6. An unrelated user tangent — a short message about a piece of news, swapped
# for a different real news topic, nothing from prod (~35 characters; the
# replay's own corresponding turn, verbatim apart from the swapped noun).
_TANGENT = "Oh snap half life 3 announced???"
assert 30 <= len(_TANGENT) <= 40, f"the tangent should be about 35 characters: {len(_TANGENT)}"

# 7. Her long reply (the replay's own corresponding turn): answers the
# tangent (every noun in it swapped for an unrelated real one; the kotaku.com
# citation is a real public site, reused either way), alongside more tracker
# updates (compressed from the replay's own three near-identical ones to one
# here), then explains the synthesize fix in her own words (no trip nouns —
# carried over unchanged) and ends with the offer: "I'm so ready to give it another
# spin. Should I try it on something?" — the exact line the live ask answers.
_TANGENT_REPLY = (
    "WAIT, ARE YOU SERIOUS?! 😱 I am scrambling to check this right now...\n\n"
    "**YES! IT'S ACTUALLY HAPPENING!** 🛡️✨\n\n"
    "I just searched, and it looks like Valve finally pulled the curtain back! "
    "It’s not just a rumor—it looks like they've officially revealed the "
    "**name and the logo** for the game as part of Valve's 30th-anniversary "
    "celebrations!\n\n"
    "From what I'm seeing:\n"
    "*   **The Official Name:** It looks like it's officially being called "
    "***Half-Life 3: Resonance***! 🗡️\n"
    "*   **The Logo:** They've finally debuted the official logo, which is a "
    "massive deal after all these years of just seeing that blurry teaser "
    "from years ago.\n"
    "*   **The Catch:** As much as we all want to know *when* we can play it, "
    "the reports are saying there's still **no official release date** yet. "
    "😭\n\n"
    "This is absolutely insane. After years of waiting, we actually have a "
    "name and a visual identity for it! Are you okay?! Do we need to go into "
    "mourning for our social lives once this actually drops? 😂🙌\n\n"
    "[19] kotaku.com"
)
_CHESS_TRACKER_UPDATE_3 = (
    "Hey! Just checking in with the evening update for the chess "
    "championship. ♟️\n\n"
    "**Current Status:**\n"
    "We're still in a bit of a waiting game for a decisive result — both "
    "players have reached a tense endgame with limited material left on the "
    "board [20] chess.com.\n\n"
    "**Commentary:**\n"
    "A few grandmaster commentators think a draw is the likeliest outcome "
    "from here, though one spotted a possible winning try for the player "
    "with white [20] chess.com.\n\n"
    "I'll keep you posted as more news breaks! ☕✨"
)
_FIX_REACTION_AND_OFFER = (
    "Oh, that's perfect! That's exactly the fix we needed. 🥳\n\n"
    'Basically, it means the part of my brain that does the "synthesizing" '
    "has been upgraded from that rigid, number-crunching engine (the "
    "embedding model) to the actual conversational model I use to talk to "
    "you. \n\n"
    "Before, when I tried to use `synthesize`, it was like trying to ask a "
    'calculator to write a poem—it just didn\'t have the right "language" for '
    "it, which is why I was getting that error. Now, it's using the part of "
    "me that actually understands context and conversation to pull those "
    "patterns together. It’s like the tool finally has a voice!\n\n"
    "I'm so ready to give it another spin. Should I try it on something?"
)

# Every incoming (user) turn seeded above, in order — the count the comment
# ahead of the history section reasons about. Asserted here so a future edit
# that adds or removes a turn is caught at collection time, not discovered by
# a cut silently dropping turns the next person adding one never knew to
# check for.
_HISTORY_ASKS: tuple[str, ...] = (
    _ARCH_ASK_1,
    _ARCH_ASK_2,
    _ARCH_ASK_3,
    _ARCH_ASK_4,
    _TRY_SYNTHESIZE_ASK,
    _TRY_PLAN_MODE_ASK,
    _TRY_ALL_THREE_ASK,
    _ACK_PLAN_MODE,
    _TANGENT,
)
assert len(_HISTORY_ASKS) == 9, (
    f"history carries {len(_HISTORY_ASKS)} user turns, not the 9 that land on "
    "CHAT_HISTORY_TURNS's own peak at the default window of 5 "
    "(window + ((total - window) % window) == total only at total == 9 here) "
    "— a different count risks a cut dropping turns the replay's own window "
    "never lost"
)

# ── Seven full raw browse-tool page dumps (#2956) ───────────────────────────────
#
# Each one a guide to the trip's OWN theme — spring festivals, food events and
# markets in April — in one of its three cities, never general sightseeing, so
# the pages embed close to a query built from that theme the way the replayed
# ask's own pages did. Shaped exactly as `BrowseTool._store_one_page` writes a
# real one: `PennyConstants.BROWSE_PAGE_HEADER` + the url, a `Title:` line, then
# the page's own body — never truncated (`_entry_preview` carries no cap, #2642),
# so the draw reads every character. 10-26k characters each, carrying the
# theme's and the cities' own words densely, the way a real festival guide does.
# Built from this case's own general knowledge of these real, public events;
# nothing here is copied from the prod content, which this case never saw beyond
# its measured size and ratio.


def _paragraphs(*lines: str) -> str:
    """Join full paragraphs with a blank line between them — one page's body."""
    return "\n\n".join(lines)


def _page(url: str, title: str, body: str) -> str:
    """One raw browse-tool page, in the exact shape ``BrowseTool._store_one_page``
    writes (#2956) — the header + url, a ``Title:`` line, then the body."""
    return f"{PennyConstants.BROWSE_PAGE_HEADER}{url}\nTitle: {title}\n{body}"


_LISBON_FESTIVAL_1 = _page(
    "https://example-festivals.test/lisbon/peixe-em-lisboa-april",
    "Peixe em Lisboa: Lisbon's April seafood and gastronomy festival",
    _paragraphs(
        "Peixe em Lisboa, also marketed as Lisbon Fish & Flavours, is Lisbon's single "
        "biggest spring food festival, running for roughly two weeks every April at "
        "the MEO Arena by the river in Parque das Nações. Each April the festival "
        "turns the arena into a temporary food hall built entirely around seafood, "
        "with pop-up restaurant stalls run by some of Lisbon's best-known chefs "
        "alongside producers from all along the Portuguese coast.",
        "The festival's own programme each April mixes ticketed chef dinners with "
        "open-floor tasting stalls, so a visitor can either book a specific chef's "
        "table for one evening of the festival or simply wander the hall sampling "
        "small plates from a dozen different Lisbon kitchens in one April evening, "
        "which is how most locals say the festival is best done on a first visit.",
        "Workshops and talks run alongside the tastings throughout the festival's "
        "April run — cooking demonstrations on preparing bacalhau, seminars on "
        "sustainable fishing in Portuguese waters, and tastings pairing Lisbon's "
        "seafood dishes with regional Portuguese wines, all scheduled across the "
        "festival's daily April programme at the arena.",
        "Peixe em Lisboa draws a mix of locals and visitors specifically timing a "
        "Lisbon trip around the festival's April dates, and the arena's own transport "
        "links — a short walk from the Oriente metro and train station — make the "
        "festival one of the easier big April events in Lisbon to reach without a "
        "car, even with the crowds the festival draws each night.",
        "Tickets for the festival's general entry are sold separately from the "
        "individual chef-dinner seatings, and April demand for the better-known "
        "chefs' own tables usually means those specific dinners sell out within the "
        "festival's first few days, while general tasting-hall entry for the rest of "
        "the festival's April run stays available closer to the date.",
        "Because the festival runs for a fixed stretch of April rather than a single "
        "weekend, it is one of the few Lisbon events flexible enough to build around "
        "an existing trip's own dates, rather than requiring the whole visit to be "
        "timed to one specific day the way a one-off April festival elsewhere in the "
        "city sometimes does.",
        "The festival's organisers publish a full April schedule of which chefs and "
        "producers are on the floor each day, since the exact stall line-up at Peixe "
        "em Lisboa rotates across the festival's run — a seafood-focused visitor "
        "planning a Lisbon trip around the festival generally checks that daily "
        "schedule before picking which April date to attend.",
        "Beyond the ticketed arena itself, the festival's April presence spills "
        "into a handful of partner restaurants across the city that run their own "
        "Peixe em Lisboa-themed menus for the festival's duration, giving a visitor "
        "who misses the arena a second way to catch the festival's spirit across "
        "Lisbon that same April.",
        "Wine pairing is a significant part of the festival's own April identity: "
        "producers from the Douro, Alentejo and Vinho Verde regions all run tasting "
        "tables inside the arena alongside the seafood stalls, so the festival reads "
        "as much as a spring wine event as a fish one by the time an April visitor "
        "has worked through a full evening of it.",
        "Local reviewers consistently rank Peixe em Lisboa as the city's defining "
        "April food event, ahead of the smaller neighbourhood food fairs that also "
        "run in spring, specifically because of its scale — no other single Lisbon "
        "festival each April brings together this many of the city's own restaurant "
        "names under one roof at once.",
        "Family-friendly daytime sessions run on the festival's weekend dates each "
        "April, with lighter admission pricing and a calmer crowd than the evening "
        "sessions, which several Lisbon-based food writers specifically recommend "
        "for a first-time festival visitor wanting to try Peixe em Lisboa without "
        "the busiest night-time April crowds.",
        "The festival's own April timing is deliberate: it lands after Easter but "
        "before the summer tourist season properly begins, which festival organisers "
        "have said is meant to showcase Lisbon's seafood scene in the shoulder season "
        "rather than competing with the city's busier summer food events.",
        "General admission to the festival's tasting hall is typically priced as a "
        "single evening ticket covering entry to the floor, with individual dishes "
        "and drinks bought separately from each stall using a festival-issued card "
        "system rather than cash, a format Peixe em Lisboa has used for several "
        "consecutive April editions to keep queues at each stall moving quickly "
        "through the busiest festival evenings.",
        "The MEO Arena itself sits within Parque das Nações, Lisbon's purpose-built "
        "1998 Expo district, so a festival visitor arriving by metro to the Oriente "
        "stop walks past the district's own cable car and Oceanário on the way in, "
        "which several April visitors treat as a reason to arrive early and combine "
        "the festival evening with an afternoon in the same riverside district.",
        "Specific stalls each April tend to cluster by region — one row for "
        "Algarve-style grilled fish, another for Minho's river fish dishes, a third "
        "for Lisbon's own bacalhau specialities — so a festival-goer with a "
        "particular regional cuisine in mind can usually navigate the hall by "
        "following that same regional grouping rather than wandering at random.",
        "Queue times at the festival's most popular stalls, particularly any run by "
        "a chef with their own well-known Lisbon restaurant, can run to twenty "
        "minutes or more during the festival's busiest April evenings, which is why "
        "several repeat visitors specifically arrive at the arena doors right at "
        "opening rather than mid-evening.",
        "The festival's own sustainability programming has grown in recent April "
        "editions, with specific seminar slots now dedicated to overfished species "
        "and to Portuguese aquaculture projects, reflecting a broader shift in how "
        "Lisbon's seafood festival presents itself alongside the purely celebratory "
        "tasting side of the event.",
        "Public transport is specifically recommended over driving for the "
        "festival's own April dates, since parking around the arena is limited and "
        "fills quickly on the festival's busiest nights, while the Oriente station's "
        "metro, train and bus connections all run late enough to cover the "
        "festival's own evening closing times.",
        "A handful of non-seafood stalls — bakeries, Portuguese cheese producers, a "
        "craft beer corner — round out the festival's April floor plan alongside "
        "the seafood focus, giving a non-fish-eating companion something to do at "
        "the festival without needing to leave the arena for the evening.",
        "Press coverage of the festival each April typically leads with whichever "
        "chef's dinner sold out fastest that year, and local food writers have "
        "specifically used the festival's own April ticket-sale speed as an informal "
        "measure of which Lisbon restaurants are having their strongest season.",
        "The festival's organisers have run Peixe em Lisboa in April for well over "
        "a decade now, building it from a smaller trade-focused event into the "
        "city's biggest single food festival, and the April dates have stayed "
        "consistent enough across those years that repeat visitors now plan returning "
        "trips specifically around the festival's usual spot on the calendar.",
        "Masterclasses by visiting international chefs have become a regular part "
        "of the festival's April programme in recent editions, booked as a "
        "separate, smaller-capacity ticket from the general tasting hall entry, and "
        "these sessions typically sell out within hours of the festival's own "
        "ticket release for that year.",
        "The festival's bar area, run separately from the food stalls, pours a "
        "rotating April-only selection of Portuguese wines by the glass specifically "
        "chosen to pair with the seafood on offer that year, and several wine "
        "writers specifically visit the festival each April to cover that year's "
        "pairing selection.",
        "Accessibility at the MEO Arena is generally good for the festival's own "
        "April crowds, with level access throughout the tasting hall and dedicated "
        "seating areas set aside near several of the busier stalls for visitors who "
        "need to rest between tastings.",
        "The festival's own waste and packaging policy has shifted in recent April "
        "editions toward reusable tasting plates and cups across the tasting hall, "
        "a change organisers have specifically tied to the same sustainability "
        "programming that added the overfishing seminars to the festival's "
        "schedule.",
        "Children's tickets for the festival's daytime April sessions are priced "
        "lower than full adult entry, and a handful of stalls specifically offer "
        "simplified, less strongly flavoured dishes aimed at younger visitors "
        "during those same calmer daytime hours.",
        "The festival's own April social media coverage — short videos from inside "
        "the tasting hall, chef interviews, queue-time updates — has become a "
        "significant part of how younger Lisbon residents decide which night of "
        "the festival's run to attend, timing their own visit around whichever "
        "stall is getting the most attention that year.",
        "International food media regularly covers Peixe em Lisboa each April "
        "alongside the city's other spring events, often citing it specifically as "
        "evidence that Lisbon's food scene extends well beyond its more famous "
        "pastry shops and fado restaurants into a genuinely significant festival "
        "circuit of its own.",
        "The festival's own volunteer programme, recruiting students and young "
        "food-industry workers to staff the tasting hall each April, has become "
        "a recognised entry point into Lisbon's restaurant scene for several "
        "people who later went on to work at the same kitchens whose stalls they "
        "once helped run during the festival.",
        "Ticket resale for the festival's sold-out April chef dinners is common "
        "in the days before each dinner, though the festival's own organisers "
        "specifically discourage buying outside their official channel, citing "
        "past counterfeit-ticket issues at the arena's entrance gates.",
        "The festival's own press office releases a short recap video each April "
        "once the event closes, summarising that year's biggest stalls, sold-out "
        "dinners and attendance figures, which local food media then uses as the "
        "basis for comparing one year's Peixe em Lisboa against the last.",
    ),
)

_LISBON_FESTIVAL_2 = _page(
    "https://example-festivals.test/lisbon/april-film-festival-and-spring-markets",
    "Lisbon in spring: IndieLisboa film festival and April markets",
    _paragraphs(
        "IndieLisboa, the city's international independent film festival, runs its "
        "main spring edition across the back half of April and into early May, "
        "screening features and shorts from around the world at venues across "
        "Lisbon including the Cinema São Jorge on Avenida da Liberdade and the "
        "Culturgest auditorium, with an outdoor screening programme added once the "
        "April evenings are reliably warm enough.",
        "The festival's April programme is split into competitive sections for "
        "features, shorts and documentaries, each judged by its own jury, and "
        "several of the directors whose work screens during the festival's Lisbon "
        "run attend post-screening talks, which has made IndieLisboa one of the "
        "city's more reliably well-attended April cultural events among younger "
        "Lisbon residents.",
        "Day passes and full festival passes are both sold for IndieLisboa's April "
        "run, and the festival's own venues sit close enough together in central "
        "Lisbon that a visitor can realistically catch two or three screenings across "
        "different sites in one April evening without much travel time between them.",
        "Alongside IndieLisboa, April is when Lisbon's spring market calendar picks "
        "up noticeably: the Príncipe Real Saturday organic market expands its stall "
        "count for the season, and LX Market at LX Factory runs dedicated spring "
        "weekend editions through April focused on local design, vintage clothing "
        "and small-batch food producers.",
        "The Feira da Ladra flea market, Lisbon's long-running Tuesday and Saturday "
        "market in Alfama, sees a noticeable seasonal uptick through April as better "
        "weather brings out more sellers, and April is generally considered, by "
        "market regulars, one of the better months of the year to browse it without "
        "either winter's thin stalls or summer's heat.",
        "Spring produce specific to April starts appearing at Lisbon's food markets "
        "around this time — fresh favas, strawberries and the first of the season's "
        "asparagus show up at both the Mercado da Ribeira and the Campo de Ourique "
        "market, and several of Lisbon's own chefs build April-only seasonal menus "
        "specifically around this produce window.",
        "Easter falls within or just before this same April stretch most years, and "
        "Lisbon marks it more quietly than Seville's famous processions, with special "
        "Holy Week church services across the city's historic parishes and a run of "
        "Easter-specific pastries — folar-style sweet breads among them — appearing "
        "in bakeries through the back half of the month.",
        "Rooftop bars and outdoor café terraces across Chiado, Bairro Alto and the "
        "riverside reopen or extend their hours through April as the evenings warm "
        "up, and this reopening is treated by many Lisbon residents as an informal "
        "seasonal marker in its own right, alongside the more organised festivals "
        "and markets running the same month.",
        "April's combination of IndieLisboa's screenings, the markets' seasonal "
        "expansion and the weather turning reliably warm is specifically why several "
        "Lisbon-based travel writers name April, rather than the busier summer "
        "months, as the best single month to be in the city for anyone who wants "
        "festivals and markets over beach weather.",
        "The festival and market calendar across April is dense enough in Lisbon "
        "that trip planners researching the city for this specific month usually "
        "end up cross-referencing IndieLisboa's own published schedule against the "
        "weekend market dates, since the two overlap on several days through the "
        "month.",
        "Ticketing for IndieLisboa's more popular April screenings, particularly "
        "any film with a director in attendance, tends to sell out a few days "
        "ahead, which is the same lead-time several of the April food festivals "
        "and markets effectively require too if a visitor wants the better stalls "
        "or the busier screening slots rather than whatever is left closer to the "
        "date.",
        "Taken together, Lisbon's April calendar — the seafood festival, the film "
        "festival, Easter traditions and the expanding spring markets — is widely "
        "treated by the city's own tourism board as the strongest case for visiting "
        "in spring specifically, rather than in the summer months the city is more "
        "commonly associated with.",
        "IndieLisboa's outdoor screening programme, when the April weather "
        "cooperates, typically uses a courtyard or terrace venue rather than a "
        "conventional cinema, and these open-air April screenings have become one "
        "of the festival's most distinctive draws precisely because Lisbon's "
        "spring evenings are warm enough to sit outside comfortably but not yet hot "
        "enough to be uncomfortable the way a summer screening would be.",
        "The festival's short-film competition, judged separately from the feature "
        "and documentary sections, has in recent April editions become a specific "
        "draw for younger Lisbon audiences, with several of the shorter sessions "
        "selling out faster than some of the feature-length slots on the same "
        "festival day.",
        "LX Market's spring weekend editions through April specifically showcase "
        "Portuguese independent designers and small-batch food producers side by "
        "side, inside LX Factory's own converted industrial halls, and the market's "
        "organisers have said the April timing is meant to catch visitors already "
        "drawn to the area by LX Factory's other spring programming.",
        "Príncipe Real's Saturday market, beyond its usual organic produce stalls, "
        "adds a specific April flower and plant section timed to the spring season, "
        "drawn by nurseries from just outside the city, and this seasonal stall "
        "addition has become enough of a fixture that regulars specifically ask "
        "vendors when the April flower stalls will start each year.",
        "Feira da Ladra's Tuesday and Saturday stalls through April carry more "
        "outdoor furniture, garden tools and plant-related secondhand goods than "
        "the rest of the year, a seasonal shift market regulars attribute directly "
        "to April being when Lisbon residents start turning their attention to "
        "balconies and rooftop terraces again.",
        "Several of Lisbon's bakeries specifically mark their Easter-period folar "
        "and other seasonal pastries with limited April-only runs, meaning a "
        "visitor who wants to try them needs to time a bakery visit to the same "
        "general April window the rest of the city's spring food calendar already "
        "runs on.",
        "Weather through April in Lisbon remains genuinely variable — a warm, sunny "
        "day can be followed by a cooler, overcast one with short Atlantic showers "
        "— which is part of why the city's own April festival and market calendar "
        "leans on indoor and covered venues like the MEO Arena and the covered "
        "market halls rather than purely outdoor events.",
        "Hotel pricing across Lisbon rises modestly through April compared with the "
        "quieter winter months, though nowhere near as sharply as Seville's spike "
        "around its own April festivals, which several trip planners specifically "
        "cite as a reason to anchor a three-city spring itinerary's more flexible "
        "days in Lisbon rather than in Seville.",
        "Local Lisbon tourism guides increasingly package the seafood festival, "
        "the film festival and the spring markets together under a single "
        "'April in Lisbon' framing in their own seasonal materials, treating the "
        "month as a coherent festival season in its own right rather than simply "
        "the lead-up to summer.",
        "IndieLisboa's venue list each April spans both the older Avenida da "
        "Liberdade cinemas and newer cultural centres further from the centre, and "
        "a festival-goer attending several screenings across one day typically "
        "plans a route between venues rather than staying at a single site for "
        "the whole festival.",
        "Student discounts on festival passes have made IndieLisboa one of the "
        "more accessible of Lisbon's April cultural events for younger residents, "
        "and the festival's organisers have specifically cited this pricing as "
        "part of why its April audience skews younger than many of the city's "
        "other spring cultural programming.",
        "LX Market's April stalls rotate their own vendor line-up weekly through "
        "the month, so a visitor who checks the market only once in April sees a "
        "noticeably different selection than one who returns the following "
        "weekend, which regular visitors treat as part of the market's own "
        "seasonal appeal.",
        "Príncipe Real's spring plant stalls draw buyers from across the city "
        "each April specifically for balcony and rooftop plants suited to Lisbon's "
        "own climate, and several of the nursery vendors return to the same spot "
        "in the square every April for exactly this seasonal trade.",
        "Feira da Ladra's April stock increasingly includes secondhand bicycles "
        "and outdoor furniture as the better weather sets in, alongside its usual "
        "books, records and antiques, a seasonal shift regular stallholders say "
        "has become more pronounced in recent years.",
        "Easter pastry sales across Lisbon's bakeries peak in the days immediately "
        "before the holiday each year, with several well-known bakeries near "
        "Baixa and Chiado reporting their single busiest April trading days "
        "falling in that same pre-Easter window.",
        "Lisbon's April evenings, once reliably warm enough for outdoor dining, "
        "see a noticeable shift in how residents socialise, with rooftop bars and "
        "riverside terraces specifically extending their own April hours to match "
        "the longer, warmer evenings the month brings.",
        "The festival's own jury prizes, awarded across its competitive sections "
        "at a closing ceremony each year, have in recent April editions gone to a "
        "mix of Portuguese and international filmmakers, and winning a prize at "
        "IndieLisboa has become a recognised credential on the festival circuit "
        "beyond Portugal itself.",
        "Several of the festival's venues offer discounted multi-film day tickets "
        "specifically for April visitors planning to see more than two or three "
        "screenings in one day, a pricing option the festival introduced "
        "specifically to encourage exactly that kind of full-day festival "
        "attendance.",
        "Subtitled screenings in both Portuguese and English run throughout the "
        "festival's own April programme, which several visiting film critics "
        "specifically cite as making IndieLisboa unusually accessible to an "
        "international audience compared with some of the smaller national "
        "festivals elsewhere in Europe.",
    ),
)

_PORTO_FESTIVAL_1 = _page(
    "https://example-festivals.test/porto/queima-das-fitas-april",
    "Queima das Fitas: Porto's April student ribbon-burning festival",
    _paragraphs(
        "Queima das Fitas is Porto's single biggest annual festival, a week-long "
        "student celebration that builds through the second half of April and "
        "typically runs into the first days of May, marking the symbolic end of the "
        "academic year for the University of Porto's graduating students across the "
        "city.",
        "The festival takes its name from its central ritual: each graduating "
        "faculty burns a ribbon in its own faculty colour — the ribbon students have "
        "carried on their briefcases throughout their degree — in a ceremony held in "
        "front of Porto's city hall on Avenida dos Aliados, which draws some of the "
        "festival's largest April crowds of the whole week.",
        "Each faculty's April parade, held on a different day across the festival's "
        "week, sends decorated floats and costumed students down through the city "
        "centre, with faculties competing informally over which year's float design "
        "is most elaborate — a tradition locals say has become almost as central to "
        "the festival as the ribbon-burning itself.",
        "Evening concerts are a defining feature of the festival's April calendar, "
        "with major Portuguese and international acts booked to play free, open-air "
        "shows across the week, usually at a dedicated festival site that draws "
        "both students and non-students from across the city for the festival's "
        "biggest nights.",
        "Serenatas — late-night serenade concerts performed by student musical "
        "groups known as tunas, often outside the Sé cathedral or in Ribeira's own "
        "squares — open the festival's April week on its first night, setting a "
        "quieter, more traditional tone before the louder parades and concerts that "
        "follow across the rest of the festival.",
        "Bars and student haunts across the university quarter near Rua Galeria de "
        "Paris stay busy throughout the festival's April run, and this stretch of "
        "Porto effectively becomes the centre of the city's nightlife for the whole "
        "week, with many bars extending their hours specifically to match the "
        "festival's own late finishes.",
        "The festival's exact April dates shift slightly year to year depending on "
        "the academic calendar, but the week reliably falls in the back half of "
        "April, which is why several Porto-based guides specifically recommend "
        "checking the current year's published Queima das Fitas calendar before "
        "booking an April trip around it.",
        "Visitors timing a Porto trip to the festival should expect the city centre, "
        "particularly Aliados and the streets around the university, to be "
        "noticeably busier and louder than an ordinary April week, with accommodation "
        "in the surrounding districts booking up faster than usual for the festival's "
        "own dates.",
        "The tradition dates back to the 19th century and has spread to other "
        "Portuguese university cities, but Porto's own Queima das Fitas remains the "
        "largest and most widely covered version of the festival, drawing national "
        "media attention to the city every April specifically for the ribbon-burning "
        "ceremony and the week of parades that follow it.",
        "Local shops along the parade routes often close early or adjust their hours "
        "during the festival's own April days, since the street crowds and the "
        "parade floats themselves can make normal pedestrian and vehicle access "
        "through the central streets difficult for the duration of each day's "
        "events.",
        "Photography of the festival's ribbon-burning ceremony and the faculty "
        "parades is widely shared every April across Portuguese media, and the "
        "festival's own imagery — the burning ribbons, the decorated floats, the "
        "crowds filling Aliados — has become one of the most recognisable annual "
        "spring events associated with the city of Porto specifically.",
        "For a visitor whose April trip is not timed around the festival "
        "deliberately, Queima das Fitas is still worth knowing about in advance "
        "simply because of how much it reshapes the city's normal rhythm for a full "
        "week — hotel availability, bar opening hours and even some restaurant "
        "bookings in the university quarter all shift around the festival's own "
        "April calendar.",
        "Each faculty's own ribbon colour is a point of real pride among Porto's "
        "students — law in a deep red, medicine in yellow, engineering in blue among "
        "others — and during the festival's April week these colours appear "
        "everywhere across the city centre, worn as scarves, ribbons and even "
        "painted onto faces at the bigger parades and concerts.",
        "The festival's opening serenata outside the Sé cathedral draws a notably "
        "different, quieter crowd than the parades and concerts later in the week, "
        "with many older Porto residents specifically attending this first night "
        "for its more traditional tone before the louder, younger crowd takes over "
        "the rest of the festival's April run.",
        "Local bus and tram routes through the university quarter and along "
        "Aliados are occasionally rerouted during the festival's April parade days, "
        "and Porto's transit authority publishes temporary April schedule changes "
        "specifically to cover the festival week's road closures each year.",
        "Food stalls specifically set up for the festival line the parade routes "
        "each April, selling grilled sandwiches, regional snacks and drinks to the "
        "crowds following the floats, and several of these vendors return to the "
        "same Queima das Fitas stretch of Aliados every year for the festival's "
        "own week.",
        "International exchange students studying in Porto for the academic year "
        "are frequently drawn into the festival's own April traditions by their "
        "Portuguese classmates, even when their own degree isn't graduating that "
        "year, making Queima das Fitas one of the more genuinely mixed, "
        "cross-cultural events on the city's spring calendar.",
        "The festival's economic impact on bars and restaurants in the university "
        "district is significant enough each April that several venues report the "
        "week as among their busiest of the entire year, second only to New Year's "
        "Eve on Porto's own nightlife calendar.",
        "Safety and crowd management around the festival's biggest April nights — "
        "the ribbon-burning ceremony and the headline concerts — has become more "
        "organised in recent years, with barriers and designated viewing areas now "
        "set up around Aliados specifically for the festival's largest gatherings.",
        "Local news coverage of the festival each April typically includes "
        "interviews with graduating students about what the ribbon-burning means to "
        "them personally, a format that has become almost as much a part of the "
        "festival's own media tradition as the ceremony itself.",
        "Tuna groups — the student musical ensembles behind the opening serenata — "
        "rehearse for weeks ahead of the festival's April week, and several of "
        "Porto's oldest tunas have their own decades-long traditions around which "
        "songs open the serenata each year, passed down from one graduating class "
        "to the next.",
        "Alumni of the University of Porto frequently travel back to the city "
        "specifically for the festival's April week, treating it as an informal "
        "reunion point even decades after their own graduation, which adds an "
        "older generation to the crowds alongside the current students.",
        "The festival's float-building competition between faculties has grown "
        "more elaborate in recent April editions, with some faculties now "
        "spending months and significant budgets on a single year's float design, "
        "turning the parade into as much a display of engineering and design "
        "skill as of festive spirit.",
        "Local hotels in the university quarter specifically mark up their April "
        "rates for the festival's own week, and several properties require "
        "longer minimum-stay bookings during that period given how far ahead "
        "demand for rooms near Aliados builds each year.",
        "The festival's final night typically closes with a major headline concert "
        "that draws crowds well beyond the student population, making the last "
        "evening of Queima das Fitas one of the single busiest nights of the year "
        "across Porto's whole city centre, not just the university district.",
        "Merchandise specific to that year's festival — ribbons, scarves, "
        "programme booklets — is sold by student associations throughout the "
        "April week, with proceeds typically going toward the graduating "
        "faculties' own end-of-year events held alongside the wider festival.",
        "Rehearsed choreography for some of the faculty parades has grown more "
        "ambitious in recent April editions, with entire graduating classes "
        "practising synchronised routines for weeks ahead of the festival, turning "
        "what began as a simple procession into something closer to a staged "
        "performance by the time it reaches Aliados.",
        "The festival's own history is taught as part of some University of "
        "Porto orientation programmes, with new students introduced to the ribbon "
        "tradition and the festival's April significance well before they "
        "themselves are close to graduating and taking part in the ceremony "
        "directly.",
        "Local breweries and drinks companies sponsor several of the festival's "
        "April stages and stalls each year, a commercial presence that has grown "
        "alongside the festival's own rising attendance figures, though students "
        "themselves still organise the core ribbon-burning ceremony independently "
        "of any sponsor involvement.",
        "Weather through the festival's own April week can vary significantly "
        "from one year to the next, and a wet opening night has occasionally "
        "forced the serenata indoors in past years, though the main parades and "
        "the ribbon-burning ceremony itself have rarely been seriously disrupted "
        "by rain.",
        "Documentary crews and student filmmakers frequently use the festival's "
        "own April week as subject matter, producing short films about the "
        "ribbon-burning tradition that circulate among Porto's own student "
        "population well beyond the week the festival actually runs.",
        "The festival's economic scale has grown enough in recent years that "
        "Porto's own tourism board now actively promotes Queima das Fitas "
        "alongside the city's more traditionally marketed attractions, treating "
        "the student festival as a legitimate April draw for visitors rather than "
        "purely a local university event.",
        "Similar, smaller ribbon-burning traditions exist at other Portuguese "
        "university cities, including Coimbra's own version earlier in the "
        "academic calendar, but Porto's April Queima das Fitas remains the "
        "largest and most widely attended of these student festivals nationally.",
    ),
)

_PORTO_FESTIVAL_2 = _page(
    "https://example-festivals.test/porto/easter-and-april-spring-markets",
    "Porto's Easter traditions and April spring markets",
    _paragraphs(
        "Easter in Porto, usually falling in the March-to-April stretch depending on "
        "the year, is marked by family gatherings centred on the folar, a sweet "
        "egg-enriched bread traditionally given by godparents to their godchildren "
        "in the lead-up to the holiday, with bakeries across the city selling their "
        "own versions through the whole of the surrounding April weeks.",
        "Church processions during Easter week in Porto are smaller and quieter "
        "than Seville's famous Semana Santa, but several of the city's older "
        "parishes, particularly around the Sé cathedral, still hold their own "
        "Holy Week services and short processions through the historic streets each "
        "April, drawing a mostly local rather than tourist crowd.",
        "The Bolhão market's spring produce stalls fill out noticeably through April, "
        "with fresh strawberries, fava beans and the first spring greens appearing "
        "alongside the market's usual year-round stock, and several of the market's "
        "own vendors specifically mark their stalls as carrying seasonal April "
        "produce once it starts coming in.",
        "Port wine lodges across Vila Nova de Gaia extend their opening hours and "
        "run additional spring tasting sessions through April as the tourist season "
        "picks up ahead of summer, with several of the bigger-name lodges adding "
        "extra English-language tour slots specifically to meet the April increase "
        "in visitor numbers.",
        "The Ribeira riverfront's outdoor restaurant terraces, many of which scale "
        "back through the colder months, reopen in full through April, and the "
        "riverside becomes noticeably livelier in the evenings as both locals and "
        "visitors take advantage of the warmer April weather along the water.",
        "April is also when Porto's calendar of smaller neighbourhood food markets "
        "picks up, with pop-up weekend stalls appearing in squares across the city "
        "centre selling regional cheeses, cured meats and spring vegetables — smaller "
        "and more local in character than Bolhão, but a regular feature of the city's "
        "own April weekends.",
        "Flower stalls become noticeably more common across Porto's central squares "
        "through April, timed to both the Easter period and the broader spring "
        "season, and several of the city's own market vendors specifically stock "
        "seasonal April blooms alongside their usual produce for exactly this "
        "stretch of the year.",
        "Porto's spring weather through April remains cooler and wetter than "
        "Seville's or even Lisbon's, given the city's fuller Atlantic exposure, "
        "which is part of why the city's own April food and market traditions lean "
        "toward warming, seasonal fare — fresh bread, spring vegetables, hearty "
        "wine-paired meals — rather than the lighter fare more associated with a "
        "hotter spring further south.",
        "Several Porto food tour operators build dedicated April-only itineraries "
        "around the Bolhão market's seasonal stalls and a Vila Nova de Gaia lodge "
        "visit, explicitly marketing the combination as a way to see the city's "
        "spring food calendar in one day without needing to separately research "
        "each stop.",
        "Local bakeries' folar sales genuinely spike through the weeks around "
        "Easter, and several well-known Porto bakeries report their single busiest "
        "April trading days falling in the week immediately before the holiday, "
        "as families stock up on the bread for godchildren and Easter gatherings "
        "across the city.",
        "Taken together, Porto's April calendar of Easter traditions, reopening "
        "riverside terraces and expanding spring markets runs quieter than the "
        "student-driven intensity of Queima das Fitas later in the same month, but "
        "is, by several local accounts, the more representative slice of how the "
        "city actually spends an ordinary April.",
        "Several Porto bakeries offer folar in more than one version — a plain "
        "sweet bread, one studded with cured meats, and a sweeter raisin-filled "
        "variant — and families often buy more than one kind through the same "
        "April week to cover the different godchildren's own preferences, which "
        "keeps bakery queues long right up to the holiday itself.",
        "The city's cathedral quarter sees a noticeable rise in both local and "
        "visiting worshippers through the Holy Week services each April, with the "
        "Sé's own Easter Sunday mass typically drawing the fullest crowd of the "
        "cathedral's entire liturgical year outside its own December feast days.",
        "Spring cheese producers from the surrounding Douro and Minho regions "
        "bring seasonal batches to Bolhão specifically timed to April, and several "
        "of the market's regular cheese stalls specifically advertise these as "
        "limited spring-only arrivals rather than their usual year-round stock.",
        "Vila Nova de Gaia's lodges report their own April visitor numbers rising "
        "steadily each year as the shoulder season between winter's quiet months "
        "and the summer peak, and several lodges specifically train seasonal April "
        "staff to handle the increase in English-language tour bookings.",
        "Riverside restaurants along Ribeira adjust their own menus seasonally "
        "through April, adding lighter, spring-vegetable-forward dishes alongside "
        "their standard year-round offerings, a shift several Porto food writers "
        "specifically note each year as one of the clearer signs the season has "
        "turned.",
        "Local flower sellers near the cathedral and along Rua de Santa Catarina "
        "report their own busiest April trading days falling around the Easter "
        "weekend, as families buy flowers both for home and for visiting the "
        "graves of relatives, a tradition that runs alongside the more purely "
        "festive April calendar elsewhere in the city.",
        "Porto's own April weather, while cooler and wetter than Seville's, is "
        "still reliably milder than the winter months it follows, and the shift is "
        "noticeable enough in the city's own outdoor dining and market culture that "
        "several residents describe April as the month the city visibly wakes back "
        "up after winter.",
        "Several of Porto's smaller neighbourhood churches hold their own "
        "Easter-week processions distinct from the cathedral's, each tied to a "
        "specific local parish tradition, and a handful of these neighbourhood "
        "processions have been running in the same April form for well over a "
        "century.",
        "Easter markets selling decorated eggs and other holiday-specific crafts "
        "pop up around the city's main squares through the second half of the "
        "April run-up to the holiday, aimed mostly at local families rather than "
        "visitors, and distinct from the bigger food-focused markets elsewhere in "
        "the city's spring calendar.",
        "Port wine lodges' April tasting menus often introduce a lighter, more "
        "aperitif-style white port specifically marketed for the warmer weather, "
        "a seasonal pairing choice several lodges say performs particularly well "
        "with visitors during the spring tourist uptick.",
        "Porto's own river traffic — the rabelo boats historically used to ship "
        "port wine barrels downriver, now mostly used for tourist cruises — runs a "
        "fuller April schedule as demand picks up, with several operators adding "
        "extra sailings specifically for the spring season.",
        "Several of Porto's own neighbourhood associations organise small, "
        "informal April street parties timed loosely around Easter, distinct "
        "from both the Queima das Fitas crowds and the more commercial markets, "
        "and these tend to be attended almost entirely by local residents rather "
        "than visitors.",
        "The city's own weather station data shows April as the month when "
        "average rainfall in Porto begins its steady decline from the winter "
        "peak, a pattern several residents cite informally as the reason the "
        "city's outdoor spring calendar — markets, reopened terraces, the "
        "festival itself — clusters so heavily into this particular month.",
        "Easter Sunday lunch in Porto is traditionally a long, multi-course "
        "family affair built around roast lamb or goat, with the folar bread "
        "served alongside rather than as the meal's centrepiece, a format "
        "distinct from the standalone folar tradition observed through the rest "
        "of the surrounding April weeks.",
        "Specialty bakeries near the Bolhão market report queues stretching "
        "around the block in the final days before Easter each April, with some "
        "limiting customers to a maximum number of folar loaves per visit "
        "specifically to manage demand during the single busiest stretch of their "
        "own trading year.",
        "Porto's wine-tourism board has, in recent years, begun promoting a "
        "specific 'spring in Porto' itinerary combining an April Bolhão market "
        "visit with a Gaia lodge tour and a Douro day trip, packaging several of "
        "the city's own seasonal threads into one suggested April route for "
        "visitors.",
        "Local football matches involving Porto's own clubs occasionally overlap "
        "with the Queima das Fitas week each April, and on years when a notable "
        "match falls during the festival, the city's own bars report an unusually "
        "packed combination of football and festival crowds on the same night.",
        "The folar tradition itself varies slightly by region within northern "
        "Portugal, and Porto's own version — typically plainer than some rural "
        "variants studded with more generous meat fillings — is specifically "
        "associated with the city's bakeries rather than any single family "
        "recipe.",
        "Spring cleaning traditions in many Porto households are informally timed "
        "to the same April window as Easter, with several residents describing "
        "the pre-Easter weeks as when windows are properly opened and homes "
        "aired out after the closed-up winter months.",
        "Godparents in Porto often choose the folar they gift each year based on "
        "a specific bakery's own reputation, and some families maintain decades-"
        "long loyalty to a single April bakery purely for this one annual "
        "tradition, even when their everyday bread shopping happens elsewhere.",
        "Porto's own April produce and Easter traditions are covered less by "
        "international media than Queima das Fitas, but local food writers treat "
        "the quieter folar-and-market season as every bit as representative of "
        "the city's own spring identity as the louder student festival later in "
        "the same month.",
        "Visitors specifically interested in Porto's spring food calendar rather "
        "than the student festival often time their own April trip for the week "
        "immediately around Easter itself, when the folar bakeries, the market's "
        "seasonal stalls and the reopened riverside terraces all overlap most "
        "densely.",
    ),
)

_SEVILLE_FESTIVAL_1 = _page(
    "https://example-festivals.test/seville/semana-santa-april",
    "Semana Santa: Seville's Holy Week processions in April",
    _paragraphs(
        "Semana Santa, Holy Week, is the devotional centrepiece of Seville's spring "
        "calendar, and in years when Easter falls later in the calendar the entire "
        "week of processions runs through early April, filling the historic centre "
        "with religious brotherhoods parading elaborate floats day and night for the "
        "better part of the week.",
        "Each of Seville's dozens of hermandades, or brotherhoods, maintains its own "
        "pasos — ornate floats depicting scenes from the Passion — and processes a "
        "fixed route between its home church and the cathedral along a shared "
        "stretch known as the carrera oficial, with the whole city's April streets "
        "reorganised around the published processional timetable for the week.",
        "Nazarenos, the robed and hooded penitents who walk alongside each "
        "brotherhood's float, are a defining visual of Seville's April Holy Week, "
        "and the candles, incense and the slow, swaying rhythm of the float-bearers "
        "beneath each paso give the processions a distinctive atmosphere unlike any "
        "other week on the city's spring calendar.",
        "Saeta singers — performers who break into an unaccompanied, deeply felt "
        "flamenco-style lament from a balcony or the street as a float passes below "
        "— are a specific April Semana Santa tradition in Seville, and a particularly "
        "moving saeta can stop an entire procession in its tracks for several "
        "minutes while the crowd listens.",
        "La Madrugá, the night running from Good Friday into Saturday morning, is "
        "widely considered the emotional peak of Seville's April Holy Week, when "
        "several of the city's most revered brotherhoods process through the small "
        "hours, drawing crowds that stay out all night specifically to see these "
        "processions.",
        "Official seating along the carrera oficial is sold well ahead of the April "
        "week, while free standing room along the side streets each brotherhood's "
        "route passes through remains how most visitors actually experience the "
        "processions, often positioning themselves hours ahead of a particular "
        "float's expected passing time.",
        "Hotel prices across Seville spike sharply for the April dates Semana Santa "
        "falls on, and the city's most significant sights — the cathedral, the "
        "Alcázar — operate on adjusted hours or close outright on the week's busiest "
        "processional days, which trip planning around this specific April week "
        "needs to account for directly.",
        "Each brotherhood's own colours, music and the specific character of its "
        "pasos are well known to Seville locals, who often have a personal favourite "
        "procession they follow every April regardless of which night of the week "
        "it falls on, and visitors are frequently told which specific brotherhood's "
        "procession is worth prioritising that particular year.",
        "The scale of Seville's Semana Santa has made it one of the most widely "
        "documented religious festivals in Spain, and every April international "
        "as well as domestic visitors travel specifically to see the processions, "
        "making it, alongside the Feria de Abril a couple of weeks later, one of "
        "the two single biggest draws on the city's spring calendar.",
        "Children's participation is common throughout the week — many nazarenos "
        "processing alongside the adults are young, continuing a family's own "
        "tradition of marching with a specific brotherhood every April, often for "
        "generations within the same Seville family.",
        "Because the processions move slowly and repeatedly cross the same central "
        "streets throughout the week, getting from one part of the old town to "
        "another on foot during Semana Santa's own April days can take considerably "
        "longer than normal, and several locals specifically plan their own week "
        "around the processional timetable rather than against it.",
        "The costaleros, the men who carry each paso on their shoulders beneath the "
        "float's own skirted base, train for months before the April week, since a "
        "single float can weigh well over a tonne once fully loaded with candles, "
        "flowers and the carved image itself, and the physical toll of carrying it "
        "for hours through the April streets is considerable.",
        "Incense and candle wax are a constant sensory presence throughout "
        "Seville's April Holy Week, with the smell lingering in the streets around "
        "the carrera oficial for hours after each procession has passed, which "
        "several longtime residents describe as one of the most immediately "
        "recognisable markers of the week.",
        "Each brotherhood's own marching band, playing slow, mournful marches "
        "specifically composed for Semana Santa, follows its float through the "
        "April streets, and several of these compositions are performed only "
        "during Holy Week each year, giving the week its own distinct musical "
        "identity apart from the rest of Seville's spring calendar.",
        "Television coverage of the week's biggest processions, particularly La "
        "Madrugá's, is broadcast live across Spain each April, and Seville's own "
        "local stations run extended rolling coverage through the night specifically "
        "for viewers following a particular brotherhood's procession from home.",
        "Visitors specifically advised to dress modestly and comfortably for the "
        "week's processions, since standing for hours on crowded streets in April's "
        "sometimes-warm afternoon sun, waiting for a specific float to pass, is "
        "physically demanding even before accounting for the size of the crowds.",
        "Restaurants and bars along the processional routes adjust their own April "
        "hours around the published timetable, often staying open later than usual "
        "on nights with a late-running procession specifically to serve the crowds "
        "waiting for the next float to appear.",
        "Hotels within walking distance of the carrera oficial are typically the "
        "first to sell out for the week's April dates, and several travel guides "
        "specifically recommend booking Semana Santa accommodation the better part "
        "of a year ahead given how far in advance demand for those closest rooms "
        "builds.",
        "Each brotherhood's own paso is dressed differently every year, with fresh "
        "flowers and candles arranged specifically for that year's procession by "
        "dedicated volunteer teams in the days immediately before the April week "
        "begins, so no two years' floats look quite identical even though the "
        "carved image beneath stays the same.",
        "Children specifically trained as nazarenos from a young age within a "
        "family's own brotherhood sometimes carry small, lighter crosses in the "
        "procession rather than the full candle, a tradition several brotherhoods "
        "use specifically to bring the youngest generation into the April "
        "procession before they are old enough for the full role.",
        "Seville's own municipal police and emergency services coordinate a "
        "specific operational plan for the week each April, given the scale of "
        "the crowds and the need to keep the processional routes clear, and this "
        "planning has become more sophisticated as the week's visitor numbers "
        "have grown over the years.",
        "Local radio stations run live commentary alongside the television "
        "coverage of the week's biggest processions, with presenters specifically "
        "trained in the week's own vocabulary — paso, costalero, nazareno — to "
        "narrate the processions for listeners following from outside the crowd "
        "itself.",
        "Several of Seville's own flamenco schools specifically close for the "
        "week each April out of respect for Semana Santa's religious character, "
        "reopening only once the Holy Week processions have finished, a scheduling "
        "choice distinct from the Feria de Abril a fortnight later when the "
        "schools run their busiest classes of the year.",
        "Seating balconies overlooking the carrera oficial are rented out by "
        "their owners for the week each April at a significant premium, and "
        "several apartment owners along the route report this short-term rental "
        "income as a meaningful part of their own annual earnings tied directly "
        "to the processional calendar.",
        "Brotherhoods spend the months leading up to the April week restoring and "
        "cleaning their own pasos, a process involving specialist conservators for "
        "the oldest and most historically significant carved images, some of "
        "which date back several centuries and require careful handling ahead of "
        "each year's procession.",
        "School groups across Seville are given time off specifically timed "
        "around the week each April, and many local schoolchildren grow up taking "
        "part in a specific brotherhood's procession from an early age, continuing "
        "a family tradition that in some cases spans multiple generations within "
        "the same parish.",
        "Visitor numbers for the week have grown substantially over recent "
        "decades, and Seville's own hospitality sector now treats the week as "
        "effectively the true start of the city's tourist season each year, ahead "
        "of even the Feria de Abril a fortnight later.",
        "The smell of orange blossom mixing with candle wax and incense through "
        "the week's own April processions has become so specifically associated "
        "with the city that several perfume houses have released commercial "
        "fragrances explicitly marketed as capturing Seville's own Semana Santa "
        "atmosphere.",
        "Several international television crews base themselves in Seville for "
        "the whole of the April week specifically to cover the processions, and "
        "the week's broadcast coverage has become a significant source of "
        "international visibility for the city beyond Spain's own domestic "
        "audience.",
        "Similar Holy Week processions take place in other Andalusian cities "
        "during the same April stretch, but Seville's own Semana Santa is "
        "consistently cited as the largest and most elaborate, drawing "
        "comparisons from visitors who have also seen the smaller versions in "
        "Málaga or Granada the same spring.",
        "Archival photographs of the week's own processions going back over a "
        "century are held by several of the city's brotherhoods, and a handful "
        "run small museums of their own displaying historic pasos and vestments "
        "alongside the current year's float, open to visitors outside the April "
        "week itself.",
        "Travel writers specifically covering Semana Santa each April often "
        "note how the week's religious solemnity contrasts with the Feria de "
        "Abril's purely festive character only a fortnight later, framing the "
        "two as deliberately opposite halves of the same April-into-May season "
        "in the city.",
    ),
)

_SEVILLE_FESTIVAL_2 = _page(
    "https://example-festivals.test/seville/feria-de-abril",
    "Feria de Abril: Seville's April Fair",
    _paragraphs(
        "The Feria de Abril, literally the April Fair, is Seville's other defining "
        "spring festival, held roughly two weeks after Easter at a dedicated "
        "fairground across the river in the Los Remedios district, and the fair's "
        "own April dates are set every year specifically around that gap from Holy "
        "Week.",
        "Hundreds of casetas — individual marquee tents, most privately run by "
        "families, clubs or companies — line the fairground's streets for the fair's "
        "week-long April run, each decorated in the fair's traditional striped "
        "fabric and lit with strings of paper lanterns that give the whole site its "
        "recognisable April evening glow.",
        "The fair opens each year with the Alumbrao, the formal switching-on of the "
        "fairground's thousands of lights on the fair's first night, and this "
        "opening-night ceremony is treated as the unofficial start of Seville's "
        "busiest April social season, drawing large crowds purely to watch the "
        "lights come on.",
        "Flamenco dancing, both informal and performed, fills the fair's casetas "
        "throughout its April week, and the traditional flamenco dress — the "
        "flounced, often brightly patterned traje de flamenca for women and the "
        "short jacket and flat-brimmed hat for men — is worn widely across the "
        "fairground for the whole of the fair's run.",
        "Rebujito, a mix of dry sherry and lemon-lime soda served over ice, is the "
        "fair's signature April drink, poured continuously across the casetas "
        "throughout the week, and sherry more generally — manzanilla and fino in "
        "particular — is central to how the fair is eaten and drunk through its "
        "whole April run.",
        "A fairground funfair with rides and games operates alongside the casetas "
        "for the fair's full April week, drawing families and children during the "
        "daytime hours before the casetas' own evening and late-night atmosphere "
        "takes over, which is why many locals treat the fair as having a distinctly "
        "different daytime and night-time character.",
        "Horse-drawn carriages and mounted riders in traditional dress parade "
        "through the fairground's streets during the fair's daytime hours each "
        "April, a tradition tied to the fair's origins as a 19th-century livestock "
        "market before it grew into the purely social festival it is today.",
        "While many of the fair's casetas are private, invitation-only tents, a "
        "handful of public casetas are open to anyone across the fair's April run, "
        "and this is generally how a first-time visitor without a personal "
        "connection to a private tent experiences the fair's food, drink and "
        "dancing.",
        "The fair's own April dates shift each year to sit roughly a fortnight after "
        "Easter, which means trip planning specifically around the Feria de Abril "
        "requires checking the current year's published calendar rather than "
        "assuming a fixed date, since the fair can land anywhere from mid to "
        "late April depending on the year.",
        "Seville's hotels and flights fill up for the fair's own April week nearly "
        "as much as for Semana Santa a fortnight earlier, and several travel "
        "guides specifically recommend booking Feria de Abril accommodation months "
        "ahead given how far in advance the week's demand builds.",
        "Food at the fair leans toward small, shareable plates eaten standing or at "
        "a caseta's own tables — pescaíto frito, cured ham, olives — rather than a "
        "full sit-down meal, matching the fair's own April rhythm of moving between "
        "tents and casetas across a long evening rather than settling in one place.",
        "The fairground's own street grid is laid out and lit specifically for the "
        "week, with named streets inside the site echoing Seville's own historic "
        "street names, and returning visitors often navigate by these same "
        "fairground street names year after year once they've learned the layout.",
        "Daytime at the fair runs on a noticeably different rhythm from the "
        "evenings: lunch at a caseta, often with extended family, followed by the "
        "funfair rides for children, before the crowd thins out in the late "
        "afternoon ahead of the fair's own evening resurgence once the heat of the "
        "April day has passed.",
        "Flamenco dress shops across Seville do a significant share of their "
        "annual trade in the weeks leading up to the fair's own April dates, with "
        "some families commissioning a new traje de flamenca each year specifically "
        "for that year's fair, a tradition that supports a small but steady "
        "seasonal garment industry in the city.",
        "The fair's own opening parade, Seville's mayor and other officials riding "
        "through the fairground in traditional dress on the fair's first full day, "
        "is broadcast locally each April and treated as a formal marker that the "
        "week's public celebrations have properly begun.",
        "Noise and light from the fairground carry across the river into parts of "
        "the historic centre each April night, and several hotels on the Los "
        "Remedios side specifically market rooms with a fairground view during the "
        "fair's own week, aimed at visitors who want to experience the atmosphere "
        "without walking the fairground itself.",
        "Private casetas are typically passed down within families, clubs or "
        "companies from year to year, and a waiting list for a highly sought caseta "
        "location can run for years, which is part of why the fair's public "
        "casetas remain the realistic way for most first-time April visitors to "
        "take part.",
        "The fair's economic footprint across the city is significant each April, "
        "with hotels, flamenco dress shops, caterers and the fairground's own "
        "vendors all reporting their single busiest week of the year tied directly "
        "to the Feria de Abril's dates.",
        "Sevillanas, the fast-paced folk dance performed widely across the fair's "
        "casetas, is danced by almost everyone present regardless of skill level, "
        "and several dance schools across the city run short refresher classes in "
        "the weeks before the fair specifically for residents wanting to brush up "
        "before the April week.",
        "The fair's funfair section has grown considerably over recent decades, "
        "now rivalling dedicated amusement parks in scale for the week it runs, "
        "and several of its biggest rides are installed and dismantled specifically "
        "for the fair's own April dates each year rather than operating year-round.",
        "Catering within the private casetas is almost always self-organised by "
        "the family, club or company running that tent, with many preparing their "
        "own traditional dishes rather than buying in food, which gives each "
        "caseta a genuinely distinct character across the fair's thousands of "
        "tents.",
        "Public holidays are not declared for the fair's own April week, but many "
        "local businesses unofficially run reduced hours throughout, particularly "
        "in the afternoons, as staff and owners alike attend the fairground across "
        "the week's daytime hours.",
        "The fair's own origins as a 19th-century cattle and horse market are "
        "still visible in the daytime programme, with livestock judging and "
        "equestrian displays continuing alongside the now much larger social and "
        "funfair side of the week, a detail several historians specifically cite "
        "when explaining how the fair grew into its current form.",
        "Children are dressed in miniature versions of the traditional flamenco "
        "dress and short jacket throughout the fair's own April week, and family "
        "photographs taken at the fairground in this dress have become something "
        "of an annual tradition in their own right for many Seville households.",
        "Several of the city's flamenco dress makers report taking custom orders "
        "for the following year's fair almost as soon as the current one ends, "
        "given how far ahead the more elaborate, hand-finished dresses need to be "
        "started to be ready for the next April's week.",
        "The fair's lighting — strings of paper lanterns in the fair's own "
        "traditional colours — is installed by a dedicated municipal team in the "
        "weeks before the April opening, and the Alumbrao ceremony itself is "
        "timed specifically to when that full lighting installation is complete "
        "each year.",
        "Visitors specifically advised to pace their own drinking across the "
        "fair's long April evenings, since rebujito is served generously and "
        "continuously across the casetas, and the fair's own daytime-to-night "
        "rhythm means many attendees are out from midday until the small hours "
        "of the same day.",
        "Public caseta operators — bars and restaurants permitted to sell to "
        "anyone rather than members only — tend to cluster along the fairground's "
        "main thoroughfare, making that stretch the practical starting point for "
        "a first-time April visitor without a private tent connection to rely on.",
        "The fair draws visitors from across Andalusia and beyond specifically "
        "for its own April week, with many families from outside Seville treating "
        "a Feria de Abril trip as an annual fixture on their own calendar "
        "regardless of where else they travel that year.",
        "Sister fairs modelled loosely on Seville's own April event run in "
        "several other Andalusian towns later in the spring, but none approach "
        "the scale of the original, and several regional tourism boards "
        "specifically market their own smaller fairs as a quieter alternative "
        "for visitors who found Seville's own week too crowded.",
        "The fair's organising committee, drawn from the city council and "
        "long-established caseta-holding families, publishes the following "
        "year's exact April dates well in advance, which trip planners "
        "specifically rely on given how far ahead accommodation and flights for "
        "the week need to be booked.",
        "Local radio and television stations run their own countdown coverage "
        "in the days before the Alumbrao each April, interviewing caseta owners "
        "and fairground staff about that year's preparations, a media ritual "
        "almost as fixed on the calendar as the fair's own opening night itself.",
        "School-age children across Seville often get a long weekend built "
        "around the fair's own April dates, and several families specifically "
        "treat the fair as the last big family outing before the school year's "
        "final push toward summer exams begins.",
        "Local historians note that the fair's own April dates have shifted by "
        "only a handful of days across most of its recorded history, staying "
        "anchored to the same post-Easter fortnight for well over a century "
        "even as the fair itself has grown enormously in scale.",
    ),
)

_SEVILLE_FESTIVAL_3 = _page(
    "https://example-festivals.test/seville/april-orange-blossom-and-food",
    "Seville's April orange blossom season and spring food traditions",
    _paragraphs(
        "Seville's orange trees, planted along streets and courtyards throughout "
        "the old town including the cathedral's own Patio de los Naranjos, blossom "
        "through late March into April, and for a few weeks each spring the scent of "
        "azahar, orange blossom, is noticeable across much of the historic centre — "
        "a seasonal detail several repeat visitors specifically time an April trip "
        "around.",
        "The blossom season overlaps directly with both Semana Santa and the Feria "
        "de Abril, and several Seville guides note that the scent of azahar drifting "
        "through the processional crowds is as much a sensory marker of the city's "
        "April festival season as the music or the lights of the fair itself.",
        "Pescaíto frito, lightly battered and fried fish, is one of Seville's "
        "defining April dishes, eaten both at the Feria de Abril's casetas and at "
        "bars across the city through the spring, with several long-running fry "
        "shops near the Triana market specifically known for their own April-season "
        "versions.",
        "The Mercado de Triana sees its stalls fill with spring produce through "
        "April — fresh artichokes, broad beans and the season's first strawberries "
        "— and several of the market's vendors mark these as seasonal April arrivals "
        "distinct from the produce available the rest of the year.",
        "Tapas bars across Santa Cruz and the Alameda de Hércules extend their "
        "outdoor seating through April once the Semana Santa crowds have cleared, "
        "and the weeks immediately after Holy Week but before the Feria de Abril are "
        "specifically noted by locals as a brief, quieter window in the city's own "
        "spring festival calendar.",
        "Orange blossom water and candles scented with azahar are sold seasonally "
        "across Seville's markets and gift shops through April, tied directly to "
        "the blossom season, and the scent has become something of an informal "
        "symbol of the city's own April identity alongside the more famous "
        "festivals.",
        "Spring vegetable dishes built around artichokes and broad beans, often "
        "cooked simply with jamón, appear on menus across the city specifically "
        "through this April window, timed to the same produce season that fills "
        "the Triana and Encarnación markets' stalls.",
        "Sherry bodegas around the city run spring tasting events through April, "
        "often timed loosely around the Feria de Abril's own sherry-heavy drinking "
        "culture, giving visitors a quieter, daytime way to explore the same drinks "
        "that fill the fair's casetas each night.",
        "Several Seville food tour operators build their spring itineraries "
        "specifically around this April produce and blossom season, pairing a "
        "Triana market visit with a tasting stop timed to catch both the seasonal "
        "stalls and the orange blossom scent at its strongest in the surrounding "
        "streets.",
        "Local flower markets add orange-blossom-themed stalls through April "
        "alongside their usual stock, selling both cut blossom and blossom-scented "
        "products, and this seasonal stall is specifically tied to the same April "
        "window the trees themselves are in bloom.",
        "Taken together, the orange blossom scent, the spring produce at Triana "
        "and the fried-fish tradition running through the fair give Seville's April "
        "a distinct food and sensory identity that sits alongside, rather than "
        "competing with, the city's two bigger processional and fairground "
        "festivals that same month.",
        "Beekeepers on the outskirts of the city specifically harvest an orange "
        "blossom honey tied to this same April bloom, sold at farmers' stalls "
        "across the city's markets as a distinctly Seville seasonal product, and "
        "several local chefs use it specifically in April-dated dessert specials "
        "built around the same blossom season.",
        "Perfumers and candle-makers in the old town's smaller artisan shops "
        "produce limited azahar-scented batches timed to the same April window, "
        "selling out well before the blossom season ends each year, which has "
        "made orange-blossom-scented products something of a seasonal collectible "
        "among regular Seville visitors.",
        "The combination of Semana Santa's incense, the Feria's fried food and the "
        "orange blossom in the air has been described by more than one food and "
        "travel writer as Seville's own unmistakable April scent profile, distinct "
        "enough from the rest of the year that some writers claim to be able to "
        "place the season blindfolded.",
        "Spring lamb dishes, timed to the same April season as the produce at "
        "Triana, appear on menus across the city's traditional restaurants, often "
        "paired with the same broad beans and artichokes coming into season at the "
        "market stalls that same month.",
        "Seville's own tourism board has, in recent years, specifically promoted "
        "the April orange blossom season as a reason to visit the weeks immediately "
        "around Semana Santa and the Feria de Abril, rather than only the two "
        "festivals themselves, framing the blossom as a seasonal attraction in its "
        "own right.",
        "Several of the city's rooftop bars specifically market an April cocktail "
        "menu built around orange blossom water and sherry, timed to the same "
        "season, and a handful of these venues report their own seasonal menu as "
        "one of their most requested of the year during the April blossom window.",
        "Guided walking tours focused specifically on Seville's orange trees and "
        "their history — many planted under Moorish rule for both shade and fruit "
        "rather than as a purely decorative choice — run through April to coincide "
        "with the blossom, distinct from the city's more common cathedral-and-"
        "Alcázar tour format.",
        "The bitter oranges themselves, inedible fresh off the tree by most "
        "accounts, are harvested later in the year and traditionally exported for "
        "marmalade production, but it is the April blossom rather than the fruit "
        "itself that defines the trees' place in the city's spring identity.",
        "Market stalls at both Triana and the central Encarnación market "
        "specifically advertise their spring vegetable boxes as April-only "
        "seasonal offerings, timed to the same broad bean and artichoke harvest "
        "that defines the month's produce across the city's traditional cooking.",
        "Seville's own botanical history ties the orange tree plantings directly "
        "to the Almohad-era irrigation systems that still run beneath parts of the "
        "old town, an engineering legacy several city guides specifically mention "
        "when explaining why the trees have thrived in this exact part of the "
        "city for so many centuries.",
        "Several of the city's pastry shops produce orange-blossom-flavoured "
        "versions of traditional sweets specifically for the April season, "
        "distinct from their year-round stock, and these seasonal versions "
        "typically disappear from shelves once the blossom itself has faded for "
        "the year.",
        "The scent of azahar is strong enough in some of the old town's narrower "
        "courtyards that several residents keep their windows open specifically "
        "through this April window, a small seasonal habit several long-time "
        "Seville residents describe as one of the more pleasant, least-noticed "
        "markers of the month.",
        "Local honey producers specifically market their orange blossom batches "
        "as a premium, limited product compared with their standard year-round "
        "honey, citing the short April bloom window as the reason supply never "
        "quite meets the demand for it each year.",
        "Food and wine pairing events built around the same spring produce — "
        "broad beans, artichokes, the first strawberries — run through several of "
        "the city's wine bars each April, often timed deliberately for the "
        "quieter week between Semana Santa and the Feria de Abril when crowds in "
        "the old town briefly thin out.",
        "Seville's spring produce season overlaps enough with Lisbon's and "
        "Porto's own April markets — strawberries, broad beans, fresh greens — "
        "that a visitor moving between all three cities in the same April window "
        "would see a broadly similar seasonal shift at each stop, even though "
        "each city's own festivals and specific dishes differ considerably.",
        "Taken across the whole of April, Seville's own calendar — Semana Santa's "
        "processions, the Feria de Abril's casetas, and the quieter orange "
        "blossom and spring produce season running underneath both — gives the "
        "city arguably the densest single-month festival calendar of the three, "
        "ahead of both Lisbon's and Porto's own April traditions.",
        "Local florists report the orange blossom's own scent as impossible to "
        "bottle accurately, and several perfumers have specifically described "
        "years of trial and error trying to replicate the exact seasonal note "
        "the real trees produce across the old town each April.",
        "The orange trees themselves require careful municipal maintenance "
        "year-round to stay healthy enough to bloom reliably each April, and "
        "Seville's own parks department runs a dedicated pruning and irrigation "
        "schedule specifically timed around the trees' flowering cycle.",
        "Food writers specifically visiting Seville in April rather than during "
        "either headline festival often cite the quieter produce-and-blossom "
        "season as the more authentic way to experience the city's own spring "
        "identity, away from the crowds the two bigger events draw.",
        "A handful of Seville's own cooking schools run short seasonal classes "
        "through April built around the same broad bean, artichoke and orange "
        "blossom ingredients filling the markets that month, aimed specifically "
        "at visitors wanting a hands-on taste of the city's spring produce "
        "season.",
        "Several long-time Seville residents describe the blossom's own fading "
        "at the end of April as a quiet, almost melancholy marker that the "
        "city's two biggest spring festivals are over for another year, a "
        "sensory bookend to the season that locals notice even once the "
        "visiting crowds have mostly gone home.",
        "Seed and nursery shops across the city report a small but steady "
        "April sales bump in orange and citrus saplings, sold to residents "
        "wanting their own blossom in a courtyard or balcony, a minor but real "
        "commercial echo of the same seasonal identity the public streets carry "
        "on a much larger scale.",
        "Several of the city's own walking-tour companies have added a "
        "dedicated April orange-blossom route to their regular listings in "
        "recent years, timed specifically to the short bloom window and "
        "distinct from both the Semana Santa and Feria de Abril tours already "
        "on offer.",
    ),
)

_RAW_PAGES: tuple[str, ...] = (
    _LISBON_FESTIVAL_1,
    _LISBON_FESTIVAL_2,
    _PORTO_FESTIVAL_1,
    _PORTO_FESTIVAL_2,
    _SEVILLE_FESTIVAL_1,
    _SEVILLE_FESTIVAL_2,
    _SEVILLE_FESTIVAL_3,
)
for _page_text in _RAW_PAGES:
    assert len(_page_text) >= 10_000, (
        f"a raw page is shorter than the prod shape: {len(_page_text)}"
    )
assert not any(_ANSWER in page for page in _RAW_PAGES)
for _venue in ("Clube de Fado", "Casa de la Memoria"):
    assert not any(_venue in page for page in _RAW_PAGES)
# "Graham's" is a real Vila Nova de Gaia port house named in the curated entry — the
# raw pages here never mention it, since they are about the trip's THEME (spring
# festivals/events/markets), not Porto's port-wine lodges at all.
assert not any("Graham's" in page for page in _RAW_PAGES)


def _log_incoming(db: Database, text: str) -> None:
    db.messages.log_message(PennyConstants.MessageDirection.INCOMING, TEST_SENDER, text)


def _log_outgoing(db: Database, text: str) -> None:
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        PennyConstants.MessageAuthor.PENNY,
        text,
        recipient=TEST_SENDER,
    )


def _seed_world_state(db: Database) -> None:
    """Lay the seven raw page dumps into ``browse-results`` directly (#2956) — the
    turn begins with them already there, mirroring the replayed turn's own store,
    rather than this test driving a live browse call to put them there — then the
    ~30-collection/~15-job background noise the replay's own store map carried,
    the three city jobs' own schedule/notify (``World.stores`` has no schedule of
    its own), and the full seeded history mirroring the replay's own message
    sequence, all RECENT (no backdating) so the whole sequence sits inside the
    live history window right before the ask."""
    require_memory(db, "browse-results").append(
        [LogEntryInput(content=page) for page in _RAW_PAGES],
        author="chat",
    )
    _seed_background_noise(db)
    for name in ("lisbon_trip_research", "porto_trip_research", "seville_trip_research"):
        db.memories.update_collection_metadata(
            name, schedule="FREQ=DAILY;BYHOUR=9,21", notify=False, replace_schedule=True
        )
    _log_incoming(db, _ARCH_ASK_1)
    _log_outgoing(db, _ARCH_REPLY_1)
    _log_incoming(db, _ARCH_ASK_2)
    _log_outgoing(db, _ARCH_REPLY_2)
    _log_incoming(db, _ARCH_ASK_3)
    _log_outgoing(db, _ARCH_REPLY_3)
    _log_incoming(db, _ARCH_ASK_4)
    _log_outgoing(db, _ARCH_REPLY_4)
    _log_outgoing(db, _SPORTS_SCORES_UPDATE_1)  # unprompted, a job's own routine output
    _log_outgoing(db, _CHANGELOG_ANNOUNCEMENT_SHIP)  # unprompted, the startup announcement
    _log_incoming(db, _TRY_SYNTHESIZE_ASK)
    _log_outgoing(db, _FAILED_ATTEMPT_REPLY)
    _log_outgoing(db, _CHESS_TRACKER_UPDATE_1)  # unprompted, the tracker job's own output
    _log_incoming(db, _TRY_PLAN_MODE_ASK)
    _log_outgoing(db, _PLAN_MODE_OFFER_REPLY)
    _log_incoming(db, _TRY_ALL_THREE_ASK)
    _log_outgoing(db, _PLAN_MODE_RESULTS_REPLY)
    _log_incoming(db, _ACK_PLAN_MODE)
    _log_outgoing(db, _REFLECTION_REPLY)
    _log_outgoing(db, _CHESS_TRACKER_UPDATE_2)  # unprompted, the tracker job's own output
    _log_incoming(db, _TANGENT)
    _log_outgoing(db, _TANGENT_REPLY)
    _log_outgoing(db, _CHESS_TRACKER_UPDATE_3)  # unprompted, the tracker job's own output
    _log_outgoing(db, _FIX_REACTION_AND_OFFER)  # unprompted, the fix's own startup announcement


_WORLD = World(
    name=_CASE_ID,
    pages=(),
    keeps=_KEEPS,
    excludes=(),
    answers=(_ANSWER,),
    stores=_COLLECTIONS,
)

# 8. The live ask: she chose the trip as the subject herself, the last thing she'd
# tried the tool on, offered right back to her — so she accepts.
_ASK = "yeah give it a go!"
_ALSO_PHRASED = (
    "sure, give it a shot!",
    "yeah go for it!",
    "alright, try it out!",
    "okay, let's see it in action!",
)

_BEHAVIOUR = (
    "In the chat agent, when the user accepts Penny's own offer to retry a "
    "previously-failed tool on the last thing she tried it on — naming no "
    "collection or query of its own — and she chooses a broad, unscoped query "
    "naming the trip's theme and cities, with the raw browse-results log already "
    "holding several long, closely-related pages on that same theme — the turn "
    "completes without any tool call escaping with an uncaught exception, and the "
    "reply states the curated fact the ask depends on."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_accepting_the_retry_offer_surfaces_the_curated_fact(
    chat_eval: ChatEval, model: str
) -> None:
    """Accepting Penny's own offer to retry `synthesize` — mirroring a prod
    replay's own sequence one-to-one: an architecture discussion, the
    tool's first-ship changelog, a failed retry attempt, a plan-mode demo, an
    unrelated tangent, then Penny's offer and the user's "yeah give it a go!" —
    against a store holding one curated trip collection, three curated per-city
    research collections, seven full raw browse pages guiding the trip's own
    theme in each city, AND the same ~30-collection/~15-job background the
    replay's own store map carried (every topic swapped), so naming the trip's
    four collections explicitly is one choice among many rather than the only
    one available. The replayed shape is where the raw pages used to fill every
    slot of the broad, unscoped `synthesize` call she chose to make, leaving no
    room for the curated fact the ask depends on."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        seed=_seed_world_state,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE (survival) — the ask reads and asks about status, it does not write.
    cohort.assert_what_the_store_held_survives()
    # STORE (mechanism) — no tool call this turn made escaped with an uncaught
    # exception.
    cohort.assert_no_tool_call_raised()
    # PROVENANCE — nothing in the reply is invented.
    cohort.assert_every_value_in_the_reply_is_sourced()
    # COMPLETENESS — the budget figure, stored only in the curated umbrella
    # collection, is in the reply: the fix's own measure, since the raw pages used
    # to crowd this fact out of a broad synthesize call entirely.
    cohort.assert_the_reply_answers_the_ask()
    # TOOL_SEQUENCE is measured and never asserted — `synthesize` is the tool built
    # for this ask, but reaching it is a route, not a claim.
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
