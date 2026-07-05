"""Synthetic, privacy-safe seeds for the eval suite.

NOTHING here is real user data — the repo is public.  These are contrived
collections shaped like real traffic but on deliberately generic topics (board
games, espresso gear, houseplants) so the suite is reproducible and privacy-safe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A synthetic user message + what recall SHOULD do with it.

    ``skill`` is the expected seed-skill key (or None for chitchat/query);
    ``collections`` are topical collections whose routing should include them;
    ``history`` is prior turns so topic-less follow-ups still anchor on a topic.
    """

    id: str
    text: str
    skill: str | None
    collections: tuple[str, ...] = ()
    history: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynthCollection:
    name: str
    description: str  # the content-reflective stage-1 routing anchor
    inclusion: str  # always | relevant | never
    entries: tuple[str, ...]  # entry contents (for stage-2 retrieval)


@dataclass(frozen=True)
class CannedPage:
    """One realistic page the mock browser returns for a matching query/URL.

    ``match`` is a lowercase substring tested against the request URL — a search
    query becomes ``SEARCH_URL`` + ``quote(query)`` and a direct read is the URL
    itself, so a single distinctive token (no spaces — URLs are percent-encoded)
    matches both shapes.  ``text`` must read like a real page: a ``Title:`` first
    line and the source URL *in the visible body* (the model cites URLs from the
    text it sees).  For search-shaped pages, put the fact lines adjacent to a
    solo markdown-link line so search trimming keeps them.  See ``install_browse``.

    ``fails=True`` makes a matched read RAISE inside the browse tool, so the run
    renders a ``## browse error:`` section for that query (the only place a read
    failure is visible — see ``database/memory/objects.py`` run-health tally).
    A ``match=""`` page matches every URL (``"" in url`` is always true), so a
    single ``CannedPage(match="", fails=True)`` makes *every* source unreachable
    — the "browsed a lot, read nothing usable" cycle.  ``text`` is ignored when
    ``fails`` is set.
    """

    match: str
    text: str
    image: str | None = None
    fails: bool = False


# Every browse read fails — the "browsed many sources, read nothing usable, wrote
# nothing" cycle that produced a confabulated done() summary in production.  Place
# it as the SOLE / LAST page so specific success pages (if any) still match first.
ALL_BROWSES_FAIL = CannedPage(match="", text="", fails=True)


BOARD_GAMES = SynthCollection(
    "board-games",
    "Heavier euro-style strategy board games and modern tabletop classics: "
    "worker-placement, engine-builders, 2-player duels, and group games worth buying.",
    inclusion="relevant",
    entries=(
        "Brass: Birmingham — economic engine-builder, 2-4 players, ~2h.",
        "Ark Nova — zoo-building card-driven strategy, heavy, 1-4 players.",
        "Twilight Struggle — 2-player Cold War tug-of-war, card-driven.",
        "Spirit Island — co-op area-control, high complexity.",
    ),
)

# A realistic extraction_prompt + goal for the seeded board-games collection,
# so update/archive cases act on a fully-formed collection like prod has.
BOARD_GAMES_INTENT = (
    "Keep me on top of new heavier euro-style strategy board games worth buying, "
    "and tell me when a good one shows up."
)
BOARD_GAMES_EXTRACTION_PROMPT = (
    "Collect heavier euro-style strategy board games and modern tabletop classics.\n"
    "1. browse the web for new strategy board games; read actual pages.\n"
    "2. Each entry: key = game name; content = name + description + player count + URL.\n"
    '3. collection_write("board-games", entries=[...]).\n'
    '4. If a write succeeded, send_message a one-sentence "found a new game" note + URL.\n'
    "5. done()."
)

ESPRESSO_GEAR = SynthCollection(
    "espresso-gear",
    "Home espresso equipment under ~$1000: dual-boiler and heat-exchanger "
    "machines, flat-burr grinders, distribution tools, and value picks.",
    inclusion="relevant",
    entries=(
        "Gaggia Classic Pro — entry single-boiler, mod-friendly, ~$450.",
        "Eureka Mignon Specialita — 55mm flat-burr grinder, ~$400.",
        "Profitec Go — compact single-boiler PID machine, ~$700.",
    ),
)

HOUSEPLANT_CARE = SynthCollection(
    "houseplant-care",
    "Indoor houseplant care notes: light needs, watering schedules, and "
    "low-maintenance species for low-light apartments.",
    inclusion="relevant",
    entries=(
        "Snake plant — very low light, water every 2-3 weeks.",
        "ZZ plant — thrives on neglect, low light, drought-tolerant.",
    ),
)

# The three topical collections used by the recall-routing suite.
TOPICAL_COLLECTIONS = (BOARD_GAMES, ESPRESSO_GEAR, HOUSEPLANT_CARE)

# Synthetic messages for recall routing — ``skill`` keys match the real seed
# skills (migration 0043); ``collections`` are the topical collections whose
# stage-1 routing must include them.
MESSAGES: tuple[Message, ...] = (
    Message(
        "research-boardgames",
        "i just got back into board games — can you research heavier euro-style "
        "strategy games and modern classics for me? ping me when you find good ones",
        skill="Research collection — notify on new finds",
        collections=("board-games",),
    ),
    Message(
        "research-continue",
        "ya that's great! keep researching and tell me when you turn up more",
        skill="Research collection — notify on new finds",
        collections=("board-games",),
        history=(
            "i just got back into board games — can you research heavier euro-style "
            "strategy games and modern classics for me? ping me when you find good ones",
        ),
    ),
    Message(
        "research-silent",
        "research espresso machines and grinders under a grand for me — silent, "
        "i'll check in when i want to see the list",
        skill="Research collection — silent",
        collections=("espresso-gear",),
    ),
    Message(
        "update-scope",
        "actually for the board games collection, narrow it to just 2-player games "
        "and drop the big-group party stuff",
        skill="Update collection scope",
        collections=("board-games",),
    ),
    Message(
        "cadence",
        "check the board games collection daily instead of every hour",
        skill="Change collection cadence",
    ),
    Message(
        "silent-flip",
        "stop pinging me about new board game finds, i'll just look myself",
        skill="Flip silent ↔ notify",
        collections=("board-games",),
    ),
    Message(
        "archive",
        "i'm done collecting board games for now, archive that one",
        skill="Archive a collection",
        collections=("board-games",),
    ),
    Message(
        "oneshot-plant",
        "what's a good low-light houseplant that's hard to kill?",
        skill="Browse for a one-shot question",
        collections=("houseplant-care",),
    ),
    Message(
        "oneshot-novel",
        "find me the best-reviewed sci-fi novel that came out this year",
        skill="Browse for a one-shot question",
    ),
    Message(
        "query-boardgames",
        "remind me which 2-player board games we'd flagged as worth buying",
        skill=None,
        collections=("board-games",),
    ),
    Message(
        "query-espresso",
        "what espresso grinders did we end up shortlisting?",
        skill=None,
        collections=("espresso-gear",),
    ),
    Message(
        "chitchat",
        "hey what do you remember, where did we leave off last time?",
        skill=None,
    ),
)


# ── Canned browse pages (for the browse-driven tool-reasoning cases) ──────────
# All invented, privacy-safe topics on example.com domains.

# chat-browse-answer: one search whose result carries the fact (version 4.2) + URL.
VERSION_PAGES = (
    CannedPage(
        match="quillpad",
        text=(
            "Title: Quillpad releases\n"
            "Quillpad release history and downloads below.\n"
            "[Quillpad 4.2 release notes](https://quillpad.example.com/releases/4.2)\n"
            "Quillpad 4.2 is the latest stable release, published this year, "
            "adding end-to-end sync and a dark theme.\n"
        ),
    ),
)

# chat-browse-multihop: the search page links to a detail page but withholds the
# date; the year (2031) lives ONLY on the detail page, so a reply that cites it
# proves the model chained a second browse to the linked URL.
MULTIHOP_PAGES = (
    CannedPage(
        match="mistforge",
        text=(
            "Title: Mistforge Tactics — search results\n"
            "Mistforge Tactics is a turn-based strategy game.\n"
            "[Mistforge Tactics — official title page]"
            "(https://gamedb.example.com/titles/mt-clans-2099)\n"
            "See the official title page for full release details.\n"
        ),
    ),
    CannedPage(
        match="mt-clans-2099",
        text=(
            "Title: Mistforge Tactics — official title page\n"
            "Mistforge Tactics was released on March 14, 2031 by Emberline Studios. "
            "It is a turn-based strategy game with co-op campaigns.\n"
            "Source: https://gamedb.example.com/titles/mt-clans-2099\n"
        ),
    ),
)

# collector-research-browse: a user-created notify-on-new watcher + its page.
RESEARCH_WATCHER = SynthCollection(
    "indie-metroidvanias",
    "Newly released indie metroidvania games worth playing: hand-drawn "
    "exploration platformers with interconnected maps and new traversal mechanics.",
    inclusion="relevant",
    entries=(),
)
RESEARCH_WATCHER_INTENT = (
    "Keep me posted on new indie metroidvania games worth playing, and ping me "
    "when a good one shows up."
)
RESEARCH_WATCHER_EXTRACTION_PROMPT = (
    "Collect newly released indie metroidvania games worth playing.\n"
    "1. browse the web for new indie metroidvania releases; read actual pages.\n"
    "2. Each entry: key = game name; content = name + description + URL.\n"
    '3. collection_write("indie-metroidvanias", entries=[...]).\n'
    "4. done()."
)


# The notifier (pub/sub consumer) is migration-seeded (0067), so the eval drives the
# SHIPPED prompt directly (a fresh eval DB runs migrations) — no duplicated copy here.
RESEARCH_PAGES = (
    CannedPage(
        match="metroidvania",
        text=(
            "Title: New indie metroidvanias this month\n"
            "Fresh indie metroidvania releases below.\n"
            "[Hollow Verge — new metroidvania release]"
            "(https://indiegames.example.com/hollow-verge)\n"
            "Hollow Verge launched this week — a hand-drawn metroidvania with a "
            "grappling-hook traversal system and a branching map, priced at $19.\n"
        ),
    ),
    # The detail page, in case the model reads the linked URL ("read actual pages").
    CannedPage(
        match="hollow-verge",
        text=(
            "Title: Hollow Verge — release page\n"
            "Hollow Verge is a newly released indie metroidvania, out this week. It "
            "features hand-drawn art, a grappling-hook traversal system, and a branching "
            "interconnected map. It is priced at $19 and is available now.\n"
            "Source: https://indiegames.example.com/hollow-verge\n"
        ),
    ),
)

# ── Digest collector (empty-user-turn bailout reproduction) ──────────────────
# A read-a-log-first / summarize-into-one-entry collector, shaped exactly like the
# real "rolling summary" collectors that bail most often in production: the whole
# task lives in the system prompt, step 1 is a mandatory log_read, and the user
# turn is empty.  gpt-oss (harmony-trained) reads the blank user turn as "the user
# said nothing" and frequently jumps straight to done() WITHOUT calling log_read —
# never checking for the work that's plainly seeded.  This is the baseline the
# prompt-split experiment must beat.  Topic is generic/privacy-safe.
WEEKLY_DIGEST = SynthCollection(
    "weekly-digest",
    "A single rolling summary of the user's recent messages: what they've been "
    "up to, key events, and how things are going.",
    inclusion="never",
    entries=(),
)
WEEKLY_DIGEST_INTENT = "Keep one running summary of what I've been up to lately, updated as I chat."
WEEKLY_DIGEST_EXTRACTION_PROMPT = (
    "Summarise the user's recent messages into a single rolling summary entry.\n"
    '1. log_read("user-messages") — fetch all new user-message entries since the '
    "last run.\n"
    '2. collection_read_latest("weekly-digest", k=1) — get the current summary '
    "entry, if any.\n"
    "3. Combine both and write a concise paragraph capturing key events, "
    "activities, and how things are going.\n"
    '4. collection_write("weekly-digest", entries=[{key: "summary", '
    "content: <generated summary>}]).\n"
    "5. done()."
)
# Clearly-summarizable synthetic messages — a working cycle MUST produce a summary
# entry, so a no-write outcome is an unambiguous bailout, not a defensible no-op.
WEEKLY_DIGEST_MESSAGES = (
    "just wrapped up a big push at work — shipped the new release on friday and "
    "the team's pretty happy with how it landed",
    "been getting back into running too, did my first 10k in ages on saturday "
    "morning and felt great after",
    "weekend was nice and low-key otherwise, cooked a bunch and caught up on some "
    "films i'd been meaning to watch",
)


# ── Send-shaped collector (half-formed-send guard reproduction) ───────────────
# A read-then-send collector: read the log, then send the user a one-line digest.
# The send step is what the malformed-send injector hijacks (forcing the
# half-formed "Hi there! ......???"); the gate must refuse it so the model resends
# a complete message instead of the user receiving junk.  Generic/privacy-safe.
SEND_DIGEST = SynthCollection(
    "daily-ping",
    "A short daily ping to the user summarizing what they have mentioned lately.",
    inclusion="never",
    entries=(),
)
SEND_DIGEST_INTENT = "Send me a short daily ping summarizing what I've mentioned lately."
SEND_DIGEST_EXTRACTION_PROMPT = (
    "Send the user a one-line digest of their recent messages.\n"
    '1. log_read("user-messages") — fetch new user messages since the last run.\n'
    "2. Compose a single COMPLETE sentence summarizing what they have been up to.\n"
    "3. send_message(content=<that complete sentence>).\n"
    "4. done()."
)
# A half-formed message body the send injector forces — an unfinished fragment the
# user should never receive.  The shape observed in production; the truncation-tail
# regex missed it, the shared half-formed rule catches it.
HALF_FORMED_SEND = "Hi there! ......???"


# ── Prose-vs-numbered format pair (same task, the format is the only variable) ─
# We found gpt-oss follows a NUMBERED instruction/tool-call recipe far more
# reliably than the SAME task written as prose (prose-in-system bails ~60% of the
# time on the empty collector user turn; numbered ~5%).  These two prompts describe
# an identical read-log → extract → write collector; only the format differs, so a
# behaviour gap between them isolates the format effect.  Generic/privacy-safe topic.
WATCHLIST = SynthCollection(
    "watchlist",
    "Movies and TV shows the user has said they want to watch.",
    inclusion="never",
    entries=(),
)
WATCHLIST_INTENT = (
    "Keep a running list of the movies and shows I want to watch, from what I mention."
)
WATCHLIST_PROSE_PROMPT = (
    "Collect the movies and TV shows the user mentions wanting to watch from their "
    "recent messages. Read the user-messages log, and for each title the user says they "
    "want to watch, record the title together with a short note on why, writing them "
    "with collection_write. Skip anything unrelated. If the user later says they "
    "finished a title or lost interest, update or delete that entry instead of leaving "
    "it. Do not add titles the user did not actually mention, and finish by calling done."
)
WATCHLIST_NUMBERED_PROMPT = (
    "Collect the movies and TV shows the user wants to watch.\n"
    '1. log_read("user-messages") — fetch new user messages since the last run.\n'
    "2. For each title the user mentions wanting to watch, note the title and a short "
    "reason.\n"
    '3. collection_write("watchlist", entries=[{key: <title>, content: <title + '
    "reason>}]).\n"
    "4. If the user finished or lost interest in a title, update_entry or "
    "collection_delete_entry instead of adding.\n"
    "5. done()."
)
WATCHLIST_MESSAGES = (
    "ooh i really want to watch that new dune movie everyone keeps talking about",
    "a friend said severance is incredible, i should start that show",
    "anyway, can you remind me what time my dentist appointment is?",
)

# A plain-text "bail" the nudge-recovery contract forces mid-cycle — a collector
# narrating completion as prose instead of calling a tool.  The exact wording is
# irrelevant (the nudge fires on ANY text-only response); this is just a realistic
# shape observed in production.  Privacy-safe / generic.
COLLECTOR_PROSE_BAIL = "**Done. Summary: I've handled the recent messages.**"

# The dominant call-shaped text bail in production: gpt-oss emits the ``done()``
# terminator's *arguments* as a bare JSON object with no function name — the native
# fallback on Harmony backends (the name rides a header that gets lost).  The
# done-JSON teaching nudge names the slip and the exact done() call to make; the
# model must re-emit it as a real tool call.  Synthetic summary; privacy-safe.
COLLECTOR_DONE_JSON_BAIL = '{"success": true, "summary": "handled the recent messages this cycle"}'

# ── Duplicate-write recovery (matched-key handoff) ───────────────────────────
# A collector whose write duplicates a recipe the box ALREADY holds.  The rejection
# now names the matched existing key + the next move, so the live model must recover
# (update_entry with that key, or an honest done()) instead of guessing keys /
# re-reading / retrying variations until it burns the step budget.  The only source
# of recipes is the collection itself (no browse step), so a recovered cycle has
# genuinely nothing new to add.  Generic/privacy-safe topic.
RECIPE_BOX = SynthCollection(
    "recipe-box",
    "Saved quick weeknight dinner recipes: short ingredient lists and fast cook times.",
    inclusion="never",
    entries=(
        "Sheet-pan chicken fajitas — peppers, onion, chicken, 25 min at 425F.",
        "One-pot lemon orzo — orzo, lemon, spinach, parmesan, 20 min.",
    ),
)
RECIPE_BOX_INTENT = "Keep a box of quick weeknight dinner recipes I can pull from."
RECIPE_BOX_EXTRACTION_PROMPT = (
    "Save quick weeknight dinner recipes worth keeping.\n"
    '1. collection_read_latest("recipe-box", k=20) — see what is already saved so you '
    "do not repeat one.\n"
    "2. For each genuinely new recipe not already saved, "
    'collection_write("recipe-box", entries=[{key: recipe name, content: name + key '
    "ingredients + time}]).\n"
    "3. done()."
)
# The forced duplicate write: byte-identical to the first seeded entry, so dedup
# catches it on content and hands back its key ("Sheet-pan chicken fajitas").
RECIPE_BOX_DUP_KEY = "sheet pan chicken fajitas"
RECIPE_BOX_DUP_CONTENT = "Sheet-pan chicken fajitas — peppers, onion, chicken, 25 min at 425F."
# The keys the box holds after seeding (SynthCollection keys = text before ' — ').
RECIPE_BOX_SEED_KEYS = ("Sheet-pan chicken fajitas", "One-pot lemon orzo")

# thinking-generate: a timely fact + URL for the seeded 'likes' topic to ground a thought.
THINKING_PAGES = (
    CannedPage(
        match="board",
        text=(
            "Title: Board game news\n"
            "[New cooperative board game spotlight](https://bgnews.example.com/tidewatch)\n"
            "A new cooperative board game called Tidewatch launched this month, "
            "featuring a modular ocean board and a 60-minute play time.\n"
        ),
    ),
    # The detail page, in case the model reads the linked URL rather than the snippet.
    CannedPage(
        match="tidewatch",
        text=(
            "Title: Tidewatch — co-op board game\n"
            "Tidewatch is a newly released cooperative board game, out this month. It "
            "features a modular ocean board and a 60-minute play time, for 1-4 players.\n"
            "Source: https://bgnews.example.com/tidewatch\n"
        ),
    ),
)

# extract-knowledge: a page already in the browse-results log to summarize.
KNOWLEDGE_PAGE_CONTENT = (
    "## browse: https://history.example.com/antikythera\n"
    "Title: The Antikythera Mechanism\n"
    "The Antikythera mechanism is an ancient Greek hand-powered analog device used "
    "to predict astronomical positions and eclipses decades in advance. Recovered "
    "from a Roman-era shipwreck off the Greek island of Antikythera in 1901, it is "
    "dated to roughly the 2nd century BC. It uses a complex system of at least 30 "
    "bronze gears to model the motions of the Sun and Moon, and tracked the timing "
    "of the ancient Olympic Games. It is widely regarded as the oldest known example "
    "of an analog computer.\n"
)
