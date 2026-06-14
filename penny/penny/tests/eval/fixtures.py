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
