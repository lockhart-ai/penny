"""Synthetic, privacy-safe seeds for the eval suite.

NOTHING here is real user data — the repo is public.  These are contrived
collections shaped like real traffic but on deliberately generic topics (board
games, espresso gear, houseplants) so the suite is reproducible and privacy-safe.
"""

from __future__ import annotations

from dataclasses import dataclass


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
