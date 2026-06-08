"""Synthetic, scrubbed fixtures for the prompt-validation harness.

NOTHING here is real user data — the repo is public.  These are
contrived messages and collections shaped like real traffic but on
deliberately generic topics (board games, espresso gear, houseplants,
sci-fi novels) so the harness is reproducible and privacy-safe.

For ad-hoc runs against real production traffic, individual runners may
accept a ``--live-db`` flag and pull messages from ``penny.db`` directly;
that data never lands in the repo.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    id: str
    text: str
    # What SHOULD happen, for scoring. ``skill`` is the expected skill key
    # (or None for one-shot / chitchat); ``collections`` are topical
    # collections whose ENTRIES should surface for this message (research,
    # topical queries, topical one-shots).  Pure lifecycle ops (cadence,
    # archive, silent-flip) leave this empty: they act on a collection but
    # don't need its entries in recall — the model uses the memory inventory
    # plus ``collection_metadata``.
    skill: str | None
    collections: tuple[str, ...] = ()
    # Prior conversation turns (oldest→newest), to model production's
    # history-aware recall anchor for topic-less follow-ups.
    history: tuple[str, ...] = ()


# Messages shaped like real intents, on synthetic topics.
MESSAGES: list[Message] = [
    # ── ongoing-research requests → research skill ──────────────────────────
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
    # ── lifecycle ops on an existing collection ─────────────────────────────
    # Lifecycle ops: ``collections`` lists a collection only when the message
    # is topical enough that surfacing it is correct (not just harmless).
    # The model drives the op via the skill + inventory + collection_metadata;
    # whether the entries also surface is a secondary, benign concern.
    Message(
        "update-scope",
        "actually for the board games collection, narrow it to just 2-player games "
        "and drop the big-group party stuff",
        skill="Update collection scope",
        collections=("board-games",),
    ),
    Message(
        # Dominated by schedule words ("daily", "every hour") — scores below
        # the topical gate, and the entries aren't needed for a cadence change.
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
    # ── one-shot lookups → browse skill, NOT a collection ───────────────────
    Message(
        "oneshot-plant",
        "what's a good low-light houseplant that's hard to kill?",
        skill="Browse for a one-shot question",
        collections=("houseplant-care",),  # topical: the houseplant collection should surface
    ),
    Message(
        "oneshot-novel",
        "find me the best-reviewed sci-fi novel that came out this year",
        skill="Browse for a one-shot question",
        collections=(),
    ),
    # ── topical query that should hit an existing topical collection ────────
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
    # ── chitchat → no skill, no topical collection ──────────────────────────
    Message(
        "chitchat",
        "hey what do you remember, where did we leave off last time?",
        skill=None,
        collections=(),
    ),
]


@dataclass
class SynthCollection:
    name: str
    description: str  # the content-reflective stage-1 routing anchor
    inclusion: str  # always | relevant | never
    entries: tuple[str, ...]  # entry contents (for stage-2 retrieval)


# Synthetic memory state for the two-stage routing prototype.  ``skills`` is
# injected separately from the real seed migration; these are the topical
# collections that stage-1 routing must include/drop correctly.
SYNTH_COLLECTIONS: list[SynthCollection] = [
    SynthCollection(
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
    ),
    SynthCollection(
        "espresso-gear",
        "Home espresso equipment under ~$1000: dual-boiler and heat-exchanger "
        "machines, flat-burr grinders, distribution tools, and value picks.",
        inclusion="relevant",
        entries=(
            "Gaggia Classic Pro — entry single-boiler, mod-friendly, ~$450.",
            "Eureka Mignon Specialita — 55mm flat-burr grinder, ~$400.",
            "Profitec Go — compact single-boiler PID machine, ~$700.",
        ),
    ),
    SynthCollection(
        "houseplant-care",
        "Indoor houseplant care notes: light needs, watering schedules, and "
        "low-maintenance species for low-light apartments.",
        inclusion="relevant",
        entries=(
            "Snake plant — very low light, water every 2-3 weeks.",
            "ZZ plant — thrives on neglect, low light, drought-tolerant.",
        ),
    ),
]
