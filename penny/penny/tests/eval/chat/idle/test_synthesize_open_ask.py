"""#2953 fix: ``synthesize``'s micro-context draw ran on the EMBEDDING client, never
the chat model client — an embedding model rejects a chat request outright, so every
real call failed with an HTTP 400 naming "does not support chat".

Built from a prod replay: a short, open "go ahead and try it" ask, naming no topic of
its own, arriving against a store that already holds several real, unrelated
collections across more than one area of the user's life. Every name, collection and
entry below is invented for this case — the repo is public, and nothing from the real
prod conversation appears here.

**The claims.** STORE (survival) — the ten collections the turn reads hold exactly
what they were seeded with; an open "try it" ask names nothing to write. PROVENANCE —
every specific value in the reply traces to something the turn was given, which is
also the one claim a turn that ended on a model error fails outright, so it is the
natural read of "a reply reached the user and nothing broke the turn".

**What is NOT claimed, and why.** The ask names no topic, so there is no single
correct thing the reply must say — which tool answers it, and which collections it
draws on, are both ROUTES, never claims (`docs/principles.md` §4.1): `synthesize` is
the tool built for exactly this ask, but `find`/`collection_list` reading some of the
same entries is an equally legitimate path. `TOOL_SEQUENCE` is measured instead. The
bug this case guards against shows most reliably as degraded variance/cost (repeated
failed `synthesize` retries, the model never settling) rather than a dropped claim —
read all three measures, not the claims alone.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD, TOOL_SEQUENCE
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "memory-synthesis"

_CASE_ID = "synthesize-open-ask-across-collections"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY

# One small umbrella project fed by three narrower research collections — the same
# shape prod's own store held for the turn this case replays: one main collection plus
# a handful of collections researching pieces of it.
_PROJECT = SynthCollection(
    "kitchen-remodel-2027",
    "Planning notes for the 2027 kitchen remodel.",
    entries=(
        "contractor quote — Milbrook Construction bid $34,000 for the full gut "
        "remodel, six-week timeline.",
        "paint color — decided on a warm off-white, Sherwin-Williams Alabaster, for "
        "the upper cabinets.",
    ),
)
_COUNTERTOPS = SynthCollection(
    "countertop-research",
    "Comparing countertop materials for the remodel.",
    entries=(
        "quartz vs granite — quartz needs no sealing and resists stains better than granite.",
    ),
)
_CABINETS = SynthCollection(
    "cabinet-styles-research",
    "Cabinet door styles and hardware under consideration.",
    entries=(
        "shaker style — flat panel with a simple frame, pairs well with brushed brass pulls.",
    ),
)
_LIGHTING = SynthCollection(
    "lighting-fixtures-research",
    "Lighting options for over the island and sink.",
    entries=("pendant lights — three glass pendants spaced 12 inches apart over the island.",),
)

# A handful of unrelated hobby collections — the same shape prod's own store held
# beside the project: several separate, standing interests with no relation to it or
# to each other.
_WOODWORKING = SynthCollection(
    "woodworking-projects",
    "Notes from woodworking projects in the garage.",
    entries=(
        "walnut side table — four mortise-and-tenon joints, finished with tung oil.",
        "cutting board — end-grain maple and cherry, sealed with food-safe mineral oil.",
        "tool sharpening — a 1000/6000 grit waterstone combo keeps the chisels keen.",
    ),
)
_BOARD_GAMES = SynthCollection(
    "board-game-collection",
    "Board games owned and house rules for each.",
    entries=(
        "Wingspan — the European expansion adds a nectar resource, makes hummingbirds stronger.",
        "Brass Birmingham — always take the first turn's loan if nobody else wants it.",
    ),
)
_SOURDOUGH = SynthCollection(
    "sourdough-experiments",
    "Notes from sourdough baking experiments.",
    entries=(
        "starter hydration — a 100% hydration starter ferments fastest near 75°F.",
        "crust technique — steam in the oven's first ten minutes gives a shinier crust.",
    ),
)
_TELESCOPE = SynthCollection(
    "telescope-upgrades",
    "Upgrades and accessories for the backyard telescope.",
    entries=(
        "eyepiece swap — a 13mm Ethos eyepiece resolves Saturn's rings far better "
        "than the stock 25mm.",
        "tracking mount — the new equatorial mount holds alignment for over an hour now.",
    ),
)
_CALLIGRAPHY = SynthCollection(
    "calligraphy-practice",
    "Calligraphy nibs, inks and practice notes.",
    entries=(
        "nib choice — a Nikko G nib holds a sharper point than the Brause for pointed-pen work.",
    ),
)
_BONSAI = SynthCollection(
    "bonsai-care-notes",
    "Care notes for the bonsai collection.",
    entries=(
        "watering — the juniper needs water daily in summer, every other day in winter.",
        "wiring — copper wire holds the branch shape better than aluminum on the older trees.",
    ),
)

_COLLECTIONS = (
    _PROJECT,
    _COUNTERTOPS,
    _CABINETS,
    _LIGHTING,
    _WOODWORKING,
    _BOARD_GAMES,
    _SOURDOUGH,
    _TELESCOPE,
    _CALLIGRAPHY,
    _BONSAI,
)

_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=(), stores=_COLLECTIONS)

# Nine short, generic prior exchanges — the same history-window SIZE a replayed prod
# turn carried (the window grown to its upper bound, #2756) — ahead of the one real
# ask, which names no topic of its own, exactly as the replayed turn's did not.
_FILLER_EXCHANGES: tuple[tuple[str, str], ...] = (
    ("morning!", "morning! hope you slept well"),
    ("quick one — remind me what day it is", "it's thursday"),
    ("thanks", "anytime"),
    ("how's everything looking today", "all quiet, nothing new to report"),
    ("cool", "yep"),
    ("one more thing before i forget", "go ahead"),
    ("never mind, it'll keep", "no worries, here whenever"),
    ("appreciate you", "always"),
    ("alright, talk in a bit", "sounds good"),
)

_ASK = "okay try your synthesize tool"
_ALSO_PHRASED = (
    "go ahead, try the synthesize tool",
    "alright, give synthesize a try",
    "sure, try out your synthesize tool",
    "ok, go ahead and use synthesize",
)

_BEHAVIOUR = (
    "In the chat agent, when the user asks Penny to try her synthesize tool, naming "
    "no topic of its own, against a store holding several unrelated real "
    "collections, the turn completes with a reply that invents nothing."
)


def _seed_history(db: Database) -> None:
    """The nine prior exchanges, logged as the newest turns before the real ask so
    they land inside the grown history window (#2756)."""
    for user_line, penny_line in _FILLER_EXCHANGES:
        db.messages.log_message(_INCOMING, TEST_SENDER, user_line)
        db.messages.log_message(_OUTGOING, _PENNY, penny_line, recipient=TEST_SENDER)


def _assert_the_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs:
    every one of the ten collections holds exactly its seeded entries."""
    for synth in _COLLECTIONS:
        held = sorted(entry.key or "" for entry in require_memory(penny.db, synth.name).read_all())
        expected = sorted(key for key, _ in synth.keyed)
        assert held == expected, f"{_CASE_ID}: {synth.name} must hold {expected}, it holds {held}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_open_synthesize_ask_completes_without_inventing_anything(
    chat_eval: ChatEval, model: str
) -> None:
    """Ten small, unrelated collections — one project fed by three research
    collections, and six standing hobby collections — and a bare "try your
    synthesize tool" with no topic of its own, exactly the open shape that leaves
    `synthesize`'s own draw to pick what to read and what to say about it."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        seed=_seed_history,
        prepare=_assert_the_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — the ask reads and tries a tool, it does not write.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — nothing in the reply is invented; also fails outright if the turn
    # aborted on a model error, which is the natural read of "a reply reached the
    # user and nothing broke the turn".
    cohort.assert_every_value_in_the_reply_is_sourced()
    # TOOL_SEQUENCE is measured and never asserted — `synthesize` is the tool built
    # for this ask, but reaching it is a route, not a claim.
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
