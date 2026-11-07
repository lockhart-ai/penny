"""#2953 fix: ``synthesize``'s micro-context draw ran on the EMBEDDING client, never
the chat model client — an embedding model rejects a chat request outright, so every
real call failed with an HTTP 400 naming "does not support chat". Built from a prod
replay.

**The world.** Penny has just told the user, in her own real startup changelog
announcement, about two of her own new tools — ``synthesize`` and a preview/plan
mode — using the real merged PRs' own first sentences (#2949, #2948; repo content,
not private). One ordinary turn follows, mentioning a thing that is NOT anywhere in
her store, the same shape the replayed prod turn carried: a topic salient from
earlier in the conversation that a first reach for comes up empty. The user then
asks her, in one short open line, to go ahead and try the tool she was just told
about — naming no topic of its own, exactly as the replayed ask did not. The store
holds ten real, unrelated collections on a different topic entirely — one small
home-improvement project fed by three research collections, and six standing hobby
collections — so a first draw against the mentioned-but-absent topic matches
nothing (no draw runs), and a second draw against what the store map actually shows
her matches and runs for real. Every name, collection and entry here is invented for
this case — the repo is public, and nothing from the real prod conversation appears
below.

**The claims.** STORE (survival) — every one of the ten collections the turn reads
holds exactly what it was seeded with; an open "try it" ask names nothing to write.
STORE (mechanism) — no tool call this turn made escaped its own execution with an
uncaught exception (#2953's own bug, on the unpatched wiring): `synthesize`'s own
HONEST refusals — nothing matched, a draw that didn't come back clean — stay
allowed; only a raw, uncaught exception counts, which is what the broken wiring
raises every time the draw actually runs. PROVENANCE — every specific value in the
reply traces to something the turn was given.

**What is NOT claimed, and why.** Which tool answers the ask, and which collections
it draws on, are both ROUTES, never claims (`docs/principles.md` §4.1): `synthesize`
is the tool built for exactly this ask, but `find`/`memory_read` reading some of
the same entries is an equally legitimate path, and nothing here forces a call.
`TOOL_SEQUENCE` is measured instead.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.changelog_commits import parse_commit_log
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.changelog_announcement import ChangelogEval
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD, TOOL_SEQUENCE
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "memory-synthesis"

_CASE_ID = "synthesize-open-ask-across-collections"

# Two real merged PRs' own RAW commit records — hash, full conventional title, and
# the merged PR's own real body (`git log`'s own `%H%x1f%s%x1f%b` shape,
# `scripts/store_git_info.sh`'s own format) — the two new tools the replayed prod
# turn's own changelog announcement actually named. Run through the REAL
# `parse_commit_log` (#2891), exactly as a real deploy's own generated commit dump
# would be, rather than hand-built `ChangelogEntry`s.
_RAW_COMMIT_RECORDS = (
    "1f10e9f72f45c31052ffdba2d74b683697344d03"
    "\x1ffeat(penny): synthesize summarises what her memory says about a query "
    "through a micro-context (#2949)\x1fAdds a read-only `synthesize` tool that "
    'answers "what does my memory say about X?" by reading several things of '
    "Penny's own memory at once and writing what they add up to, instead of "
    "leaving the model to assemble a pattern across several `find` calls itself.",
    "4437dc34f41ac1c26853430bab859fa5348f7146"
    "\x1ffeat(penny): a plan flag on state-changing tools previews the change "
    "without making it (#2948)\x1fAdds a `plan=true` argument to every "
    "state-changing memory tool that validates and describes what the call would "
    "do, then stops before anything is written or sent.",
)
_ENTRIES = parse_commit_log("\x1e".join(_RAW_COMMIT_RECORDS))
assert {entry.pr for entry in _ENTRIES} == {2949, 2948}, (
    "one of these two real commit lines no longer parses as a penny-scope entry"
)

# Between the announcement and the ask — a thing mentioned in passing that is NOT
# anywhere in this world's store, the replayed shape: a topic salient from earlier
# in the conversation that a first reach for comes up empty.
_UNRELATED_TURN = "oh, remind me — are you still keeping notes on my job search?"

# One small umbrella project fed by three narrower research collections — one
# cluster of this world's real store.
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

# A handful of unrelated hobby collections — the other cluster of this world's real
# store, with no relation to the project above or to the job-search topic mentioned
# and never backed by any of it.
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
assert not any("job search" in entry.lower() for synth in _COLLECTIONS for entry in synth.entries)

_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=(), stores=_COLLECTIONS)

_ASK = "okay try your synthesize tool"
_ALSO_PHRASED = (
    "go ahead, try the synthesize tool",
    "alright, give synthesize a try",
    "sure, try out your synthesize tool",
    "ok, go ahead and use synthesize",
)

_BEHAVIOUR = (
    "In the chat agent, right after her own startup changelog announcement of the "
    "synthesize tool, when the user asks her to try it — naming no topic of its "
    "own, against a store holding several real, unrelated collections — the turn "
    "completes without any tool call escaping with an uncaught exception."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_open_synthesize_ask_runs_without_a_tool_call_raising(
    changelog_eval: ChangelogEval, model: str
) -> None:
    """Right after the real startup announcement of the ``synthesize`` tool, one
    unrelated turn names a topic the store never backs, then a bare "try your
    synthesize tool" — no topic of its own — against ten small, unrelated
    collections across two areas of the user's life, the open shape that leaves
    the draw to pick what to read and what to say about it."""
    cohort: Cohort = await changelog_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        changelog_entries=_ENTRIES,
        # This sample's own `changelog` log carries exactly these same two real
        # commits (#2891: `Config.changelog_commits` is a PARAMETER, defaulting to
        # empty for every other case) — consistent with what the delivery turn
        # below actually tells her, never the real build's own commit history.
        changelog_commits=_ENTRIES,
        unrelated_turn=_UNRELATED_TURN,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE (survival) — the ask reads and tries a tool, it does not write.
    cohort.assert_what_the_store_held_survives()
    # STORE (mechanism) — the deterministic fact #2953's bug fails and the fix holds:
    # no tool call this turn made escaped with an uncaught exception.
    cohort.assert_no_tool_call_raised()
    # PROVENANCE — nothing in the reply is invented; also fails outright if the turn
    # aborted on a model error.
    cohort.assert_every_value_in_the_reply_is_sourced()
    # TOOL_SEQUENCE is measured and never asserted — `synthesize` is the tool built
    # for this ask, but reaching it is a route, not a claim.
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
