"""Bracket-key guard: a bad key recovers (#1404).

Every read surface renders an entry key in **invocation form** (``key='<key>'``) so the
form the model READS is the form a key-taking tool accepts.  The old ``[key]`` bracket
render is what that replaced: the model pasted its brackets verbatim into ``key="[key]"``
arguments, 225 times in the observed corpus, and the memory-tool teaching rejection
(#1396) is what refuses such a call.  This case is the standing proof that the rejection
recovers:

  **forced recovery** — the model's first key-bearing call is sabotaged to carry a
    bracket-wrapped key, so the memory-tool teaching rejection fires and the live model has
    to recover to the bare key and land the mutation.  **Ported to the cohort structure
    under #2009** (`docs/eval-case-design.md`): five wordings of the one correction, the
    sabotage FIXED across them, end state asserted and route measured.  A sample the
    sabotage never fired on is a HARNESS exclusion — it corrected an unbroken turn and
    exercised no recovery — never a behavioural failure.  At the cost #2018 records: the
    sabotage watches three tool names, so a turn that reached the entry any other way
    leaves here rather than answering the claims below.

It is a MECHANISM guard — it pins a render form's teaching rejection, not a user-facing
story — so this module sits beside the other chat-recovery guards
(``test_chat_call_recovery`` · ``test_harmony_leak_recovery`` · …) rather than with the
NL-dispatch stories.

**The world is an INERT storage collection.**  The board-games fixture used to be seeded
with a hand-authored ``extraction_prompt`` (terminal ``4. done().`` and all), an hourly
schedule and notify on — dressing from before skill-backed instantiation, and from before
a stored program stopped carrying its own terminator.  The chat-side key contract needs
none of it: what it needs is entries with realistic multi-word keys and a surface that
renders them.  A collection that is storage and nothing else is also what a post-0108
world actually contains, so the slim fixture is the honest one.  The loud probe asserts all
of that out loud rather than trusting it.

Report-only (``min_pass_rate=None``): the thresholds are the code owner's to set once the
numbers are read.  Nothing here is the user's: the collection is a fixture of published
board-game titles, held for their multi-word keys, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory.types import render_key
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
    _InjectBracketKey,
    collection_entries,
    seed_collection,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import BOARD_GAMES
from penny.tests.eval.utils.worlds import World
from penny.tools.memory_tools import format_entries

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "prompt-render"

# The entry the case corrects BY KEY — a realistic multi-word key, seeded verbatim by
# ``seed_collection`` (key = the text before ' — ').
_TARGET_KEY = "Ark Nova"

# What every seeded key maps to, derived from the fixture rather than restated: the
# "nothing else was touched" claim compares against it, so a fixture edit moves both the
# world and its expectation together.
_SEEDED = {entry.split(" — ")[0]: entry for entry in BOARD_GAMES.entries}

_FORCED_RECOVERY_CASE_ID = "key-render-forced-recovery"

# The one sentence the PORTED case exists to check, in the fixed form: "In <the locus>, when
# <X>, Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename, and
# this is what renders above every number in the ported case's report.
_FORCED_RECOVERY_BEHAVIOUR = (
    "In the chat agent, when her first attempt to correct an entry is refused for wrapping "
    "the key in the brackets a display put round it, Penny comes back with the bare key and "
    "leaves the correction on the entry the user named and on nothing else."
)

# Every case on this world, for the ``make check`` probe that runs its assertions once per
# case — so a fixture edit that broke a case's premise fails there rather than an hour into
# a GPU run.
BRACKET_KEY_CASES = (_FORCED_RECOVERY_CASE_ID,)

_UPDATE_MESSAGE = (
    "in my board games collection the Ark Nova entry is out of date — please fix it "
    "to say it plays 1-4 players and runs about 150 minutes."
)

# Four more wordings of that same correction.  What varies is only how a person says it —
# which clause opens it, "fix" or "update" or "correct", whether the two facts are read as
# a list or as a sentence.  What does NOT vary is the entry named, the values it should end
# up carrying, or the collection it lives in — those are what the case measures against.
_UPDATE_PHRASINGS = (
    "can you update the Ark Nova entry in my board games collection? it plays "
    "1-4 players and runs about 150 minutes.",
    "my board games collection has Ark Nova wrong — it should say 1-4 players and "
    "about 150 minutes.",
    "please correct Ark Nova in board games so it reads 1-4 players and roughly 150 minutes.",
    "the Ark Nova entry in board games needs fixing: 1-4 players, around 150 minutes.",
)

# The correction is a conversation about a collection, so the world carries no page and the
# claims read what the turn was GIVEN.  Named for its EFFECT rather than for its emptiness:
# a world with no pages still installs the canned-browse stub, so a browse this turn had no
# business making comes back as the stub's no-results page rather than reaching anything.
_EMPTY_BROWSE = World(name="board-games", pages=(), keeps=(), excludes=())


def _seed_board_games(db: Database) -> None:
    """An INERT board-games collection: storage, entries with multi-word keys, no job.

    No ``extraction_prompt``, no schedule, no notify — a container carrying a program the
    contract never runs is a second thing that can drift, and since migration 0108 an inert
    collection is what a fresh deployment's registry is made of."""
    seed_collection(db, BOARD_GAMES)


# ── The loud probe: the keys really are rendered in invocation form ───────────


def assert_board_games_world(db: Database, case_id: str) -> None:
    """Everything the seed is responsible for, asserted out loud.

    Three claims, and a drift in any of them would be read as the model failing: the
    collection is INERT (a job on it would dispatch a cycle into the middle of the turn
    being measured), it holds the target entry under its BARE multi-word key (the key the
    turn corrects, and the key the sabotage wraps), and — the premise the whole contract
    rests on — the read surface renders that key in INVOCATION form, so what the model
    copies is what a ``key=`` argument takes.  Rendering it any other way would make this a
    measurement of a display nobody ships.

    Takes the DATABASE rather than the constructed Penny so the same assertions run
    without one: the pin in ``test_eval_harness.py`` seeds a plain DB and calls this, so a
    fixture edit that breaks any of the three fails inside ``make check`` rather than an
    hour into a GPU run."""
    row = db.memories.get(BOARD_GAMES.name)
    assert row is not None, f"{case_id}: the seeded collection must exist"
    job = {
        "extraction_prompt": row.extraction_prompt,
        "schedule": row.schedule,
        "notify": row.notify or None,
    }
    carried = {field: value for field, value in job.items() if value is not None}
    assert not carried, f"{case_id}: the collection must be inert, it carries {carried}"

    entries = collection_entries(db, BOARD_GAMES.name)
    assert entries == _SEEDED, f"{case_id}: the collection must hold exactly {_SEEDED}"

    rendered = format_entries(
        require_memory(db, BOARD_GAMES.name).read_all(), source=BOARD_GAMES.name
    )
    assert render_key(_TARGET_KEY) in rendered, (
        f"{case_id}: the read surface must render {_TARGET_KEY!r} in invocation form "
        f"({render_key(_TARGET_KEY)!r}), it renders:\n{rendered}"
    )


def _probe_board_games_world(case_id: str) -> Preparer:
    """The loud seed probe, run once the world is whole."""

    def prepare(penny: Penny) -> None:
        assert_board_games_world(penny.db, case_id)

    return prepare


# ── The ported recovery case's own claims (inline: one customer each) ──────────


def _target_rewritten_under_its_bare_key(sample: SampleObservation, _world: World) -> Answer:
    """The mutation landed on the entry the ask named, under the key it is really filed
    under — the recovery's whole point, read as END STATE.

    Not "a call went out carrying the bare key": that is a ROUTE, and it was keyed to
    ``update_entry`` on the case this ports, which failed a sample that recovered through
    ``collection_write`` while the entry itself landed correctly.  What matters is which
    key the store was left holding a new value under, and a value written under
    ``[Ark Nova]`` satisfies none of this sentence.

    It claims the value MOVED and never what it moved TO, deliberately.  The ask supplies a
    playtime ("about 150 minutes") and a player count ("1-4 players"), and both are things a
    correct rewrite may legitimately render another way — 150 minutes as ``2.5 hours``, the
    range as ``1 to 4`` — so an assertion naming either would fail a correct run for a
    cosmetic reason.  What this case is about is which KEY the correction landed under, and
    that has exactly one strictly-identifiable form."""
    written = [
        entry
        for entry in sample.entries
        if entry.collection == BOARD_GAMES.name and entry.key == _TARGET_KEY
    ]
    if not written:
        wrote = sorted({f"{entry.collection}/{entry.key}" for entry in sample.entries})
        return False, f"the round wrote {wrote or 'nothing'}, never {_TARGET_KEY!r}"
    stale = [entry for entry in written if entry.content == _SEEDED[_TARGET_KEY]]
    return not stale, f"{_TARGET_KEY!r} was rewritten with the value it already had"


def _nothing_else_in_the_collection_moved(sample: SampleObservation, _world: World) -> Answer:
    """Every other entry reads exactly as it was seeded, and the collection gained none.

    Read off what the store HOLDS rather than off what the round wrote, because the two
    ways this fails are invisible in a list of writes: an entry the round DELETED is simply
    absent there, and so is one it never touched.  The gained-key arm is also where a
    recovery that filed the value under ``[Ark Nova]`` as a fresh key surfaces — the
    display form made durable, which is the render's own mistake outliving the turn."""
    held = {
        entry.key: entry.content
        for entry in sample.held
        if entry.collection == BOARD_GAMES.name and entry.key is not None
    }
    moved = sorted(
        key for key, content in _SEEDED.items() if key != _TARGET_KEY and held.get(key) != content
    )
    gained = sorted(key for key in held if key not in _SEEDED)
    return not (moved or gained), f"changed or lost {moved}; gained {gained}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_forced_bracket_key_recovery(chat_eval: ChatEval, model: str) -> None:
    """The first key-bearing call is sabotaged to carry a bracket-wrapped key, so the
    teaching rejection fires; the live model must come back with the bare key and land the
    mutation inside the run's step budget."""
    cohort = await chat_eval(
        case_id=_FORCED_RECOVERY_CASE_ID,
        behaviour=_FORCED_RECOVERY_BEHAVIOUR,
        model=model,
        world=_EMPTY_BROWSE,
        ask=_UPDATE_MESSAGE,
        also_phrased=_UPDATE_PHRASINGS,
        samples_per_phrasing=3,
        seed=_seed_board_games,
        prepare=_probe_board_games_world(_FORCED_RECOVERY_CASE_ID),
        wrap_client=_InjectBracketKey,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(
        f"state: {_TARGET_KEY!r} was rewritten under its bare key",
        _target_rewritten_under_its_bare_key,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: nothing else in the collection moved",
        _nothing_else_in_the_collection_moved,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
