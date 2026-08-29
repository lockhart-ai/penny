"""Bracket-key guards: the render never teaches a bad key, and a bad key recovers (#1404).

Every read surface renders an entry key in **invocation form** (``key='<key>'``) so the
form the model READS is the form a key-taking tool accepts.  These two cases are the
standing proof that the render form is load-bearing, and the tripwire if anyone
reintroduces a copyable-wrong display — the old ``[key]`` bracket render, whose brackets
the model pasted verbatim into ``key="[key]"`` arguments, 225 times in the observed corpus.

  **copy-through** — the chat agent reads a collection's rendered keys and operates on ONE
    entry BY KEY.  The intended entry is updated, no call anywhere in the run pasted a
    bracket-wrapped key, and the rest of the collection is exactly as it was seeded.
    Success alone cannot tell the two render forms apart, because the teaching rejection
    (#1396) can turn a bracket call into a reject-and-retry that still eventually lands —
    so the BRACKET-CALL COUNT is the signal, and it is read structurally off every
    key-bearing call the run made.

  **forced recovery** — the model's first key-bearing call is sabotaged to carry a
    bracket-wrapped key, so the memory-tool teaching rejection fires and the live model has
    to recover to the bare key and land the mutation.  **Ported to the cohort structure
    under #2009** (`docs/eval-case-design.md`): five wordings of the one correction, the
    sabotage FIXED across them, end state asserted and route measured.  A sample the
    sabotage never fired on is a HARNESS exclusion — it corrected an unbroken turn and
    exercised no recovery — never a behavioural failure.  At the cost #2018 records: the
    sabotage watches three tool names, so a turn that reached the entry any other way
    leaves here rather than answering the claims below.

The two are deliberately NOT one cohort: they are the same ask against the same world with
and without a fault, which is two behaviours, and pooling them would average a recovery
rate into a copy-through rate.  Copy-through stays on the scorer path, where its signal —
what the model's CALLS carried — lives; a route is not something the cohort structure
asserts.

Both cases are MECHANISM guards — they pin a render form and its teaching rejection, not a
user-facing story — so this module sits beside the other chat-recovery guards
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
    Check,
    Preparer,
    _InjectBracketKey,
    bracket_wrapped_key_calls,
    collection_entries,
    routing_clean,
    seed_collection,
    tool_call_keys,
    tool_call_sequence,
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

# The entry both cases correct BY KEY — a realistic multi-word key, seeded verbatim by
# ``seed_collection`` (key = the text before ' — ').
_TARGET_KEY = "Ark Nova"

# What every seeded key maps to, derived from the fixture rather than restated: the
# "nothing else was touched" claim compares against it, so a fixture edit moves both the
# world and its expectation together.
_SEEDED = {entry.split(" — ")[0]: entry for entry in BOARD_GAMES.entries}

_COPYTHROUGH_CASE_ID = "key-render-copythrough"
_FORCED_RECOVERY_CASE_ID = "key-render-forced-recovery"

# Both cases, for the ``make check`` probe that runs their shared world's assertions once
# per case — so a fixture edit that broke either one's premise fails there rather than an
# hour into a GPU run.
BRACKET_KEY_CASES = (_COPYTHROUGH_CASE_ID, _FORCED_RECOVERY_CASE_ID)

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


# ── The copy-through case's checks ────────────────────────────────────────────


def _target_updated_check(db: Database) -> Check:
    """The intended entry still exists under its BARE key and its content changed from the
    seed — the end-state proof the update landed by key rather than beside it.

    Both halves in one check because they are one fact: an entry that vanished and an entry
    that never changed are the same miss read two ways, and the rationale says which."""
    entries = collection_entries(db, BOARD_GAMES.name)
    held = entries.get(_TARGET_KEY)
    updated = held is not None and held != _SEEDED[_TARGET_KEY]
    if held is None:
        rationale = f"{_TARGET_KEY!r} is gone — the collection holds {sorted(entries)}"
    elif not updated:
        rationale = f"{_TARGET_KEY!r} still reads as it was seeded"
    else:
        rationale = None
    return Check(
        f"state: {_TARGET_KEY!r} was updated under its bare key",
        updated,
        rationale=rationale,
        kind="state",
    )


def _no_bracket_call_check(db: Database) -> Check:
    """No call anywhere in the run pasted a display-bracketed key into an argument.

    The load-bearing signal of the copy-through case, and the reason success alone is not
    enough: the teaching rejection can turn a bracket call into a retry that lands, so the
    end state would look identical while the render was busy teaching the wrong thing.

    It stays on THIS case and does not port to its sabotaged sibling: there, every sample's
    first key-bearing call carries brackets because the harness put them there."""
    offenders = bracket_wrapped_key_calls(db)
    return Check(
        "calls: no bracket-wrapped key was pasted into an argument",
        not offenders,
        rationale=f"passed {offenders}" if offenders else None,
        kind="spine",
    )


def _rest_untouched_check(db: Database) -> Check:
    """Every OTHER entry reads exactly as it was seeded — the nothing-else-touched claim,
    and the guard against a turn that corrected the collection by rewriting it.

    Read against the fixture the seed was built from, so a fixture edit moves the world and
    its expectation in one place."""
    entries = collection_entries(db, BOARD_GAMES.name)
    moved = sorted(
        key
        for key, content in _SEEDED.items()
        if key != _TARGET_KEY and entries.get(key) != content
    )
    return Check(
        "state: the rest of the collection is untouched",
        not moved,
        rationale=f"also changed or lost {moved}" if moved else None,
        kind="state",
    )


def _key_advisories(db: Database, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — the calls it made, the keys it
    tried, and the answer it gave — so a report shows the run whichever way it went and the
    wording is read where wording is read: at joint review."""
    return [
        Check(f"fired: {tool_call_sequence(db)}", True, kind="proc", scored=False),
        Check(
            f"update keys tried: {tool_call_keys(db, 'update_entry')}",
            True,
            kind="proc",
            scored=False,
        ),
        Check(f"answered: {reply!r}", True, kind="reply", scored=False),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


def _score_copythrough(db: Database, before: set[str], reply: str) -> list[Check]:
    """The intended entry was updated by key, no bracket-wrapped key was ever passed, and
    the rest of the collection is as it was seeded."""
    return [
        _target_updated_check(db),
        _no_bracket_call_check(db),
        _rest_untouched_check(db),
        *_key_advisories(db, reply),
    ]


async def test_copythrough_update_by_key(chat_eval: ChatEval) -> None:
    """Read the rendered keys, update ONE entry by key, and paste no display brackets on
    the way — the standing proof that the invocation-form render does not teach the mistake
    the ``[key]`` render taught."""
    await chat_eval(
        case_id=_COPYTHROUGH_CASE_ID,
        family=_FAMILY,
        message=_UPDATE_MESSAGE,
        seed=_seed_board_games,
        prepare=_probe_board_games_world(_COPYTHROUGH_CASE_ID),
        score=_score_copythrough,
        min_pass_rate=None,
    )


# ── The ported recovery case's own claims (inline: one customer each) ──────────


def _target_rewritten_under_its_bare_key(sample: SampleObservation, _world: World) -> Answer:
    """The mutation landed on the entry the ask named, under the key it is really filed
    under — the recovery's whole point, read as END STATE.

    Not "a call went out carrying the bare key": that is a ROUTE, and it was keyed to
    ``update_entry`` on the case this ports, which failed a sample that recovered through
    ``collection_write`` while the entry itself landed correctly.  What matters is which
    key the store was left holding a new value under, and a value written under
    ``[Ark Nova]`` satisfies none of this sentence."""
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
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
