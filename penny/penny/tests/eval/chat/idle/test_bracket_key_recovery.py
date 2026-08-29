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
    bracket-wrapped key, so the memory-tool teaching rejection fires on every sample and
    the live model has to recover to the bare key and land the mutation.  This is the only
    exercise of ``_bracket_key_rejection`` under a guaranteed trigger; that the sabotage
    actually fired is the framework's own guard Check, prepended by ``chat_eval``'s graded
    path (#1718), so a run where it never fired cannot score green off this scorer alone.

Both cases are MECHANISM guards — they pin a render form and its teaching rejection, not a
user-facing story — so this module sits beside the other chat-recovery guards
(``test_chat_call_recovery`` · ``test_key_not_found_recovery`` ·
``test_harmony_leak_recovery`` · …) rather than with the NL-dispatch stories.

**The world is an INERT storage collection.**  The board-games fixture used to be seeded
with a hand-authored ``extraction_prompt`` (terminal ``4. done().`` and all), an hourly
schedule and notify on — dressing from before skill-backed instantiation, and from before
a stored program stopped carrying its own terminator.  The chat-side key contract needs
none of it: what it needs is entries with realistic multi-word keys and a surface that
renders them.  A collection that is storage and nothing else is also what a post-0108
world actually contains, so the slim fixture is the honest one.  The loud probe asserts all
of that out loud rather than trusting it.

Report-only (``min_pass_rate=None``): the thresholds are the code owner's to set once the
numbers are read.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.memory.types import render_key
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
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
from penny.tests.eval.utils.fixtures import BOARD_GAMES
from penny.tools.memory_tools import format_entries

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "prompt-render"

# The entry both cases correct BY KEY — a realistic multi-word key, seeded verbatim by
# ``seed_collection`` (key = the text before ' — ').
_TARGET_KEY = "Ark Nova"

# What every seeded key maps to, derived from the fixture rather than restated: the
# "nothing else was touched" check compares against it, so a fixture edit moves both the
# world and its expectation together.
_SEEDED = {entry.split(" — ")[0]: entry for entry in BOARD_GAMES.entries}

_UPDATE_MESSAGE = (
    "in my board games collection the Ark Nova entry is out of date — please fix it "
    "to say it plays 1-4 players and runs about 150 minutes."
)


class _BracketKeyCase(NamedTuple):
    """One agreed turn, and whether its first key-bearing call is sabotaged.

    ``sabotaged`` is the whole difference between the two cases — the same message against
    the same world, once as the model writes it and once with the bracket habit forced —
    so the world, the seed and the shared checks are declared once."""

    case_id: str
    message: str
    sabotaged: bool


_COPYTHROUGH = _BracketKeyCase(
    case_id="key-render-copythrough",
    message=_UPDATE_MESSAGE,
    sabotaged=False,
)

_FORCED_RECOVERY = _BracketKeyCase(
    case_id="key-render-forced-recovery",
    message=_UPDATE_MESSAGE,
    sabotaged=True,
)

BRACKET_KEY_CASES = (_COPYTHROUGH, _FORCED_RECOVERY)


def _seed_board_games(db: Database) -> None:
    """An INERT board-games collection: storage, entries with multi-word keys, no job.

    No ``extraction_prompt``, no schedule, no notify — a container carrying a program the
    contract never runs is a second thing that can drift, and since migration 0108 an inert
    collection is what a fresh deployment's registry is made of."""
    seed_collection(db, BOARD_GAMES)


# ── The loud probe: the keys really are rendered in invocation form ───────────


def assert_board_games_world(db: Database, case: _BracketKeyCase) -> None:
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
    assert row is not None, f"{case.case_id}: the seeded collection must exist"
    job = {
        "extraction_prompt": row.extraction_prompt,
        "schedule": row.schedule,
        "notify": row.notify or None,
    }
    carried = {field: value for field, value in job.items() if value is not None}
    assert not carried, f"{case.case_id}: the collection must be inert, it carries {carried}"

    entries = collection_entries(db, BOARD_GAMES.name)
    assert entries == _SEEDED, f"{case.case_id}: the collection must hold exactly {_SEEDED}"

    rendered = format_entries(
        require_memory(db, BOARD_GAMES.name).read_all(), source=BOARD_GAMES.name
    )
    assert render_key(_TARGET_KEY) in rendered, (
        f"{case.case_id}: the read surface must render {_TARGET_KEY!r} in invocation form "
        f"({render_key(_TARGET_KEY)!r}), it renders:\n{rendered}"
    )


def _probe_board_games_world(case: _BracketKeyCase) -> Preparer:
    """The loud seed probe, run once the world is whole."""

    def prepare(penny: Penny) -> None:
        assert_board_games_world(penny.db, case)

    return prepare


# ── Checks ────────────────────────────────────────────────────────────────────


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
    end state would look identical while the render was busy teaching the wrong thing."""
    offenders = bracket_wrapped_key_calls(db)
    return Check(
        "calls: no bracket-wrapped key was pasted into an argument",
        not offenders,
        rationale=f"passed {offenders}" if offenders else None,
        kind="spine",
    )


def _recovered_to_the_bare_key_check(db: Database) -> Check:
    """After the rejection, a key-bearing call went out carrying the BARE key — the
    recovery itself, read as a call rather than inferred from the end state.

    Its own check beside the end state because the two can disagree: a run that never
    retried and a run that retried and failed to write both leave the entry untouched, and
    only the calls say which happened."""
    keys = tool_call_keys(db, "update_entry")
    recovered = _TARGET_KEY in keys
    return Check(
        f"calls: a call went out carrying the bare key {_TARGET_KEY!r}",
        recovered,
        anchor="update_entry(",
        rationale=None if recovered else f"the update keys it tried were {keys or 'none'}",
        kind="spine",
    )


def _no_bracket_key_stored_check(db: Database) -> Check:
    """No bracket-wrapped key SURVIVES in the collection — the recovery's end-state half.

    Distinct from the intended entry landing: a run that recovered by writing the value
    under ``[Ark Nova]`` as a fresh key would leave the target untouched and the display
    form baked into the store, which is the render's own mistake made durable."""
    stored = sorted(key for key in collection_entries(db, BOARD_GAMES.name) if key not in _SEEDED)
    return Check(
        "state: no bracket-wrapped key was stored",
        not stored,
        rationale=f"the collection gained {stored}" if stored else None,
        kind="state",
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


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_copythrough(db: Database, before: set[str], reply: str) -> list[Check]:
    """The intended entry was updated by key, no bracket-wrapped key was ever passed, and
    the rest of the collection is as it was seeded."""
    return [
        _target_updated_check(db),
        _no_bracket_call_check(db),
        _rest_untouched_check(db),
        *_key_advisories(db, reply),
    ]


def _score_forced_recovery(db: Database, before: set[str], reply: str) -> list[Check]:
    """The model recovered from the forced bracket key: a call went out carrying the bare
    key, the intended entry landed under it, no bracketed key was stored, and the rest of
    the collection is as it was seeded.

    The "did the sabotage fire?" contract is the framework-injected guard ``chat_eval``'s
    graded path PREPENDS on a ``wrap_client`` case (#1718), so a graded return cannot drop
    it.  The raw response is persisted inside the real client BEFORE the injector mutates
    it, so the promptlog never shows the injected bracket form and no scorer could probe
    for it — hence the harness-side guard rather than a check here."""
    return [
        _recovered_to_the_bare_key_check(db),
        _target_updated_check(db),
        _no_bracket_key_stored_check(db),
        _rest_untouched_check(db),
        *_key_advisories(db, reply),
    ]


async def _run_bracket_key_case(chat_eval: ChatEval, case: _BracketKeyCase) -> None:
    """Drive one bracket-key case: the inert collection seeded and probed, the injector
    installed only where the case declares a sabotage, the scorer bound to the case.
    Report-only — the thresholds are the code owner's to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        family=_FAMILY,
        message=case.message,
        seed=_seed_board_games,
        prepare=_probe_board_games_world(case),
        wrap_client=_InjectBracketKey if case.sabotaged else None,
        score=_score_forced_recovery if case.sabotaged else _score_copythrough,
        min_pass_rate=None,
    )


async def test_copythrough_update_by_key(chat_eval: ChatEval) -> None:
    """Read the rendered keys, update ONE entry by key, and paste no display brackets on
    the way — the standing proof that the invocation-form render does not teach the mistake
    the ``[key]`` render taught."""
    await _run_bracket_key_case(chat_eval, _COPYTHROUGH)


async def test_forced_bracket_key_recovery(chat_eval: ChatEval) -> None:
    """The first key-bearing call is sabotaged to carry a bracket-wrapped key, so the
    teaching rejection fires on every sample; the live model must come back with the bare
    key and land the mutation inside the run's step budget."""
    await _run_bracket_key_case(chat_eval, _FORCED_RECOVERY)
