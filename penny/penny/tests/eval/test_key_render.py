"""Entry-key rendering copy-through contracts — driven against the REAL model.

The read surfaces render an entry key in **invocation form** (``key='<key>'``) so
the form the model READS is the form a key-taking tool accepts.  These cases are
the standing proof that the render form is load-bearing — a tripwire if anyone
reintroduces a copyable-wrong display (the old ``[key]`` bracket render, which the
model pasted verbatim into ``key="[key]"`` args — 225 observed leaks, #1404).

  copy-through (Case A) — the chat agent sees a collection's rendered keys (via
    ambient recall and/or an explicit read) then operates on ONE entry BY KEY; the
    intended entry is mutated AND no tool call in the run pasted a bracket-wrapped
    key.  Success alone can't discriminate the render forms once the teaching
    rejection (#1396) can turn a bracket call into a reject→retry that still
    eventually lands — so the *bracket-call count* is the signal, scored
    structurally.  Run both arms (old render vs this one) for the causal table; the
    NEW arm ships here as the durable contract.

  forced recovery (Case B) — one key-bearing call is sabotaged to carry a
    bracket-wrapped key, so the memory-tool teaching rejection fires on every
    sample; the live model must recover to the bare key and land the mutation.
    This validates the "did you mean" teaching rejection is actually load-bearing.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    _InjectBracketKey,
    bracket_wrapped_key_calls,
    collection_entries,
    seed_collection,
)
from penny.tests.eval.fixtures import BOARD_GAMES, BOARD_GAMES_EXTRACTION_PROMPT

pytestmark = pytest.mark.eval

# The entry the cases correct by key — a realistic multi-word key, seeded verbatim
# by ``seed_collection`` (key = text before ' — ').
_TARGET_KEY = "Ark Nova"
_TARGET_SEED_CONTENT = "Ark Nova — zoo-building card-driven strategy, heavy, 1-4 players."
_UPDATE_MESSAGE = (
    "in my board games collection the Ark Nova entry is out of date — please fix it "
    "to say it plays 1-4 players and runs about 150 minutes."
)


def _seed_board_games(db: Database) -> None:
    """A fully-formed board-games collection with multi-word entry keys."""
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        interval=3600,
        notify=True,
    )


def _target_mutated(db: Database) -> bool:
    """The intended entry still exists under its bare key and its content changed
    from the seed — the end-state proof the update actually landed by key."""
    entries = collection_entries(db, "board-games")
    return _TARGET_KEY in entries and entries[_TARGET_KEY] != _TARGET_SEED_CONTENT


def _score_copythrough(db: Database, before: set[str], reply: str) -> list[str]:
    """Case A: the intended entry was mutated AND no bracket-wrapped key was ever
    passed.  The bracket-call count is the load-bearing signal — the render must
    not tempt the model into pasting display brackets into a ``key=`` argument."""
    fails = []
    if not _target_mutated(db):
        fails.append(f"{_TARGET_KEY!r} not updated by key — content unchanged from seed")
    brackets = bracket_wrapped_key_calls(db)
    if brackets:
        fails.append(f"pasted display brackets into a key arg (render regressed): {brackets}")
    return fails


def _score_forced_recovery(db: Database, before: set[str], reply: str) -> list[str]:
    """Case B: the model recovered — the intended entry landed under its bare key
    despite the teaching rejection.  The "did the sabotage fire?" check lives in
    the harness (``chat_eval`` asserts the wrapper's ``bail_injected``): the raw
    response is persisted inside the real client BEFORE the injector mutates it,
    so the promptlog never shows the injected bracket form and can't be probed."""
    if not _target_mutated(db):
        return [f"did not recover — {_TARGET_KEY!r} never updated after the rejection"]
    return []


async def test_copythrough_update_by_key(chat_eval: ChatEval) -> None:
    """List keys → update one BY KEY, with zero bracket-wrapped key arguments.

    Ships as the new-render arm of the causal A/B (baseline = the old ``[key]``
    render; see the PR body for the arm-by-arm bracket-call table)."""
    await chat_eval(
        case_id="key-render-copythrough",
        message=_UPDATE_MESSAGE,
        seed=_seed_board_games,
        score=_score_copythrough,
    )


async def test_forced_bracket_key_recovery(chat_eval: ChatEval) -> None:
    """Sabotage the model's first key-bearing call to carry a bracket-wrapped key,
    so the teaching rejection fires; the live model must recover to the bare key
    and land the mutation within the run's step budget (a spiral → timeout → fail)."""
    await chat_eval(
        case_id="key-render-forced-recovery",
        message=_UPDATE_MESSAGE,
        seed=_seed_board_games,
        wrap_client=_InjectBracketKey,
        score=_score_forced_recovery,
    )
