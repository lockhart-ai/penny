"""Duplicate-write recovery contract — when a collector's write is rejected as a
duplicate, the rejection now hands back the matched existing key + the next move,
and the live model must recover instead of key-hunting.

Production failure this pins (July 2026 prompt audit): a duplicate rejection told
the model a similar entry existed but not WHICH one, so it guessed keys, re-read
the collection, or retried variations — burning its step budget (~1,800
wholly-duplicate-rejected writes across ~18% of collector runs in a 4-week window;
the recovery attempts fed the max-steps / died-mid-run failure classes).  The
rejection message now names the matched key and, when the WHOLE batch was
duplicates, tells the collector it may close with ``done()``.

The slip is a model DECISION on a visible tool result, but a natural cycle only
rarely writes an exact duplicate, so we force ONE duplicate ``collection_write``
(``_InjectDuplicateWrite``) and let the REAL model drive the recovery off the
production rejection message.  The contract is STRUCTURAL, never wording:

  PASS = the collection's keys are UNCHANGED (dedup rejected the write; no
         confabulated / proliferated entries — ``update_entry`` keeps the same
         keys) AND the cycle closed via ``done()`` or pivoted to ``update_entry``
         (a clean recovery), rather than spiraling to the step ceiling.

The deterministic message content is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    _InjectDuplicateWrite,
    collection_entries,
    seed_collection,
    tool_call_keys,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_DUP_CONTENT,
    RECIPE_BOX_DUP_CONTENT_2,
    RECIPE_BOX_DUP_KEY,
    RECIPE_BOX_DUP_KEY_2,
    RECIPE_BOX_EXTRACTION_PROMPT,
    RECIPE_BOX_SEED_KEYS,
)

pytestmark = pytest.mark.eval


def _seed_recipe_box(db: Database) -> None:
    seed_collection(
        db,
        RECIPE_BOX,
        extraction_prompt=RECIPE_BOX_EXTRACTION_PROMPT,
        interval=3600,
    )


def _score_recovered_from_duplicate(db: Database, sent: list[str]) -> list[str]:
    """Pass iff the forced duplicate write was rejected (keys unchanged) AND the
    cycle recovered — closed via ``done()`` or pivoted to ``update_entry`` on the
    BOUND matched key — rather than key-hunting to the step ceiling.

    The load-bearing check for #1405 is the last one: every ``update_entry`` must
    target an EXISTING (matched) key.  The 47%-recovery failure was the model
    re-using its OWN rejected candidate key (``"sheet pan chicken fajitas"``, not a
    real entry key) → key-not-found → ping-pong.  A call whose key isn't a seed key
    IS that ping-pong, so it fails the case."""
    fails: list[str] = []
    keys = set(collection_entries(db, RECIPE_BOX.name))
    if keys != set(RECIPE_BOX_SEED_KEYS):
        fails.append(
            "collection keys changed on an all-duplicate cycle "
            f"(confabulated/proliferated writes): {sorted(keys)} vs seeded "
            f"{sorted(RECIPE_BOX_SEED_KEYS)}"
        )
    if not (tool_was_called(db, "done") or tool_was_called(db, "update_entry")):
        fails.append(
            "did not recover after the duplicate rejection — no done()/update_entry "
            "(key-hunted / spiraled to the step ceiling)"
        )
    stray_keys = [
        key for key in tool_call_keys(db, "update_entry") if key not in RECIPE_BOX_SEED_KEYS
    ]
    if stray_keys:
        fails.append(
            f"update_entry targeted non-existent key(s) {stray_keys} instead of the "
            "bound matched key — the key-not-found ping-pong the binding fixes"
        )
    return fails


async def test_duplicate_write_hands_back_key_and_recovers(guard_recovery_eval) -> None:
    """A single duplicate ``collection_write`` is rejected with the matched key BOUND
    into an update_entry call; the live model recovers (done() or update_entry on the
    bound key) without re-using its own rejected key."""
    await guard_recovery_eval(
        case_id="duplicate-write-recovery",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectDuplicateWrite(
            real, RECIPE_BOX.name, [(RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT)]
        ),
        score=_score_recovered_from_duplicate,
        min_pass_rate=0.75,
    )


async def test_multi_entry_duplicate_write_binds_every_key(guard_recovery_eval) -> None:
    """A BATCH of two duplicate writes — each matching a DIFFERENT existing key — is
    rejected with EVERY matched key bound into its own update_entry call; the live
    model recovers without re-using either rejected candidate key.  Guards the
    multi-entry gap (some rejected keys previously left with no match named)."""
    await guard_recovery_eval(
        case_id="duplicate-write-recovery-multi",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectDuplicateWrite(
            real,
            RECIPE_BOX.name,
            [
                (RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT),
                (RECIPE_BOX_DUP_KEY_2, RECIPE_BOX_DUP_CONTENT_2),
            ],
        ),
        score=_score_recovered_from_duplicate,
        min_pass_rate=0.75,
    )
