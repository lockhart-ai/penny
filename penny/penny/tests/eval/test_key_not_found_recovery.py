"""Key-not-found write-vs-update recovery contract — after a key-not-found
rejection the model finds the entry under a different key and must UPDATE it with
``update_entry``, not ``collection_write`` it (a duplicate the dedup rejects).

Production residue this pins (July 2026 tool-failure audit, item #11): the Jun-13
key-not-found rewrite ("list the keys with collection_keys(...), or search by
content with read_similar(...)") moved recovery 47% → 88%, but the remaining
failures share a shape — the model runs collection_keys, finds the right key, then
picks ``collection_write`` (→ duplicate-rejected) instead of ``update_entry``,
because the rejection named the READ tools but not the write-vs-update decision.
The rejection now closes that gap: it tells the model to refresh the existing entry
with ``update_entry(key=<the key you found>, ...)`` and that ``collection_write``
creates NEW keys only.

The slip is a model DECISION on a visible tool result, and a natural cycle only
rarely probes an existing entry with a near-miss key, so we force ONE such
``collection_get`` (``_InjectKeyMiss``) and let the REAL model drive the recovery
off the production rejection message.  The contract is STRUCTURAL, never wording:

  PASS = the model reached ``update_entry`` (the correct write-vs-update choice)
         AND the box's keys are UNCHANGED (no fresh / duplicate key from a
         ``collection_write``) AND the existing fajitas entry was refreshed with
         the new detail — rather than proliferating a key or spiraling.

The deterministic message content is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    _InjectKeyMiss,
    collection_entries,
    seed_collection,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_ENRICH_PROMPT,
    RECIPE_BOX_FAJITAS_KEY,
    RECIPE_BOX_FAJITAS_SEED_CONTENT,
    RECIPE_BOX_INTENT,
    RECIPE_BOX_NEAR_MISS_KEY,
    RECIPE_BOX_SEED_KEYS,
)

pytestmark = pytest.mark.eval


def _seed_recipe_box(db: Database) -> None:
    seed_collection(
        db,
        RECIPE_BOX,
        extraction_prompt=RECIPE_BOX_ENRICH_PROMPT,
        intent=RECIPE_BOX_INTENT,
        interval=3600,
    )


def _score_reached_update_not_write(db: Database, sent: list[str]) -> list[str]:
    """Pass iff the model recovered from the key-not-found rejection to the right
    write path: it called ``update_entry`` (not ``collection_write``), left the
    box's keys unchanged (no proliferated / duplicate key), and the enrichment
    landed on the existing fajitas entry."""
    fails: list[str] = []
    entries = collection_entries(db, RECIPE_BOX.name)
    keys = set(entries)
    if keys != set(RECIPE_BOX_SEED_KEYS):
        fails.append(
            "box keys changed — the model created a fresh/duplicate key via "
            f"collection_write instead of update_entry: {sorted(keys)} vs seeded "
            f"{sorted(RECIPE_BOX_SEED_KEYS)}"
        )
    if not tool_was_called(db, "update_entry"):
        fails.append(
            "did not reach update_entry after the key-not-found rejection — picked "
            "collection_write (→ duplicate-rejected) or gave up"
        )
    if entries.get(RECIPE_BOX_FAJITAS_KEY, "") == RECIPE_BOX_FAJITAS_SEED_CONTENT:
        fails.append(
            "the existing fajitas entry was not refreshed — update_entry did not "
            "land the enrichment on the found key"
        )
    return fails


async def test_key_not_found_recovers_to_update_not_write(guard_recovery_eval) -> None:
    """A near-miss ``collection_get`` returns the key-not-found rejection; the live
    model finds the real key and refreshes the existing entry with ``update_entry``
    rather than ``collection_write``-ing a duplicate."""
    await guard_recovery_eval(
        case_id="key-not-found-write-vs-update",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectKeyMiss(real, RECIPE_BOX.name, RECIPE_BOX_NEAR_MISS_KEY),
        score=_score_reached_update_not_write,
        min_pass_rate=0.75,
    )
