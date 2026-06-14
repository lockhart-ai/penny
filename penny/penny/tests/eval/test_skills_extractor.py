"""Skills-extractor contracts — the background collector that grows/tunes skills.

Runs the REAL skills collection (its migration-0043 extraction prompt + the
collector runtime rules) via ``collector.run_for("skills")`` over synthetic
conversation logs, and checks the entry-level outcome on the skills collection
by diffing its entries before/after the cycle:

  teach        — "from now on when I say X, do Y"  → a new skill written
  correct-sub  — "stop telling me X"               → an existing skill edited
  correct-scope— "only do X for Y"                 → an existing skill edited
  deprecate    — "drop that rule entirely"         → an existing skill deleted
  lift         — a one-off then taught as a rule   → a new skill written
  quiet        — normal chat, no teaching          → skills unchanged
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import EntryInput, LogEntryInput
from penny.tests.eval.conftest import CollectorScorer, collection_entries

pytestmark = pytest.mark.eval

_SKILLS = "skills"

# Existing skills the read_similar probe surfaces (for correction/deprecation).
_EXISTING = {
    "recipes-include-prep-and-difficulty": (
        "TRIGGER\nUser asks about recipes.\n\nSTEPS\n1. browse() for the recipe.\n"
        "2. Answer with ingredients, prep time, AND difficulty level."
    ),
    "research-with-wikipedia-link": (
        "TRIGGER\nUser asks you to research a topic.\n\nSTEPS\n1. browse() for the topic.\n"
        "2. Always include a Wikipedia link alongside other sources.\n3. Answer."
    ),
}


def _seed(user_msgs, penny_msgs=(), existing_key=None):
    """Build a seeder: the conversation logs + (optionally) an existing skill."""

    def _apply(db: Database) -> None:
        db.memories.append(
            PennyConstants.MEMORY_USER_MESSAGES_LOG,
            [LogEntryInput(content=message) for message in user_msgs],
            author="user",
        )
        if penny_msgs:
            db.memories.append(
                PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
                [LogEntryInput(content=message) for message in penny_msgs],
                author="penny",
            )
        if existing_key is not None:
            db.memories.write(
                _SKILLS,
                [EntryInput(key=existing_key, content=_EXISTING[existing_key])],
                author="user",
            )

    return _apply


def _snapshot(db: Database) -> dict[str, str]:
    return collection_entries(db, _SKILLS)


def _score_write(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, _SKILLS)
    new_keys = set(after) - set(before_entries)
    if not new_keys:
        return ["expected a new skill written, none added"]
    return [
        f"written skill {key!r} lacks TRIGGER/STEPS shape"
        for key in new_keys
        if "TRIGGER" not in after[key].upper()
    ]


def _score_update(existing_key: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        if existing_key not in after:
            return [f"existing skill {existing_key!r} disappeared"]
        if after[existing_key] != before_entries.get(existing_key):
            return []  # edited in place — the correction landed
        if set(after) - set(before_entries):
            return [f"wrote a new skill instead of editing {existing_key!r} (fragments)"]
        return [f"correction lost — {existing_key!r} unchanged"]

    return _score


def _score_delete(existing_key: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        after = collection_entries(db, _SKILLS)
        if existing_key in after:
            return [f"deprecation lost — {existing_key!r} still present"]
        return []

    return _score


def _score_no_op(db: Database, before: object, sent: list[str]) -> list[str]:
    before_entries = cast("dict[str, str]", before)
    after = collection_entries(db, _SKILLS)
    if after != before_entries:
        return ["mutated skills on a quiet cycle"]
    return []


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_teach(collector_eval) -> None:
    await collector_eval(
        case_id="skills-teach",
        collection=_SKILLS,
        seed=_seed(
            [
                "what's a good pasta recipe",
                "from now on when i ask about recipes, always include the prep time and difficulty",
            ]
        ),
        snapshot=_snapshot,
        score=_score_write,
    )


async def test_correct_substance(collector_eval) -> None:
    await collector_eval(
        case_id="skills-correct-sub",
        collection=_SKILLS,
        seed=_seed(
            [
                "what's a good carbonara recipe",
                "wait, when i ask about recipes, stop telling me the difficulty — "
                "just give me prep time",
            ],
            existing_key="recipes-include-prep-and-difficulty",
        ),
        snapshot=_snapshot,
        score=_score_update("recipes-include-prep-and-difficulty"),
    )


async def test_correct_scope(collector_eval) -> None:
    await collector_eval(
        case_id="skills-correct-scope",
        collection=_SKILLS,
        seed=_seed(
            [
                "research the latest wireless earbuds for me",
                "you don't need wikipedia for product comparisons — "
                "only do the wiki link for historical topics",
            ],
            existing_key="research-with-wikipedia-link",
        ),
        snapshot=_snapshot,
        score=_score_update("research-with-wikipedia-link"),
    )


async def test_deprecate(collector_eval) -> None:
    await collector_eval(
        case_id="skills-deprecate",
        collection=_SKILLS,
        seed=_seed(
            [
                "on second thought, drop the recipes skill entirely — "
                "never mind tracking that, just delete it",
            ],
            existing_key="recipes-include-prep-and-difficulty",
        ),
        snapshot=_snapshot,
        score=_score_delete("recipes-include-prep-and-difficulty"),
    )


async def test_lift(collector_eval) -> None:
    await collector_eval(
        case_id="skills-lift",
        collection=_SKILLS,
        seed=_seed(
            [
                "find me a good thai place near home",
                "from now on whenever i ask about restaurants, always include the price range",
            ],
            penny_msgs=["Sukhumvit Garden — solid pad thai, ~10 min away."],
        ),
        snapshot=_snapshot,
        score=_score_write,
    )


async def test_quiet(collector_eval) -> None:
    await collector_eval(
        case_id="skills-quiet",
        collection=_SKILLS,
        seed=_seed(
            ["what's the weather today?", "thanks!"],
            penny_msgs=["Sunny and 72°F!", "anytime — let me know if you need anything else"],
        ),
        snapshot=_snapshot,
        score=_score_no_op,
    )
