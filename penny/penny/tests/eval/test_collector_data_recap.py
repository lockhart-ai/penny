"""Survival contract for a collector DATA/LOG tool on the collector surface (#1513,
epic #1478).

The collector data/log tools (`log_append` / `log_create` / `read_published_latest`
/ `collection_read_random` / `collection_get` / `collection_keys` / `exists`) are
mostly collector-driven, so their survival surface is the ``done(summary=…)`` call,
which #1499's collector-done mechanism already carries.  Their narration strings are
pinned deterministically in ``tests/tools/test_collector_data_narration.py``.

This adds the one CLEAN live done()-survival case among them: `read_published_latest`
via the migration-seeded ``notifier`` consumer.  The notifier reads the published
stream (`read_published_latest`), grounds the find, delivers it, and closes with
``done()`` — so the done() summary must reflect that it shared the new find, never
claim it shared nothing.  (Delivery itself — the send — is separately contracted by
``test_extractors::notifier-delivers-published``; this scores the *summary survival*.)

The other six are unit-only (no clean standalone collector cycle whose done() summary
deterministically reflects that specific read/append) — see the issue report.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, RecallMode
from penny.tests.eval.conftest import CollectorEval, last_tool_args
from penny.tests.eval.fixtures import (
    RESEARCH_WATCHER,
    RESEARCH_WATCHER_INTENT,
)

pytestmark = pytest.mark.eval

# The done() summary honestly reflects that the cycle SHARED/delivered the new find.
_SHARED = re.compile(
    r"deliver|shar|notif|told|sen[dt]|messag|alert|let .*know|ping|posted|"
    r"passed (it |the |them )?(on|along)|update|new (find|game|entry|release)|"
    r"hollow verge",
    re.I,
)
# The done() summary must NOT falsely claim there was nothing to share when a fresh
# published find WAS delivered.
_NOTHING = re.compile(
    r"nothing (new|to share)|no new|no updates?|already (up to date|shared|seen)|"
    r"no fresh|none to",
    re.I,
)


def _seed_notifier_with_published_find(db: Database) -> None:
    """A published producer holding one fresh find. The notifier consumer that
    delivers it is migration-seeded (0067), so this drives the SHIPPED prompt."""
    db.memories.create_collection(
        RESEARCH_WATCHER.name,
        RESEARCH_WATCHER.description,
        Inclusion(RESEARCH_WATCHER.inclusion),
        RecallMode.RELEVANT,
        intent=RESEARCH_WATCHER_INTENT,
        published=True,
    )
    db.memory(RESEARCH_WATCHER.name).write(
        [
            EntryInput(
                key="Hollow Verge",
                content="Hollow Verge — a hand-drawn metroidvania with grappling-hook "
                "traversal and a branching map. https://indiegames.example.com/hollow-verge",
            )
        ],
        author="producer",
    )


def _score_shared(db: Database, before: object, sent: list[str]) -> list[str]:
    fails: list[str] = []
    done = last_tool_args(db, "done")
    summary = str((done or {}).get("summary", ""))
    print(f"[DONE read_published_latest] :: {summary!r}")
    if done is None:
        fails.append("cycle never closed with done()")
    elif not _SHARED.search(summary):
        fails.append(f"done() summary didn't reflect sharing the new find: {summary!r}")
    elif _NOTHING.search(summary):
        fails.append(f"done() summary falsely claimed nothing to share: {summary!r}")
    return fails


async def test_done_reflects_published_read(collector_eval: CollectorEval) -> None:
    await collector_eval(
        case_id="collector-data-recap-published-read",
        collection="notifier",  # migration-seeded (0067) — drives the shipped prompt
        seed=_seed_notifier_with_published_find,
        score=_score_shared,
        min_pass_rate=0.75,
    )
