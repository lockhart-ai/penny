"""Two-stage recall routing contract — driven through the REAL recall path.

The old scripts/prompt_validation/retrieval.py reimplemented stage-1 routing and
stage-2 hybrid ranking with the ``similarity`` primitives.  This drives the
production code instead: seed the topical collections, embed them, then render
``ChatAgent._recall_section`` for each synthetic message and assert on the block
it produces.  Embeddings are deterministic, so each message is scored once and a
pass-rate asserted (routing is imperfect on real vectors, as it is in prod).

  stage 1 — each ``relevant`` collection is included iff the message matches its
            description anchor; expected topical collections in, others out.
  stage 2 — the expected seed skill surfaces in the rendered block.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import RecallEval, seed_collection
from penny.tests.eval.fixtures import MESSAGES, TOPICAL_COLLECTIONS, Message

pytestmark = pytest.mark.eval


def _seed_topicals(db: Database) -> None:
    for collection in TOPICAL_COLLECTIONS:
        seed_collection(db, collection)


def _check_routing(recall_block: str, message: Message) -> list[str]:
    block = recall_block.lower()
    fails = []
    for expected in message.collections:
        if expected.lower() not in block:
            fails.append(f"[{message.id}] expected collection '{expected}' not routed in")
    for collection in TOPICAL_COLLECTIONS:
        if collection.name not in message.collections and collection.name.lower() in block:
            fails.append(f"[{message.id}] collection '{collection.name}' should have dropped")
    if message.skill is not None and message.skill.lower() not in block:
        fails.append(f"[{message.id}] skill '{message.skill}' not surfaced in top recall")
    return fails


async def test_recall_routing(recall_eval: RecallEval) -> None:
    await recall_eval(
        case_id="recall-routing",
        seed=_seed_topicals,
        messages=MESSAGES,
        check=_check_routing,
    )
