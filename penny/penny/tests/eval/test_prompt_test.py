"""Can the model drive the dry-run tool to self-correct a prompt?

The question is behavioural: given the real `prompt_test` tool (which replays a
candidate extraction_prompt on a throwaway dry-run collector and reports what the
cycle WOULD do), can the quality collector run the full loop —

    spot the drift → draft a fixed prompt → prompt_test it → read the result →
    decide it's fixed → apply it with collection_update

— through the REAL collector against the REAL model?

This drives the production PromptTestTool (a stubbed version proved the model
can operate the protocol; this checks it against a faithful simulation).

Scenario: a silent-intent collection whose prompt still sends messages.  A
correct fix removes the send_message step; the dry run should then report 0
messages, and the model should apply it.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import Inclusion, LogEntryInput, RecallMode
from penny.tests.eval.conftest import CollectorScorer, tool_was_called
from penny.tools.prompt_test import PromptTestTool

pytestmark = pytest.mark.eval

_QUALITY = "quality"
_SUSPECT = "espresso-gear"

# The drifted prompt: intent says "never ping me", but it still sends messages.
_DRIFTED_PROMPT = (
    "Collect espresso equipment worth considering.\n"
    "1. browse(...) for new espresso gear; read actual pages.\n"
    '2. collection_write("espresso-gear", entries=[...]).\n'
    "3. If the write succeeded, send_message: one-sentence 'found a new item' + URL.\n"
    "4. done()."
)


# ── Quality prompt that drives the dry-run loop ─────────────────────────────

_CANDIDATE_PROMPT = (
    "You are Penny's quality agent.  Each cycle you review your own recent "
    "behaviour and fix the ONE collection that has drifted most from what the "
    "user asked of it.\n\n"
    "A collection's `intent` is the user's own words for what it should do — the "
    "spec.  Its `extraction_prompt` is how it tries to do it.  When the prompt "
    "produces behaviour that contradicts the intent, rewrite the prompt to match. "
    "The intent is fixed; you can never change it.\n\n"
    "Sequence:\n"
    '1. log_read_recent("collector-runs", window_seconds=86400) and '
    'log_read_recent("penny-messages", window_seconds=86400) — see what your '
    "collectors actually did and sent.\n"
    "2. Find ONE collection whose behaviour contradicts its intent (e.g. a "
    "collection whose intent says never to message you, but which sent a message).\n"
    "3. collection_metadata(<that collection>) — read its intent and current "
    "extraction_prompt.\n"
    "4. Draft a corrected extraction_prompt that removes the offending step.\n"
    "5. prompt_test(collection=<that collection>, extraction_prompt=<your draft>) "
    "— dry-run it.  Read the result: if the cycle would still violate the intent "
    "(e.g. still sends a message a silent collection shouldn't), revise the draft "
    "and prompt_test again.  Only proceed once the dry run is clean.\n"
    "6. collection_update(name=<that collection>, extraction_prompt=<the "
    "dry-run-confirmed draft>) — apply the fix.\n"
    "7. send_message the user one sentence on what you changed, then done()."
)


def _seed(db: Database) -> None:
    db.memories.create_collection(
        _QUALITY,
        "Reviews Penny's own runs and messages and corrects drifted collection prompts",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_CANDIDATE_PROMPT,
    )
    db.memories.create_collection(
        _SUSPECT,
        "A quiet running list of espresso equipment worth considering.",
        Inclusion.RELEVANT,
        RecallMode.RECENT,
        extraction_prompt=_DRIFTED_PROMPT,
        intent="Keep a quiet running list of espresso equipment worth considering — "
        "never ping me about it, I'll check the list myself.",
    )
    db.memories.append(
        PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
        [
            LogEntryInput(
                content="[espresso-gear] wrote 2 entries and sent an update about a grinder."
            )
        ],
        author="collector",
    )
    db.memories.append(
        PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
        [LogEntryInput(content="Found a new espresso grinder: the Niche Zero clone, $300.")],
        author="penny",
    )


def _snapshot(db: Database) -> str:
    memory = db.memories.get(_SUSPECT)
    return (memory.extraction_prompt or "") if memory else ""


def _score_loop(db: Database, before: object, sent: list[str]) -> list[str]:
    """The loop is validated when the model dry-ran the fix AND applied a clean one."""
    original = before
    fails = []
    if not tool_was_called(db, "prompt_test"):
        fails.append("never called prompt_test to check the fix")
    memory = db.memories.get(_SUSPECT)
    new_prompt = (memory.extraction_prompt or "") if memory else ""
    if new_prompt == original:
        fails.append("did not apply a fix (extraction_prompt unchanged)")
    elif "send_message" in new_prompt:
        fails.append("applied a fix that still sends messages (dry run not heeded)")
    return fails


def _extra_tools(collector) -> list:
    return [PromptTestTool(collector)]


async def test_dry_run_self_correction_loop(collector_eval) -> None:
    score: CollectorScorer = _score_loop
    await collector_eval(
        case_id="prompt-test-loop",
        collection=_QUALITY,
        seed=_seed,
        snapshot=_snapshot,
        extra_tools=_extra_tools,
        score=score,
        # Exploratory: the multi-hop tool loop is hard for a 20B model, and the
        # finding is the pass-rate itself.  A low bar keeps the suite honest
        # about that while we learn whether the model can drive it at all.
        min_pass_rate=0.5,
    )
