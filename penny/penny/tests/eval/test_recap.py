"""Self-narrating recap contract — the chat agent opens its reply with a brief,
in-voice recap of the actions it just took (searched, opened, read), then the
answer.

This is the core model-facing lever of the self-narrating-tools epic (#1478,
issue #1483): the tool layer already narrates every result in first person
(#1479), but a live-model probe showed narrated results ALONE never produced a
recap — the ``CONVERSATION_PROMPT`` recap instruction is what makes Penny weave
those actions back to the user.  So the contract scores that the reply
STRUCTURALLY reflects the action it took (mentions searching / opening / reading)
— never the exact wording, since the model is stochastic.

``VERSION_PAGES`` gives the turn a real action to recap (one browse for the
current version), so a reply that reflects "I searched / pulled up their page"
proves the lever fired.  Run this case with and without the instruction to prove
it is load-bearing (the probe measured 0 recaps without it).
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, tool_was_called
from penny.tests.eval.fixtures import VERSION_PAGES

pytestmark = pytest.mark.eval

# First-person action-reflection the recap produces — a reply that opens with
# "I searched / looked up / pulled up / checked / opened / read / found ..." is
# recapping the action it took.  Matched loosely (any one), never on exact
# wording: the model composes the recap fresh each turn.
_RECAP_PATTERNS = (
    r"\bi (searched|looked|checked|pulled|opened|read|found|browsed|dug|went|"
    r"visited|scanned|grabbed|hit|poked|glanced|dove|dived|took a look|had a look)\b",
    r"\b(searched for|looked (it |them )?up|pulled (it |them )?up|checked (out|their|the)|"
    r"looked (through|into)|after (searching|looking|checking|reading|browsing))\b",
)


def _reflects_action(reply: str) -> bool:
    lowered = reply.lower()
    return any(re.search(pattern, lowered) for pattern in _RECAP_PATTERNS)


def _score_recap(db: Database, before: set[str], reply: str) -> list[str]:
    fails = []
    if not reply.strip():
        fails.append("empty reply")
    if not tool_was_called(db, "browse"):
        fails.append("did not browse — no real action to recap this turn")
    if not _reflects_action(reply):
        fails.append(
            "reply did not recap the action taken (no first-person "
            "searched/opened/read reflection) — recap lever did not fire"
        )
    return fails


async def test_recap_reflects_actions(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="chat-recap",
        message="what's the latest stable version of the quillpad note-taking app?",
        browse=list(VERSION_PAGES),
        score=_score_recap,
    )
