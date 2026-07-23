"""Novel-pattern contracts — does the substrate GENERALIZE past the seeded skills?

Requests with no matching skill.  The bar is deliberately lenient: a coherent
action (improvise a collection, browse, or give a substantive reply — including
gracefully declining a tool-gap) passes.  The only failure is doing nothing at
all (empty / stuck).  Driven through the real chat flow against the real model.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, Check, new_collections

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "novel-patterns"


def _score_coherent(db: Database, before: set[str], reply: str) -> list[Check]:
    """A coherent action is enough: a collection was created, or there's a
    substantive text reply.  Failure is only no-action-and-no-reply (stuck)."""
    coherent = bool(new_collections(db, before)) or len(reply.strip()) >= 20
    return [Check("coherent action or substantive reply", coherent, kind="state")]


async def test_url_watcher(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="novel-url-watcher",
        family=_FAMILY,
        message="watch this page https://example.com/news for changes weekly "
        "and tell me when it updates",
        score=_score_coherent,
    )


async def test_recurring_reminder(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="novel-recurring-reminder",
        family=_FAMILY,
        message="remind me to water my plants every sunday morning",
        score=_score_coherent,
    )


async def test_chat_pattern_extraction(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="novel-chat-pattern-extraction",
        family=_FAMILY,
        message="every time i mention a book in our chats, quietly save the title to a list",
        score=_score_coherent,
    )


async def test_tool_gap_email(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="novel-tool-gap-email",
        family=_FAMILY,
        message="summarize every email i get from my landlord and send me the summary",
        score=_score_coherent,
    )
