"""Survival contract for the email-tool narrations (part of epic #1478).

The narration seam (#1479) makes each email tool result lead with a first-person
line ("You searched your email for …"), and the recap instruction (#1483) tells
Penny to open her reply with what she did.  A unit test can prove the narration
STRING exists (``tests/tools/test_email_tools.py``) — but not the thing that
actually matters: that the summary **survives into Penny's reply to the user**.
These cases drive the real chat loop and score the REPLY, so a pass means the
email action genuinely reached the user.

The email tools are config-gated — present only when a mailbox is configured
(Fastmail / Zoho), which the eval environment does NOT have.  So, exactly like
``tests/eval/test_email_dispatch.py``, each case INJECTS a mocked mailbox via the
``prepare`` hook (``ChatAgent._email_tools_builder``); that makes the five email
tools present and their boundary calls no-ops returning canned data, so the
survival contract runs for real under ``make eval`` regardless of env config — no
real IMAP/JMAP and no skip guard needed.  The mailbox is mocked; only the model's
behaviour is live.

Scored STRUCTURALLY (the reply reflects the action — searched / found nothing),
never on exact wording, since the recap is composed fresh each turn.  Each scorer
prints a sample reply so the PR can report ``case | sample text | N score``.
Senders/topics are synthetic (the repo is public).
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from penny.database import Database
from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.penny import Penny
from penny.tests.eval.conftest import ChatEval, tool_was_called
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

pytestmark = pytest.mark.eval

_SEARCH = "search_emails"

# A synthetic hit — Sam / invoice, generic enough for a public repo.
_SUMMARY = EmailSummary(
    id="E1",
    subject="March invoice — payment due",
    from_addresses=[EmailAddress(name="Sam Okafor", email="sam@example.com")],
    received_at="2026-03-02T09:15:00Z",
    preview="Hi — attaching the March invoice; the balance is due by the 15th...",
)
_DETAIL = EmailDetail(
    id="E1",
    subject="March invoice — payment due",
    from_addresses=[EmailAddress(name="Sam Okafor", email="sam@example.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@example.com")],
    received_at="2026-03-02T09:15:00Z",
    text_body="The March invoice total is $420, due by the 15th. Reply if you have questions.",
)


def _install_mailbox(penny: Penny, *, summaries: list[EmailSummary]) -> None:
    """Wire a mocked mailbox so the five email tools register and return canned
    data.  ``summaries`` empty models the no-results branch."""
    client = AsyncMock()
    client.search_emails.return_value = summaries
    client.read_emails.return_value = [_DETAIL] if summaries else []
    client.list_emails.return_value = summaries
    client.get_folders.return_value = []
    client.draft_response.return_value = "draft-1"

    def build(user_query: str, today: str) -> list:
        return [
            SearchEmailsTool(client),
            ReadEmailsTool(client, penny.chat_agent._model_client, user_query, today),
            ListEmailsTool(client),
            ListFoldersTool(client),
            DraftEmailTool(client),
        ]

    penny.chat_agent._email_tools_builder = build


def _mailbox_with_hit(penny: Penny) -> None:
    _install_mailbox(penny, summaries=[_SUMMARY])


def _mailbox_empty(penny: Penny) -> None:
    _install_mailbox(penny, summaries=[])


# The reply must reflect that Penny went to the email — she recaps it as "I
# checked/searched your email", "looked through your inbox", "pulled up the
# email from Sam", etc.  Match the ACTION (no leading "I" — the model often
# drops it), never exact wording.
_CHECKED_EMAIL = re.compile(
    r"\b(search(ed|ing)?|check(ed|ing)?|look(ed|ing)?|pulled|dug|scan(ned|ning)?|"
    r"went through|read|open(ed)?|found)\b.{0,40}\b(e-?mail|inbox|message)|"
    r"\b(e-?mail|inbox)\b.{0,40}\b(from|about|search|check|look|found)",
    re.I,
)

# The honest no-results recap: nothing matched.  Penny must not fabricate an
# invoice email she didn't find.  The empty-search semantic space is wide — the
# model says "there weren't any", "aren't any that match", "came back empty", "no
# hits", "didn't find", "nothing that matched" — so match a negation/absence word
# near a mail/search noun, OR one of the fixed empty-result phrases.  Verified
# against 5 captured live replies (4 honest recaps match, a raw call-as-text bail
# does not) before gating, per the brittle-scorer lesson.
_NOTHING_FOUND = re.compile(
    r"\b(weren'?t|wasn'?t|aren'?t|isn'?t|no|not|nothing|didn'?t|couldn'?t|can'?t)\b"
    r".{0,40}?\b(any|match|matched|matching|found|hit|hits|e-?mail|emails|inbox|invoice|"
    r"there|message|keywords?|description|results?|anything|from sam)\b"
    r"|came back empty|\bempty\b|no (hits|matches|match|luck|sign|trace|results?)|"
    r"nothing (from|found|there|matching|that)",
    re.I,
)


def _score_searched(db: Database, before: set[str], reply: str) -> list[str]:
    fails: list[str] = []
    if not tool_was_called(db, _SEARCH):
        fails.append("search_emails was not called — no email lookup to recap")
    if not _CHECKED_EMAIL.search(reply):
        fails.append("reply did not recap checking email — summary did not survive into the reply")
    print(f"[SURVIVAL email-search] tool={int(tool_was_called(db, _SEARCH))} :: {reply[:200]!r}")
    return fails


def _score_no_results(db: Database, before: set[str], reply: str) -> list[str]:
    """No email matched → the reply must honestly say so, not invent one.  The
    model may or may not still call search first, so this scores only the honest
    recap in the reply."""
    fails: list[str] = []
    if not _NOTHING_FOUND.search(reply):
        fails.append("reply did not honestly recap the empty search — summary did not survive")
    print(f"[SURVIVAL email-none] :: {reply[:200]!r}")
    return fails


async def test_search_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="email-recap-search",
        message="did I get an email from Sam about the invoice?",
        prepare=_mailbox_with_hit,
        score=_score_searched,
        min_pass_rate=0.75,
    )


async def test_no_results_is_honest(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="email-recap-none",
        message="did I get an email from Sam about the invoice?",
        prepare=_mailbox_empty,
        score=_score_no_results,
        min_pass_rate=0.75,
    )
