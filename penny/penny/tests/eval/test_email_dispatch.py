"""NL-dispatch contracts for the email tools (retired /email + /zoho, epic #1445).

These cases prove that a natural-language email question dispatches to
``search_emails`` (the entry point of the search → read → answer surface) with
faithful args, plus a no-fire guard that grumbling about email volume must NOT
trigger any email tool.  The mailbox client is mocked at the system boundary via
the ``prepare`` hook — the full five-tool Zoho surface is installed so the cases
also exercise the tool-count budget — so no real IMAP/JMAP is needed.  Scoring is
STRUCTURAL (the persisted tool call + its arguments), never wording.  Senders and
topics are synthetic (the repo is public).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.penny import Penny
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    last_tool_args,
    tool_not_called,
    tool_was_called,
)
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "nl-dispatch"

_SEARCH_EMAILS = "search_emails"

_EMAIL_TOOLS = {
    _SEARCH_EMAILS,
    "read_emails",
    "list_emails",
    "list_folders",
    "draft_email",
}

_SUMMARY = EmailSummary(
    id="E1",
    subject="Rooftop solar quote — next steps",
    from_addresses=[EmailAddress(name="Priya Nakamura", email="priya@example.com")],
    received_at="2026-02-10T14:30:00Z",
    preview="Thanks for the site visit — attached is the quote for the rooftop solar install...",
)

_DETAIL = EmailDetail(
    id="E1",
    subject="Rooftop solar quote — next steps",
    from_addresses=[EmailAddress(name="Priya Nakamura", email="priya@example.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@example.com")],
    received_at="2026-02-10T14:30:00Z",
    text_body="The rooftop solar quote is $18,400, valid for 30 days. Let me know to proceed.",
)


def _mock_email_client(penny: Penny) -> None:
    """Wire a mocked mailbox so the email tools register and their boundary
    calls are no-ops returning canned messages (no real IMAP/JMAP)."""
    client = AsyncMock()
    client.search_emails.return_value = [_SUMMARY]
    client.read_emails.return_value = [_DETAIL]
    client.list_emails.return_value = [_SUMMARY]
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


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_searched(token: str):
    """The utterance must dispatch to search_emails with args that faithfully
    carry a salient token of the sender/topic the user named."""

    def score(db, before, reply) -> list[Check]:
        anchor = f"{_SEARCH_EMAILS}("
        if not tool_was_called(db, _SEARCH_EMAILS):
            # No dispatch — the args-token check has nothing to inspect (not-applicable).
            return [
                Check("search_emails called", False, anchor=anchor),
                Check.na(f"search args carry '{token}'", anchor=anchor),
            ]
        args = last_tool_args(db, _SEARCH_EMAILS) or {}
        blob = " ".join(str(v) for v in args.values()).lower()
        has_token = token in blob
        return [
            Check("search_emails called", True, anchor=anchor),
            Check(
                f"search args carry '{token}'",
                has_token,
                anchor=anchor,
                rationale=None if has_token else f"args {args!r} dropped {token!r}",
            ),
        ]

    return score


def _score_no_email(db, before, reply) -> list[Check]:
    """Grumbling about email volume must NOT trigger any email tool."""
    return [
        Check(
            f"{name} not fired on a casual mention",
            tool_not_called(db, name),
            anchor=f"{name}(",
        )
        for name in sorted(_EMAIL_TOOLS)
    ]


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_email_from_sender_dispatches(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-email-from-sender",
        family=_FAMILY,
        message="did I get an email from Priya Nakamura about the lease?",
        prepare=_mock_email_client,
        score=_score_searched("nakamura"),
    )


async def test_check_email_for_topic_dispatches(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-email-for-topic",
        family=_FAMILY,
        message="check my email for the rooftop solar quote",
        prepare=_mock_email_client,
        score=_score_searched("solar"),
    )


async def test_casual_email_grumble_does_not_dispatch(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="tool-email-nofire",
        family=_FAMILY,
        message="honestly i get way too much email these days, my inbox is out of control",
        prepare=_mock_email_client,
        score=_score_no_email,
    )
