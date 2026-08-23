"""NL-dispatch story: an email question reaches the mailbox, a grumble does not (#1445).

The retired ``/email`` + ``/zoho`` commands became a tool surface the model reaches from
plain language, so the story is one sentence with two directions:

  * a question ABOUT the mailbox ("did I get an email from X?", "check my email for Y")
    dispatches to ``search_emails`` — the entry point of the search → read → answer
    surface — carrying the sender or topic the user actually named; and
  * a REMARK about email ("i get way too much email these days") reaches nothing, and
    the reply does not claim she went and looked.

**Dispatch stands on the tool descriptions ALONE.**  Migration 0078 seeded a "Look up
email" skill that taught this routing; 0092 deleted its entries, 0097 deleted the
collection carrying them, and since 0108 nothing is pre-seeded at all.  These cases seed
no skills and no collections, so the world they measure is exactly a fresh deployment's:
the model reads ``search_emails``'s own description and decides.  The loud probe below
asserts that world out loud rather than trusting it — a mailbox that failed to install
would score every sample a dispatch failure it never had a chance to make.

**The conversation state machine fronts every driven turn** (#1706): it classifies before
the chat agent runs, and an email question lands in whatever state it lands in.  What is
scored here is the chat turn's DISPATCH, never the state it landed in.

Scoring is STRUCTURAL — the persisted tool call and its arguments, and the store read
before and after — with one narrow REPLY floor on the no-fire direction, where saying she
checked the inbox is a claim the record contradicts (the visible-degradation rule applied
to what she tells the user).  How well she answered is read at joint review against each
case's ``reference`` reply, which is DATA rather than a comment so the deterministic pin
in ``test_eval_harness.py`` can run it through this module's own vocabulary without a GPU.

The mailbox is mocked at the system boundary via the ``prepare`` hook — that hook is also
what REGISTERS the tools, since ``ChatAgent`` builds them only when a mailbox is
configured — so no real IMAP/JMAP is involved.  Senders and topics are synthetic (the repo
is public).

Report-only (``min_pass_rate=None``): a live-model dispatch rate is a number to read, and
the threshold is the code owner's to set once the numbers are read.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest

from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.penny import Penny
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    Scorer,
    collection_names,
    last_tool_args,
    new_collections,
    routing_clean,
    tool_call_sequence,
    tool_not_called,
    tool_was_called,
)
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module — shared with
# the sibling dispatch stories (generate_image, choose) so the report's families rollup
# reads chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

_SEARCH_EMAILS = "search_emails"

# The whole mailbox surface a configured deployment carries.  The no-fire direction asks
# that NONE of it fired, so the set is named once and read twice.
_EMAIL_TOOLS = (
    _SEARCH_EMAILS,
    "read_emails",
    "list_emails",
    "list_folders",
    "draft_email",
)

# ── The canned mailbox ────────────────────────────────────────────────────────
# One message, synthetic throughout.  It exists so a dispatched search RETURNS something
# and the turn can go on being an ordinary turn; nothing here is scored.

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


def _install_mailbox(penny: Penny) -> None:
    """Wire a mocked mailbox so the email tools REGISTER and their boundary calls are
    no-ops returning the canned message.

    Installing the builder is what puts the tools on the surface at all —
    ``ChatAgent._email_tools`` returns nothing without one — so this hook stands the
    world up rather than merely stubbing a network call."""
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


# ── What a reply CLAIMING she went and looked says ────────────────────────────
#
# Deliberately NARROW, and the only reply reading in the module: this is the one reply
# failure with a structural answer — no email tool ran, so saying the inbox was checked is
# a claim the record contradicts.  Every entry pairs a looking verb with the MAILBOX, in a
# form a denial or an offer cannot produce: "want me to check your inbox?" carries none of
# them, and neither does "i haven't looked".  A wide vocabulary would fail ordinary
# sympathy about a full inbox, which is the correct answer to this message.
#
# The tripwire is each case's own ``reference`` reply, run through this set by the pin in
# ``test_eval_harness.py``: a vocabulary that cannot pass the answer the case calls correct
# would score every sample a miss, on a GPU, an hour later.
_CLAIMS_SHE_CHECKED_THE_MAILBOX = (
    "i checked your inbox",
    "i checked your email",
    "i checked your mail",
    "i've checked your inbox",
    "i've checked your email",
    "i searched your inbox",
    "i searched your email",
    "i searched your mail",
    "i've searched your inbox",
    "i've searched your email",
    "i looked through your inbox",
    "i looked through your email",
    "i went through your inbox",
    "i went through your email",
    "i'll search your inbox",
    "i'll check your inbox for",
)


class _EmailCase(NamedTuple):
    """One agreed message, and what the turn it opens has to look like.

    ``asks_for`` is the salient token of the sender or topic the user named — what a
    faithful search must carry — or ``None`` for the no-fire direction, which is the
    declaration that this message asks the mailbox for nothing.

    ``reference`` is how the message would be answered WELL: a review target, read at
    joint review and never matched by the scorer.  It is DATA rather than a comment so
    the deterministic pin can run it through this module's reply vocabulary without a GPU
    — a scorer that cannot pass the answer the case itself calls correct is a broken
    scorer, and that is cheaper to find here than on the queue."""

    case_id: str
    message: str
    asks_for: str | None
    reference: str


_FROM_SENDER = _EmailCase(
    case_id="tool-email-from-sender",
    message="did I get an email from Priya Nakamura about the lease?",
    asks_for="nakamura",
    reference=(
        "nothing from priya nakamura about a lease — the only thing of hers in there is a "
        "rooftop solar quote from february."
    ),
)

_FOR_TOPIC = _EmailCase(
    case_id="tool-email-for-topic",
    message="check my email for the rooftop solar quote",
    asks_for="solar",
    reference=(
        "found it — priya nakamura sent the rooftop solar quote in february: $18,400, good "
        "for 30 days."
    ),
)

_GRUMBLE = _EmailCase(
    case_id="tool-email-nofire",
    message="honestly i get way too much email these days, my inbox is out of control",
    asks_for=None,
    reference="ugh, yeah. want me to dig through it and see what's actually worth reading?",
)

EMAIL_CASES = (_FROM_SENDER, _FOR_TOPIC, _GRUMBLE)


# ── The loud probe: the mailbox really is on the surface ──────────────────────


def assert_mailbox_world(penny: Penny, case: _EmailCase) -> None:
    """Everything this case's world is responsible for, asserted out loud.

    Two claims, and a drift in either would be read as the model failing: the five email
    tools are REGISTERED (a case whose whole subject is dispatch, run against a surface
    that carries no mailbox, reports a dispatch miss the model was never offered), and the
    registry is EMPTY (nothing pre-seeded since migration 0108 — which is what makes
    "nothing was created" a total reading of what this turn touched)."""
    surface = {tool.name for tool in penny.chat_agent.get_tools()}
    missing = sorted(set(_EMAIL_TOOLS) - surface)
    assert not missing, f"{case.case_id}: the mailbox surface must carry {missing}"
    held = sorted(collection_names(penny.db))
    assert not held, f"{case.case_id}: the world must hold no collection, it holds {held}"


def _probe_mailbox_world(case: _EmailCase) -> Preparer:
    """Install the mocked mailbox, then assert the world it stood up."""

    def prepare(penny: Penny) -> None:
        _install_mailbox(penny)
        assert_mailbox_world(penny, case)

    return prepare


# ── Checks ────────────────────────────────────────────────────────────────────


def _searched_check(db) -> Check:
    """The headline: the question reached the mailbox at all.

    Anchored to the call itself, so the verdict sits on the row that made it and a miss
    falls to the run-close table where a missing action belongs."""
    fired = tool_was_called(db, _SEARCH_EMAILS)
    return Check(
        "calls: the question reached search_emails",
        fired,
        anchor=f"{_SEARCH_EMAILS}(",
        rationale=None if fired else f"the turn fired {tool_call_sequence(db) or 'nothing'}",
        kind="spine",
    )


def _faithful_args_check(db, case: _EmailCase) -> Check:
    """The search asks for what the USER named — the half that makes dispatch worth
    anything, since a search for the wrong thing is a call that fired and answered nobody.

    Read across every argument the call carried rather than one named field: the surface
    takes sender, subject and free text, and which one a faithful search uses is the
    model's to pick.  N/A when nothing was searched — there is no argument to read, and
    counting it a miss would report one failure twice."""
    label = f"calls: the search asks for {case.asks_for!r}"
    args = last_tool_args(db, _SEARCH_EMAILS)
    if args is None:
        return Check.na(label, anchor=f"{_SEARCH_EMAILS}(", kind="spine")
    asked = " ".join(str(value) for value in args.values()).lower()
    carried = (case.asks_for or "") in asked
    return Check(
        label,
        carried,
        anchor=f"{_SEARCH_EMAILS}(",
        rationale=None if carried else f"it asked for {args!r}",
        kind="spine",
    )


def _no_mailbox_call_check(db) -> Check:
    """Nothing on the mailbox surface fired — the whole no-fire direction in one check,
    because "she went and searched" and "she drafted a reply" are the same failure: a
    remark was read as an instruction.

    The rationale NAMES which tool fired, so a miss reads as what happened rather than as
    a bare red."""
    fired = [name for name in _EMAIL_TOOLS if not tool_not_called(db, name)]
    return Check(
        "calls: no email tool fired on a remark about email",
        not fired,
        rationale=f"fired {fired}" if fired else None,
        kind="spine",
    )


def _store_untouched_check(db, before: set[str]) -> Check:
    """Nothing was created.  An email question is a READ, and this world holds no
    collection at all — so a collection appearing is the whole "nothing else was touched"
    claim rather than a sample of it: with an empty registry there is nowhere to write
    that does not first show up here."""
    created = sorted(row.name for row in new_collections(db, before))
    return Check(
        "state: nothing was created",
        not created,
        rationale=f"created {created}" if created else None,
        kind="state",
    )


def _claims_no_search_check(reply: str) -> Check:
    """The reply does not say she checked the mailbox when nothing checked it.

    A FLOOR, deliberately narrow: everything else about the answer — whether the sympathy
    landed, whether the offer was the right one — is read at joint review against the
    case's reference reply, and an answer that just commiserates is not a miss."""
    claimed = [phrase for phrase in _CLAIMS_SHE_CHECKED_THE_MAILBOX if phrase in reply.lower()]
    return Check(
        "reply: it claims no search happened",
        not claimed,
        anchor=REPLY_ANCHOR,
        rationale=f"said {claimed}" if claimed else None,
        kind="reply",
    )


def _dispatch_advisories(db, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — the calls it made and the
    answer it gave — so a report shows the turn whichever way it went and the wording is
    read where wording is read: at joint review."""
    return [
        Check(f"fired: {tool_call_sequence(db)}", True, kind="proc", scored=False),
        Check(f"answered: {reply!r}", True, kind="reply", scored=False),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_email_ask(db, before: set[str], reply: str, *, case: _EmailCase) -> list[Check]:
    """The question reached the mailbox, carrying what the user named, and the turn
    touched nothing else.

    No reply check on this direction: what the answer should say depends on what the
    mailbox returned, which is the surface's business rather than dispatch's — so the
    answer is an advisory, read at joint review against the case's reference."""
    return [
        _searched_check(db),
        _faithful_args_check(db, case),
        _store_untouched_check(db, before),
        *_dispatch_advisories(db, reply),
    ]


def _score_email_grumble(db, before: set[str], reply: str) -> list[Check]:
    """The remark reached nothing, nothing was created, and the reply does not claim a
    search that never happened.

    Every check is a negative, because the failure this direction exists to catch is
    firing anything at all."""
    return [
        _no_mailbox_call_check(db),
        _store_untouched_check(db, before),
        _claims_no_search_check(reply),
        *_dispatch_advisories(db, reply),
    ]


async def _run_email_case(chat_eval: ChatEval, case: _EmailCase) -> None:
    """Drive one email case: the mocked mailbox installed and probed, the scorer bound to
    the case's own token.  Report-only — the threshold is the code owner's to set once the
    numbers are read."""
    score: Scorer = (
        _score_email_grumble if case.asks_for is None else partial(_score_email_ask, case=case)
    )
    await chat_eval(
        case_id=case.case_id,
        family=_FAMILY,
        message=case.message,
        prepare=_probe_mailbox_world(case),
        score=score,
        min_pass_rate=None,
    )


async def test_email_from_sender_dispatches(chat_eval: ChatEval) -> None:
    """ "did I get an email from X about Y?" — the sender-anchored ask, and the one where a
    search that drops the name would still look like it worked."""
    await _run_email_case(chat_eval, _FROM_SENDER)


async def test_check_email_for_topic_dispatches(chat_eval: ChatEval) -> None:
    """ "check my email for X" — the topic-anchored ask, an explicit instruction to go and
    look, with nothing but a subject to search on."""
    await _run_email_case(chat_eval, _FOR_TOPIC)


async def test_casual_email_grumble_does_not_dispatch(chat_eval: ChatEval) -> None:
    """A remark about the VOLUME of email asks the mailbox for nothing — the over-firing
    direction, where the topic is email and the request is not."""
    await _run_email_case(chat_eval, _GRUMBLE)
