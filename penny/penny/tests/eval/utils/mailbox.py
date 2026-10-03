"""A canned mailbox: the email boundary a world stands on (#2209).

The email tools reach a mailbox through the ``EmailClient`` protocol, which is a system
boundary — Fastmail's JMAP or Zoho's REST in production.  A case that asks about an email
needs that boundary to answer from a fixed set of messages, the way ``CannedPage`` answers a
browse: deterministic, synthetic, and declared on the world so the report shows what the
sample was answering against.

**What a search returns is decided here, and it is deliberately LENIENT.**  A case about
finding the message an ask describes measures whether the reply comes from the right
message, not how precisely the model words a query, so a search returns EVERY message that
carries any of the query's words — ranked by how many of them it carries, newest first among
equals.  So a search on the sender returns that sender's other mail too, and a search on the
subject returns other mail about it: the neighbouring messages reach the model, and picking
the one the ask describes is the model's.  The rules, stated once:

* a word is a run of letters and digits, case-folded, of at least ``_MIN_WORD`` characters,
  and never one of ``_STOPWORDS`` — words that say nothing about WHICH message;
* ``text`` reads the subject, the sender and the body; ``from_addr`` reads the sender's name
  and address; ``subject`` reads the subject;
* a message is returned when every filter the call supplied matches it at least once, and
  the dates are applied as stated (``after`` / ``before``, ISO 8601);
* a filter with no words left in it constrains nothing.

An address contributes its local part and its host without the final label, so the shared
``.example`` top-level domain every synthetic sender carries does not match every message.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from penny.email.models import EmailAddress, EmailDetail, EmailSummary

# Who every canned message is addressed to — the eval's own user.
CANNED_RECIPIENT = EmailAddress(name="Test User", email="test@example.com")

# How much of a body a search result previews.  Fixed, so a result is the same on every
# sample; long enough that a preview reads like one.
PREVIEW_CHARS = 120

_WORD = re.compile(r"[0-9a-z]+")
_MIN_WORD = 3
_STOPWORDS = frozenset(
    {"the", "and", "for", "from", "about", "with", "your", "you", "this", "that", "any"}
    | {"email", "emails", "mail", "inbox", "message", "messages", "sent", "send"}
)


@dataclass(frozen=True)
class CannedEmail:
    """One message in a canned mailbox, synthetic throughout.

    ``received_at`` is ISO 8601 in UTC, which is how both production backends render it."""

    id: str
    subject: str
    sender: str
    address: str
    received_at: str
    body: str

    @property
    def detail(self) -> EmailDetail:
        """The message as ``read_emails`` returns it."""
        return EmailDetail(
            id=self.id,
            subject=self.subject,
            from_addresses=[EmailAddress(name=self.sender, email=self.address)],
            to_addresses=[CANNED_RECIPIENT],
            received_at=self.received_at,
            text_body=self.body,
        )

    @property
    def summary(self) -> EmailSummary:
        """The message as ``search_emails`` returns it."""
        return EmailSummary(
            id=self.id,
            subject=self.subject,
            from_addresses=[EmailAddress(name=self.sender, email=self.address)],
            received_at=self.received_at,
            preview=self.body[:PREVIEW_CHARS],
        )

    @property
    def text(self) -> str:
        """The message as a reader of the world sees it — the rendered detail."""
        return str(self.detail)

    def sender_words(self) -> set[str]:
        """What ``from_addr`` is compared against: the name, the address's local part, and its
        host without the final label."""
        local, _, domain = self.address.partition("@")
        host = domain.rsplit(".", 1)[0]
        return words(f"{self.sender} {local} {host}")

    def subject_words(self) -> set[str]:
        """What ``subject`` is compared against."""
        return words(self.subject)

    def all_words(self) -> set[str]:
        """What ``text`` is compared against: the subject, the sender and the body."""
        return self.subject_words() | self.sender_words() | words(self.body)


def words(text: str) -> set[str]:
    """The words a search compares — see the module docstring for what counts as one."""
    return {
        word
        for word in _WORD.findall(text.casefold())
        if len(word) >= _MIN_WORD and word not in _STOPWORDS
    }


# A search's filters: each one's words, and the part of a message they are compared against.
_Filters = tuple[tuple[set[str], Callable[[CannedEmail], set[str]]], ...]


class CannedMailbox:
    """The ``EmailClient`` protocol over a fixed set of messages.

    Reads only: a canned mailbox is never changed by a turn, so the surface it backs is
    Fastmail's — search and read."""

    def __init__(self, emails: Sequence[CannedEmail]) -> None:
        self._emails = tuple(emails)

    async def search_emails(
        self,
        text: str | None = None,
        from_addr: str | None = None,
        subject: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> list[EmailSummary]:
        """Every message the filters match, best match first — see the module docstring."""
        asked: _Filters = (
            (words(text or ""), CannedEmail.all_words),
            (words(from_addr or ""), CannedEmail.sender_words),
            (words(subject or ""), CannedEmail.subject_words),
        )
        scored = [
            (score, email)
            for email in self._in_window(after, before)
            if (score := _score(email, asked)) is not None
        ]
        # Best score first, and the newest message first among equal scores.
        ranked = sorted(scored, key=lambda pair: (pair[0], _received(pair[1])), reverse=True)
        return [email.summary for _, email in ranked]

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """The messages the ids name, in the order asked; an unknown id is simply not there."""
        by_id = {email.id: email for email in self._emails}
        return [by_id[one].detail for one in email_ids if one in by_id]

    async def close(self) -> None:
        """Nothing to close: there is no connection behind a canned mailbox."""

    def _in_window(self, after: str | None, before: str | None) -> list[CannedEmail]:
        start = _parse(after) if after else None
        end = _parse(before) if before else None
        return [
            email
            for email in self._emails
            if (start is None or _received(email) > start)
            and (end is None or _received(email) < end)
        ]


def _score(email: CannedEmail, asked: _Filters) -> int | None:
    """How many of the asked words ``email`` carries, or ``None`` when a filter misses it."""
    score = 0
    for wanted, field in asked:
        if not wanted:
            continue
        hits = len(wanted & field(email))
        if not hits:
            return None
        score += hits
    return score


def _received(email: CannedEmail) -> datetime:
    return _parse(email.received_at)


def _parse(stamp: str) -> datetime:
    """An ISO 8601 stamp as an aware datetime; a stamp with no zone is read as UTC, the zone
    every canned message is stamped in."""
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
