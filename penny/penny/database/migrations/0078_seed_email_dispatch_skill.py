"""Seed a skill that dispatches email questions to the email tools.

Type: data

The ``/email`` + ``/zoho`` commands retired (epic #1445) onto the chat agent's
tool surface — ``search_emails`` / ``read_emails`` (both mailboxes) plus
``list_emails`` / ``list_folders`` / ``draft_email`` (Zoho).  This skill is the
NL trigger that makes the dispatch reliable: a TRIGGER (intent + example
phrasings) plus numbered tool-call STEPS, in the one clean shape migration 0069
established.  It also names the no-fire boundary (grumbling about email volume is
not a lookup request) so the skill can't push the model toward over-firing.

The STEPS lead with the two tools every mailbox has (search → read → answer) and
mention folder-listing and drafting as "when available" (only Zoho exposes them),
so the recipe holds for both backends without pointing the model at a tool that
isn't on its surface.

Operate-the-system skill (``author='system'``, no source collection), so the
skills reconcile loop leaves it alone, exactly like the mute/schedule/scope
skills.  Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_EMAIL_LOOKUP = """TRIGGER
User asks about their email — whether a message arrived, what it said, or to draft a reply.
Example phrasings:
- "did I get an email from my accountant?"
- "check my email for the invoice from the plumber"
- "any emails about the conference this week?"
- "what did the school say in their last email?"
- "reply to that email and tell them I'll be there"
A casual grumble about email volume ("I get too much email these days") is NOT a lookup
request — do nothing.

STEPS
1. search_emails(text=<keywords>) — find candidate emails; narrow with from_addr=<sender>,
   subject=<subject text>, after=<ISO date>, or before=<ISO date>. Each result carries an id.
   (To browse a whole folder when that tool is available, list_emails(folder=<name>); call
   list_folders() first if you are unsure which folders exist.)
2. read_emails(email_ids=[<id>, <id>]) — read the full bodies of the promising hits; pass ALL
   relevant ids in ONE call, not one at a time.
3. If the user asked you to reply and draft_email is available, draft_email(to=[<address>],
   subject=<subject>, body=<text>) — this saves a draft to their Drafts folder for review; it
   NEVER sends.
4. Answer in plain text with the concrete details you found — specific dates, names, and
   amounts — and name the email (sender + subject) each fact came from. Ground every claim in
   an email you actually read; never guess at a detail you did not see."""


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory_entry "
        "(memory_name, key, content, author, key_embedding, content_embedding, created_at) "
        "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
        ("Look up email", _EMAIL_LOOKUP, now),
    )
    conn.commit()
