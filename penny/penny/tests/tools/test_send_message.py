"""Tests for SendMessageTool — message-validity validation + delivery + enqueue.

Two layers, tested where each lives:

- **Message validity** is a validator on ``SendMessageArgs`` (the tool's
  ``args_model``), so a half-formed body — blank / punctuation-only, bare URL,
  bail-out phrase, unfinished fragment (``"Hi there! ......???"``), or
  ellipsis-truncated tail — fails validation BEFORE ``execute`` runs.  ``Tool.run``
  turns that into a ``success=False`` actionable error tool response.  This is the
  shared ``half_formed_send_reason`` rule the run-health classifier flags
  ``⚠ HALF-FORMED SEND`` on — one definition for both.
- **Delivery decisions** live in ``execute`` (they need runtime state or are
  correct no-op declines): a refusal body, no recipient, a muted user.

The tool enqueues into ``db.send_queue`` rather than delivering directly (see
``test_send_queue_drainer.py``); a clean send returns ``"Message sent."``
(``mutated=True``) — the successful handoff the collector prompts gate on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from penny.database import Database
from penny.tools.models import SendMessageArgs
from penny.tools.send_message import SendMessageTool

_RECIPIENT = "+15551234567"
_AGENT = "notify"


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    return db


def _make_tool(db) -> SendMessageTool:
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    return SendMessageTool(agent_name=_AGENT, db=db)


@pytest.mark.asyncio
async def test_send_message_enqueues_when_not_gated(tmp_path):
    """Happy path: no mute, valid content → message appended to the send queue."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)

    result = await tool.execute(content="hey there!")

    # Enqueue is the successful handoff — the literal the collector prompts gate on.
    assert result.message == "Message sent."
    assert result.mutated is True
    pending = db.send_queue.next_pending()
    assert pending is not None
    assert pending.content == "hey there!"
    assert pending.collection == _AGENT
    assert pending.sent_at is None


@pytest.mark.asyncio
async def test_send_message_refuses_when_user_muted(tmp_path):
    """Muted recipient: tool refuses without enqueueing."""
    db = _make_db(tmp_path)
    db.users.set_muted(_RECIPIENT)
    tool = _make_tool(db)

    result = await tool.execute(content="hey there!")

    # A muted send is a correct decline, not a failure — nothing was queued
    # (mutated False) but the call succeeded.
    assert result.success is True
    assert result.mutated is False
    assert "muted" in result.message.lower()
    assert "done" in result.message.lower()
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_send_message_refuses_when_content_is_a_refusal(tmp_path):
    """Refusal content ("I'm sorry, I can't...") is not enqueued as a reply."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)

    result = await tool.execute(
        content="I'm sorry, I can't help with that as an AI language model."
    )

    # Refusal content is a correct decline, not a failure.
    assert result.success is True
    assert result.mutated is False
    assert "refusal" in result.message.lower()
    assert "done" in result.message.lower()
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_run_refuses_half_formed_with_actionable_error(tmp_path):
    """Going through ``Tool.run`` (the executor's entry point), a half-formed body
    is refused by the ``args_model`` validator with a ``success=False`` actionable
    error tool response — names the field + the reason + how to fix — and
    ``execute`` never runs, so nothing is queued.  This is the unfinished-fragment
    shape the old truncation regex missed."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)

    result = await tool.run(content="Hi there! ......???")

    assert result.success is False
    assert result.mutated is False
    assert "unfinished fragment" in result.message.lower()  # the specific reason
    assert "complete message" in result.message.lower()  # how to fix
    assert "send_message" in result.message  # which tool to retry
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_send_message_allows_conversational_mid_sentence_ellipsis(tmp_path):
    """A '…' followed by trailing text (e.g. 'Anyway… 🤓') is a complete
    message, not a truncation — the tool enqueues it normally."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)

    result = await tool.execute(content="anyway… that's the gist 🤓")

    assert result.message == "Message sent."
    pending = db.send_queue.next_pending()
    assert pending is not None
    assert pending.content == "anyway… that's the gist 🤓"


@pytest.mark.asyncio
async def test_send_message_refuses_when_no_primary_user(tmp_path):
    """No registered user → no recipient → refuse without enqueueing."""
    db = _make_db(tmp_path)
    tool = SendMessageTool(agent_name=_AGENT, db=db)  # no save_info → no primary sender

    result = await tool.execute(content="hello?")

    assert result.mutated is False
    assert db.send_queue.next_pending() is None


def test_send_message_args_rejects_half_formed_bodies():
    """The ``SendMessageArgs`` validator is the single message-validity gate the
    ToolExecutor enforces before ``execute`` ever runs.  It rejects every
    half-formed shape — blank / punctuation-only, bare URL, bail-out phrase,
    unfinished fragment, and the ellipsis-truncated tails captured from production
    — and accepts complete messages (including a conversational mid-text ellipsis
    and a link with surrounding prose)."""
    half_formed = [
        "lets you play 2-player co-op in style-themed ……",
        "still uses the original …",
        "all-time-best-efficiency - …?",
        "Hello world...",
        "Hi there! ......???",
        "???!!! ...",
        "https://example.com/page",
        "I don't know",
    ]
    for body in half_formed:
        with pytest.raises(ValidationError):
            SendMessageArgs(content=body)

    complete = [
        "anyway… that's the gist 🤓",
        "What a great find!",
        "Source: https://example.com/page 🚀",
        "Heads up — a new title dropped, details inside.",
    ]
    for body in complete:
        SendMessageArgs(content=body)  # must not raise


def test_send_queue_store_round_trip(tmp_path):
    """Enqueue → next_pending (FIFO) → mark_sent removes it from the pending tail."""
    db = _make_db(tmp_path)
    first = db.send_queue.enqueue(content="one", collection="likes")
    db.send_queue.enqueue(content="two", collection="notified-thoughts")

    # FIFO: oldest pending first.
    pending = db.send_queue.next_pending()
    assert pending is not None and pending.id == first and pending.content == "one"

    # pending_items returns the whole pending tail, oldest-first (the eval harness
    # reads a cycle's enqueued sends here, since run_for never runs the drainer).
    assert [item.content for item in db.send_queue.pending_items()] == ["one", "two"]

    db.send_queue.mark_sent(first)
    # First is delivered; the next pending is now "two".
    nxt = db.send_queue.next_pending()
    assert nxt is not None and nxt.content == "two"
    assert [item.content for item in db.send_queue.pending_items()] == ["two"]
