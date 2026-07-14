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
    error tool response — names the field + the SPECIFIC defect + the next move —
    and ``execute`` never runs, so nothing is queued.  ``"Hi there! ......???"``
    trails off into a degenerate tail; the message names that and how to fix it
    (finish the sentence), NOT the old misdirecting generic "send the COMPLETE
    message" (which read as wrong when the send was already substantive)."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)

    result = await tool.run(content="Hi there! ......???")

    assert result.success is False
    assert result.mutated is False
    assert "degenerate" in result.message.lower()  # the specific defect
    assert "finish the sentence" in result.message.lower()  # the next move
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
    """No registered user → no recipient → decline without enqueueing, naming the
    real condition (an environment/config state) and binding the terminal move.
    This decline must be DISTINCT from the refusal-content decline: telling the
    model its content read as a refusal would misdirect it into pointlessly
    rewriting a fine message when the actual fault is that no recipient exists."""
    db = _make_db(tmp_path)
    tool = SendMessageTool(agent_name=_AGENT, db=db)  # no save_info → no primary sender

    result = await tool.execute(content="hello?")

    assert result.mutated is False
    assert db.send_queue.next_pending() is None
    # Names the real condition and binds the correct terminal move — this cycle
    # cannot deliver, so done(success=false), not a content rewrite.
    assert "recipient" in result.message.lower()
    assert "done" in result.message.lower()
    assert "success=false" in result.message.lower()
    # The two declines are DISTINCT strings — the no-recipient response must not
    # reuse the refusal-content message (the bug this fixes).
    assert result.message != SendMessageTool._REFUSAL_RESPONSE
    assert "refusal" not in result.message.lower()


def test_send_message_args_rejects_half_formed_bodies():
    """The ``SendMessageArgs`` validator is the single message-validity gate the
    ToolExecutor enforces before ``execute`` ever runs.  It judges the message AS A
    WHOLE: it rejects every whole-message half-formed shape — blank /
    punctuation-only, bare URL, bail-out phrase, an unfinished ellipsis+?/! TAIL,
    and the ellipsis-truncated tails captured from production — and accepts complete
    messages, INCLUDING a substantive message that merely EMBEDS a degenerate
    fragment mid-text (a `quality` suggestion quoting the bad send it observed)."""
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
        # Degenerate runs EMBEDDED mid-message (real words follow) are no longer
        # refused by the send gate: a substantive, deliberate message that quotes a
        # fragment is complete.  Catching an in-flight collapse in the model's OWN
        # output is the agent-loop reroll guard's job (is_degenerate_run), not the
        # send gate's.  The canonical case is a `quality` suggestion that reports —
        # and quotes — the half-formed send it observed.
        "New restaurant … … … … openings this week",
        "Big AI update ……? for you today",
        "Delivered a find about Boss ..??.. gear",
        "HALF-FORMED SEND on board-game-news. Observed: the collector sent "
        '"Hi there! ......???" before the real note. Proposed fix: compose the '
        "complete message first, then send once. New prompt: 1. browse for the "
        "game. 2. write the entry. 3. compose the full sentence. 4. send it. 5. done.",
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
