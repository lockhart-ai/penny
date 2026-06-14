"""Tests for SendMessageTool — mute and cooldown gates.

The tool is the universal outbound delivery primitive.  Two gates
are enforced before the channel dispatch:

1. ``users.is_muted(recipient)`` — refuses with a string that
   instructs the model to call ``done``.
2. Flat-interval cooldown (``SEND_COOLDOWN_SECONDS``) bypassed when
   the user has replied since the agent's last send in
   ``penny-messages``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import Inclusion, RecallMode
from penny.tools.send_message import SendMessageTool, _appears_truncated

_PENNY_LOG = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_USER_LOG = PennyConstants.MEMORY_USER_MESSAGES_LOG

_RECIPIENT = "+15551234567"
_AGENT = "notify"


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    # The cooldown helper reads the system penny-messages and user-messages
    # logs; create them up-front so the tool's lookups don't ImportError.
    db.memories.create_log(_PENNY_LOG, "outbound", Inclusion.NEVER, RecallMode.RECENT)
    db.memories.create_log(_USER_LOG, "inbound", Inclusion.NEVER, RecallMode.RECENT)
    return db


def _make_config(cooldown_seconds: float = 600.0):
    """Stand-in config exposing the runtime knobs the tool reads."""
    runtime = type("Runtime", (), {"SEND_COOLDOWN_SECONDS": cooldown_seconds})()
    return type("Config", (), {"runtime": runtime})()


def _penny_sent(db, content: str) -> None:
    """Record a Penny outgoing message — the cooldown reads ``penny-messages``,
    a facade over ``messagelog`` (direction=outgoing)."""
    db.messages.log_message(PennyConstants.MessageDirection.OUTGOING, "penny", content)


def _user_said(db, content: str) -> None:
    """Record an incoming user message (``user-messages`` facade)."""
    db.messages.log_message(PennyConstants.MessageDirection.INCOMING, _RECIPIENT, content)


def _make_channel():
    """Mock channel — only ``send_response`` is exercised."""
    channel = type("Channel", (), {})()
    channel.send_response = AsyncMock(return_value=42)
    return channel


def _make_tool(db, channel=None, config=None):
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    return SendMessageTool(
        channel=channel or _make_channel(),
        agent_name=_AGENT,
        db=db,
        config=config or _make_config(),
    )


@pytest.mark.asyncio
async def test_send_message_dispatches_when_not_gated(tmp_path):
    """Happy path: no mute, no prior sends → dispatch + ack."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="hey there!")

    assert result == "Message sent."
    channel.send_response.assert_awaited_once()
    kwargs = channel.send_response.await_args.kwargs
    assert kwargs["recipient"] == _RECIPIENT
    assert kwargs["content"] == "hey there!"
    assert kwargs["author"] == _AGENT


@pytest.mark.asyncio
async def test_send_message_refuses_when_user_muted(tmp_path):
    """Muted recipient: tool refuses without dispatching."""
    db = _make_db(tmp_path)
    db.users.set_muted(_RECIPIENT)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="hey there!")

    assert "muted" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_refuses_when_content_is_a_refusal(tmp_path):
    """Refusal content ("I'm sorry, I can't...") is not dispatched as a reply."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(
        content="I'm sorry, I can't help with that as an AI language model."
    )

    assert "refusal" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_rejects_ellipsis_truncated_content(tmp_path):
    """Content ending mid-thought with '…' returns a failure-prefixed string
    so the agent loop marks the call as failed; the model retries with the
    complete body on its next step."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="here's the news, the model …")

    assert result.startswith("Error: ")
    assert "ended with an ellipsis" in result.lower()
    assert "complete" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_allows_conversational_mid_sentence_ellipsis(tmp_path):
    """A '…' followed by trailing text (e.g. 'Anyway… 🤓') is a complete
    message, not a truncation — the tool dispatches it normally."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="anyway… that's the gist 🤓")

    assert result == "Message sent."
    channel.send_response.assert_awaited_once()


def test_appears_truncated_detects_production_failure_tails():
    """Regression cases captured from production: model self-truncations."""
    truncated = [
        "lets you play 2-player co-op in style-themed ……",
        "still uses the original …",
        "precision engineering. Scientists …",
        "all-time-best-efficiency - …?",
        "Hello world...",
    ]
    for body in truncated:
        assert _appears_truncated(body), f"should detect truncation in: {body!r}"

    complete = [
        "anyway… that's the gist 🤓",
        "Hello world.",
        "What a great find!",
        "Source: https://example.com/page 🚀",
    ]
    for body in complete:
        assert not _appears_truncated(body), f"should NOT detect truncation in: {body!r}"


@pytest.mark.asyncio
async def test_send_message_refuses_when_cooldown_not_elapsed(tmp_path):
    """A recent send from the same agent (no user reply since) → cooldown gate fires."""
    db = _make_db(tmp_path)
    # Seed a prior send authored by this agent — count = 1 (no user reply since),
    # so the cooldown applies and the new send is refused.
    _penny_sent(db, "prior")
    channel = _make_channel()
    tool = _make_tool(db, channel=channel, config=_make_config(cooldown_seconds=3600.0))

    result = await tool.execute(content="hey again!")

    assert "cooldown" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_cooldown_skipped_when_user_replied_since_last_send(tmp_path):
    """User has spoken since the agent's last send → count=0 → cooldown bypassed.

    The cooldown is meant to throttle *autonomous* outreach
    (background agent spamming the user when ignored).  When a new user
    message has arrived between sends the conversation is alive, so the
    next send is conversational, not autonomous — no cooldown applies.
    """
    db = _make_db(tmp_path)
    # Prior send, then a user reply, then this new send attempt.
    _penny_sent(db, "prior")
    _user_said(db, "actually here's a follow-up")
    channel = _make_channel()
    tool = _make_tool(db, channel=channel, config=_make_config(cooldown_seconds=3600.0))

    result = await tool.execute(content="responding to your follow-up")

    assert result == "Message sent."
    channel.send_response.assert_awaited_once()


def test_user_reply_resets_cooldown_count(tmp_path):
    """A user-messages entry newer than prior sends resets the autonomous-send count to zero."""
    db = _make_db(tmp_path)
    # Old send → would otherwise be counted as autonomous outreach.
    _penny_sent(db, "old")
    # User replied since — entries are timestamped at write time, so this
    # user-messages entry's created_at is newer than the old send.
    _user_said(db, "hi back")
    tool = _make_tool(db)

    # The count walks newest-first and breaks once entries are older than the
    # latest user message — an immediate break gives count = 0.
    assert tool._count_sends_since_user_message() == 0


def test_latest_send_time_is_penny_last_outgoing(tmp_path):
    """The cooldown is per-Penny, not per-internal-agent: ``_latest_send_time``
    returns Penny's most recent outgoing message regardless of which agent
    produced it (a message has two authors — Penny or the user)."""
    db = _make_db(tmp_path)
    _penny_sent(db, "first")
    _penny_sent(db, "most recent")
    tool = _make_tool(db)

    latest = tool._latest_send_time()
    assert latest is not None


def test_latest_send_time_none_when_no_prior_sends(tmp_path):
    """Empty log → None, which the cooldown helper treats as 'no cooldown to wait out'."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)
    assert tool._latest_send_time() is None
