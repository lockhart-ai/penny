"""Tests for the notifications tool — the chat-agent surface that replaced the
retired ``/mute`` + ``/unmute`` commands, then the retired ``notifications_mute``
/ ``notifications_unmute`` tool pair (#2812).

One tool over the existing ``MuteState`` row (``db.users``): the direction is an
explicit ``on`` bool argument rather than the tool's NAME.  It toggles the
primary recipient's mute state, returns an honest mirror-back, and is
idempotent — an already-in-that-state call is a successful ``mutated=False``
no-op, not an error.  With no user registered, it fails loudly rather than
silently no-op.

The NL-dispatch contract (a naive phrasing reaches the right tool, and a casual
mention does NOT) lives in ``tests/eval/chat/idle/test_notifications.py`` against
the live model; this pins the deterministic mechanism for ``make check``.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.llm.call_link import CallLink
from penny.tools.notifications import NotificationsTool

pytestmark = pytest.mark.bare_db


_RECIPIENT = "+15551234567"


def _register_user(db: Database) -> None:
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="New York",
        timezone="America/New_York",
        date_of_birth="1990-01-01",
    )


@pytest.mark.asyncio
async def test_mute_sets_state_and_mirrors_back(db):
    """``on=false`` writes the MuteState row and reports it as a real change."""
    _register_user(db)

    result = await NotificationsTool(db).execute(CallLink.new_top(), on=False)

    assert result.success is True
    assert result.mutated is True
    assert "muted" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is True
    # The result twin of to_action_str: a real mute narrates the action first-person
    # (the seam adds the tag), so the chat recap can fold it into Penny's reply (#1481).
    assert (
        NotificationsTool.to_result_narration({"on": False}, result) == "You paused notifications:"
    )


@pytest.mark.asyncio
async def test_mute_when_already_muted_is_a_noop(db):
    """Muting an already-muted user is a successful no-op (mutated=False), state kept."""
    _register_user(db)
    db.users.set_muted(_RECIPIENT)

    result = await NotificationsTool(db).execute(CallLink.new_top(), on=False)

    assert result.success is True
    assert result.mutated is False
    assert "already" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is True
    # No-op branch: narrate the honest "already paused" so the recap can't claim a change.
    assert (
        NotificationsTool.to_result_narration({"on": False}, result)
        == "Notifications were already paused:"
    )


@pytest.mark.asyncio
async def test_unmute_clears_state_and_mirrors_back(db):
    """``on=true`` on a muted user deletes the MuteState row and reports the change."""
    _register_user(db)
    db.users.set_muted(_RECIPIENT)

    result = await NotificationsTool(db).execute(CallLink.new_top(), on=True)

    assert result.success is True
    assert result.mutated is True
    assert "back on" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is False
    assert (
        NotificationsTool.to_result_narration({"on": True}, result)
        == "You turned notifications back on:"
    )


@pytest.mark.asyncio
async def test_unmute_when_not_muted_is_a_noop(db):
    """``on=true`` on a user who isn't muted is a successful no-op (mutated=False)."""
    _register_user(db)

    result = await NotificationsTool(db).execute(CallLink.new_top(), on=True)

    assert result.success is True
    assert result.mutated is False
    assert "already on" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is False
    assert (
        NotificationsTool.to_result_narration({"on": True}, result)
        == "Notifications were already on:"
    )


@pytest.mark.asyncio
async def test_tool_fails_loudly_with_no_registered_user(db):
    """With no primary user there is nothing to toggle — the tool fails visibly
    (no silent no-op) in both directions, naming the config condition rather than
    pretending success."""
    mute = await NotificationsTool(db).execute(CallLink.new_top(), on=False)
    unmute = await NotificationsTool(db).execute(CallLink.new_top(), on=True)

    for result in (mute, unmute):
        assert result.success is False
        assert result.mutated is False
        assert "no user" in result.message.lower()
    # Failure branch narrates honestly ("didn't work"), never a false success.
    assert (
        NotificationsTool.to_result_narration({"on": False}, mute)
        == "You tried to pause notifications but it didn't work:"
    )
    assert (
        NotificationsTool.to_result_narration({"on": True}, unmute)
        == "You tried to turn notifications back on but it didn't work:"
    )
