"""Tests for the notification mute/unmute tools — the chat-agent surface that
replaced the retired ``/mute`` + ``/unmute`` commands.

Two thin tools over the existing ``MuteState`` row (``db.users``): the tool NAME
carries the direction, so the model never picks a direction argument.  Each
toggles the primary recipient's mute state, returns an honest mirror-back, and is
idempotent — an already-in-that-state call is a successful ``mutated=False``
no-op, not an error.  With no user registered, both fail loudly rather than
silently no-op.

The NL-dispatch contract (a naive phrasing reaches the right tool, and a casual
mention does NOT) lives in ``tests/eval/test_notifications.py`` against the live
model; this pins the deterministic mechanism for ``make check``.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.schema_template import schema_only_db
from penny.tools.notifications import NotificationsMuteTool, NotificationsUnmuteTool

_RECIPIENT = "+15551234567"


def _make_db(tmp_path) -> Database:
    db = schema_only_db(str(tmp_path / "test.db"))
    return db


def _register_user(db: Database) -> None:
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )


@pytest.mark.asyncio
async def test_mute_sets_state_and_mirrors_back(tmp_path):
    """A first mute writes the MuteState row and reports it as a real change."""
    db = _make_db(tmp_path)
    _register_user(db)

    result = await NotificationsMuteTool(db).execute()

    assert result.success is True
    assert result.mutated is True
    assert "muted" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is True
    # The result twin of to_action_str: a real mute narrates the action first-person
    # (the seam adds the tag), so the chat recap can fold it into Penny's reply (#1481).
    assert NotificationsMuteTool.to_result_narration({}, result) == "You paused notifications:"


@pytest.mark.asyncio
async def test_mute_when_already_muted_is_a_noop(tmp_path):
    """Muting an already-muted user is a successful no-op (mutated=False), state kept."""
    db = _make_db(tmp_path)
    _register_user(db)
    db.users.set_muted(_RECIPIENT)

    result = await NotificationsMuteTool(db).execute()

    assert result.success is True
    assert result.mutated is False
    assert "already" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is True
    # No-op branch: narrate the honest "already paused" so the recap can't claim a change.
    assert (
        NotificationsMuteTool.to_result_narration({}, result)
        == "Notifications were already paused:"
    )


@pytest.mark.asyncio
async def test_unmute_clears_state_and_mirrors_back(tmp_path):
    """An unmute on a muted user deletes the MuteState row and reports the change."""
    db = _make_db(tmp_path)
    _register_user(db)
    db.users.set_muted(_RECIPIENT)

    result = await NotificationsUnmuteTool(db).execute()

    assert result.success is True
    assert result.mutated is True
    assert "back on" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is False
    assert (
        NotificationsUnmuteTool.to_result_narration({}, result)
        == "You turned notifications back on:"
    )


@pytest.mark.asyncio
async def test_unmute_when_not_muted_is_a_noop(tmp_path):
    """Unmuting a user who isn't muted is a successful no-op (mutated=False)."""
    db = _make_db(tmp_path)
    _register_user(db)

    result = await NotificationsUnmuteTool(db).execute()

    assert result.success is True
    assert result.mutated is False
    assert "already on" in result.message.lower()
    assert db.users.is_muted(_RECIPIENT) is False
    assert (
        NotificationsUnmuteTool.to_result_narration({}, result) == "Notifications were already on:"
    )


@pytest.mark.asyncio
async def test_tools_fail_loudly_with_no_registered_user(tmp_path):
    """With no primary user there is nothing to toggle — both tools fail visibly
    (no silent no-op), naming the config condition rather than pretending success."""
    db = _make_db(tmp_path)  # no save_info → no primary sender

    mute = await NotificationsMuteTool(db).execute()
    unmute = await NotificationsUnmuteTool(db).execute()

    for result in (mute, unmute):
        assert result.success is False
        assert result.mutated is False
        assert "no user" in result.message.lower()
    # Failure branch narrates honestly ("didn't work"), never a false success.
    assert (
        NotificationsMuteTool.to_result_narration({}, mute)
        == "You tried to pause notifications but it didn't work:"
    )
    assert (
        NotificationsUnmuteTool.to_result_narration({}, unmute)
        == "You tried to turn notifications back on but it didn't work:"
    )
