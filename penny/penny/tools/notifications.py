"""Notification mute/unmute tools — the chat-agent surface for silencing and
resuming Penny's proactive notifications.

Two thin tools over the existing ``MuteState`` row (``db.users``): the tool NAME
carries the direction (``notifications_mute`` / ``notifications_unmute``), so the
model never has to pick a direction argument — the least-ambiguous shape for
dispatch.  Both take no arguments (``NoArgs``) and resolve the recipient the same
way ``send_message`` does — ``db.users.get_primary_sender()`` — so the row they
toggle is the exact key the notifier's ``is_muted`` gate reads before delivering.

The result is an **honest mirror-back**: the message states the state that is now
true (muted / resumed), or — on a no-op — that it was already in that state, and
carries ``mutated`` accordingly (a real toggle vs. an already-in-that-state
no-op) so the collector work/no-work signals stay accurate for any future
background caller.  The chat agent composes its user reply from this result.

The recipient-None guard is shared: ``execute`` resolves the recipient and fails
loudly if there is no registered user, then delegates the direction-specific
toggle to ``_apply`` — a template method, so neither subclass repeats the guard.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import NoArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database


class _NotificationsTool(Tool):
    """Shared base: resolve the primary recipient, then toggle its mute state.

    The recipient is always the single Penny user (``get_primary_sender``) — the
    same identity the notifier checks with ``is_muted`` — so a chat-driven mute
    silences exactly the notifications the background pipeline would otherwise
    deliver.  With no user registered there is nothing to mute; ``execute`` turns
    that into an actionable failure once, rather than each subclass repeating it.
    """

    args_model = NoArgs

    _NO_RECIPIENT_RESPONSE = (
        "No user is registered yet, so there are no notifications to change. "
        "This is a config state, not something to retry."
    )

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            return ToolResult(message=self._NO_RECIPIENT_RESPONSE, success=False)
        return self._apply(recipient)

    @abstractmethod
    def _apply(self, recipient: str) -> ToolResult:
        """Toggle ``recipient``'s mute state in this tool's direction."""


class NotificationsMuteTool(_NotificationsTool):
    """Silence Penny's proactive notifications for the user."""

    name = "notifications_mute"
    description = (
        "Mute Penny's proactive notifications — thought discoveries, news and "
        "collection updates, and other autonomous pings.  Use this when the user "
        "asks to stop being messaged for a while, quiet down, take a break from "
        "updates, or otherwise pause proactive messages.  Replies to the user's "
        "own messages are never affected.  Takes no arguments."
    )

    _MUTED = (
        "Notifications are now muted — no proactive updates will go out until they "
        "are unmuted.  Replies to your messages still come through."
    )
    _ALREADY_MUTED = "Notifications were already muted — nothing changed."

    def _apply(self, recipient: str) -> ToolResult:
        if self._db.users.is_muted(recipient):
            return ToolResult(message=self._ALREADY_MUTED)
        self._db.users.set_muted(recipient)
        return ToolResult(message=self._MUTED, mutated=True)

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Muting notifications"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to pause notifications but it didn't work:"
        if not result.mutated:
            return "Notifications were already paused:"
        return "You paused notifications:"


class NotificationsUnmuteTool(_NotificationsTool):
    """Resume Penny's proactive notifications for the user."""

    name = "notifications_unmute"
    description = (
        "Unmute Penny's proactive notifications after they were muted — resume "
        "thought discoveries, news and collection updates, and other autonomous "
        "pings.  Use this when the user asks to be messaged again, turn updates "
        "back on, or resume notifications.  Takes no arguments."
    )

    _UNMUTED = "Notifications are back on — proactive updates will resume."
    _ALREADY_ON = "Notifications weren't muted — they're already on, nothing changed."

    def _apply(self, recipient: str) -> ToolResult:
        if not self._db.users.is_muted(recipient):
            return ToolResult(message=self._ALREADY_ON)
        self._db.users.set_unmuted(recipient)
        return ToolResult(message=self._UNMUTED, mutated=True)

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Resuming notifications"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to turn notifications back on but it didn't work:"
        if not result.mutated:
            return "Notifications were already on:"
        return "You turned notifications back on:"
