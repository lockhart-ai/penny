"""NotificationsTool — the chat-agent surface for silencing and resuming Penny's
proactive notifications.

One tool over the existing ``MuteState`` row (``db.users``), folding the retired
``notifications_mute`` / ``notifications_unmute`` pair (#2812) onto a single
boolean argument (``on``) rather than two argless tools whose NAME carried the
direction.  Neither tool had a single use in 444 measured prod calls, so the fold
costs nothing in dispatch and saves a whole tool definition's worth of tokens on
every chat turn.  The recipient is resolved the same way ``send_message`` does —
``db.users.get_primary_sender()`` — so the row it toggles is the exact key the
notifier's ``is_muted`` gate reads before delivering.

The result is an **honest mirror-back**: the message states the state that is now
true (muted / resumed), or — on a no-op — that it was already in that state, and
carries ``mutated`` accordingly (a real toggle vs. an already-in-that-state
no-op) so the collector work/no-work signals stay accurate for any future
background caller.  The chat agent composes its user reply from this result.

The recipient-None guard is shared: ``execute`` resolves the recipient and fails
loudly if there is no registered user, then dispatches on ``on`` to the matching
direction — no per-direction subclass repeats the guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from penny.llm.call_link import CallLink
from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database


class NotificationsArgs(ToolArgs):
    """The one argument ``notifications_set`` carries: the direction, as a bool
    rather than a tool name to pick between (#2812)."""

    on: bool


class NotificationsTool(Tool):
    """Turn Penny's proactive notifications on or off for the user.

    ``on=False`` silences thought discoveries, news, collection updates and
    other autonomous pings; ``on=True`` resumes them.  Replies to the user's
    own messages are never affected either way.
    """

    name = "notifications_set"
    args_model = NotificationsArgs

    _NO_RECIPIENT_RESPONSE = (
        "No user is registered yet, so there are no notifications to change. "
        "This is a config state, not something to retry."
    )

    _MUTED = (
        "Notifications are now muted — no proactive updates will go out until they "
        "are unmuted.  Replies to your messages still come through."
    )
    _ALREADY_MUTED = "Notifications were already muted — nothing changed."
    _UNMUTED = "Notifications are back on — proactive updates will resume."
    _ALREADY_ON = "Notifications weren't muted — they're already on, nothing changed."

    # Two shapes reach the OFF->ON direction, and only one of them is a request
    # (#1919).  The user can ASK for updates back, or they can GRANT PERMISSION for
    # them to resume — and a permission granted over a mute IS the ask, since
    # nothing else was waiting on it.  Naming only the request shape left the
    # permission shape reading as ordinary agreement (measured: 0.33 against 1.00
    # on the sibling that asks outright).  The STATE is named rather than any
    # phrasing, so a wording nobody enumerated lands too.
    description = (
        "Turn Penny's proactive notifications on or off — thought discoveries, "
        "news and collection updates, and other autonomous pings. Replies to the "
        "user's own messages are never affected either way.\n\n"
        "Pass `on=false` to mute: use this when the user asks to stop being "
        "messaged for a while, quiet down, take a break from updates, or "
        "otherwise pause proactive messages.\n\n"
        "Pass `on=true` to unmute: use this when the user asks for updates to "
        "resume, turns them back on, or lifts an earlier pause. Lifting the "
        "pause is often phrased as PERMISSION rather than as a request — the "
        "user saying it is fine to message them again, or that they are free "
        "now, is asking for exactly this, because proactive messages are the "
        "only thing that permission releases. A remark about how quiet or busy "
        "things have been is NOT that: it describes their day, grants nothing, "
        "and leaves the switch exactly where it is."
    )
    parameters = {
        "type": "object",
        "properties": {
            "on": {
                "type": "boolean",
                "description": (
                    "true to turn notifications on (unmute), false to turn them off (mute)"
                ),
            },
        },
        "required": ["on"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        args = NotificationsArgs(**kwargs)
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            return ToolResult(message=self._NO_RECIPIENT_RESPONSE, success=False)
        if args.on:
            return self._turn_on(recipient)
        return self._turn_off(recipient)

    def _turn_on(self, recipient: str) -> ToolResult:
        if not self._db.users.is_muted(recipient):
            return ToolResult(message=self._ALREADY_ON)
        self._db.users.set_unmuted(recipient)
        return ToolResult(message=self._UNMUTED, mutated=True)

    def _turn_off(self, recipient: str) -> ToolResult:
        if self._db.users.is_muted(recipient):
            return ToolResult(message=self._ALREADY_MUTED)
        self._db.users.set_muted(recipient)
        return ToolResult(message=self._MUTED, mutated=True)

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Resuming notifications" if arguments.get("on") else "Muting notifications"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        on = bool(arguments.get("on"))
        if not result.success:
            verb = "turn notifications back on" if on else "pause notifications"
            return f"You tried to {verb} but it didn't work:"
        if not result.mutated:
            return "Notifications were already on:" if on else "Notifications were already paused:"
        return "You turned notifications back on:" if on else "You paused notifications:"
