"""The one slot for the active foreground chat turn (#2985).

Penny is single-user, so at most one foreground chat turn runs at a time —
across every channel and device.  A message arriving while that turn is still
in its tool loop joins the turn instead of starting a second one: the joining
channel queues it on the active turn's inbox (``TurnInbox``,
penny/agents/models.py), the loop drains the inbox after each tool step, and
the claiming channel's dispatch task owns the whole series until the inbox is
empty.  This module holds the registry every channel shares.

The registry is created once at startup (``create_channel_manager``) and handed
to every channel as a constructor parameter — never ambient state.  All of its
methods are synchronous with no ``await`` inside, so each is atomic under
asyncio's single-threaded scheduling: a channel's claim-or-join and a turn's
end-check can never interleave halfway.

Background callers that run the agent loop with no live user to drain for
(collector cycles, the delivery turns) pass the ``NO_INBOX`` null object
(penny/agents/models.py) — explicit at every call site, never ``None`` that
quietly means "not linked".
"""

from __future__ import annotations

from penny.agents.models import TurnInbox
from penny.channels.base import QueuedTurnMessage


class TurnRegistry:
    """The one slot for the active foreground chat turn (#2985).

    Penny is single-user: one registry instance, created at startup and shared
    by every channel.  ``enter`` is the atomic claim-or-join — both outcomes in
    one synchronous method, so a channel either owns the turn it just started
    or is joining the one that already owns the slot; there is no window where
    two turns both believe they are alone.
    """

    def __init__(self) -> None:
        self._inbox: TurnInbox | None = None

    def enter(self, sender: str) -> tuple[TurnInbox, bool]:
        """Claim the slot for a new turn (returns the fresh inbox, ``False``) or
        join the active turn's inbox (returns it, ``True``).  Synchronous with no
        ``await`` — atomic under asyncio, and the caller puts its message into
        the returned inbox before its next ``await``, so a turn can never end
        with a just-joined message stranded in a dead inbox."""
        if self._inbox is not None:
            return self._inbox, True
        self._inbox = TurnInbox(sender)
        return self._inbox, False

    def take_next_or_end(self, inbox: TurnInbox) -> QueuedTurnMessage | None:
        """The series continuation point: the oldest still-queued message (keep
        the turn registered — the caller runs it as the next turn), or ``None``
        when nothing is queued and the slot is released.  Synchronous — the
        pending-check and the release are one atomic step, so a message joining
        in the same instant either lands before the check (the turn continues)
        or after the release (it starts its own turn)."""
        if inbox.pending():
            return inbox.take_next()
        if self._inbox is inbox:
            self._inbox = None
        return None

    def abandon(self, inbox: TurnInbox) -> list[QueuedTurnMessage]:
        """Release the slot unconditionally and return anything still queued —
        the backstop for a turn series that died before its clean end.  The
        caller answers each returned message itself; nothing is left stranded."""
        if self._inbox is inbox:
            self._inbox = None
        return inbox.drain_all()
