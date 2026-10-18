"""CallLink: the run and tool-call linkage threaded through every model call (#2458).

The code owner's ruling: no ambient state (no ``contextvars``, no ``get_current_X``).
The linkage is an explicit, REQUIRED parameter everywhere a model call is made, or a
tool call is dispatched — so a kitten adding, moving, or removing a micro-context, a
tool, or a run starter has to pass it in, and the type checker enforces that. No
``X or <mint>`` fallback anywhere a link already exists; the one place minting still
belongs is :meth:`CallLink.new_top`, for a one-shot caller with no run id of its own.

Four fields, two kinds:

- ``run_id`` / ``parent_run_id`` / ``parent_tool_call_id`` are this call's LOGGED
  identity — exactly what a ``promptlog`` row stores (:class:`~penny.llm.client.
  LlmClient` reads only these three, never ``tool_call_id``).
- ``tool_call_id`` is POSITION, not identity: which dispatched tool call this link is
  presently executing inside, if any. It exists only so :meth:`child` can convert a
  link's current position into the child run's logged ``parent_tool_call_id`` — it is
  never logged itself.

Keeping position and identity separate is what makes nesting correct at any depth. A
draw made from INSIDE a dispatched tool call (browse's extraction) is handed a link
whose ``tool_call_id`` names that call; its own :meth:`child` folds that into its
``parent_tool_call_id``. A draw made from inside ANOTHER draw (no tool call of its
own) is handed a link whose ``tool_call_id`` is ``None`` — because the first draw's
own :meth:`child` already consumed its position into ITS ``parent_tool_call_id`` and
reset it — so the second draw's ``parent_tool_call_id`` is correctly ``None``, not
the outer tool call's id. Carrying ``parent_tool_call_id`` straight through unchanged
(rather than converting it from ``tool_call_id`` and resetting) is exactly the bug
this split was built to rule out.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class CallLink(BaseModel):
    """Immutable linkage for one model call or one dispatched tool call."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    parent_run_id: str | None
    parent_tool_call_id: str | None
    tool_call_id: str | None

    @staticmethod
    def top(run_id: str) -> CallLink:
        """A run starter that already has its own id — the channel minting a
        turn's id, a collector cycle minting its own, both of which need the
        bare string for DB/ledger purposes regardless of this link."""
        return CallLink(
            run_id=run_id, parent_run_id=None, parent_tool_call_id=None, tool_call_id=None
        )

    @staticmethod
    def new_top() -> CallLink:
        """A run starter with no run id of its own yet — a one-shot with
        nothing above it (startup's announcement, ``/profile``'s parse calls,
        image captioning). Mints one here — the one place minting still
        belongs."""
        return CallLink.top(uuid.uuid4().hex)

    def for_tool_call(self, tool_call_id: str) -> CallLink:
        """This same run, now POSITIONED inside one dispatched tool call —
        identity (``run_id``/``parent_run_id``/``parent_tool_call_id``)
        unchanged. Built by the tool dispatcher and handed to that call's
        ``execute``."""
        return CallLink(
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            parent_tool_call_id=self.parent_tool_call_id,
            tool_call_id=tool_call_id,
        )

    def child(self) -> CallLink:
        """The link for a single model call made ON BEHALF of this one — a
        micro-context's own draw, which mints its OWN run id for its own
        ledger attribution and links back to THIS run as its parent.

        Converts THIS link's POSITION (the tool call it is presently inside,
        if any) into the child's logged ``parent_tool_call_id``, then resets
        position to ``None`` — the child is a fresh run with no tool call of
        its own yet. A draw nested inside ANOTHER draw therefore gets
        ``parent_tool_call_id=None`` (the enclosing draw's own position was
        already consumed into ITS parent id and cleared), while a draw made
        straight from inside a dispatched tool call gets that call's real id."""
        return CallLink(
            run_id=uuid.uuid4().hex,
            parent_run_id=self.run_id,
            parent_tool_call_id=self.tool_call_id,
            tool_call_id=None,
        )
