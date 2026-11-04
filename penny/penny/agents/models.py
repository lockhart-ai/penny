"""Pydantic models and enums for agent loop."""

import asyncio
import logging
import uuid
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, Field

from penny.constants import PennyConstants, WriteGateOutcome
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.channels.base import QueuedTurnMessage

logger = logging.getLogger(__name__)


class TurnInbox:
    """The active chat turn's queue of messages waiting to join it (#2985).

    Lives on the agent side (the loop drains it) but carries no channel
    dependency, so both the loop and the channels import it freely.  Two drain
    points, with deliberately different accounting:

    - ``drain_ready()`` is the agent loop's mid-run drain (after a tool step,
      before the next model call).  Everything it pops JOINS the run — recorded
      in ``_drained`` so the channel logs each one's ``messagelog`` row under
      the run it joined when the turn ends.
    - ``take_next()`` is the series loop's tail pick (the turn ended with
      messages still queued).  What it pops becomes its OWN turn's triggering
      message, logged by that turn's ``_finish_turn`` — so it is deliberately
      NOT recorded as drained.

    ``finished`` is set exactly once when the whole turn series ends, success
    or failure; a joining channel's typing indicator waits on it.
    """

    def __init__(self, owner_sender: str) -> None:
        self._owner_sender = owner_sender
        self._pending: deque[QueuedTurnMessage] = deque()
        self._drained: list[QueuedTurnMessage] = []
        self._finished = asyncio.Event()
        self._closed = False

    @classmethod
    def closed(cls) -> TurnInbox:
        """The null inbox background callers pass — drains nothing, accepts nothing."""
        return _CLOSED_INBOX

    @property
    def owner_sender(self) -> str:
        """The platform sender of the message that started this turn."""
        return self._owner_sender

    @property
    def finished(self) -> asyncio.Event:
        """Set once when the whole turn series ends (success or failure)."""
        return self._finished

    def put(self, message: QueuedTurnMessage) -> None:
        """Queue a message to join the active turn."""
        if self._closed:
            raise RuntimeError("Cannot join a message to a closed turn inbox")
        self._pending.append(message)

    def pending(self) -> bool:
        """Whether any message is still waiting to join."""
        return bool(self._pending)

    def drain_ready(self) -> list[QueuedTurnMessage]:
        """The mid-run drain: pop everything queued, in arrival order, recording
        each as joined to this run.  A closed inbox (no live user) drains nothing."""
        if self._closed:
            return []
        ready = list(self._pending)
        self._pending.clear()
        self._drained.extend(ready)
        return ready

    def take_next(self) -> QueuedTurnMessage | None:
        """The series tail pick: the oldest still-queued message, NOT recorded as
        drained — it becomes the next turn's own triggering message."""
        if self._closed:
            return None
        return self._pending.popleft() if self._pending else None

    def take_drained(self) -> list[QueuedTurnMessage]:
        """Everything drained into the run so far, in drain order — consumed by
        the channel once, when it logs the joined rows at turn end."""
        drained = self._drained
        self._drained = []
        return drained

    def drain_all(self) -> list[QueuedTurnMessage]:
        """Pop everything still queued WITHOUT recording it as drained — the
        abandon path: these messages never joined a run, so no row of theirs is
        logged under one; the caller answers them itself."""
        if self._closed:
            return []
        pending = list(self._pending)
        self._pending.clear()
        return pending

    async def release(self, queued: QueuedTurnMessage) -> None:
        """Remove the waiting mark the joining channel left on this message
        (#2985) — the moment it leaves the pending queue, however it leaves:
        injected into the running turn, picked up by a follow-up turn, or
        abandoned.  Best-effort by contract: a failed release is logged and
        never fails the turn."""
        try:
            await queued.clear_queued(queued.message)
        except Exception:
            logger.exception("Failed to clear the waiting mark on a queued message")

    def finish(self) -> None:
        """Release everything waiting on the series' end."""
        self._finished.set()


_CLOSED_INBOX = TurnInbox("")
_CLOSED_INBOX._closed = True

NO_INBOX = TurnInbox.closed()


class MessageRole(StrEnum):
    """Valid message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class HistoryTurn(NamedTuple):
    """One turn of the chat history window, as ``Agent._build_conversation`` (and
    ``ChatAgent.get_history``'s quote-reply path) reads it off ``MessageLog``: the
    role, the text as it was said, and — for an outgoing reply only — the run id
    the framework stamped on it (#2630).

    ``run_id`` is what lets ``Agent._build_messages`` tell a talk-only reply from
    one a real tool call backed (#2915/#2888): with no run id, or a run that made
    no calls, a turn renders as the single plain message it always has; a run
    that DID call tools expands into its own native call + tool-result turns
    ahead of the reply text, so the act is visible beside the words rather than
    only the words riding into the next turn's history.  ``None`` on a user turn
    (which carries no run) and on an autonomous send (which has no run behind it
    at all)."""

    role: str
    content: str
    run_id: str | None = None


def new_history_replay_call_id() -> str:
    """Mint a fresh id for a tool call rendered into history from a PAST run —
    never the id that call originally logged under, so a later run's own
    ``messages`` snapshot structurally distinguishes "a call this run made" from
    "a call replayed from an earlier run's own record"."""
    return f"{PennyConstants.HISTORY_REPLAY_CALL_ID_PREFIX}{uuid.uuid4().hex}"


def is_history_replay_call_id(call_id: str | None) -> bool:
    """Whether ``call_id`` was minted by :func:`new_history_replay_call_id` — the
    one check every reader of a logged run's tool turns applies before counting
    one as a call THAT run made (see the constant's own docstring)."""
    return call_id is not None and call_id.startswith(PennyConstants.HISTORY_REPLAY_CALL_ID_PREFIX)


class ToolCallRecord(BaseModel):
    """Record of a tool call made during an agent run."""

    tool: str = Field(description="Tool name")
    arguments: dict = Field(description="Arguments passed to the tool")
    failed: bool = Field(
        default=False, description="Whether the tool returned an error or empty result"
    )
    mutated: bool = Field(
        default=False,
        description=(
            "Whether this call actually changed durable state (a row written, a "
            "message sent).  False for reads, refusals, and successful no-ops "
            "(duplicate-rejected write, update/delete/move on a missing key).  "
            "Drives the collector's work/no-work split and auto-throttle."
        ),
    )
    result: str | None = Field(
        default=None, description="The tool's result/error text, set after execution"
    )
    stop_reason: WriteGateOutcome | None = Field(
        default=None,
        description=(
            "The enumerated write-gate outcome that ends a must-act (collector) run "
            "at the chokepoint (#1587).  Set from ``ToolResult.stop`` (collection_write "
            "on a collector-scoped write); the collector's ``should_stop_loop`` reads "
            "it to exit, and ``_cycle_result`` stamps it as the run's stop reason.  "
            "None for every call that doesn't carry a STOP."
        ),
    )
    media_id: int | None = Field(
        default=None,
        description=(
            "The id of a media row this call created that egress must attach to the "
            "reply — the deterministic generate→deliver link (generate_image).  None "
            "for calls that produce no deliverable image."
        ),
    )


# The error class stamped when the reroll budget ran out rather than one call raising:
# every draw was discarded as unusable, so the failure is the budget, and WHICH condition
# kept tripping rides in the message.
REROLL_EXHAUSTED = "reroll-exhausted"

# The error class stamped when a draw came back cut by its own token budget
# (``LlmResponse.finish_reason == PennyConstants.FINISH_REASON_LENGTH``, #2475) rather
# than discarded as merely unusable.  Distinct from ``REROLL_EXHAUSTED``: this fires on
# the FIRST cut draw, never re-rolled on the same budget and thinking settings first,
# because a cap that cut once would almost certainly cut the retry too.
BUDGET_EXCEEDED = "budget-exceeded"


class ModelCallError(BaseModel):
    """Why one model call came back with nothing usable (#1909).

    A call that raises never reaches the client's persist step, so it leaves no
    ``promptlog`` row — this is the only record of it.  ``error_class`` is the
    exception's own class name (``LlmTimeoutError``, ``LlmConnectionError``,
    ``LlmResponseError``) or ``REROLL_EXHAUSTED`` when every draw was discarded.
    """

    error_class: str = Field(
        description="The exception's class name, or REROLL_EXHAUSTED for a spent budget"
    )
    message: str = Field(description="The error's own message, or which conditions tripped")

    def __str__(self) -> str:
        return f"{self.error_class}: {self.message}"


class ModelCallAbortedError(Exception):
    """A model call came back with nothing usable, so the run must end (#1909).

    Raised rather than returned: the failure carries a MESSAGE, and a caller handed an
    error object in the success channel has to discriminate it at every call site
    instead of catching it once.  Self-rendering — ``str(exc)`` is the failure's own
    line — with the structured ``error`` for the record that outlives the run.
    """

    def __init__(self, error: ModelCallError) -> None:
        super().__init__(str(error))
        self.error = error


class RunAbort(BaseModel):
    """Where an agent run died and why (#1909).

    A run that aborts on a model call closes with no ``done()``, and the outcome it
    used to stamp said only that — so 31 of 75 measured collector cycles died with no
    observable cause.  These are the structural facts of the abort, assembled where
    each is known: the loop supplies WHERE (the step index, and the tool the last
    successful step ran — the anchor the step number alone doesn't give), the invoke
    layer supplies WHY.
    """

    step: int = Field(description="The 1-based loop step whose model call failed")
    after_tool: str | None = Field(
        default=None,
        description="The tool the last successful step executed, or None if none had landed",
    )
    error: ModelCallError = Field(description="What the failed call came back with")

    def render(self) -> str:
        """The run's stamped reason — one plain line, since it renders into the run
        record the model reads (``[target] <reason>``)."""
        after = f" after {self.after_tool}" if self.after_tool else ""
        return f"model call failed at step {self.step}{after}: {self.error}"


class ControllerResponse(BaseModel):
    """Response from the agentic controller."""

    answer: str = Field(description="The final answer from the controller")
    thinking: str | None = Field(
        default=None, description="Optional thinking/reasoning trace from the model"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Tool calls made during this run"
    )
    abort: RunAbort | None = Field(
        default=None,
        description=(
            "The structural cause of a run that ended on a failed model call (#1909) — "
            "set only on the aborted-run response, None for every run that reached a "
            "terminal state of its own."
        ),
    )
    ran_out_of_steps: bool = Field(
        default=False,
        description=(
            "Whether this run hit its step cap without a final answer (#2370).  "
            "Structural, read by _run_outcome / Collector._unfinished_reason instead "
            "of matching `answer` against a sentinel string — `answer` now carries "
            "whatever the model's (or the deterministic account's) honest text is."
        ),
    )

    @property
    def generated_media_ids(self) -> list[int]:
        """Media rows created this run that egress must attach to the reply.

        Derived from ``tool_calls`` (no denormalized field): the deterministic
        generate→deliver link.  ``generate_image`` stamps the id of the row it
        stored onto its ``ToolCallRecord.media_id``; the channel fetches exactly
        those rows at egress rather than fuzzy-matching the media table.
        """
        return [record.media_id for record in self.tool_calls if record.media_id is not None]


def account_of(facts: str, tool_calls: list[ToolCallRecord]) -> str:
    """The deterministic account for when even the narrated draw comes back
    unusable (#2370): ``facts`` — the plain statement of this run's own state,
    handed to that draw — plus a plain list of what was tried, built from the
    record.  Never an apology, and never a second model call: this is the
    fallback for when the model had nothing more to say, not a retry of it.
    Reused at the channel layer (``channels/base.py``) for the rare case a
    reply reaches send time still empty (e.g. a deliberate write-gate STOP)."""
    if not tool_calls:
        return f"{facts} {PennyResponse.ACCOUNT_NO_TOOL_CALLS}"
    tried = ", ".join(
        f"{call.tool}{PennyResponse.ACCOUNT_FAILED_SUFFIX}" if call.failed else call.tool
        for call in tool_calls
    )
    return f"{facts} {PennyResponse.ACCOUNT_TRIED_PREFIX}{tried}."
