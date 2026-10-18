"""Pydantic models and enums for agent loop."""

from enum import StrEnum

from pydantic import BaseModel, Field

from penny.constants import WriteGateOutcome
from penny.responses import PennyResponse


class MessageRole(StrEnum):
    """Valid message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallRecord(BaseModel):
    """Record of a tool call made during an agent run."""

    tool: str = Field(description="Tool name")
    arguments: dict = Field(description="Arguments passed to the tool")
    reasoning: str | None = Field(default=None, description="Model's reasoning for this tool call")
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
