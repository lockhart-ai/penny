"""Pydantic models and enums for agent loop."""

from enum import StrEnum

from pydantic import BaseModel, Field

from penny.constants import WriteGateOutcome


class MessageRole(StrEnum):
    """Valid message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        """Convert to dict for Ollama API."""
        return {"role": self.role.value, "content": self.content}


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

    @property
    def generated_media_ids(self) -> list[int]:
        """Media rows created this run that egress must attach to the reply.

        Derived from ``tool_calls`` (no denormalized field): the deterministic
        generate→deliver link.  ``generate_image`` stamps the id of the row it
        stored onto its ``ToolCallRecord.media_id``; the channel fetches exactly
        those rows at egress rather than fuzzy-matching the media table.
        """
        return [record.media_id for record in self.tool_calls if record.media_id is not None]
