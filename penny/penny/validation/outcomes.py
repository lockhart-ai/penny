"""The live side of model-I/O validation: dispositions a validator returns and
the dispatcher the agentic loop drives the validator chain with.

A ``ResponseValidator`` inspects one model response against a ``LoopContext`` and
returns a ``ValidationOutcome`` — *what the loop should do*, not just "reject".
The loop matches on it.  This is the dynamic-disposition analogue of the static
tool-arg validators: there, the only outcome is "reject the call"; here, a
malformed response might warrant a fresh draw in its place, a quiet in-place repair, an
error tool-result, a continue-with-nudge, or a hard stop.

Returning a typed disposition (rather than raising) is deliberate: the loop
catches and reshapes immediately into control flow, so a typed return is clearer
than exception-as-control-flow (see the project's error-handling guidance).
"""

from __future__ import annotations

from typing import Protocol, assert_never, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.llm.models import LlmResponse
from penny.tools.link_handles import HandleTable
from penny.validation.conditions import ConditionKey


class Proceed(BaseModel):
    """The response is acceptable — use it.  ``response`` carries the value the
    chain settled on (a validator's repairs are threaded through), so the loop
    always reads the post-repair response from here, never the raw one."""

    model_config = ConfigDict(frozen=True)
    response: LlmResponse | None = None


class Retry(BaseModel):
    """Discard the bad draw and re-draw on the unchanged conversation — once per
    ``condition`` (a repeat of the same condition proceeds with what it has).

    A retry re-samples the state the draft was drawn from.  Nothing is written into
    the conversation: not the draft, and not a note about it.  The conditions this
    serves — markup, a refusal, an ungrounded URL — are usually not repeated by a
    fresh draw, and when one is, the second draw proceeds because its condition is
    spent.

    ``note`` is the one exception: when a retry carries a note, the loop shows the
    model its draft (assistant turn) plus the note (user turn) before re-drawing —
    for a defect the model cannot usually see in its own draft.  The dangling-link
    rejection (#2717) is one: the note names which numbered links exist, so the
    re-draw cites a real one."""

    model_config = ConfigDict(frozen=True)
    condition: ConditionKey
    note: str | None = None


class Repair(BaseModel):
    """Transform the response in place and continue the chain (e.g. strip
    hallucinated tool calls, append an omitted source URL, clean malformed
    URLs).  A repair is silent — no re-call, no nudge."""

    model_config = ConfigDict(frozen=True)
    response: LlmResponse


class RejectToolCall(BaseModel):
    """Refuse the response's tool call(s) with an error tool-result and continue
    the loop (the model sees the error and retries) — e.g. a first-move
    ``done()``.  ``message`` is the error body shown for the rejected call(s)."""

    model_config = ConfigDict(frozen=True)
    message: str


class NudgeContinue(BaseModel):
    """Append the response plus a user-turn ``message`` and continue the loop — e.g.
    a chat run that just auto-extracted a skill and must narrate it in the same turn.

    Only ever applied to a draw that is VALID for the agent's shape: an invalid one is
    discarded and re-rolled before the loop sees it (#1839), so this disposition never
    stands in for recovery from malformed output."""

    model_config = ConfigDict(frozen=True)
    message: str


class Stop(BaseModel):
    """End the loop now, returning ``response`` — e.g. every tool call so far has
    failed, so there's no point continuing."""

    model_config = ConfigDict(frozen=True)
    response: ControllerResponse


# The closed set of things the loop can be told to do about a model response.
ValidationOutcome = Proceed | Retry | Repair | RejectToolCall | NudgeContinue | Stop


class LoopContext(BaseModel):
    """The run state a validator needs, passed explicitly (no fishing from agent
    state).  Carries only reads — applying a disposition is the loop's job.

    A validator that needs more should have it added here rather than reaching
    into the agent, so the validator stays a pure function of (response, ctx)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: int
    is_final_step: bool
    tools_available: bool
    # Concatenated source material (tool results + prompt + history) a response's
    # URLs must be grounded in — empty when there's nothing to check against.
    source_text: str = ""
    # The run's numbered link table (#2717) — what a ``[n]`` citation in the
    # response must resolve against.  Defaults to an empty registry, the honest
    # value for a run whose surfaces rendered no handles: with no links rendered,
    # no citation can dangle.
    handles: HandleTable = Field(default_factory=HandleTable)
    # Every tool-call record accumulated this run so far (for run-shape guards
    # like premature-done: "has any non-done work happened yet?").
    records: list[ToolCallRecord] = Field(default_factory=list)
    # Reasons already retried this model call — a validator keyed on one of these
    # must not ask to retry it again (the loop also enforces this).
    retried: set[ConditionKey] = Field(default_factory=set)
    # Neither a writes-landed frame nor a skill-narration frame is stamped here
    # anymore (#2455 naive baseline / #2473): what a chat run wrote is delivered
    # straight onto the tool result the moment its call lands (see
    # ``ChatAgent.after_tool_step``), and there is no skill machinery left to
    # narrate — so no run-shape validator needs a field on this ctx to redraw
    # from.


@runtime_checkable
class ResponseValidator(Protocol):
    """A single composable guard over a model response.

    Implementations are pure: read ``response`` + ``ctx``, return a disposition,
    mutate nothing.  Each validator owns exactly one condition from the taxonomy;
    a new guard is a new validator added to an agent's chain, not a new branch in
    the loop."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome: ...


def run_validators(
    validators: list[ResponseValidator],
    response: LlmResponse,
    ctx: LoopContext,
) -> ValidationOutcome:
    """Run the chain in order and return the loop's disposition.

    ``Repair`` threads its transformed response into the rest of the chain and
    continues; ``Proceed`` passes; the first ``Retry`` / ``RejectToolCall`` /
    ``NudgeContinue`` / ``Stop`` short-circuits and is returned.
    When the chain completes with no objection, returns ``Proceed`` carrying the
    (possibly-repaired) response."""
    working = response
    for validator in validators:
        match validator.check(working, ctx):
            case Repair(response=repaired):
                working = repaired
            case Proceed():
                pass
            case (Retry() | RejectToolCall() | NudgeContinue() | Stop()) as terminal:
                return terminal
            case unreachable:
                assert_never(unreachable)
    return Proceed(response=working)
