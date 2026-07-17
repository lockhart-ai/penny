"""The live side of model-I/O validation: dispositions a validator returns and
the dispatcher the agentic loop drives the validator chain with.

A ``ResponseValidator`` inspects one model response against a ``LoopContext`` and
returns a ``ValidationOutcome`` — *what the loop should do*, not just "reject".
The loop matches on it.  This is the dynamic-disposition analogue of the static
tool-arg validators: there, the only outcome is "reject the call"; here, a
malformed response might warrant a retry-with-nudge, a quiet in-place repair, an
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
from penny.validation.conditions import ConditionKey


class Proceed(BaseModel):
    """The response is acceptable — use it.  ``response`` carries the value the
    chain settled on (a validator's repairs are threaded through), so the loop
    always reads the post-repair response from here, never the raw one."""

    model_config = ConfigDict(frozen=True)
    response: LlmResponse | None = None


class Retry(BaseModel):
    """Re-call the model with a nudge appended.  The loop appends the bad
    response and ``nudge`` as turns, then re-invokes — once per ``condition`` (a
    repeat of the same condition exhausts and the loop proceeds with what it
    has)."""

    model_config = ConfigDict(frozen=True)
    condition: ConditionKey
    nudge: str


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
    """Append the response plus a user-turn ``message`` and continue the loop —
    e.g. a collector that narrated prose where a tool call was required."""

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
    # Every tool-call record accumulated this run so far (for run-shape guards
    # like premature-done: "has any non-done work happened yet?").
    records: list[ToolCallRecord] = Field(default_factory=list)
    # Reasons already retried this model call — a validator keyed on one of these
    # must not ask to retry it again (the loop also enforces this).
    retried: set[ConditionKey] = Field(default_factory=set)
    # The rendered frame of a skill this run just auto-extracted (#1658), stamped
    # onto the text-branch ctx by ``ChatAgent._prepare_text_shape`` so the chat
    # ``SkillNarrationValidator`` can narrate it (SAID==DID).  ``None`` when the run
    # did not qualify for extraction, so every non-chat / non-qualifying run reads
    # the same empty default.
    learned_skill_frame: str | None = None


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
    ``NudgeContinue`` / ``Stop`` short-circuits and is returned.  When the chain
    completes with no objection, returns ``Proceed`` carrying the
    (possibly-repaired) response."""
    working = response
    for validator in validators:
        match validator.check(working, ctx):
            case Repair(response=repaired):
                working = repaired
            case Proceed():
                pass
            case Retry() | RejectToolCall() | NudgeContinue() | Stop() as terminal:
                return terminal
            case unreachable:
                assert_never(unreachable)
    return Proceed(response=working)
