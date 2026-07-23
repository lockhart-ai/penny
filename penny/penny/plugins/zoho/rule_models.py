"""Typed condition/action models for email rules (#1737).

The condition and action of an email rule were raw JSON dicts; these Pydantic
models replace them so the shape is validated at the tool boundary rather than
absorbed blindly.  Both are ``extra="forbid"`` and require at least one field —
an empty or unknown shape is a loud, teachable rejection, never a silent no-op
rule that would match everything or nothing.

Kept in this dedicated module (not ``plugins/zoho/models.py``) so both the tool
layer (``email_tools.py``) and the application engine (``rules.py``) can import
them without pulling the Zoho API model surface, and so the database layer never
imports a plugin model (the ``EmailRule`` row stores their serialized JSON).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The reject-and-teach messages — one per shape, naming every supported field so a
# rejected call carries the fix (the actionable-tool-failure rule).
CONDITION_EMPTY_MESSAGE = (
    "a rule condition needs at least one of: from, subject_contains, body_contains"
)
ACTION_EMPTY_MESSAGE = "a rule action needs at least one of: move_to, label"


def describe_fields(model: BaseModel) -> str:
    """Render a model's set fields as ``key=value, …`` (by alias, None omitted).

    The human-readable form used in the create echo and the rule listing — shows
    exactly the matchers/actions the model supplied, nothing else."""
    fields = model.model_dump(by_alias=True, exclude_none=True)
    return ", ".join(f"{key}={value}" for key, value in fields.items())


class EmailRuleCondition(BaseModel):
    """What an email must match for a rule to fire — at least one matcher set."""

    model_config = ConfigDict(extra="forbid")

    from_: str | None = Field(default=None, alias="from")
    subject_contains: str | None = None
    body_contains: str | None = None

    @model_validator(mode="after")
    def _require_one(self) -> EmailRuleCondition:
        if self.from_ is None and self.subject_contains is None and self.body_contains is None:
            raise ValueError(CONDITION_EMPTY_MESSAGE)
        return self


class EmailRuleAction(BaseModel):
    """What to do with an email a rule matched — at least one action set."""

    model_config = ConfigDict(extra="forbid")

    move_to: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _require_one(self) -> EmailRuleAction:
        if self.move_to is None and self.label is None:
            raise ValueError(ACTION_EMPTY_MESSAGE)
        return self
