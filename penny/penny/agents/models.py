"""Pydantic models and enums for agent loop."""

from enum import StrEnum

from pydantic import BaseModel, Field


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
    media_id: int | None = Field(
        default=None,
        description=(
            "The id of a media row this call created that egress must attach to the "
            "reply — the deterministic generate→deliver link (generate_image).  None "
            "for calls that produce no deliverable image."
        ),
    )


class ControllerResponse(BaseModel):
    """Response from the agentic controller."""

    answer: str = Field(description="The final answer from the controller")
    thinking: str | None = Field(
        default=None, description="Optional thinking/reasoning trace from the model"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Tool calls made during this run"
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
