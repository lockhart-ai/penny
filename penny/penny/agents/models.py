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


class ControllerResponse(BaseModel):
    """Response from the agentic controller."""

    answer: str = Field(description="The final answer from the controller")
    thinking: str | None = Field(
        default=None, description="Optional thinking/reasoning trace from the model"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Tool calls made during this run"
    )
