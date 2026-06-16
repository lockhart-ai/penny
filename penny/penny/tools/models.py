"""Pydantic models for tool calling."""

from typing import Any

from pydantic import BaseModel, Field


class ToolOutcome(BaseModel):
    """The single structured result every ``Tool.execute`` returns.

    One uniform contract instead of bare strings (or the ``str | T`` half-measure):

    - ``message``: the model-facing body, rendered into the tool result the LLM reads.
    - ``success``: ``False`` for errors, refusals, or empty/no-result outcomes —
      becomes ``ToolCallRecord.failed``.
    - ``mutated``: the call changed durable state or had an outbound side effect
      (a row written, an entry moved/deleted, a message sent).  ``False`` for reads
      and *successful no-ops* (a duplicate-rejected write, an update/delete/move on a
      missing key) — this is the signal the collector's work/no-work split and
      auto-throttle ride on.
    - ``source_urls``: URLs the final reply should cite (browse) — threaded into the
      response's source-appending.

    Images are not carried here: the browse tool stores them in the media table at
    capture time and they are matched back side-channel at egress.
    """

    message: str
    success: bool = True
    mutated: bool = False
    source_urls: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return self.message


class BrowsePage(BaseModel):
    """A single page read by the browse tool, before sections are assembled.

    Carries the page image (a base64 ``data:`` URI), source URL, and title out
    to the tool's media-capture step; none of these reach the model.
    """

    text: str
    image: str | None = None
    title: str | None = None
    url: str | None = None


class BrowseArgs(BaseModel):
    """Validated arguments for the browse tool."""

    queries: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class SendMessageArgs(BaseModel):
    """Validated arguments for the send_message tool."""

    content: str


class SearchEmailsArgs(BaseModel):
    """Validated arguments for the search_emails tool."""

    text: str | None = None
    from_addr: str | None = None
    subject: str | None = None
    after: str | None = None
    before: str | None = None


class ReadEmailsArgs(BaseModel):
    """Validated arguments for the read_emails tool."""

    email_ids: list[str]


class ListEmailsArgs(BaseModel):
    """Validated arguments for the list_emails tool."""

    folder: str | None = None


class DraftEmailArgs(BaseModel):
    """Validated arguments for the draft_email tool."""

    to: list[str]
    subject: str
    body: str
    cc: list[str] | None = None


class ToolCall(BaseModel):
    """A tool call from the model."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool: str
    result: Any
    error: str | None = None
    id: str | None = None


class ToolDefinition(BaseModel):
    """Definition of a tool for the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
