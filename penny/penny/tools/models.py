"""Pydantic models for tool calling."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from penny.text_validity import half_formed_send_reason


class NoArgs(BaseModel):
    """Args model for a tool that takes no parameters.

    The default ``Tool.args_model`` — validation is a no-op (extra keys the model
    may pass are ignored), so an argless tool needs no per-tool model."""


class ToolResult(BaseModel):
    """The single structured result of running a tool.

    One uniform contract — what a tool returns from ``execute`` AND what the
    ``ToolExecutor`` hands back (it synthesises a failed result for framework
    errors a tool can't report itself: tool-not-found, bad arguments, timeout,
    uncaught exception).  No separate envelope, no bare strings, no ``str | T``.

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
    """Validated arguments for the send_message tool.

    The ``content`` validator is the tool's *message-validity* gate: it rejects a
    half-formed body (blank / punctuation-only, bare URL, bail-out phrase,
    unfinished fragment, ellipsis-truncated) via the shared
    ``half_formed_send_reason`` — the same rule the run-health classifier flags
    ``⚠ HALF-FORMED SEND`` on.  Running here (not inside ``execute``) means the
    ``ToolExecutor`` refuses the call with an actionable error tool response
    before the tool runs; ``execute`` then handles only delivery decisions
    (refusal/mute/recipient).
    """

    content: str

    @field_validator("content")
    @classmethod
    def _reject_half_formed(cls, value: str) -> str:
        if reason := half_formed_send_reason(value):
            raise ValueError(
                f"{reason} — that is not a complete message the user should receive. "
                "Send the COMPLETE message body: a finished, substantive sentence (or "
                "more), no placeholder punctuation, no bare link, no trailing ellipsis."
            )
        return value


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


class ToolDefinition(BaseModel):
    """Definition of a tool for the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
