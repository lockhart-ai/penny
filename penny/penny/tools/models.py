"""Pydantic models for tool calling."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from penny.text_validity import half_formed_send_reason, is_blank


def _require_non_blank(value: str) -> str:
    """Reject a string that carries no word tokens (blank / punctuation only).

    The pure-arg "this field can't be empty" rule, shared by every text field a
    tool genuinely needs filled (an email subject, an email body, an image
    description).  ``is_blank`` (from ``text_validity``) is the same predicate the
    corpus write path uses, so "what counts as empty" is one definition across the
    codebase.
    """
    if is_blank(value):
        raise ValueError(
            "must not be blank — provide non-empty text (real content), not "
            "whitespace or punctuation"
        )
    return value


# A required text field that must carry at least one word token.
NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]


def _require_queries(value: list[str]) -> list[str]:
    """Reject an empty browse ``queries`` list with an actionable message."""
    if not value:
        raise ValueError("provide at least one search query or URL")
    return value


# Browse queries: at least one search query or URL.
QueryList = Annotated[list[str], AfterValidator(_require_queries)]


def _require_email_ids(value: list[str]) -> list[str]:
    """Reject an empty ``email_ids`` list, pointing at where IDs come from.

    A bare ``Field(min_length=1)`` would surface Pydantic's generic "list should
    have at least 1 item" — true but not actionable.  Name the source so the
    model knows the fix is to read IDs from a prior listing call first."""
    if not value:
        raise ValueError(
            "provide at least one email ID from a prior search_emails or list_emails result"
        )
    return value


# Email IDs to read: at least one, sourced from a prior search/list call.
EmailIdList = Annotated[list[str], AfterValidator(_require_email_ids)]


def _require_recipients(value: list[str]) -> list[str]:
    """Reject an empty recipient ``to`` list with an actionable message."""
    if not value:
        raise ValueError("provide at least one recipient email address")
    return value


# Draft recipients: at least one address.
RecipientList = Annotated[list[str], AfterValidator(_require_recipients)]


class ToolArgs(BaseModel):
    """Base for every tool's Pydantic arg model.

    ``extra="forbid"`` routes an unknown parameter — a misspelled optional
    argument (``count=`` where the tool takes ``k=``), a stale field name —
    through the arg-validation envelope (``Tool._validation_error_message``) as an
    actionable rejection ("unknown parameter 'count' …") instead of Pydantic's
    default of silently dropping it and running the tool with default behaviour: a
    silent no-op parameter the model has no signal to correct.

    The framework-injected ``reasoning`` param (``Tool.to_ollama_tool``) is popped
    from the call arguments *before* validation (``Agent._dedup_tool_calls``), so
    ``forbid`` never sees it — real tool calls are unaffected.  (``browse``
    additionally declares ``reasoning`` as a genuine field, since it advertises the
    param in its own schema.)
    """

    model_config = ConfigDict(extra="forbid")


class NoArgs(ToolArgs):
    """Args model for a tool that takes no parameters.

    The default ``Tool.args_model``.  It declares no fields, so — inheriting
    ``extra="forbid"`` from ``ToolArgs`` — any argument the model passes to an
    argless tool is an unknown parameter and is rejected through the envelope,
    rather than silently ignored."""


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
    - ``narration``: an explicit first-person frame for ``Tool.format_result`` to lead
      with, overriding the registry-dispatched ``to_result_narration``.  Set only for
      framework-synthesised failures the tool itself can't narrate — a tool-not-found
      result has NO registered class to dispatch a narration from, and a timeout /
      uncaught-exception / bad-arguments result knows *why* it failed in a way the
      generic per-tool narration can't.  ``None`` (the default) means "narrate via the
      normal dispatch", so every tool-returned result is unchanged.  The ``message``
      stays the actionable remedy (the #1414 house template's diagnosis + how-to-fix
      tail); this field is only the frame around it.

    Images are not carried here: the browse tool stores them in the media table at
    capture time and they are matched back side-channel at egress.
    """

    message: str
    success: bool = True
    mutated: bool = False
    source_urls: list[str] = Field(default_factory=list)
    narration: str | None = None

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


class BrowseArgs(ToolArgs):
    """Validated arguments for the browse tool.

    ``queries`` must carry at least one entry: an empty browse call did nothing
    silently, so it's rejected at the arg gate with an actionable message rather
    than reaching ``execute`` and no-op'ing.
    """

    queries: QueryList
    reasoning: str | None = None


class SendMessageArgs(ToolArgs):
    """Validated arguments for the send_message tool.

    The ``content`` validator is the tool's *message-validity* gate: it rejects a
    WHOLE-message half-formed body (blank / punctuation-only, bare URL, bail-out
    phrase, or an unfinished/ellipsis-truncated TAIL) via the shared
    ``half_formed_send_reason`` — the same rule the run-health classifier flags
    ``⚠ HALF-FORMED SEND`` on.  Running here (not inside ``execute``) means the
    ``ToolExecutor`` refuses the call with an actionable error tool response
    before the tool runs; ``execute`` then handles only delivery decisions
    (refusal/mute/recipient).

    ``half_formed_send_reason`` is already an actionable message (the specific
    defect + the next move), so the validator raises it verbatim — it must NOT be
    wrapped in a generic "send the COMPLETE message" tail, which misdirects when
    the defect is specific (e.g. a truncated tail on an otherwise complete note).
    """

    content: str

    @field_validator("content")
    @classmethod
    def _reject_half_formed(cls, value: str) -> str:
        if reason := half_formed_send_reason(value):
            raise ValueError(reason)
        return value


class GenerateImageArgs(ToolArgs):
    """Validated arguments for the generate_image tool.

    ``description`` must carry at least one word token — an empty description
    can't be drawn, so the blank case is rejected at the arg gate with an
    actionable message (via the shared ``NonBlankText`` rule) rather than
    reaching ``execute`` and calling the image model with nothing.
    """

    description: NonBlankText


class SearchEmailsArgs(ToolArgs):
    """Validated arguments for the search_emails tool.

    Every field is optional, but an all-empty search is meaningless (it would
    match the whole mailbox), so at least one criterion must be supplied — the
    ``_require_a_criterion`` validator rejects the empty call at the arg gate.
    """

    text: str | None = None
    from_addr: str | None = None
    subject: str | None = None
    after: str | None = None
    before: str | None = None

    @model_validator(mode="after")
    def _require_a_criterion(self) -> SearchEmailsArgs:
        if not any((self.text, self.from_addr, self.subject, self.after, self.before)):
            raise ValueError(
                "provide at least one search criterion — text, from_addr, subject, after, or before"
            )
        return self


class ReadEmailsArgs(ToolArgs):
    """Validated arguments for the read_emails tool.

    ``email_ids`` must be non-empty: reading no emails is a no-op, so the empty
    list is rejected at the arg gate (the ``NO_EMAILS_TO_READ`` guidance) rather
    than reaching ``execute``.
    """

    email_ids: EmailIdList


class ListEmailsArgs(ToolArgs):
    """Validated arguments for the list_emails tool."""

    folder: str | None = None


class DraftEmailArgs(ToolArgs):
    """Validated arguments for the draft_email tool.

    Structural validation only — a recipient list with at least one address and a
    non-blank subject and body.  Address *format* is not checked here; a malformed
    address is a runtime send failure handled in ``execute``.
    """

    to: RecipientList
    subject: NonBlankText
    body: NonBlankText
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
