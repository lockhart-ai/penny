"""Zoho Mail organisation tools — LLM-callable tools for the Zoho plugin."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from penny.llm.call_link import CallLink
from penny.plugins.zoho.mail_models import ZohoFolder
from penny.plugins.zoho.rule_models import (
    EmailRuleAction,
    EmailRuleCondition,
    describe_fields,
)
from penny.tools.base import PLAN_ARG_DESCRIPTION, Tool, as_plan_result
from penny.tools.models import ToolResult
from penny.tools.plan_tool import ResolvingMutationTool

if TYPE_CHECKING:
    from penny.database import Database
    from penny.plugins.zoho.mail_client import ZohoClient

logger = logging.getLogger(__name__)

# The account a client failure gets (#2369): what happened, with no order attached — the
# model reads the state and reasons from it, same as every other tool result. Catches the
# mail client's own failures — a non-2xx response or a transport error (both
# ``httpx.HTTPError``), or a failed OAuth token refresh / missing-account response
# (``RuntimeError``, raised by ``ZohoClient`` itself — it has its own OAuth logic, not the
# shared ``ZohoOAuthClient``) — never a bug in this file's own logic, which still propagates
# to the executor's generic handler as today. ``create_email_rule``/``list_email_rules``
# reach only ``db.email_rules`` (no network client), so they get no catch here.
MOVE_EMAILS_FAILED = "The emails couldn't be moved ({e}), so nothing was moved in the mailbox."
CREATE_FOLDER_FAILED = (
    "The folder couldn't be created ({e}), so nothing was created in the mailbox."
)
APPLY_LABEL_FAILED = "The label couldn't be applied ({e}), so nothing was changed in the mailbox."
LIST_LABELS_FAILED = (
    "The mailbox labels couldn't be listed ({e}), so nothing was read from the mailbox."
)


class MoveEmailsArgs(BaseModel):
    """Arguments for moving emails to a folder."""

    message_ids: list[str] = Field(description="List of email message IDs to move")
    folder_path: str = Field(description="Destination folder path (e.g., 'Clients/John Smith')")
    create_if_missing: bool = Field(default=True, description="Create folder if it doesn't exist")
    # The structural plan mark (#2947, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CreateFolderArgs(BaseModel):
    """Arguments for creating an email folder."""

    folder_path: str = Field(description="Folder path to create (e.g., 'Accounting/Expenses/AWS')")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class ApplyLabelArgs(BaseModel):
    """Arguments for applying a label to emails."""

    message_ids: list[str] = Field(description="List of email message IDs to label")
    label_name: str = Field(description="Label name to apply (e.g., 'completed')")
    create_if_missing: bool = Field(default=True, description="Create label if it doesn't exist")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CreateEmailRuleArgs(BaseModel):
    """Arguments for creating an email rule.

    ``condition`` and ``action`` are typed (not raw dicts): an unknown field or
    an all-empty shape is rejected at the tool boundary with an actionable
    message naming the supported fields, so a rule can never be saved that would
    match everything or nothing."""

    name: str = Field(description="Human-readable rule name")
    condition: EmailRuleCondition = Field(
        description="Rule condition (from, subject_contains, etc.)"
    )
    action: EmailRuleAction = Field(description="Rule action (move_to, label, etc.)")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


@dataclass
class _ResolvedMoveEmails:
    """What ``MoveEmailsTool.resolve`` works out before the mutating call(s)
    — the existing folder, or ``None`` meaning "doesn't exist yet, would be
    created" (resolve never calls ``create_nested_folder`` itself — that's a
    mutation, deferred to ``apply``)."""

    args: MoveEmailsArgs
    folder: ZohoFolder | None


class MoveEmailsTool(ResolvingMutationTool[MoveEmailsArgs]):
    """Move emails to a folder, creating the folder if needed."""

    name = "move_emails"
    mutates = True
    description = (
        "Move one or more emails to a destination folder. "
        "The folder path can include nested folders like 'Clients/John Smith' or "
        "'Accounting/Expenses/AWS'. Folders will be created if they don't exist. "
        "Use this after reading emails to organize them into appropriate folders.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email message IDs to move",
            },
            "folder_path": {
                "type": "string",
                "description": (
                    "Destination folder path. Can be nested like 'Clients/John Smith' "
                    "or 'Accounting/Expenses/AWS'"
                ),
            },
            "create_if_missing": {
                "type": "boolean",
                "description": "Create the folder if it doesn't exist (default: true)",
            },
        },
        "required": ["message_ids", "folder_path"],
    }
    args_model = MoveEmailsArgs
    failure_message = MOVE_EMAILS_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Moving emails"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def resolve(self, args: MoveEmailsArgs) -> _ResolvedMoveEmails | ToolResult:
        """Look up the destination folder — never create it here, even when
        ``create_if_missing`` — so a plan never calls a mutating endpoint."""
        if not args.message_ids:
            return ToolResult(message="No message IDs provided.", success=False)

        folder = await self._client.get_folder_by_name(args.folder_path.split("/")[-1])
        if not folder and not args.create_if_missing:
            return ToolResult(message=f"Folder not found: {args.folder_path}", success=False)
        return _ResolvedMoveEmails(args=args, folder=folder)

    def describe(self, resolved: _ResolvedMoveEmails) -> ToolResult:
        """The plan headline — names whether the folder already exists."""
        count = len(resolved.args.message_ids)
        if resolved.folder:
            return ToolResult(message=f"move {count} email(s) to '{resolved.args.folder_path}'.")
        return ToolResult(
            message=(
                f"create folder '{resolved.args.folder_path}' (it doesn't exist yet) "
                f"and move {count} email(s) into it."
            )
        )

    async def apply(self, resolved: _ResolvedMoveEmails) -> ToolResult:
        """Create the folder if it was missing, then move the batch."""
        args = resolved.args
        folder = resolved.folder
        if not folder:
            folder = await self._client.create_nested_folder(args.folder_path)
            if not folder:
                return ToolResult(
                    message=f"Failed to create folder: {args.folder_path}",
                    success=False,
                )

        success = await self._client.move_messages(args.message_ids, folder.folder_id)
        if success:
            # The Zoho endpoint moves the whole batch in one call and answers with a
            # single aggregate status, not a per-message confirmation (#2768) — so the
            # count in the request is never stated as a count the backend confirmed.
            return ToolResult(
                message=f"Moved the email(s) to '{args.folder_path}'.",
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to move emails to '{args.folder_path}'",
            success=False,
        )


class CreateFolderTool(ResolvingMutationTool[CreateFolderArgs]):
    """Create an email folder with optional nesting."""

    name = "create_folder"
    mutates = True
    description = (
        "Create a new email folder. Supports nested folder paths like "
        "'Clients/John Smith' or 'Accounting/Expenses/AWS'. "
        "Parent folders will be created automatically if they don't exist.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Folder path to create. Can be nested like 'Clients/John Smith'",
            },
        },
        "required": ["folder_path"],
    }
    args_model = CreateFolderArgs
    failure_message = CREATE_FOLDER_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating folder"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def resolve(self, args: CreateFolderArgs) -> CreateFolderArgs:
        """Nothing to look up — the whole call is the args themselves."""
        return args

    def describe(self, resolved: CreateFolderArgs) -> ToolResult:
        return ToolResult(message=f"create folder '{resolved.folder_path}'.")

    async def apply(self, resolved: CreateFolderArgs) -> ToolResult:
        """The one mutating call — create the (possibly nested) folder."""
        folder = await self._client.create_nested_folder(resolved.folder_path)
        if folder:
            return ToolResult(
                message=f"Successfully created folder: {resolved.folder_path}",
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to create folder: {resolved.folder_path}",
            success=False,
        )


@dataclass
class _ResolvedApplyLabel:
    """What ``ApplyLabelTool.resolve`` works out before the mutating
    call(s) — the existing label (a plain dict, as ``get_label_by_name``
    returns it), or ``None`` meaning "doesn't exist yet, would be created"."""

    args: ApplyLabelArgs
    label: dict[str, Any] | None


class ApplyLabelTool(ResolvingMutationTool[ApplyLabelArgs]):
    """Apply a label to emails."""

    name = "apply_label"
    mutates = True
    description = (
        "Apply a label to one or more emails. Labels help categorize emails "
        "without moving them. Common labels include 'completed', 'pending', "
        "'urgent', etc. The label will be created if it doesn't exist.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email message IDs to label",
            },
            "label_name": {
                "type": "string",
                "description": "Label name to apply (e.g., 'completed', 'pending')",
            },
            "create_if_missing": {
                "type": "boolean",
                "description": "Create the label if it doesn't exist (default: true)",
            },
        },
        "required": ["message_ids", "label_name"],
    }
    args_model = ApplyLabelArgs
    failure_message = APPLY_LABEL_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Applying label"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def resolve(self, args: ApplyLabelArgs) -> _ResolvedApplyLabel | ToolResult:
        """Look up the label — never create it here, even when
        ``create_if_missing`` — so a plan never calls a mutating endpoint."""
        if not args.message_ids:
            return ToolResult(message="No message IDs provided.", success=False)

        label = await self._client.get_label_by_name(args.label_name)
        if not label and not args.create_if_missing:
            return ToolResult(message=f"Label not found: {args.label_name}", success=False)
        return _ResolvedApplyLabel(args=args, label=label)

    def describe(self, resolved: _ResolvedApplyLabel) -> ToolResult:
        """The plan headline — names whether the label already exists."""
        count = len(resolved.args.message_ids)
        if resolved.label:
            return ToolResult(
                message=f"apply label '{resolved.args.label_name}' to {count} email(s)."
            )
        return ToolResult(
            message=(
                f"create label '{resolved.args.label_name}' (it doesn't exist yet) "
                f"and apply it to {count} email(s)."
            )
        )

    async def apply(self, resolved: _ResolvedApplyLabel) -> ToolResult:
        """Create the label if it was missing, then apply it to the batch."""
        args = resolved.args
        label = resolved.label
        if not label:
            label = await self._client.create_label(args.label_name)
            if not label:
                return ToolResult(
                    message=f"Failed to create label: {args.label_name}",
                    success=False,
                )

        label_id = label.get("labelId", "")
        success = await self._client.apply_label(args.message_ids, label_id)
        if success:
            # Same bulk-endpoint shape as move_messages (#2768): one aggregate status
            # for the whole batch, no per-message confirmation to count.
            return ToolResult(
                message=f"Applied label '{args.label_name}' to the email(s).",
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to apply label '{args.label_name}'",
            success=False,
        )


class ListLabelsTool(Tool):
    """List available email labels."""

    name = "list_labels"
    description = (
        "List all available email labels in the user's mailbox. "
        "Returns label names and colors. Use this to see what labels exist "
        "before applying them to emails."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing labels"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List all labels."""
        try:
            labels = await self._client.get_labels()
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("Listing labels failed")
            return ToolResult(message=LIST_LABELS_FAILED.format(e=e), success=False)
        if not labels:
            return ToolResult(message="No labels found.")

        lines = [f"Found {len(labels)} label(s):\n"]
        for label in labels:
            name = label.get("displayName", "Unknown")
            color = label.get("color", "")
            lines.append(f"- {name} ({color})")
        return ToolResult(message="\n".join(lines))


class CreateEmailRuleTool(Tool):
    """Create a persistent email rule for automatic organization.

    Unlike the other mutating tools in this plugin, the one mutating call
    here is a write to Penny's OWN database (``db.email_rules.create``), not
    an external API — so, unlike ``ResolvingMutationTool`` (#2947), a plan
    doesn't need to skip the call entirely: it runs the EXACT same store
    call with ``plan=True``, which rolls the transaction back instead of
    committing it (mirroring ``Collection.write``/``_create_memory``, #2933/
    #2946), rather than a second, separately-written "describe what this
    would do" path."""

    name = "create_email_rule"
    description = (
        "Create a persistent email rule that will be automatically applied "
        "during scheduled email checks. Rules can match emails by sender, "
        "subject, or content, and can move emails to folders or apply labels. "
        "Example: Create a rule to move all emails from AWS to Accounting/Expenses/AWS.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable rule name (e.g., 'AWS invoices to expenses')",
            },
            "condition": {
                "type": "object",
                "description": (
                    "Rule condition. Supported fields: 'from' (sender email/domain), "
                    "'subject_contains' (text in subject), 'body_contains' (text in body)"
                ),
                "properties": {
                    "from": {
                        "type": "string",
                        "description": "Sender email or domain (partial match)",
                    },
                    "subject_contains": {
                        "type": "string",
                        "description": "Text in the subject (case-insensitive)",
                    },
                    "body_contains": {
                        "type": "string",
                        "description": "Text in the body (case-insensitive)",
                    },
                },
            },
            "action": {
                "type": "object",
                "description": (
                    "Rule action. Supported fields: 'move_to' (folder path), "
                    "'label' (label name to apply)"
                ),
                "properties": {
                    "move_to": {
                        "type": "string",
                        "description": "Folder path to move matching emails to",
                    },
                    "label": {
                        "type": "string",
                        "description": "Label name to apply to matching emails",
                    },
                },
            },
        },
        "required": ["name", "condition", "action"],
    }
    mutates = True
    args_model = CreateEmailRuleArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating email rule"

    def __init__(self, db: Database, provider: str) -> None:
        self._db = db
        self._provider = provider

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Create an email rule — under ``plan``, the store call runs
        identically but rolls back instead of committing (#2947)."""
        args = CreateEmailRuleArgs(**kwargs)

        self._db.email_rules.create(
            provider=self._provider,
            name=args.name,
            condition=args.condition.model_dump_json(by_alias=True, exclude_none=True),
            action=args.action.model_dump_json(by_alias=True, exclude_none=True),
            plan=args.plan,
        )

        result = ToolResult(
            message=(
                f"Email rule '{args.name}' saved.\n\n"
                f"Condition: {describe_fields(args.condition)}\n"
                f"Action: {describe_fields(args.action)}\n\n"
                "Note: saved rules aren't applied automatically yet — this stores the rule "
                "for future use."
            ),
            mutated=True,
        )
        return as_plan_result(result) if args.plan else result


class ListEmailRulesTool(Tool):
    """List all active email rules."""

    name = "list_email_rules"
    description = (
        "List all active email rules that are applied during scheduled email checks. "
        "Shows rule names, conditions, and actions."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing email rules"

    def __init__(self, db: Database, provider: str) -> None:
        self._db = db
        self._provider = provider

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List all email rules."""
        rules = self._db.email_rules.list_active(self._provider)

        if not rules:
            return ToolResult(message="No email rules configured.")

        lines = [f"Found {len(rules)} active email rule(s):\n"]
        for idx, rule in enumerate(rules, start=1):
            condition = EmailRuleCondition.model_validate_json(rule.condition)
            action = EmailRuleAction.model_validate_json(rule.action)
            lines.append(f"{idx}. **{rule.name}**")
            lines.append(f"   Condition: {describe_fields(condition)}")
            lines.append(f"   Action: {describe_fields(action)}")
            lines.append("")

        return ToolResult(message="\n".join(lines))
