"""Zoho Mail organisation tools — LLM-callable tools for the Zoho plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from penny.plugins.zoho.rule_models import (
    EmailRuleAction,
    EmailRuleCondition,
    describe_fields,
)
from penny.tools.base import Tool
from penny.tools.models import ToolResult

if TYPE_CHECKING:
    from penny.database import Database
    from penny.plugins.zoho.mail_client import ZohoClient

logger = logging.getLogger(__name__)


class MoveEmailsArgs(BaseModel):
    """Arguments for moving emails to a folder."""

    message_ids: list[str] = Field(description="List of email message IDs to move")
    folder_path: str = Field(description="Destination folder path (e.g., 'Clients/John Smith')")
    create_if_missing: bool = Field(default=True, description="Create folder if it doesn't exist")


class CreateFolderArgs(BaseModel):
    """Arguments for creating an email folder."""

    folder_path: str = Field(description="Folder path to create (e.g., 'Accounting/Expenses/AWS')")


class ApplyLabelArgs(BaseModel):
    """Arguments for applying a label to emails."""

    message_ids: list[str] = Field(description="List of email message IDs to label")
    label_name: str = Field(description="Label name to apply (e.g., 'completed')")
    create_if_missing: bool = Field(default=True, description="Create label if it doesn't exist")


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


class MoveEmailsTool(Tool):
    """Move emails to a folder, creating the folder if needed."""

    name = "move_emails"
    description = (
        "Move one or more emails to a destination folder. "
        "The folder path can include nested folders like 'Clients/John Smith' or "
        "'Accounting/Expenses/AWS'. Folders will be created if they don't exist. "
        "Use this after reading emails to organize them into appropriate folders."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Moving emails"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Move emails to a folder."""
        args = MoveEmailsArgs(**kwargs)

        if not args.message_ids:
            return ToolResult(
                message="No message IDs provided.",
                success=False,
            )

        folder = await self._client.get_folder_by_name(args.folder_path.split("/")[-1])

        if not folder and args.create_if_missing:
            folder = await self._client.create_nested_folder(args.folder_path)
            if not folder:
                return ToolResult(
                    message=f"Failed to create folder: {args.folder_path}",
                    success=False,
                )

        if not folder:
            return ToolResult(
                message=f"Folder not found: {args.folder_path}",
                success=False,
            )

        success = await self._client.move_messages(args.message_ids, folder.folder_id)
        if success:
            return ToolResult(
                message=(
                    f"Successfully moved {len(args.message_ids)} email(s) to '{args.folder_path}'"
                ),
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to move emails to '{args.folder_path}'",
            success=False,
        )


class CreateFolderTool(Tool):
    """Create an email folder with optional nesting."""

    name = "create_folder"
    description = (
        "Create a new email folder. Supports nested folder paths like "
        "'Clients/John Smith' or 'Accounting/Expenses/AWS'. "
        "Parent folders will be created automatically if they don't exist."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating folder"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create a folder."""
        args = CreateFolderArgs(**kwargs)

        folder = await self._client.create_nested_folder(args.folder_path)
        if folder:
            return ToolResult(
                message=f"Successfully created folder: {args.folder_path}",
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to create folder: {args.folder_path}",
            success=False,
        )


class ApplyLabelTool(Tool):
    """Apply a label to emails."""

    name = "apply_label"
    description = (
        "Apply a label to one or more emails. Labels help categorize emails "
        "without moving them. Common labels include 'completed', 'pending', "
        "'urgent', etc. The label will be created if it doesn't exist."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Applying label"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Apply a label to emails."""
        args = ApplyLabelArgs(**kwargs)

        if not args.message_ids:
            return ToolResult(
                message="No message IDs provided.",
                success=False,
            )

        label = await self._client.get_label_by_name(args.label_name)

        if not label and args.create_if_missing:
            label = await self._client.create_label(args.label_name)
            if not label:
                return ToolResult(
                    message=f"Failed to create label: {args.label_name}",
                    success=False,
                )

        if not label:
            return ToolResult(
                message=f"Label not found: {args.label_name}",
                success=False,
            )

        label_id = label.get("labelId", "")
        success = await self._client.apply_label(args.message_ids, label_id)
        if success:
            return ToolResult(
                message=(
                    f"Successfully applied label '{args.label_name}' to "
                    f"{len(args.message_ids)} email(s)"
                ),
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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List all labels."""
        labels = await self._client.get_labels()
        if not labels:
            return ToolResult(message="No labels found.")

        lines = [f"Found {len(labels)} label(s):\n"]
        for label in labels:
            name = label.get("displayName", "Unknown")
            color = label.get("color", "")
            lines.append(f"- {name} ({color})")
        return ToolResult(message="\n".join(lines))


class CreateEmailRuleTool(Tool):
    """Create a persistent email rule for automatic organization."""

    name = "create_email_rule"
    description = (
        "Create a persistent email rule that will be automatically applied "
        "during scheduled email checks. Rules can match emails by sender, "
        "subject, or content, and can move emails to folders or apply labels. "
        "Example: Create a rule to move all emails from AWS to Accounting/Expenses/AWS."
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
    args_model = CreateEmailRuleArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating email rule"

    def __init__(self, db: Database, provider: str) -> None:
        self._db = db
        self._provider = provider

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create an email rule."""
        args = CreateEmailRuleArgs(**kwargs)

        self._db.email_rules.create(
            provider=self._provider,
            name=args.name,
            condition=args.condition.model_dump_json(by_alias=True, exclude_none=True),
            action=args.action.model_dump_json(by_alias=True, exclude_none=True),
        )

        return ToolResult(
            message=(
                f"Email rule '{args.name}' saved.\n\n"
                f"Condition: {describe_fields(args.condition)}\n"
                f"Action: {describe_fields(args.action)}\n\n"
                "Note: saved rules aren't applied automatically yet — this stores the rule "
                "for future use."
            ),
            mutated=True,
        )


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

    async def execute(self, **kwargs: Any) -> ToolResult:
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
