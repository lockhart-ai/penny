"""Draft email tool — compose and save email drafts."""

from __future__ import annotations

import logging
from typing import Any

from penny.tools.base import Tool
from penny.tools.models import DraftEmailArgs, ToolResult
from penny.zoho import ZohoClient

logger = logging.getLogger(__name__)


class DraftEmailTool(Tool):
    """Compose and save an email draft for user review."""

    name = "draft_email"
    description = (
        "Compose an email and save it as a draft for the user to review — it is saved to "
        "the Drafts folder for them to edit and send, and is NEVER sent automatically. Use "
        "this after `read_emails(email_ids=[<id>])` when the user asks you to reply."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of recipient email addresses",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "body": {
                "type": "string",
                "description": "Email body content (plain text)",
            },
            "cc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of CC recipient email addresses",
            },
        },
        "required": ["to", "subject", "body"],
    }
    args_model = DraftEmailArgs

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Save an email draft and return confirmation."""
        args = DraftEmailArgs(**kwargs)
        to_addresses = args.to
        subject = args.subject
        body = args.body
        cc_addresses = args.cc

        try:
            message_id = await self._client.draft_response(
                to_addresses=to_addresses,
                subject=subject,
                content=body,
                cc_addresses=cc_addresses,
            )

            if message_id:
                recipients = ", ".join(to_addresses)
                return ToolResult(
                    message=(
                        f"Draft saved successfully!\n\n"
                        f"To: {recipients}\n"
                        f"Subject: {subject}\n\n"
                        f"The draft has been saved to your Drafts folder for review before sending."
                    ),
                    mutated=True,
                )
            else:
                return ToolResult(
                    message="Draft was saved but could not confirm the message ID.",
                    mutated=True,
                )

        except Exception as e:
            logger.exception("Failed to save draft")
            return ToolResult(
                message=f"Could not save the draft — {e}. Verify the recipient addresses are "
                f"well-formed and the subject and body are non-empty, then try again; if it "
                f"keeps failing, tell the user the mail service is unavailable.",
                success=False,
            )
