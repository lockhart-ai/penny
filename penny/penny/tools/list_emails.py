"""List emails tool — browse emails in a specific folder."""

from __future__ import annotations

import logging
from typing import Any

from penny.tools.base import Tool
from penny.tools.models import ListEmailsArgs, ToolResult
from penny.zoho import ZohoClient

logger = logging.getLogger(__name__)

NO_EMAILS_FOUND = (
    "No emails found in that folder. Confirm the folder name with list_folders, or try a "
    "different folder."
)


class ListEmailsTool(Tool):
    """List emails from a specific folder."""

    name = "list_emails"
    description = (
        "List emails from a specific folder in the user's mailbox. "
        "Available folders include: Inbox, Sent, Drafts, Trash, Spam. "
        "Returns email summaries with IDs, subjects, senders, dates, and previews. "
        "Use this to browse a folder, then use read_emails to get full details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": (
                    "Name of the folder to list emails from. "
                    "Common folders: Inbox, Sent, Drafts, Trash, Spam. "
                    "Defaults to Inbox if not specified."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List emails from a folder and return formatted summaries."""
        args = ListEmailsArgs(**kwargs)
        folder = args.folder

        results = await self._client.list_emails(folder_name=folder)
        if not results:
            return ToolResult(message=NO_EMAILS_FOUND)

        folder_name = folder or "Inbox"
        header = f"Found {len(results)} email(s) in {folder_name}:\n\n"
        return ToolResult(message=header + "\n\n".join(str(r) for r in results))
