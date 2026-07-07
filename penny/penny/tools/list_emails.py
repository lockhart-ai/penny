"""List emails tool — browse emails in a specific folder."""

from __future__ import annotations

import logging
from typing import Any

from penny.tools.base import Tool
from penny.tools.models import ListEmailsArgs, ToolResult
from penny.zoho import ZohoClient

logger = logging.getLogger(__name__)

# The folder listed when the caller doesn't name one — the single source of truth
# for both the ``execute`` default and the result narration.
DEFAULT_FOLDER = "Inbox"

NO_EMAILS_FOUND = (
    "No emails found in that folder. Confirm the folder name with `list_folders()`, or try "
    "a different folder."
)


class ListEmailsTool(Tool):
    """List emails from a specific folder."""

    name = "list_emails"
    description = (
        "List emails from one folder of the user's mailbox (Inbox, Sent, Drafts, Trash, "
        "Spam; defaults to Inbox). Returns email summaries — each with an id, subject, "
        "sender, date, and preview. Browse a folder here, then pass ids to "
        "`read_emails(email_ids=[<id>])` for full bodies. Call `list_folders()` first if "
        "you are unsure which folders exist."
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
    args_model = ListEmailsArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person recap of the folder listing (part of epic #1478).  Names
        the folder; ``NO_EMAILS_FOUND`` is the tool's own empty-folder body."""
        folder = arguments.get("folder") or DEFAULT_FOLDER
        if not result.success:
            return f"You tried to list the emails in {folder} but it didn't work:"
        if result.message == NO_EMAILS_FOUND:
            return f"You looked in {folder} but found no emails:"
        return f"You listed the emails in {folder}:"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List emails from a folder and return formatted summaries."""
        args = ListEmailsArgs(**kwargs)
        folder = args.folder

        results = await self._client.list_emails(folder_name=folder)
        if not results:
            return ToolResult(message=NO_EMAILS_FOUND)

        folder_name = folder or DEFAULT_FOLDER
        header = f"Found {len(results)} email(s) in {folder_name}:\n\n"
        return ToolResult(message=header + "\n\n".join(str(r) for r in results))
