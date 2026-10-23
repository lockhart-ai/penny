"""List folders tool — show available email folders."""

from __future__ import annotations

import logging
from typing import Any

from penny.llm.call_link import CallLink
from penny.plugins.zoho.mail_client import ZohoClient
from penny.tools.base import Tool
from penny.tools.models import NoArgs, ToolResult

logger = logging.getLogger(__name__)

# The account a backend failure gets (#2368): what happened, with no order attached — the
# model reads the state and reasons from it, same as every other tool result.
MAILBOX_FOLDERS_FAILED = (
    "The mailbox folders couldn't be listed ({e}), so nothing was read from the mailbox."
)


class ListFoldersTool(Tool):
    """List available email folders."""

    name = "list_folders"
    # A folder is named so its messages can be listed.
    advises = ("list_emails",)
    description = (
        "List every email folder in the user's mailbox. Returns each folder's name and "
        "type (Inbox, Sent, Drafts, etc.). Use this to discover what folders exist before "
        "`list_emails(folder=<folder name>)`."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    args_model = NoArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person recap of the folder-list call (part of epic #1478)."""
        if not result.success:
            return "You tried to list your email folders but it didn't work:"
        return "You looked at your email folders:"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List all folders and return formatted list."""
        try:
            folders = await self._client.get_folders()
        except Exception as e:
            logger.exception("Mailbox folder listing failed")
            return ToolResult(message=MAILBOX_FOLDERS_FAILED.format(e=e), success=False)
        if not folders:
            # Every mailbox has at least Inbox/Sent/Drafts, so an empty list is the
            # account being unreachable, not a legitimately empty one — the message
            # already said so; ``success`` now agrees (#2768) rather than reading as
            # a completed, if boring, listing.
            return ToolResult(
                message="No mail folders returned — this usually means the mail account isn't "
                "reachable or has no access right now. Let the user know rather than retrying.",
                success=False,
            )

        lines = [f"Found {len(folders)} folder(s):\n"]
        for folder in folders:
            lines.append(f"- {folder.folder_name} ({folder.folder_type})")

        return ToolResult(message="\n".join(lines))
