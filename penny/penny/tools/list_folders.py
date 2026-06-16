"""List folders tool — show available email folders."""

from __future__ import annotations

import logging
from typing import Any

from penny.tools.base import Tool
from penny.tools.models import ToolOutcome
from penny.zoho import ZohoClient

logger = logging.getLogger(__name__)


class ListFoldersTool(Tool):
    """List available email folders."""

    name = "list_folders"
    description = (
        "List all available email folders in the user's mailbox. "
        "Returns folder names and types (Inbox, Sent, Drafts, etc.). "
        "Use this to discover what folders exist before listing emails from them."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> ToolOutcome:
        """List all folders and return formatted list."""
        folders = await self._client.get_folders()
        if not folders:
            return ToolOutcome(message="No folders found.")

        lines = [f"Found {len(folders)} folder(s):\n"]
        for folder in folders:
            lines.append(f"- {folder.folder_name} ({folder.folder_type})")

        return ToolOutcome(message="\n".join(lines))
