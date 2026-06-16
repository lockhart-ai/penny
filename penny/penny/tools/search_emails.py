"""Search emails tool — search the user's email inbox."""

from __future__ import annotations

import logging
from typing import Any

from penny.email.protocol import EmailClient
from penny.tools.base import Tool
from penny.tools.models import SearchEmailsArgs, ToolResult

logger = logging.getLogger(__name__)

NO_EMAILS_FOUND = (
    "No emails found matching that query. Try a broader or differently worded search "
    "(fewer terms, a sender or subject keyword), or use list_folders to confirm where to look."
)


class SearchEmailsTool(Tool):
    """Search emails by text, sender, subject, or date range."""

    name = "search_emails"
    description = (
        "Search the user's email inbox. Returns a list of matching email summaries "
        "with IDs, subjects, senders, dates, and previews. "
        "Use this to find relevant emails, then use read_emails to get full details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Full-text search query across subject, body, and sender",
            },
            "from_addr": {
                "type": "string",
                "description": "Filter by sender email address or name",
            },
            "subject": {
                "type": "string",
                "description": "Filter by subject line text",
            },
            "after": {
                "type": "string",
                "description": (
                    "Only emails after this date (ISO 8601, e.g., 2026-01-01T00:00:00Z)"
                ),
            },
            "before": {
                "type": "string",
                "description": "Only emails before this date (ISO 8601)",
            },
        },
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Searching emails"

    def __init__(self, email_client: EmailClient) -> None:
        self._client = email_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Search emails and return formatted summaries."""
        args = SearchEmailsArgs(**kwargs)
        results = await self._client.search_emails(**args.model_dump(exclude_none=True))
        if not results:
            return ToolResult(message=NO_EMAILS_FOUND)
        header = f"Found {len(results)} email(s):\n\n"
        return ToolResult(message=header + "\n\n".join(str(r) for r in results))
