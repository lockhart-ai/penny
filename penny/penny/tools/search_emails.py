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
    "(fewer terms, or a sender or subject keyword), or `list_folders()` to confirm where "
    "to look."
)


def _search_criteria(arguments: dict) -> str:
    """A short human phrase for the search terms the call used, for narration.

    Joins whichever filters the call supplied (text / sender / subject / date
    window) into one readable clause; falls back to a generic noun when the call
    carried no filters (an arg-validation failure still narrates)."""
    parts: list[str] = []
    if arguments.get("text"):
        parts.append(str(arguments["text"]))
    if arguments.get("from_addr"):
        parts.append(f"from {arguments['from_addr']}")
    if arguments.get("subject"):
        parts.append(f"subject {arguments['subject']}")
    if arguments.get("after"):
        parts.append(f"after {arguments['after']}")
    if arguments.get("before"):
        parts.append(f"before {arguments['before']}")
    return " · ".join(parts) if parts else "your recent mail"


class SearchEmailsTool(Tool):
    """Search emails by text, sender, subject, or date range."""

    name = "search_emails"
    description = (
        "Search the user's email by keyword, sender, subject, or date range. Returns "
        "matching email summaries — each with an id, subject, sender, date, and preview. "
        "Find candidates here, then pass their ids to `read_emails(email_ids=[<id>])` for "
        "the full bodies."
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
    args_model = SearchEmailsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Searching emails"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person recap of the search (part of epic #1478).  Branches on
        success and on whether anything matched (``NO_EMAILS_FOUND`` is the tool's
        own empty-result body), so a fruitless search narrates honestly."""
        if not result.success:
            return "You tried to search your email but it didn't work:"
        if result.message == NO_EMAILS_FOUND:
            return "You searched your email but found nothing matching:"
        return f'You searched your email for "{_search_criteria(arguments)}":'

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
