"""Read emails tool — read full email content by ID."""

from __future__ import annotations

import logging
from typing import Any

from penny.constants import PennyConstants
from penny.email.protocol import EmailClient
from penny.llm.client import LlmClient
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.models import ReadEmailsArgs, ToolResult

logger = logging.getLogger(__name__)

NO_EMAILS_TO_READ = (
    "No emails to read — pass one or more email IDs from a prior search_emails or "
    "list_emails call (those return the IDs this tool expects)."
)


class ReadEmailsTool(Tool):
    """Read the full body of one or more emails by ID."""

    name = "read_emails"
    description = (
        "Read the full content of one or more emails by their IDs. "
        "Use this after search_emails to get the complete bodies of relevant emails."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "email_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email IDs from search_emails results",
            },
        },
        "required": ["email_ids"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Reading emails"

    def __init__(
        self,
        email_client: EmailClient,
        ollama_client: LlmClient,
        user_query: str,
    ) -> None:
        self._client = email_client
        self._ollama = ollama_client
        self._user_query = user_query

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Read emails and summarize relevant content."""
        args = ReadEmailsArgs(**kwargs)
        email_ids = args.email_ids
        if not email_ids:
            return ToolResult(message=NO_EMAILS_TO_READ)
        emails = await self._client.read_emails(email_ids)
        if not emails:
            return ToolResult(message=NO_EMAILS_TO_READ)

        raw_content = PennyConstants.SECTION_SEPARATOR.join(str(e) for e in emails)
        prompt = Prompt.EMAIL_SUMMARIZE_PROMPT.format(
            query=self._user_query,
            emails=raw_content,
        )
        response = await self._ollama.chat([{"role": "user", "content": prompt}])
        return ToolResult(message=response.content or raw_content)
