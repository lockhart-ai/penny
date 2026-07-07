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
    "No emails to read — pass one or more ids from a prior `search_emails(text=<keywords>)` "
    "or `list_emails(folder=<folder>)` call (those return the ids this tool expects)."
)


class ReadEmailsTool(Tool):
    """Read the full body of one or more emails by ID."""

    name = "read_emails"
    description = (
        "Read the full content of one or more emails by their ids. Run this after a search "
        "(`search_emails(text=<keywords>)`) or a folder listing to get the complete "
        "bodies — pass every relevant id in ONE call, not one at a time."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "email_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The ids to read, from a prior search_emails or list_emails result"
                ),
            },
        },
        "required": ["email_ids"],
    }
    args_model = ReadEmailsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Reading emails"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person recap of the read (part of epic #1478).  Names how many
        emails were opened from the call's ``email_ids``; ``NO_EMAILS_TO_READ`` is
        the tool's own empty body (no matching ids)."""
        if not result.success:
            return "You tried to read your email but it didn't work:"
        if result.message == NO_EMAILS_TO_READ:
            return "You tried to open some email but there was nothing to read:"
        count = len(arguments.get("email_ids") or [])
        noun = "email" if count == 1 else "emails"
        return f"You read {count} {noun}:"

    def __init__(
        self,
        email_client: EmailClient,
        ollama_client: LlmClient,
        user_query: str,
        today: str,
    ) -> None:
        self._client = email_client
        self._ollama = ollama_client
        self._user_query = user_query
        self._today = today

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Read emails and summarize relevant content."""
        args = ReadEmailsArgs(**kwargs)
        emails = await self._client.read_emails(args.email_ids)
        if not emails:
            return ToolResult(message=NO_EMAILS_TO_READ)

        raw_content = PennyConstants.SECTION_SEPARATOR.join(str(e) for e in emails)
        prompt = Prompt.EMAIL_SUMMARIZE_PROMPT.format(
            today=self._today,
            query=self._user_query,
            emails=raw_content,
        )
        response = await self._ollama.chat([{"role": "user", "content": prompt}])
        return ToolResult(message=response.content or raw_content)
