"""Read emails tool — read full email content by ID."""

from __future__ import annotations

import logging
from typing import Any

from penny.constants import PennyConstants
from penny.email.protocol import EmailClient
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.models import ReadEmailsArgs, ToolResult

logger = logging.getLogger(__name__)

NO_EMAILS_TO_READ = (
    "No emails to read — pass one or more ids from a prior `search_emails(text=<keywords>)` "
    "or `list_emails(folder=<folder>)` call (those return the ids this tool expects)."
)

# The account a backend failure gets (#2368): what happened, with no order attached — the
# model reads the state and reasons from it, same as every other tool result.
MAILBOX_READ_FAILED = "That email couldn't be opened ({e}), so nothing was read from it."

# Named, not silent (#2768): the backend read some of the requested ids but not all of
# them (``ZohoClient.read_emails`` logs a warning and drops a failed id rather than
# raising), so the body says exactly which ones didn't open rather than letting the
# summary quietly cover a subset while the headline claims the full request.
_SOME_IDS_NOT_OPENED = "\n\n(Couldn't open {count} of the requested {noun}: {ids}.)"


class ReadEmailsTool(Tool):
    """Read the full body of one or more emails by ID."""

    name = "read_emails"
    # Nothing to read points back at the two ways of finding messages.
    advises = ("search_emails", "list_emails")
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
        emails were ACTUALLY opened (``result.read_email_ids``, #2768) — never the
        call's requested ``email_ids``, which the backend can silently drop one of
        (the client logs a warning and omits it rather than raising) — so a partial
        fetch is never claimed as a full one.  ``NO_EMAILS_TO_READ`` is the tool's
        own empty body (no matching ids)."""
        if not result.success:
            return "You tried to read your email but it didn't work:"
        if result.message == NO_EMAILS_TO_READ:
            return "You tried to open some email but there was nothing to read:"
        count = len(result.read_email_ids)
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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Read emails and summarize relevant content."""
        args = ReadEmailsArgs(**kwargs)
        try:
            emails = await self._client.read_emails(args.email_ids)
        except Exception as e:
            logger.exception("Mailbox read failed")
            return ToolResult(message=MAILBOX_READ_FAILED.format(e=e), success=False)
        if not emails:
            return ToolResult(message=NO_EMAILS_TO_READ)

        read_ids = [e.id for e in emails]
        raw_content = PennyConstants.SECTION_SEPARATOR.join(str(e) for e in emails)
        prompt = Prompt.EMAIL_SUMMARIZE_PROMPT.format(
            today=self._today,
            query=self._user_query,
            emails=raw_content,
        )
        response = await self._ollama.chat(
            [LlmMessage(role="user", content=prompt)], call_link=call_link.child()
        )
        message = (response.content or raw_content) + self._not_opened_note(
            args.email_ids, read_ids
        )
        return ToolResult(message=message, read_email_ids=read_ids)

    @staticmethod
    def _not_opened_note(requested: list[str], read: list[str]) -> str:
        """Which of the requested ids never came back from the backend — ``""``
        when every id was read, so the common full-fetch case renders byte-identical
        to before."""
        missing = [i for i in requested if i not in read]
        if not missing:
            return ""
        noun = "id" if len(missing) == 1 else "ids"
        return _SOME_IDS_NOT_OPENED.format(count=len(missing), noun=noun, ids=", ".join(missing))
