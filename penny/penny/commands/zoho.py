"""Zoho email search command using Zoho Mail API."""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.datetime_utils import current_datetime_line
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool
from penny.zoho.client import ZohoClient

logger = logging.getLogger(__name__)


class ZohoCommand(Command):
    """Search Zoho email and answer questions about it."""

    name = "zoho"
    description = "Search your Zoho email and answer questions"
    help_text = (
        "Usage: /zoho <question>\n\n"
        "Ask a question about your Zoho email and Penny will search and read "
        "relevant messages to find the answer.\n\n"
        "Examples:\n"
        "• /zoho what packages am I expecting\n"
        "• /zoho when is my dentist appointment\n"
        "• /zoho any emails from mom this week"
    )

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the zoho email command."""
        prompt = args.strip()

        if not prompt:
            return CommandResult(text=PennyResponse.ZOHO_NO_QUERY_TEXT)

        zoho_client = ZohoClient(
            self._client_id,
            self._client_secret,
            self._refresh_token,
            timeout=context.config.runtime.JMAP_REQUEST_TIMEOUT,
            max_body_length=int(context.config.runtime.EMAIL_BODY_MAX_LENGTH),
            search_limit=int(context.config.runtime.EMAIL_SEARCH_LIMIT),
            list_limit=int(context.config.runtime.EMAIL_LIST_LIMIT),
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                SearchEmailsTool(zoho_client),
                ListEmailsTool(zoho_client),
                ListFoldersTool(zoho_client),
                ReadEmailsTool(
                    zoho_client, context.model_client, prompt, current_datetime_line(context.db)
                ),
                DraftEmailTool(zoho_client),
            ]

            agent = Agent(
                system_prompt=Prompt.ZOHO_SYSTEM_PROMPT,
                model_client=context.model_client,
                embedding_model_client=context.embedding_model_client,
                tools=tools,
                db=context.db,
                config=context.config,
                allow_repeat_tools=True,
            )

            response = await agent.run(prompt, max_steps=context.config.email_max_steps)
            return CommandResult(text=response.answer)

        except Exception as e:
            logger.exception("Zoho email search failed")
            return CommandResult(text=PennyResponse.ZOHO_ERROR.format(error=e))

        finally:
            if agent:
                await agent.close()
            await zoho_client.close()
