"""Email search command using Fastmail JMAP."""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.datetime_utils import current_datetime_line
from penny.jmap import JmapClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

logger = logging.getLogger(__name__)


class EmailCommand(Command):
    """Search email and answer questions about it."""

    name = "email"
    description = "Search your email and answer questions"
    help_text = (
        "Usage: /email <question>\n\n"
        "Ask a question about your email and Penny will search and read "
        "relevant messages to find the answer.\n\n"
        "Examples:\n"
        "• /email what packages am I expecting\n"
        "• /email when is my dentist appointment\n"
        "• /email any emails from mom this week"
    )

    def __init__(self, fastmail_api_token: str) -> None:
        self._fastmail_api_token = fastmail_api_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the email command."""
        prompt = args.strip()

        if not prompt:
            return CommandResult(text=PennyResponse.EMAIL_NO_QUERY_TEXT)

        jmap_client = JmapClient(
            self._fastmail_api_token,
            timeout=context.config.runtime.JMAP_REQUEST_TIMEOUT,
            max_body_length=int(context.config.runtime.EMAIL_BODY_MAX_LENGTH),
            search_limit=int(context.config.runtime.EMAIL_SEARCH_LIMIT),
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                SearchEmailsTool(jmap_client),
                ReadEmailsTool(
                    jmap_client, context.model_client, prompt, current_datetime_line(context.db)
                ),
            ]

            agent = Agent(
                system_prompt=Prompt.EMAIL_SYSTEM_PROMPT,
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
            logger.exception("Email search failed")
            return CommandResult(text=PennyResponse.EMAIL_ERROR.format(error=e))

        finally:
            if agent:
                await agent.close()
            await jmap_client.close()
