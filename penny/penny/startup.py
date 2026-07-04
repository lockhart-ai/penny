"""Startup announcement logic for Penny."""

import logging
import os

from penny.database.database import Database
from penny.datetime_utils import current_datetime_line
from penny.llm.client import LlmClient
from penny.llm.models import LlmError
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


async def get_restart_message(db: Database, ollama_client: LlmClient) -> str:
    """
    Generate a casual restart announcement based on the latest git commit.

    Args:
        db: Database, for the current-date/time anchor (user timezone)
        ollama_client: Ollama client for transforming commit message

    Returns:
        A casual, first-person restart message (e.g., "I added a new command! /debug")
        or fallback PennyResponse.RESTART_FALLBACK if commit message unavailable
    """
    # Get commit message from environment variable (set at build time)
    commit_message = os.environ.get("GIT_COMMIT_MESSAGE", "").strip()

    if not commit_message or commit_message == "unknown":
        logger.info("No git commit message available, using fallback")
        return PennyResponse.RESTART_FALLBACK

    logger.info("Latest commit message: %s", commit_message)

    # Transform commit message into casual first-person announcement
    prompt = (
        f"{current_datetime_line(db)}\n\n"
        f"Transform this git commit message into a casual, first-person "
        f"announcement (one sentence, under 100 characters). "
        f"Keep it friendly and simple, like texting a friend.\n\n"
        f"Commit: {commit_message}\n\n"
        f"Examples:\n"
        f'- "fix: add /debug command" → "I added a new command! /debug"\n'
        f'- "feat: implement reminders" → "I added reminders! '
        f'You can ask me to remind you about stuff now"\n'
        f'- "fix: message handling bug" → "I fixed a bug with message handling"\n\n'
        f"Your announcement:"
    )

    try:
        response = await ollama_client.generate(
            prompt, agent_name="startup", prompt_type="startup_announcement"
        )
        announcement = response.content.strip()

        if not announcement or len(announcement) > 150:
            logger.warning("LLM response invalid, using fallback")
            return PennyResponse.RESTART_FALLBACK

        logger.info("Generated restart announcement: %s", announcement)
        return announcement

    except LlmError as e:
        logger.warning("LLM transformation failed: %s, using fallback", e)
        return PennyResponse.RESTART_FALLBACK
