"""The /profile command — view or update user profile (name, location, DOB)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import dateparser
from pydantic import BaseModel, Field

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.datetime_utils import current_datetime_line, get_timezone
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ProfileUpdateParse(BaseModel):
    """Schema for parsing profile update arguments."""

    name: str | None = Field(
        default=None, description="User's name, or null if not specified in the input"
    )
    location: str | None = Field(
        default=None, description="User's location, or null if not specified in the input"
    )


class ProfileCreateParse(BaseModel):
    """Schema for parsing profile creation arguments."""

    name: str = Field(description="User's name")
    location: str = Field(description="User's location")
    date_of_birth: str = Field(
        description="User's date of birth in natural language (e.g., 'January 10, 1995')"
    )


class ProfileCommand(Command):
    """View or update your basic user profile (name, location, date of birth)."""

    name = "profile"
    description = "View or update your profile (name, location, DOB)"
    help_text = (
        "View your current profile or create/update your profile information.\n\n"
        "**Usage**:\n"
        "• `/profile` — View your current profile\n"
        "• `/profile <name> <location> <date of birth>` — Create profile (if new)\n"
        "• `/profile <name> <location>` — Update name/location (if profile exists)\n\n"
        "**Examples**:\n"
        "• `/profile sam denver march 5 1990` (initial setup)\n"
        "• `/profile sam denver` (update existing)\n"
        "• `/profile denver` (update location only)\n\n"
        "**Note**: Timezone is automatically derived from your location."
    )

    async def _parse_profile_create(
        self, args: str, ollama_client: Any, today: str
    ) -> ProfileCreateParse | None:
        """
        Parse profile creation arguments using LLM.

        Args:
            args: User input string
            ollama_client: Ollama client for structured parsing
            today: The current date/time anchor, so relative dates ("30 years
                ago") resolve against the right calendar day

        Returns:
            ProfileCreateParse if parsing succeeded, None otherwise
        """
        try:
            prompt = (
                f"{today}\n\n"
                f"Extract the user's name, location, and date of birth "
                f'from this input: "{args}"\n\n'
                "Return your response as JSON matching this schema:\n"
                "- name (string): user's name\n"
                "- location (string): user's location\n"
                "- date_of_birth (string): date of birth in natural language format "
                "(e.g., 'January 10, 1995')"
            )

            response = await ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ProfileCreateParse.model_json_schema(),
            )

            # Parse JSON response with Pydantic schema
            return ProfileCreateParse.model_validate_json(response.content)

        except Exception as e:
            logger.warning("Failed to parse profile creation args: %s", e)
            return None

    async def _parse_profile_update(
        self, args: str, ollama_client: Any, today: str
    ) -> ProfileUpdateParse | None:
        """
        Parse profile update arguments using LLM.

        Args:
            args: User input string
            ollama_client: Ollama client for structured parsing
            today: The current date/time anchor for grounding

        Returns:
            ProfileUpdateParse if parsing succeeded, None otherwise
        """
        try:
            prompt = (
                f"{today}\n\n"
                f'Extract the user\'s name and/or location from this input: "{args}"\n\n'
                "Return your response as JSON matching this schema:\n"
                "- name (string or null): user's name, or null if not mentioned\n"
                "- location (string or null): user's location, or null if not mentioned"
            )

            response = await ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ProfileUpdateParse.model_json_schema(),
            )

            # Parse JSON response with Pydantic schema
            return ProfileUpdateParse.model_validate_json(response.content)

        except Exception as e:
            logger.warning("Failed to parse profile update args: %s", e)
            return None

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute profile command."""

        args = args.strip()

        # No args - show current profile
        if not args:
            user_info = context.db.users.get_info(context.user)
            if not user_info:
                return CommandResult(text=PennyResponse.PROFILE_NO_PROFILE)

            # Format date of birth for display
            dob_formatted = datetime.strptime(user_info.date_of_birth, "%Y-%m-%d").strftime(
                "%B %d, %Y"
            )

            lines = [
                PennyResponse.PROFILE_HEADER,
                "",
                PennyResponse.PROFILE_NAME.format(name=user_info.name),
                PennyResponse.PROFILE_LOCATION.format(location=user_info.location),
                PennyResponse.PROFILE_TIMEZONE.format(timezone=user_info.timezone),
                PennyResponse.PROFILE_DOB.format(dob=dob_formatted),
            ]
            return CommandResult(text="\n".join(lines))

        user_info = context.db.users.get_info(context.user)

        # NEW PROFILE CREATION (no existing profile)
        if not user_info:
            # Use LLM to parse profile creation arguments
            parsed = await self._parse_profile_create(
                args, context.model_client, current_datetime_line(context.db)
            )
            if not parsed:
                return CommandResult(text=PennyResponse.PROFILE_CREATE_PARSE_ERROR)

            # Parse date of birth
            dob_date = dateparser.parse(
                parsed.date_of_birth, settings={"PREFER_DATES_FROM": "past"}
            )
            if not dob_date:
                return CommandResult(
                    text=PennyResponse.PROFILE_DATE_PARSE_ERROR.format(date=parsed.date_of_birth)
                )

            dob_formatted = dob_date.strftime("%Y-%m-%d")

            # Derive timezone from location
            timezone = await get_timezone(parsed.location)
            if not timezone:
                return CommandResult(
                    text=PennyResponse.PROFILE_TIMEZONE_ERROR.format(location=parsed.location)
                )

            # Save new profile
            context.db.users.save_info(
                sender=context.user,
                name=parsed.name,
                location=parsed.location,
                timezone=timezone,
                date_of_birth=dob_formatted,
            )

            response = PennyResponse.PROFILE_CREATED.format(name=parsed.name)

            # Prompt for interests if no preferences exist yet (onboarding)
            if not context.db.preferences.get_for_user(context.user):
                response += "\n\n" + PennyResponse.ONBOARDING_INTERESTS_PROMPT

            return CommandResult(text=response)

        # PROFILE UPDATE (existing profile)

        # Use LLM to parse profile update arguments
        parsed = await self._parse_profile_update(
            args, context.model_client, current_datetime_line(context.db)
        )
        if not parsed:
            return CommandResult(text=PennyResponse.PROFILE_UPDATE_PARSE_ERROR)

        # Use parsed values or keep existing
        new_name = parsed.name if parsed.name else user_info.name
        new_location = parsed.location if parsed.location else user_info.location

        # Derive new timezone from location if it changed
        if new_location != user_info.location:
            timezone = await get_timezone(new_location)
            if not timezone:
                return CommandResult(
                    text=PennyResponse.PROFILE_TIMEZONE_ERROR.format(location=new_location)
                )
        else:
            timezone = user_info.timezone

        # Update database
        context.db.users.save_info(
            sender=context.user,
            name=new_name,
            location=new_location,
            timezone=timezone,
            date_of_birth=user_info.date_of_birth,  # Keep existing DOB
        )

        # Build confirmation message
        changes = []
        if new_name != user_info.name:
            changes.append(PennyResponse.PROFILE_UPDATE_NAME.format(name=new_name))
        if new_location != user_info.location:
            changes.append(
                PennyResponse.PROFILE_UPDATE_LOCATION.format(
                    location=new_location, timezone=timezone
                )
            )

        if changes:
            change_text = " and ".join(changes)
            return CommandResult(text=PennyResponse.PROFILE_UPDATED.format(changes=change_text))
        else:
            return CommandResult(text=PennyResponse.PROFILE_UNCHANGED)
