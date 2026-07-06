"""Command system for Penny."""

from typing import TYPE_CHECKING

from penny.commands.base import Command, CommandRegistry
from penny.commands.config import ConfigCommand
from penny.commands.dislike import DislikeCommand
from penny.commands.index import IndexCommand
from penny.commands.like import LikeCommand
from penny.commands.models import CommandContext, CommandError, CommandResult
from penny.commands.mute import MuteCommand
from penny.commands.profile import ProfileCommand
from penny.commands.schedule import ScheduleCommand
from penny.commands.undislike import UndislikeCommand
from penny.commands.unlike import UnlikeCommand
from penny.commands.unmute import UnmuteCommand
from penny.commands.unschedule import UnscheduleCommand

if TYPE_CHECKING:
    from penny.llm.image_client import OllamaImageClient
    from penny.zoho.models import ZohoCredentials

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandContext",
    "CommandResult",
    "CommandError",
    "create_command_registry",
]


def create_command_registry(
    image_model_client: OllamaImageClient | None = None,
    fastmail_api_token: str | None = None,
    zoho_credentials: ZohoCredentials | None = None,
) -> CommandRegistry:
    """
    Factory to create registry with builtin commands.

    Args:
        image_model_client: Optional image generation OllamaImageClient (required for draw command)
        fastmail_api_token: Optional Fastmail API token (required for email command)
        zoho_credentials: Optional ZohoCredentials for Zoho Mail API (required for zoho command)

    Returns:
        CommandRegistry with all builtin commands registered
    """
    registry = CommandRegistry()

    # Register IndexCommand with self-reference for listing commands
    commands_cmd = IndexCommand(registry)
    registry.register(commands_cmd)

    # Register other builtin commands
    registry.register(ConfigCommand())
    registry.register(ProfileCommand())
    registry.register(ScheduleCommand())
    registry.register(MuteCommand())
    registry.register(UnmuteCommand())
    registry.register(UnscheduleCommand())
    registry.register(LikeCommand())
    registry.register(UnlikeCommand())
    registry.register(DislikeCommand())
    registry.register(UndislikeCommand())

    # Register draw command if image model client is configured
    if image_model_client:
        from penny.commands.draw import DrawCommand

        registry.register(DrawCommand())

    # Register email command if Fastmail API token is configured
    if fastmail_api_token:
        from penny.commands.email import EmailCommand

        registry.register(EmailCommand(fastmail_api_token))

    # Register zoho command if Zoho credentials are configured
    if zoho_credentials:
        from penny.commands.zoho import ZohoCommand

        registry.register(
            ZohoCommand(
                zoho_credentials.client_id,
                zoho_credentials.client_secret,
                zoho_credentials.refresh_token,
            )
        )

    return registry
