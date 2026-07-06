"""Command system for Penny."""

from penny.commands.base import Command, CommandRegistry
from penny.commands.config import ConfigCommand
from penny.commands.index import IndexCommand
from penny.commands.models import CommandContext, CommandError, CommandResult
from penny.commands.profile import ProfileCommand

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandContext",
    "CommandResult",
    "CommandError",
    "create_command_registry",
]


def create_command_registry() -> CommandRegistry:
    """Factory to create the registry with the built-in commands.

    Email retired onto the chat tool surface (epic #1445) — there are no
    config-gated commands left, so this takes no arguments.
    """
    registry = CommandRegistry()

    # Register IndexCommand with self-reference for listing commands
    commands_cmd = IndexCommand(registry)
    registry.register(commands_cmd)

    # Register other builtin commands
    registry.register(ConfigCommand())
    registry.register(ProfileCommand())

    return registry
