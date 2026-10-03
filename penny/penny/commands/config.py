"""The /config command — view and modify runtime configuration parameters."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.config_params import RUNTIME_CONFIG_PARAMS, get_params_by_group
from penny.responses import PennyResponse


class ConfigCommand(Command):
    """View and modify runtime configuration parameters."""

    name = "config"
    description = "View and modify runtime configuration parameters"
    help_text = (
        "View and modify runtime configuration parameters like timing settings. "
        "Configuration is stored in the database and takes effect immediately.\n\n"
        "**Usage**:\n"
        "• `/config` — List all available configuration parameters and their current values\n"
        "• `/config <key>` — Show the value of a specific configuration parameter\n"
        "• `/config <key> <value>` — Update a configuration parameter"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute config command."""
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        parts = args.strip().split(maxsplit=1)

        # Case 1: List all config
        if not args.strip():
            lines = [PennyResponse.CONFIG_HEADER, ""]

            for group_name, params in get_params_by_group():
                lines.append(PennyResponse.CONFIG_GROUP_HEADER.format(group=group_name))
                for param in params:
                    current_value = getattr(context.config.runtime, param.key)
                    lines.append(
                        PennyResponse.CONFIG_PARAM_DISPLAY.format(
                            key=param.key,
                            value=current_value,
                            description=param.description,
                        )
                    )
                lines.append("")

            lines.append(PennyResponse.CONFIG_FOOTER)
            return CommandResult(text="\n".join(lines))

        # Case 2: Get specific config
        if len(parts) == 1:
            key = parts[0].upper()
            if key not in RUNTIME_CONFIG_PARAMS:
                return CommandResult(text=PennyResponse.CONFIG_UNKNOWN_PARAM.format(key=key))

            param = RUNTIME_CONFIG_PARAMS[key]
            current_value = getattr(context.config.runtime, key)
            return CommandResult(text=f"**{key}**: {current_value} ({param.description})")

        # Case 3: Set config value
        key = parts[0].upper()
        value_str = parts[1]

        if key not in RUNTIME_CONFIG_PARAMS:
            return CommandResult(text=PennyResponse.CONFIG_UNKNOWN_PARAM.format(key=key))

        param = RUNTIME_CONFIG_PARAMS[key]

        # Validate value
        try:
            parsed_value = param.validator(value_str)
        except ValueError as e:
            return CommandResult(text=PennyResponse.CONFIG_INVALID_VALUE.format(key=key, error=e))

        # Store in database
        with Session(context.db.engine) as session:
            existing = session.exec(select(RuntimeConfig).where(RuntimeConfig.key == key)).first()

            if existing:
                existing.value = str(parsed_value)
                existing.updated_at = context.db.clock.now()
                session.add(existing)
            else:
                new_config = RuntimeConfig(
                    key=key,
                    value=str(parsed_value),
                    description=param.description,
                    updated_at=context.db.clock.now(),
                )
                session.add(new_config)

            session.commit()

        # Config changes take effect immediately via RuntimeParams DB lookup
        return CommandResult(text=PennyResponse.CONFIG_UPDATED.format(key=key, value=parsed_value))
