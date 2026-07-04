"""Discord implementation of MessageChannel using discord.py."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import discord

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.discord.models import DiscordMessage, DiscordUser
from penny.constants import ChannelType

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)

# Sender ID for Discord bot messages in the database
DISCORD_SENDER_ID = "penny"

# Discord developer portal where privileged gateway intents are toggled.
DISCORD_DEVELOPER_PORTAL_URL = "https://discord.com/developers/applications"

# Actionable one-liner emitted when Discord rejects the connection because the
# Message Content Intent hasn't been enabled for the bot. Surfaced instead of a
# raw PrivilegedIntentsRequired traceback so a first-boot user knows the exact fix.
DISCORD_PRIVILEGED_INTENTS_ERROR = (
    "Discord refused the connection: enable the Message Content Intent for this bot "
    "under Bot -> Privileged Gateway Intents in the Discord developer portal "
    f"({DISCORD_DEVELOPER_PORTAL_URL}), then restart Penny."
)


class DiscordChannel(MessageChannel):
    """
    Discord channel implementation using discord.py.

    Unlike Signal which uses a simple WebSocket, Discord.py manages its own
    connection internally. This channel provides a message queue that the
    agent can consume, bridging the event-driven discord.py model with
    the pull-based MessageChannel interface.
    """

    def __init__(
        self,
        token: str,
        channel_id: str,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        """
        Initialize Discord channel.

        Args:
            token: Discord bot token
            channel_id: The channel ID to listen to and send messages in
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
            command_registry: Optional command registry for handling commands
        """
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self._token = token
        self.channel_id = channel_id

        # Set up Discord intents - need guilds to see channels and reactions
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.reactions = True

        # Create Discord client
        self.client = discord.Client(intents=intents)
        self._channel: discord.TextChannel | None = None
        self._ready = asyncio.Event()

        # Register event handlers
        self._setup_events()

        logger.info("Initialized Discord channel for channel_id=%s", channel_id)

    @property
    def sender_id(self) -> str:
        """Get the identifier for outgoing messages."""
        return DISCORD_SENDER_ID

    def _setup_events(self) -> None:
        """Set up Discord event handlers."""

        @self.client.event
        async def on_ready() -> None:
            await self._on_ready()

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            await self._on_message(message)

        @self.client.event
        async def on_reaction_add(reaction: discord.Reaction, user: discord.User) -> None:
            await self._on_reaction_add(reaction, user)

    async def _on_ready(self) -> None:
        """Handle the bot ready event — resolve target channel."""
        logger.info("Discord bot logged in as %s", self.client.user)
        self._log_guilds()

        channel = self.client.get_channel(int(self.channel_id))
        if channel and isinstance(channel, discord.TextChannel):
            self._channel = channel
            logger.info("Connected to channel: %s", channel.name)
        else:
            logger.error(
                "Could not find channel with ID: %s. "
                "Make sure the bot is invited to the server and has access to this channel.",
                self.channel_id,
            )

        self._ready.set()

    def _log_guilds(self) -> None:
        """Log available guilds and channels for debugging."""
        logger.info("Bot is in %d guild(s)", len(self.client.guilds))
        for guild in self.client.guilds:
            logger.info("  Guild: %s (ID: %s)", guild.name, guild.id)
            for ch in guild.text_channels[:5]:
                logger.info("    Channel: %s (ID: %s)", ch.name, ch.id)

    async def _on_message(self, message: discord.Message) -> None:
        """Handle an incoming Discord message."""
        if message.author == self.client.user:
            return
        if str(message.channel.id) != self.channel_id:
            return

        logger.debug(
            "Received Discord message from %s: %s",
            message.author.name,
            message.content[:100],
        )

        raw_data = self._build_message_data(message)
        await self.handle_message(raw_data)

    def _build_message_data(self, message: discord.Message) -> dict:
        """Build a Pydantic-validated dict from a Discord message."""
        author = DiscordUser(
            id=str(message.author.id),
            username=message.author.name,
            discriminator=message.author.discriminator,
            bot=message.author.bot,
            global_name=message.author.global_name,
        )
        discord_message = DiscordMessage(
            id=str(message.id),
            channel_id=str(message.channel.id),
            author=author,
            content=message.content,
            timestamp=message.created_at.isoformat(),
            guild_id=str(message.guild.id) if message.guild else None,
        )
        return discord_message.model_dump(by_alias=True)

    async def _on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Handle a reaction added to a message."""
        if user == self.client.user:
            return
        if str(reaction.message.channel.id) != self.channel_id:
            return

        logger.debug(
            "Received Discord reaction from %s: %s on message %s",
            user.name,
            str(reaction.emoji),
            reaction.message.id,
        )

        sender = f"{user.name}#{user.id}"
        incoming = IncomingMessage(
            sender=sender,
            content=str(reaction.emoji),
            channel_type=ChannelType.DISCORD,
            device_identifier=sender,
            is_reaction=True,
            reacted_to_external_id=str(reaction.message.id),
        )
        await self._handle_reaction(incoming)

    async def listen(self) -> None:
        """Start listening for messages via Discord gateway."""
        logger.info("Starting Discord client...")
        try:
            await self.client.start(self._token)
        except discord.errors.PrivilegedIntentsRequired as intents_error:
            # Surface an actionable one-liner (which intent to enable + portal link)
            # instead of dumping a raw traceback, then exit via the ConnectionError
            # path main() already handles for startup connectivity failures.
            logger.error(DISCORD_PRIVILEGED_INTENTS_ERROR)
            raise ConnectionError(DISCORD_PRIVILEGED_INTENTS_ERROR) from intents_error

    async def wait_until_ready(self) -> None:
        """Wait until the Discord gateway has resolved the target channel."""
        await self._ready.wait()
        logger.info("Discord client is ready, channel_id=%s", self.channel_id)

    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """
        Deliver a prepared message via Discord.

        Logging happens in the base ``_log_and_send`` chokepoint before this
        is called.

        Args:
            recipient: Channel ID (for Discord, we send to the configured channel)
            message: Message content
            attachments: Optional list of base64-encoded attachments (not yet implemented)
            quote_message: Optional message to quote-reply to (not yet implemented for Discord)

        Returns:
            Discord message ID on success, None on failure
        """
        try:
            await self._ready.wait()

            if not self._channel:
                logger.error("Discord channel not available")
                return None

            sent_message = await self._send_discord_message(self._channel, message)
            logger.info("Sent message to Discord channel (length: %d)", len(message))
            return int(sent_message.id) if sent_message else None

        except discord.HTTPException as e:
            logger.error("Failed to send Discord message: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error sending Discord message: %s", e)
            return None

    def _chunk_message(self, text: str, limit: int = 2000) -> list[str]:
        """Split text into chunks that fit within Discord's character limit."""
        if len(text) <= limit:
            return [text]
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    async def _send_discord_message(
        self, channel: discord.TextChannel, text: str
    ) -> discord.Message | None:
        """Send a message to a Discord channel, chunking if necessary."""
        sent_message: discord.Message | None = None
        for chunk in self._chunk_message(text):
            sent_message = await channel.send(chunk)
        return sent_message

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """
        Send a typing indicator via Discord.

        Args:
            recipient: Channel ID (unused, we use configured channel)
            typing: True to start typing (Discord typing lasts ~10 seconds)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not typing:
                # Discord doesn't have a "stop typing" API, it auto-expires
                return True

            await self._ready.wait()

            if not self._channel:
                logger.warning("Discord channel not available for typing indicator")
                return False

            await self._channel.typing()
            logger.debug("Sent typing indicator to Discord channel")
            return True

        except discord.HTTPException as e:
            logger.warning("Failed to send typing indicator: %s", e)
            return False
        except Exception as e:
            logger.warning("Unexpected error sending typing indicator: %s", e)
            return False

    def get_connection_url(self) -> str:
        """
        Get the connection identifier for Discord.

        Returns:
            A descriptive string (Discord manages its own gateway connection)
        """
        return f"discord-gateway:channel={self.channel_id}"

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """
        Extract a message from Discord event data.

        Args:
            raw_data: Raw message data from Discord event

        Returns:
            IncomingMessage if valid, None if should be ignored
        """
        try:
            # Parse and validate using Pydantic model
            message = DiscordMessage.model_validate(raw_data)

            # Ignore bot messages (own messages already filtered in on_message)
            if message.author.bot:
                logger.debug("Ignoring bot message from %s", message.author.username)
                return None

            content = message.content.strip()

            if not content:
                logger.debug("Ignoring empty message from %s", message.author.username)
                return None

            # Use username as sender for readability in logs/db
            sender = f"{message.author.username}#{message.author.id}"

            logger.info("Extracted Discord message - sender: %s, content: '%s'", sender, content)

            return IncomingMessage(
                sender=sender,
                content=content,
                channel_type=ChannelType.DISCORD,
                device_identifier=sender,
            )

        except Exception as e:
            logger.error("Failed to extract Discord message: %s", e)
            logger.debug("Raw data: %s", raw_data)
            return None

    async def close(self) -> None:
        """Stop listening and close Discord client."""
        logger.info("Closing Discord client...")
        await self.client.close()
        logger.info("Discord channel closed")
