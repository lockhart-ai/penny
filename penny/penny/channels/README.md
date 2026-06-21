# Channels Module

This module provides an abstraction layer for communication channels, allowing Penny to work with different messaging platforms.

## Architecture

The `MessageChannel` abstract base class defines the interface that all channel implementations must follow. This allows the agent to work with any messaging platform without being tightly coupled to a specific implementation.

## Directory Structure

Each channel implementation follows this structure:

```
penny/channels/
├── base.py                 # Abstract MessageChannel interface
├── signal/                 # Signal implementation
│   ├── __init__.py
│   ├── channel.py         # SignalChannel class
│   └── models.py          # Signal-specific Pydantic models
└── discord/                # Discord template
    ├── __init__.py
    ├── channel.py         # DiscordChannel class
    └── models.py          # Discord-specific Pydantic models
```

## Creating a New Channel

To add support for a new platform (e.g., Slack, Telegram):

1. Create a new subdirectory (e.g., `slack/`)
2. Create `channel.py` and implement the `MessageChannel` interface:

```python
# slack/channel.py
from penny.channels.base import MessageChannel, IncomingMessage

class SlackChannel(MessageChannel):
    async def _send_raw(self, recipient, message, attachments=None, quote_message=None) -> int | None:
        """Deliver a prepared message to the platform — the raw network send.

        Implement ONLY this. The base class's concrete `send_message` /
        `send_response` log every outgoing message to `messagelog` (so it
        surfaces in the `penny-messages` facade) before calling `_send_raw`,
        so no send can bypass the conversation record. Do not log here.
        """
        # Implementation here
        pass

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send typing indicator."""
        # Implementation here
        pass

    def get_connection_url(self) -> str:
        """Get connection URL/identifier."""
        # Return connection string
        pass

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract message from platform-specific data."""
        # Parse platform data and return IncomingMessage
        pass

    async def close(self) -> None:
        """Cleanup resources."""
        # Close connections
        pass
```

3. Create `models.py` for platform-specific Pydantic models:

```python
# slack/models.py
from pydantic import BaseModel

class SlackMessage(BaseModel):
    """Slack message structure."""
    channel: str
    user: str
    text: str
```

4. Create `__init__.py` to export your channel:

```python
# slack/__init__.py
from penny.channels.slack.channel import SlackChannel
from penny.channels.slack.models import SlackMessage

__all__ = ["SlackChannel", "SlackMessage"]
```

5. Optionally add to main `channels/__init__.py` for convenience:

```python
from penny.channels.slack import SlackChannel
```

6. Use it in the agent:

```python
from penny.channels import SlackChannel

channel = SlackChannel(...)
agent = PennyAgent(config, channel=channel)
```

## Reference Implementation: Signal

See the [`signal/`](./signal/) directory for a complete reference implementation:
- [`signal/channel.py`](./signal/channel.py) - SignalChannel implementation
- [`signal/models.py`](./signal/models.py) - Signal-specific Pydantic models
- [`signal/__init__.py`](./signal/__init__.py) - Module exports

## Discord configuration

To use the Discord channel integration you need:

- **A Discord bot token** (`DISCORD_BOT_TOKEN`)
- **A target channel ID** (`DISCORD_CHANNEL_ID`)

### Create and configure the bot

1. Create an application in the Discord Developer Portal.
2. In the **Bot** tab, click **Add Bot** and copy the bot token.
3. Enable **Message Content Intent** under **Privileged Gateway Intents**.

### Invite the bot to your server

1. In **OAuth2**, generate an invite URL with these scopes:
   - `bot`
   - `applications.commands`
2. Select the permissions you want the bot to have, then open the generated URL in your browser.
3. Authorise the bot for your server while logged into your Discord account.

### Allow the bot to read/write in the target channel

If the target channel is private:

1. Right-click the channel and choose **Edit Channel**.
2. Go to **Permissions**.
3. Click **+** next to **Roles/Members** and add the bot.

### Get the channel ID

1. In Discord, enable **Developer Mode** within the admin settings.
2. Right-click the channel and select **Copy Channel ID**.

### Environment variables

Set the following in your environment (see `.env.example`):

- `DISCORD_BOT_TOKEN="..."`
- `DISCORD_CHANNEL_ID=...`

