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
├── discord/                # Discord implementation
│   ├── __init__.py
│   ├── channel.py         # DiscordChannel class
│   └── models.py          # Discord-specific Pydantic models
└── ios/                    # iOS WebSocket + APNs implementation
    ├── __init__.py
    ├── apns.py            # APNs HTTP/2 client
    ├── channel.py         # IosChannel class
    └── models.py          # iOS WebSocket protocol models
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

## iOS channel

The iOS channel is designed for a native client with foreground WebSocket
delivery and background APNs notifications:

- The app connects to `ws://<host>:9091` by default.
- The app must send `register` first. If `IOS_PAIRING_TOKEN` is configured, the
  registration's `pairing_token` must match.
- Penny stores/updates the generic device row and the iOS APNs registration, then
  responds with `registered`.
- Outgoing Penny messages always go into `ios_outbox`.
- If the target device has an active WebSocket connection, Penny sends
  `outbox_changed`; the client should then send `pull_messages`.
- If the target device is disconnected, Penny sends an APNs alert preview with a
  summarized body and source hint, then the client pulls the durable outbox on
  next open.

### Client messages

- `register`: `device_id`, `label`, optional `pairing_token`,
  optional `device_secret`, optional `apns_token`, `apns_environment`,
  optional `app_version`
- `message`: `content`
- `pull_messages`: optional `limit`
- `ack_messages`: `ids`
- `heartbeat`

### Server messages

- `status`: connection or protocol error status
- `registered`: device id, default status, pending count
- `outbox_changed`: pending count hint
- `messages`: durable outbox rows
- `messages_acked`: ack count
- `typing`: typing indicator

### APNs configuration

Set these environment variables for background notifications:

- `IOS_APNS_TEAM_ID`
- `IOS_APNS_KEY_ID`
- `IOS_APNS_KEY_PATH`
- `IOS_BUNDLE_ID`
- `IOS_APNS_SANDBOX`

`IOS_APNS_KEY_PATH` must point to an Apple `.p8` APNs auth key inside the
container. The project mounts `./data` at `/penny/data`, so
`/penny/data/private/AuthKey_XXXX.p8` is a convenient location. `.p8` files are
gitignored.

### Diagnostics

When testing APNs from the iOS client, send a normal `message` whose content is
`send me a test push`, `test push`, or `send a test notification`. Penny bypasses
the chat agent and forces an APNs test notification to the registered device even
when the WebSocket is currently connected.

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
