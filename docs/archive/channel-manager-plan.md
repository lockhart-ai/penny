# Channel Manager Implementation Plan

## Overview

Transform Penny from a single-channel architecture into a multi-channel system where a ChannelManager routes messages to/from multiple channels (Signal, Discord, Browser). All messages share a single conversation history. The single-user model is preserved; "devices" represent different ways the same user connects.

## Design Decisions

1. **ChannelManager implements MessageChannel ABC** — acts as a routing proxy. Everything holding a "channel" reference (agents, scheduler, commands) gets the manager. `send_message(recipient)` looks up the device table, finds the channel type, delegates to the right channel.

2. **Device table** — new table tracking all ways the user connects. One user (Penny is single-user), many devices. Fields: `id` (PK), `channel_type`, `identifier` (unique — phone number, discord ID, browser label), `label` (human-readable), `is_default`, `created_at`.

3. **MessageLog gets `device_id` FK** — replaces the current `sender` column. Used for reply routing. `sender` kept during transition, dropped in a future migration.

4. **Conversation history = all messages** — no sender filtering needed since single user. Cross-channel context works automatically.

5. **BrowserChannel** — evolve echo-only BrowserServer into a proper MessageChannel. First sidebar open prompts for a device label, stored in `browser.storage.local`, sent with every message. Penny auto-registers unknown devices.

6. **Proactive notifications go to default channel for MVP** — configurable routing tabled for later.

---

## Progress

- [x] Step 1: Device model + DeviceStore
- [x] Step 2: `device_id` FK on MessageLog
- [x] Step 3: Migration 0016
- [x] Step 4: IncomingMessage + channel_type on channels
- [x] Step 5: MessageStore dual-write
- [x] Step 6: ChannelManager
- [x] Step 7: BrowserChannel
- [x] Step 8: Factory + penny.py rewire
- [x] Step 9: Device-aware logging in handle_message
- [x] Step 10: ChannelType enum
- [x] Step 11: Tests (migration counts updated, all 393 passing)
- [ ] Step 12: Deferred cleanup (sender column drop — future migration)

---

## Step 1: Device model + DeviceStore

**Create** `database/device_store.py`, **modify** `database/models.py`, `database/database.py`

New `Device` SQLModel:
- `id`, `channel_type`, `identifier` (unique — phone number, discord ID, browser label), `label` (human-readable), `is_default` (bool), `created_at`

`DeviceStore` methods: `get_by_identifier()`, `get_by_id()`, `get_default()`, `get_all()`, `register()` (upsert — idempotent), `set_default()`.

Wired into `Database` facade as `db.devices`.

**Depends on**: nothing

---

## Step 2: `device_id` FK on MessageLog

**Modify** `database/models.py`

Add `device_id: int | None = Field(default=None, foreign_key="device.id")` to `MessageLog`.

Keep `sender` column during transition — both are written, queries migrate incrementally.

**Depends on**: Step 1

---

## Step 3: Migration 0016

**Create** `database/migrations/0016_add_device_table.py`

1. CREATE TABLE `device`
2. Seed Signal device from existing messages (`SELECT DISTINCT sender FROM messagelog WHERE direction='incoming'`; identifier starting with `+` → signal)
3. ALTER TABLE `messagelog` ADD COLUMN `device_id` + backfill from device lookup
4. Create index on `device_id`

`sender` column NOT dropped yet — future migration after full cutover.

**Depends on**: Steps 1-2

---

## Step 4: IncomingMessage + channel_type on channels

**Modify** `channels/base.py`, `channels/signal/channel.py`, `channels/discord/channel.py`

Add `channel_type` and `device_identifier` fields to `IncomingMessage`. Each channel's `extract_message()` sets these.

**Depends on**: nothing

---

## Step 5: MessageStore dual-write

**Modify** `database/message_store.py`

`log_message()` gains `device_id` parameter. New code writes both `sender` and `device_id`. Existing queries unchanged (they still work via `sender`).

**Depends on**: Step 2

---

## Step 6: ChannelManager

**Create** `channels/manager.py`

Implements `MessageChannel`. Key design:

- **Incoming**: each concrete channel handles its own receive loop + `handle_message` + reply. The manager is NOT in the incoming path.
- **Outgoing/proactive**: `send_message(recipient)` looks up device table → resolves channel type → delegates to concrete channel. This is what NotifyAgent, ScheduleExecutor, and startup announcements use.
- `listen()` → `asyncio.gather` all concrete channels
- `close()` → close all
- `set_scheduler()` / `set_command_context()` → forward to all concrete channels
- `register_channel(channel_type, channel)` — adds a channel to the routing table
- `sender_id` property → returns default channel's sender_id

**Depends on**: Steps 1-5

---

## Step 7: BrowserChannel

**Modify** `channels/browser/channel.py`, `channels/browser/models.py`, `channels/browser/__init__.py`

Evolve `BrowserServer` into `BrowserChannel(MessageChannel)`:

- Tracks connections in `dict[str, ServerConnection]` (device_label → ws)
- `_handle_connection()`: receives messages, auto-registers unknown devices via `db.devices.register()`, calls `self.handle_message()`
- `extract_message()` → `IncomingMessage` with `channel_type="browser"`
- `send_message(recipient)` → looks up ws connection by device label
- `send_typing()` → typing indicator to the right ws
- `listen()` → starts WebSocket server
- `close()` → shuts down server

**Depends on**: Step 6

---

## Step 8: Factory + penny.py rewire

**Modify** `channels/__init__.py`, `penny.py`

New `create_channel_manager()` factory: creates all configured channels (Signal/Discord based on config + Browser if enabled), registers them on the manager, seeds devices.

`penny.py` changes:
- `self.channel` becomes a `ChannelManager`
- Remove `_init_browser_server()` entirely — browser is now a channel
- `listen()` starts all channels via manager
- `shutdown()` closes all via manager
- Startup announcements and profile prompts use default channel

**Depends on**: Steps 6-7

---

## Step 9: Device-aware logging in handle_message

**Modify** `channels/base.py`

`_dispatch_to_agent()` resolves `device_id` from `message.device_identifier` via `db.devices.get_by_identifier()`, passes it to `log_message()`. Same for `send_response()` on outgoing.

**Depends on**: Steps 4-5

---

## Step 10: ChannelType enum

**Modify** `constants.py`

`ChannelType(StrEnum)` with `SIGNAL`, `DISCORD`, `BROWSER`. Old string constants in `channels/__init__.py` become aliases.

**Depends on**: nothing

---

## Step 11: Tests

- Seed test devices in `conftest.py` fixtures
- New `test_channel_manager.py` — routing, listen, close
- New `test_browser_channel.py` — extract, send, auto-registration
- New `test_device_store.py` — CRUD, upsert idempotency
- Update `test_message.py` — verify `device_id` on logged messages

**Depends on**: all previous steps

---

## Step 12: Deferred cleanup

- `UserStore.get_all_senders()` → query device table instead
- Drop `sender` column in a future migration
- Migrate remaining sender-filtered queries to device_id

---

## Execution Order

Steps 1, 4, 10 can be parallelized (no dependencies). Then 2 → 3 → 5 → 6 → 7 → 8 → 9 → 11.

```
[1: Device model]  ──→  [2: device_id FK]  ──→  [3: Migration]  ──→  [5: Dual-write]  ──┐
[4: IncomingMessage]  ────────────────────────────────────────────────────────────────────┤
[10: ChannelType enum]  ──────────────────────────────────────────────────────────────────┤
                                                                                          ↓
                                                                    [6: ChannelManager]  ──→  [7: BrowserChannel]  ──→  [8: Factory + penny.py]
                                                                                          ↓
                                                                    [9: Device-aware logging]
                                                                                          ↓
                                                                    [11: Tests]
                                                                                          ↓
                                                                    [12: Deferred cleanup]
```

## Potential Challenges

1. **ChannelManager as MessageChannel**: `extract_message()` is never called on the manager (each concrete channel extracts its own). `send_message()` and `send_typing()` must route correctly via device lookup. Since each concrete channel calls its own `handle_message` on itself (not on the manager), the incoming path works natively.

2. **set_command_context / set_scheduler propagation**: Must be forwarded to all concrete channels.

3. **Browser auto-registration race**: Multiple tabs could connect simultaneously. The `register()` upsert handles this — if the label already exists, it's a no-op.

4. **sender column backfill**: Migration must handle empty databases (fresh install). Seed step is skipped if no incoming messages exist.

5. **Test isolation**: Test fixtures must seed devices alongside test users, otherwise `device_id` lookups return None.
