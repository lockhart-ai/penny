"""Main agent loop for Penny."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

from penny.agents import (
    Agent,
    ChatAgent,
    Collector,
)
from penny.channels import MessageChannel, create_channel_manager
from penny.channels.browser import BrowserChannel
from penny.channels.ios.channel import IosChannel
from penny.channels.manager import ChannelManager
from penny.channels.permission_manager import PermissionManager
from penny.channels.signal.channel import SignalChannel
from penny.commands import create_command_registry
from penny.config import Config, setup_logging
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.migrate import migrate
from penny.database.models import MemoryEntry
from penny.email.protocol import EmailClient
from penny.jmap import JmapClient
from penny.llm.client import LlmClient
from penny.llm.embeddings import serialize_embedding
from penny.llm.image_client import OllamaImageClient
from penny.llm.models import LlmError
from penny.preflight import Preflight, PreflightError
from penny.responses import PennyResponse
from penny.scheduler import (
    BackgroundScheduler,
    PeriodicSchedule,
    Schedule,
)
from penny.scheduler.send_queue_drainer import SendQueueDrainer
from penny.startup import get_restart_message
from penny.tools import Tool
from penny.tools.draft_email import DraftEmailTool
from penny.tools.list_emails import ListEmailsTool
from penny.tools.list_folders import ListFoldersTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool
from penny.zoho import ZohoClient
from penny.zoho.models import ZohoCredentials

logger = logging.getLogger(__name__)

# Builds the chat agent's email tool surface for one turn, given the current
# user message and the dated datetime line ``read_emails`` summarises against.
EmailToolsBuilder = Callable[[str, str], list[Tool]]


class Penny:
    """AI agent powered by local LLM inference via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize Penny — summary method."""
        self.config = config
        self.start_time = datetime.now()
        self._init_database(config)
        self._init_llm_clients(config)
        self._init_email(config)
        self._init_agents(config)
        self._init_commands(config)
        self._init_channel(config, channel)
        self._init_scheduler(config)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_database(self, config: Config) -> None:
        """Set up database, create tables, then run migrations.

        Order matters: ``migrate()`` is a no-op when the DB file doesn't
        exist (fresh deploy), so ``create_tables()`` must run first to
        materialise the schema. Migrations then apply on top — including
        data-insert migrations like 0026 that seed system log memories.
        """
        self.db = Database(config.db_path, runtime=config.runtime)
        self.db.create_tables()
        migrate(config.db_path)
        config.runtime._db = self.db

    def _create_llm_client(
        self,
        model: str,
        db: Database | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> LlmClient:
        """Create an LlmClient with standard configuration."""
        return LlmClient(
            api_url=api_url or self.config.llm_api_url,
            model=model,
            db=db if db is not None else self.db,
            max_retries=self.config.llm_max_retries,
            retry_delay=self.config.llm_retry_delay,
            api_key=api_key or self.config.llm_api_key,
            timeout=self.config.llm_timeout,
        )

    def _init_llm_clients(self, config: Config) -> None:
        """Create shared LLM model clients."""
        self.model_client = self._create_llm_client(config.llm_model)
        self.vision_model_client = (
            self._create_llm_client(
                config.llm_vision_model,
                api_url=config.llm_vision_api_url,
                api_key=config.llm_vision_api_key,
            )
            if config.llm_vision_model
            else None
        )
        # Embedding model is a required prerequisite (validated at config load),
        # so the client is always constructed — memory dedup and similarity
        # recall never run in a degraded, embedding-less mode.
        self.embedding_model_client = self._create_llm_client(
            config.llm_embedding_model,
            api_url=config.llm_embedding_api_url,
            api_key=config.llm_embedding_api_key,
        )
        self.image_client = (
            OllamaImageClient(
                api_url=config.image_api_url,
                model=config.llm_image_model,
                max_retries=config.llm_max_retries,
                retry_delay=config.llm_retry_delay,
            )
            if config.llm_image_model
            else None
        )

    def _init_email(self, config: Config) -> None:
        """Build the config-gated email client + the chat-surface tool builder.

        Fastmail (``JmapClient``) and Zoho (``ZohoClient``) both satisfy the
        ``EmailClient`` protocol.  Fastmail exposes search + read; Zoho adds
        folder listing and draft composition, so its chat surface carries all
        five tools.  When both are configured Fastmail wins — a single user has
        one mailbox in practice.  The client is long-lived (created here, closed
        in ``shutdown``); the builder wraps it fresh each turn with the current
        message + date so ``read_emails`` can summarise against the question.
        """
        self.email_client: EmailClient | None = None
        self.email_tools_builder: EmailToolsBuilder | None = None
        if config.fastmail_api_token:
            jmap_client = JmapClient(
                config.fastmail_api_token,
                timeout=config.runtime.JMAP_REQUEST_TIMEOUT,
                max_body_length=int(config.runtime.EMAIL_BODY_MAX_LENGTH),
                search_limit=int(config.runtime.EMAIL_SEARCH_LIMIT),
            )
            self.email_client = jmap_client
            self.email_tools_builder = self._fastmail_tools_builder(jmap_client)
            return
        credentials = self._get_zoho_credentials(config)
        if credentials:
            zoho_client = ZohoClient(
                credentials.client_id,
                credentials.client_secret,
                credentials.refresh_token,
                timeout=config.runtime.JMAP_REQUEST_TIMEOUT,
                max_body_length=int(config.runtime.EMAIL_BODY_MAX_LENGTH),
                search_limit=int(config.runtime.EMAIL_SEARCH_LIMIT),
                list_limit=int(config.runtime.EMAIL_LIST_LIMIT),
            )
            self.email_client = zoho_client
            self.email_tools_builder = self._zoho_tools_builder(zoho_client)

    def _fastmail_tools_builder(self, client: EmailClient) -> EmailToolsBuilder:
        """Fastmail's chat surface: search + read (JMAP has no folders/drafts)."""

        def build(user_query: str, today: str) -> list[Tool]:
            return [
                SearchEmailsTool(client),
                ReadEmailsTool(client, self.model_client, user_query, today),
            ]

        return build

    def _zoho_tools_builder(self, client: ZohoClient) -> EmailToolsBuilder:
        """Zoho's chat surface: search + read + folder listing + draft."""

        def build(user_query: str, today: str) -> list[Tool]:
            return [
                SearchEmailsTool(client),
                ReadEmailsTool(client, self.model_client, user_query, today),
                ListEmailsTool(client),
                ListFoldersTool(client),
                DraftEmailTool(client),
            ]

        return build

    def _init_agents(self, config: Config) -> None:
        """Create chat agent + collector dispatcher + schedule executor.

        The single ``Collector`` covers everything previously handled by
        the per-collection extractor agents AND the bespoke thinking +
        notify agents.  Each tick the dispatcher reads the memory table,
        picks the most-overdue collection with an extraction_prompt, and
        runs the agent loop bound to that target.
        """
        self.chat_agent = ChatAgent(
            model_client=self.model_client,
            db=self.db,
            config=config,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
            image_client=self.image_client,
            email_tools_builder=self.email_tools_builder,
        )
        self.collector = Collector(
            model_client=self.model_client,
            db=self.db,
            config=config,
            embedding_model_client=self.embedding_model_client,
        )
        self.chat_agent.set_collector(self.collector)
        # Deterministic task (no LLM) that delivers queued send_message output
        # once the autonomous-send cooldown clears.
        self.send_queue_drainer = SendQueueDrainer(db=self.db, config=config)

    def _init_commands(self, config: Config) -> None:
        """Create command registry.

        Email is no longer a command — the ``/email`` + ``/zoho`` slash commands
        retired onto the chat tool surface (epic #1445); the mailbox client is
        built in ``_init_email`` and driven from natural language instead.
        """
        self.command_registry = create_command_registry()

    def _get_zoho_credentials(self, config: Config) -> ZohoCredentials | None:
        """Get Zoho credentials if all required values are configured."""
        if config.zoho_api_id and config.zoho_api_secret and config.zoho_refresh_token:
            return ZohoCredentials(
                client_id=config.zoho_api_id,
                client_secret=config.zoho_api_secret,
                refresh_token=config.zoho_refresh_token,
            )
        return None

    def _init_channel(self, config: Config, channel: MessageChannel | None) -> None:
        """Create channel manager and connect agents that send notifications."""
        self.channel = channel or create_channel_manager(
            config=config,
            message_agent=self.chat_agent,
            db=self.db,
            command_registry=self.command_registry,
        )
        self.chat_agent.set_channel(self.channel)
        self.send_queue_drainer.set_channel(self.channel)
        # Collector needs the channel so a notify-shaped cycle (a collection whose
        # ``notify`` flag drives the run-time notify suffix, #1557) can call
        # send_message to tell the user about a new find.
        self.collector.set_channel(self.channel)
        if isinstance(self.channel, IosChannel):
            self.collector._progress_factory = self.channel.make_background_progress_callback
        self._wire_browser_tools(config)

    def _wire_browser_tools(self, config: Config) -> None:
        """Connect browser tools to agents when a browser channel is available."""
        if not isinstance(self.channel, ChannelManager):
            return
        browser_ch = self.channel.get_channel(ChannelType.BROWSER)
        ios_ch = self.channel.get_channel(ChannelType.IOS)
        if not isinstance(browser_ch, BrowserChannel) and not isinstance(ios_ch, IosChannel):
            return

        perm_mgr = PermissionManager(db=self.db, channel_manager=self.channel, config=config)
        if isinstance(browser_ch, BrowserChannel):
            browser_ch.set_permission_manager(perm_mgr)
            # Let the addon run a collection's extractor on demand.
            browser_ch.set_collector(self.collector)
        if isinstance(ios_ch, IosChannel):
            ios_ch.set_permission_manager(perm_mgr)
            ios_ch.set_collector(self.collector)
            self.collector._progress_factory = ios_ch.make_background_progress_callback
        signal_ch = self.channel.get_channel(ChannelType.SIGNAL)
        if isinstance(signal_ch, SignalChannel):
            signal_ch.set_permission_manager(perm_mgr)

        if not isinstance(browser_ch, BrowserChannel):
            return

        # Browse provider — agents build fresh BrowseTools each cycle.
        def browse_provider():
            if not browser_ch.has_browser_connection:
                return None
            return browser_ch.send_tool_request, perm_mgr

        self.chat_agent._browse_provider = browse_provider
        self.collector._browse_provider = browse_provider
        self.collector._on_tool_start_factory = browser_ch.make_background_tool_callback

    def _init_scheduler(self, config: Config) -> None:
        """Create background scheduler — send-queue drainer + collector dispatcher.

        The Collector is a single idle-gated schedule that ticks fast
        (COLLECTOR_TICK_INTERVAL).  Each tick it picks the most-overdue
        collection from ``memory`` (per-row ``collector_interval_seconds``)
        and runs that collection's extraction prompt.  Idle gating keeps
        collector work out of the way during active conversation; the
        store fills up "between conversations".
        """
        schedules: list[Schedule] = [
            # Drain before the collector so a queued message is delivered
            # promptly once its cooldown clears, rather than waiting behind a
            # collection cycle.  Idle-gated: queued autonomous messages never
            # interrupt an active conversation.
            PeriodicSchedule(
                agent=self.send_queue_drainer,
                interval=lambda: PennyConstants.SEND_QUEUE_DRAIN_INTERVAL,
                requires_idle=True,
            ),
            PeriodicSchedule(
                agent=self.collector,
                interval=lambda: config.runtime.COLLECTOR_TICK_INTERVAL,
                requires_idle=True,
            ),
        ]
        self.scheduler = BackgroundScheduler(
            schedules=schedules,
            idle_threshold=lambda: config.runtime.IDLE_SECONDS,
            tick_interval=config.scheduler_tick_interval,
        )
        self._connect_scheduler(config)

    def _connect_scheduler(self, config: Config) -> None:
        """Connect scheduler to channel and set command context."""
        self.channel.set_scheduler(self.scheduler)
        self.channel.set_command_context(
            config=config,
            channel_type=config.channel_type,
            start_time=self.start_time,
            model_client=self.model_client,
            embedding_model_client=self.embedding_model_client,
        )

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.scheduler.stop()

    async def _run_preflight(self) -> None:
        """Run the startup setup-health checks; abort on a hard prerequisite failure.

        Consolidates the startup prerequisite checks (LLM endpoint + chat model,
        embedding model, vision/image models, browser addon, primary-channel
        routing) into one legible log summary. Hard failures raise
        ``PreflightError`` — caught in ``main()`` and surfaced in ``penny.log``
        before exiting — instead of letting every downstream call fail opaquely.
        """
        report = await self._build_preflight().run()
        report.log(logger)
        if report.has_failures:
            raise PreflightError(report.failure_summary())

    def _build_preflight(self) -> Preflight:
        """Assemble the preflight with a snapshot of the current channel/routing facts."""
        browser = self._browser_channel()
        return Preflight(
            config=self.config,
            model_client=self.model_client,
            embedding_client=self.embedding_model_client,
            vision_client=self.vision_model_client,
            image_client=self.image_client,
            browser_enabled=self.config.browser_enabled,
            browser_connected=bool(browser and browser.has_browser_connection),
            configured_channel_type=self.config.channel_type,
            resolved_channel_type=self._resolved_channel_type(),
        )

    def _browser_channel(self) -> BrowserChannel | None:
        """The registered browser channel, if the channel manager has one."""
        if not isinstance(self.channel, ChannelManager):
            return None
        channel = self.channel.get_channel(ChannelType.BROWSER)
        return channel if isinstance(channel, BrowserChannel) else None

    def _resolved_channel_type(self) -> str | None:
        """Channel type proactive sends will route to (for the routing preflight)."""
        if isinstance(self.channel, ChannelManager):
            return self.channel.default_channel_type
        return self.config.channel_type

    async def _backfill_memory_embeddings(self, batch_limit: int) -> int:
        """Backfill memory entries missing embeddings. Returns count embedded.

        Covers the unified memory framework (skills, ``user-messages``, any
        migration-seeded collection content).  Scoped by the store to
        non-archived, non-``off`` memories so recall-relevant entries get
        vectors while bulk ``off`` logs (``collector-runs``) are skipped.
        Fills whichever vector each entry is missing — content, key, or both —
        so a keyed entry whose key never embedded is matchable by
        ``read_similar``, not just one whose content is null.
        """
        total = 0
        while True:
            entries = self.db.memories.get_entries_without_embeddings(limit=batch_limit)
            if not entries:
                break
            try:
                await self._embed_memory_entry_batch(entries)
                total += len(entries)
            except LlmError as e:
                logger.warning("Startup embedding backfill failed for memory entries: %s", e)
                break
        return total

    async def _embed_memory_entry_batch(self, entries: list[MemoryEntry]) -> None:
        """Embed only the vector(s) each selected entry is missing, then persist.

        Content is (re-)embedded only for entries lacking a content vector; the
        key is embedded only for keyed entries lacking a key vector — so a keyed
        entry that already carries its content but not its key gets the key
        filled without needlessly re-embedding the content it has.
        """
        content_items: list[tuple[int, str]] = []
        key_items: list[tuple[int, str]] = []
        for entry in entries:
            entry_id = entry.id
            if entry_id is None:
                continue
            if entry.content_embedding is None:
                content_items.append((entry_id, entry.content))
            key = entry.key
            if key is not None and entry.key_embedding is None:
                key_items.append((entry_id, key))
        content_by_id = await self._embed_by_id(content_items)
        key_by_id = await self._embed_by_id(key_items)
        for entry in entries:
            entry_id = entry.id
            if entry_id is None:
                continue
            self.db.memories.set_entry_embeddings(
                entry_id,
                key_embedding=key_by_id.get(entry_id),
                content_embedding=content_by_id.get(entry_id),
            )

    async def _embed_by_id(self, items: list[tuple[int, str]]) -> dict[int, list[float]]:
        """Embed each ``(id, text)`` pair in one batched call; ``{}`` when empty."""
        if not items:
            return {}
        vectors = await self.embedding_model_client.embed([text for _, text in items])
        return {item_id: vec for (item_id, _), vec in zip(items, vectors, strict=True)}

    async def _backfill_message_embeddings(self, batch_limit: int) -> int:
        """Backfill ``messagelog`` rows missing a content embedding.

        The user/penny message logs are read facades over ``messagelog``, and
        ``read_similar`` over them ranks on ``messagelog.embedding`` — so any
        table with an embedding column gets vectorized here (no row is copied
        from another table).  Idempotent: a row keeps its embedding once set.
        """
        total = 0
        while True:
            messages = self.db.messages.messages_without_embeddings(limit=batch_limit)
            if not messages:
                break
            try:
                vectors = await self.embedding_model_client.embed([m.content for m in messages])
                for message, vector in zip(messages, vectors, strict=True):
                    if message.id is not None:
                        self.db.messages.set_embedding(message.id, serialize_embedding(vector))
                total += len(messages)
            except LlmError as e:
                logger.warning("Startup embedding backfill failed for messages: %s", e)
                break
        return total

    async def _backfill_description_embeddings(self, batch_limit: int) -> int:
        """Backfill memory description anchors missing an embedding.

        The stage-1 routing gate compares the conversation against each
        ``relevant`` memory's ``description_embedding``.  Descriptions seeded
        by migrations have no vector (migrations can't embed), so vectorize
        them at startup — scoped by the store to active, routable memories.
        Returns count embedded.
        """
        total = 0
        while True:
            memories = self.db.memories.get_memories_without_description_embedding(
                limit=batch_limit
            )
            if not memories:
                break
            try:
                vecs = await self.embedding_model_client.embed([m.description for m in memories])
                for memory, vec in zip(memories, vecs, strict=True):
                    self.db.memories.set_description_embedding(memory.name, vec)
                total += len(memories)
            except LlmError as e:
                logger.warning("Startup embedding backfill failed for descriptions: %s", e)
                break
        return total

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Channel: %s (sender_id=%s)", self.config.channel_type, self.channel.sender_id)
        logger.info("Ollama model: %s", self.config.llm_model)
        if self.config.llm_vision_model:
            logger.info("Ollama model: %s (vision)", self.config.llm_vision_model)
        if self.config.llm_image_model:
            logger.info("Ollama model: %s (image generation)", self.config.llm_image_model)

        # Validate channel connectivity before starting
        await self.channel.validate_connectivity()

        await self._run_preflight()
        await self._run_startup_backfills()

        listen_task = asyncio.create_task(self.channel.listen(), name="channel.listen")
        scheduler_task = asyncio.create_task(self.scheduler.run(), name="scheduler.run")
        startup_task = asyncio.create_task(
            self._run_startup_notifications(), name="startup.notifications"
        )
        tasks = {listen_task, scheduler_task, startup_task}
        try:
            while True:
                done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                if scheduler_task in done:
                    logger.info("Scheduler stopped; shutting down channel listener")
                    break
                if listen_task in done:
                    logger.info("Channel listener stopped; shutting down scheduler")
                    break
                tasks -= done
        finally:
            await self.shutdown()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_startup_backfills(self) -> None:
        """Vectorize any embedding-less rows across the embedding-bearing tables.

        The embedding model is a required prerequisite, so this always runs —
        memory entries, description anchors, and message rows all get vectors at
        startup rather than lazily.
        """
        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)
        total_entries = await self._backfill_memory_embeddings(batch_limit)
        if total_entries:
            logger.info("Startup embedding backfill complete: %d memory entries", total_entries)
        total_descriptions = await self._backfill_description_embeddings(batch_limit)
        if total_descriptions:
            logger.info("Startup embedding backfill complete: %d descriptions", total_descriptions)
        total_messages = await self._backfill_message_embeddings(batch_limit)
        if total_messages:
            logger.info("Startup embedding backfill complete: %d messages", total_messages)

    async def _run_startup_notifications(self) -> None:
        """Send startup notifications after the outgoing channel is ready."""
        try:
            await self.channel.wait_until_ready()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Startup notifications skipped; channel never became ready: %s", e)
            return

        await self._send_startup_announcement()
        await self._prompt_for_missing_profiles()

    async def _send_startup_announcement(self) -> None:
        """Send a startup announcement to the user's default device."""
        try:
            sender = self.db.users.get_primary_sender()
            if not sender:
                logger.info("No user profile found for startup announcement")
                return

            # Only announce if the user has chatted before (not a fresh profile).
            # This same gate covers the iOS push path so a fresh install (or the
            # watcher's per-deploy restarts before any conversation) can't spam a
            # notification.
            user_messages = self.db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
            if user_messages is None or not user_messages.newest_entries(k=1):
                logger.info("No message history yet, skipping startup announcement")
                return

            if await self._send_ios_operational_announcement():
                return

            restart_msg = await get_restart_message(self.db, self.model_client)
            announcement = f"👋 {restart_msg}"

            logger.info("Sending startup announcement to %s", sender)
            await self.channel.send_message(sender, announcement)
        except Exception as e:
            logger.warning("Failed to send startup announcement: %s", e)

    async def _send_ios_operational_announcement(self) -> bool:
        """Send a simple startup notification when iOS is the default channel."""
        default = self.db.devices.get_default()
        if default is None or default.channel_type != ChannelType.IOS:
            return False
        logger.info("Sending iOS operational startup announcement to %s", default.identifier)
        await self.channel.send_message(default.identifier, "Penny is operational.")
        return True

    async def _prompt_for_missing_profiles(self) -> None:
        """Prompt the user if they don't have a profile set up yet (single-user)."""
        try:
            if self.db.users.get_primary_sender():
                return  # Profile exists, nothing to do

            # No profile — send prompt to any known sender from message history
            senders = self.db.users.get_all_senders()
            for sender in senders:
                try:
                    logger.info("User %s has no profile, sending prompt", sender)
                    await self.channel.send_message(sender, PennyResponse.PROFILE_REQUIRED)
                except Exception as e:
                    logger.warning("Failed to send profile prompt to %s: %s", sender, e)
        except Exception as e:
            logger.warning("Failed to send profile prompts: %s", e)

    async def shutdown(self) -> None:
        """Clean shutdown of resources."""
        logger.info("Shutting down agent...")
        self.scheduler.stop()
        await self.channel.close()
        await Agent.close_all()
        await self.model_client.close()
        if self.vision_model_client:
            await self.vision_model_client.close()
        if self.embedding_model_client:
            await self.embedding_model_client.close()
        if self.email_client:
            await self.email_client.close()
        logger.info("Agent shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = Config.load()
    setup_logging(config.log_level, config.log_file, config.log_max_bytes, config.log_backup_count)

    logger.info("Starting Penny with config:")
    logger.info("  channel_type: %s", config.channel_type)
    logger.info("  llm_model: %s", config.llm_model)
    logger.info("  llm_api_url: %s", config.llm_api_url)
    logger.info("  idle_threshold: %.0fs", config.runtime.IDLE_SECONDS)

    agent = Penny(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)
    except ConnectionError as connection_error:
        # Surface startup connectivity failures (e.g. signal-api) in penny.log
        # so the docker restart loop is debuggable from the file logs alone.
        logger.error("Startup connectivity check failed: %s", connection_error)
        sys.exit(1)
    except PreflightError as preflight_error:
        # Surface hard setup-health failures (unreachable LLM endpoint, an
        # unresolvable chat/embedding model) in penny.log before exiting.
        logger.error("Setup preflight failed:\n%s", preflight_error)
        sys.exit(1)
