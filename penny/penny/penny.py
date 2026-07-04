"""Main agent loop for Penny."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from penny.agents import (
    Agent,
    ChatAgent,
    Collector,
)
from penny.channels import MessageChannel, create_channel_manager
from penny.channels.browser import BrowserChannel
from penny.channels.manager import ChannelManager
from penny.channels.permission_manager import PermissionManager
from penny.channels.signal.channel import SignalChannel
from penny.commands import create_command_registry
from penny.config import Config, setup_logging
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.migrate import migrate
from penny.llm.client import LlmClient
from penny.llm.embeddings import serialize_embedding
from penny.llm.image_client import OllamaImageClient
from penny.llm.models import LlmError
from penny.preflight import Preflight, PreflightError
from penny.responses import PennyResponse
from penny.scheduler import (
    AlwaysRunSchedule,
    BackgroundScheduler,
    PeriodicSchedule,
    Schedule,
)
from penny.scheduler.schedule_runner import ScheduleExecutor
from penny.scheduler.send_queue_drainer import SendQueueDrainer
from penny.startup import get_restart_message
from penny.zoho.models import ZohoCredentials

logger = logging.getLogger(__name__)


class Penny:
    """AI agent powered by local LLM inference via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize Penny — summary method."""
        self.config = config
        self.start_time = datetime.now()
        self._init_database(config)
        self._init_llm_clients(config)
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
        )
        self.collector = Collector(
            model_client=self.model_client,
            db=self.db,
            config=config,
            embedding_model_client=self.embedding_model_client,
        )
        self.chat_agent.set_collector(self.collector)
        self.schedule_executor = ScheduleExecutor(
            model_client=self.model_client,
            db=self.db,
            config=config,
            embedding_model_client=self.embedding_model_client,
        )
        # Deterministic task (no LLM) that delivers queued send_message output
        # once the autonomous-send cooldown clears.
        self.send_queue_drainer = SendQueueDrainer(db=self.db, config=config)

    def _init_github_client(self, config: Config) -> Any:
        """Initialize GitHub API client if configured. Returns GitHubAPI or None."""
        if not (
            config.github_app_id
            and config.github_app_private_key_path
            and config.github_app_installation_id
        ):
            return None
        try:
            from pathlib import Path

            from github_api.api import GitHubAPI
            from github_api.auth import GitHubAuth

            key_path = Path(config.github_app_private_key_path)
            if not key_path.is_absolute():
                key_path = Path.cwd() / key_path
            github_auth = GitHubAuth(
                app_id=int(config.github_app_id),
                private_key_path=key_path,
                installation_id=int(config.github_app_installation_id),
            )
            github_api = GitHubAPI(
                github_auth.get_token,
                PennyConstants.GITHUB_REPO_OWNER,
                PennyConstants.GITHUB_REPO_NAME,
            )
            logger.info("GitHub API client initialized")
            return github_api
        except Exception:
            logger.exception("Failed to initialize GitHub client")
            return None

    def _init_commands(self, config: Config) -> None:
        """Create command registry with GitHub client and optional integrations."""
        github_api = self._init_github_client(config)
        zoho_credentials = self._get_zoho_credentials(config)
        self.command_registry = create_command_registry(
            github_api=github_api,
            image_model_client=self.image_client,
            fastmail_api_token=config.fastmail_api_token,
            zoho_credentials=zoho_credentials,
        )

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
        self.schedule_executor.set_channel(self.channel)
        self.chat_agent.set_channel(self.channel)
        self.send_queue_drainer.set_channel(self.channel)
        # Collector needs the channel so consumer cycles (e.g. the ``notifier``,
        # which drains published collections) can call send_message.
        self.collector.set_channel(self.channel)
        self._wire_browser_tools(config)

    def _wire_browser_tools(self, config: Config) -> None:
        """Connect browser tools to agents when a browser channel is available."""
        if not isinstance(self.channel, ChannelManager):
            return
        browser_ch = self.channel.get_channel(ChannelType.BROWSER)
        if not isinstance(browser_ch, BrowserChannel):
            return

        # Wire up permission manager
        perm_mgr = PermissionManager(db=self.db, channel_manager=self.channel, config=config)
        browser_ch.set_permission_manager(perm_mgr)
        # Let the addon run a collection's extractor on demand.
        browser_ch.set_collector(self.collector)
        signal_ch = self.channel.get_channel(ChannelType.SIGNAL)
        if isinstance(signal_ch, SignalChannel):
            signal_ch.set_permission_manager(perm_mgr)

        # Browse provider — agents build fresh BrowseTools each cycle.
        def browse_provider():
            if not browser_ch.has_browser_connection:
                return None
            return browser_ch.send_tool_request, perm_mgr

        self.chat_agent._browse_provider = browse_provider
        self.collector._browse_provider = browse_provider
        self.collector._on_tool_start_factory = browser_ch.make_background_tool_callback

    def _init_scheduler(self, config: Config) -> None:
        """Create background scheduler — schedule_executor + collector dispatcher.

        The Collector is a single idle-gated schedule that ticks fast
        (COLLECTOR_TICK_INTERVAL).  Each tick it picks the most-overdue
        collection from ``memory`` (per-row ``collector_interval_seconds``)
        and runs that collection's extraction prompt.  Idle gating keeps
        collector work out of the way during active conversation; the
        store fills up "between conversations".
        """
        schedules: list[Schedule] = [
            AlwaysRunSchedule(agent=self.schedule_executor, interval=60.0),
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
            image_model_client=self.image_client,
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

    async def _backfill_preference_embeddings(self, batch_limit: int) -> int:
        """Backfill preferences with missing embeddings. Returns count embedded."""
        total = 0
        while True:
            prefs = self.db.preferences.get_without_embeddings(limit=batch_limit)
            if not prefs:
                break
            try:
                texts = [p.content for p in prefs]
                vecs = await self.embedding_model_client.embed(texts)
                for pref, vec in zip(prefs, vecs, strict=True):
                    assert pref.id is not None
                    self.db.preferences.update_embedding(pref.id, serialize_embedding(vec))
                    logger.info("Embedded preference %d: %s", pref.id, pref.content[:120])
                total += len(prefs)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for preferences: %s", e)
                break
        return total

    async def _backfill_memory_embeddings(self, batch_limit: int) -> int:
        """Backfill memory entries missing embeddings. Returns count embedded.

        Covers the unified memory framework (skills, ``user-messages``, any
        migration-seeded collection content) that the preference backfill
        doesn't reach.  Scoped by the store to non-archived, non-``off``
        memories so recall-relevant entries get vectors while bulk ``off``
        logs (``collector-runs``) are skipped.  Embeds both key and content
        so keyed entries are matchable by ``read_similar``.
        """
        total = 0
        while True:
            entries = self.db.memories.get_entries_without_embeddings(limit=batch_limit)
            if not entries:
                break
            try:
                content_vecs = await self.embedding_model_client.embed([e.content for e in entries])
                keyed = [(e.id, e.key) for e in entries if e.key is not None and e.id is not None]
                key_vec_by_id: dict[int, list[float]] = {}
                if keyed:
                    key_vecs = await self.embedding_model_client.embed([key for _, key in keyed])
                    key_vec_by_id = {
                        entry_id: vec for (entry_id, _), vec in zip(keyed, key_vecs, strict=True)
                    }
                for entry, content_vec in zip(entries, content_vecs, strict=True):
                    assert entry.id is not None
                    self.db.memories.set_entry_embeddings(
                        entry.id,
                        key_embedding=key_vec_by_id.get(entry.id),
                        content_embedding=content_vec,
                    )
                total += len(entries)
            except LlmError as e:
                logger.warning("Startup embedding backfill failed for memory entries: %s", e)
                break
        return total

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
        preferences, memory entries, description anchors, and message rows all
        get vectors at startup rather than lazily.
        """
        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)
        total_prefs = await self._backfill_preference_embeddings(batch_limit)
        if total_prefs:
            logger.info("Startup embedding backfill complete: %d preferences", total_prefs)
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
            if await self._send_ios_operational_announcement():
                return

            sender = self.db.users.get_primary_sender()
            if not sender:
                logger.info("No user profile found for startup announcement")
                return

            # Only announce if the user has chatted before (not a fresh profile)
            user_messages = self.db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
            if user_messages is None or not user_messages.newest_entries(k=1):
                logger.info("No message history yet, skipping startup announcement")
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
