"""Base abstractions for communication channels."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from penny.agents.models import ControllerResponse, ToolCallRecord, account_of
from penny.config import Config
from penny.constants import PennyConstants, PermissionResolution
from penny.database.media_store import ImageSelectionPolicy
from penny.database.models import Device, Media, MessageLog
from penny.image_utils import normalize_data_uri
from penny.llm import LlmClient
from penny.llm.call_link import CallLink
from penny.llm.embeddings import serialize_embedding
from penny.llm.similarity import embed_text
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import Command, CommandRegistry
    from penny.database import Database
    from penny.scheduler import BackgroundScheduler

logger = logging.getLogger(__name__)

# URLs Penny cites in an outgoing message — used to attach the cited page's own
# captured image at egress (see ``MediaStore.select_image``).
_MESSAGE_URL_RE = re.compile(r"https?://[^\s)>\]]+")


class PageContext(BaseModel):
    """The page the user is currently viewing in the browser."""

    title: str
    url: str
    text: str


class IncomingMessage(BaseModel):
    """A message received from any channel."""

    sender: str
    content: str
    channel_type: str | None = None  # ChannelType enum value
    device_identifier: str | None = None  # Device identifier for routing
    quoted_text: str | None = None
    # The quoted message's own platform id when the channel carries it (Signal's
    # ``Quote.id`` — the timestamp stamped as our outgoing row's ``external_id``),
    # so a quote-reply's parent is a read, not a content guess.
    quoted_external_id: str | None = None
    signal_timestamp: int | None = None  # Original Signal timestamp (ms since epoch)
    is_reaction: bool = False  # True if this is a reaction message
    reacted_to_external_id: str | None = None  # External ID of message being reacted to
    images: list[str] = Field(default_factory=list)  # Image attachments as data URIs
    page_context: PageContext | None = None  # Current page the user is viewing (browser only)


class ProgressTracker(ABC):
    """Lightweight in-flight progress indicator for an in-progress agent run.

    Channels return one of these from ``MessageChannel._begin_progress`` to
    surface what the agent is doing while it works (e.g., emoji reactions on
    the user's message that morph as tool calls fire). The dispatch loop
    calls ``update`` whenever a tool batch starts and ``clear`` exactly once
    when the run finishes — success, delivery failure, or exception. The
    final response is always delivered via the channel's normal
    ``send_response`` path so attachments and quotes work correctly.

    ``clear`` must be idempotent — the dispatch loop may invoke it on the
    success path and again from the ``finally`` cleanup.
    """

    @abstractmethod
    async def update(self, tools: list[tuple[str, dict]]) -> None:
        """Surface that the agent has started a new batch of tool calls."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove any in-flight progress indicator. Idempotent."""


class MessageChannel(ABC):
    """Abstract base class for communication channels."""

    def __init__(
        self,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        """
        Initialize channel with dependencies.

        Args:
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
            command_registry: Optional command registry for handling commands
        """
        self._message_agent = message_agent
        self._db = db
        self._command_registry = command_registry
        self._scheduler: BackgroundScheduler | None = None
        self._config: Config | None = None
        self._embedding_model_client: LlmClient | None = None

    def set_scheduler(self, scheduler: BackgroundScheduler) -> None:
        """Set the scheduler for message notifications."""
        self._scheduler = scheduler

    def set_command_context(
        self,
        config: Config,
        channel_type: str,
        start_time: datetime,
        model_client: LlmClient,
        embedding_model_client: LlmClient,
    ) -> None:
        """
        Set command context for command execution.

        Args:
            config: Penny config
            channel_type: Channel type ("signal" or "discord")
            start_time: Penny startup time
            model_client: Shared LlmClient for commands
            embedding_model_client: Shared embedding LlmClient for similarity
        """
        self._config = config
        self._model_client = model_client
        self._embedding_model_client = embedding_model_client

        from penny.commands import CommandContext

        self._command_context = CommandContext(
            db=self._db,
            config=config,
            model_client=model_client,
            user="",  # Will be set per-command
            channel_type=channel_type,
            start_time=start_time,
            embedding_model_client=embedding_model_client,
            scheduler=self._scheduler,
        )

    async def _embed_message(self, text: str) -> list[float] | None:
        """Best-effort embed helper for channels before command context wiring."""
        if self._embedding_model_client is None:
            return None
        return await embed_text(self._embedding_model_client, text)

    @property
    @abstractmethod
    def sender_id(self) -> str:
        """Get the identifier for this channel's outgoing messages."""
        pass

    @abstractmethod
    async def listen(self) -> None:
        """
        Start listening for messages and dispatch to handle_message.

        This method blocks until the channel is closed.
        """
        pass

    @abstractmethod
    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        source_name: str | None = None,
        message_log_id: int | None = None,
    ) -> int | None:
        """
        Deliver an already-prepared message to the platform.

        This is the single per-channel delivery primitive — the raw network
        send (Signal REST, Discord, browser WebSocket). It performs NO logging:
        every outgoing message is logged exactly once in ``send_message`` /
        ``send_response`` (which funnel through ``_log_and_send`` before calling
        this), so no send can bypass the conversation record. ``ChannelManager``
        overrides this to route to the resolved concrete channel.

        Args:
            recipient: Identifier for the recipient (platform-specific)
            message: Message content (already prepared via prepare_outgoing)
            attachments: Optional list of base64-encoded attachments
            quote_message: Optional message to quote-reply to
        source_name: Optional source attribution for durable channels
            message_log_id: Canonical messagelog ID for channels with durable delivery metadata

        Returns:
            Platform message id / timestamp on success, None on failure
        """
        pass

    async def validate_connectivity(self) -> None:
        """Validate connectivity to the channel's backend.

        No-op by default. Channels with expensive/flaky backends (e.g. Signal)
        override this to probe the backend at startup and raise on failure.
        """
        return

    async def wait_until_ready(self) -> None:
        """Wait until the channel can accept proactive outgoing sends.

        Channels whose send path depends on an async listener becoming ready
        override this. The default covers channels that can send immediately
        after construction/connectivity validation.
        """
        return

    def prepare_outgoing(self, text: str) -> str:
        """
        Prepare text for sending via this channel.

        Override in subclasses to apply channel-specific formatting.
        The result is what goes on the wire; the conversation record keeps
        the text as the model wrote it (#2722), so the channel conversion is
        applied only at send time, never stored.

        Args:
            text: Raw text from the agent

        Returns:
            Text formatted for this channel
        """
        return text

    @abstractmethod
    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """
        Send a typing indicator to a recipient.

        Args:
            recipient: Identifier for the recipient
            typing: True to start typing, False to stop

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """
        Extract a message from raw channel data.

        Args:
            raw_data: Raw data from the channel (WebSocket message, API event, etc.)

        Returns:
            IncomingMessage if valid message, None if should be ignored
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the channel and cleanup resources."""
        pass

    # --- Permission prompts (overridden by channels that support them) ---

    async def handle_permission_prompt(
        self,
        request_id: str,
        domain: str,
        url: str,
    ) -> None:
        """Handle a permission prompt broadcast. Override in subclasses."""
        return  # no-op default

    async def handle_permission_dismiss(
        self, request_id: str, resolution: PermissionResolution
    ) -> None:
        """Handle a resolved permission prompt. Override in subclasses.

        ``resolution`` says how the prompt ended, for channels that acknowledge
        it (Signal marks its own prompt message with the matching emoji).
        Channels whose prompt is a dismissable UI element just close it.
        """
        return  # no-op default

    async def handle_domain_permissions_changed(self) -> None:
        """Handle domain permissions update. Override in subclasses."""
        return  # no-op default

    async def _fetch_attachments(self, message: IncomingMessage, raw_data: dict) -> IncomingMessage:
        """
        Fetch attachment data for the message. Override in subclasses.

        Default implementation returns the message unchanged.
        """
        return message

    async def _typing_loop(self, recipient: str, interval: float = 4.0) -> None:
        """Send typing indicators on a loop until cancelled."""
        try:
            while True:
                await self.send_typing(recipient, True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def send_message(
        self,
        recipient: str,
        content: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """
        Log and deliver a plain outgoing message.

        The chokepoint for every non-conversational send — command results,
        error notices, onboarding prompts, threading rejections, permission
        prompts, startup announcements. The message is logged ``OUTGOING`` to
        ``messagelog`` (so it surfaces in the ``penny-messages`` facade) before
        delivery, so nothing Penny sends can skip the conversation record.
        It computes the message embedding before persistence.  If embedding
        generation fails, delivery continues and startup backfill remains the
        recovery path for that row.

        Returns the platform external id (or None on failure).
        """
        prepared = self.prepare_outgoing(content)
        embedding = await self._embed_message(content) if content.strip() else None
        _, external_id = await self._log_and_send(
            recipient,
            content,
            prepared,
            attachments,
            quote_message,
            embedding=embedding,
        )
        logger.info("Sent message to %s (%d chars)", recipient, len(prepared))
        return external_id

    async def send_response(
        self,
        recipient: str,
        content: str,
        parent_id: int | None,
        author: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        media_ids: list[int] | None = None,
        mechanism: str | None = None,
        run_id: str | None = None,
    ) -> int | None:
        """
        Log and deliver a conversational reply with embedding + side-channel media.

        Args:
            recipient: Identifier for the recipient
            content: Message content
            parent_id: Parent message ID for thread linking
            author: Name of the agent producing this message — stamped onto
                the side-effect write to ``penny-messages``.  Always passed
                explicitly by the caller (chat reply path, notify, scheduler)
                so attribution is correct without ambient state.
            attachments: Optional list of base64-encoded image attachments
            quote_message: Optional message to quote-reply to
            media_ids: Media rows this run generated (``generate_image``) that must
                be attached deterministically to *this* reply — the exact rows,
                not an embedding-nearest guess.
            mechanism: The bound collection whose autonomous cycle produced this
                send (#1568) — stamped on the delivered ``messagelog`` row so it
                names its cause.  ``None`` for a direct reply (a chat turn with a
                live triggering user message).  Only the ``SendQueueDrainer``
                passes it, from the queued row.
            run_id: The run whose reply this direct reply delivers (#2630) —
                stamped on the delivered ``messagelog`` row so the conversation
                history pairs a run with the message that answered for it by
                identity.  ``None`` for an autonomous send, which has no run
                behind it.  Only the chat reply path passes it.

        Returns:
            Database message ID if send was successful, None otherwise
        """
        prepared = self.prepare_outgoing(content)
        # Embed once: stored on the messagelog row (the penny-messages facade's
        # read_similar ranks on it) and reused for nearest-image matching. The
        # embedding is of the model's own text — the content stored on the row —
        # so recall and the startup backfill rank on the same words.
        embedding = await self._embed_message(content)
        attachments = self._resolve_media(attachments, content, embedding, media_ids)
        message_id, external_id = await self._log_and_send(
            recipient,
            content,
            prepared,
            attachments,
            quote_message,
            parent_id=parent_id,
            embedding=embedding,
            source_name=author,
            mechanism=mechanism,
            run_id=run_id,
        )
        logger.info("Sent response to %s (%d chars)", recipient, len(content))
        return message_id if external_id is not None else None

    async def _log_and_send(
        self,
        recipient: str,
        content: str,
        prepared: str,
        attachments: list[str] | None,
        quote_message: MessageLog | None,
        *,
        parent_id: int | None = None,
        embedding: list[float] | None = None,
        source_name: str | None = None,
        mechanism: str | None = None,
        run_id: str | None = None,
    ) -> tuple[int | None, int | None]:
        """Log an ``OUTGOING`` message to messagelog, deliver it, stamp external_id.

        The single funnel both ``send_message`` and ``send_response`` pass
        through, so logging happens exactly once immediately before the raw
        platform send and no outgoing message can bypass the record. The
        conversation record keeps ``content`` — the reply as the model wrote
        it (#2722), so history shows the model its own formatting — while
        ``prepared`` (the channel-converted form) is what goes on the wire.
        ``mechanism`` (#1568) is stamped for an autonomous send and NULL for
        a direct reply; ``run_id`` (#2630) the opposite side — the run whose
        reply a direct reply delivers, NULL for an autonomous send. Returns
        ``(message_id, external_id)``.
        """
        if (not prepared or not prepared.strip()) and not attachments:
            logger.error("Attempted to send empty message to %s", recipient)
            raise ValueError("Cannot send empty or whitespace-only message")
        device = self._db.devices.get_by_identifier(recipient)
        device_id = device.id if device else None
        message_id = self._db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            self.sender_id,
            content,
            parent_id=parent_id,
            recipient=recipient,
            device_id=device_id,
            embedding=serialize_embedding(embedding) if embedding is not None else None,
            mechanism=mechanism,
            run_id=run_id,
        )
        external_id = await self._send_raw(
            recipient,
            prepared,
            attachments,
            quote_message,
            source_name,
            message_id,
        )
        # Store the external ID for future reactions and quote replies
        if external_id and message_id:
            self._db.messages.set_external_id(message_id, str(external_id))
        return message_id, external_id

    def _resolve_media(
        self,
        attachments: list[str] | None,
        text: str,
        embedding: list[float] | None,
        media_ids: list[int] | None = None,
    ) -> list[str] | None:
        """Resolve the image(s) to attach to this reply, most-authoritative first.

        1. Caller-supplied ``attachments`` win outright.
        2. ``media_ids`` — rows this run *generated* (``generate_image``) — are
           attached deterministically: exactly those images, fetched by id, land
           on the reply that describes them.  The fuzzy ladder does not also run
           (no double attach).
        3. Otherwise the nearest-image ladder: ``select_image`` prefers a cited
           page's own image (exact URL, then same domain) and falls back to a
           jittered embedding-nearest pick — so every reply carries an image
           whenever one can be matched.
        """
        if attachments:
            return attachments
        generated = self._encode_media(media_ids) if media_ids else None
        if generated:
            return generated
        urls = _MESSAGE_URL_RE.findall(text)
        media = self._db.media.select_image(urls, embedding, policy=self._image_selection_policy())
        if media is None:
            return None
        return [self._encode_media_row(media)]

    def _image_selection_policy(self) -> ImageSelectionPolicy:
        """Channels can restrict automatic images without changing explicit delivery."""
        return ImageSelectionPolicy()

    def _encode_media(self, media_ids: list[int]) -> list[str] | None:
        """Fetch each generated media row by id and encode it as a data URI.

        A missing id (the row was just committed, so this should not happen)
        is logged and skipped rather than crashing egress — a visible signal,
        not a silent swallow.  Returning None (nothing resolved) lets
        ``_resolve_media`` fall back to the nearest-image ladder, so the reply
        still carries an image.
        """
        encoded: list[str] = []
        for media_id in media_ids:
            media = self._db.media.get(media_id)
            if media is None:
                logger.error("Generated media %d vanished before egress — not attached", media_id)
                continue
            encoded.append(self._encode_media_row(media))
        return encoded or None

    @staticmethod
    def _encode_media_row(media: Media) -> str:
        """Encode a media row's bytes as a base64 ``data:`` URI attachment."""
        encoded = base64.b64encode(media.data).decode()
        return f"data:{media.mime_type};base64,{encoded}"

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message through the agent.

        This is the main message handling logic, shared by all channel implementations.

        A turn never ends in silence (#2371): the paths below that know enough to
        account for their own failure handle it themselves (``_finish_turn``, the
        command executor); this is the backstop for everything else.  ``message``
        starts unbound-but-named so the ``except`` below can tell whether we ever
        learned who to answer — before ``extract_message`` returns, there is no
        recipient, so logging is the honest ceiling.
        """
        message: IncomingMessage | None = None
        try:
            message = self.extract_message(envelope_data)
            if message is None:
                return

            if message.is_reaction:
                await self._handle_reaction(message)
                return

            message = await self._fetch_attachments(message, envelope_data)

            if not await self._validate_message(message):
                return

            if await self._dispatch_command(message):
                return

            if await self._reject_unsupported_thread(message):
                return

            await self._dispatch_to_agent(message)

        except Exception as e:
            logger.exception("Error handling message: %s", e)
            if message is not None:
                await self._report_turn_failure(
                    message.sender, PennyResponse.FACTS_TURN_FAILED_BEFORE_REPLY, None
                )

    async def _report_turn_failure(
        self, recipient: str, facts: str, tool_calls: list[ToolCallRecord] | None
    ) -> None:
        """Tell the user a turn died on an escaped exception — never silence, and
        never ``AGENT_MODEL_ERROR`` (that fact is only for the model itself being
        unreachable, decided inside the agent, not here).  Built deterministically
        from what this layer knows (``account_of``, reused from #2370): no second
        model call, since by the time we're here the model's own state is unknown.

        ``tool_calls=None`` means UNKNOWN, not empty — a turn can die before this
        layer ever sees whether the agent called a tool, and ``account_of`` reading
        an empty list as "no tool calls" would claim something the record doesn't
        show.  Send ``facts`` alone in that case; a real list (even ``[]``, when the
        run really made none) goes through ``account_of`` as usual.

        Guarded so a second failure (e.g. the DB that just broke) is logged rather
        than raised — ``handle_message`` must never crash its channel's receive loop."""
        content = facts if tool_calls is None else account_of(facts, tool_calls)
        try:
            await self.send_message(recipient, content)
        except Exception:
            logger.exception("Also failed to tell %s this turn failed", recipient)

    async def _validate_message(self, message: IncomingMessage) -> bool:
        """Notify the scheduler of the incoming message. Returns False if dropped."""
        if self._scheduler:
            self._scheduler.notify_message()

        logger.info("Received message from %s: %s", message.sender, message.content)
        return True

    async def _dispatch_command(self, message: IncomingMessage) -> bool:
        """Detect and route slash commands. Returns True if message was a command."""
        if not message.content.strip().startswith("/"):
            return False

        command_name = message.content.strip()[1:].split(maxsplit=1)[0].lower()
        logger.info("Command detected: /%s from %s", command_name, message.sender)
        commands_supporting_quotes = {"bug"}

        if message.quoted_text and command_name not in commands_supporting_quotes:
            await self.send_message(message.sender, PennyResponse.THREADING_NOT_SUPPORTED_COMMANDS)
            return True

        await self._handle_command(message)
        return True

    def _is_thread_reply_to_command(self, message: IncomingMessage) -> bool:
        """Check if the message is a thread reply to a slash command."""
        return bool(message.quoted_text and message.quoted_text.strip().startswith("/"))

    async def _reject_unsupported_thread(self, message: IncomingMessage) -> bool:
        """Reject thread replies to commands. Returns True if rejected."""
        if self._is_thread_reply_to_command(message):
            await self.send_message(message.sender, PennyResponse.THREADING_NOT_SUPPORTED_COMMANDS)
            return True

        return False

    def _resolve_device_id(self, message: IncomingMessage) -> int | None:
        """Look up the device ID from the message's device identifier.

        A miss falls through to ``_bootstrap_device`` — a channel whose
        endpoint has no independent pairing flow (Signal: #2765) registers it
        here, the first time it's seen, instead of never. Browser and iOS
        register on connect/pair already, so their default no-op costs
        nothing extra on the miss path.
        """
        if not message.device_identifier:
            return None
        device = self._db.devices.get_by_identifier(message.device_identifier)
        if device is None:
            device = self._bootstrap_device(message)
        return device.id if device else None

    def _bootstrap_device(self, message: IncomingMessage) -> Device | None:
        """Register a device the first time its identifier is ever seen.

        Default no-op. Override on a channel whose endpoint has no explicit
        pairing step (Signal) to claim it lazily — the identifier is known
        the moment a message arrives, even on a fresh install with no profile
        yet, which is earlier than startup can know it.
        """
        return None

    def _needs_profile(self) -> bool:
        """Check if any user profile exists (Penny is single-user)."""
        try:
            return self._db.users.get_primary_sender() is None
        except SQLAlchemyError:
            logger.exception("Failed to check for existing user profile")
            return False

    def _resolve_user_sender(self, device_sender: str) -> str:
        """Resolve a device identifier to the primary user sender for DB lookups."""
        primary = self._db.users.get_primary_sender()
        return primary if primary else device_sender

    def _make_handle_kwargs(
        self, message: IncomingMessage, progress: ProgressTracker | None = None
    ) -> dict:
        """Return extra kwargs for ChatAgent.handle(). Override in subclasses.

        ``progress`` is the optional tracker returned by ``_begin_progress``.
        Default forwards ``progress.update`` as ``on_tool_start`` so any
        channel that returns a tracker gets tool-call progress updates for
        free; channels with custom progress UI (e.g. browser) can override.
        """
        if progress is None:
            return {}
        return {"on_tool_start": progress.update}

    async def _begin_progress(self, message: IncomingMessage) -> ProgressTracker | None:
        """Start an in-flight progress indicator for this message.

        Channels that surface progress (e.g., emoji reactions on the user's
        message) override this and return a ``ProgressTracker``. Default
        returns ``None`` — the dispatch loop runs unmodified.
        """
        return None

    async def _dispatch_to_agent(self, message: IncomingMessage) -> None:
        """Run the message through the agent loop with typing indicators."""
        device_id = self._resolve_device_id(message)
        user_sender = self._resolve_user_sender(message.sender)

        if self._needs_profile():
            await self._handle_profile_required(message, user_sender, device_id)
            return

        typing_task = asyncio.create_task(self._typing_loop(message.sender))
        progress: ProgressTracker | None = None
        try:
            if self._scheduler:
                self._scheduler.notify_foreground_start()
            progress = await self._begin_progress(message)
            await self._run_message_through_agent(message, user_sender, device_id, progress)
        finally:
            if progress is not None:
                await progress.clear()
            typing_task.cancel()
            await self.send_typing(message.sender, False)
            if self._scheduler:
                self._scheduler.notify_foreground_end()

    async def _handle_profile_required(
        self, message: IncomingMessage, user_sender: str, device_id: int | None
    ) -> None:
        """Log the message but redirect the user to profile setup."""
        embedding = await self._embed_message(message.content)
        self._db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            user_sender,
            message.content,
            signal_timestamp=message.signal_timestamp,
            device_id=device_id,
            embedding=serialize_embedding(embedding) if embedding is not None else None,
        )
        await self.send_message(message.sender, PennyResponse.PROFILE_REQUIRED)

    def _resolve_quoted_parent(self, message: IncomingMessage) -> int | None:
        """The parent a quote-reply continues, most-authoritative first.

        When the channel carries the quoted message's own id (Signal's
        ``Quote.id`` — the timestamp stamped as our outgoing row's
        ``external_id``), the parent is a read by that anchor. Content
        matching stays the fallback: the quote text arriving over the wire is
        the channel-converted form, while the conversation record keeps the
        reply as the model wrote it (#2722), so matching is reliable only for
        quotes on messages the channel never reformats — the user's own.
        The resolved id is what the thread link and the turn's history both
        read; the quote's text is never re-matched downstream.
        """
        if message.quoted_external_id:
            parent = self._db.messages.find_by_external_id(message.quoted_external_id)
            if parent is not None and parent.id is not None:
                return parent.id
        if message.quoted_text:
            parent = self._db.messages.find_outgoing_by_content(message.quoted_text)
            if parent is not None and parent.id is not None:
                return parent.id
        return None

    async def _run_message_through_agent(
        self,
        message: IncomingMessage,
        user_sender: str,
        device_id: int | None,
        progress: ProgressTracker | None,
    ) -> None:
        """Invoke the agent, log the incoming message, and deliver the response.

        The egress write inside ``send_response`` is attributed to the
        message agent's ``name`` (template-method override on the agent
        subclass) — passed explicitly down the call chain rather than
        threaded through ambient state.

        The true top-level run starter for a chat turn (#2458): mints the top
        ``CallLink`` before the turn's own loop even draws, passed explicitly down
        the call chain with no ambient state anywhere.
        """
        logger.info("Dispatching to message agent for %s", message.sender)
        # Normalize every image attachment to the stored-image size rules (#2780)
        # before it ever reaches the model request or the prompt log — the one
        # shared chokepoint every channel's incoming message passes through here,
        # so a channel that starts accepting image attachments later needs no
        # change of its own.
        if message.images:
            message = message.model_copy(
                update={"images": [normalize_data_uri(uri) for uri in message.images]}
            )
        # Mint the turn's run id here so the same id stamps every promptlog row of
        # the run AND is recorded on any collection the run creates
        # (``created_by_run_id``, #1566).  The incoming message is logged AFTER the
        # run (below), so it never doubles into the turn's own recall — then it is
        # linked structurally to the run by ``link_source_message``.
        # ``user-messages`` is a read facade over ``messagelog`` — no separate append.
        run_id = uuid.uuid4().hex
        call_link = CallLink.top(run_id)
        parent_id = self._resolve_quoted_parent(message)
        response = await self._message_agent.handle(
            content=message.content,
            sender=user_sender,
            call_link=call_link,
            images=message.images or None,
            page_context=message.page_context,
            thread_parent_id=parent_id,
            **self._make_handle_kwargs(message, progress),
        )
        try:
            await self._finish_turn(
                message, user_sender, device_id, response, run_id, parent_id, progress
            )
        except Exception:
            logger.exception("Failed to finish turn for %s", message.sender)
            await self._report_turn_failure(
                message.sender, PennyResponse.FACTS_TURN_FAILED_AFTER_REPLY, response.tool_calls
            )

    async def _finish_turn(
        self,
        message: IncomingMessage,
        user_sender: str,
        device_id: int | None,
        response: ControllerResponse,
        run_id: str,
        parent_id: int | None,
        progress: ProgressTracker | None,
    ) -> None:
        """Log the incoming message, link it to anything the run created, and
        deliver the reply — the part of the turn that has an answer in hand
        already (#2371: split out so a failure here can be reported with the
        run's own ``response.tool_calls``, which ``_run_message_through_agent``
        doesn't have once control returns to it)."""
        # The conversation record keeps a short deterministic marker where the image was
        # (#2635) — the attachment itself rides only the turn's own user message and
        # is never resent into history, so a later turn reads this line instead.
        logged_content = message.content
        if message.images:
            logged_content = f"{logged_content} {PennyConstants.IMAGE_HISTORY_MARKER}".strip()
        incoming_embedding = await self._embed_message(logged_content)
        incoming_id = self._db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            user_sender,
            logged_content,
            parent_id=parent_id,
            signal_timestamp=message.signal_timestamp,
            device_id=device_id,
            embedding=serialize_embedding(incoming_embedding)
            if incoming_embedding is not None
            else None,
        )
        # Link the spawning message to any mechanism this run created, now that the
        # message has an id (#1566) — matched by the unique per-turn run id.
        if incoming_id is not None:
            self._db.memories.link_source_message(run_id, incoming_id)
        await self._deliver_agent_response(
            message, user_sender, response, incoming_id, progress, self._message_agent.name, run_id
        )

    async def _deliver_agent_response(
        self,
        message: IncomingMessage,
        user_sender: str,
        response: Any,
        incoming_id: int | None,
        progress: ProgressTracker | None,
        author: str,
        run_id: str,
    ) -> None:
        """Send the agent's response and surface delivery failures."""
        # Everything upstream (Agent._narrate_or_account, #2370) already tries to have
        # the model say SOMETHING, so an empty answer reaching send time is the rare
        # structural case (e.g. a deliberate write-gate STOP) rather than the model
        # refusing — the deterministic account built from this run's own tool calls,
        # never a generic apology.
        answer = (
            response.answer.strip()
            if response.answer
            else account_of(PennyResponse.FACTS_NO_REPLY_AT_SEND, response.tool_calls)
        )
        incoming_log = MessageLog(
            id=incoming_id,
            direction=PennyConstants.MessageDirection.INCOMING,
            sender=user_sender,
            content=message.content,
            signal_timestamp=message.signal_timestamp,
        )
        # Clear progress before sending so the user sees "done working"
        # immediately even if the send takes a moment. ``clear`` is idempotent
        # so the finally block can call it again on exception paths.
        if progress is not None:
            await progress.clear()
        sent = await self.send_response(
            message.sender,
            answer,
            parent_id=incoming_id,
            author=author,
            quote_message=incoming_log,
            media_ids=response.generated_media_ids,
            run_id=run_id,
        )
        if sent is None:
            logger.error("Failed to deliver response to %s — notifying user", message.sender)
            await self.send_message(message.sender, PennyResponse.DELIVERY_FAILURE)

    async def _handle_reaction(self, message: IncomingMessage) -> None:
        """Log a reaction as a regular incoming message in the thread."""
        if not message.reacted_to_external_id:
            logger.warning("Reaction message missing reacted_to_external_id")
            return

        reacted_msg = self._db.messages.find_by_external_id(message.reacted_to_external_id)
        if not reacted_msg or not reacted_msg.id:
            logger.warning(
                "Could not find message with external_id=%s for reaction",
                message.reacted_to_external_id,
            )
            return

        device_id = self._resolve_device_id(message)
        user_sender = self._resolve_user_sender(message.sender)
        embedding = await self._embed_message(message.content)
        self._db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            user_sender,
            message.content,
            parent_id=reacted_msg.id,
            is_reaction=True,
            device_id=device_id,
            embedding=serialize_embedding(embedding) if embedding is not None else None,
        )

        logger.info(
            "Logged reaction from %s: %s (parent_id=%d)",
            message.sender,
            message.content,
            reacted_msg.id,
        )

    def _parse_command(self, text: str) -> tuple[str, str]:
        """Parse command name and arguments from a slash command string."""
        parts = text.strip()[1:].split(maxsplit=1)  # Skip leading /
        command_name = parts[0].lower()
        command_args = parts[1] if len(parts) > 1 else ""
        return command_name, command_args

    async def _execute_command(
        self,
        message: IncomingMessage,
        command_name: str,
        command_args: str,
        command: Command,
    ) -> None:
        """Execute a known command with typing indicator and send the result."""
        user_sender = self._resolve_user_sender(message.sender)
        typing_task = asyncio.create_task(self._typing_loop(message.sender))
        try:
            context = self._command_context
            context.user = user_sender
            context.message = message

            result = await command.execute(command_args, context)
            response = result.text

            await self.send_message(
                message.sender, response, attachments=result.attachments, quote_message=None
            )
            self._log_command_result(user_sender, command_name, command_args, response)
            logger.info("Executed command /%s for %s", command_name, message.sender)

        except Exception as e:
            logger.exception("Error executing command /%s: %s", command_name, e)
            error_response = PennyResponse.COMMAND_ERROR.format(error=e)
            await self.send_message(message.sender, error_response)
            self._log_command_result(
                user_sender, command_name, command_args, error_response, error=str(e)
            )
        finally:
            typing_task.cancel()
            await self.send_typing(message.sender, False)

    def _log_command_result(
        self,
        sender: str,
        command_name: str,
        command_args: str,
        response: str,
        error: str | None = None,
    ) -> None:
        """Log a command execution to the database."""
        self._db.messages.log_command(
            user=sender,
            channel_type=self._command_context.channel_type,
            command_name=command_name,
            command_args=command_args,
            response=response,
            error=error,
        )

    async def _handle_command(self, message: IncomingMessage) -> None:
        """Handle a command message: parse, look up, and execute."""
        if not self._command_registry:
            logger.warning("Command received but no registry configured")
            return

        command_name, command_args = self._parse_command(message.content)
        command = self._command_registry.get(command_name)

        if not command:
            response = PennyResponse.UNKNOWN_COMMAND.format(command_name=command_name)
            await self.send_message(message.sender, response)
            user_sender = self._resolve_user_sender(message.sender)
            self._log_command_result(
                user_sender, command_name, command_args, response, error="unknown command"
            )
            return

        await self._execute_command(message, command_name, command_args, command)
