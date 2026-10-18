"""Message store — logging, threading, and queries for messages."""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, NamedTuple

from pydantic import BaseModel
from sqlalchemy import and_, bindparam, func, inspect, or_, text
from sqlmodel import Session, col, select
from sqlmodel.sql.expression import SelectOfScalar

from penny.agents.models import MessageRole
from penny.clock import Clock
from penny.constants import ChatPromptType, PennyConstants, RunOutcome
from penny.database.memory.objects import _run_tool_calls, classify_run, render_run_record
from penny.database.models import (
    CommandLog,
    Device,
    IosOutboxItem,
    MessageLog,
    PromptLog,
)

logger = logging.getLogger(__name__)

# Patterns for stripping markdown formatting from outgoing messages
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_STRIKETHROUGH_RE = re.compile(r"~{1,2}(.+?)~{1,2}")
_MONOSPACE_RE = re.compile(r"`(.+?)`")
_TILDE_OPERATOR = "\u223c"


class CallStarted(BaseModel):
    """A model call about to be made (#2439) — published BEFORE any ``promptlog``
    row exists for it, from ``LlmClient.chat()`` itself (the one chokepoint every
    caller shares: the agent loop, a micro-context's own draw via
    ``CallLink.child()``, the conversation-machine classifier, round framing,
    skill extraction).  Carries exactly the identity the eventual logged row
    will carry (``run_id``/``parent_run_id``/``parent_tool_call_id``), so a live
    Prompts tab can show a running row in the same place the logged draw will
    land, and later match the two by ``run_id`` — a ``child()`` draw mints a
    fresh one every time (unique, trivial match); a run's own sequential steps
    share one, but the agent loop awaits each call before starting the next, so
    at most one is ever outstanding for a given ``run_id`` (unambiguous FIFO)."""

    run_id: str
    parent_run_id: str | None
    parent_tool_call_id: str | None
    agent_name: str | None
    prompt_type: str | None
    model: str
    run_target: str | None
    started_at: datetime


class ToolStarted(BaseModel):
    """A dispatched tool call about to execute (#2439) — published from the
    dispatcher (``Agent._execute_single_tool``) before ``ToolExecutor.execute``
    runs.  ``(run_id, tool_call_id)`` is an exact identity (unlike a model
    call's), so a running tool entry never needs FIFO matching."""

    run_id: str
    tool_call_id: str
    tool_name: str
    status: str
    started_at: datetime


class ToolFinished(BaseModel):
    """A dispatched tool call that just finished (#2439) — published right
    after ``ToolExecutor.execute`` returns, from the same site as
    ``ToolStarted``.  ``success`` is the tool's own structural result, so the
    addon can flip a running row to a checkmark/x immediately, ahead of the
    authoritative logged record a later call's ``messages`` will carry."""

    run_id: str
    tool_call_id: str
    tool_name: str
    status: str
    success: bool


class ChatTurnTrace(NamedTuple):
    """One chat turn's run as the conversation history reads it (#2630): when the
    run's last logged call happened, the id of that last promptlog row (the tie-break
    when a call and the send it produced share a timestamp), and the tool names the
    run's OWN prompt-log records show it called — the ledger's projection, never a
    reconstruction from the reply's wording."""

    last_call: datetime
    row_id: int
    tools: tuple[str, ...]


class PromptPerf(NamedTuple):
    """Aggregate wall time + token usage across logged prompts.

    Sourced from data the real LLM path already records \u2014 ``duration_ms`` per
    call plus the token usage stored inside each response \u2014 so the eval suite
    can report throughput without any new instrumentation.

    ``output_tokens`` (the OpenAI ``completion_tokens``) already *includes* the
    reasoning trace. ``reasoning_tokens`` is the provider's OWN count of that part
    (``usage.completion_tokens_details.reasoning_tokens``) — a read, not a guess. Backends
    that do not report it leave it 0, which is why the character lengths of the stored
    ``thinking`` trace and the visible ``content`` are still carried: their ratio is the
    ESTIMATE a reader falls back to, and keeping both is what lets a report say which of
    the two it is showing rather than presenting an estimate as a measurement.

    Why this matters beyond curiosity: a remote provider's wall time says nothing about
    how the same model will run locally, but its token counts do. Two runs that score the
    same while one spends far more tokens thinking are not equivalent — the second is a
    performance regression on the machine that has to generate them.
    """

    calls: int
    duration_ms: int
    input_tokens: int
    output_tokens: int
    thinking_chars: int = 0
    output_chars: int = 0
    reasoning_tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        seconds = self.duration_ms / 1000
        return self.output_tokens / seconds if seconds else 0.0

    @property
    def reasoning_share(self) -> float:
        """Fraction of generated characters that were reasoning (0.0\u20131.0).

        ``output_tokens`` bundles reasoning + visible output, so this char-based
        ratio is how we split it \u2014 multiply by ``output_tokens`` for an estimated
        reasoning-token count."""
        total = self.thinking_chars + self.output_chars
        return self.thinking_chars / total if total else 0.0


class RunActivity(BaseModel):
    """One completed collector run at rollup altitude, for the self-state header's
    recent-activity block (#1555).

    ``run_id`` is the typed anchor the header names verbatim; ``target`` is the
    collection it served (also the ``run_list`` drill-down anchor);
    ``outcome`` is the ``RunOutcome`` value; ``finished_at`` is the run's end
    time (its outcome-bearing last prompt); ``call_count`` is the number of tool
    calls the run made \u2014 the rollup summary in place of the per-call detail."""

    run_id: str
    target: str
    outcome: str
    finished_at: datetime
    call_count: int


class ChatFailureActivity(BaseModel):
    """One chat turn that needed the honest-fallback machinery, at rollup altitude,
    for the self-state header's recent-activity block (#2355 part D).

    ``run_id`` is the typed anchor the header names verbatim (consumed by
    ``run_get(run='run <id>')``, exactly as a collector run's is); ``reason`` is the
    THIRD-PERSON structural fact ``ChatAgent._record_turn_failure`` stamped
    (``run_reason`` — never a model-authored summary); ``finished_at`` is the
    stamped prompt's own timestamp."""

    run_id: str
    reason: str
    finished_at: datetime


class RunOutcomeStamp(BaseModel):
    """A collector's most recent completed run \u2014 its ``RunOutcome`` value and
    finish time.  The mechanism-inventory 'last run' line reads one of these per
    mechanism (#1555)."""

    outcome: str
    finished_at: datetime


class EmissionActivity(BaseModel):
    """One delivered autonomous send at rollup altitude, for the self-state
    header's recent-activity block (#1568).

    ``mechanism`` is the bound collection whose cycle produced it (the
    ``memory_get`` anchor); ``sent_at`` is when it went out; ``snippet`` is a
    short, whitespace-collapsed excerpt of the content so the model sees *what* it
    autonomously said, not only that it did.  Direct replies (``mechanism`` NULL)
    are excluded \u2014 they are the conversation, not its complement."""

    mechanism: str
    sent_at: datetime
    snippet: str


# First N characters of a delivered autonomous send, rendered as the
# activity-block emission snippet (#1568).  A short, whitespace-collapsed excerpt
# — enough to recognise what Penny said, not the whole message.
_EMISSION_SNIPPET_CHARS = 50


def _emission_snippet(content: str) -> str:
    """A one-line excerpt (first ``_EMISSION_SNIPPET_CHARS`` chars) of an outgoing
    autonomous send, whitespace collapsed so a multi-line body reads as one line."""
    collapsed = " ".join(content.split())
    if len(collapsed) <= _EMISSION_SNIPPET_CHARS:
        return collapsed
    return f"{collapsed[:_EMISSION_SNIPPET_CHARS].rstrip()}…"


def _count_run_tool_calls(prompts: list[PromptLog]) -> int:
    """Total tool calls a run made, across its prompt rows \u2014 the rollup count the
    self-state activity block shows in place of the per-call trace."""
    total = 0
    for prompt in prompts:
        response = json.loads(prompt.response) if prompt.response else {}
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            total += len(message.get("tool_calls") or [])
    return total


# How every ``messagelog`` read orders: by the datetime column, with ``id`` breaking a tie
# under it.  The house rule is that recency is a datetime column's answer and never an id's
# (ids are for joins and lookups) \u2014 and it still is: ``timestamp`` decides, and ``id`` is
# consulted only where ``timestamp`` has no answer to give.  A tie is not hypothetical; a
# burst of writes lands several rows in one microsecond, and with nothing to separate them
# the rows come back in whatever order the query plan or a Python merge happened to produce
# \u2014 a reply and the message that answers it swapping places (#1963).  ``id`` is monotonic in
# write order, which is exactly the fact the equal timestamps lost.
#
# It costs nothing: SQLite carries the rowid as the trailing column of every index, so
# ``ORDER BY timestamp DESC, id DESC`` still resolves straight off the timestamp index.
_NEWEST_MESSAGE_FIRST = (col(MessageLog.timestamp).desc(), col(MessageLog.id).desc())


def _in_send_order(message: MessageLog) -> tuple[datetime, int]:
    """One message's place in the conversation \u2014 the Python-side twin of
    ``_NEWEST_MESSAGE_FIRST``, for a result merged from several queries.

    ``id or 0`` satisfies the type only: a row read back from the database always has one."""
    return (message.timestamp, message.id or 0)


class MessageStore:
    """Manages MessageLog, PromptLog, and CommandLog records.

    Table of contents, in file order: live step pub/sub for a connected addon
    (``subscribe_*``/``publish_*``) · message logging (``log_message``,
    ``log_prompt``, ``log_command``) · message metadata
    (``set_signal_timestamp``, ``set_external_id``) · embedding backfill
    (``messages_without_embeddings``, ``set_embedding``) · message lookup
    (``get_by_id``, ``get_by_ids``, ``find_by_external_id``,
    ``find_outgoing_by_content``) · thread context (``get_thread_context``) ·
    conversation queries (``get_conversation_leaves``, ``get_user_messages``,
    ``get_messages_since``, the iOS history page via ``ios_history_page``/
    ``ios_history_count``, ``get_unprocessed``, ``get_user_reactions``,
    ``mark_processed``) · aggregate + run queries (``count``/
    ``count_active_threads``, ``set_run_outcome``, the ``recent_*`` activity
    reads, ``run_call_groups``, ``get_prompt_log_runs``, ``get_run``,
    ``get_target_runs``, ``prompt_perf``, and the run/prompt serializers
    behind them)."""

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()
        # Lists, not slots (#2441): every connected channel — browser extension AND
        # iOS — subscribes its own listener here, so each gets every push instead of
        # only the one built last silently taking over the other's.
        self._prompt_logged_listeners: list[Callable[[dict], None]] = []
        self._run_outcome_set_listeners: list[Callable[[str, str, str], None]] = []
        # Live step markers (#2439) — published ahead of any DB row, from
        # LlmClient.chat() and the tool dispatcher, so a connected addon can show
        # a running step before the logged record exists.  No channel subscribes
        # these by default (browser does; iOS doesn't yet — no live renderer).
        self._call_started_listeners: list[Callable[[CallStarted], None]] = []
        self._tool_started_listeners: list[Callable[[ToolStarted], None]] = []
        self._tool_finished_listeners: list[Callable[[ToolFinished], None]] = []

    def subscribe_prompt_logged(self, listener: Callable[[dict], None]) -> None:
        """Add a listener notified after every logged prompt that carries a run id."""
        self._prompt_logged_listeners.append(listener)

    def subscribe_run_outcome_set(self, listener: Callable[[str, str, str], None]) -> None:
        """Add a listener notified whenever a run's outcome is set."""
        self._run_outcome_set_listeners.append(listener)

    def subscribe_call_started(self, listener: Callable[[CallStarted], None]) -> None:
        """Add a listener notified just before every model call is made (#2439)."""
        self._call_started_listeners.append(listener)

    def subscribe_tool_started(self, listener: Callable[[ToolStarted], None]) -> None:
        """Add a listener notified just before every dispatched tool call runs (#2439)."""
        self._tool_started_listeners.append(listener)

    def subscribe_tool_finished(self, listener: Callable[[ToolFinished], None]) -> None:
        """Add a listener notified right after every dispatched tool call finishes (#2439)."""
        self._tool_finished_listeners.append(listener)

    def _publish_prompt_logged(self, payload: dict) -> None:
        """Notify every subscriber; one raising must not stop the rest, or the
        write that already landed before this fires."""
        for listener in tuple(self._prompt_logged_listeners):
            try:
                listener(payload)
            except Exception:
                logger.exception("A prompt-logged listener failed")

    def _publish_run_outcome_set(self, run_id: str, outcome: str, reason: str) -> None:
        """Notify every subscriber; one raising must not stop the rest."""
        for listener in tuple(self._run_outcome_set_listeners):
            try:
                listener(run_id, outcome, reason)
            except Exception:
                logger.exception("A run-outcome-set listener failed")

    def publish_call_started(
        self,
        *,
        run_id: str,
        parent_run_id: str | None,
        parent_tool_call_id: str | None,
        agent_name: str | None,
        prompt_type: str | None,
        model: str,
        run_target: str | None,
    ) -> None:
        """Notify every subscriber that a model call is about to be made (#2439).

        Called from ``LlmClient.chat()`` BEFORE the network request — the one
        chokepoint every caller (the agent loop, a micro-context draw, the
        classifier, round framing, skill extraction) shares — so a connected
        addon can show a running row before any ``promptlog`` row exists for
        this call.  A no-op with nobody subscribed (no connected browser), the
        same guard every publish here uses."""
        if not self._call_started_listeners:
            return
        event = CallStarted(
            run_id=run_id,
            parent_run_id=parent_run_id,
            parent_tool_call_id=parent_tool_call_id,
            agent_name=agent_name,
            prompt_type=prompt_type,
            model=model,
            run_target=run_target,
            started_at=self._clock.now(),
        )
        for listener in tuple(self._call_started_listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("A call-started listener failed")

    def publish_tool_started(
        self, *, run_id: str, tool_call_id: str, tool_name: str, status: str
    ) -> None:
        """Notify every subscriber that a dispatched tool call is about to run
        (#2439).  Called from the dispatcher (``Agent._execute_single_tool``)
        before ``ToolExecutor.execute`` — a no-op with nobody subscribed."""
        if not self._tool_started_listeners:
            return
        event = ToolStarted(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            started_at=self._clock.now(),
        )
        for listener in tuple(self._tool_started_listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("A tool-started listener failed")

    def publish_tool_finished(
        self, *, run_id: str, tool_call_id: str, tool_name: str, status: str, success: bool
    ) -> None:
        """Notify every subscriber that a dispatched tool call just finished
        (#2439).  Called from the same dispatcher site, right after
        ``ToolExecutor.execute`` returns — a no-op with nobody subscribed."""
        if not self._tool_finished_listeners:
            return
        event = ToolFinished(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            success=success,
        )
        for listener in tuple(self._tool_finished_listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("A tool-finished listener failed")

    def _session(self) -> Session:
        return Session(self.engine)

    @staticmethod
    def strip_formatting(text: str) -> str:
        """Strip markdown formatting for quote lookup.

        Signal converts **bold**/etc. to native formatting, so quotes come back
        as plain text. We strip these markers to enable reliable matching.
        """
        text = _BOLD_ITALIC_RE.sub(r"\1", text)
        text = _STRIKETHROUGH_RE.sub(r"\1", text)
        text = _MONOSPACE_RE.sub(r"\1", text)
        text = text.replace(_TILDE_OPERATOR, "~")
        return text

    # --- Message logging ---

    def log_message(
        self,
        direction: str,
        sender: str,
        content: str,
        parent_id: int | None = None,
        signal_timestamp: int | None = None,
        external_id: str | None = None,
        is_reaction: bool = False,
        recipient: str | None = None,
        device_id: int | None = None,
        embedding: bytes | None = None,
        mechanism: str | None = None,
    ) -> int | None:
        """Log a user message or agent response. Returns the message ID or None.

        ``embedding`` (serialized float32) is stored at write time so the message
        is immediately searchable via the ``user-messages``/``penny-messages``
        facades' ``read_similar``/relevant-recall path — the startup backfill only
        catches rows logged without one.

        ``mechanism`` (#1568) names the autonomous cycle that produced an outgoing
        send — NULL for a direct reply."""
        if direction == PennyConstants.MessageDirection.OUTGOING:
            content = self.strip_formatting(content)
        try:
            with self._session() as session:
                log = MessageLog(
                    direction=direction,
                    sender=sender,
                    content=content,
                    parent_id=parent_id,
                    signal_timestamp=signal_timestamp,
                    external_id=external_id,
                    is_reaction=is_reaction,
                    recipient=recipient,
                    device_id=device_id,
                    embedding=embedding,
                    mechanism=mechanism,
                    timestamp=self._clock.now(),
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                logger.debug("Logged %s message from %s (id=%d)", direction, sender, log.id)
                return log.id
        except Exception as e:
            logger.error("Failed to log message: %s", e)
            return None

    def log_prompt(
        self,
        model: str,
        messages: list[dict],
        response: dict | None,
        tools: list[dict] | None = None,
        thinking: str | None = None,
        duration_ms: int | None = None,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
        run_target: str | None = None,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        call_outcome: str | None = None,
        call_error_class: str | None = None,
        call_error_message: str | None = None,
        call_error_fault: str | None = None,
    ) -> None:
        """Log a prompt/response exchange with Ollama.

        ``response`` is ``None`` for a call that never got one — cancelled, timed
        out, or raised (#2355 part A).  The row still carries the literal ``"{}"``
        (never a NULL ``response`` column — see migration 0114's own note on why),
        so every existing reader's ``if prompt.response else {}`` guard sees
        exactly what it already expects; the NEW ``call_outcome`` column is what
        actually distinguishes a call that never got a response from a completed
        one with a genuinely empty body (which never happens — a completed call
        always has at least one choice)."""
        response_payload = response if response is not None else {}
        try:
            with self._session() as session:
                log = PromptLog(
                    model=model,
                    messages=json.dumps(messages),
                    tools=json.dumps(tools) if tools else None,
                    response=json.dumps(response_payload),
                    thinking=thinking,
                    duration_ms=duration_ms,
                    agent_name=agent_name,
                    prompt_type=prompt_type,
                    run_id=run_id,
                    run_target=run_target,
                    parent_run_id=parent_run_id,
                    parent_tool_call_id=parent_tool_call_id,
                    call_outcome=call_outcome,
                    call_error_class=call_error_class,
                    call_error_message=call_error_message,
                    call_error_fault=call_error_fault,
                    timestamp=self._clock.now(),
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                logger.debug("Logged prompt exchange (model=%s)", model)
                if self._prompt_logged_listeners and run_id:
                    self._notify_prompt_logged(log, messages, response_payload)
        except Exception as e:
            logger.error("Failed to log prompt: %s", e)

    def _notify_prompt_logged(
        self, log: PromptLog, messages: list[dict], response_payload: dict
    ) -> None:
        """Publish a just-logged prompt to every subscriber.

        ``log`` already carries everything ``log_prompt`` was handed, so every
        scalar field is read straight off it; ``messages``/``response_payload``
        are passed as the live Python objects the caller already built —
        ``log`` stores them only as serialized JSON strings, and re-parsing
        them here would be pointless work."""
        input_tokens, output_tokens = self._extract_token_usage(response_payload)
        self._publish_prompt_logged(
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "model": log.model,
                "agent_name": log.agent_name or "",
                "prompt_type": log.prompt_type or "",
                "duration_ms": log.duration_ms or 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "run_id": log.run_id,
                "run_target": log.run_target,
                "messages": messages,
                "response": response_payload,
                "thinking": log.thinking or "",
                "has_tools": log.tools is not None,
                "made_tool_call": self._made_tool_call(response_payload),
                "parent_run_id": log.parent_run_id,
                "parent_tool_call_id": log.parent_tool_call_id,
                "call_outcome": log.call_outcome,
                "call_error_class": log.call_error_class,
                "call_error_message": log.call_error_message,
            }
        )

    def log_command(
        self,
        user: str,
        channel_type: str,
        command_name: str,
        command_args: str,
        response: str,
        error: str | None = None,
    ) -> None:
        """Log a command invocation."""
        try:
            with self._session() as session:
                log = CommandLog(
                    user=user,
                    channel_type=channel_type,
                    command_name=command_name,
                    command_args=command_args,
                    response=response,
                    error=error,
                    timestamp=self._clock.now(),
                )
                session.add(log)
                session.commit()
                logger.debug("Logged command: /%s %s", command_name, command_args)
        except Exception as e:
            logger.error("Failed to log command: %s", e)

    # --- Message metadata ---

    def set_signal_timestamp(self, message_id: int, signal_timestamp: int) -> None:
        """Update the Signal timestamp on a message after sending."""
        try:
            with self._session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.signal_timestamp = signal_timestamp
                    session.add(msg)
                    session.commit()
        except Exception as e:
            logger.error("Failed to set signal_timestamp: %s", e)

    def set_external_id(self, message_id: int, external_id: str) -> None:
        """Update the external ID on a message after sending."""
        try:
            with self._session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.external_id = external_id
                    session.add(msg)
                    session.commit()
        except Exception as e:
            logger.error("Failed to set external_id: %s", e)

    # --- Embeddings (startup backfill) ---

    def messages_without_embeddings(self, limit: int) -> list[MessageLog]:
        """Real messages still missing a content embedding (reactions excluded).

        ``messagelog.embedding`` powers ``read_similar`` over the user/penny
        message logs (now read facades over this table).  The startup backfill
        fills any gaps — historical rows + anything logged since the last run."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        col(MessageLog.embedding).is_(None),
                        col(MessageLog.is_reaction).is_(False),
                    )
                    .limit(limit)
                ).all()
            )

    def set_embedding(self, message_id: int, embedding: bytes) -> None:
        """Store a serialized content embedding on a message row."""
        with self._session() as session:
            message = session.get(MessageLog, message_id)
            if message is not None:
                message.embedding = embedding
                session.add(message)
                session.commit()

    # --- Message lookup ---

    def last_outgoing_content(self) -> str | None:
        """What Penny said most recently, or ``None`` if she hasn't yet.

        The conversation state machine's ``penny_last_turn`` (#1706): the newest
        message is a REPLY, and a reply is only classifiable against what it
        answers ("just the headline" is steps-arrived only against "what should
        I look for?")."""
        with self._session() as session:
            message = session.exec(
                select(MessageLog)
                .where(MessageLog.direction == PennyConstants.MessageDirection.OUTGOING)
                .order_by(*_NEWEST_MESSAGE_FIRST)
            ).first()
            return message.content if message is not None else None

    def get_by_id(self, message_id: int) -> MessageLog | None:
        """Get a message by its database ID."""
        with self._session() as session:
            return session.get(MessageLog, message_id)

    def get_by_ids(self, message_ids: set[int]) -> dict[int, MessageLog]:
        """Get multiple messages in one query, keyed by database ID."""
        if not message_ids:
            return {}
        with self._session() as session:
            messages = session.exec(
                select(MessageLog).where(col(MessageLog.id).in_(message_ids))
            ).all()
            return {message.id: message for message in messages if message.id is not None}

    def find_by_external_id(self, external_id: str) -> MessageLog | None:
        """Find a message by its platform-specific external ID."""
        with self._session() as session:
            return session.exec(
                select(MessageLog).where(MessageLog.external_id == external_id)
            ).first()

    def find_outgoing_by_content(self, content: str) -> MessageLog | None:
        """Find the most recent outgoing message matching the given content prefix."""
        content = self.strip_formatting(content)
        with self._session() as session:
            return session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    col(MessageLog.content).startswith(content),
                )
                .order_by(*_NEWEST_MESSAGE_FIRST)
            ).first()

    # --- Thread context ---

    def get_thread_context(
        self, quoted_text: str
    ) -> tuple[int | None, list[tuple[str, str]] | None]:
        """Look up a quoted message and return its id and conversation context."""
        parent_msg = self.find_outgoing_by_content(quoted_text)
        if not parent_msg or parent_msg.id is None:
            logger.warning("Could not find quoted message in database")
            return None, None

        thread = self._walk_thread(parent_msg.id)
        history: list[tuple[str, str]] = [
            (
                str(
                    MessageRole.USER
                    if m.direction == PennyConstants.MessageDirection.INCOMING
                    else MessageRole.ASSISTANT
                ),
                m.content,
            )
            for m in thread
        ]
        logger.info("Built thread history with %d messages", len(history))
        return parent_msg.id, history

    def _walk_thread(self, message_id: int, limit: int = 20) -> list[MessageLog]:
        """Walk up the parent chain. Returns messages oldest-first."""
        history: list[MessageLog] = []
        with self._session() as session:
            current_id: int | None = message_id
            while current_id is not None and len(history) < limit:
                msg = session.get(MessageLog, current_id)
                if msg is None:
                    break
                history.append(msg)
                current_id = msg.parent_id
        history.reverse()
        return history

    # --- Conversation queries ---

    def get_conversation_leaves(self) -> list[MessageLog]:
        """Get outgoing leaf messages eligible for spontaneous continuation."""
        with self._session() as session:
            has_child = select(MessageLog.parent_id).where(col(MessageLog.parent_id).isnot(None))
            incoming_ids = select(MessageLog.id).where(
                MessageLog.direction == PennyConstants.MessageDirection.INCOMING
            )
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                        col(MessageLog.id).notin_(has_child),
                        col(MessageLog.parent_id).in_(incoming_ids),
                    )
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                ).all()
            )

    def get_user_messages(self, sender: str, limit: int = 100) -> list[MessageLog]:
        """Get incoming messages from a specific user, oldest first."""
        with self._session() as session:
            messages = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    )
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                    .limit(limit)
                ).all()
            )
            messages.reverse()
            return messages

    def _get_threaded_replies(self, session: Any, incoming: list[MessageLog]) -> list[MessageLog]:
        """Fetch outgoing messages that are direct replies to the given incoming messages."""
        incoming_ids = [m.id for m in incoming if m.id is not None]
        if not incoming_ids:
            return []
        return list(
            session.exec(
                select(MessageLog).where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    col(MessageLog.parent_id).in_(incoming_ids),
                )
            ).all()
        )

    def _get_autonomous_outgoing(
        self, session: Any, recipient: str, since: datetime, limit: int
    ) -> list[MessageLog]:
        """Fetch autonomous outgoing messages (no parent thread) sent to a
        user.  ``since`` bounds the time window so the conversation
        builder doesn't drag in stale notifications from days ago."""
        return list(
            session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.parent_id == None,  # noqa: E711
                    MessageLog.recipient == recipient,
                    MessageLog.timestamp >= since,
                )
                .order_by(*_NEWEST_MESSAGE_FIRST)
                .limit(limit)
            ).all()
        )

    def get_messages_since(
        self, sender: str, since: datetime, limit: int = 100
    ) -> list[MessageLog]:
        """Get conversation messages since a timestamp, oldest first, capped at limit.

        Includes:
          - incoming user messages
          - Penny's threaded replies to those messages
          - autonomous outgoing sends (notifications, ``send_message`` from
            collector cycles) within the same window

        Autonomous sends are conversational events too — when the user
        replies to one, ``_build_conversation`` needs the prior turn so
        Penny knows what the reply is about.  Without this they'd be
        invisible to the chat turns array (no parent_id linking them to
        anything incoming) and Penny would see only the user's reply.

        The three legs are merged in ``_in_send_order`` — timestamp, then id
        for a tie.  On timestamp alone the merge is a stable sort over a
        concatenation, so rows sharing a microsecond come back in the order
        the legs happen to be joined in — every incoming row ahead of every
        outgoing one, whatever the write order was.  That is what swapped a
        reply and the message answering it in a seeded conversation (#1963).
        """
        with self._session() as session:
            incoming = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.timestamp >= since,
                    )
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                    .limit(limit)
                ).all()
            )
            threaded = self._get_threaded_replies(session, incoming)
            autonomous = self._get_autonomous_outgoing(session, sender, since, limit)
            all_messages = incoming + threaded + autonomous
            all_messages.sort(key=_in_send_order)
            return all_messages[-limit:]

    def chat_turn_tool_traces(self, limit: int = 500) -> list[ChatTurnTrace]:
        """Every chat turn's run, oldest first, as the history's tool line reads it:
        the run's last logged call, and the tool names the run's own prompt-log
        records show it called (the run record's projection, deduplicated).

        A chat turn's reply reaches ``messagelog`` without its ``run_id`` — the
        history therefore pairs each run with the first send at or after the run's
        last logged call, which is the pairing production produces (a turn replies
        right after its last call).  Bounded to the newest ``limit`` promptlog rows;
        a run straddling that window's oldest edge is dropped rather than rendered
        with a truncated call list."""
        with self._session() as session:
            rows = list(
                session.exec(
                    select(PromptLog)
                    .where(
                        PromptLog.agent_name == PennyConstants.CHAT_AGENT_NAME,
                        PromptLog.prompt_type == ChatPromptType.USER_MESSAGE.value,
                    )
                    .order_by(col(PromptLog.timestamp).desc(), col(PromptLog.id).desc())
                    .limit(limit)
                ).all()
            )
        rows.reverse()
        # Group by run id in call order; a row with no run id stands alone.
        groups: dict[tuple[str, int | str], list[PromptLog]] = {}
        for row in rows:
            key: tuple[str, int | str] = ("run", row.run_id) if row.run_id else ("row", row.id or 0)
            groups.setdefault(key, []).append(row)
        ordered = list(groups.values())
        if len(rows) == limit and ordered:
            ordered = ordered[1:]  # the oldest visible run may be truncated
        traces: list[ChatTurnTrace] = []
        for group_rows in ordered:
            last = group_rows[-1]
            traces.append(
                ChatTurnTrace(
                    last_call=last.timestamp,
                    row_id=last.id or 0,
                    tools=tuple(dict.fromkeys(name for name, _ in _run_tool_calls(group_rows))),
                )
            )
        return traces

    def ios_history_page(
        self,
        *,
        channel_types: list[str] | None,
        before: tuple[datetime, int] | None,
        limit: int,
    ) -> tuple[list[tuple[MessageLog, Device | None, IosOutboxItem | None]], bool]:
        """Return one newest-first boundary page for the iOS history surface.

        Device identifiers are used as a compatibility fallback because older
        message-log rows may not have a populated ``device_id``.
        """
        with self._session() as session:
            message_columns = inspect(MessageLog).local_table.c
            outbox_columns = inspect(IosOutboxItem).local_table.c
            devices = self._ios_scoped_devices(session, channel_types)
            if not devices:
                return [], False
            scope = self._ios_device_scope(message_columns, devices)
            query = self._ios_history_query(message_columns, scope, before)
            rows = list(
                session.exec(
                    query.order_by(
                        message_columns.timestamp.desc(), message_columns.id.desc()
                    ).limit(limit + 1)
                ).all()
            )
            has_more = len(rows) > limit
            ordered = list(reversed(rows[:limit]))
            outbox_by_message_id, outbox_by_id = self._ios_outbox_maps(
                session, outbox_columns, ordered
            )
            resolved = self._resolve_ios_rows(ordered, devices, outbox_by_message_id, outbox_by_id)
            return resolved, has_more

    @staticmethod
    def _ios_scoped_devices(session: Session, channel_types: list[str] | None) -> list[Device]:
        """Devices eligible for the iOS history surface, narrowed to the given
        channel types when any are named."""
        devices = list(session.exec(select(Device)).all())
        if channel_types:
            devices = [device for device in devices if device.channel_type in channel_types]
        return devices

    @staticmethod
    def _ios_device_scope(message_columns: Any, devices: list[Device]) -> Any:
        """The WHERE clause scoping messagelog rows to the given devices, by
        ``device_id`` or by sender/recipient identifier (the compatibility
        fallback for older rows with no populated ``device_id``)."""
        device_ids = [device.id for device in devices if device.id is not None]
        identifiers = [device.identifier for device in devices]
        return or_(
            message_columns.device_id.in_(device_ids),
            message_columns.sender.in_(identifiers),
            message_columns.recipient.in_(identifiers),
        )

    @staticmethod
    def _ios_history_query(
        message_columns: Any, scope: Any, before: tuple[datetime, int] | None
    ) -> Any:
        """The base newest-first history query for the given scope, optionally
        bounded to strictly before a ``(timestamp, id)`` page boundary.  The
        caller applies its own ordering + ``limit(limit + 1)`` on top (the
        page-size budget), so ordering/paging stay colocated at the call site."""
        query = (
            select(MessageLog, Device)
            .join(Device, isouter=True)
            .where(
                message_columns.direction.in_(
                    [
                        PennyConstants.MessageDirection.INCOMING,
                        PennyConstants.MessageDirection.OUTGOING,
                    ]
                ),
                message_columns.is_reaction.is_(False),
                scope,
            )
        )
        if before is not None:
            timestamp, message_id = before
            query = query.where(
                or_(
                    message_columns.timestamp < timestamp,
                    and_(
                        message_columns.timestamp == timestamp,
                        message_columns.id < message_id,
                    ),
                )
            )
        return query

    @staticmethod
    def _ios_outbox_maps(
        session: Session,
        outbox_columns: Any,
        ordered: list[tuple[MessageLog, Device | None]],
    ) -> tuple[dict[int, IosOutboxItem], dict[int, IosOutboxItem]]:
        """Outbox rows matching the page's messages, keyed both by
        ``message_log_id`` and by ``id`` (for legacy rows whose ``external_id``
        is itself an outbox id)."""
        message_ids = [message.id for message, _ in ordered if message.id is not None]
        outbox_ids = [
            int(message.external_id)
            for message, _ in ordered
            if message.external_id and message.external_id.isdecimal()
        ]
        outbox_rows = list(
            session.exec(
                select(IosOutboxItem).where(
                    or_(
                        outbox_columns.message_log_id.in_(message_ids),
                        outbox_columns.id.in_(outbox_ids),
                    )
                )
            ).all()
        )
        outbox_by_message_id = {
            row.message_log_id: row for row in outbox_rows if row.message_log_id is not None
        }
        outbox_by_id = {row.id: row for row in outbox_rows if row.id is not None}
        return outbox_by_message_id, outbox_by_id

    @staticmethod
    def _resolve_ios_rows(
        ordered: list[tuple[MessageLog, Device | None]],
        devices: list[Device],
        outbox_by_message_id: dict[int, IosOutboxItem],
        outbox_by_id: dict[int, IosOutboxItem],
    ) -> list[tuple[MessageLog, Device | None, IosOutboxItem | None]]:
        """Pair each page message with its device (falling back to an
        identifier match for legacy rows with no ``device_id``) and its
        outbox row, if any."""
        resolved: list[tuple[MessageLog, Device | None, IosOutboxItem | None]] = []
        for message, device in ordered:
            if device is None:
                device = next(
                    (
                        candidate
                        for candidate in devices
                        if candidate.identifier in {message.sender, message.recipient}
                    ),
                    None,
                )
            outbox = outbox_by_message_id.get(message.id)
            if outbox is None and message.external_id and message.external_id.isdecimal():
                outbox = outbox_by_id.get(int(message.external_id))
            resolved.append((message, device, outbox))
        return resolved

    def ios_history_count(self, *, channel_types: list[str] | None) -> int:
        """Count eligible history rows once at the start of an iOS sync."""
        with self._session() as session:
            message_columns = inspect(MessageLog).local_table.c
            devices = self._ios_scoped_devices(session, channel_types)
            if not devices:
                return 0
            scope = self._ios_device_scope(message_columns, devices)
            return int(
                session.exec(
                    select(func.count())
                    .select_from(MessageLog)
                    .where(
                        message_columns.direction.in_(
                            [
                                PennyConstants.MessageDirection.INCOMING,
                                PennyConstants.MessageDirection.OUTGOING,
                            ]
                        ),
                        message_columns.is_reaction.is_(False),
                        scope,
                    )
                ).one()
            )

    def get_unprocessed(self, sender: str, limit: int) -> list[MessageLog]:
        """Get recent unprocessed non-reaction messages from a specific user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                    .limit(limit)
                ).all()
            )

    def get_user_reactions(self, sender: str, limit: int) -> list[MessageLog]:
        """Get recent unprocessed reactions from a specific user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                    .limit(limit)
                ).all()
            )

    def mark_processed(self, message_ids: list[int]) -> None:
        """Mark multiple messages as processed."""
        if not message_ids:
            return
        try:
            with self._session() as session:
                for message_id in message_ids:
                    msg = session.get(MessageLog, message_id)
                    if msg:
                        msg.processed = True
                        session.add(msg)
                session.commit()
                logger.debug("Marked %d messages as processed", len(message_ids))
        except Exception as e:
            logger.error("Failed to mark messages as processed: %s", e)

    # --- Aggregate queries ---

    def count(self) -> int:
        """Count total number of messages."""
        with self._session() as session:
            return session.exec(select(func.count()).select_from(MessageLog)).one()

    def count_active_threads(self) -> int:
        """Count leaf messages (those with no children)."""
        with self._session() as session:
            has_child = select(MessageLog.parent_id).where(col(MessageLog.parent_id).isnot(None))
            return session.exec(
                select(func.count())
                .select_from(MessageLog)
                .where(col(MessageLog.id).notin_(has_child))
            ).one()

    def set_run_outcome(
        self,
        run_id: str,
        outcome: str,
        reason: str,
        tool_failures: int = 0,
    ) -> None:
        """Set the run outcome (a ``RunOutcome`` value + reason) on the last
        prompt log row for ``run_id``.  Drives the outcome badge on the prompts
        tab.  The run's ``run_target`` is stamped on every prompt at write time
        (see ``log_prompt``), so it isn't set here.

        ``tool_failures`` is the run's count of failed tool calls, stamped on the
        same last row so the run-health classifier can read it structurally
        rather than parsing tool-result text."""
        try:
            with self._session() as session:
                last_prompt = self._last_run_prompt(session, run_id)
                if last_prompt:
                    last_prompt.run_outcome = outcome
                    last_prompt.run_reason = reason
                    last_prompt.tool_failures = tool_failures
                    session.add(last_prompt)
                    session.commit()
                    if self._run_outcome_set_listeners:
                        self._publish_run_outcome_set(run_id, outcome, reason)
        except Exception as e:
            logger.error("Failed to set run outcome for %s: %s", run_id, e)

    def set_run_trailing_messages(self, run_id: str, messages: list[dict]) -> None:
        """Stamp the run's TRAILING turns — the conversation the agent loop appended
        after its final model call — onto the last prompt row for ``run_id`` (#1778).

        A tool result is otherwise durable only as a side effect of being fed back:
        it rides into the NEXT call's ``messages`` and is written down there.  A run
        that ends immediately after executing a tool (a write-gate STOP, ``max_steps``
        on a tool step, a reroll abort, an exception) has no next call, so its
        terminal call's outcome was recorded nowhere — and those are exactly the runs
        worth reading afterwards.  Stamping the tail here makes the record complete
        however the loop ended: the run renderers and the eval transcript extractor
        read ``messages`` + this tail as one sequence, so neither can be missing an
        entry.

        The tail is stored in the SAME wire shape ``messages`` uses (it is literally
        what the next call's ``messages`` would have ended with), so one decode path
        serves both and the two can't drift.  Empty tail → no write at all, so a run
        that round-tripped every result is untouched and renders byte-identically.

        Best-effort like ``log_prompt``: a failed record write must never fail the
        run it records."""
        if not messages:
            return
        try:
            with self._session() as session:
                last_prompt = self._last_run_prompt(session, run_id)
                if last_prompt:
                    last_prompt.trailing_messages = json.dumps(messages)
                    session.add(last_prompt)
                    session.commit()
        except Exception as e:
            logger.error("Failed to persist trailing messages for %s: %s", run_id, e)

    @staticmethod
    def _last_run_prompt(session: Session, run_id: str) -> PromptLog | None:
        """The run's final prompt row — the one row per run that carries the run's
        derived stamps (outcome, tool failures, the trailing turns)."""
        return session.exec(
            select(PromptLog)
            .where(PromptLog.run_id == run_id)
            .order_by(col(PromptLog.timestamp).desc())
            .limit(1)
        ).first()

    def recent_conversation(self, limit: int) -> list[tuple[str, str]]:
        """The last ``limit`` conversational messages (both directions, reactions
        excluded) as ``(direction, content)``, OLDEST first — the turns that led
        up to the present run.  The skill-naming step reads the user's INTENT
        from these (why the routine exists), not just the demonstration's how
        (#1658 intent grounding)."""
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MessageLog)
                    .where(MessageLog.is_reaction == False)  # noqa: E712
                    .order_by(*_NEWEST_MESSAGE_FIRST)
                    .limit(limit)
                ).all()
            )
        return [(row.direction, row.content) for row in reversed(rows)]

    def recent_run_outcomes(self, run_target: str, limit: int) -> list[tuple[datetime, str]]:
        """A collector's own most recent completed runs as ``(timestamp, outcome)``,
        newest first — what its previous invocations did, and when (#1569).

        The outcome line is STRUCTURAL, generated from the ledger: the run's
        ``run_reason`` when it carries one (a write-gate stop reason, or the
        no-``done()`` close reason), else the ``run_outcome`` enum.  Never a
        model-authored ``done()`` summary — ``done()`` is argless, so
        ``run_reason`` is empty on a clean close and the outcome enum shows
        instead.  Each run stamps ``run_outcome`` on exactly one prompt row, so the
        completion rows ARE the run index — one row per run, served by
        ``ix_promptlog_target_runs`` (a bounded ``ORDER BY ... LIMIT``, not a scan).
        Cancelled runs (preempted by a foreground message — not a real cycle
        outcome) are excluded.  The completion-row timestamp is the run's finish
        time.
        """
        if limit <= 0:
            return []
        with self._session() as session:
            rows = session.exec(
                select(PromptLog.timestamp, PromptLog.run_outcome, PromptLog.run_reason)
                .where(
                    col(PromptLog.run_outcome).isnot(None),
                    PromptLog.run_target == run_target,
                    PromptLog.run_outcome != RunOutcome.CANCELLED.value,
                )
                .order_by(col(PromptLog.timestamp).desc())
                .limit(limit)
            ).all()
        return [(timestamp, reason or outcome) for timestamp, outcome, reason in rows if outcome]

    def count_completed_runs(self, run_target: str) -> int:
        """How many completed (non-cancelled) cycles this collector has run.

        Each run stamps ``run_outcome`` on exactly one prompt row, so the
        completion rows ARE the run count — one row per run.  Cancelled runs
        (preempted by a foreground message, not a real cycle) are excluded, so a
        preemption never burns a ``max_runs`` allotment.  Read from the ledger,
        never re-decided by the model — the run-quota retire gate (#1556, now the
        schedule's own ``COUNT=``, #1857).
        """
        with self._session() as session:
            return session.exec(
                select(func.count())
                .select_from(PromptLog)
                .where(
                    col(PromptLog.run_outcome).isnot(None),
                    PromptLog.run_target == run_target,
                    PromptLog.run_outcome != RunOutcome.CANCELLED.value,
                )
            ).one()

    def run_call_groups(
        self, target: str, cursor: datetime | None, limit: int
    ) -> list[list[PromptLog]]:
        """Recent runs for ``target`` as raw prompt groups, oldest-first — the raw
        material ``run_list`` renders into tool-call sequences (``render_run_calls``
        does the shaping; this only selects and groups).

        ``target`` is either ``"chat"`` (conversational chat-agent runs — chat stamps no
        ``run_outcome``, so the run index is the distinct ``run_id``s under
        ``agent_name='chat'``) or a collector/collection name (that collector's completed
        runs, keyed on ``run_target`` + a non-null, non-cancelled ``run_outcome``).  One
        group per run, its representative time the run's last prompt.  Cursor semantics
        mirror ``Log.read_batch``: no cursor → the most recent ``limit`` (returned
        oldest-first); a cursor → the next ``limit`` since it.
        """
        if limit <= 0:
            return []
        finished = func.max(PromptLog.timestamp)
        with self._session() as session:
            query = self._run_call_groups_query(target, finished)
            rows = self._run_id_window(session, query, finished, cursor, limit)
            run_ids = [run_id for run_id, _ in rows if run_id is not None]
            if not run_ids:
                return []
            grouped = self._group_runs(session, run_ids)
        return [grouped[run_id] for run_id in run_ids if run_id in grouped]

    @staticmethod
    def _run_call_groups_query(target: str, finished: Any) -> Any:
        """The base run_id-grouped query for ``run_call_groups``: the chat
        agent's distinct runs, or one collector's completed, non-cancelled
        runs."""
        query = (
            select(PromptLog.run_id, finished.label("finished"))
            .where(col(PromptLog.run_id).isnot(None))
            .group_by(PromptLog.run_id)
        )
        if target == PennyConstants.CHAT_AGENT_NAME:
            return query.where(PromptLog.agent_name == PennyConstants.CHAT_AGENT_NAME)
        return query.where(
            PromptLog.run_target == target,
            col(PromptLog.run_outcome).isnot(None),
            PromptLog.run_outcome != RunOutcome.CANCELLED.value,
        )

    @staticmethod
    def _run_id_window(
        session: Session,
        query: Any,
        finished: Any,
        cursor: datetime | None,
        limit: int,
    ) -> list[Any]:
        """Cursor semantics mirroring ``Log.read_batch``: no cursor returns the
        most recent ``limit`` rows (oldest-first); a cursor returns the next
        ``limit`` since it."""
        if cursor is not None:
            return session.exec(
                query.having(finished > cursor).order_by(finished.asc()).limit(limit)
            ).all()
        return list(reversed(session.exec(query.order_by(finished.desc()).limit(limit)).all()))

    def get_run_prompts(self, run_id: str) -> list[PromptLog]:
        """Every prompt of one run, ascending time order — the raw material the
        run-end skill extractor (#1658) projects into ordinaled tool calls (#1590).
        Empty when the run id is unknown."""
        with self._session() as session:
            return list(
                session.exec(
                    select(PromptLog)
                    .where(PromptLog.run_id == run_id)
                    .order_by(col(PromptLog.timestamp).asc())
                ).all()
            )

    def get_child_draws(self, run_id: str) -> list[PromptLog]:
        """Every model call a micro-context made ON BEHALF of ``run_id`` (#2355 part
        D) — a browse-extract draw, say, found by ``parent_run_id`` rather than
        ``run_id`` (a micro-context mints its own, for its own ledger attribution),
        ascending time order.  ``render_run_calls`` nests each under the step it was
        made for.  Empty when the run made no such calls (the ordinary case)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(PromptLog)
                    .where(PromptLog.parent_run_id == run_id)
                    .order_by(col(PromptLog.timestamp).asc())
                ).all()
            )

    def recent_collector_runs(self, limit: int) -> list[RunActivity]:
        """The most recent completed collector runs across ALL collections, newest
        first — the run half of the self-state header's activity block (#1555).

        Scoped to ``run_target IS NOT NULL`` — collector cycles alone; an ORDINARY
        chat turn still stamps no ``run_outcome`` at all and so is excluded by
        construction (the header renders the *complement* of the conversation —
        background runs, not the turns already in context).  A chat turn that
        FAILED is the one exception (#2355 part D): it stamps ``run_outcome`` too,
        but never ``run_target`` (a chat run names no collection), so it is still
        excluded HERE and surfaced instead through ``recent_failed_chat_turns`` —
        its own query, not a relaxed filter on this one, so an ordinary collector
        activity line's shape is untouched.  Cancelled runs (foreground-preempted,
        not a real cycle) are excluded, like every other run index.  Each run's
        outcome lands on exactly one prompt row (its last, via ``set_run_outcome``),
        so that row's timestamp is the finish time and one row = one run.  A second
        bounded query groups those runs' prompts to count their tool calls (the
        rollup summary)."""
        if limit <= 0:
            return []
        with self._session() as session:
            rows = session.exec(self._recent_collector_run_rows_query(limit)).all()
            run_ids = [run_id for run_id, _, _, _ in rows if run_id is not None]
            grouped = self._group_runs(session, run_ids) if run_ids else {}
        return self._run_activities(rows, grouped)

    @staticmethod
    def _recent_collector_run_rows_query(limit: int) -> Any:
        """The newest-first completed-collector-run rows (run_id, target,
        outcome, finished_at), ``id`` breaking same-timestamp ties
        deterministically so the activity render is stable across runs."""
        return (
            select(
                PromptLog.run_id,
                PromptLog.run_target,
                PromptLog.run_outcome,
                PromptLog.timestamp,
            )
            .where(
                col(PromptLog.run_outcome).isnot(None),
                PromptLog.run_outcome != RunOutcome.CANCELLED.value,
                col(PromptLog.run_target).isnot(None),
            )
            .order_by(col(PromptLog.timestamp).desc(), col(PromptLog.id).desc())
            .limit(limit)
        )

    @staticmethod
    def _run_activities(rows: Any, grouped: dict[str, list[PromptLog]]) -> list[RunActivity]:
        """Rows → ``RunActivity``, with each run's tool-call rollup count read
        off its grouped prompts."""
        return [
            RunActivity(
                run_id=run_id,
                target=target,
                outcome=outcome,
                finished_at=finished,
                call_count=_count_run_tool_calls(grouped.get(run_id, [])),
            )
            for run_id, target, outcome, finished in rows
            if run_id is not None and target is not None and outcome is not None
        ]

    def recent_failed_chat_turns(self, limit: int) -> list[ChatFailureActivity]:
        """The most recent chat turns stamped FAILED, newest first — the chat half
        of the self-state header's recent-activity block (#2355 part D).

        A chat run stamps ``run_outcome`` only through
        ``ChatAgent._record_turn_failure`` (never ``WORKED``/``NO_WORK`` — an
        ordinary successful reply is the conversation already, not this
        complement's business), so ``run_outcome = FAILED`` alone is the filter;
        ``run_target IS NULL`` is what tells a chat run apart from a collector's
        (which always stamps one), and is what keeps this query's rows out of
        ``recent_collector_runs``'s own (which requires a target)."""
        if limit <= 0:
            return []
        with self._session() as session:
            rows = session.exec(
                select(PromptLog.run_id, PromptLog.run_reason, PromptLog.timestamp)
                .where(
                    PromptLog.run_outcome == RunOutcome.FAILED.value,
                    col(PromptLog.run_target).is_(None),
                )
                .order_by(col(PromptLog.timestamp).desc(), col(PromptLog.id).desc())
                .limit(limit)
            ).all()
        return [
            ChatFailureActivity(run_id=run_id, reason=reason or "", finished_at=finished)
            for run_id, reason, finished in rows
            if run_id is not None
        ]

    def recent_emissions(self, limit: int) -> list[EmissionActivity]:
        """The most recent delivered autonomous sends across ALL mechanisms, newest
        first — the emission half of the self-state header's activity block (#1568).

        Only outgoing rows carrying a ``mechanism`` (an autonomous send that named
        its cause) qualify; a direct reply stamps NULL and is excluded (it is the
        conversation, already in context).  The snippet is the whitespace-collapsed
        head of the content."""
        if limit <= 0:
            return []
        with self._session() as session:
            rows = session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    col(MessageLog.mechanism).isnot(None),
                )
                .order_by(*_NEWEST_MESSAGE_FIRST)
                .limit(limit)
            ).all()
        return [
            EmissionActivity(
                mechanism=row.mechanism,  # ty: ignore[invalid-argument-type]
                sent_at=row.timestamp,
                snippet=_emission_snippet(row.content),
            )
            for row in rows
            if row.mechanism is not None
        ]

    def latest_mechanism_send(self, mechanism: str) -> MessageLog | None:
        """The newest outgoing message this mechanism sent, or ``None`` if never.

        Read off the emission provenance the drainer stamps (#1568): the outgoing
        row whose ``mechanism`` names the collection, newest first.  This is what
        the collector's "what you last told the user" line renders (#2385) — the
        anchor the close's ``notify`` choice reads against, so "have I already said
        this?" is a read, not a recollection.  Served by the partial
        ``ix_messagelog_emission_time`` index (mechanism-bearing rows are sparse)."""
        with self._session() as session:
            return session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.mechanism == mechanism,
                )
                .order_by(*_NEWEST_MESSAGE_FIRST)
            ).first()

    def latest_run_outcomes(self) -> dict[str, RunOutcomeStamp]:
        """Each collector's most recent completed run outcome + finish time, keyed
        by ``run_target`` — one grouped query (#1555).

        The self-state mechanism inventory reads one of these per mechanism for
        its 'last run' line.  ``run_outcome`` is the value from the row carrying
        the max timestamp per target (SQLite resolves a bare column alongside
        ``MAX()`` to that row).  Cancelled runs are excluded so 'last run' reports
        the last real cycle, not a preemption."""
        with self._session() as session:
            rows = session.exec(
                select(
                    PromptLog.run_target,
                    PromptLog.run_outcome,
                    func.max(PromptLog.timestamp),
                )
                .where(
                    col(PromptLog.run_outcome).isnot(None),
                    PromptLog.run_outcome != RunOutcome.CANCELLED.value,
                    col(PromptLog.run_target).isnot(None),
                )
                .group_by(PromptLog.run_target)
            ).all()
        return {
            target: RunOutcomeStamp(outcome=outcome, finished_at=finished)
            for target, outcome, finished in rows
            if target is not None and outcome is not None
        }

    @staticmethod
    def _group_runs(session: Session, run_ids: list[str]) -> dict[str, list[PromptLog]]:
        """Load every prompt of the given runs, grouped by ``run_id`` (each run's
        prompts in ascending time order — the trace order ``render_run_record``
        expects)."""
        grouped: dict[str, list[PromptLog]] = {}
        prompts = session.exec(
            select(PromptLog)
            .where(col(PromptLog.run_id).in_(run_ids))
            .order_by(col(PromptLog.timestamp).asc())
        ).all()
        for prompt in prompts:
            if prompt.run_id is not None:
                grouped.setdefault(prompt.run_id, []).append(prompt)
        return grouped

    # A micro-context can itself be moved to run inside another one (a draw whose
    # own ``parent_run_id`` names another draw's ``run_id``, not a top-level run) —
    # the code owner may rearrange these at any time with no code change here, so
    # the walk below follows the chain to WHATEVER depth it actually goes, capped
    # only so a runaway chain can't make one page's query unbounded.
    _CHILD_DRAW_MAX_DEPTH = 8

    @staticmethod
    def _group_child_draws(session: Session, run_ids: list[str]) -> dict[str, list[PromptLog]]:
        """Every child draw (#2434) made on behalf of any of the given TOP-LEVEL
        runs, AT ANY DEPTH (#2435) — the batch form of ``get_child_draws``, one
        query per LEVEL for a whole page of runs instead of one per run.

        Walks outward from ``run_ids`` by ``parent_run_id`` a level at a time
        (each level's query keyed on the previous level's own run_ids), so a
        draw dispatched from INSIDE another draw is found exactly like one
        dispatched from a top-level run — nothing here is keyed to which
        micro-context a row's ``agent_name`` names.  Every row is kept, even
        when several share one run_id (a micro-context's own poison-reroll
        attempts keep one run_id across tries), then each is attributed back to
        the top-level run it ultimately descends from by walking its own
        ``parent_run_id`` chain up through what this page just found.

        Grouped by that top-level run_id, ascending time order (the order
        ``_serialize_run`` renders them in) — a FLAT list per top-level run,
        same shape #2434 shipped: every entry still carries its own
        ``parent_run_id``/``parent_tool_call_id``, so the addon builds the
        whole nesting tree itself, to any depth, from those ids alone."""
        top_level_ids = set(run_ids)
        all_draws, parent_of = MessageStore._walk_child_draw_levels(session, top_level_ids)
        return MessageStore._attribute_to_top_level(all_draws, parent_of, top_level_ids)

    @staticmethod
    def _walk_child_draw_levels(
        session: Session, top_level_ids: set[str]
    ) -> tuple[list[PromptLog], dict[str, str]]:
        """Walk ``parent_run_id`` outward from ``top_level_ids`` one level at a
        time (capped at ``_CHILD_DRAW_MAX_DEPTH``), collecting every descendant
        draw plus each run_id's own immediate parent_run_id."""
        all_draws: list[PromptLog] = []
        parent_of: dict[str, str] = {}
        frontier = top_level_ids
        for _ in range(MessageStore._CHILD_DRAW_MAX_DEPTH):
            if not frontier:
                break
            level = session.exec(
                select(PromptLog)
                .where(col(PromptLog.parent_run_id).in_(frontier))
                .order_by(col(PromptLog.timestamp).asc())
            ).all()
            if not level:
                break
            next_frontier: set[str] = set()
            for draw in level:
                if draw.run_id is None or draw.parent_run_id is None:
                    continue
                all_draws.append(draw)
                if draw.run_id not in parent_of:
                    next_frontier.add(draw.run_id)
                parent_of[draw.run_id] = draw.parent_run_id
            frontier = next_frontier
        return all_draws, parent_of

    @staticmethod
    def _attribute_to_top_level(
        all_draws: list[PromptLog], parent_of: dict[str, str], top_level_ids: set[str]
    ) -> dict[str, list[PromptLog]]:
        """Group every draw under the top-level run it ultimately descends
        from (via ``_top_level_origin``), each group in ascending time order."""
        grouped: dict[str, list[PromptLog]] = {}
        for draw in all_draws:
            origin = MessageStore._top_level_origin(draw.parent_run_id, parent_of, top_level_ids)
            if origin is not None:
                grouped.setdefault(origin, []).append(draw)
        for draws in grouped.values():
            draws.sort(key=lambda d: d.timestamp)
        return grouped

    @staticmethod
    def _top_level_origin(
        parent_run_id: str | None, parent_of: dict[str, str], top_level_ids: set[str]
    ) -> str | None:
        """Walk a draw's own ``parent_run_id`` chain up to the top-level run it
        ultimately descends from, through the parent-of-each-run_id map
        ``_group_child_draws`` already built for this page.  ``None`` for a
        chain this page never fully resolved — its own origin fell outside
        ``run_ids``, or the chain ran past ``_CHILD_DRAW_MAX_DEPTH``."""
        current = parent_run_id
        for _ in range(MessageStore._CHILD_DRAW_MAX_DEPTH):
            if current is None:
                return None
            if current in top_level_ids:
                return current
            current = parent_of.get(current)
        return None

    def get_prompt_log_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        agent_name: str | None = None,
        query: str | None = None,
        flagged_only: bool = False,
    ) -> list[dict]:
        """Get prompt logs grouped by run_id, newest first.

        Returns a list of run summaries with their individual prompts.
        Pagination happens at the run level in SQL: stage one selects only
        the requested page of run_ids (ordered by each run's newest prompt),
        stage two loads the heavy prompt rows for just those runs.  This
        keeps the query cost proportional to the page size, not to the whole
        (multi-GB) promptlog table.

        ``query`` filters to runs that have at least one prompt whose
        ``response`` or ``thinking`` (the output the run produced — not its
        shared input scaffolding) matches the text.

        ``flagged_only`` keeps only runs the run-health classifier marks
        regressive (a bail / incomplete / tool-failure / half-formed send),
        paging over that filtered stream — ``offset`` then counts flagged runs,
        matching the addon's offset-by-displayed-count model.
        """
        with self._session() as session:
            if flagged_only:
                return self._flagged_runs(session, agent_name, query)
            run_ids_ordered = self._page_of_run_ids(session, limit, offset, agent_name, query)
            return self._runs_for(session, run_ids_ordered)

    def get_run(self, run_id: str) -> dict | None:
        """One run, fully serialized (run → prompts), by its ``run_id`` — the
        fetch-one counterpart to ``get_prompt_log_runs``/``get_target_runs``'s
        pages, for a caller that already holds the id (a skill's
        ``source_run_id``, say) and wants to open it directly rather than paging
        to find it.  ``None`` when no prompt in the ledger carries that run_id."""
        with self._session() as session:
            runs = self._runs_for(session, [run_id])
        return runs[0] if runs else None

    def get_target_runs(self, run_target: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Full serialized runs for one collection's collector, newest-first —
        the per-collection Activity panel (run → prompts → turns, the same shape
        the prompts tab renders).  Each completed run stamps ``run_outcome`` on
        exactly one row (its last prompt), so the completion rows ARE the run
        index — one per run, served by ``ix_promptlog_target_runs`` (a bounded
        ``ORDER BY ... LIMIT``, not a scan), matching the old record-only panel's
        filter (``run_outcome IS NOT NULL AND run_target = ?``)."""
        with self._session() as session:
            run_ids = self._page_of_target_run_ids(session, run_target, limit, offset)
            return self._runs_for(session, run_ids)

    @staticmethod
    def _page_of_target_run_ids(
        session: Session, run_target: str, limit: int, offset: int
    ) -> list[str]:
        """One newest-first page of completed run_ids for ``run_target``."""
        rows = session.exec(
            select(PromptLog.run_id)
            .where(
                col(PromptLog.run_outcome).isnot(None),
                PromptLog.run_target == run_target,
            )
            .order_by(col(PromptLog.timestamp).desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [run_id for run_id in rows if run_id is not None]

    def _runs_for(self, session: Session, run_ids_ordered: list[str]) -> list[dict]:
        """Load + serialize the given runs (heavy prompt rows), preserving order.

        Also batches each run's child draws (#2434) in the same page — one extra
        query, not one per run — so every fetched run carries its own micro-context
        draws alongside its prompts.

        Every existing caller first selects ``run_ids_ordered`` off this same
        table, so each id always has at least one prompt — except ``get_run``,
        whose id comes from outside (a skill's ``source_run_id``, say) with no
        guarantee it ever existed.  So a requested id with no prompt rows is
        skipped rather than assumed present (#2443)."""
        if not run_ids_ordered:
            return []
        grouped: dict[str, list[PromptLog]] = {}
        prompts = session.exec(
            select(PromptLog)
            .where(col(PromptLog.run_id).in_(run_ids_ordered))
            .order_by(col(PromptLog.timestamp).asc())
        ).all()
        for prompt in prompts:
            if prompt.run_id is None:
                continue
            grouped.setdefault(prompt.run_id, []).append(prompt)
        child_draws_by_run = self._group_child_draws(session, run_ids_ordered)
        runs = []
        for run_id in run_ids_ordered:
            run_prompts = grouped.get(run_id)
            if run_prompts is None:
                continue
            # Model time: this run's OWN prompts only — _serialize_run keeps
            # child draws out of this sum on purpose (#2438).
            total_duration_ms = sum(p.duration_ms or 0 for p in run_prompts)
            runs.append(
                self._serialize_run(
                    run_id, run_prompts, total_duration_ms, child_draws_by_run.get(run_id)
                )
            )
        return runs

    # The flagged-only triage is a view of RECENT regressions, so it scans a
    # bounded window of the newest runs rather than the whole multi-GB history.
    # Classification reads only light columns (no messages/thinking), so this
    # window is cheap to sweep; the window comfortably covers many days of runs.
    _FLAGGED_SCAN_RUNS = 1000
    _FLAGGED_SCAN_PAGE = 200

    def _flagged_runs(
        self,
        session: Session,
        agent_name: str | None,
        query: str | None,
    ) -> list[dict]:
        """Every regressive run in the recent-runs window, newest-first.

        Sweeps the newest ``_FLAGGED_SCAN_RUNS`` runs, classifying each from a
        LIGHT column read (no ``messages``/``thinking`` — the heavy fields), then
        heavy-serializes only the flagged ones.  Single-shot (no pagination): a
        triage of recent regressions, not a page into all history."""
        flagged_ids: list[str] = []
        scanned = 0
        while scanned < self._FLAGGED_SCAN_RUNS:
            run_ids = self._page_of_run_ids(
                session, self._FLAGGED_SCAN_PAGE, scanned, agent_name, query
            )
            if not run_ids:
                break
            flagged_ids.extend(self._regressive_among(session, run_ids))
            scanned += len(run_ids)
            if len(run_ids) < self._FLAGGED_SCAN_PAGE:
                break
        return self._runs_for(session, flagged_ids)

    @staticmethod
    def _regressive_among(session: Session, run_ids: list[str]) -> list[str]:
        """The subset of ``run_ids`` whose run is regressive, in input order.

        Classifies from the LIGHT columns only — the run's tool calls
        (``response``) plus the ``run_outcome``/``tool_failures`` the classifier's
        boolean flags read — so the flagged sweep never loads the heavy
        ``messages`` scaffolding.  Order within a run is irrelevant to the flags
        (bail = any non-done call; degenerate = any bad send; the outcome and
        failure count sit on a single row), so no ``ORDER BY`` is needed."""
        sql = text(
            "SELECT run_id, response, run_outcome, tool_failures "
            "FROM promptlog WHERE run_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        grouped: dict[str, list[PromptLog]] = {}
        for run_id, response, run_outcome, tool_failures in session.execute(
            sql, {"ids": run_ids}
        ).all():
            light = PromptLog(
                model="",
                messages="",
                # A NULL response (no model reply logged) means no tool calls —
                # classify_run reads "" as an empty call set, which is correct.
                response=response if response is not None else "",
                run_id=run_id,
                run_outcome=run_outcome,
                tool_failures=tool_failures,
            )
            grouped.setdefault(run_id, []).append(light)
        return [run_id for run_id in run_ids if classify_run(grouped.get(run_id, [])).regressive]

    @staticmethod
    def _page_of_run_ids(
        session: Session,
        limit: int,
        offset: int,
        agent_name: str | None,
        search: str | None = None,
    ) -> list[str]:
        """Return one page of TOP-LEVEL run_ids, ordered newest-first by each
        run's latest activity.  Touches only the indexed run_id/parent_run_id/
        timestamp columns — no heavy JSON payloads — so it stays cheap as the
        table grows.

        ``search`` keeps only runs with a prompt whose ``response`` or
        ``thinking`` matches it via the ``promptlog_fts`` full-text index
        (migration 0051) — a per-word prefix MATCH, so it stays fast on the
        multi-GB table instead of scanning every JSON blob.
        """
        if search:
            return MessageStore._page_of_run_ids_fts(session, limit, offset, agent_name, search)
        query = MessageStore._top_level_run_ids_by_activity(agent_name).limit(limit).offset(offset)
        return [run_id for run_id in session.exec(query).all() if run_id is not None]

    @staticmethod
    def _top_level_run_ids_by_activity(agent_name: str | None) -> SelectOfScalar[str]:
        """Every top-level run_id (#2435 — one with at least one prompt of its
        OWN, never only a micro-context draw made on another run's behalf),
        ordered by latest activity across both its own prompts AND any child
        draw (#2434) made on its behalf — so a run whose own steps are stale
        but whose draw just landed still sorts to the top, matching the live
        push's reorder-on-update.

        A run_id with zero top-level prompts of its own (every row a child
        draw) never enters this result at all — the bug this exists to fix, a
        draw's own run_id surfacing as its own anonymous top-level run.
        ``agent_name`` filters the TOP-LEVEL rows only; a run's children always
        come along with it regardless of their own agent_name — the addon
        attaches them client-side once the page is loaded.

        Built from two grouped subqueries rather than a raw SQL union, so the
        "latest of two sources" comparison stays expressed through SQLAlchemy:
        SQLite's scalar (two-plus-argument) ``MAX()`` — distinct from its
        single-argument AGGREGATE ``MAX()`` — picks the larger of a run's own
        latest timestamp and its latest child draw's, coalescing a run with no
        draws (no match on the outer join) back to its own latest."""
        own_activity = MessageStore._own_activity_subquery(agent_name)
        draw_activity = MessageStore._draw_activity_subquery()
        latest_activity = func.max(
            own_activity.c.own_max,
            func.coalesce(draw_activity.c.draw_max, own_activity.c.own_max),
        )
        return (
            select(own_activity.c.run_id)
            .outerjoin(draw_activity, draw_activity.c.run_id == own_activity.c.run_id)
            .order_by(latest_activity.desc())
        )

    @staticmethod
    def _own_activity_subquery(agent_name: str | None) -> Any:
        """Each top-level run's own latest prompt timestamp, optionally
        narrowed to one agent."""
        own = select(
            col(PromptLog.run_id).label("run_id"),
            func.max(PromptLog.timestamp).label("own_max"),
        ).where(col(PromptLog.run_id).isnot(None), col(PromptLog.parent_run_id).is_(None))
        if agent_name:
            own = own.where(PromptLog.agent_name == agent_name)
        return own.group_by(PromptLog.run_id).subquery()

    @staticmethod
    def _draw_activity_subquery() -> Any:
        """Each run's latest child-draw timestamp (#2434), keyed by the
        parent run_id the draw was made on behalf of."""
        return (
            select(
                col(PromptLog.parent_run_id).label("run_id"),
                func.max(PromptLog.timestamp).label("draw_max"),
            )
            .where(col(PromptLog.parent_run_id).isnot(None))
            .group_by(PromptLog.parent_run_id)
            .subquery()
        )

    @staticmethod
    def _page_of_run_ids_fts(
        session: Session, limit: int, offset: int, agent_name: str | None, search: str
    ) -> list[str]:
        """Run-id page for a full-text search, newest-first by latest activity,
        via promptlog_fts — the search-path twin of
        ``_top_level_run_ids_by_activity``.  Raw SQL throughout (not just for the
        latest-activity comparison): ``promptlog_fts`` is a virtual table with no
        SQLModel mapping, so the MATCH join already has to be text().

        The user's text becomes a per-word prefix query (``morning news`` →
        ``morning* news*``, implicit AND).  With no searchable word characters
        there is nothing to match, so return no runs."""
        match = " ".join(f"{token}*" for token in re.findall(r"\w+", search.lower()))
        if not match:
            return []
        agent_clause = "AND p.agent_name = :agent" if agent_name else ""
        sql = text(
            "WITH child_activity AS ("
            "  SELECT parent_run_id AS run_id, MAX(timestamp) AS draw_max"
            "  FROM promptlog WHERE parent_run_id IS NOT NULL GROUP BY parent_run_id"
            ") "
            "SELECT p.run_id FROM promptlog p "
            "JOIN promptlog_fts f ON f.rowid = p.id "
            "LEFT JOIN child_activity c ON c.run_id = p.run_id "
            f"WHERE p.run_id IS NOT NULL AND p.parent_run_id IS NULL "
            f"AND promptlog_fts MATCH :q {agent_clause} "
            "GROUP BY p.run_id "
            "ORDER BY MAX(MAX(p.timestamp), COALESCE(MAX(c.draw_max), MAX(p.timestamp))) DESC "
            "LIMIT :limit OFFSET :offset"
        )
        params: dict[str, Any] = {"q": match, "limit": limit, "offset": offset}
        if agent_name:
            params["agent"] = agent_name
        rows = session.execute(sql, params).all()
        return [row[0] for row in rows if row[0] is not None]

    def recent_prompts(self, limit: int = 200) -> list[PromptLog]:
        """The most recent prompt-log rows, newest first — for inspection/eval."""
        with self._session() as session:
            return list(
                session.exec(
                    select(PromptLog).order_by(col(PromptLog.timestamp).desc()).limit(limit)
                ).all()
            )

    def prompt_perf(self) -> PromptPerf:
        """Aggregate wall time + token usage across every logged prompt.

        Reads the timing the real LLM path already records (``duration_ms`` per
        call) plus the token usage stored inside each response — the eval suite
        sums this across a case's samples to report throughput (tok/s).
        """
        with self._session() as session:
            rows = list(session.exec(select(PromptLog)).all())
        input_tokens = 0
        output_tokens = 0
        thinking_chars = 0
        output_chars = 0
        reasoning_tokens = 0
        for row in rows:
            response = json.loads(row.response) if row.response else {}
            prompt_tokens, completion_tokens = self._extract_token_usage(response)
            input_tokens += prompt_tokens
            output_tokens += completion_tokens
            reasoning_tokens += self._extract_reasoning_tokens(response)
            thinking_chars += len(row.thinking or "")
            output_chars += len(self._extract_content(response))
        duration_ms = sum(row.duration_ms or 0 for row in rows)
        return PromptPerf(
            len(rows),
            duration_ms,
            input_tokens,
            output_tokens,
            thinking_chars,
            output_chars,
            reasoning_tokens,
        )

    @staticmethod
    def _extract_content(response: dict) -> str:
        """The visible assistant text of a response (excludes the reasoning trace)."""
        choices = response.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content") or ""

    @staticmethod
    def _extract_token_usage(response: dict) -> tuple[int, int]:
        """Extract prompt and completion token counts from an OpenAI response."""
        usage = response.get("usage")
        if not usage:
            return 0, 0
        return usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0

    @staticmethod
    def _made_tool_call(response: dict) -> bool:
        """Whether this call's response actually INVOKED a tool, as opposed to
        merely being offered some (``has_tools``) — the addon step icon's
        tool-vs-plain-reply discriminator (#2434).  Reads the same
        ``choices[0].message.tool_calls`` shape ``_run_logged_steps`` sweeps
        across a whole run, for one response."""
        choices = response.get("choices") or []
        if not choices:
            return False
        message = choices[0].get("message") or {}
        return bool(message.get("tool_calls"))

    @staticmethod
    def _extract_reasoning_tokens(response: dict) -> int:
        """The provider's own count of the reasoning part of ``completion_tokens``.

        Reported by providers that serve reasoning models under
        ``usage.completion_tokens_details.reasoning_tokens``; 0 where a backend omits it,
        which the reader treats as "not reported" and falls back to the character ratio.
        """
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return details.get("reasoning_tokens", 0) or 0

    @staticmethod
    def _serialize_prompt_record(p: PromptLog) -> dict:
        """One ``promptlog`` row in the addon's per-prompt wire shape — shared by a
        run's own prompts and its child draws (#2434), which carry the identical
        shape so the addon nests the latter under the former by matching
        ``parent_tool_call_id`` itself, with no second schema to keep in sync.

        ``made_tool_call`` is this ONE response's own tool-call bit (whether the
        call actually invoked one), distinct from ``has_tools`` (whether tools were
        merely offered).  ``parent_run_id``/``parent_tool_call_id`` link a child
        draw back to the run/step it was made for (``None`` for an ordinary
        top-level call); ``call_outcome``/``call_error_class``/``call_error_message``
        carry how the call itself ended.  ``run_id`` is carried on every record
        (not only a draw's) so a draw's OWN children — a micro-context nested
        inside another, to any depth (#2435) — can be found by matching their
        ``parent_run_id`` against THIS id, purely structurally."""
        response = json.loads(p.response) if p.response else {}
        input_tokens, output_tokens = MessageStore._extract_token_usage(response)
        return {
            "id": p.id,
            "run_id": p.run_id,
            "timestamp": p.timestamp.isoformat(),
            "model": p.model,
            "agent_name": p.agent_name or "",
            "prompt_type": p.prompt_type or "",
            "duration_ms": p.duration_ms or 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "run_target": p.run_target,
            "messages": json.loads(p.messages) if p.messages else [],
            "response": response,
            "thinking": p.thinking or "",
            "has_tools": p.tools is not None,
            "made_tool_call": MessageStore._made_tool_call(response),
            "parent_run_id": p.parent_run_id,
            "parent_tool_call_id": p.parent_tool_call_id,
            "call_outcome": p.call_outcome,
            "call_error_class": p.call_error_class,
            "call_error_message": p.call_error_message,
        }

    @staticmethod
    def _run_window(rows: list[PromptLog]) -> tuple[datetime, datetime]:
        """The structural start/end of a run: the earliest ``timestamp -
        duration_ms`` and the latest ``timestamp`` across EVERY row handed to it
        (#2438).  ``PromptLog.timestamp`` is stamped when a call is logged, i.e.
        when it ENDS, so a row's own start is recovered by subtracting how long it
        took; a null ``duration_ms`` (a row logged before the column existed, or a
        call that never finished) is read as 0, same as the duration sums
        elsewhere in this module.

        Callers hand this the run's own prompts AND its child draws together (a
        flat list, any nesting depth) — nothing here is keyed to an agent or tool
        name, so it adapts to however many micro-contexts a cycle makes without a
        code change."""
        starts = [p.timestamp - timedelta(milliseconds=p.duration_ms or 0) for p in rows]
        ends = [p.timestamp for p in rows]
        return min(starts), max(ends)

    @staticmethod
    def _serialize_run(
        run_id: str,
        prompts: list[PromptLog],
        total_duration_ms: int,
        child_draws: list[PromptLog] | None = None,
    ) -> dict:
        """Serialize a single run, its own prompts, and its child draws (#2434 —
        the model calls a micro-context made ON BEHALF of this run, found by
        ``get_child_draws``/``_group_child_draws``) to a dict.

        ``child_draws`` is a FLAT list in the same per-prompt shape as ``prompts``
        — each entry already carries its own ``parent_tool_call_id`` (or ``None``
        for the five micro-context customers that aren't nested inside a dispatched
        tool call), so the addon nests them itself rather than this needing a
        second, run-shaped schema.

        Two timings, kept deliberately separate (#2438): ``wall_duration_ms`` is
        the real elapsed time of the whole run tree — ``started_at`` to
        ``ended_at``, both computed by ``_run_window`` over this run's own
        prompts AND its child draws, so a browse round whose extraction read
        pages for minutes shows that whole span.  ``total_duration_ms`` stays
        exactly what it always was — this run's OWN prompts' summed model-call
        time, EXCLUDING child draws — labelled "model time" by the addon; a
        child draw can run long after the model itself has gone quiet (reading a
        page, say), and keeping it out of ``total_duration_ms`` keeps that number
        paired with ``total_output_tokens`` for an honest tok/s."""
        all_rows = [*prompts, *(child_draws or [])]
        started_at, ended_at = MessageStore._run_window(all_rows)
        wall_duration_ms = int((ended_at - started_at).total_seconds() * 1000)
        serialized_prompts, total_input_tokens, total_output_tokens = (
            MessageStore._serialize_run_prompts(prompts)
        )
        run_outcome, run_reason = MessageStore._run_outcome_and_reason(prompts)

        # The bound collection is fixed for the whole cycle and stamped on every
        # prompt at write time, so all prompts in a run carry the same run_target
        # — read it off the first one.  It must NOT be coupled to the outcome
        # (only the last prompt carries that); doing so dropped run_target for
        # outcome-less runs (in-progress / never-tagged) and the addon then
        # rendered the bare agent identity ("collector") instead of the name.
        run_target = prompts[0].run_target

        # Run health + concise record: the SAME representation the self-state
        # header and the ``collector-runs`` read facade render of a run
        # (render_run_record / classify_run), so the addon's badges + "flagged
        # only" filter and Penny's ambient view of her own runs draw from one
        # classifier.  ``record`` is copy-pasteable straight back to a deeper
        # analysis.
        return {
            "run_id": run_id,
            "agent_name": prompts[0].agent_name or "unknown",
            "prompt_count": len(prompts),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_duration_ms": total_duration_ms,
            "wall_duration_ms": wall_duration_ms,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "run_outcome": run_outcome,
            "run_reason": run_reason,
            "run_target": run_target,
            "health": classify_run(prompts).model_dump(),
            "record": render_run_record(prompts),
            "prompts": serialized_prompts,
            "child_draws": [
                MessageStore._serialize_prompt_record(draw) for draw in (child_draws or [])
            ],
        }

    @staticmethod
    def _serialize_run_prompts(prompts: list[PromptLog]) -> tuple[list[dict], int, int]:
        """Each prompt serialized (``_serialize_prompt_record``), plus this
        run's own summed input/output tokens — excludes child draws, the same
        scope as ``total_duration_ms`` (#2438)."""
        serialized_prompts = []
        total_input_tokens = 0
        total_output_tokens = 0
        for p in prompts:
            record = MessageStore._serialize_prompt_record(p)
            total_input_tokens += record["input_tokens"]
            total_output_tokens += record["output_tokens"]
            serialized_prompts.append(record)
        return serialized_prompts, total_input_tokens, total_output_tokens

    @staticmethod
    def _run_outcome_and_reason(prompts: list[PromptLog]) -> tuple[str | None, str | None]:
        """The run's outcome + reason, read off the last prompt that carries
        one (every run stamps exactly one row, via ``set_run_outcome``)."""
        for p in reversed(prompts):
            if p.run_outcome is not None or p.run_reason:
                return p.run_outcome, p.run_reason
        return None, None
