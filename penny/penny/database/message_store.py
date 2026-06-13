"""Message store — logging, threading, and queries for messages."""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from penny.agents.models import MessageRole
from penny.constants import PennyConstants
from penny.database.models import CommandLog, MessageLog, PromptLog

logger = logging.getLogger(__name__)

# Patterns for stripping markdown formatting from outgoing messages
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_STRIKETHROUGH_RE = re.compile(r"~{1,2}(.+?)~{1,2}")
_MONOSPACE_RE = re.compile(r"`(.+?)`")
_TILDE_OPERATOR = "\u223c"


class MessageStore:
    """Manages MessageLog, PromptLog, and CommandLog records."""

    def __init__(self, engine):
        self.engine = engine
        self._on_prompt_logged: Callable[[dict], None] | None = None
        self._on_run_outcome_set: Callable[[str, bool, str, str | None], None] | None = None

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
        thought_id: int | None = None,
        device_id: int | None = None,
    ) -> int | None:
        """Log a user message or agent response. Returns the message ID or None."""
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
                    thought_id=thought_id,
                    device_id=device_id,
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
        response: dict,
        tools: list[dict] | None = None,
        thinking: str | None = None,
        duration_ms: int | None = None,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log a prompt/response exchange with Ollama."""
        try:
            with self._session() as session:
                log = PromptLog(
                    model=model,
                    messages=json.dumps(messages),
                    tools=json.dumps(tools) if tools else None,
                    response=json.dumps(response),
                    thinking=thinking,
                    duration_ms=duration_ms,
                    agent_name=agent_name,
                    prompt_type=prompt_type,
                    run_id=run_id,
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                logger.debug("Logged prompt exchange (model=%s)", model)
                if self._on_prompt_logged and run_id:
                    input_tokens, output_tokens = self._extract_token_usage(response)
                    self._on_prompt_logged(
                        {
                            "id": log.id,
                            "timestamp": log.timestamp.isoformat(),
                            "model": model,
                            "agent_name": agent_name or "",
                            "prompt_type": prompt_type or "",
                            "duration_ms": duration_ms or 0,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "run_id": run_id,
                            "messages": messages,
                            "response": response,
                            "thinking": thinking or "",
                            "has_tools": tools is not None,
                        }
                    )
        except Exception as e:
            logger.error("Failed to log prompt: %s", e)

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

    # --- Message lookup ---

    def get_by_id(self, message_id: int) -> MessageLog | None:
        """Get a message by its database ID."""
        with self._session() as session:
            return session.get(MessageLog, message_id)

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
                    MessageLog.content.startswith(content),
                )
                .order_by(MessageLog.timestamp.desc())
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
            has_child = select(MessageLog.parent_id).where(MessageLog.parent_id.isnot(None))
            incoming_ids = select(MessageLog.id).where(
                MessageLog.direction == PennyConstants.MessageDirection.INCOMING
            )
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                        MessageLog.id.notin_(has_child),
                        MessageLog.parent_id.in_(incoming_ids),
                    )
                    .order_by(MessageLog.timestamp.desc())
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
                    .order_by(MessageLog.timestamp.desc())
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
                    MessageLog.parent_id.in_(incoming_ids),
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
                .order_by(MessageLog.timestamp.desc())
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
                    .order_by(MessageLog.timestamp.desc())
                    .limit(limit)
                ).all()
            )
            threaded = self._get_threaded_replies(session, incoming)
            autonomous = self._get_autonomous_outgoing(session, sender, since, limit)
            all_messages = incoming + threaded + autonomous
            all_messages.sort(key=lambda m: m.timestamp)
            return all_messages[-limit:]

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
                    .order_by(MessageLog.timestamp.desc())
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
                    .order_by(MessageLog.timestamp.desc())
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
            has_child = select(MessageLog.parent_id).where(MessageLog.parent_id.isnot(None))
            return session.exec(
                select(func.count()).select_from(MessageLog).where(MessageLog.id.notin_(has_child))
            ).one()

    def set_run_outcome(
        self,
        run_id: str,
        success: bool,
        reason: str,
        target: str | None = None,
    ) -> None:
        """Set the run outcome (success / reason / target) on the last prompt
        log row for ``run_id``.  Drives the green/red tag on the prompts
        tab.  ``target`` is the collection name for collector cycles, None
        for other agents."""
        try:
            with self._session() as session:
                last_prompt = session.exec(
                    select(PromptLog)
                    .where(PromptLog.run_id == run_id)
                    .order_by(PromptLog.timestamp.desc())
                    .limit(1)
                ).first()
                if last_prompt:
                    last_prompt.run_success = success
                    last_prompt.run_reason = reason
                    last_prompt.run_target = target
                    session.add(last_prompt)
                    session.commit()
                    if self._on_run_outcome_set:
                        self._on_run_outcome_set(run_id, success, reason, target)
        except Exception as e:
            logger.error("Failed to set run outcome for %s: %s", run_id, e)

    def get_prompt_log_agent_names(self) -> list[str]:
        """Get distinct agent names from prompt logs."""
        with self._session() as session:
            rows = session.exec(
                select(PromptLog.agent_name)
                .where(
                    PromptLog.run_id.isnot(None),  # ty: ignore[unresolved-attribute]
                    PromptLog.agent_name.isnot(None),  # ty: ignore[unresolved-attribute]
                )
                .distinct()
            ).all()
            return sorted(name for name in rows if name)

    def get_prompt_log_runs(
        self, limit: int = 50, offset: int = 0, agent_name: str | None = None
    ) -> list[dict]:
        """Get prompt logs grouped by run_id, newest first.

        Returns a list of run summaries with their individual prompts.
        Pagination happens at the run level in SQL: stage one selects only
        the requested page of run_ids (ordered by each run's newest prompt),
        stage two loads the heavy prompt rows for just those runs.  This
        keeps the query cost proportional to the page size, not to the whole
        (multi-GB) promptlog table.
        """
        with self._session() as session:
            run_ids_ordered = self._page_of_run_ids(session, limit, offset, agent_name)
            if not run_ids_ordered:
                return []

            grouped: dict[str, list[PromptLog]] = {}
            prompts = session.exec(
                select(PromptLog)
                .where(PromptLog.run_id.in_(run_ids_ordered))  # ty: ignore[unresolved-attribute]
                .order_by(PromptLog.timestamp.asc())
            ).all()
            for prompt in prompts:
                if prompt.run_id is None:
                    continue
                grouped.setdefault(prompt.run_id, []).append(prompt)

            runs = []
            for run_id in run_ids_ordered:
                run_prompts = grouped[run_id]
                total_duration_ms = sum(p.duration_ms or 0 for p in run_prompts)
                runs.append(self._serialize_run(run_id, run_prompts, total_duration_ms))
            return runs

    @staticmethod
    def _page_of_run_ids(
        session: Session, limit: int, offset: int, agent_name: str | None
    ) -> list[str]:
        """Return one page of run_ids, ordered newest-first by each run's most
        recent prompt.  Touches only the indexed run_id/timestamp columns — no
        heavy JSON payloads — so it stays cheap as the table grows."""
        query = select(PromptLog.run_id).where(PromptLog.run_id.isnot(None))  # ty: ignore[unresolved-attribute]
        if agent_name:
            query = query.where(PromptLog.agent_name == agent_name)
        query = (
            query.group_by(PromptLog.run_id)
            .order_by(func.max(PromptLog.timestamp).desc())
            .limit(limit)
            .offset(offset)
        )
        return [run_id for run_id in session.exec(query).all() if run_id is not None]

    @staticmethod
    def _extract_token_usage(response: dict) -> tuple[int, int]:
        """Extract prompt and completion token counts from an OpenAI response."""
        usage = response.get("usage")
        if not usage:
            return 0, 0
        return usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0

    @staticmethod
    def _serialize_run(
        run_id: str,
        prompts: list[PromptLog],
        total_duration_ms: int,
    ) -> dict:
        """Serialize a single run and its prompts to a dict."""
        total_input_tokens = 0
        total_output_tokens = 0
        serialized_prompts = []
        for p in prompts:
            response = json.loads(p.response) if p.response else {}
            input_tokens, output_tokens = MessageStore._extract_token_usage(response)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            serialized_prompts.append(
                {
                    "id": p.id,
                    "timestamp": p.timestamp.isoformat(),
                    "model": p.model,
                    "agent_name": p.agent_name or "",
                    "prompt_type": p.prompt_type or "",
                    "duration_ms": p.duration_ms or 0,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "messages": json.loads(p.messages) if p.messages else [],
                    "response": response,
                    "thinking": p.thinking or "",
                    "has_tools": p.tools is not None,
                }
            )

        # Run outcome is set on the last prompt that has one
        run_success: bool | None = None
        run_reason: str | None = None
        run_target: str | None = None
        for p in reversed(prompts):
            if p.run_success is not None or p.run_reason:
                run_success = p.run_success
                run_reason = p.run_reason
                run_target = p.run_target
                break

        return {
            "run_id": run_id,
            "agent_name": prompts[0].agent_name or "unknown",
            "prompt_count": len(prompts),
            "started_at": prompts[0].timestamp.isoformat(),
            "ended_at": prompts[-1].timestamp.isoformat(),
            "total_duration_ms": total_duration_ms,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "run_success": run_success,
            "run_reason": run_reason,
            "run_target": run_target,
            "prompts": serialized_prompts,
        }
