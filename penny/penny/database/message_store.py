"""Message store — logging, threading, and queries for messages."""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, NamedTuple

from sqlalchemy import bindparam, func, text
from sqlmodel import Session, select

from penny.agents.models import MessageRole
from penny.constants import PennyConstants, RunOutcome
from penny.database.memory.objects import classify_run, render_run_record
from penny.database.models import CommandLog, MessageLog, PromptLog

logger = logging.getLogger(__name__)

# Patterns for stripping markdown formatting from outgoing messages
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_STRIKETHROUGH_RE = re.compile(r"~{1,2}(.+?)~{1,2}")
_MONOSPACE_RE = re.compile(r"`(.+?)`")
_TILDE_OPERATOR = "\u223c"


class PromptPerf(NamedTuple):
    """Aggregate wall time + token usage across logged prompts.

    Sourced from data the real LLM path already records \u2014 ``duration_ms`` per
    call plus the token usage stored inside each response \u2014 so the eval suite
    can report throughput without any new instrumentation.

    ``output_tokens`` (the OpenAI ``completion_tokens``) already *includes* the
    reasoning trace, so to expose how much of the generation was reasoning we
    also carry the character lengths of the stored ``thinking`` trace and the
    visible ``content``; their ratio gives the reasoning share of generation.
    """

    calls: int
    duration_ms: int
    input_tokens: int
    output_tokens: int
    thinking_chars: int = 0
    output_chars: int = 0

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


class MessageStore:
    """Manages MessageLog, PromptLog, and CommandLog records."""

    def __init__(self, engine):
        self.engine = engine
        self._on_prompt_logged: Callable[[dict], None] | None = None
        self._on_run_outcome_set: Callable[[str, str, str], None] | None = None

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
        embedding: bytes | None = None,
    ) -> int | None:
        """Log a user message or agent response. Returns the message ID or None.

        ``embedding`` (serialized float32) is stored at write time so the message
        is immediately searchable via the ``user-messages``/``penny-messages``
        facades' ``read_similar``/relevant-recall path — the startup backfill only
        catches rows logged without one."""
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
                    embedding=embedding,
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
        run_target: str | None = None,
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
                    run_target=run_target,
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
                            "run_target": run_target,
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
                        MessageLog.embedding.is_(None),  # ty: ignore[unresolved-attribute]
                        MessageLog.is_reaction.is_(False),  # ty: ignore[unresolved-attribute]
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
                last_prompt = session.exec(
                    select(PromptLog)
                    .where(PromptLog.run_id == run_id)
                    .order_by(PromptLog.timestamp.desc())
                    .limit(1)
                ).first()
                if last_prompt:
                    last_prompt.run_outcome = outcome
                    last_prompt.run_reason = reason
                    last_prompt.tool_failures = tool_failures
                    session.add(last_prompt)
                    session.commit()
                    if self._on_run_outcome_set:
                        self._on_run_outcome_set(run_id, outcome, reason)
        except Exception as e:
            logger.error("Failed to set run outcome for %s: %s", run_id, e)

    def recent_run_summaries(self, run_target: str, limit: int) -> list[tuple[datetime, str]]:
        """A collector's own most recent completed runs as ``(timestamp, summary)``,
        newest first — what its previous invocations did, and when.

        Each run stamps ``run_outcome`` + ``run_reason`` (the ``done()`` summary)
        on exactly one prompt row, so the completion rows ARE the run index — one
        row per run, served by ``ix_promptlog_target_runs`` (a bounded
        ``ORDER BY ... LIMIT``, not a scan).  Cancelled runs (preempted by a
        foreground message — not a real cycle outcome) are excluded.  The
        completion-row timestamp is the run's finish time.
        """
        if limit <= 0:
            return []
        with self._session() as session:
            rows = session.exec(
                select(PromptLog.timestamp, PromptLog.run_reason)
                .where(
                    PromptLog.run_outcome.isnot(None),  # ty: ignore[unresolved-attribute]
                    PromptLog.run_target == run_target,
                    PromptLog.run_outcome != RunOutcome.CANCELLED.value,
                )
                .order_by(PromptLog.timestamp.desc())
                .limit(limit)
            ).all()
        return [(timestamp, reason) for timestamp, reason in rows if reason]

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
                PromptLog.run_outcome.isnot(None),  # ty: ignore[unresolved-attribute]
                PromptLog.run_target == run_target,
            )
            .order_by(PromptLog.timestamp.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [run_id for run_id in rows if run_id is not None]

    def _runs_for(self, session: Session, run_ids_ordered: list[str]) -> list[dict]:
        """Load + serialize the given runs (heavy prompt rows), preserving order."""
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
        """Return one page of run_ids, ordered newest-first by each run's most
        recent prompt.  Touches only the indexed run_id/timestamp columns — no
        heavy JSON payloads — so it stays cheap as the table grows.

        ``search`` keeps only runs with a prompt whose ``response`` or
        ``thinking`` matches it via the ``promptlog_fts`` full-text index
        (migration 0051) — a per-word prefix MATCH, so it stays fast on the
        multi-GB table instead of scanning every JSON blob.
        """
        if search:
            return MessageStore._page_of_run_ids_fts(session, limit, offset, agent_name, search)
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
    def _page_of_run_ids_fts(
        session: Session, limit: int, offset: int, agent_name: str | None, search: str
    ) -> list[str]:
        """Run-id page for a full-text search, newest-first, via promptlog_fts.

        The user's text becomes a per-word prefix query (``morning news`` →
        ``morning* news*``, implicit AND).  With no searchable word characters
        there is nothing to match, so return no runs."""
        match = " ".join(f"{token}*" for token in re.findall(r"\w+", search.lower()))
        if not match:
            return []
        agent_clause = "AND p.agent_name = :agent" if agent_name else ""
        sql = text(
            "SELECT p.run_id FROM promptlog p "
            "JOIN promptlog_fts f ON f.rowid = p.id "
            f"WHERE p.run_id IS NOT NULL AND promptlog_fts MATCH :q {agent_clause} "
            "GROUP BY p.run_id ORDER BY MAX(p.timestamp) DESC LIMIT :limit OFFSET :offset"
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
                    select(PromptLog).order_by(PromptLog.timestamp.desc()).limit(limit)
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
        for row in rows:
            response = json.loads(row.response) if row.response else {}
            prompt_tokens, completion_tokens = self._extract_token_usage(response)
            input_tokens += prompt_tokens
            output_tokens += completion_tokens
            thinking_chars += len(row.thinking or "")
            output_chars += len(self._extract_content(response))
        duration_ms = sum(row.duration_ms or 0 for row in rows)
        return PromptPerf(
            len(rows), duration_ms, input_tokens, output_tokens, thinking_chars, output_chars
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
                    "run_target": p.run_target,
                    "messages": json.loads(p.messages) if p.messages else [],
                    "response": response,
                    "thinking": p.thinking or "",
                    "has_tools": p.tools is not None,
                }
            )

        # Run outcome is set on the last prompt that has one
        run_outcome: str | None = None
        run_reason: str | None = None
        for p in reversed(prompts):
            if p.run_outcome is not None or p.run_reason:
                run_outcome = p.run_outcome
                run_reason = p.run_reason
                break

        # The bound collection is fixed for the whole cycle and stamped on every
        # prompt at write time, so all prompts in a run carry the same run_target
        # — read it off the first one.  It must NOT be coupled to the outcome
        # (only the last prompt carries that); doing so dropped run_target for
        # outcome-less runs (in-progress / never-tagged) and the addon then
        # rendered the bare agent identity ("collector") instead of the name.
        run_target = prompts[0].run_target

        # Run health + concise record: the SAME representation Penny's quality
        # collector reads of her own runs (render_run_record / classify_run), so
        # the addon's badges + "flagged only" filter and Penny's self-review draw
        # from one classifier.  ``record`` is copy-pasteable straight back to a
        # deeper analysis.
        return {
            "run_id": run_id,
            "agent_name": prompts[0].agent_name or "unknown",
            "prompt_count": len(prompts),
            "started_at": prompts[0].timestamp.isoformat(),
            "ended_at": prompts[-1].timestamp.isoformat(),
            "total_duration_ms": total_duration_ms,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "run_outcome": run_outcome,
            "run_reason": run_reason,
            "run_target": run_target,
            "health": classify_run(prompts).model_dump(),
            "record": render_run_record(prompts),
            "prompts": serialized_prompts,
        }
