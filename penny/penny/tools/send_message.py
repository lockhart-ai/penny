"""SendMessageTool — model-driven outbound message delivery.

Bound at construction to a (channel, agent_name) pair plus the database
and runtime config.  The recipient is always the primary user (Penny is
single-user) and is resolved from ``db`` at execute time, not plumbed
through agent construction.  The model calls this tool with a message
body when it has decided what to say.  The tool checks three gates
before dispatching:

- **Refusal**: if the content is itself a model refusal ("I'm sorry,
  I can't..."), don't dispatch — that's not a real reply.  Tells
  the model to call ``done`` instead.
- **Mute**: if the recipient has muted notifications, the tool
  refuses with a string that tells the model to call ``done``.
- **Cooldown**: flat interval between autonomous sends from the same
  agent.  Bypassed when the user has replied since the agent's last
  send — the next send is then conversational, not autonomous.
  Otherwise the cooldown is ``SEND_COOLDOWN_SECONDS`` (runtime-tunable).
- **Truncation**: if the content tail looks like a model self-
  truncation (ending in ``…`` or three-or-more dots, mid-thought),
  return a failure string with the ``Error:`` prefix so the agent
  loop marks the call as failed.  The model sees the rejection on
  its next step and re-emits with the complete body.

The first three are graceful-exit signals (model is told to call
``done``); truncation is a retry signal (model is told to redo
``send_message`` with the complete body).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants
from penny.llm.refusal import is_refusal
from penny.tools.base import Tool
from penny.tools.memory_tools import DoneTool
from penny.tools.models import SendMessageArgs

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel
    from penny.config import Config
    from penny.database import Database

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a message to the user through the bound channel."""

    name = "send_message"
    description = (
        "Send a message to the user.  Use this once you have decided "
        "what to say — the ``content`` is the exact text the user will "
        "see.  The send is gated on refusal detection, mute state, and "
        "a flat-interval cooldown between autonomous sends; if any "
        f"refuses, the response will say so and you should call "
        f"``{DoneTool.name}`` to exit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text to send to the user.",
            }
        },
        "required": ["content"],
    }

    _REFUSAL_RESPONSE = (
        "Message NOT sent: the content reads as a model refusal "
        "(\"I'm sorry, I can't...\") rather than a substantive reply.  "
        f"Call ``{DoneTool.name}`` to exit — do not retry with the same content."
    )
    _MUTED_RESPONSE = (
        "Message NOT sent: the user has muted autonomous messages.  "
        f'Call ``{DoneTool.name}(success=true, summary="muted — skipped")`` '
        "to exit — do not retry.  This is normal cycle behaviour, not a failure."
    )
    _COOLDOWN_RESPONSE = (
        "Message NOT sent: cooldown has not elapsed since the last "
        f'autonomous send.  Call ``{DoneTool.name}(success=true, summary="cooldown — '
        'skipped this cycle")`` to exit — do not retry.  This is normal '
        "cycle behaviour, not a failure."
    )
    # ``Error:`` prefix triggers ``record.failed=True`` in the agent loop,
    # which counts toward the abort threshold so we don't infinite-loop
    # if the model keeps producing truncated content.
    _TRUNCATION_REJECTION = (
        "Error: Message NOT sent: the content ended with an ellipsis "
        "('…' or '...'), which means it was cut off mid-thought.  "
        "Call send_message again with the COMPLETE message body — "
        "finish every sentence and bullet you start, no ellipses, "
        "no 'etc.', no 'and more', no teaser phrasing."
    )

    def __init__(
        self,
        channel: MessageChannel,
        agent_name: str,
        db: Database,
        config: Config,
    ) -> None:
        self._channel = channel
        self._agent_name = agent_name
        self._db = db
        self._config = config

    async def execute(self, **kwargs: Any) -> str:
        args = SendMessageArgs(**kwargs)
        if is_refusal(args.content):
            logger.info("send_message refused (refusal content): %s", self._agent_name)
            return self._REFUSAL_RESPONSE
        if _appears_truncated(args.content):
            logger.info("send_message rejected (truncation): %s", self._agent_name)
            return self._TRUNCATION_REJECTION
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            logger.info("send_message refused (no primary user): %s", self._agent_name)
            return self._REFUSAL_RESPONSE
        if self._db.users.is_muted(recipient):
            logger.info("send_message refused (muted): %s", recipient)
            return self._MUTED_RESPONSE
        if not self._cooldown_elapsed():
            logger.info("send_message refused (cooldown): %s → %s", self._agent_name, recipient)
            return self._COOLDOWN_RESPONSE
        await self._channel.send_response(
            recipient=recipient,
            content=args.content,
            parent_id=None,
            author=self._agent_name,
            quote_message=None,
        )
        logger.info("send_message: %s → %s", self._agent_name, recipient)
        return "Message sent."

    # ── Gating helpers ──────────────────────────────────────────────────

    def _cooldown_elapsed(self) -> bool:
        """Flat-interval cooldown between *autonomous* sends.

        The gate stops a background agent from spamming the user when
        there's been no reply.  When ``count == 0`` the user has spoken
        since Penny last sent, so the next message is conversational, not
        autonomous — no cooldown applies.

        Otherwise the next send must wait ``SEND_COOLDOWN_SECONDS`` since
        Penny's previous outgoing message.  A message has two authors —
        Penny or the user — so the cooldown is per-Penny, not per-internal-
        agent: any recent Penny send (a chat reply included) holds off an
        autonomous ping.
        """
        count = self._count_sends_since_user_message()
        if count == 0:
            return True
        latest = self._latest_send_time()
        if latest is None:
            return True
        elapsed = (_naive_utc_now() - _to_naive(latest)).total_seconds()
        return elapsed >= self._config.runtime.SEND_COOLDOWN_SECONDS

    def _latest_send_time(self) -> datetime | None:
        """Created-at of Penny's most recent outgoing message."""
        entries = self._db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG, k=1)
        return entries[0].created_at if entries else None

    def _count_sends_since_user_message(self) -> int:
        """Number of Penny's outgoing messages newer than the latest user message."""
        latest_user = self._latest_user_message_time()
        cutoff = _to_naive(latest_user) if latest_user is not None else None
        count = 0
        for entry in self._db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG):
            if cutoff is not None and _to_naive(entry.created_at) <= cutoff:
                break
            count += 1
        return count

    def _latest_user_message_time(self) -> datetime | None:
        """Created-at of the most recent ``user-messages`` entry."""
        entries = self._db.memories.read_latest(PennyConstants.MEMORY_USER_MESSAGES_LOG, k=1)
        return entries[0].created_at if entries else None


_TRUNCATION_TAIL_PATTERN = re.compile(r"(?:…+|\.{3,})\s*[?!.]?\s*$")


def _appears_truncated(content: str) -> bool:
    """Return True if ``content`` looks like a model self-truncation.

    Matches a tail of one-or-more ``…`` characters or three-or-more ASCII
    dots, optionally followed by a single ``?``/``!``/``.`` and trailing
    whitespace.  Production failures look like ``"...the original …"`` or
    ``"all-time-best ‑ …?"``.  Conversational mid-sentence ellipsis
    (``"Anyway… 🤓"``) doesn't match because the message ends with text
    after the ellipsis.
    """
    return bool(_TRUNCATION_TAIL_PATTERN.search(content))


def _naive_utc_now() -> datetime:
    """Naive UTC ``now`` to compare against ``MemoryEntry.created_at``,
    which round-trips through SQLite as a tz-naive value."""
    return datetime.now(UTC).replace(tzinfo=None)


def _to_naive(value: datetime) -> datetime:
    """Strip tzinfo if present so naive/aware mixes don't crash arithmetic."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
