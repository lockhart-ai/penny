"""SendMessageTool — model-driven outbound message delivery.

Bound at construction to an ``agent_name`` (the bound collection, for
collectors) plus the database.  The recipient is always the primary user
(Penny is single-user) and is resolved from ``db`` at execute time.  The
model calls this tool with a message body when it has decided what to say.
The tool checks three *content/availability* gates, then **enqueues** the
message for delivery — it does not send directly:

- **Refusal**: if the content is itself a model refusal ("I'm sorry,
  I can't..."), don't enqueue — that's not a real reply.  Tells the
  model to call ``done`` instead.
- **Truncation**: if the content tail looks like a model self-
  truncation (ending in ``…`` or three-or-more dots, mid-thought),
  return a failure string with the ``Error:`` prefix so the agent
  loop marks the call as failed and the model re-emits the complete body.
- **Mute**: if the recipient has muted notifications, the tool refuses
  with a string that tells the model to call ``done``.

If all three pass, the message is appended to ``db.send_queue`` and the
tool returns ``"Message sent."`` (``mutated=True``).  Enqueue **is** the
successful handoff: the background drain schedule (``SendQueueDrainer``)
owns *when* the message actually goes out, honouring the flat-interval
autonomous-send cooldown and delivering the message later rather than
dropping it.  The literal ``"Message sent."`` is preserved so the
collector prompts that gate a follow-up move on it ("only move the entry
once send_message returned Message sent.") keep working unchanged — from
the collector's point of view the message has been accepted for delivery,
which is true.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from penny.llm.refusal import is_refusal
from penny.tools.base import Tool
from penny.tools.memory_tools import DoneTool
from penny.tools.models import SendMessageArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Queue a message to the user for delivery through the bound channel."""

    name = "send_message"
    description = (
        "Send a message to the user.  Use this once you have decided "
        "what to say — the ``content`` is the exact text the user will "
        "see.  The send is gated on refusal detection and mute state; if "
        f"either refuses, the response will say so and you should call "
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

    # Preserved verbatim: collector prompts gate a follow-up ``collection_move``
    # on this exact string ("only move the entry once send_message returned
    # Message sent.").  Enqueue is the successful handoff, so it returns this.
    _SENT_RESPONSE = "Message sent."
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
    # Returned with ``success=False`` so the agent loop sets
    # ``record.failed=True``, which counts toward the abort threshold — we don't
    # infinite-loop if the model keeps producing truncated content.
    _TRUNCATION_REJECTION = (
        "Message NOT sent: the content ended with an ellipsis "
        "('…' or '...'), which means it was cut off mid-thought.  "
        "Call send_message again with the COMPLETE message body — "
        "finish every sentence and bullet you start, no ellipses, "
        "no 'etc.', no 'and more', no teaser phrasing."
    )

    def __init__(self, agent_name: str, db: Database) -> None:
        self._agent_name = agent_name
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = SendMessageArgs(**kwargs)
        # Not-enqueued gates all carry mutated=False (nothing was queued).  Most
        # are *successful* no-ops — a correct decline the model shouldn't retry
        # (refusal content, no recipient, muted), so success=True keeps them out
        # of the failure budget and matches their "this is normal, not a failure"
        # bodies.  Truncation is the one real failure: the model produced cut-off
        # content, so success=False marks the call failed and steers a retry with
        # the complete body (and feeds the abort threshold).
        if is_refusal(args.content):
            logger.info("send_message refused (refusal content): %s", self._agent_name)
            return ToolResult(message=self._REFUSAL_RESPONSE)
        if _appears_truncated(args.content):
            logger.info("send_message rejected (truncation): %s", self._agent_name)
            return ToolResult(message=self._TRUNCATION_REJECTION, success=False)
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            logger.info("send_message refused (no primary user): %s", self._agent_name)
            return ToolResult(message=self._REFUSAL_RESPONSE)
        if self._db.users.is_muted(recipient):
            logger.info("send_message refused (muted): %s", recipient)
            return ToolResult(message=self._MUTED_RESPONSE)
        # Enqueue for delivery — the drain schedule honours the cooldown and
        # sends later, so a cooldown no longer drops the message.
        self._db.send_queue.enqueue(content=args.content, collection=self._agent_name)
        logger.info("send_message queued: %s → %s", self._agent_name, recipient)
        return ToolResult(message=self._SENT_RESPONSE, mutated=True)


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
