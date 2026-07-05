"""SendMessageTool — model-driven outbound message delivery.

Bound at construction to an ``agent_name`` (the bound collection, for
collectors) plus the database.  The recipient is always the primary user
(Penny is single-user) and is resolved from ``db`` at execute time.  The
model calls this tool with a message body when it has decided what to say.

**Message-validity is validated before ``execute`` runs**, on ``SendMessageArgs``
(the tool's ``args_model``): a half-formed body — blank / punctuation-only, a
bare URL, a bail-out phrase, an unfinished fragment like ``"Hi there! ......???"``,
or an ellipsis-truncated tail — fails validation via the shared
``half_formed_send_reason`` rule (the same one the run-health classifier flags
``⚠ HALF-FORMED SEND`` on), so the ``ToolExecutor`` returns an actionable error
tool response and ``execute`` never sees it.  ``execute`` is therefore left with
only the *delivery decisions* — ones that need runtime state or are correct
no-op declines rather than content failures — then **enqueues**; it does not
send directly:

- **Refusal**: if the content is itself a model refusal ("I'm sorry,
  I can't..."), don't enqueue — that's not a real reply.  A correct no-op
  decline (``success=True``); tells the model to call ``done`` instead.
- **No recipient**: if there's no primary user, decline as a no-op naming the
  real condition — an environment/config state, not a content problem — and
  bind ``done(success=false, ...)`` since this cycle cannot deliver.  This is a
  distinct decline from Refusal: rewriting the message would be pointless when
  the fault is that no recipient exists.
- **Mute**: if the user has muted autonomous messages, decline as a no-op —
  normal cycle behaviour, not a failure — and bind ``done(success=true, ...)``.

If those pass, the message is appended to ``db.send_queue`` and the
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
    args_model = SendMessageArgs

    # The success signal a consumer's prompt can key on. Enqueue is the
    # successful handoff (delivery is the drainer's job), so it returns this.
    _SENT_RESPONSE = "Message sent."
    _REFUSAL_RESPONSE = (
        "Message NOT sent: the content reads as a model refusal "
        "(\"I'm sorry, I can't...\") rather than a substantive reply.  "
        f"Call ``{DoneTool.name}`` to exit — do not retry with the same content."
    )
    _NO_RECIPIENT_RESPONSE = (
        "Message NOT sent: no user is registered to receive messages, so this "
        "cycle cannot deliver.  This is an environment/config state, not a problem "
        "with your content.  "
        f'Call ``{DoneTool.name}(success=false, summary="no recipient — cannot deliver")`` '
        "to exit — do not rewrite the message."
    )
    _MUTED_RESPONSE = (
        "Message NOT sent: the user has muted autonomous messages.  "
        f'Call ``{DoneTool.name}(success=true, summary="muted — skipped")`` '
        "to exit — do not retry.  This is normal cycle behaviour, not a failure."
    )

    def __init__(self, agent_name: str, db: Database) -> None:
        self._agent_name = agent_name
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = SendMessageArgs(**kwargs)
        # Content validity is already enforced (the args_model validator ran in
        # ToolExecutor before we got here).  What's left are delivery decisions —
        # no-op declines (success=True, mutated=False) that need runtime state,
        # not content failures: a refusal body, no recipient, or a muted user.
        # Each names its own condition and binds the right terminal move — a
        # refusal/mute is normal cycle behaviour (done(success=true)), while no
        # recipient is an environment/config fault this cycle can't deliver into
        # (done(success=false)).  All three stay out of the tool-failure budget.
        if is_refusal(args.content):
            logger.info("send_message refused (refusal content): %s", self._agent_name)
            return ToolResult(message=self._REFUSAL_RESPONSE)
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            logger.info("send_message refused (no primary user): %s", self._agent_name)
            return ToolResult(message=self._NO_RECIPIENT_RESPONSE)
        if self._db.users.is_muted(recipient):
            logger.info("send_message refused (muted): %s", recipient)
            return ToolResult(message=self._MUTED_RESPONSE)
        # Enqueue for delivery — the drain schedule honours the cooldown and
        # sends later, so a cooldown no longer drops the message.
        self._db.send_queue.enqueue(content=args.content, collection=self._agent_name)
        logger.info("send_message queued: %s → %s", self._agent_name, recipient)
        return ToolResult(message=self._SENT_RESPONSE, mutated=True)
