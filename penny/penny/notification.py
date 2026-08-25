"""Telling the user about a finished collector cycle (#1911).

Notification used to be part of the cycle: four numbered steps appended to the
collection's own program — two ``read_similar`` calls, "compose one short friendly
message", and ``send_message`` — carried out by the same loop, on the same context.
Measured on an instrumented run, that tail is where the cycles died: 42 of 49
reroll-exhaustion deaths landed in it, immediately after a ``read_similar``, because
by then the context was long and chat-flavoured and the model's tool-call envelope
decayed.  86% of the deaths, and almost none of the value.

So the tail is taken apart along the line of what actually needs a model:

- the two lookups are STORE QUERIES — embed what the run found, read the nearest
  entries out of ``user-messages`` and ``penny-messages``.  Python, always;
- the composing is a MODEL job — it is prose about one particular find — so it runs
  as its own micro-context (``MicroContext.compose_notification``) on a FRESH, short
  document with NO tool channel at all, which is why a decayed tool-call envelope can
  no longer end a cycle here: there is no call left to decay into;
- the send is DETERMINISTIC — the refusal / no-recipient / mute gate and the enqueue
  the ``SendQueueDrainer`` drains on the cooldown.  Python, always.

What is left in the model's hands is exactly the sentence the user reads, and every
fact in it was put in front of it by the framework.

That document is BOUNDED and ORDERED (#1934).  The first live cycle after the reset
assembled 50,805 characters of it for an ordinary three-page news round — a browse
result inlined whole, a write call restating its whole payload in its own arguments,
and the entries holding that text a third time — on a model whose degeneration onset is
~4K prompt tokens; and the message it composed led with a headline that was not the
fetched page's top story but WAS the lead of the PREVIOUS notification, which sat in
the closest prior messages.  So the entries the cycle wrote lead the document, the
calls follow them, and the earlier conversation comes last under a heading that says
what it is.  Every section is bounded in the RENDER against a stated budget, and every
cut states what it left out — a condensed value is something the draw can act on, where
a silent one is not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from penny.agents.models import ToolCallRecord
from penny.constants import CycleEnd, CycleTrigger, PennyConstants
from penny.database.memory.objects import render_tool_call
from penny.database.models import MemoryEntry, MemoryRow
from penny.datetime_utils import format_log_timestamp
from penny.llm.refusal import is_refusal
from penny.llm.similarity import embed_text
from penny.tools.micro_context import MicroContext
from penny.tools.models import SendMessageArgs

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


class NotificationOutcome(StrEnum):
    """How a finished cycle's notification ended — the closed set the run record is
    stamped from, so "did the user hear about this?" is READ off the run rather than
    inferred from whether a ``send_message`` call happens to be in the trace.

    ``QUEUED`` is the whole success case (the queue IS the handoff; the drainer owns
    when it goes out).  ``NOT_DRAWN`` is a model that could not produce a sendable
    message within its reroll budget — an honest degradation, never a crash and never
    a silent skip.  ``NOT_DELIVERABLE`` is the send gate declining for a reason no
    rewrite would fix: the content read as a refusal, nobody is registered to receive
    it, or the user has muted autonomous messages."""

    QUEUED = "queued"
    NOT_DRAWN = "not_drawn"
    NOT_DELIVERABLE = "not_deliverable"


# What the run record says about each outcome — the honest note the ticket asks for,
# appended to the completed-cycle reason so all four terminal shapes (aborted,
# stopped, completed-quiet, completed-notified) read differently on the run.
NOTIFICATION_NOTES: dict[NotificationOutcome, str] = {
    NotificationOutcome.QUEUED: "the user was told",
    NotificationOutcome.NOT_DRAWN: "nothing was sent — no usable message could be written",
    NotificationOutcome.NOT_DELIVERABLE: "nothing was sent — the send was declined",
}


# How a done-closed cycle ENDED, per notification outcome (#1936) — the other table
# every outcome is read through, and the one that decides whether the cycle's staged
# writes land.
#
# ``QUEUED`` and ``NOT_DELIVERABLE`` are both ENDS: the cycle did its job, and a decline
# is a decision rather than a failure — the user is muted, or nobody is registered to
# receive it, and attempting the same cycle again reaches the same answer.  (Two of the
# gate's own refusals — content that reads as a refusal, or that is not a whole message
# — are theoretically reachable here and read as declines too; the draw's acceptance
# rule screens both before the queue ever sees them.)  ``NOT_DRAWN`` is not an end at
# all: the compose micro-context exhausted its rerolls, so the cycle never reached the
# point of reporting anything, and the whole run is thrown out and retried.
NOTIFIED_CYCLE_ENDS: dict[NotificationOutcome, CycleEnd] = {
    NotificationOutcome.QUEUED: CycleEnd.CLOSED_NOTIFIED,
    NotificationOutcome.NOT_DELIVERABLE: CycleEnd.CLOSED_DECLINED,
    NotificationOutcome.NOT_DRAWN: CycleEnd.NOTIFICATION_NOT_DRAWN,
}


class CycleCall(BaseModel):
    """One executed call of the cycle, as the notify document renders it: the call in
    the canonical ``tool(args)`` notation and its result.

    Both halves are carried WHOLE here and CONDENSED at the render, against a stated
    budget (#1934).  #1911 carried them whole all the way down, on the reasoning that a
    truncated result is where an invented detail comes from; the first live cycle after
    the reset measured what that costs — one browse result inlined at 31,725 characters
    and one ``collection_write`` whose ARGUMENTS restated the whole payload at 14,923
    on a single line, on a model that degenerates past ~4K prompt tokens.  A document
    the model cannot read is not a safer document, so the cut stands but STATES itself:
    a condensed value ends in the count of what was left out, which is a fact the draw
    can act on, where a silent cut is not."""

    call: str
    result: str = ""
    failed: bool = False


class WrittenEntry(BaseModel):
    """One entry the cycle wrote the current value of — where it went, under what key
    (``None`` for a log append, which has none), and what it says."""

    memory: str
    key: str | None = None
    content: str


class RelatedMessages(BaseModel):
    """The nearest past messages from ONE message log, best-first, already rendered.

    Both logs are carried even when one came back empty, so the document states the
    absence rather than leaving a section out and letting it read as an oversight."""

    source: str
    lines: tuple[str, ...] = ()


class NotificationInput(BaseModel):
    """Everything the notify micro-context is given, assembled framework-side.

    A single typed value rather than five arguments, because the render and the
    assembly are two halves of one contract and a test drives the render alone."""

    collection: str
    description: str
    routine: str | None = None
    calls: tuple[CycleCall, ...] = ()
    written: tuple[WrittenEntry, ...] = ()
    related: tuple[RelatedMessages, ...] = ()


_LEAD = "The `{collection}` routine just ran on its own, and it is time to tell the user."
_ABOUT = "What this collection is for: {description}"
_ROUTINE = "The routine it runs: {routine}"
_WROTE_HEAD = "## What it wrote this cycle"
_WROTE_NONE = "It wrote nothing down this cycle."
_DID_HEAD = "## How it got there"
_DID_NONE = "It made no calls."
_RELATED_HEAD = "## Background: earlier messages"
_RELATED_LEAD = (
    "These messages are from before this cycle ran. What this cycle found is at the "
    "top of this document."
)
_RELATED_NONE = "Nothing in your past messages matched this — there is no callback to make."
_RESULT_PREFIX = "   → "
_FAILED_PREFIX = "   → (this call failed) "
# The two honest-omission markers (#1934).  A bound that cuts says what it cut: the
# characters a condensed value dropped, and the items a capped section left out.
_CHARS_OMITTED = "… [{count} characters omitted]"
_MORE_ITEMS = "{count} more {noun} not shown."
# What each capped section counts, singular and plural — declared beside the marker
# they fill rather than passed as bare words at each call site.
_ENTRY_NOUNS = ("entry", "entries")
_CALL_NOUNS = ("call", "calls")
_MESSAGE_NOUNS = ("message", "messages")


def render_notification(document: NotificationInput) -> str:
    """The whole notify document, rendered from framework-assembled facts alone.

    Pure and deterministic — no database, no model — so the exact text the draw reads
    is pinned by a whole-render test rather than by whatever a live run happened to
    produce.

    Sections in the order the message is written FROM (#1934): what ran, then what the
    cycle WROTE — the payload, at the head — then how it got there, then the earlier
    conversation, labelled as background and last.  The order is load-bearing: with the
    closest prior messages sitting nearest the draw, one measured composition took the
    PREVIOUS notification's lead as its template and re-reported it as today's top
    story.  Every section is BOUNDED here rather than at the assembly, so the bound
    holds whatever put the facts in front of it, and every cut states itself."""
    return "\n".join(
        [
            _LEAD.format(collection=document.collection),
            _ABOUT.format(
                description=_condensed(
                    document.description, PennyConstants.NOTIFY_DESCRIPTION_CHARS
                )
            ),
            *([_ROUTINE.format(routine=document.routine)] if document.routine else []),
            "",
            _WROTE_HEAD,
            _render_written(document.written),
            "",
            _DID_HEAD,
            _render_calls(document.calls),
            "",
            _RELATED_HEAD,
            _render_related(document.related),
        ]
    )


def _condensed(text: str, budget: int) -> str:
    """``text`` within its budget, or its head plus the count of what was left out.

    Never a silent cut: the marker is the difference between a draw reading a partial
    value knowingly and one reading it as the whole thing."""
    if len(text) <= budget:
        return text
    return f"{text[:budget]}{_CHARS_OMITTED.format(count=len(text) - budget)}"


def _overflow_line(shown: int, total: int, nouns: tuple[str, str]) -> list[str]:
    """The honest count of the items a section's cap left out, or nothing at all."""
    rest = total - shown
    if rest <= 0:
        return []
    singular, plural = nouns
    return [_MORE_ITEMS.format(count=rest, noun=singular if rest == 1 else plural)]


def _render_written(written: Sequence[WrittenEntry]) -> str:
    """The durable outcome, and the document's LEAD: the entries whose current value
    this cycle wrote, in the order the run wrote them, up to the section's cap.

    Empty is a real answer and says so plainly — a routine with no write at all
    completes and notifies like any other (the exit is the cycle's own close, never a
    write), so "nothing written" is an ordinary shape here."""
    if not written:
        return _WROTE_NONE
    shown = written[: PennyConstants.NOTIFY_WRITTEN_ENTRIES]
    lines = [_written_line(entry) for entry in shown]
    return "\n".join([*lines, *_overflow_line(len(shown), len(written), _ENTRY_NOUNS)])


def _written_line(entry: WrittenEntry) -> str:
    content = _condensed(entry.content, PennyConstants.NOTIFY_WRITTEN_CONTENT_CHARS)
    if entry.key is None:
        return f"- into `{entry.memory}`: {content}"
    key = _condensed(entry.key, PennyConstants.NOTIFY_ENTRY_KEY_CHARS)
    return f"- into `{entry.memory}`, under `{key}`: {content}"


def _render_calls(calls: Sequence[CycleCall]) -> str:
    """The cycle's calls in order, each with its result under it — the run-record
    register (``tool(args)``, one call per line) with the results the record itself
    leaves out, because a record is read for what a run DID and this document is read
    for what it FOUND.

    Both halves are condensed: a call's own ARGUMENTS restate whatever payload it
    carried (a measured ``collection_write`` ran to 14,923 characters on one line), and
    a browse result carries a whole fetched page."""
    if not calls:
        return _DID_NONE
    shown = calls[: PennyConstants.NOTIFY_CYCLE_CALLS]
    lines: list[str] = []
    for index, call in enumerate(shown, start=1):
        lines.append(f"{index}. {_condensed(call.call, PennyConstants.NOTIFY_CALL_CHARS)}")
        if call.result:
            prefix = _FAILED_PREFIX if call.failed else _RESULT_PREFIX
            result = _condensed(call.result, PennyConstants.NOTIFY_CALL_RESULT_CHARS)
            lines.append(f"{prefix}{result}")
    return "\n".join([*lines, *_overflow_line(len(shown), len(calls), _CALL_NOUNS)])


def _render_related(related: Sequence[RelatedMessages]) -> str:
    """Both message logs' nearest entries as BACKGROUND, or the one plain
    nothing-matched line.

    Nothing matching is the ORDINARY case, so it renders as a statement rather than as
    two empty sections: a callback line is worth having when there is something to
    call back to, and inventing one is the failure this wording exists to prevent.
    When there IS something, the section says what it is — messages from before this
    cycle — and where the cycle's own findings are, because these are the only lines in
    the document that were never produced by the run being reported."""
    populated = [group for group in related if group.lines]
    if not populated:
        return _RELATED_NONE
    return "\n\n".join([_RELATED_LEAD, *(_related_group(group) for group in populated)])


def _related_group(group: RelatedMessages) -> str:
    """One log's nearest messages, capped and each condensed to its own budget."""
    shown = group.lines[: PennyConstants.NOTIFY_RELATED_MESSAGES]
    lines = [_condensed(line, PennyConstants.NOTIFY_RELATED_LINE_CHARS) for line in shown]
    overflow = _overflow_line(len(shown), len(group.lines), _MESSAGE_NOUNS)
    return "\n".join([f"### {group.source}", *lines, *overflow])


class CollectorNotifier:
    """Assembles a finished cycle's notify document, draws the message, and queues it.

    Framework-entered: the collector calls it after the cycle's program is covered, so
    nothing here is reachable from a model's tool call and no prompt can ask for it.
    """

    def __init__(self, db: Database, embedding_client: LlmClient, model_client: LlmClient) -> None:
        self._db = db
        self._embedding_client = embedding_client
        self._micro_context = MicroContext(model_client)

    async def notify(
        self,
        target: MemoryRow,
        run_id: str,
        records: Sequence[ToolCallRecord],
        trigger: CycleTrigger,
    ) -> NotificationOutcome:
        """Tell the user about this cycle — assemble, draw, queue.

        ``trigger`` is what set the cycle running, carried through to the queued row so
        the drainer knows whether anyone is waiting on this message (#1939).

        Returns the enumerated outcome the collector stamps on the run.  Every branch
        is a recorded fact: a message that could not be written and a send that was
        declined are both stated on the run record rather than passing as a quiet
        success."""
        document = await self._document(target, run_id, records)
        message = await self._micro_context.compose_notification(
            render_notification(document), run_target=target.name
        )
        if message is None:
            logger.warning(
                "No sendable notification could be drawn for '%s' — run %s sends nothing",
                target.name,
                run_id,
            )
            return NotificationOutcome.NOT_DRAWN
        return self.queue(target.name, message, trigger)

    def queue(self, mechanism: str, content: str, trigger: CycleTrigger) -> NotificationOutcome:
        """The send gate every autonomous message passes, then the enqueue.

        The three declines are the ones that need runtime state or are correct no-ops:
        content that reads as a model refusal, no registered recipient, and a muted
        user.  Enqueue IS the successful handoff — the ``SendQueueDrainer`` owns when
        the message actually goes out, so a cooldown delays a message rather than
        losing it.  ``trigger`` is stamped on the row and decides WHICH lane it waits
        in: a cadence cycle's message rides the full autonomous-send cooldown, while a
        cycle the user pressed "run this now" on goes out on the next drain tick,
        because they are sitting there waiting for it (#1939).

        Content validity is checked one layer up, in the draw's own acceptance rule
        (a half-formed message is re-drawn rather than queued), and re-validated here
        through the same :class:`SendMessageArgs` model so nothing reaches the queue
        unvalidated whatever put it there."""
        try:
            args = SendMessageArgs(content=content)
        except ValidationError:
            logger.warning("Notification for '%s' was not a complete message", mechanism)
            return NotificationOutcome.NOT_DELIVERABLE
        if is_refusal(args.content):
            logger.info("Notification for '%s' read as a model refusal", mechanism)
            return NotificationOutcome.NOT_DELIVERABLE
        recipient = self._db.users.get_primary_sender()
        if recipient is None:
            logger.info("Notification for '%s' has no registered recipient", mechanism)
            return NotificationOutcome.NOT_DELIVERABLE
        if self._db.users.is_muted(recipient):
            logger.info("Notification for '%s' withheld — the user is muted", mechanism)
            return NotificationOutcome.NOT_DELIVERABLE
        self._db.send_queue.enqueue(content=args.content, collection=mechanism, origin=trigger)
        logger.info("Notification queued (%s): %s → %s", trigger.value, mechanism, recipient)
        return NotificationOutcome.QUEUED

    async def _document(
        self, target: MemoryRow, run_id: str, records: Sequence[ToolCallRecord]
    ) -> NotificationInput:
        """The framework's whole assembly: the run's own calls, what it wrote, and the
        past-message lookups Python performs on the caller's behalf."""
        written = self._db.memories.entries_written_by_run(run_id)
        return NotificationInput(
            collection=target.name,
            description=target.description,
            routine=self._routine_line(target),
            calls=tuple(_cycle_call(record) for record in records),
            written=tuple(_written_entry(entry) for entry in written),
            related=await self._related(target, written),
        )

    @staticmethod
    def _routine_line(target: MemoryRow) -> str | None:
        """Which routine ran, by name — absent for a hand-authored collection, which
        has none, so its document renders one line shorter rather than saying so."""
        return target.skill_name

    async def _related(
        self, target: MemoryRow, written: Sequence[MemoryEntry]
    ) -> tuple[RelatedMessages, ...]:
        """The two message logs' nearest past entries — the retired notify steps'
        ``read_similar(memory=…, anchor=<what you just found>, k=5)`` pair, done here.

        The ANCHOR is what the cycle found: the contents of everything it wrote,
        joined.  A cycle that wrote nothing has no find to anchor on, so it falls back
        to the collection's own description — the same meaning anchor the registry
        resolves the collection by — which keeps a no-write routine's callback line
        answerable instead of arbitrary.  A transient embed failure yields no groups
        at all, which the render states as nothing-matched: honest, and the message is
        still written."""
        anchor = "\n".join(entry.content for entry in written) or target.description
        vector = await embed_text(self._embedding_client, anchor)
        if vector is None:
            logger.warning("Notify anchor for '%s' could not be embedded", target.name)
            return ()
        return tuple(
            self._nearest(log_name, vector)
            for log_name in (
                PennyConstants.MEMORY_USER_MESSAGES_LOG,
                PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
            )
        )

    def _nearest(self, log_name: str, vector: list[float]) -> RelatedMessages:
        """One log's nearest entries, best-first, rendered with their timestamps — the
        same plain ``[stamp] content`` shape a log read shows the model."""
        log = self._db.memory(log_name)
        if log is None:
            return RelatedMessages(source=log_name)
        entries = log.read_similar(vector, PennyConstants.NOTIFY_RELATED_MESSAGES)
        return RelatedMessages(
            source=log_name,
            lines=tuple(
                f"{index}. [{format_log_timestamp(entry.created_at)}] {entry.content}"
                for index, entry in enumerate(entries, start=1)
            ),
        )


def _cycle_call(record: ToolCallRecord) -> CycleCall:
    """One executed call as the document renders it — the canonical call notation the
    run trace uses everywhere, so what the draw reads of the cycle matches what every
    other surface says about it."""
    return CycleCall(
        call=render_tool_call(record.tool, record.arguments),
        result=record.result or "",
        failed=record.failed,
    )


def _written_entry(entry: MemoryEntry) -> WrittenEntry:
    return WrittenEntry(memory=entry.memory_name, key=entry.key, content=entry.content)
