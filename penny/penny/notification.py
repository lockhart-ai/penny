"""Telling the user about a finished collector cycle (#1911, reshaped by #2656).

Notification used to be part of the cycle: four numbered steps appended to the
collection's own program — two ``read_similar`` calls, "compose one short friendly
message", and ``send_message`` — carried out by the same loop, on the same context.
Measured on an instrumented run, that tail is where the cycles died: 42 of 49
reroll-exhaustion deaths landed in it.  So the tail was taken apart: the lookups
moved to Python, the composing became its own micro-context on a fresh short
document with no tool channel, and the send stayed deterministic.

#2656 retires the micro-context.  Its document CUT what the cycle wrote —
measured in prod, a 12-headline digest entry was condensed to a 500-character
preview and the message named only the first source's stories — and its tone read
as a template, because it had no conversation, no standing rules and no voice but
its own prompt.  So the composing moves to the chat agent itself:

- the cycle stays on the task: it ends with ``done(notify=…)``, as now, and
  ``_notify_if_due`` enqueues a REPORT ROW — the job's name, the trigger, and the
  run id whose entries the report is built from.  No message is written here;
- the send-queue drainer, when it reaches that row, builds the :class:`JobReport`
  from the cycle's written entries — FULL, never cut — and runs ONE chat-agent
  turn on it: the message is written with Penny's whole context (persona,
  history, standing rules, the user's formatting asks) and then passes the same
  send gate and rides the same queue and drainer as every autonomous send.

What this module still holds is the framework's half of that hand-off: the
outcome vocabulary the run record is stamped from, the typed report, its
deterministic render, and the send gate.  Nothing here calls the model.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import CycleEnd, CycleTrigger
from penny.llm.refusal import is_refusal

if TYPE_CHECKING:
    from penny.database import Database

logger = logging.getLogger(__name__)


class NotificationOutcome(StrEnum):
    """How a finished cycle's notification ended — the closed set the run record is
    stamped from, so "did the user hear about this?" is READ off the run rather than
    inferred from whether a ``send_message`` call happens to be in the trace.

    ``QUEUED`` is the whole success case (the queue IS the handoff; the drainer owns
    composing the message and when it goes out).  ``NOT_DELIVERABLE`` is the send gate
    declining for a reason no rewrite would fix: nobody is registered to receive it,
    or the user has muted autonomous messages.  (The compose failures that used to be
    ``NOT_DRAWN`` live with the delivery turn now (#2656): the message is written
    after the cycle has settled, so a draw that never became a message retries on the
    row, bounded, instead of throwing the cycle's writes away.)"""

    QUEUED = "queued"
    NOT_DELIVERABLE = "not_deliverable"


# What the run record says about each outcome — the honest note the ticket asks for,
# appended to the completed-cycle reason so the terminal shapes (aborted, stopped,
# completed-quiet, completed-notified) read differently on the run.
NOTIFICATION_NOTES: dict[NotificationOutcome, str] = {
    NotificationOutcome.QUEUED: "the report was queued to reach the user",
    NotificationOutcome.NOT_DELIVERABLE: "nothing was sent — the send was declined",
}


# How a done-closed cycle ENDED, per notification outcome (#1936) — the other table
# every outcome is read through, and the one that decides whether the cycle's entry
# writes survive.  Both outcomes are ENDS: the cycle did its job, and a decline is a
# decision rather than a failure — the user is muted, or nobody is registered to
# receive it, and attempting the same cycle again reaches the same answer.  A message
# that is never WRITTEN is no longer a cycle outcome at all: the composing happens on
# the queued row after the cycle has settled (#2656), bounded there.
NOTIFIED_CYCLE_ENDS: dict[NotificationOutcome, CycleEnd] = {
    NotificationOutcome.QUEUED: CycleEnd.CLOSED_NOTIFIED,
    NotificationOutcome.NOT_DELIVERABLE: CycleEnd.CLOSED_DECLINED,
}


class WrittenEntry(BaseModel):
    """One entry the cycle wrote the current value of — where it went, under what key
    (``None`` for a log append, which has none), and what it says."""

    memory: str
    key: str | None = None
    content: str


class JobReport(BaseModel):
    """Everything the delivery chat turn is given about a finished cycle (#2656).

    Assembled framework-side, as ONE typed value a test can build and render alone.
    The entries are carried WHOLE — never condensed, never capped: the composer's cut
    was the measured failure, and the chat agent can write a message the user can act
    on only from what it is actually shown."""

    collection: str
    description: str
    trigger: CycleTrigger
    run_id: str
    entries: tuple[WrittenEntry, ...]


_CADENCE_LINE = "It ran on its own schedule."
_ON_DEMAND_LINE = "It ran just now because the user asked it to run."
_WROTE_HEAD = "## What the job wrote this run"
_WROTE_NONE = "It wrote nothing down this run."
_WRITE_LINE = "- into `{memory}`, under `{key}`: {content}"
_WRITE_LINE_NO_KEY = "- into `{memory}`: {content}"
_ASK = "Write the one message to send the user about this run, from what the job wrote."
_REPORT_LEAD = 'Your job "{collection}" just finished a run and asked to tell the user.'
_REPORT_ABOUT = "What the job is for: {description}"


def render_job_report(report: JobReport) -> str:
    """The whole delivery turn's input, rendered from framework-assembled facts alone.

    Pure and deterministic — no database, no model — so the exact text the chat agent
    reads is pinned by a whole-render test rather than by whatever a live run happened
    to produce.  Every written entry renders FULL and in the order the cycle wrote
    them, with the key it is stored under: the key is the anchor a follow-up
    (``collection_get``) needs, and the content is the payload the message is written
    from.  Nothing is condensed and nothing is capped."""
    lines = [
        _REPORT_LEAD.format(collection=report.collection),
        _REPORT_ABOUT.format(description=report.description),
        _CADENCE_LINE if report.trigger is CycleTrigger.CADENCE else _ON_DEMAND_LINE,
        "",
        _WROTE_HEAD,
    ]
    if not report.entries:
        lines.append(_WROTE_NONE)
    else:
        lines.extend(_written_report_line(entry) for entry in report.entries)
    lines.extend(["", _ASK])
    return "\n".join(lines)


def _written_report_line(entry: WrittenEntry) -> str:
    if entry.key is None:
        return _WRITE_LINE_NO_KEY.format(memory=entry.memory, content=entry.content)
    return _WRITE_LINE.format(memory=entry.memory, key=entry.key, content=entry.content)


def build_job_report(
    db: Database, collection: str, run_id: str, *, trigger: CycleTrigger
) -> JobReport | None:
    """Assemble the report for one queued row: the collection row's own words, and the
    FULL current value of every entry that run wrote.

    ``trigger`` is the queued row's own ``origin`` — a required parameter, carried
    never inferred: it is what the report says about why the cycle ran, and what the
    drainer's delivery lane already read off the row.

    ``None`` when the collection has vanished between the enqueue and the compose —
    logged, and the caller records the failure on the row rather than composing from
    a half-assembled report.  (A collection archived after the enqueue already had
    its pending rows cancelled at the archive chokepoint, so reaching here with no
    row means something else moved it; the honest answer is a named failure.)"""
    row = db.memories.get(collection)
    if row is None:
        logger.error(
            "Report for '%s' (run %s) cannot be built — the collection no longer exists",
            collection,
            run_id,
        )
        return None
    written = db.memories.entries_written_by_run(run_id)
    return JobReport(
        collection=collection,
        description=row.description,
        trigger=trigger,
        run_id=run_id,
        entries=tuple(
            WrittenEntry(memory=entry.memory_name, key=entry.key, content=entry.content)
            for entry in written
        ),
    )


def withheld_reason(db: Database, content: str) -> str | None:
    """The send gate every composed message passes, as a reason or ``None``.

    The declines are the ones that need runtime state or are correct no-ops —
    decisions no retry changes: content that reads as a model refusal, no registered
    recipient, and a muted user.  ``None`` means the content may be stored on the
    row and delivered.

    The recipient and mute checks run here as well as at the enqueue because the two
    moments can be far apart — a cooldown can hold a report for minutes, and the
    user's mute state may have changed since the cycle queued it.  (A message that
    is not WHOLE is not a decline: that is a compose failure, retried on the row —
    the drainer reads it off ``half_formed_send_reason`` directly, the same rule the
    retired composer's acceptance rule enforced, so what the framework is willing to
    deliver, what a ``send_message`` would carry, and what a failed send is flagged
    on are one definition.)"""
    if is_refusal(content):
        return "the message read as a model refusal"
    recipient = db.users.get_primary_sender()
    if recipient is None:
        return "there is no registered recipient"
    if db.users.is_muted(recipient):
        return "the user is muted"
    return None
