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

from penny.constants import (
    SANDBOX_INTERRUPTED_REASON,
    CycleEnd,
    CycleTrigger,
    MutationAction,
    MutationActor,
    MutationEntityType,
)
from penny.database.memory import render_run_calls
from penny.database.mutation_store import mutation_detail
from penny.datetime_utils import user_timezone_name
from penny.llm.refusal import is_refusal
from penny.tools.link_handles import CITATION_LINE, HandleTable

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
_WROTE_LEAD = "These entries are what the job kept for the user this run."
_WROTE_NONE = "It wrote nothing down this run."
_WRITE_LINE = "- into `{memory}`, under `{key}`: {content}"
_WRITE_LINE_NO_KEY = "- into `{memory}`: {content}"
_REPORT_LEAD = 'Your job "{collection}" just finished a run and asked to tell the user.'
_REPORT_ABOUT = "What the job is for: {description}"


def render_job_report(report: JobReport, handles: HandleTable | None = None) -> str:
    """The whole delivery turn's input, rendered from framework-assembled facts alone
    (#2656, the trigger reworked #2974 — facts only, no instruction: she decides for
    herself whether, how, and how many times to reply, the same as on any chat turn).

    Pure and deterministic — no database, no model — so the exact text the chat agent
    reads is pinned by a whole-render test rather than by whatever a live run happened
    to produce.  Every written entry renders FULL and in the order the cycle wrote
    them, with the key it is stored under: the key is the anchor a follow-up
    (``memory_get``) needs, and the content is the payload the message is written
    from.  Nothing is condensed and nothing is capped.

    ``handles`` is the delivery turn's own link table (#2717): a stored entry keeps
    real URLs, so the render converts them to that turn's handles — a URL the model
    READS always arrives as a handle, and the message it writes cites numbers the
    send path expands.  ``None`` renders the entries verbatim (the test shape).

    This is the ONE render used both live (prefixed with ``PennyConstants.
    SYSTEM_NOTIFICATION_PREFIX`` as the delivery turn's own USER message,
    ``ChatAgent.handle_job_report``) and, byte-identical, in every later turn's history
    (``Agent._run_trigger_message`` reads it back off this run's own promptlog row —
    never stored a second time)."""
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
        lines.append(_WROTE_LEAD)
        size_before = len(handles) if handles is not None else 0
        lines.extend(_written_report_line(entry, handles) for entry in report.entries)
        if handles is not None and len(handles) > size_before:
            lines.append(CITATION_LINE)
    return "\n".join(lines)


def _written_report_line(entry: WrittenEntry, handles: HandleTable | None = None) -> str:
    content = handles.convert(entry.content) if handles is not None else entry.content
    if entry.key is None:
        return _WRITE_LINE_NO_KEY.format(memory=entry.memory, content=content)
    return _WRITE_LINE.format(memory=entry.memory, key=entry.key, content=content)


def build_job_report(
    db: Database, collection: str, run_id: str, *, trigger: CycleTrigger
) -> JobReport | None:
    """Assemble the report for one queued row: the collection row's own words, and the
    FULL current value of every entry that run wrote THROUGH ITS OWN WRITE TOOLS.

    ``trigger`` is the queued row's own ``origin`` — a required parameter, carried
    never inferred: it is what the report says about why the cycle ran, and what the
    drainer's delivery lane already read off the row.

    ``entries_written_by_run`` is read with ``exclude_system_logs=True`` (#2803): a run
    that browsed stamped a ``browse-results`` entry with the same run id a kept entry
    carries, but that page is Python's own record of what the run READ, not something
    the job itself decided to keep — rendering it alongside a kept entry told the
    delivery turn its message "covers every one" when it hadn't even read most of them.
    The exclusion is the memory's own kind (``PennyConstants.SYSTEM_LOGS`` — the same
    mark the ``memory_write`` gate already refuses a model write against), never a name
    this call invents.

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
    written = db.memories.entries_written_by_run(run_id, exclude_system_logs=True)
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


class SandboxReport(BaseModel):
    """Everything the delivery chat turn is given about a finished SANDBOX preview
    (#2964) — the sibling of :class:`JobReport` for a cycle that wrote nothing
    real.  ``trace`` is the previewed cycle's own tool-call sequence, each
    mutating call rendered with the plan wording it WOULD have shown (#2995 —
    the job read the ordinary wording instead, so the run's stamped
    ``plan_descriptions`` is what names it here) — never the written entries
    ``JobReport`` carries, because nothing here ever committed.

    ``interrupted`` (#3024) is whether the preview was cut off before it
    finished — read off the run's own outcome stamp at compose time, never
    guessed — and picks the lead the render opens with."""

    collection: str
    description: str
    run_id: str
    trace: str
    interrupted: bool


_SANDBOX_REPORT_LEAD = (
    'The sandbox run of "{collection}" is complete — its writes were not applied '
    "because it was sandboxed."
)

_SANDBOX_REPORT_INTERRUPTED_LEAD = (
    'The sandbox run of "{collection}" was interrupted by a restart before it '
    "finished; these are the steps it got through:"
)


def render_sandbox_report(report: SandboxReport) -> str:
    """The whole sandbox delivery turn's input, rendered from framework-assembled
    facts alone — the sibling of ``render_job_report`` (trigger reworked #2974;
    lead reworded #2995 — the run looks live to the job, so THIS is the one
    surface that says its writes were not applied).

    Pure and deterministic, same as ``render_job_report``.  ``report.trace`` is
    already the full tool-call sequence (``render_run_calls``), so there is
    nothing here to condense or cap.  The lead is the one fact that keeps the
    model from mistaking a preview for a thing that happened — an interrupted
    run's lead (#3024) states that the run never finished, instead of reporting
    a partial run as complete; nothing here instructs her to say so; she decides
    for herself whether, how, and how many times to reply, the same as on any
    chat turn."""
    lead = (
        _SANDBOX_REPORT_INTERRUPTED_LEAD if report.interrupted else _SANDBOX_REPORT_LEAD
    ).format(collection=report.collection)
    return "\n".join(
        [
            lead,
            _REPORT_ABOUT.format(description=report.description),
            "",
            report.trace,
        ]
    )


def build_sandbox_report(db: Database, collection: str, run_id: str) -> SandboxReport | None:
    """Assemble the report for one queued SANDBOX row: the collection row's own
    words, and the previewed cycle's own tool-call trace — never written
    entries, since a sandbox cycle commits none (every mutating call went
    through its plan path, #2933).

    ``interrupted`` (#3024) is read at COMPOSE time off the run's own outcome
    stamp — the same read ``build_job_failed_report`` makes.  A finished preview
    stamps ``sandbox`` on its last prompt row; one a shutdown cut off stamps
    ``cancelled`` carrying ``SANDBOX_INTERRUPTED_REASON`` — which always lands,
    because even a call cancelled mid-flight logs its own promptlog row
    (``LlmClient.chat``), so the interrupted stamp has a row to sit on.  A run
    carrying no stamp at all reads as not interrupted — there is nothing its
    record names the interruption from.

    The trace is ``render_run_calls`` — the same renderer ``run_get``/``run_list``
    use to show a run's own tool-call sequence, call syntax plus each call's
    compact result — with the run's stamped plan wording (#2995) rendered for
    every call that went through its plan path: the job itself read the as-applied
    wording (a sandbox run looks live to the job), so the stamp is the only record
    of what each call WOULD have done.  ``with_thinking=False``: the user-facing
    report needs what each step did, not why the model chose it.
    ``schedule_words=""``: a sandbox preview names no cadence
    (``dispatch_line``'s own SANDBOX branch renders "as a sandbox preview"
    regardless of this argument).

    ``None`` when the collection has vanished between the enqueue and the
    compose — the same honest failure ``build_job_report`` names for the real
    path."""
    row = db.memories.get(collection)
    if row is None:
        logger.error(
            "Sandbox report for '%s' (run %s) cannot be built — the collection no longer exists",
            collection,
            run_id,
        )
        return None
    prompts = db.messages.get_run_prompts(run_id)
    stamped = next((prompt for prompt in reversed(prompts) if prompt.run_outcome), None)
    trace = render_run_calls(
        prompts,
        db.messages.get_child_draws(run_id),
        with_thinking=False,
        delivered_text=db.messages.delivered_text_for_run(run_id),
        timezone_name=user_timezone_name(db),
        schedule_words="",
        # The stamped plan wording (#2995) replaces each forced-plan call's
        # recorded result — the as-applied wording the job read, since the run
        # looks live to it.  What each call WOULD have done is what this report
        # exists to show.
        plan_descriptions=db.messages.plan_descriptions_for_run(run_id),
    )
    return SandboxReport(
        collection=collection,
        description=row.description,
        run_id=run_id,
        trace=trace,
        interrupted=stamped is not None and stamped.run_reason == SANDBOX_INTERRUPTED_REASON,
    )


# ── The autonomous job events (#2975) ────────────────────────────────────────
#
# Two facts Penny is told as system messages, through the SAME queue/drainer/
# delivery-turn hand-off a notifying cycle's report rides: the system archive of
# a retired job, and a cadence run that failed for good (its occurrence-retry
# budget spent out).  Each fact set is assembled from the chokepoints that
# already record the event — the mutation ledger's system-actor archive note,
# the failed run's own promptlog outcome stamp — never re-derived and never a
# job's name deciding anything.


class JobRetiredReport(BaseModel):
    """Everything the delivery chat turn is given about a job the system archived
    (#2975): ``why`` is the ledger's own note on the collection's system-actor
    ARCHIVED event (the run limit spent out, or the expiry that passed) — the
    record the archive path already wrote, read back verbatim."""

    collection: str
    description: str
    why: str


_RETIRED_REPORT_LEAD = 'Your job "{collection}" was archived: {why}.'


def render_job_retired_report(report: JobRetiredReport) -> str:
    """The whole retirement delivery turn's input, rendered from
    framework-assembled facts alone — the sibling of ``render_job_report``
    (#2975).  Pure and deterministic, same as every render here, so the exact
    text the chat agent reads is pinned by a whole-render test.  No link table:
    the facts carry no URLs."""
    return "\n".join(
        [
            _RETIRED_REPORT_LEAD.format(collection=report.collection, why=report.why),
            _REPORT_ABOUT.format(description=report.description),
        ]
    )


def build_job_retired_report(db: Database, collection: str) -> JobRetiredReport | None:
    """Assemble the retirement fact set for one queued row (#2975): the
    collection row's own words, and the archive event's own note off the
    mutation ledger — the latest ARCHIVED event with actor SYSTEM, which is
    exactly the event that fired this report (a chat-side archive carries
    ``USER_RUN`` and never queues one).

    ``None`` when the collection has vanished or the ledger carries no system
    archive for it — the honest failure ``build_job_report`` names, logged and
    recorded on the row rather than composed from a guess."""
    row = db.memories.get(collection)
    if row is None:
        logger.error(
            "Retirement event for '%s' cannot be built — the collection no longer exists",
            collection,
        )
        return None
    events = db.mutations.history(collection, 1, entity_type=MutationEntityType.COLLECTION)
    event = next((e for e in events if e.action == MutationAction.ARCHIVED.value), None)
    detail = mutation_detail(event) if event is not None else None
    why = detail.note if detail is not None else None
    if event is None or event.actor != MutationActor.SYSTEM.value or why is None:
        logger.error(
            "Retirement event for '%s' cannot be built — the ledger carries no "
            "system-actor archive note to read the cause from",
            collection,
        )
        return None
    return JobRetiredReport(collection=collection, description=row.description, why=why)


class JobFailedReport(BaseModel):
    """Everything the delivery chat turn is given about a job run that failed for
    good (#2975): a CADENCE cycle whose occurrence-retry budget spent out.
    ``reason`` is the failed run's own recorded outcome stamp — the terminal
    cause an aborted run names (#1909), or the unhealthy-end line — read off
    the run's promptlog rows at compose time, never re-derived."""

    collection: str
    description: str
    run_id: str
    reason: str


_FAILED_REPORT_LEAD = 'Your job "{collection}"\'s run failed: {reason}.'


def render_job_failed_report(report: JobFailedReport) -> str:
    """The whole failure delivery turn's input, rendered from
    framework-assembled facts alone — the sibling of ``render_job_report``
    (#2975).  Pure and deterministic, same as every render here.  No link
    table: the facts carry no URLs."""
    return "\n".join(
        [
            _FAILED_REPORT_LEAD.format(collection=report.collection, reason=report.reason),
            _REPORT_ABOUT.format(description=report.description),
        ]
    )


def build_job_failed_report(db: Database, collection: str, run_id: str) -> JobFailedReport | None:
    """Assemble the failure fact set for one queued row (#2975): the collection
    row's own words, and the failed run's own outcome stamp.  The stamp is read
    at COMPOSE time — the row is enqueued in the cycle's settlement, before
    ``_tag_promptlog_run`` writes it — and it is the same read
    ``recent_run_outcomes`` serves: the run's ``run_reason`` when it carries
    one (an abort's terminal cause, a stop reason), else the ``run_outcome``
    enum.

    ``None`` when the collection has vanished or the run's stamp is unreadable
    (rows pruned) — logged, and the row is failed outright by the caller
    rather than composed from a guess."""
    row = db.memories.get(collection)
    if row is None:
        logger.error(
            "Failure event for '%s' (run %s) cannot be built — the collection no longer exists",
            collection,
            run_id,
        )
        return None
    stamped = [
        prompt
        for prompt in reversed(db.messages.get_run_prompts(run_id))
        if prompt.run_outcome is not None
    ]
    if not stamped:
        logger.error(
            "Failure event for '%s' (run %s) cannot be built — the run carries no "
            "outcome stamp to read the failure from",
            collection,
            run_id,
        )
        return None
    prompt = stamped[0]
    outcome = prompt.run_outcome
    assert outcome is not None  # the list comprehension above filtered for it
    reason = prompt.run_reason or outcome
    return JobFailedReport(
        collection=collection, description=row.description, run_id=run_id, reason=reason
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
