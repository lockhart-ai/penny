"""SendQueueDrainer — delivers queued outbound messages on the send cooldown.

A finished cycle's notification is enqueued into ``db.send_queue`` rather than
dropped when the autonomous-send cooldown hasn't elapsed.  This task is the other
half: each tick it pops the oldest pending message and delivers it, but only once
the flat-interval cooldown has cleared — so no message is lost, just delayed.

It's a plain ``ScheduledTask`` with ONE model-shaped step (#2656): a row with no
content yet is a composing report — the hand-off a notifying cycle queued, or an
autonomous job event did (#2975).  For one of those the drainer builds the fact
set the row's own ``kind`` names — a :class:`JobReport` from the cycle's written
entries (FULL, never cut), a :class:`SandboxReport` from the previewed trace
(#2964), or a retirement/failure fact set (#2975) — runs ONE chat-agent turn on
it, and the message the turn writes is what gets gated, stored on the row and
delivered.  Every ordinary row (content already set) goes out with no model call
at all.

``deliver_composed`` (#2913) is the OTHER entry point: a caller that already ran
its own delivery turn and gated what it wrote (the startup changelog announcement,
#2846) sends it straight through the channel the same way a drained row is —
recipient resolution, the messagelog stamp naming the collection and the run that
wrote it — with no queue row and no idle wait in between.  It shares ``_deliver``
with ``execute()`` so the two paths record a send identically.

The cooldown logic is identical to the gate the retired ``send_message`` tool
used to apply inline: bypassed when the user has spoken since Penny's last message (the queued
send is then conversational, not autonomous), otherwise the message waits
``SEND_COOLDOWN_SECONDS`` since Penny's previous outgoing message of any kind
(chat reply included — the cooldown is per-Penny, not per-collection).

There are delivery lanes, read off the queued row's own ``origin`` (#1939).  The
cooldown is anti-spam for messages nobody asked for; a message from a cycle the USER
pressed "run this now" on is one they are sitting in front of waiting for, so it is
conversational in exactly the sense the user-spoke-since bypass already means, and it
goes out on the next tick.  Observed live: a user's on-demand run finished and its
notification then waited ~10.5 minutes, because an unrelated chat reply seconds earlier
had started the cooldown.  Which lane a row is in is a READ of what set its cycle
running, never a re-decision here.  A sandbox preview's own origin (#2964) shares
``ON_DEMAND``'s cooldown-skipping lane for the same reason — someone is sitting in the
very conversation that asked for the preview — but composes a different report (the
previewed cycle's own tool-call trace, never written entries, since nothing it did
was real).

A queued report is also held back while ANY job run is in progress (#3025): the
drainer's delivery turn is itself a model call, and running it beside a cycle on the
one local model halves both.  The gate is the collector's own ``is_cycle_running()`` —
one structural read covering scheduled cycles, a user's "run this now" dispatch and a
sandbox preview (every kind of run passes through the collector's cycle lock) — read
at each tick, never a timer.  A held-back row stays pending: delayed, never dropped,
and the next tick with no run in progress delivers it.  ``deliver_composed`` is not
gated — it is a caller's own already-composed send, not this task's tick.

Compose failures are bounded and visible (#2656): a turn that produced nothing
sendable bumps the row's ``attempts`` and leaves it pending for the next idle tick,
until ``COLLECTOR_RETRY_ATTEMPTS`` is spent and the row is stamped ``failed_at`` —
kept, never delivered, logged loudly.  A DECLINE (a refusal-shaped reply, a mute
taken since the enqueue, no recipient) is a decision rather than a failure: the row
is stamped ``cancelled_at`` and the reason is logged.  A foreground message cancels
the compose mid-turn like any background work — the row stays pending untouched and
the next idle window tries again.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from penny.constants import CycleTrigger, PennyConstants, QueueItemKind
from penny.llm.call_link import CallLink
from penny.notification import (
    JobFailedReport,
    JobRetiredReport,
    SandboxReport,
    build_job_failed_report,
    build_job_report,
    build_job_retired_report,
    build_sandbox_report,
    withheld_reason,
)
from penny.text_validity import half_formed_send_reason

if TYPE_CHECKING:
    from penny.agents.chat import ChatAgent
    from penny.agents.models import ControllerResponse
    from penny.channels.base import MessageChannel
    from penny.config import Config
    from penny.database import Database
    from penny.database.models import SendQueueItem
    from penny.notification import JobReport

logger = logging.getLogger(__name__)


class JobRunTracker(Protocol):
    """The collector surface the drainer reads (#3025): whether any job run is in
    progress.  Structural, like ``ScheduledTask`` — the real ``Collector``
    satisfies it (``is_cycle_running`` reads its cycle lock, the one chokepoint
    every scheduled cycle, "run it now" dispatch and sandbox preview passes
    through), and the drainer never imports the agent module."""

    def is_cycle_running(self) -> bool: ...


class SendQueueDrainer:
    """Deliver the oldest queued message its lane allows out this tick."""

    name = "send_queue_drain"

    def __init__(
        self, db: Database, config: Config, chat_agent: ChatAgent, collector: JobRunTracker
    ) -> None:
        self._db = db
        self._config = config
        self._chat_agent = chat_agent
        # The collector's own run state (#3025) — ``is_cycle_running()`` is the one
        # structural read this drainer makes off it: a job run in progress holds a
        # queued report's delivery turn back, since both are model calls on the one
        # local model.
        self._collector = collector
        self._channel: MessageChannel | None = None

    def set_channel(self, channel: MessageChannel) -> None:
        """Bind the channel the drained messages are delivered through."""
        self._channel = channel

    async def execute(self) -> bool:
        """Deliver one queued message if its lane allows; return whether one went out.

        Returns False (no work) when a job run is in progress (#3025), when the
        queue holds nothing deliverable this tick, or when no channel/recipient is
        wired — the scheduler then moves on to the next schedule this tick.  A
        composing report whose turn produced nothing usable also returns False:
        the row stays pending (or fails out) and a later tick picks the queue up
        again — but no message behind it goes out until this head resolves,
        because the queue is FIFO.
        """
        if self._collector.is_cycle_running():
            # A job run — a scheduled cycle, a "run this now" dispatch or a sandbox
            # preview — is mid-flight on the local model.  The delivery turn is a
            # model call too; running the two together halves both (#3025).  The
            # row stays pending exactly as it is: delayed, never dropped, and the
            # next tick with no run in progress delivers it.
            return False
        if self._channel is None:
            return False
        item = self._next_deliverable()
        if item is None:
            return False
        if self._is_report(item):
            composed = await self._compose(item)
            if composed is None:
                return False
            content, composed_run_id = composed
            if item.id is not None:
                self._db.send_queue.store_composed(item.id, content)
        else:
            # An ordinary row (content already set, no model call here) still
            # carries whatever run wrote it — ``None`` when there never was one
            # (a plain ``enqueue``), or an EARLIER tick's own id when that tick
            # already composed and stored it (a report row whose first delivery
            # attempt found no recipient: ``item.content`` is no longer empty, so
            # this branch is taken without re-composing, and ``report_run_id`` is
            # still the turn that wrote it, not lost to a bare ``None``).
            content, composed_run_id = item.content, item.report_run_id
        recipient = await self._deliver(content, collection=item.collection, run_id=composed_run_id)
        if recipient is None:
            return False
        if item.id is not None:
            self._db.send_queue.mark_sent(item.id)
        logger.info("send_queue drained: %s → %s", item.collection, recipient)
        return True

    async def deliver_composed(self, content: str, *, collection: str, run_id: str) -> bool:
        """Deliver an already-composed, already-gated message straight through
        the channel — no queue row, no idle/cooldown wait (#2913).

        For a caller whose own turn wrote and gated ``content`` up front (the
        startup changelog announcement, #2846) and just needs it to reach the
        user the moment the channel is ready. Shares ``_deliver`` with
        ``execute()`` so this is recorded exactly like a drained row would be:
        the same recipient resolution, and the same messagelog stamp (mechanism
        ``collection``, ``run_id`` the run that wrote it). Returns whether a
        channel/recipient was available to deliver through.
        """
        recipient = await self._deliver(content, collection=collection, run_id=run_id)
        return recipient is not None

    async def _deliver(self, content: str, *, collection: str, run_id: str | None) -> str | None:
        """Resolve the recipient and send one message through the channel,
        stamped the way every autonomous send is (#1568/#2630).

        Shared by ``execute()`` (a drained queue row) and ``deliver_composed()``
        (a caller's own pre-composed, already-gated send, #2913) so both record
        a delivered message identically. Returns the recipient reached, or
        ``None`` when there is no channel or no recipient to reach.
        """
        if self._channel is None:
            return None
        recipient = self._resolve_recipient()
        if recipient is None:
            return None
        await self._channel.send_response(
            recipient=recipient,
            content=content,
            parent_id=None,
            author=collection,
            quote_message=None,
            # Emission provenance (#1568): names the mechanism (the collector,
            # or the startup changelog announcement) onto the delivered
            # messagelog row so the send names its cause and "which mechanism
            # sent this?" is a read, not a diagnosis.  For a composed report
            # this is what keeps "what you last told the user" (#2385) a read:
            # the newest outgoing row stamped with this collection.
            mechanism=collection,
            # The run whose reply this delivery carries (#2630) — the chat turn
            # that wrote the message, whether composed by the drainer or handed
            # in already composed.  ``None`` only for a row with no real run
            # behind it at all.
            run_id=run_id,
        )
        return recipient

    @staticmethod
    def _is_report(item: SendQueueItem) -> bool:
        """A composing row (#2656, widened #2975): the hand-off row a notifying
        cycle or an autonomous job event queued — no content yet, the message
        written later by the delivery chat turn.  The structural mark, read
        once: ``content`` being empty on an ORDINARY row is not a state the
        queue holds (the enqueue chokepoints refuse it), so it is unambiguous.
        WHAT to build is the row's own ``kind`` (#2975), read in
        ``_build_report``."""
        return not item.content

    def _build_report(
        self, item: SendQueueItem
    ) -> JobReport | SandboxReport | JobRetiredReport | JobFailedReport | None:
        """The fact set one composing row's message is built from, by the row's
        own ``kind`` (#2975) — the structural mark set once at enqueue, never a
        guess from the row's content (a run-limit one-shot's last run looks
        exactly like an ordinary report's run, so no read off the run record
        alone could tell them apart).

        ``report`` — the notifying cycle's :class:`JobReport` off the cycle's
        written entries, or a SANDBOX origin's :class:`SandboxReport` off the
        previewed cycle's own tool-call trace (#2964).  ``job_retired`` — the
        :class:`JobRetiredReport` off the mutation ledger's system-actor
        archive note (``report_run_id`` may be ``None``: a while-down expiry
        has no run to attribute).  ``job_failed`` — the
        :class:`JobFailedReport` off the failed run's own outcome stamp, keyed
        on ``report_run_id``, which that kind always carries.

        ``None`` when the fact set cannot be assembled — the collection gone,
        or the recorded cause unreadable; the builders log which."""
        kind = QueueItemKind(item.kind)
        if kind is QueueItemKind.REPORT:
            assert item.report_run_id is not None
            if CycleTrigger(item.origin) is CycleTrigger.SANDBOX:
                return build_sandbox_report(self._db, item.collection, item.report_run_id)
            return build_job_report(
                self._db, item.collection, item.report_run_id, trigger=CycleTrigger(item.origin)
            )
        if kind is QueueItemKind.JOB_RETIRED:
            return build_job_retired_report(self._db, item.collection)
        assert item.report_run_id is not None
        return build_job_failed_report(self._db, item.collection, item.report_run_id)

    async def _compose(self, item: SendQueueItem) -> tuple[str, str] | None:
        """Write the message for one composing report, and account for it either way.

        Builds the report off the row's own kind (``_build_report``, #2975) and
        runs the one delivery turn it rides — then runs what the turn wrote
        through the same send gate either way.  Returns ``(content, run_id)`` —
        the message and the turn that wrote it — or ``None`` when this tick
        delivered nothing and the row was dealt with: a DECLINE was stamped
        cancelled, a failure was counted against the row's bounded budget, and
        a cancellation left the row exactly as it was."""
        report = self._build_report(item)
        if report is None:
            self._fail(item, "the fact set could not be built")
            return None
        sender = self._db.users.get_primary_sender()
        if sender is None:
            self._decline(item, "there is no registered recipient")
            return None
        run_id = uuid.uuid4().hex
        try:
            response = await self._dispatch_report_turn(report, sender, run_id)
        except asyncio.CancelledError:
            # Foreground priority — the row stays pending, untouched, and the next
            # idle window composes it.  Nothing about the attempt is recorded: the
            # attempt did not fail, it was interrupted.
            raise
        except Exception:
            logger.exception("Report turn for '%s' (row %s) died", item.collection, item.id)
            self._record_failure(item, "the delivery turn died")
            return None
        answer = (response.answer or "").strip() if response is not None else ""
        if not answer:
            self._record_failure(item, "the delivery turn produced no reply")
            return None
        # Not a WHOLE message is a COMPOSE failure (retried, bounded) — the same rule
        # the retired composer's acceptance rule re-rolled on; a refusal, a mute or a
        # missing recipient is a DECISION (withheld on purpose).
        half_formed = half_formed_send_reason(answer)
        if half_formed is not None:
            self._record_failure(item, f"the message was not a complete one ({half_formed})")
            return None
        reason = withheld_reason(self._db, answer)
        if reason is not None:
            self._decline(item, reason)
            return None
        return answer, run_id

    async def _dispatch_report_turn(
        self,
        report: JobReport | SandboxReport | JobRetiredReport | JobFailedReport,
        sender: str,
        run_id: str,
    ) -> ControllerResponse:
        """The one delivery turn this report rides — ``handle_job_report`` for a
        :class:`JobReport`, ``handle_sandbox_report`` for a :class:`SandboxReport`
        (#2964), ``handle_job_event_report`` for either autonomous job event
        (#2975).  Dispatched on the report's OWN type, never the queue row's
        origin a second time, so this can never disagree with which report
        ``_compose`` actually built."""
        call_link = CallLink.top(run_id)
        if isinstance(report, SandboxReport):
            return await self._chat_agent.handle_sandbox_report(report, sender, call_link=call_link)
        if isinstance(report, (JobRetiredReport, JobFailedReport)):
            return await self._chat_agent.handle_job_event_report(
                report, sender, call_link=call_link
            )
        return await self._chat_agent.handle_job_report(report, sender, call_link=call_link)

    def _record_failure(self, item: SendQueueItem, reason: str) -> None:
        """One failed compose against the row's bounded budget (#2656).

        Within ``COLLECTOR_RETRY_ATTEMPTS`` the row stays pending for the next idle
        tick; once the budget is spent it is stamped ``failed_at`` — kept, never
        delivered, logged loudly.  Visible degradation, never silence: the cycle's
        entries stand either way, and the queue's own row says what happened."""
        if item.id is None:
            return
        attempts = self._db.send_queue.bump_compose_attempts(item.id)
        if attempts >= PennyConstants.COLLECTOR_RETRY_ATTEMPTS:
            self._db.send_queue.mark_failed(item.id)
            logger.error(
                "Job report for '%s' failed %d times (%s) — sending nothing",
                item.collection,
                attempts,
                reason,
            )
        else:
            logger.warning(
                "Job report for '%s' compose failed (%s) — attempt %d of %d, row stays pending",
                item.collection,
                reason,
                attempts,
                PennyConstants.COLLECTOR_RETRY_ATTEMPTS,
            )

    def _fail(self, item: SendQueueItem, reason: str) -> None:
        """A report that cannot even be BUILT cannot improve by waiting — straight to
        the terminal state, bypassing the retry budget the composable failures get."""
        if item.id is None:
            return
        self._db.send_queue.mark_failed(item.id)
        logger.error(
            "Job report for '%s' failed outright (%s) — sending nothing", item.collection, reason
        )

    def _decline(self, item: SendQueueItem, reason: str) -> None:
        """A DELIBERATE withhold: the gate declined, not the model.  The row is
        stamped cancelled — the same audit-trail state an archived collector's
        queued sends get (#1634) — and the reason is logged."""
        if item.id is not None:
            self._db.send_queue.mark_cancelled(item.id)
        logger.info("Job report for '%s' withheld — %s", item.collection, reason)

    def _next_deliverable(self) -> SendQueueItem | None:
        """The oldest queued message this tick is allowed to deliver.

        The head first, so an empty queue costs one indexed read and nothing else —
        this runs on every idle tick.  With something waiting, the cooldown decides:
        cleared, the head goes; still running, the only messages that may go out are
        the ones the user is waiting on, and they are asked for BY LANE rather than
        read off the head, because an older cadence message sitting out its cooldown
        would otherwise hold them behind it for the rest of the window.  A sandbox
        preview's report (#2964) is asked for the same way ``on_demand`` is — the
        user is sitting in the very conversation that asked for it — tried second
        only because ``on_demand`` already existed first; neither lane's head
        blocks the other's."""
        item = self._db.send_queue.next_pending()
        if item is None or self._cooldown_elapsed():
            return item
        return self._db.send_queue.next_pending(
            origin=CycleTrigger.ON_DEMAND
        ) or self._db.send_queue.next_pending(origin=CycleTrigger.SANDBOX)

    def _resolve_recipient(self) -> str | None:
        """Where a drained message is delivered — the default device's own
        identifier (that's the routing #1373 needed for iOS).

        #2765 made this true for every channel: Signal and Discord now
        register the user's own endpoint (not Penny's) as the default
        device, so there is no longer a channel whose default device row
        would misroute a read-by-identifier to Note-to-Self. The primary
        sender is still the fallback for the one case with no default
        device registered at all (a fresh Signal install before its first
        message ever arrives).
        """
        default = self._db.devices.get_default()
        if default is not None:
            return default.identifier
        return self._db.users.get_primary_sender()

    # ── Cooldown (moved verbatim from SendMessageTool) ────────────────────

    def _cooldown_elapsed(self) -> bool:
        """Flat-interval cooldown between *autonomous* sends.

        When ``count == 0`` the user has spoken since Penny last sent, so the
        queued message is conversational, not autonomous — deliver immediately.
        Otherwise wait ``SEND_COOLDOWN_SECONDS`` since Penny's previous outgoing
        message (any author — a chat reply holds off a queued autonomous ping).
        """
        count = self._count_sends_since_user_message()
        if count == 0:
            return True
        latest = self._latest_send_time()
        if latest is None:
            return True
        elapsed = (_to_naive(self._db.clock.now()) - _to_naive(latest)).total_seconds()
        return elapsed >= self._config.runtime.SEND_COOLDOWN_SECONDS

    def _latest_send_time(self) -> datetime | None:
        """Created-at of Penny's most recent outgoing message."""
        log = self._db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        entries = log.newest_entries(k=1) if log is not None else []
        return entries[0].created_at if entries else None

    def _count_sends_since_user_message(self) -> int:
        """Number of Penny's outgoing messages newer than the latest user message.

        Bounded read: ``read_since(cutoff)`` pushes the ``timestamp > cutoff``
        filter into SQL, so this touches only the post-cutoff tail.  With no user
        message yet, any prior Penny send counts — a single bounded existence
        probe."""
        log = self._db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        if log is None:
            return 0
        cutoff = self._latest_user_message_time()
        if cutoff is None:
            return len(log.newest_entries(k=1))
        return len(log.read_since(cutoff))

    def _latest_user_message_time(self) -> datetime | None:
        """Created-at of the most recent ``user-messages`` entry."""
        log = self._db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
        entries = log.newest_entries(k=1) if log is not None else []
        return entries[0].created_at if entries else None


def _to_naive(value: datetime) -> datetime:
    """Strip tzinfo if present so naive/aware mixes don't crash arithmetic."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
