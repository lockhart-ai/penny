"""SelfStateHeader — Penny's operational self-state, rendered deterministically
into the chat agent's system prompt (#1555).  Jobs and Your memory ride the
STATIC prefix (``SelfStateHeader.render_static``); Recent activity and About the
user ride the per-turn DYNAMIC message (``render_dynamic``) — see #2814 and each
method's own docstring below.

The ambient inversion: instead of *predicting* which of the user's stored content
might be relevant this turn (speculative, usually unused, occasionally
misleading), the header carries **deterministic facts about Penny's own
situation** — what mechanisms she's running, what she just did, what stores exist
to read from, and the durable facts about the user. All of it is a pure *render*
of the registry (``memory`` rows, #1566) and the ledger (``promptlog`` runs +
``mutation_event``, #1560) — no model judgment, no relevance guess.

Its job is twofold: it is the L1 cache of Penny's situation (facts needed nearly
every operational turn, which she has no anchor to fetch mid-turn), and it is the
index into everything else (every id it names — a collection, a run's target — is
one guess-free tool call from the detail). Sections:

- **Active mechanisms** — EVERY collector, archived-inclusive, no cap (#2288): name,
  status, cadence, end condition, last-run outcome. "what's running right now?" A live
  job hangs a second, two-space-indented line under its row (#2495): the page it reads
  (the first ``https?://`` URL in its prompt) and the instructions its cycles follow —
  capped at a word-visible budget, with a note naming the ``memory_get`` hop to the
  whole prompt, so what a job actually does is ambient and an ask that resembles a
  running job can be answered by pointing at it.
- **Recent activity** — background runs, configuration mutations, and autonomous
  sends, interleaved in one time-ordered block at rollup altitude (one line each;
  one run's per-call detail is one ``run_get(run <id>)`` hop away). A run line
  that wrote carries a ``· wrote '<key>' → `<collection>``` clause naming what it
  changed (from the #1560 ``last_written_by_run_id`` stamp, #1641), so a just-
  written fact is 0 calls away and its full entry is one ``collection_list``
  hop. The *complement* of the
  conversation — chat turns (direct replies) are already in context, so they're
  never duplicated here; only mechanism-authored sends (``mechanism`` non-NULL,
  #1568) appear. **One exception (#2355 part D):** a chat turn that FAILED (fell
  back to the last-draw-or-account machinery) IS a background fact in the sense
  that matters here — asked about it a turn, or a day, later, there is no
  conversation left to carry it — so it gets its own ``run <id> · <when> · chat
  turn failed: <reason>`` line, its typed ``run <id>`` the same ``run_get`` anchor
  every other activity line renders.
- **Your memory** — the map of stores: EVERY live collection and log, no cap
  (#2288), names + one-line scope — the index for an anchored lookup, never the
  content. Its entry count is shown for a collection or log the user or a job
  owns (moves only on a real write); the five system-log facades
  (``PennyConstants.SYSTEM_LOGS``) show their name and scope with NO live count
  (#2814) — that count ticks on every ordinary turn, and this section rides the
  cached static prefix now, so showing it would rebuild the prefix every turn.
  One compact trailing line names every ARCHIVED collection and log by
  name (``archived: `<name>`, …``), omitted entirely when nothing is archived — an
  archived row with no job has nowhere else in this header to appear, so this line
  is the complete list, not a remainder.
- **About the user** — the durable user-fact core (name, timezone, location):
  deterministic facts, not a relevance guess, so personality survives without a
  lookup.
- A pointers line naming the fetch tools for anything deeper.

Standing rules (#2633): the seeded ``rules`` collection — the things the user has
asked Penny to always or never do — renders in full wherever she acts, oldest first,
each entry as ``- <key>: <content> (added <date>)``, from ONE shared render
(``render_standing_rules``; collector cycle and notify composer show the whole
section).  Only this collection renders this way: another collection's content never
enters the prompt as standing instructions.  In the chat prompt the rules render in
the instruction part instead (``standing_rules_instruction``), where she reads
instructions rather than state.  With no rule set the section is omitted entirely
(#2686) — the ``rules`` collection stays reachable through the store map, which
always lists it with its description.

Autonomous-send (emission) rows join the recent-activity block (#1568): each
delivered mechanism send renders one ``sent · <when> · <mechanism> — "<snippet>"``
line, interleaved by time with the runs and mutations, so Penny sees her own
background emissions ambiently; older sends carry the same provenance inline on
every ``penny-messages`` read, and ``memory_get(<mechanism>)`` is the hop to
the creating request (#1566).
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from penny.constants import PennyConstants
from penny.database.memory.types import render_key_value
from penny.database.mutation_store import mutation_change_summary
from penny.datetime_utils import format_log_timestamp, format_user_timestamp, user_timezone_name
from penny.tools.collection_instantiation import describe_schedule
from penny.tools.job_summary import job_summary_line

if TYPE_CHECKING:
    from datetime import datetime

    from penny.database.database import Database
    from penny.database.memory.types import RunWrite
    from penny.database.message_store import (
        ChatFailureActivity,
        EmissionActivity,
        RunActivity,
        RunOutcomeStamp,
    )
    from penny.database.models import MemoryEntry, MemoryRow, MutationEvent


class SelfStateHeader:
    """Renders Penny's operational self-state from the registry + ledger.

    Constructed per chat turn and read purely — every method is a deterministic
    projection of stored rows. Two sections — Jobs and Your memory — ride
    ``ChatAgent``'s cached STATIC prefix (:meth:`render_static`, #2814): both are
    pure renders of the registry alone (schedule rule, ``notify``, ``expires_at``,
    last-run stamp, entry counts, descriptions — never ``now``, confirmed turn to
    turn by ``test_static_system_prefix_rebuilds_when_a_job_or_collection_changes``),
    so they are byte-identical across turns unless a job or a collection actually
    changed — rare next to every turn, the same bet the conversation map makes.
    The rest — Recent activity (which churns on every run/mutation/emission) and
    About the user — stay in the per-turn dynamic tail (:meth:`render_dynamic`),
    since those genuinely can change every turn. ``render()`` keeps the whole,
    historical, single-message shape (identical output, unused by ``ChatAgent``
    since #2814) for the many tests below that assert on ``SelfStateHeader`` as a
    unit rather than on where its sections land.
    """

    HEADER = "## Penny's current state"
    MECHANISMS_HEADER = "### Jobs"
    ACTIVITY_HEADER = "### Recent activity"
    MAP_HEADER = "### Your memory"
    STANDING_RULES_HEADER = "### Standing rules"
    DURABLE_HEADER = "### About the user"

    # Whether proactive notifications are going out, rendered in BOTH directions and
    # ALWAYS present (#1919).  Nothing said whether the user was muted, so the mute
    # tools' descriptions were the only thing that could carry the state — and they
    # conditioned every trigger on "an earlier pause" the model had no way to verify,
    # so it correctly declined to assume one.  A visible "on" is as load-bearing as a
    # visible "muted": with no line at all there is a hidden state to imagine, and the
    # guard against acting on an imagined one is that the real one is always stated.
    #
    # Both lines name WHAT is paused, because the two channels are different things —
    # a mute silences the jobs, never the replies — and a line that said only
    # "notifications are paused" would read as Penny having been told to stop talking.
    NOTIFICATIONS_MUTED = (
        "notifications: muted — proactive task notifications are paused; replies to "
        "the user's own messages are unaffected"
    )
    NOTIFICATIONS_ON = "notifications: on — proactive task notifications are delivered"

    # Whether ONE mechanism tells the user what it finds — a clause on its own row, in the
    # same both-directions-always shape as the global line above (#1919).
    #
    # Named by its PARAMETER: `notify` is the exact `collection_configure` argument that flips
    # this row, so display form == invocation form and the anchor is copied rather than
    # translated.  Human wording was measured and is worse HERE — "tells you" read as the
    # `notifications_*` tool family, and a sample asked to silence ONE watch reached for
    # the global mute — while the global line one row up keeps human wording for the
    # mirror-image reason: its lever really IS that tool pair.
    #
    # This clause is HALF of a mechanism; on its own it moved nothing (measured: silence
    # 0.92 → 0.80 → 0.68 across two wordings, restored here with the other half — the
    # `collection_configure` description that says the argument turns a collection's
    # notifications on or off).  A state that shows a switch and a tool that never
    # mentions it are two ways of saying the same thing badly.
    MECHANISM_NOTIFIES = "notify: on"
    MECHANISM_QUIET = "notify: off"

    EMPTY_MECHANISMS = "(no jobs yet)"
    EMPTY_ACTIVITY = "(no recent activity)"
    EMPTY_MAP = "(no stores yet)"
    NO_PROFILE = "(no profile set yet)"

    # The overflow tail activity still shows past its cap — the fetch tool named so
    # the remainder is one guess-free call away (nothing is silently dropped; n≤1
    # holds). Mechanisms and the store map carry no cap (#2288): the whole
    # registry renders, so there is no overflow tail to name for either.
    MORE_ACTIVITY = "+ older activity — run_list(<source>)"

    # The store map's compact index of archived names (#2288) — the complete list,
    # not a remainder: an archived collection with no job appears nowhere else.
    ARCHIVED_LABEL = "archived:"

    POINTERS = (
        "To look deeper: memory_get(<name>) for a collection's full config and "
        "change history, run_get(run <id>) for one run's tool calls, "
        "collection_list(<name>) reads a store's entries, and find(query=<text>) "
        "searches them all by meaning — a collection, a stored entry, or "
        "something said in the conversation."
    )

    def __init__(self, db: Database, user: str | None) -> None:
        self.db = db
        self.user = user

    # ── Summary ──────────────────────────────────────────────────────────────

    def render(self) -> str:
        """The whole ``## Penny's current state`` block — a table of contents of
        Penny's operational situation, each section a deterministic render.  A
        section that has nothing to show renders nothing.  The standing rules do not
        render here at all: with a rule standing they belong to the chat prompt's
        instruction part (``standing_rules_instruction``), and with none the section
        is omitted entirely (#2686) — the store map still names ``rules``.

        Kept for its many existing callers that test ``SelfStateHeader`` as a
        self-contained unit (its own section-rendering logic — caps, cuts,
        mute-state wording, …) rather than where a section lands in the chat
        prompt.  ``ChatAgent`` itself no longer calls this: it calls
        :meth:`render_static` and :meth:`render_dynamic` instead (#2814), so this
        method's section order no longer matches either message — it is this
        class's own legacy whole-render, not a third shape the model ever sees."""
        sections = [
            self.HEADER,
            self._mechanisms_section(),
            self._activity_section(),
            self._memory_map_section(),
            self._durable_core_section(),
            self.POINTERS,
        ]
        return "\n\n".join(section for section in sections if section)

    def render_static(self) -> str:
        """Jobs + Your memory — the two SLOW-CHANGING sections (#2814) that ride
        ``ChatAgent``'s cached static system-prompt prefix instead of the per-turn
        dynamic tail.  Both are pure renders of the registry alone — no activity,
        no ``now`` — so this string is byte-identical turn to turn unless a job or
        a collection actually changed, the same cache bet the conversation map
        makes for the rest of the static message."""
        sections = [self._mechanisms_section(), self._memory_map_section()]
        return "\n\n".join(section for section in sections if section)

    def render_dynamic(self) -> str:
        """``## Penny's current state`` + Recent activity + About the user + the
        pointers line — what rides ``ChatAgent``'s per-turn dynamic system message
        (#2814): the activity block churns on every run/mutation/emission, so it
        can never be part of a cached prefix; the durable core and the pointers
        line are cheap enough to just ride along beside it rather than earn their
        own split."""
        sections = [
            self.HEADER,
            self._activity_section(),
            self._durable_core_section(),
            self.POINTERS,
        ]
        return "\n\n".join(section for section in sections if section)

    # ── Active mechanisms ────────────────────────────────────────────────────

    def _mechanisms_section(self) -> str:
        """The collectors — EVERY one, recency-first and archived-inclusive, each on
        one line: status · cadence · end condition · last-run outcome (#1566
        registry). No cap (#2288): the ambient index is her own situation in full,
        not a sample of it."""
        rows = self._mechanism_rows()
        outcomes = self.db.messages.latest_run_outcomes()
        lines = [self.MECHANISMS_HEADER, self._notifications_line()]
        lines.extend(self._mechanism_line(row, outcomes.get(row.name)) for row in rows)
        if not rows:
            lines.append(self.EMPTY_MECHANISMS)
        return "\n".join(lines)

    def _notifications_line(self) -> str:
        """Whether proactive notifications are going out right now — the state every
        mechanism line below it is subject to, which is why it opens the section.

        Read off the SAME key the notifier gates on (the primary sender, falling back to
        the turn's own user), so what this says and what actually happens at delivery
        cannot disagree.  With no user registered at all there is no mute row to hold,
        which reads as unmuted; whether anyone is reachable is the preflight's to say."""
        recipient = self.db.users.get_primary_sender() or self.user
        muted = self.db.users.is_muted(recipient) if recipient else False
        return self.NOTIFICATIONS_MUTED if muted else self.NOTIFICATIONS_ON

    def _mechanism_rows(self) -> list[MemoryRow]:
        """Every collector (a ``memory`` row with an ``extraction_prompt``), active
        and archived, sorted by name — ALL of them (#2288), so the order just needs
        to be deterministic regardless of seed timing (the inventory convention),
        which keeps the whole-render assertion stable."""
        rows = [row for row in self.db.memories.list_all() if row.extraction_prompt is not None]
        rows.sort(key=lambda row: row.name)
        return rows

    def _mechanism_line(self, row: MemoryRow, stamp: RunOutcomeStamp | None) -> str:
        """``- <name> — <status> · <cadence> · <notify> · <end> · <last run>`` — a retired
        mechanism drops the running clauses (it isn't running). A live job's line hangs a
        second, indented line under it (#2495) naming what it reads and follows; an
        archived row or a prompt-less one renders exactly as before."""
        if row.archived:
            parts = [f"archived {format_log_timestamp(row.updated_at)}", self._last_run(stamp)]
        else:
            parts = [
                "active",
                self._cadence(row),
                self._telling(row),
                self._end_condition(row),
                self._last_run(stamp),
            ]
        clauses = " · ".join(part for part in parts if part)
        line = f"- {row.name} — {clauses}"
        if not row.archived and row.extraction_prompt is not None:
            line = f"{line}\n{self._instructions_line(row)}"
        return line

    def _instructions_line(self, row: MemoryRow) -> str:
        """The hung second line under a live job's row (#2495) — the shared
        ``job_summary_line`` render (``penny.tools.job_summary``), indented two
        spaces under its row."""
        return f"  {job_summary_line(row)}"

    def _telling(self, row: MemoryRow) -> str:
        """Whether THIS mechanism tells the user what it finds — both directions, always,
        on a live row, exactly like the section's global notifications line.

        The line above says whether proactive messages go out AT ALL; this says whether
        this particular job is one of the things that would send one, which is the question
        a "turn off notifications for X" ask turns on.  A visible ``notify: off`` is as
        load-bearing as a visible ``notify: on`` — it is what a "turn them back on" ask has
        to flip, and with nothing rendered either way both directions are a hidden state to
        be guessed at."""
        return self.MECHANISM_NOTIFIES if row.notify else self.MECHANISM_QUIET

    def _cadence(self, row: MemoryRow) -> str:
        """The mechanism's schedule in plain words, with its exact rule alongside in a
        code span (#2425) — #1857's display == invocation contract stays true (the
        model still reads the copyable form it would write back), it just no longer
        stands alone as the text a reply would echo.  Empty when the collection has no
        schedule yet, so the mechanism line drops the clause."""
        return describe_schedule(row, user_timezone_name(self.db))

    def _end_condition(self, row: MemoryRow) -> str:
        """The mechanism's end condition — an expiry time, on the USER's own clock
        (#2206/#2251: the same clock a stated end is read on, so Penny's own ambient
        view of a job's end never disagrees with the clock it reads back on elsewhere),
        a one-shot / run quota, or empty when it runs indefinitely."""
        if row.expires_at is not None:
            when = format_user_timestamp(row.expires_at, user_timezone_name(self.db))
            return f"expires {when}"
        if row.max_runs == 1:
            return "one-shot"
        if row.max_runs is not None:
            return f"ends after {row.max_runs} runs"
        return ""

    @staticmethod
    def _last_run(stamp: RunOutcomeStamp | None) -> str:
        """``last run <OUTCOME> <when>`` for the mechanism's most recent completed
        cycle, or ``no runs yet`` when it has never completed one."""
        if stamp is None:
            return "no runs yet"
        return f"last run {stamp.outcome.upper()} {format_log_timestamp(stamp.finished_at)}"

    # ── Recent activity ──────────────────────────────────────────────────────

    def _activity_section(self) -> str:
        """Background runs, config mutations, and autonomous sends, interleaved
        newest-first at rollup altitude (one line each). Chat turns (direct replies)
        are excluded by construction — they're already the conversation; this is its
        complement (ledger, #1560; emissions, #1568)."""
        events = self._activity_events()
        limit = PennyConstants.SELF_STATE_ACTIVITY_LIMIT
        shown = events[:limit]
        lines = [self.ACTIVITY_HEADER]
        lines.extend(line for _, line in shown)
        if not shown:
            lines.append(self.EMPTY_ACTIVITY)
        if len(events) > limit:
            lines.append(self.MORE_ACTIVITY)
        return "\n".join(lines)

    def _activity_events(self) -> list[tuple[datetime, str]]:
        """(timestamp, rendered line) for recent runs + mutations + emissions,
        newest first.

        Each source is fetched at ``cap + 1`` (so the section can tell that a
        ``cap + 1``-th event exists and show the overflow tail), merged, and
        sorted by time — so the block stays flat as activity grows and the newest
        events of any kind win a slot.

        The runs' writes are joined once (``writes_by_run`` over the fetched run
        ids, #1641) rather than per line, so the writes clause stays one bounded
        read."""
        fetch = PennyConstants.SELF_STATE_ACTIVITY_LIMIT + 1
        runs = self.db.messages.recent_collector_runs(fetch)
        writes = self.db.memories.writes_by_run([run.run_id for run in runs])
        events: list[tuple[datetime, str]] = [
            (run.finished_at, self._run_line(run, writes.get(run.run_id, []))) for run in runs
        ]
        events.extend(
            (event.created_at, self._mutation_line(event))
            for event in self.db.mutations.recent(fetch)
        )
        events.extend(
            (emission.sent_at, self._emission_line(emission))
            for emission in self.db.messages.recent_emissions(fetch)
        )
        events.extend(
            (failure.finished_at, self._chat_failure_line(failure))
            for failure in self.db.messages.recent_failed_chat_turns(fetch)
        )
        events.sort(key=lambda event: event[0], reverse=True)
        return events

    @staticmethod
    def _emission_line(emission: EmissionActivity) -> str:
        """``sent · <when> · <mechanism> — "<snippet>"`` — a delivered autonomous
        send (#1568). The mechanism is the ``memory_get`` anchor (its #1566
        block names the creating request); the snippet is what Penny autonomously
        said, at rollup altitude."""
        return (
            f"sent · {format_log_timestamp(emission.sent_at)} · "
            f'{emission.mechanism} — "{emission.snippet}"'
        )

    @staticmethod
    def _chat_failure_line(failure: ChatFailureActivity) -> str:
        """``run <run_id> · <when> · chat turn failed: <reason>`` — a chat turn that
        fell back to the last-draw-or-account machinery (#2355 part D), the ONE
        exception to "chat turns are already the conversation": asked about it
        later, there is no conversation left to carry it, so it gets the same typed
        ``run <id>`` anchor every other activity line renders — ``run_get`` on it
        shows the tool call and the model call that did not finish."""
        return (
            f"{PennyConstants.RUN_EVENT_PREFIX}{failure.run_id} · "
            f"{format_log_timestamp(failure.finished_at)} · chat turn failed: {failure.reason}"
        )

    @staticmethod
    def _run_line(run: RunActivity, writes: list[RunWrite]) -> str:
        """``run <run_id> · <when> · <target> → <OUTCOME> (<n> calls)``, then — when
        the run wrote — a ``· wrote …`` clause naming what it changed (#1641).

        The run id is the typed anchor, consumed verbatim by ``run_get(run
        <id>)`` (its drill-down); the ``run `` tag is single-sourced with that
        tool's parse (``RUN_EVENT_PREFIX``) so the rendered token IS the argument.
        The clause GROWS the line — a no-write run renders byte-identical to the
        pre-#1641 shape."""
        plural = "" if run.call_count == 1 else "s"
        return (
            f"{PennyConstants.RUN_EVENT_PREFIX}{run.run_id} · "
            f"{format_log_timestamp(run.finished_at)} · "
            f"{run.target} → {run.outcome.upper()} ({run.call_count} call{plural})"
            f"{SelfStateHeader._writes_clause(writes)}"
        )

    @staticmethod
    def _writes_clause(writes: list[RunWrite]) -> str:
        """The ``· wrote …`` tail naming what a run wrote — one clause per
        collection it wrote keyed entries to, so the model sees its own writes
        ambiently (#1641). Empty (byte-identical to the pre-#1641 line) for a
        no-write run; keys render invocation-form so each pastes into a read
        tool's ``key=`` argument."""
        return "".join(
            f" · {SelfStateHeader._collection_writes(name, keys)}"
            for name, keys in SelfStateHeader._group_writes(writes)
        )

    @staticmethod
    def _group_writes(writes: list[RunWrite]) -> list[tuple[str, list[str]]]:
        """``(collection, keys)`` groups, preserving the store's ``(memory_name,
        created_at, id)`` order — same-collection writes are contiguous, so
        ``groupby`` clusters each collection's keys oldest-first."""
        return [
            (name, [write.key for write in group])
            for name, group in groupby(writes, key=lambda write: write.memory_name)
        ]

    @staticmethod
    def _collection_writes(name: str, keys: list[str]) -> str:
        """One collection's writes clause: a single write names its key
        (``wrote '<key>' → `<name>```); several compact to a count plus a bounded
        key sample (``wrote 3 entries → `<name>` ('k1', 'k2', …)``), the ``…`` tail
        showing more keys exist than are named (#1641)."""
        if len(keys) == 1:
            return f"wrote {render_key_value(keys[0])} → `{name}`"
        cap = PennyConstants.SELF_STATE_WRITES_KEY_SAMPLE
        sample = ", ".join(render_key_value(key) for key in keys[:cap])
        if len(keys) > cap:
            sample = f"{sample}, …"
        return f"wrote {len(keys)} entries → `{name}` ({sample})"

    @staticmethod
    def _mutation_line(event: MutationEvent) -> str:
        """``change · <when> · <entity> <action> by <actor> (run <id>) — <detail>``
        — the causing run id (present for a user-run mutation, absent for a system
        one) is a ``run_get(run <id>)`` anchor, single-sourced with that tool's
        parse (``RUN_EVENT_PREFIX``); the entity name resolves via
        ``memory_get``/``find``, and the detail carries the changed
        fields."""
        parts = [
            f"change · {format_log_timestamp(event.created_at)} · "
            f"{event.entity_name} {event.action} by {event.actor}"
        ]
        if event.run_id is not None:
            parts.append(f"({PennyConstants.RUN_EVENT_PREFIX}{event.run_id})")
        line = " ".join(parts)
        summary = mutation_change_summary(event)
        return f"{line} — {summary}" if summary else line

    # ── The map of stores ────────────────────────────────────────────────────

    def _memory_map_section(self) -> str:
        """The index of readable stores — EVERY non-archived memory (collection or
        log) by name, shape, and one-line scope, plus an entry count for the ones
        whose count changes only on a real write. Names, never content: any user
        fact is one anchored read away using a name shown here.  No cap (#2288) —
        the live rows render in full, and the archived ones get their own compact
        line below (``_archived_line``) rather than a count that hides them.

        This section rides the static prefix (#2814, ``render_static``), so the
        count is OMITTED for the five system-log facades
        (``PennyConstants.SYSTEM_LOGS`` — ``user-messages``/``penny-messages``/
        ``browse-results``/``collector-runs``/``run-health``): their ``entry_counts()`` value is
        read straight off ``messagelog``/``promptlog``, which grows on every
        ordinary turn (an incoming message, a reply, a browse call), not on a job
        or collection changing — showing it here would rebuild the cached prefix
        every turn, defeating the point of moving this section into it.  The
        facade's NAME still renders (the ``log_read`` anchor, n≤1 reachability),
        only its live count does not; a collection or log the user or a job owns
        keeps its count, which only moves on a real write."""
        rows = self.db.memories.list_all()
        live = [row for row in rows if not row.archived]
        counts = self.db.memories.entry_counts()
        lines = [self.MAP_HEADER]
        lines.extend(self._memory_map_line(row, counts) for row in live)
        if not live:
            lines.append(self.EMPTY_MAP)
        archived = self._archived_line(rows)
        if archived:
            lines.append(archived)
        return "\n".join(lines)

    @staticmethod
    def _memory_map_line(row: MemoryRow, counts: dict[str, int]) -> str:
        """One store-map row: ``- <name> (<type>, <n> entries) — <description>``,
        or ``- <name> (<type>) — <description>`` for a system-log facade
        (``PennyConstants.SYSTEM_LOGS``), whose live count would otherwise rebuild
        the static prefix on every ordinary turn (see ``_memory_map_section``)."""
        if row.name in PennyConstants.SYSTEM_LOGS:
            return f"- {row.name} ({row.type}) — {row.description}"
        return f"- {row.name} ({row.type}, {counts.get(row.name, 0)} entries) — {row.description}"

    def _archived_line(self, rows: list[MemoryRow]) -> str:
        """One compact line naming every archived collection and log, by name
        (#2288). An archived
        collection with no job (no mechanism row) appears nowhere else in the
        header, so this line is the complete index of archived names — even one
        that ALSO renders (as ``archived <when>``) in the mechanisms section
        above, since this line's job is to be the whole list, not the
        remainder. Empty string — no line at all — when nothing is archived."""
        names = sorted(row.name for row in rows if row.archived)
        if not names:
            return ""
        rendered = ", ".join(f"`{name}`" for name in names)
        return f"{self.ARCHIVED_LABEL} {rendered}"

    # ── Durable user-fact core ───────────────────────────────────────────────

    def _durable_core_section(self) -> str:
        """The small, stable set of deterministic user facts — name, timezone,
        location — so the personal tone survives without a lookup. These change
        when the fact changes, never per message."""
        lines = [self.DURABLE_HEADER]
        info = self.db.users.get_info(self.user) if self.user else None
        if info is None:
            lines.append(self.NO_PROFILE)
        else:
            lines.append(f"- name: {info.name}")
            lines.append(f"- timezone: {info.timezone}")
            lines.append(f"- location: {info.location}")
        return "\n".join(lines)


# ── The shared standing-rules render (#2633) ──────────────────────────────────


def standing_rules_entries(db: Database) -> list[MemoryEntry]:
    """Every entry in the ``rules`` collection — empty when the collection does not
    exist (``db.memory`` returns ``None`` on a name nothing has created)."""
    collection = db.memory(PennyConstants.MEMORY_RULES_COLLECTION)
    return collection.read_all() if collection is not None else []


def render_standing_rules(db: Database) -> str:
    """The standing rules' BODY: every entry in the ``rules`` collection in full,
    oldest first, each as its own sentence — ``- <content>`` — and nothing else
    (#2647).  A rule reads as an instruction, never as a callable name: the entry's
    key is a snake_case name, and a bare snake_case name followed by an instruction
    reads exactly like the tool list the prompt carries two sections down, which is
    how one prod turn came to call a rule's key as a tool.  A rule is still one
    guess-free call to read or change: the section names the collection it renders,
    and ``collection_keys``/``collection_list`` on ``rules`` read the keys and the
    ``added`` stamps the render leaves out.

    With none — or no collection row at all — empty string (#2686): the section is
    omitted wherever it appears, and the ``rules`` collection stays reachable through
    the store map, which always lists it with its description."""
    entries = standing_rules_entries(db)
    if not entries:
        return ""
    return "\n".join(f"- {entry.content}" for entry in entries)


def standing_rules_section(db: Database) -> str:
    """The whole ``### Standing rules`` section — the ONE render the model-facing
    surfaces that read the rules as a DATA section share (#2633): every collector
    cycle's system prompt and the notify composer's document.  Written once so they
    can never disagree about what a rule says.  The chat prompt renders the rules in
    its instruction part instead (``standing_rules_instruction``), where she reads
    instructions rather than state.  With no rule set the whole section — heading
    included — is omitted (#2686)."""
    body = render_standing_rules(db)
    if not body:
        return ""
    return f"{SelfStateHeader.STANDING_RULES_HEADER}\n{body}"


# The one line that frames the rules where the chat agent reads INSTRUCTIONS.  Stated
# in plain words, no imperative: the rules govern the whole message, its last
# characters included, and a rule wins over anything else the prompt says — how she
# looks things up and what she does, as well as how she writes — named as the general
# case, never by any one rule's words (#2647).
STANDING_RULES_FRAMING = (
    "The user's standing rules: follow each one in everything you send, to the very "
    "end of the message. Where a rule asks for something different from any other "
    "guidance in this prompt — how you look things up, what you do, or how you "
    "write — the rule is what they want."
)


def standing_rules_instruction(db: Database) -> str | None:
    """The standing rules presented where the chat agent reads instructions — the
    framing line above the shared rules render, as a ``### Standing rules`` section
    of the chat prompt between her identity and her instructions.

    Carries the SAME section title as the collector/composer data section, so a rule
    is sourced the same wherever it renders (the eval harness's provenance split
    reads the title, not the surface).  ``None`` when there are no rules — the
    section is omitted entirely (#2686), and the ``rules`` collection stays named in
    the store map, which always lists it with its description; an instruction section
    with nothing to follow would only blur the one rule that later lands."""
    if not standing_rules_entries(db):
        return None
    return (
        f"{SelfStateHeader.STANDING_RULES_HEADER}\n"
        f"{STANDING_RULES_FRAMING}\n"
        f"\n{render_standing_rules(db)}"
    )
