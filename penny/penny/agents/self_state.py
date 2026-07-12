"""SelfStateHeader — Penny's operational self-state, rendered deterministically
into the chat agent's system prompt (#1555).

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

- **Active mechanisms** — the collectors, archived-inclusive: name, status,
  cadence, end condition, last-run outcome. "what's running right now?"
- **Recent activity** — background runs and configuration mutations, interleaved
  in one time-ordered block at rollup altitude (one line each; per-call detail is
  one ``read_run_calls`` hop away). The *complement* of the conversation — chat
  turns are already in context, so they're never duplicated here.
- **Your memory** — the map of stores (collections + logs): names + one-line
  scope. The index for an anchored lookup, never the content.
- **About the user** — the durable user-fact core (name, timezone, location):
  deterministic facts, not a relevance guess, so personality survives without a
  lookup.
- A pointers line naming the fetch tools for anything deeper.

Emission/autonomous-send rows are deliberately absent: #1568 adds them to this
same block and the same whole-render test literal, so this renderer is structured
so that extension is additive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.constants import PennyConstants
from penny.database.mutation_store import mutation_change_summary
from penny.datetime_utils import format_interval, format_log_timestamp

if TYPE_CHECKING:
    from datetime import datetime

    from penny.database.database import Database
    from penny.database.message_store import RunActivity, RunOutcomeStamp
    from penny.database.models import MemoryRow, MutationEvent


class SelfStateHeader:
    """Renders the ``## Penny's current state`` block from the registry + ledger.

    Constructed per chat turn (the activity block churns on every event, so it
    lives in the dynamic tail of the prompt, never a cached static prefix) and
    read purely — every method is a deterministic projection of stored rows.
    """

    HEADER = "## Penny's current state"
    MECHANISMS_HEADER = "### Active mechanisms"
    ACTIVITY_HEADER = "### Recent activity"
    MAP_HEADER = "### Your memory"
    DURABLE_HEADER = "### About the user"

    EMPTY_MECHANISMS = "(no mechanisms yet)"
    EMPTY_ACTIVITY = "(no recent activity)"
    EMPTY_MAP = "(no stores yet)"
    NO_PROFILE = "(no profile set yet)"

    # The overflow tail each bounded section shows when it has more rows than its
    # cap — the fetch tool named so the remainder is one guess-free call away
    # (nothing is silently dropped; n≤1 holds).
    MORE_MECHANISMS = "+{count} more — collection_catalog()"
    MORE_ACTIVITY = "+ older activity — read_run_calls(<target>)"
    MORE_MAP = "+{count} more — collection_catalog()"

    POINTERS = (
        "To look deeper: memory_metadata(<name>) for a collection's full config and "
        "change history, read_run_calls(<target>) for a run's tool calls, "
        "collection_read_latest(<name>) or read_similar(query=<text>) for stored "
        "entries, and collection_catalog() for every collection."
    )

    def __init__(self, db: Database, user: str | None) -> None:
        self.db = db
        self.user = user

    # ── Summary ──────────────────────────────────────────────────────────────

    def render(self) -> str:
        """The whole ``## Penny's current state`` block — a table of contents of
        Penny's operational situation, each section a deterministic render."""
        sections = [
            self.HEADER,
            self._mechanisms_section(),
            self._activity_section(),
            self._memory_map_section(),
            self._durable_core_section(),
            self.POINTERS,
        ]
        return "\n\n".join(sections)

    # ── Active mechanisms ────────────────────────────────────────────────────

    def _mechanisms_section(self) -> str:
        """The collectors, recency-first and archived-inclusive, each on one line:
        status · cadence · end condition · last-run outcome (#1566 registry)."""
        rows = self._mechanism_rows()
        outcomes = self.db.messages.latest_run_outcomes()
        shown = rows[: PennyConstants.SELF_STATE_MECHANISMS_LIMIT]
        lines = [self.MECHANISMS_HEADER]
        lines.extend(self._mechanism_line(row, outcomes.get(row.name)) for row in shown)
        if not shown:
            lines.append(self.EMPTY_MECHANISMS)
        overflow = len(rows) - len(shown)
        if overflow > 0:
            lines.append(self.MORE_MECHANISMS.format(count=overflow))
        return "\n".join(lines)

    def _mechanism_rows(self) -> list[MemoryRow]:
        """Every collector (a ``memory`` row with an ``extraction_prompt``), active
        and archived, sorted by name.

        Name order is deterministic regardless of seed timing (the inventory
        convention), so the whole-render assertion is stable; the overflow tail
        therefore drops the alphabetical tail rather than the stalest — a
        recency-weighted overflow is a deferred refinement (the cap is generous,
        so overflow is rare in practice)."""
        rows = [row for row in self.db.memories.list_all() if row.extraction_prompt is not None]
        rows.sort(key=lambda row: row.name)
        return rows

    def _mechanism_line(self, row: MemoryRow, stamp: RunOutcomeStamp | None) -> str:
        """``- <name> — <status> · <cadence> · <end> · <last run>`` — a retired
        mechanism drops the cadence/end clauses (it isn't running)."""
        if row.archived:
            parts = [f"archived {format_log_timestamp(row.updated_at)}", self._last_run(stamp)]
        else:
            parts = ["active", self._cadence(row), self._end_condition(row), self._last_run(stamp)]
        clauses = " · ".join(part for part in parts if part)
        return f"- {row.name} — {clauses}"

    @staticmethod
    def _cadence(row: MemoryRow) -> str:
        """``every <interval>`` from the collector's current cadence, or empty."""
        if row.collector_interval_seconds is None:
            return ""
        return f"every {format_interval(row.collector_interval_seconds)}"

    @staticmethod
    def _end_condition(row: MemoryRow) -> str:
        """The mechanism's end condition — an expiry time, a one-shot / run quota,
        or empty when it runs indefinitely."""
        if row.expires_at is not None:
            return f"expires {format_log_timestamp(row.expires_at)}"
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
        """Background runs and config mutations, interleaved newest-first at rollup
        altitude (one line each). Chat turns are excluded by construction — they're
        already the conversation; this is its complement (ledger, #1560)."""
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
        """(timestamp, rendered line) for recent runs + mutations, newest first.

        Each source is fetched at ``cap + 1`` (so the section can tell that a
        ``cap + 1``-th event exists and show the overflow tail), merged, and
        sorted by time — so the block stays flat as activity grows and the newest
        events of either kind win a slot."""
        fetch = PennyConstants.SELF_STATE_ACTIVITY_LIMIT + 1
        events: list[tuple[datetime, str]] = [
            (run.finished_at, self._run_line(run))
            for run in self.db.messages.recent_collector_runs(fetch)
        ]
        events.extend(
            (event.created_at, self._mutation_line(event))
            for event in self.db.mutations.recent(fetch)
        )
        events.sort(key=lambda event: event[0], reverse=True)
        return events

    @staticmethod
    def _run_line(run: RunActivity) -> str:
        """``run <run_id> · <when> · <target> → <OUTCOME> (<n> calls)`` — the run
        id is the typed anchor; ``read_run_calls(<target>)`` is the drill-down."""
        plural = "" if run.call_count == 1 else "s"
        return (
            f"run {run.run_id} · {format_log_timestamp(run.finished_at)} · "
            f"{run.target} → {run.outcome.upper()} ({run.call_count} call{plural})"
        )

    @staticmethod
    def _mutation_line(event: MutationEvent) -> str:
        """``change · <when> · <entity> <action> by <actor> (run <id>) — <detail>``
        — the run id (present for a user-run mutation, absent for a system one) is
        the anchor; the detail carries the cause or the changed fields."""
        parts = [
            f"change · {format_log_timestamp(event.created_at)} · "
            f"{event.entity_name} {event.action} by {event.actor}"
        ]
        if event.run_id is not None:
            parts.append(f"(run {event.run_id})")
        line = " ".join(parts)
        summary = mutation_change_summary(event)
        return f"{line} — {summary}" if summary else line

    # ── The map of stores ────────────────────────────────────────────────────

    def _memory_map_section(self) -> str:
        """The index of readable stores — every non-archived memory (collection or
        log) by name, shape, entry count, and one-line scope. Names, never
        content: any user fact is one anchored read away using a name shown here."""
        memories = [row for row in self.db.memories.list_all() if not row.archived]
        counts = self.db.memories.entry_counts()
        shown = memories[: PennyConstants.SELF_STATE_MAP_LIMIT]
        lines = [self.MAP_HEADER]
        lines.extend(
            f"- {row.name} ({row.type}, {counts.get(row.name, 0)} entries) — {row.description}"
            for row in shown
        )
        if not shown:
            lines.append(self.EMPTY_MAP)
        overflow = len(memories) - len(shown)
        if overflow > 0:
            lines.append(self.MORE_MAP.format(count=overflow))
        return "\n".join(lines)

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
