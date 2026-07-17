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
- **Recent activity** — background runs, configuration mutations, and autonomous
  sends, interleaved in one time-ordered block at rollup altitude (one line each;
  one run's per-call detail is one ``get_event(run <id>)`` hop away). A run line
  that wrote carries a ``· wrote '<key>' → `<collection>``` clause naming what it
  changed (from the #1560 ``last_written_by_run_id`` stamp, #1641), so a just-
  written fact is 0 calls away and its full entry is one ``collection_read_latest``
  hop. The *complement* of the
  conversation — chat turns (direct replies) are already in context, so they're
  never duplicated here; only mechanism-authored sends (``mechanism`` non-NULL,
  #1568) appear.
- **Your memory** — the map of stores (collections + logs): names + one-line
  scope. The index for an anchored lookup, never the content.
- **Skills and rules** — the pinned firing channel for taught skills (#1471).
  Rendered deterministically (all of them, never a relevance guess) so a taught
  behavior fires *ambiently* — the skill is in the prompt, so firing costs
  **0 calls**; its full recipe is one ``skill_read(<name>)`` hop away. One feed:
  the taught-skill registry (``db.skills``, #1590) — the sole skills store. (The
  legacy ``skills`` collection's standing-rules feed retired with the collection,
  #1624/migration 0092.)
- **About the user** — the durable user-fact core (name, timezone, location):
  deterministic facts, not a relevance guess, so personality survives without a
  lookup.
- A pointers line naming the fetch tools for anything deeper.

Autonomous-send (emission) rows join the recent-activity block (#1568): each
delivered mechanism send renders one ``sent · <when> · <mechanism> — "<snippet>"``
line, interleaved by time with the runs and mutations, so Penny sees her own
background emissions ambiently; older sends carry the same provenance inline on
every ``penny-messages`` read, and ``memory_metadata(<mechanism>)`` is the hop to
the creating request (#1566).
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from penny.constants import PennyConstants
from penny.database.memory.types import render_key_value
from penny.database.mutation_store import mutation_change_summary
from penny.datetime_utils import format_log_timestamp
from penny.tools.collection_instantiation import render_trigger_clause

if TYPE_CHECKING:
    from datetime import datetime

    from penny.database.database import Database
    from penny.database.memory.types import RunWrite
    from penny.database.message_store import EmissionActivity, RunActivity, RunOutcomeStamp
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
    SKILLS_HEADER = "### Skills and rules"
    DURABLE_HEADER = "### About the user"

    EMPTY_MECHANISMS = "(no mechanisms yet)"
    EMPTY_ACTIVITY = "(no recent activity)"
    EMPTY_MAP = "(no stores yet)"
    EMPTY_SKILLS = (
        "(no skills yet — when a task needs one, ask the user to walk you through it "
        "once and you'll learn it automatically)"
    )
    NO_PROFILE = "(no profile set yet)"

    # The taught-skill feed's group label names its OWN guess-free drill-down
    # (``skill_read(<name>)`` — the rendered name IS the argument), so a rendered
    # name is never a failed guess.  ``skill_read`` is named here, not in
    # POINTERS, so the section stays surgical (POINTERS is shared, #1580
    # territory).
    TAUGHT_SKILLS_LABEL = "Skills you've been taught — skill_read(<name>) for the full recipe:"

    # The overflow tail each bounded section shows when it has more rows than its
    # cap — the fetch tool named so the remainder is one guess-free call away
    # (nothing is silently dropped; n≤1 holds).
    MORE_MECHANISMS = "+{count} more — collection_catalog()"
    MORE_ACTIVITY = "+ older activity — read_run_calls(<target>)"
    MORE_MAP = "+{count} more — collection_catalog()"

    POINTERS = (
        "To look deeper: memory_metadata(<name>) for a collection's full config and "
        "change history, get_event(run <id>) for one run's tool calls, "
        "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) "
        "for stored entries, find(query=<text>) to find anything of yours by meaning "
        "(a collection, a skill, or a stored entry), and collection_catalog() for "
        "every collection."
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
            self._skills_section(),
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
        """The mechanism's trigger clause rendered AS the copyable ``trigger`` input
        (#1631, display form == invocation form): ``on advance of <log>`` · ``once at
        <ISO> [xN]`` · ``every <seconds>``.  Empty when the collection has no trigger yet
        (an adopted skill awaiting a cadence), so the mechanism line drops the clause."""
        has_trigger = (
            row.source_log is not None
            or row.run_at is not None
            or row.collector_interval_seconds is not None
        )
        return render_trigger_clause(row) if has_trigger else ""

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
        events.sort(key=lambda event: event[0], reverse=True)
        return events

    @staticmethod
    def _emission_line(emission: EmissionActivity) -> str:
        """``sent · <when> · <mechanism> — "<snippet>"`` — a delivered autonomous
        send (#1568). The mechanism is the ``memory_metadata`` anchor (its #1566
        block names the creating request); the snippet is what Penny autonomously
        said, at rollup altitude."""
        return (
            f"sent · {format_log_timestamp(emission.sent_at)} · "
            f'{emission.mechanism} — "{emission.snippet}"'
        )

    @staticmethod
    def _run_line(run: RunActivity, writes: list[RunWrite]) -> str:
        """``run <run_id> · <when> · <target> → <OUTCOME> (<n> calls)``, then — when
        the run wrote — a ``· wrote …`` clause naming what it changed (#1641).

        The run id is the typed anchor, consumed verbatim by ``get_event(run
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
        one) is a ``get_event(run <id>)`` anchor, single-sourced with that tool's
        parse (``RUN_EVENT_PREFIX``); the entity name resolves via
        ``memory_metadata``/``find``, and the detail carries the changed
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

    # ── Skills and rules ─────────────────────────────────────────────────────

    def _skills_section(self) -> str:
        """The pinned firing channel for taught skills (#1471).

        Renders ALL of the taught-skill registry (no relevance gating, no budget
        cap — wholesale; trimming is a later tuning knob) so a taught behavior
        fires ambiently: it is *in the prompt*, so firing costs 0 calls. The feed
        is a labeled group naming its own guess-free drill-down; the section
        collapses to one honest placeholder when nothing has been taught. (The
        legacy standing-rules feed retired with the ``skills`` collection,
        #1624.)"""
        taught = self._taught_skill_lines()
        lines = [self.SKILLS_HEADER, *taught]
        if not taught:
            lines.append(self.EMPTY_SKILLS)
        return "\n".join(lines)

    def _taught_skill_lines(self) -> list[str]:
        """``- <name> — <intent>`` per taught skill (the ``skill`` registry,
        #1590, name order), under a label naming ``skill_read(<name>)`` as the
        drill-down — the rendered name IS that call's argument (n≤1). Empty when
        nothing has been taught yet (a fresh install ships the table empty)."""
        skills = self.db.skills.list_all()
        if not skills:
            return []
        lines = [self.TAUGHT_SKILLS_LABEL]
        lines.extend(f"- {skill.name} — {self._one_line(skill.intent)}" for skill in skills)
        return lines

    @staticmethod
    def _one_line(text: str) -> str:
        """Collapse whitespace/newlines to a single line so a taught skill's
        intent (a user utterance, possibly multi-line) stays one bullet."""
        return " ".join(text.split())

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
