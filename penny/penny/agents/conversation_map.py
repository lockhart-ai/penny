"""ConversationMap — the deterministic map of the conversation older than the
history window (#2752).

The history window is ``CHAT_HISTORY_TURNS`` exchanges, growing to twice that
minus one before cutting back (#2756/#2757) — so anything older than its own
FIRST exchange is out of sight unless she searches her logs.  Full-text recall
of her own replies failed before on three counts — size (about 6k tokens
re-read every turn), fixation (showing her own past prose makes her copy it,
§3.6), and stale-request confusion — so this map carries NONE of it: one line
per exchange, built by the framework from the user's own words and the run
records (§2.3), never the model; her replies appear only as the items they
MARKED (bold spans and link texts), never as sentences; and it renders at the
END of the LEADING static system message (#2757), after the identity and
instructions and before the tool schemas — never in the per-turn state message,
and never near the user's own turn.

One line, per exchange older than the window::

    - Oct 20 19:38 · "which bands did the Talking Heads members form after…" →
      read 3 pages · saved 'remain-in-light' → albums_to_hear · named: Tom Tom Club

- The date and time are absolute, on the user's own clock.
- The user's message is its first line, clipped at ``CLIP_CHARS``.
- The outcome is the exchange's recorded run: pages read (browse-results entries
  stamped with the run), keyed writes (the ledger stamp), registry mutations (the
  mutation ledger).  Omitted when the turn made no recorded action — never her
  claimed actions, only records (a false claim can't ride into the map).
- ``named:`` — the items her stored reply marked, parsed deterministically, at
  most ``MAX_NAMES`` per line.  Stored replies keep real URLs (#2729), so a link
  contributes its TEXT and never its address.

Append-only (§3 cache): a line's text depends only on records fixed at that
exchange's creation — its message rows, its closed run's stamps — so the map is
a byte-stable rebuild every turn.  Because the history window's own FIRST
exchange stays fixed through its whole grow phase and moves only at a cut turn
(#2757), the map changes ONLY on a cut turn too — the same shared
``Agent._history_window_size`` boundary decides both — and on that turn it
grows by exactly the exchanges the history dropped, at its end, except that a
trailing short-exchange collapse group may absorb the newcomer and change that
one last line.  Every OTHER turn the map is byte-identical, so the reused
static-system prefix (identity + instructions + map + tool schemas) survives
whole.  The two boundings both remove from the FRONT whole: the date horizon
(``CHAT_MAP_DAYS``, by local DATE, so a line leaves at most once a day) and the
line ceiling (``CHAT_MAP_MAX_LINES``, collapsing the OLDEST WHOLE DAY into one
line, never line-by-line).  The closing anchor is fixed text.  So the map
before an exchange ages out is a byte-identical prefix of the map after it,
through every line but possibly the last content line — the cached prefix runs
through the rest.

Reachability: each line carries its absolute time, so the full text of any
exchange is one anchored read away — ``log_read`` with the ``day`` the line's
date names, on the log the closing line points at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.datetime_utils import stored_as_utc, user_timezone_name, zone_or_utc

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
    from penny.database.database import Database
    from penny.database.models import MessageLog

# ── The line's named constants ────────────────────────────────────────────────

# The user's message is clipped to this many characters, with "…" when cut (§2.8);
# each rendered name is clipped to the same length.
CLIP_CHARS = 60
# At most this many of her marked items render on one line, deduplicated.
MAX_NAMES = 5
# A "short exchange" — no recorded outcome, and both sides under this many
# characters — is collapse fodder; the rule reads only lengths and records, so
# it is deterministic.
SHORT_CHARS = 80
# The bounded read the map's exchanges come from: the horizon's messages, capped.
# Past it the horizon's OLDEST rows drop — the same once-a-day front loss the
# date horizon itself has, never a mid-map churn.
MESSAGE_POOL = 200
# At most this many recorded outcomes (reads + writes + changes) render on one
# line, with a "…" tail naming that more exist.
MAX_OUTCOME_PARTS = 3
# At most this many written keys render per collection, with a "…" tail.
MAX_WRITE_KEYS = 3


# Her stored reply's markdown, parsed deterministically: bold spans, and the
# TEXT of markdown links (a stored reply keeps real URLs since #2729, so a
# link's label is the name and its address is discarded).  A link whose text is
# its own URL contributes nothing, and an image's alt text is not a name.
_BOLD_SPAN_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_LINK_TEXT_RE = re.compile(r"\[([^\]\n]+)\]\(https?://[^)\s]+\)")


@dataclass(frozen=True)
class _Exchange:
    """One user turn with the reply it drew — the map's unit of grouping.

    ``user_at`` is the turn's own instant (the line's time anchor); ``run_ids``
    are the reply rows' run ids, the join to every recorded outcome; ``reply``
    is the reply text whole, read only for the names it marks and the
    short-exchange length, never rendered."""

    user: str
    reply: str
    user_at: datetime
    run_ids: tuple[str, ...]


class ConversationMap:
    """Renders the ``## Conversation map`` block from the conversation log and
    the run records — a pure read, constructed per chat turn beside the rest of
    the per-turn system message (#2745)."""

    HEADER = "## Conversation map"
    # The closing line: the anchored reads that reach any line's full text.  The
    # ``day`` argument is the map's own horizon read — chat's no-argument
    # ``log_read`` reaches only its recent window, so the anchor names the form
    # that answers at the horizon's age.
    ANCHOR_PREFIX = "Full text: "
    CLIP_SUFFIX = "…"
    NAMES_PREFIX = "named: "
    SHORT_LINE_TAIL = "short messages"
    DAY_LINE_TAIL = "exchanges"

    def __init__(self, db: Database, user: str | None, runtime: RuntimeParams) -> None:
        self.db = db
        self.user = user
        self.runtime = runtime

    # ── Summary ──────────────────────────────────────────────────────────────

    def render(self) -> str:
        """The whole block — header, one line per exchange older than the
        window (collapses applied), and the closing anchor.  Empty string when
        nothing in the horizon is older than the window: a section with nothing
        to show renders nothing."""
        exchanges = self._older_than_window(self._in_horizon())
        if not exchanges:
            return ""
        lines = self._collapse_short(exchanges)
        lines = self._collapse_at_ceiling(lines)
        return "\n".join([self.HEADER, *(line.text for line in lines), self._anchor()])

    # ── The exchanges ────────────────────────────────────────────────────────

    def _in_horizon(self) -> list[_Exchange]:
        """The horizon's exchanges, oldest first: the direct messages since the
        horizon start, grouped as ``_build_conversation`` groups them —
        consecutive incoming rows are one user turn, the outgoing rows that
        follow are its reply.  Leading reply rows with no user turn before them
        belong to an exchange whose ask is outside the horizon, so they drop."""
        rows = self.db.messages.recent_direct_messages(
            since=self._horizon_start(), limit=MESSAGE_POOL
        )
        exchanges: list[_Exchange] = []
        user_rows: list[MessageLog] = []
        reply_rows: list[MessageLog] = []
        for row in rows:
            if row.direction == PennyConstants.MessageDirection.INCOMING:
                if user_rows:
                    exchanges.append(self._exchange(user_rows, reply_rows))
                    reply_rows = []
                    user_rows = []
                user_rows.append(row)
            elif user_rows:
                reply_rows.append(row)
        if user_rows:
            exchanges.append(self._exchange(user_rows, reply_rows))
        return exchanges

    def _exchange(self, user_rows: list[MessageLog], reply_rows: list[MessageLog]) -> _Exchange:
        """One exchange from its rows: merged user text, merged reply, the turn's
        own instant, and the reply rows' run ids."""
        return _Exchange(
            user="\n".join(row.content for row in user_rows),
            reply="\n".join(row.content for row in reply_rows),
            user_at=user_rows[0].timestamp,
            run_ids=tuple(row.run_id for row in reply_rows if row.run_id is not None),
        )

    def _older_than_window(self, exchanges: list[_Exchange]) -> list[_Exchange]:
        """Everything in the horizon before the history window's own first
        exchange — computed by the SAME grow/cut function the history itself
        is built from (``Agent._history_window_size``), anchored on the SAME
        uncapped turn count (``MessageStore.count_user_turns``, #2756) the
        history's own grow/cut cycle reads.  One shared function, so the map
        and the history can never disagree about where the window starts.

        The window grows from ``CHAT_HISTORY_TURNS`` to twice that minus one
        before cutting back to the window — so its FIRST exchange stays fixed
        through the whole growth phase and only moves at a cut turn.  That is
        the only turn this boundary moves: between cuts, the map changes only
        by the one exchange that ages out at its end (#2757)."""
        window = int(self.runtime.CHAT_HISTORY_TURNS)
        if window <= 0 or not exchanges:
            return []
        total_user_turns = (
            self.db.messages.count_user_turns(self.user) if self.user else len(exchanges)
        )
        size = Agent._history_window_size(total_user_turns, window)
        if size <= 0 or len(exchanges) <= size:
            return []
        return exchanges[:-size]

    def _horizon_start(self) -> datetime:
        """Local midnight of the horizon's first day — ``CHAT_MAP_DAYS`` local
        dates including today, so a line leaves the map only when its DATE
        leaves, at most once a day.  Returned naive, because that is how every
        timestamp is stored."""
        zone = self._zone()
        today = self.db.clock.now().astimezone(zone).date()
        days = max(int(self.runtime.CHAT_MAP_DAYS), 1)
        start_local = datetime.combine(
            today - timedelta(days=days - 1), datetime.min.time(), tzinfo=zone
        )
        return start_local.astimezone(UTC).replace(tzinfo=None)

    def _zone(self) -> tzinfo:
        return zone_or_utc(user_timezone_name(self.db))

    # ── The lines ────────────────────────────────────────────────────────────

    @dataclass(frozen=True)
    class _Line:
        """One rendered line plus what the boundings need to reason about it:
        its local date (the ceiling collapses whole days), that day's ISO form
        (the collapse line's read anchor), and how many exchanges it covers."""

        text: str
        date: str
        date_iso: str
        exchanges: int

    def _exchange_line(self, exchange: _Exchange) -> ConversationMap._Line:
        """One line for one exchange: its absolute time, the user's clipped
        first line, then the recorded outcome and the marked names — each
        omitted when the exchange has none.  Every input is a record fixed at
        the exchange's creation, so the line's text is fixed at creation."""
        outcome = self._outcome(exchange)
        names = self._names(exchange.reply)
        parts = [part for part in (outcome, names) if part]
        tail = f" → {' · '.join(parts)}" if parts else ""
        date, date_iso = self._local_date(exchange.user_at)
        return self._Line(
            text=f'- {self._stamp(exchange.user_at)} · "{self._clip(exchange.user)}"{tail}',
            date=date,
            date_iso=date_iso,
            exchanges=1,
        )

    def _stamp(self, at: datetime) -> str:
        """The line's time anchor — absolute, on the user's own clock, so the
        full text is one anchored read away and the render never drifts with
        the turn it is rebuilt in."""
        return stored_as_utc(at).astimezone(self._zone()).strftime("%b %d %H:%M")

    def _local_date(self, at: datetime) -> tuple[str, str]:
        local = stored_as_utc(at).astimezone(self._zone())
        return local.strftime("%b %d"), local.date().isoformat()

    def _clip(self, text: str) -> str:
        first_line = text.split("\n", 1)[0]
        if len(first_line) > CLIP_CHARS:
            return f"{first_line[:CLIP_CHARS]}{self.CLIP_SUFFIX}"
        return first_line

    def _outcome(self, exchange: _Exchange) -> str:
        """The exchange's recorded run: pages read, keyed writes, registry
        changes — joined `` · ``, at most ``MAX_OUTCOME_PARTS`` parts with a "…"
        tail.  All three are framework-stamped records; an exchange whose runs
        recorded nothing has no outcome, and never a claimed one."""
        if not exchange.run_ids:
            return ""
        run_ids = list(exchange.run_ids)
        parts: list[str] = []
        pages = sum(self.db.memories.pages_read_by_run(run_ids).values())
        if pages:
            plural = "" if pages == 1 else "s"
            parts.append(f"read {pages} page{plural}")
        for name, keys in self._writes(run_ids):
            shown = ", ".join(f"'{key}'" for key in keys[:MAX_WRITE_KEYS])
            more = self.CLIP_SUFFIX if len(keys) > MAX_WRITE_KEYS else ""
            parts.append(f"saved {shown}{more} → {name}")
        changes = self.db.mutations.events_for_runs(run_ids)
        for events in changes.values():
            for event in events:
                parts.append(f"{event.action} {event.entity_name}")
        if len(parts) > MAX_OUTCOME_PARTS:
            return " · ".join(parts[:MAX_OUTCOME_PARTS]) + f" · {self.CLIP_SUFFIX}"
        return " · ".join(parts)

    def _writes(self, run_ids: list[str]) -> list[tuple[str, list[str]]]:
        """The runs' keyed writes grouped per collection, in the store's own
        order — the ``writes_by_run`` join the self-state header's writes clause
        renders, at the map's altitude."""
        writes = self.db.memories.writes_by_run(run_ids)
        grouped: dict[str, list[str]] = {}
        for run_id in run_ids:
            for write in writes.get(run_id, []):
                grouped.setdefault(write.memory_name, []).append(write.key)
        return list(grouped.items())

    def _names(self, reply: str) -> str:
        """``named: …`` — the items her stored reply MARKED, never her
        sentences: bold spans and link texts, deduplicated in order of
        appearance, at most ``MAX_NAMES``, each clipped.  A reply that marked
        nothing contributes no names, so none of her prose is here to copy
        (§3.6) and a false claim can't ride in."""
        found: list[str] = []
        for match in (*_BOLD_SPAN_RE.findall(reply), *_LINK_TEXT_RE.findall(reply)):
            name = match.strip()
            if name and name not in found:
                found.append(name)
            if len(found) == MAX_NAMES:
                break
        if not found:
            return ""
        named = ", ".join(self._clip(name) for name in found)
        more = self.CLIP_SUFFIX if len(found) == MAX_NAMES else ""
        return f"{self.NAMES_PREFIX}{named}{more}"

    # ── The collapses ────────────────────────────────────────────────────────

    def _is_short(self, exchange: _Exchange) -> bool:
        """No recorded outcome, no marked names, and both sides short — the
        deterministic collapse rule.  It reads only the records its exchange is
        built from, so an exchange's collapsibility is as fixed as its line."""
        return (
            not self._outcome(exchange)
            and not self._names(exchange.reply)
            and len(exchange.user) <= SHORT_CHARS
            and len(exchange.reply) <= SHORT_CHARS
        )

    def _collapse_short(self, exchanges: list[_Exchange]) -> list[ConversationMap._Line]:
        """Consecutive short exchanges collapse into one line, ranged
        ``19:30–19:34 · 3 short messages``.  A group never spans a local date
        (the range would be ambiguous), and a group only ever extends at its
        tail — an exchange crossing into the map joins the trailing group if it
        qualifies, changing nothing before it."""
        lines: list[ConversationMap._Line] = []
        group: list[_Exchange] = []
        for exchange in exchanges:
            joins_group = self._is_short(exchange) and (
                not group
                or self._local_date(group[-1].user_at)[1] == self._local_date(exchange.user_at)[1]
            )
            if joins_group:
                group.append(exchange)
                continue
            self._flush_short_group(group, lines)
            group = [exchange] if self._is_short(exchange) else []
            if not group:
                lines.append(self._exchange_line(exchange))
        self._flush_short_group(group, lines)
        return lines

    def _flush_short_group(
        self, group: list[_Exchange], lines: list[ConversationMap._Line]
    ) -> None:
        """One finished run of short exchanges: a LONE short exchange renders as
        its own ordinary line (a collapse is for consecutive shorts, plural);
        two or more collapse into the ranged line."""
        if not group:
            return
        if len(group) == 1:
            lines.append(self._exchange_line(group[0]))
            return
        first, last = group[0], group[-1]
        span = f"{self._stamp(first.user_at)}–{self._stamp(last.user_at)}"
        date, date_iso = self._local_date(first.user_at)
        lines.append(
            self._Line(
                text=f"- {span} · {len(group)} {self.SHORT_LINE_TAIL}",
                date=date,
                date_iso=date_iso,
                exchanges=len(group),
            )
        )

    def _collapse_at_ceiling(
        self, lines: list[ConversationMap._Line]
    ) -> list[ConversationMap._Line]:
        """Past ``CHAT_MAP_MAX_LINES``, the OLDEST WHOLE DAY collapses into one
        line — ``Oct 20 · 23 exchanges — log_read('user-messages', day=…)`` —
        repeated while the map is over the ceiling and more than one day remains.
        Never line-by-line: trimming the front one line per turn would break the
        cached prefix every turn, while a whole-day line is fixed once rendered."""
        ceiling = int(self.runtime.CHAT_MAP_MAX_LINES)
        lines = list(lines)
        while len(lines) > ceiling and len({line.date for line in lines}) > 1:
            oldest_date, oldest_iso = lines[0].date, lines[0].date_iso
            covered = sum(line.exchanges for line in lines if line.date == oldest_date)
            rest = [line for line in lines if line.date != oldest_date]
            lines = [
                self._Line(
                    text=(
                        f"- {oldest_date} · {covered} {self.DAY_LINE_TAIL} — "
                        f"log_read{{'user-messages', day: '{oldest_iso}'}}"
                    ),
                    date=oldest_date,
                    date_iso=oldest_iso,
                    exchanges=covered,
                ),
                *rest,
            ]
        return lines

    def _anchor(self) -> str:
        """The closing reads — FIXED text, never recomputed: it names the form
        of the call, and each line's own date stamp names the day to pass.  A
        recomputed anchor (one that named the map's newest day) would change as
        the horizon rolls and break the byte-identical-prefix property the map's
        cache stability rests on."""
        return (
            f"{self.ANCHOR_PREFIX}log_read{{'user-messages', day: 'YYYY-MM-DD'}}, "
            "log_read{'penny-messages', day: 'YYYY-MM-DD'}"
        )
