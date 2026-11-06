"""ConversationMap — the deterministic map of the conversation older than the
history window (#2752, reworked #2982).

The history window is every exchange since the current session began — cut
back to ``CHAT_HISTORY_TURNS`` at the session's first turn and again at the
``CHAT_SESSION_MAX_TURNS`` cap (#3077; previously grew to twice that minus one
before cutting back, #2756/#2757) — so anything older than its own
FIRST exchange is out of sight unless she searches her logs.  Full-text recall
of her own replies failed before on three counts — size (about 6k tokens
re-read every turn), fixation (showing her own past prose makes her copy it,
§3.6), and stale-request confusion — so this map carries NONE of her reply
prose: one line per exchange, built by the framework from the user's own
words and the run records (§2.3), never the model.  Her replies appear only
as the actions the records show they DID — pages read, keyed writes, registry
changes — never as sentences, and the ``named:`` heading extraction that
belonged to her reply's own markup is gone (#2982): a heading names a topic,
never an outcome, and it was her prose riding the map.  What replaced it is
the user's own words, whole up to ``USER_CLIP_CHARS`` — a correction renders
to its point instead of stopping short of it, which is the decision the map
exists to carry forward.

One line, per exchange older than the window::

    - Oct 20 19:38 · "which bands did the Talking Heads members form after…" →
      read 3 pages · saved 'remain-in-light' → albums_to_hear

- The date and time are absolute, on the user's own clock.
- The user's message is its first line, whole to ``USER_CLIP_CHARS``; a longer
  one keeps its head and its tail around one " … " — conclusions often come
  last.
- The outcome is the exchange's recorded run: pages read (browse-results entries
  stamped with the run), keyed writes (the ledger stamp), registry mutations (the
  mutation ledger).  Omitted when the turn made no recorded action — never her
  claimed actions, only records (a false claim can't ride into the map).

One line, per AUTONOMOUS send that has left the window (#2982) — a job's
report, a sandbox preview, the startup notice::

    - Nov 04 09:12 · job report: daily_headlines
      (run 55d0aa1b2c3d4e5f6a7b8c9d0e1f2a3b) → you replied

- Detection is STRUCTURAL: the outgoing ``messagelog`` row's non-null
  ``mechanism`` — the same mark #2974/#2979 chose for autonomy in history
  (``HistoryTurn.mechanism``) — never the message text, never a
  mechanism-name list.  The kind reads off the delivery run's own
  ``promptlog.prompt_type`` stamp (``ChatPromptType``); an unknown or
  run-less row renders the generic ``notification:`` label, so a future
  delivery mechanism degrades visibly rather than vanishing.
- ``(run <id>)`` is the verbatim anchor ``run_get(run: 'run <id>')`` consumes
  (#2974's self-state convention); the delivered message is one
  ``log_read{'penny-messages', day: …}`` away — the day the line's own stamp
  names.  The ``→ you replied`` fragment states the one record-derived fact
  the row carries: the send happened.

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
date names, on the log the closing line points at.  An autonomous line's own
anchor is ``(run <id>)``, read by ``run_get`` verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.constants import ChatPromptType, PennyConstants
from penny.database.message_store import SessionAnchor
from penny.datetime_utils import stored_as_utc, user_timezone_name, zone_or_utc

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
    from penny.database.database import Database
    from penny.database.models import MessageLog

# ── The line's named constants ────────────────────────────────────────────────

# The user's message renders whole up to this many characters; a longer one
# keeps its head and its tail around one " … " (§2.8) — conclusions often
# come last.
USER_CLIP_CHARS = 280
# A "short exchange" — no recorded outcome, and both sides under this many
# characters — is collapse fodder; the rule reads only lengths and records, so
# it is deterministic.
SHORT_CHARS = 80
# The bounded read the map's exchanges come from: the horizon's messages, capped.
# Past it the horizon's OLDEST rows drop — the same once-a-day front loss the
# date horizon itself has, never a mid-map churn.  A day folded past the cap
# counts its true size off the store (`_day_item_count`), never off this read
# (#3056).
MESSAGE_POOL = 200
# At most this many recorded outcomes (reads + writes + changes) render on one
# line, with a "…" tail naming that more exist.
MAX_OUTCOME_PARTS = 3
# At most this many written keys render per collection, with a "…" tail.
MAX_WRITE_KEYS = 3

# The separator between a clipped message's head and its tail, counted inside
# ``USER_CLIP_CHARS``.
_CLIP_SEPARATOR = " … "

# The render label per delivery-run ``prompt_type`` — a framework stamp, never
# the message text or a mechanism-name list.  An unknown prompt_type (a future
# delivery mechanism) renders ``_NOTIFICATION_FALLBACK`` with the row's own
# ``mechanism`` instead, so nothing vanishes silently.
_NOTIFICATION_LABELS: dict[str, str] = {
    ChatPromptType.JOB_REPORT: "job report",
    ChatPromptType.SANDBOX_REPORT: "sandbox report",
    ChatPromptType.CHANGELOG: "startup notice",
}
_NOTIFICATION_FALLBACK = "notification"
# The startup notice renders WITHOUT its mechanism: the changelog row's
# ``mechanism`` is the changelog memory row's own name, internal naming, never
# a job name the line could anchor.
_CHANGELOG_LABEL = _NOTIFICATION_LABELS[ChatPromptType.CHANGELOG]
# The one fragment every autonomous line ends with — the record-derived fact
# the row itself carries: the send happened (#2982).
_SENT_FRAGMENT = "you replied"

# A run id renders as the verbatim ``run_get`` anchor (#2974's convention).
_RUN_ANCHOR = "(run {run_id})"


def _user_zone(db: Database) -> tzinfo:
    """The user's timezone, UTC when none is set — the map's date arithmetic and
    the shared horizon cutoff both read it."""
    return zone_or_utc(user_timezone_name(db))


def utc_day_bounds(zone: tzinfo, date_iso: str) -> tuple[datetime, datetime]:
    """One local date's bounds ``[start, end)`` in stored (naive UTC) time —
    the window a per-day count reads over."""
    local_date = date.fromisoformat(date_iso)
    start_local = datetime.combine(local_date, datetime.min.time(), tzinfo=zone)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(UTC).replace(tzinfo=None),
        end_local.astimezone(UTC).replace(tzinfo=None),
    )


def map_horizon_start(db: Database, runtime: RuntimeParams) -> datetime:
    """Local midnight of the conversation map's horizon's first day — the date
    cutoff the map's own lines read (#2752) and, since #3079, the recent-activity
    block reads too: an autonomous send inside the horizon is already covered by
    the history window or the map, so only a send OLDER than it still renders an
    emission line there.  One function for both readers, on the same
    ``CHAT_MAP_DAYS`` the map reads, so the two can never disagree about the
    boundary.  Returned naive, because that is how every timestamp is stored."""
    zone = _user_zone(db)
    today = db.clock.now().astimezone(zone).date()
    days = max(int(runtime.CHAT_MAP_DAYS), 1)
    return utc_day_bounds(zone, (today - timedelta(days=days - 1)).isoformat())[0]


@dataclass(frozen=True)
class _Exchange:
    """One user turn with the reply it drew — the map's unit of grouping.

    ``user_at`` is the turn's own instant (the line's time anchor); ``run_ids``
    are the reply rows' run ids, the join to every recorded outcome; ``reply``
    is the reply text whole, read only for the short-exchange length, never
    rendered."""

    user: str
    reply: str
    user_at: datetime
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Notification:
    """One autonomous send that has left the history window (#2982) — the
    map's other line kind.

    Built from the outgoing row's own stamps (``mechanism``, ``run_id``,
    timestamp) plus the delivery run's ``prompt_type`` — every input a record
    fixed at the send's creation, so the line is as byte-stable as any
    exchange's.  ``label`` is the line's whole head, built once here."""

    at: datetime
    run_id: str | None
    label: str


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
    SHORT_LINE_TAIL = "short messages"
    DAY_LINE_TAIL = "exchanges"

    def __init__(self, db: Database, user: str | None, runtime: RuntimeParams) -> None:
        self.db = db
        self.user = user
        self.runtime = runtime

    # ── Summary ──────────────────────────────────────────────────────────────

    def render(self) -> str:
        """The whole block — header, one line per exchange and per autonomous
        send older than the window (collapses applied), and the closing anchor.
        Empty string when nothing in the horizon is older than the window: a
        section with nothing to show renders nothing."""
        items = self._older_than_window(self._in_horizon())
        if not items:
            return ""
        lines = self._collapse_short(items)
        lines = self._collapse_at_ceiling(lines)
        return "\n".join([self.HEADER, *(line.text for line in lines), self._anchor()])

    # ── The exchanges ────────────────────────────────────────────────────────

    def _in_horizon(self) -> list[_Exchange | _Notification]:
        """The horizon's items, oldest first: the direct messages since the
        horizon start, grouped as ``_build_conversation`` groups them —
        consecutive incoming rows are one user turn, the outgoing rows that
        follow are its reply — and every mechanism-stamped outgoing row
        (#2982) is its own ``_Notification``, flushing any open exchange
        first.  Leading reply rows with no user turn before them and no
        mechanism behind them drop."""
        rows = self.db.messages.recent_direct_messages(
            since=self._horizon_start(), limit=MESSAGE_POOL
        )
        items: list[_Exchange | _Notification] = []
        user_rows: list[MessageLog] = []
        reply_rows: list[MessageLog] = []
        for row in rows:
            if row.direction == PennyConstants.MessageDirection.OUTGOING and (
                row.mechanism is not None
            ):
                if user_rows:
                    items.append(self._exchange(user_rows, reply_rows))
                    reply_rows = []
                    user_rows = []
                items.append(self._notification(row))
                continue
            if row.direction == PennyConstants.MessageDirection.INCOMING:
                if user_rows:
                    items.append(self._exchange(user_rows, reply_rows))
                    reply_rows = []
                    user_rows = []
                user_rows.append(row)
            elif user_rows:
                reply_rows.append(row)
        if user_rows:
            items.append(self._exchange(user_rows, reply_rows))
        return items

    def _exchange(self, user_rows: list[MessageLog], reply_rows: list[MessageLog]) -> _Exchange:
        """One exchange from its rows: merged user text, merged reply, the turn's
        own instant, and the reply rows' run ids."""
        return _Exchange(
            user="\n".join(row.content for row in user_rows),
            reply="\n".join(row.content for row in reply_rows),
            user_at=user_rows[0].timestamp,
            run_ids=tuple(row.run_id for row in reply_rows if row.run_id is not None),
        )

    def _notification(self, row: MessageLog) -> _Notification:
        """One autonomous send from its row: the row's own stamps plus the
        delivery run's ``prompt_type`` label, rendered with the mechanism
        behind it — except the startup notice, whose ``mechanism``
        (``changelog``) is internal naming, never a job name.  A run-less or
        unknown-kind row renders the fallback label with the row's own
        ``mechanism`` — visible, never silent (visible degradation over
        silent success)."""
        label = self._notification_label(row.run_id)
        mechanism = row.mechanism or ""
        head = label if label == _CHANGELOG_LABEL or not mechanism else f"{label}: {mechanism}"
        return _Notification(
            at=row.timestamp,
            run_id=row.run_id,
            label=head,
        )

    def _notification_label(self, run_id: str | None) -> str:
        """The line's kind label, read off the delivery run's own
        ``promptlog.prompt_type`` stamp — a framework stamp, never the message
        text.  Falls back to the generic label when the run is unknown."""
        if run_id is None:
            return _NOTIFICATION_FALLBACK
        prompts = self.db.messages.prompts_for_run(run_id)
        prompt_type = prompts[0].prompt_type if prompts else None
        return _NOTIFICATION_LABELS.get(prompt_type or "", _NOTIFICATION_FALLBACK)

    def _older_than_window(
        self, items: list[_Exchange | _Notification]
    ) -> list[_Exchange | _Notification]:
        """Everything in the horizon before the history window's own first
        exchange — computed by the SAME function the history itself is built
        from (``Agent._history_window_size``), anchored on the SAME session
        anchor (``MessageStore.session_anchor``, #3077) the history's own
        size is computed from.  One shared function, so the map and the
        history can never disagree about where the window starts — the map
        stops exactly one exchange before the history's own first: the open
        live turn's exchange (its reply has not followed it) occupies none of
        the window's slots, here or in the history's own trim (#3077's
        boundary fix).

        The slice runs to the window's first USER exchange (#2982): a
        notification rides the map exactly when it has left the history —
        notifications sit between user turns in the send order, so the
        boundary is that exchange's position, not a count of items.

        The window keeps every exchange since the current session began and
        cuts back only at the session's first turn or at the
        ``CHAT_SESSION_MAX_TURNS`` cap (#3077) — so its FIRST exchange stays
        fixed between cuts and moves only at one.  That is the only turn this
        boundary moves: between cuts, the map changes only by the one item
        that ages out at its end (#2757)."""
        window = int(self.runtime.CHAT_HISTORY_TURNS)
        if window <= 0 or not items:
            return []
        if self.user:
            anchor = self.db.messages.session_anchor(
                self.user, int(self.runtime.CHAT_SESSION_IDLE_SECONDS)
            )
        else:
            count = sum(1 for item in items if isinstance(item, _Exchange))
            anchor = SessionAnchor(
                total_user_turns=count, session_turns=count, live_turn_open=False
            )
        if anchor.live_turn_open and items and isinstance(items[-1], _Exchange):
            # The open live turn's exchange (its reply has not followed it —
            # that is the mark the anchor read) is the live message's own, not
            # a past one: it must not occupy one of the window's slots here
            # either, or the map's boundary lands one exchange older than the
            # history's own first and the two disagree by one (#3077).
            items = items[:-1]
        size = Agent._history_window_size(
            anchor.session_turns,
            anchor.total_user_turns,
            window,
            int(self.runtime.CHAT_SESSION_MAX_TURNS),
        )
        if size <= 0:
            return []
        user_positions = [index for index, item in enumerate(items) if isinstance(item, _Exchange)]
        if len(user_positions) <= size:
            return []
        return items[: user_positions[-size]]

    def _horizon_start(self) -> datetime:
        """The shared horizon cutoff — :func:`map_horizon_start`, the one
        function the recent-activity block reads too (#3079)."""
        return map_horizon_start(self.db, self.runtime)

    def _utc_day_bounds(self, date_iso: str) -> tuple[datetime, datetime]:
        """One local date's bounds ``[start, end)`` in stored (naive UTC)
        time — the window a per-day count reads over."""
        return utc_day_bounds(self._zone(), date_iso)

    def _zone(self) -> tzinfo:
        return _user_zone(self.db)

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

    def _item_line(self, item: _Exchange | _Notification) -> ConversationMap._Line:
        """One line for one horizon item, of whichever kind it is."""
        if isinstance(item, _Notification):
            return self._notification_line(item)
        return self._exchange_line(item)

    def _exchange_line(self, exchange: _Exchange) -> ConversationMap._Line:
        """One line for one exchange: its absolute time, the user's first line
        whole to ``USER_CLIP_CHARS``, then the recorded outcome — omitted when
        the exchange has none.  Every input is a record fixed at the
        exchange's creation, so the line's text is fixed at creation."""
        outcome = self._outcome(exchange)
        tail = f" → {outcome}" if outcome else ""
        date, date_iso = self._local_date(exchange.user_at)
        return self._Line(
            text=f'- {self._stamp(exchange.user_at)} · "{self._clip(exchange.user)}"{tail}',
            date=date,
            date_iso=date_iso,
            exchanges=1,
        )

    def _notification_line(self, note: _Notification) -> ConversationMap._Line:
        """One line for one autonomous send (#2982): its absolute time, the
        head built at construction (kind label with the mechanism behind it),
        the run anchor, and the record-derived send fragment."""
        anchor = f" {_RUN_ANCHOR.format(run_id=note.run_id)}" if note.run_id else ""
        date, date_iso = self._local_date(note.at)
        return self._Line(
            text=f"- {self._stamp(note.at)} · {note.label}{anchor} → {_SENT_FRAGMENT}",
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
        """The user's message, whole to ``USER_CLIP_CHARS``; a longer first
        line keeps its head and its tail around one " … " — conclusions often
        come last (#2982)."""
        first_line = text.split("\n", 1)[0]
        if len(first_line) <= USER_CLIP_CHARS:
            return first_line
        budget = USER_CLIP_CHARS - len(_CLIP_SEPARATOR)
        head_len = (budget + 1) // 2
        tail_len = budget - head_len
        return f"{first_line[:head_len]}{_CLIP_SEPARATOR}{first_line[-tail_len:]}"

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

    # ── The collapses ────────────────────────────────────────────────────────

    def _is_short(self, exchange: _Exchange) -> bool:
        """No recorded outcome and both sides short — the deterministic
        collapse rule.  It reads only the records its exchange is built from,
        so an exchange's collapsibility is as fixed as its line."""
        return (
            not self._outcome(exchange)
            and len(exchange.user) <= SHORT_CHARS
            and len(exchange.reply) <= SHORT_CHARS
        )

    def _collapse_short(
        self, items: list[_Exchange | _Notification]
    ) -> list[ConversationMap._Line]:
        """Consecutive short exchanges collapse into one line, ranged
        ``19:30–19:34 · 3 short messages``.  A group never spans a local date
        (the range would be ambiguous), and a group only ever extends at its
        tail — an exchange crossing into the map joins the trailing group if it
        qualifies, changing nothing before it.  A notification is never
        collapse fodder: its line is the record the map exists to carry
        (#2982)."""
        lines: list[ConversationMap._Line] = []
        group: list[_Exchange] = []
        for item in items:
            if isinstance(item, _Notification):
                self._flush_short_group(group, lines)
                group = []
                lines.append(self._item_line(item))
                continue
            joins_group = self._is_short(item) and (
                not group
                or self._local_date(group[-1].user_at)[1] == self._local_date(item.user_at)[1]
            )
            if joins_group:
                group.append(item)
                continue
            self._flush_short_group(group, lines)
            group = [item] if self._is_short(item) else []
            if not group:
                lines.append(self._item_line(item))
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
        repeated while the map is over the ceiling.  Each pass folds the OLDEST
        day that still has more than one line, never the newest day, so every
        pass strictly reduces the line count; when no such day remains the fold
        stops, even with the map still over the ceiling.  An already-folded day
        is never folded again — that pass would change nothing, so the loop
        would never end (#3014).  Never line-by-line: trimming the front one
        line per turn would break the cached prefix every turn, while a
        whole-day line is fixed once rendered.

        The fold line's count is the day's TRUE item count (``_day_item_count``)
        — a pure function of the day (#3056), never of whatever part of it the
        bounded ``MESSAGE_POOL`` read still holds, which shrinks turn by turn
        on a day that outgrew the pool."""
        ceiling = int(self.runtime.CHAT_MAP_MAX_LINES)
        lines = list(lines)
        while len(lines) > ceiling:
            newest_date = lines[-1].date
            counts: dict[str, int] = {}
            for line in lines:
                if line.date != newest_date:
                    counts[line.date] = counts.get(line.date, 0) + 1
            # ``lines`` is oldest-first, so the dict's insertion order is too.
            oldest_date = next((date for date, count in counts.items() if count > 1), None)
            if oldest_date is None:
                break
            oldest_iso = next(line.date_iso for line in lines if line.date == oldest_date)
            covered = self._day_item_count(oldest_iso)
            fold = self._Line(
                text=(
                    f"- {oldest_date} · {covered} {self.DAY_LINE_TAIL} — "
                    f"{PennyConstants.USER_MESSAGES_DAY_ANCHOR.format(day=oldest_iso)}"
                ),
                date=oldest_date,
                date_iso=oldest_iso,
                exchanges=covered,
            )
            first = next(i for i, line in enumerate(lines) if line.date == oldest_date)
            rest = [line for line in lines if line.date != oldest_date]
            lines = [*rest[:first], fold, *rest[first:]]
        return lines

    def _day_item_count(self, date_iso: str) -> int:
        """The day's true item count — ``db.messages.count_map_items`` over the
        whole local day, not a sum off the bounded read (#3056): a day that
        outgrew ``MESSAGE_POOL`` loses its OLDEST rows from the read as new
        messages arrive, so a count read off the lines would shrink turn by
        turn and bust the cached prefix at the fold line.  The store's own
        definition is the row-level half of ``_in_horizon``'s grouping."""
        since, until = self._utc_day_bounds(date_iso)
        return self.db.messages.count_map_items(since=since, until=until)

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
