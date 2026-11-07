"""RelatedExchanges — the per-turn ``## Related earlier exchanges`` block (#3010).

The earlier exchanges most related to the live user message, rendered IN FULL in
the per-turn dynamic system message, so a plan agreed with the user stays
visible after it has left the history window and the conversation map.  Measured
on a replayed prod turn of exactly this shape, showing the top 3 related
exchanges whole moved citing-the-plan from 0/12 samples to 11/12 — a first-sentence
preview left the plan's own words out of view and never surfaced them.

The retrieval is deterministic, never a model judgment: each exchange is ONE
text (the user message + Penny's reply to it), embedded at write time with the
same model as every other message vector and stored on the reply's
``messagelog`` row (``exchange_embedding``); the live user message is embedded
once per turn and scored by cosine similarity against every exchange older than
the history window.  The exchange — not the user message alone — is the unit,
because the user's later message often refers to the plan by a pronoun ("do
those") while the plan's words sit in her reply.

What is excluded and why: exchanges INSIDE the history window are already
visible, so the boundary is the same function the history and the conversation
map share (``Agent._history_window_size`` over the shared ``session_anchor``,
#3077).  Exchanges shown as map lines are still selected — the map carries only
the user's own words, never Penny's reply, which is where an agreed plan
lives — but a match whose user words the map ALREADY renders whole, on a line
of their own, drops its own ``User:`` side here (#3085).  Two conditions, both
required: (1) ``_block`` is handed the map's own ``ConversationMap.
one_to_one_user_ids()`` set (``ChatAgent`` builds it once per turn, off the
SAME ``ConversationMap`` instance the static prefix renders from — never a
second map, never a re-derived horizon or window) and checks every one of the
match's user-message ids is in it — the STRUCTURAL half: is this exchange a
lone line at all, never a ranged short-collapse line, a folded day, or a
notification line, none of which carries any one exchange's words; (2) that
lone line must show the text WHOLE — ``ConversationMap.clips_whole(match.
user)`` calls the map's own clip rule directly, so a text over
``USER_CLIP_CHARS`` or carrying more than one line, which the map's line would
cut to a head+tail excerpt or to its first line alone, still renders both
sides here, because the map's version of it is not the same text.  Still
within the history window, or failing either check, renders both sides,
exactly as before.  No retrieval runs for a system-notification trigger: the
section is keyed to the user's own words, and a delivery turn's trigger is the
framework's.

Both runtime bounds are config (``RELATED_EXCHANGES_MAX``,
``RELATED_EXCHANGES_MIN_SIMILARITY``).  No matches, no section: the quiet
default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.conversation_map import ConversationMap
from penny.constants import PennyConstants
from penny.database.message_store import SessionAnchor
from penny.datetime_utils import stored_as_utc, user_timezone_name, zone_or_utc
from penny.llm.embeddings import cosine_similarity, deserialize_embedding

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
    from penny.database import Database
    from penny.database.models import MessageLog


@dataclass(frozen=True)
class _Match:
    """One selected exchange: its merged sides, the turn's own instant, and the
    message ids the block renders — the ids are the anchor for reading the
    whole exchange or its day back."""

    user_at: datetime
    user_ids: tuple[int, ...]
    reply_id: int
    user: str
    reply: str


class RelatedExchanges:
    """Retrieval + render for one chat turn's related earlier exchanges.

    Table of contents, in file order: the render (``render``) composes the
    boundary, the scored matches and the blocks · the boundary
    (``_boundary``) is the history window's own oldest kept user turn, read
    off the same window math the map and the history share · the matches
    (``_select``/``_match_from``) score every older exchange vector and take
    the top few at or above the similarity floor · the block (``_block``)
    renders one exchange's reply, and its user side too unless the
    conversation map already carries it whole (#3085), ids and day anchor
    included."""

    HEADER = "## Related earlier exchanges"
    BLOCK_HEAD = "Related earlier exchange · "
    # The issue's render spec (#3010): each side in full up to 600 characters,
    # a longer side cut to its first 600 ending in an ellipsis — the form the
    # measured arm showed.
    SIDE_CHARS = 600
    ELLIPSIS = "…"
    # Local-time stamp format, the conversation map's own form — an absolute
    # instant on the user's clock, so the block's time never drifts with the
    # turn it is rebuilt in.
    STAMP_FORMAT = "%b %d %H:%M"

    def __init__(
        self,
        db: Database,
        user: str,
        runtime: RuntimeParams,
        anchor: SessionAnchor,
        live_embedding: list[float],
        map_one_to_one_user_ids: frozenset[int],
    ) -> None:
        self.db = db
        self.user = user
        self.runtime = runtime
        # The caller's own single ``session_anchor`` read (#3125), REQUIRED
        # — this class never re-derives its own boundary, so the boundary
        # here and the history window it shares ``Agent._history_window_size``
        # with can never disagree.
        self.anchor = anchor
        self.live_embedding = live_embedding
        # The conversation map's own one-to-one line ids (#3085,
        # ``ConversationMap.one_to_one_user_ids``) — required, never derived
        # here: this class never re-reads the map's horizon or window, it only
        # asks whether a match's user ids are already in this set.
        self.map_one_to_one_user_ids = map_one_to_one_user_ids

    def render(self) -> str:
        """The whole block — header, one block per selected exchange, oldest
        first.  Empty string when nothing scores at or above the floor: a
        section with nothing to show renders nothing."""
        boundary = self._boundary()
        if boundary is None:
            return ""
        matches = self._select(boundary)
        if not matches:
            return ""
        blocks = [self._block(match) for match in matches]
        return "\n\n".join([self.HEADER, *blocks])

    # ── The boundary ─────────────────────────────────────────────────────────

    def _boundary(self) -> MessageLog | None:
        """The history window's own oldest kept user turn — the exchange rows
        before it in send order are exactly the ones the window has dropped.

        Computed by the SAME function the history and the map are built from
        (``Agent._history_window_size``, over ``self.anchor`` (#3125), the
        caller's own single ``session_anchor`` read passed in at
        construction), so the retrieval can never show an exchange the
        window already carries — the boundary is exactly the history's own
        oldest kept exchange, and when the live turn's row is open in the
        store the read skips it (#3077's boundary fix)."""
        window = int(self.runtime.CHAT_HISTORY_TURNS)
        if window <= 0:
            return None
        anchor = self.anchor
        size = Agent._history_window_size(
            anchor.session_turns,
            anchor.total_user_turns,
            window,
            int(self.runtime.CHAT_SESSION_MAX_TURNS),
        )
        if size <= 0:
            return None
        # The anchor counts CLOSED exchanges (#3077): when the live turn's row
        # is open in the store, it is the newest run-less incoming row and the
        # boundary read must skip it — one exchange further back — so the
        # scored pool is exactly what the window has dropped, never an
        # exchange the history still carries.
        offset = size - 1 + (1 if anchor.live_turn_open else 0)
        return self.db.messages.user_turn_at_offset(self.user, offset)

    # ── The matches ──────────────────────────────────────────────────────────

    def _select(self, boundary: MessageLog) -> list[_Match]:
        """Every exchange vector older than the window, scored by cosine
        against the live message's own vector; the top ``RELATED_EXCHANGES_MAX``
        at or above ``RELATED_EXCHANGES_MIN_SIMILARITY``, rendered oldest first.

        The full older-exchange pool is scanned — a few thousand vectors is
        cheap, and a cap below "all exchanges older than the window" would
        silently drop the match that mattered."""
        floor = float(self.runtime.RELATED_EXCHANGES_MIN_SIMILARITY)
        scored: list[tuple[float, MessageLog]] = []
        for row in self.db.messages.exchange_vectors_before(boundary):
            vector = row.exchange_embedding
            if vector is None:
                continue
            similarity = cosine_similarity(self.live_embedding, deserialize_embedding(vector))
            if similarity >= floor:
                scored.append((similarity, row))
        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp, pair[1].id or 0), reverse=True)
        top = scored[: int(self.runtime.RELATED_EXCHANGES_MAX)]
        matches = [self._match_from(row) for _, row in top]
        matches.sort(key=lambda match: match.user_at)
        return matches

    def _match_from(self, row: MessageLog) -> _Match:
        """One match from its reply row: both sides reconstructed off the run
        stamps (``MessageStore.exchange_sides_for`` — the same read the
        write-time and backfill embeddings cover), so what renders is exactly
        what was embedded."""
        sides = self.db.messages.exchange_sides_for(row)
        return _Match(
            user_at=sides.user_rows[0].timestamp if sides.user_rows else row.timestamp,
            user_ids=tuple(row_id for row in sides.user_rows if (row_id := row.id) is not None),
            reply_id=row.id or 0,
            user="\n".join(side.content for side in sides.user_rows),
            reply="\n".join(side.content for side in sides.reply_rows),
        )

    # ── The block ────────────────────────────────────────────────────────────

    def _block(self, match: _Match) -> str:
        """One exchange's block: the head (stamp, ids, the conversation map's
        own day anchor, byte-for-byte its form), then its side(s) in quotes,
        each clipped to ``SIDE_CHARS``.

        The ``User:`` side drops only when BOTH hold (#3085): every one of
        the match's user-message ids is already on one of the map's own
        one-to-one lines (structural — this exchange is a lone line at all,
        never a ranged short-collapse line, a folded day, or a notification
        line), AND the map's own clip of the text is the text itself
        (``ConversationMap.clips_whole`` — content: that lone line shows it
        WHOLE, never a head+tail excerpt or a first-line-only cut).  Either
        check failing renders both sides, exactly as before — nothing here is
        a duplicate of anything the map actually shows."""
        ids = ", ".join(str(user_id) for user_id in match.user_ids)
        day = stored_as_utc(match.user_at).astimezone(self._zone()).date().isoformat()
        head = (
            f"{self.BLOCK_HEAD}{self._stamp(match.user_at)} "
            f"(messages {ids}, reply {match.reply_id}) — "
            f"{PennyConstants.USER_MESSAGES_DAY_ANCHOR.format(day=day)}"
        )
        if (
            match.user_ids
            and all(user_id in self.map_one_to_one_user_ids for user_id in match.user_ids)
            and ConversationMap.clips_whole(match.user)
        ):
            return f'{head}\nPenny: "{self._clip(match.reply)}"'
        return f'{head}\nUser: "{self._clip(match.user)}"\nPenny: "{self._clip(match.reply)}"'

    def _clip(self, text: str) -> str:
        """Whole to ``SIDE_CHARS``; a longer side is its first ``SIDE_CHARS``
        characters ending in an ellipsis (#3010) — the form the measured arm
        showed, whose plan's words sat inside the first 600 characters."""
        if len(text) <= self.SIDE_CHARS:
            return text
        return f"{text[: self.SIDE_CHARS]}{self.ELLIPSIS}"

    def _stamp(self, at: datetime) -> str:
        """The block's time anchor — absolute, on the user's own clock, the
        conversation map's own form."""
        return stored_as_utc(at).astimezone(self._zone()).strftime(self.STAMP_FORMAT)

    def _zone(self) -> tzinfo:
        return zone_or_utc(user_timezone_name(self.db))
