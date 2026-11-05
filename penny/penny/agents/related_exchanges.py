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
visible, so the boundary is the same grow/cut function the history and the
conversation map share (``Agent._history_window_size`` over the uncapped
``count_user_turns`` anchor).  Exchanges shown as map lines are included — the
map carries only the user's own words, so her half of the exchange is nowhere.
No retrieval runs for a system-notification trigger: the section is keyed to
the user's own words, and a delivery turn's trigger is the framework's.

Both runtime bounds are config (``RELATED_EXCHANGES_MAX``,
``RELATED_EXCHANGES_MIN_SIMILARITY``).  No matches, no section: the quiet
default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.constants import PennyConstants
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
    renders one exchange whole, both sides clipped, ids and day anchor
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
        live_embedding: list[float],
    ) -> None:
        self.db = db
        self.user = user
        self.runtime = runtime
        self.live_embedding = live_embedding

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

        Computed by the SAME grow/cut function the history and the map are
        built from (``Agent._history_window_size``, over the uncapped
        ``count_user_turns`` anchor), so the retrieval can never show an
        exchange the window already carries."""
        window = int(self.runtime.CHAT_HISTORY_TURNS)
        if window <= 0:
            return None
        total_user_turns = self.db.messages.count_user_turns(self.user)
        size = Agent._history_window_size(total_user_turns, window)
        if size <= 0:
            return None
        return self.db.messages.user_turn_at_offset(self.user, size - 1)

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
        own day anchor, byte-for-byte its form), then both sides in quotes,
        each clipped to ``SIDE_CHARS``."""
        ids = ", ".join(str(user_id) for user_id in match.user_ids)
        day = stored_as_utc(match.user_at).astimezone(self._zone()).date().isoformat()
        head = (
            f"{self.BLOCK_HEAD}{self._stamp(match.user_at)} "
            f"(messages {ids}, reply {match.reply_id}) — "
            f"{PennyConstants.USER_MESSAGES_DAY_ANCHOR.format(day=day)}"
        )
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
