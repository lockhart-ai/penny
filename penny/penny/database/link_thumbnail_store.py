"""Link thumbnail store — link-card image addresses recorded at browse time."""

from __future__ import annotations

import logging

from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlmodel import Session, col, select

from penny.clock import Clock
from penny.database.models import LinkThumbnail
from penny.tools.link_handles import normalise_link_address

logger = logging.getLogger(__name__)

# The table is a bounded cache of where a link card's picture lives, not an
# archive: past this many rows the oldest are dropped on insert.  A front-page
# read records up to ~60 entries, so this is weeks of heavy browsing.
LINK_THUMBNAIL_CAP = 5_000


class LinkThumbnailEntry(BaseModel):
    """One link card's thumbnail as recorded from a browse read — the link's
    address, its card image's address, and the card's label."""

    url: str
    image_url: str
    title: str


class LinkThumbnailStore:
    """Records and reads the ``link_thumbnail`` table: each link card's image
    address, keyed by the link's URL normalised the way the link-handle table
    normalises a link, so a lookup matches what a handle resolves to.

    Recording is best-effort at the channel seam and never downloads anything;
    the egress capture step (``MessageChannel._capture_cited_page_image``)
    looks a cited URL up here first and downloads the card image only then.
    """

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()

    def _session(self) -> Session:
        return Session(self.engine)

    def record(self, entries: list[LinkThumbnailEntry]) -> None:
        """Upsert each entry, keyed by its normalised URL, and cap the table.

        A link seen again replaces its row in place — the newest card image
        wins.  One statement, one commit; the cap keeps the newest
        ``LINK_THUMBNAIL_CAP`` rows by ``seen_at``.
        """
        if not entries:
            return
        now = self._clock.now()
        with self._session() as session:
            for entry in entries:
                row = LinkThumbnail(
                    url=normalise_link_address(entry.url),
                    image_url=entry.image_url,
                    title=entry.title,
                    seen_at=now,
                )
                session.merge(row)
            session.commit()
            self._cap(session)
        logger.debug("Recorded %d link thumbnails", len(entries))

    def lookup(self, url: str) -> LinkThumbnail | None:
        """The card recorded for ``url`` — normalised the same way on both
        sides — or ``None`` when no read ever captured one for it."""
        with self._session() as session:
            return session.get(LinkThumbnail, normalise_link_address(url))

    def _cap(self, session: Session) -> None:
        """Drop rows past the cap, oldest ``seen_at`` first (URL breaks ties)."""
        doomed = session.exec(
            select(LinkThumbnail.url)
            .order_by(col(LinkThumbnail.seen_at).desc(), col(LinkThumbnail.url))
            .offset(LINK_THUMBNAIL_CAP)
        ).all()
        if not doomed:
            return
        session.exec(sa_delete(LinkThumbnail).where(col(LinkThumbnail.url).in_(doomed)))
        session.commit()
        logger.info("Link thumbnail table capped: dropped %d oldest rows", len(doomed))
