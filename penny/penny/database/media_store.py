"""Media store — images captured while browsing, delivered side-channel at egress."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

from similarity.embeddings import find_similar
from sqlmodel import Session, select

from penny.database.models import Media
from penny.llm.embeddings import deserialize_embedding

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Lower-case and strip trailing punctuation so a URL written into a message
    matches the captured ``source_url`` of the same page."""
    return url.rstrip(".,);:'\"<>").lower()


def _domain(url: str) -> str:
    """The registrable host of a URL (``www.`` stripped), or '' if unparseable."""
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


class _Candidate(NamedTuple):
    """A media row reduced to the columns egress matching needs — never the image
    ``data`` blob, so selection doesn't load the whole (multi-GB) media table."""

    id: int
    source_url: str
    created_at: datetime
    vector: list[float] | None


class MediaStore:
    """Store browsed images and retrieve the cited-page match to an egress message.

    Selection is deterministic: a browsed image is attached only when the outgoing
    message links its source page (exact URL, then same domain).  A reply that
    cites no source gets no image — there is no random embedding-nearest fallback,
    so ordinary messages never carry an unrelated image.  (Generated images are
    delivered by id via ``send_response(media_ids=...)``, not through this path.)"""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def put(
        self,
        data: bytes,
        mime_type: str,
        source_url: str | None = None,
        title: str | None = None,
        embedding: bytes | None = None,
    ) -> int:
        """Insert an image blob with its metadata and return its assigned id."""
        with self._session() as session:
            row = Media(
                mime_type=mime_type,
                data=data,
                source_url=source_url,
                title=title,
                embedding=embedding,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("media row was inserted but has no id")
            logger.debug("Stored %d bytes as media %d (%s)", len(data), row.id, mime_type)
            return row.id

    def get(self, media_id: int) -> Media | None:
        with self._session() as session:
            return session.get(Media, media_id)

    def select_image(self, urls: list[str], embedding: list[float] | None) -> Media | None:
        """Pick the cited-page image to attach to an egress message, if any.

        ``urls`` are the links the message itself contains (Penny cites her
        source), ``embedding`` is the message text's vector.  Two deterministic
        tiers, both keyed on a cited source:

        1. **Exact URL** — the message links a page we captured an image from:
           attach that page's own image (newest capture).  It *is* the right image.
        2. **Same domain** — the message links a site we have images from but not
           that exact page: the embedding-nearest image from that domain.

        Returns None when the message cites no source we have an image for — an
        ordinary reply is left imageless rather than carrying a random, unrelated
        picture (there is no embedding-nearest fallback).
        """
        rows = self._candidates()
        chosen = self._cited_page_image(rows, urls) or self._cited_domain_image(
            rows, urls, embedding
        )
        return self.get(chosen) if chosen is not None else None

    def _candidates(self) -> list[_Candidate]:
        """Every media row as a lightweight candidate (no ``data`` blob)."""
        with self._session() as session:
            rows = session.exec(
                select(Media.id, Media.source_url, Media.created_at, Media.embedding)
            ).all()
        return [
            _Candidate(
                id=row[0],
                source_url=row[1] or "",
                created_at=row[2],
                vector=deserialize_embedding(row[3]) if row[3] else None,
            )
            for row in rows
            if row[0] is not None
        ]

    def _cited_page_image(self, rows: list[_Candidate], urls: list[str]) -> int | None:
        """Tier 1: the newest image captured from a page the message links."""
        linked = {_normalize_url(url) for url in urls}
        matches = [row for row in rows if _normalize_url(row.source_url) in linked]
        if not matches:
            return None
        best = max(matches, key=lambda row: row.created_at)
        logger.debug("Matched media %d by exact cited URL", best.id)
        return best.id

    def _cited_domain_image(
        self, rows: list[_Candidate], urls: list[str], embedding: list[float] | None
    ) -> int | None:
        """Tier 2: embedding-nearest image from a domain the message links."""
        if embedding is None:
            return None
        domains = {_domain(url) for url in urls} - {""}
        scoped = [row for row in rows if row.vector and _domain(row.source_url) in domains]
        best_id = self._nearest_id(embedding, scoped, top_k=1)
        if best_id is not None:
            logger.debug("Matched media %d by cited domain", best_id)
        return best_id

    def _scored(self, embedding: list[float], rows: list[_Candidate]) -> list[tuple[int, float]]:
        """(id, cosine) for ``rows``, nearest first, no floor."""
        return find_similar(
            embedding,
            [(row.id, row.vector) for row in rows if row.vector is not None],
            top_k=len(rows) or 1,
            threshold=-1.0,
        )

    def _nearest_id(self, embedding: list[float], rows: list[_Candidate], top_k: int) -> int | None:
        scored = self._scored(embedding, rows)
        return scored[0][0] if scored else None
