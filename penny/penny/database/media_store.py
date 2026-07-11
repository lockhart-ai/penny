"""Media store — images captured while browsing, delivered side-channel at egress."""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

from similarity.embeddings import find_similar
from sqlmodel import Session, select

from penny.constants import PennyConstants
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
    """Store browsed images and retrieve the most relevant match to an egress
    message — the cited page's own image when the message links one, else a
    jittered embedding-nearest pick.

    A reply whose run *generated* an image bypasses this matching entirely — it
    is delivered by id via ``send_response(media_ids=...)`` — but generated rows
    are stored with an embedding, so they join this pool for future replies."""

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
        """Pick the image to attach to an egress message, most-relevant first.

        ``urls`` are the links the message itself contains (Penny cites her
        source), ``embedding`` is the message text's vector.  Three tiers:

        1. **Exact URL** — the message links a page we captured an image from:
           attach that page's own image (newest capture).  Deterministic — it
           *is* the right image.
        2. **Same domain** — the message links a site we have images from but not
           that exact page: the embedding-nearest image from that domain.
           Deterministic.
        3. **No source linked** — embedding-nearest over everything, but a uniform
           random pick among the top-K so a centroid "magnet" image can't repeat
           on consecutive messages (jitter applies *only* to this fallback).

        Returns None only when nothing qualifies (no URL match and no embedded
        media), so a reply still carries an image whenever one can be matched.
        """
        rows = self._candidates()
        chosen = (
            self._cited_page_image(rows, urls)
            or self._cited_domain_image(rows, urls, embedding)
            or self._jittered_nearest(rows, embedding)
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

    def _jittered_nearest(
        self, rows: list[_Candidate], embedding: list[float] | None
    ) -> int | None:
        """Tier 3: uniform random among the top-K embedding-nearest images."""
        if embedding is None:
            return None
        scored = self._scored(embedding, [row for row in rows if row.vector])
        if not scored:
            return None
        pool = [media_id for media_id, _ in scored[: PennyConstants.MEDIA_MATCH_JITTER_TOPK]]
        chosen = random.choice(pool)
        logger.debug("Matched media %d by jittered embedding (pool of %d)", chosen, len(pool))
        return chosen

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
