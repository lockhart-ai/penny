"""Media store — images captured while browsing, delivered side-channel at egress."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import NamedTuple
from urllib.parse import urlparse

from pydantic import BaseModel
from similarity.embeddings import find_similar
from sqlmodel import Session, select

from penny.clock import Clock
from penny.constants import PennyConstants
from penny.database.models import Media
from penny.image_utils import normalize_image
from penny.llm.embeddings import deserialize_embedding

logger = logging.getLogger(__name__)

# Trailing prose punctuation stripped from a URL on either side of an egress
# match — egress extraction (``channels.base``) strips the same set off every
# URL a message cites, so both sides of a match agree however it was written
# (#3064).  Defined here, the lower-level module, and imported by egress.
URL_TRAILING_PUNCTUATION = ".,;:!?()<>]}>'\""


def normalize_url(url: str) -> str:
    """Lower-case and strip trailing punctuation so a URL written into a message
    matches the captured ``source_url`` of the same page.  Public — a link-card
    image is stored keyed by this same normal form (#3113), so storage and the
    exact-URL tier agree on one identity for a URL without either side needing
    to guess the other's casing or trailing punctuation."""
    return url.rstrip(URL_TRAILING_PUNCTUATION).lower()


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


class ImageSelectionPolicy(BaseModel):
    """Automatic image categories allowed by a delivery channel."""

    automatic: bool = True
    cited_page: bool = True
    same_site: bool = True
    related: bool = True

    def allows(self, row: _Candidate, urls: list[str]) -> bool:
        """Classify by the strongest relationship so blocked rows cannot fall through."""
        if normalize_url(row.source_url) in {normalize_url(url) for url in urls}:
            return self.cited_page
        if _domain(row.source_url) in ({_domain(url) for url in urls} - {""}):
            return self.same_site
        return self.related


class MediaStore:
    """Store browsed images and retrieve the most relevant match to an egress
    message — the cited page's own image when the message links one, else a
    jittered embedding-nearest pick.

    A reply whose run *generated* an image bypasses this matching entirely — it
    is delivered by id via ``send_response(media_ids=...)`` — but generated rows
    are stored with an embedding, so they join this pool for future replies."""

    def __init__(self, engine, clock: Clock | None = None):
        self.engine = engine
        self._clock = clock if clock is not None else Clock()

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
        """Insert an image blob with its metadata and return its assigned id.

        Normalized to the stored-image size rules (#2780) before the row is
        built, so every stored image — a browse capture, a generated image —
        is already within bounds at both storage and egress (egress reads
        this same row, never re-encodes)."""
        data, mime_type = normalize_image(data, mime_type)
        with self._session() as session:
            row = Media(
                mime_type=mime_type,
                data=data,
                source_url=source_url,
                title=title,
                embedding=embedding,
                created_at=self._clock.now(),
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

    def select_image(
        self,
        urls: list[str],
        embedding: list[float] | None,
        *,
        policy: ImageSelectionPolicy | None = None,
    ) -> Media | None:
        """Pick the image to attach to an egress message, most-relevant first.

        ``urls`` are the links the message itself contains (Penny cites her
        source), ``embedding`` is the message text's vector.  Two tiers:

        1. **Exact URL** — the message links a page we hold an image for:
           attach that page's own image (newest capture).  Deterministic — it
           *is* the right image.  A capture can come from actually browsing the
           page, or from a link card's own image data stored at read time for a
           link she never opened (``BrowserChannel._store_link_card_media``,
           #3113) — either way this tier fires for pages she cites without
           having fetched anything at send time.
        2. **No captured source linked** — embedding-nearest over everything,
           but a uniform random pick among the top-K so a centroid "magnet"
           image can't repeat on consecutive messages (jitter applies *only*
           to this fallback).

        A same-domain image is no longer a tier of its own (#3064): it competes
        in the fallback like any other image.  The optional channel policy still
        excludes disabled categories before matching, so a policy with
        ``same_site`` off cannot be fallen through to.  Returns None when
        automatic images are off or no eligible image qualifies.
        """
        policy = policy or ImageSelectionPolicy()
        if not policy.automatic:
            return None
        rows = [row for row in self._candidates() if policy.allows(row, urls)]
        chosen = self._cited_page_image(rows, urls) or self._jittered_nearest(rows, embedding)
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

    def has_capture(self, url: str) -> bool:
        """Whether some media row already holds an image for ``url`` (normalised
        the same way the exact-URL tier matches) — the dedup check a link-card
        capture makes before storing, so a re-read of the same link never writes
        a second row (#3113)."""
        target = normalize_url(url)
        return any(normalize_url(row.source_url) == target for row in self._candidates())

    def _cited_page_image(self, rows: list[_Candidate], urls: list[str]) -> int | None:
        """Tier 1: the newest image captured from a page the message links."""
        linked = {normalize_url(url) for url in urls}
        matches = [row for row in rows if normalize_url(row.source_url) in linked]
        if not matches:
            return None
        best = max(matches, key=lambda row: row.created_at)
        logger.debug("Matched media %d by exact cited URL", best.id)
        return best.id

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
