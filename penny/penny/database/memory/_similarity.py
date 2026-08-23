"""Pure scoring primitives for the memory layer — no DB, no Memory classes.

Two families of shape-independent math the ``Memory`` objects compose:

  * dedup — the three-signal collision rule used by ``Collection.write`` and
    the ``exists`` probe (key TCR, key cosine, content cosine).
  * retrieval — embedding stacking and plain cosine nearest-neighbor scoring for
    the explicit ``read_similar`` search and resolve-by-meaning.

Everything here is a free function over plain values so it stays trivially
testable and reusable from both the entity classes and the registry.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
    token_containment_ratio,
)

from penny.database.memory.types import DedupThresholds, EntrySide


def maybe_serialize(vec: list[float] | None) -> bytes | None:
    return serialize_embedding(vec) if vec is not None else None


def maybe_deserialize(blob: bytes | None) -> list[float] | None:
    return deserialize_embedding(blob) if blob is not None else None


# ── Dedup ────────────────────────────────────────────────────────────────────


class _Signal(NamedTuple):
    """One dedup signal's score against its two thresholds.

    ``is_content`` marks the signal that speaks to the VALUE rather than to the name —
    the one the write gate reads to tell a re-observation that found nothing new from a
    collision on the key alone."""

    score: float
    strict: float
    relaxed: float
    is_content: bool = False


class DuplicateMatch(NamedTuple):
    """The existing entry a candidate collided with, and WHETHER the collision was on
    the value itself.

    Returning the matched side (instead of a bool) is what lets a caller surface *which*
    existing entry blocked the write.  ``same_value`` is the second thing a caller needs
    and the dedup rule is the only place that knows it: the STRICT CONTENT signal fired,
    so the stored entry already says what this write says — the wording of either may
    differ, which is exactly why an embedding signal answers it and a text comparison
    could not.

    The distinction is the write gate's (#1919): a collision on the VALUE is a watch's
    no-news, while a collision on the KEY alone is a stored entry this write has FRESHER
    information about — "the release moved to June" arriving under a reworded key must
    never be read as nothing having happened.
    """

    side: EntrySide
    same_value: bool


def is_duplicate(
    candidate: EntrySide,
    existing: list[EntrySide],
    thresholds: DedupThresholds,
) -> DuplicateMatch | None:
    """Return the first existing entry that ``candidate`` collides with under the
    dedup rule — with whether the collision was on the value — or ``None`` if no
    match."""
    for side in existing:
        match = _pair_match(candidate, side, thresholds)
        if match is not None:
            return match
    return None


def _pair_match(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> DuplicateMatch | None:
    """Apply the three-signal dedup rule to a single candidate/existing pair, and read
    the value question off the same scores.

    Signals that can't be computed (missing keys, missing embeddings) are skipped. Fire
    if any one signal hits its strict threshold or any two signals hit their relaxed
    thresholds.
    """
    signals = _score_signals(candidate, existing, thresholds)
    if not _collides(signals):
        return None
    return DuplicateMatch(side=existing, same_value=_says_the_same_thing(signals))


def _collides(signals: list[_Signal]) -> bool:
    """The dedup rule itself: any one strict hit, or any two relaxed hits."""
    if any(signal.score >= signal.strict for signal in signals):
        return True
    return sum(1 for signal in signals if signal.score >= signal.relaxed) >= 2


def _says_the_same_thing(signals: list[_Signal]) -> bool:
    """Whether the CONTENT signal itself hit STRICT — the stored entry says what this
    write says.

    Strict and not relaxed, because the relaxed band is where two entries about the same
    SUBJECT sit, and a change is nearly always a small edit to a sentence about the
    subject it is a change to.  An unscored content signal (either side has no vector) is
    NOT the same value: there is no evidence it is, and reading absence as sameness would
    silence a change on exactly the entries dedup can say least about."""
    return any(signal.is_content and signal.score >= signal.strict for signal in signals)


def _score_signals(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> list[_Signal]:
    """Every applicable signal, scored against its own two thresholds."""
    out: list[_Signal] = []
    if candidate.key is not None and existing.key is not None:
        out.append(
            _Signal(
                token_containment_ratio(candidate.key, existing.key),
                thresholds.key_tcr_strict,
                thresholds.key_tcr_relaxed,
            )
        )
    key_cos = _safe_cosine(candidate.key_vec, existing.key_vec)
    if key_cos is not None:
        out.append(_Signal(key_cos, thresholds.key_sim_strict, thresholds.key_sim_relaxed))
    content_cos = _safe_cosine(candidate.content_vec, existing.content_vec)
    if content_cos is not None:
        out.append(
            _Signal(
                content_cos,
                thresholds.content_sim_strict,
                thresholds.content_sim_relaxed,
                is_content=True,
            )
        )
    return out


def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return cosine_similarity(a, b)


# ── Retrieval scoring ────────────────────────────────────────────────────────


def stack_normalized(blobs: Iterable[bytes]) -> np.ndarray:
    """Stack serialized embeddings into an L2-normalized (N, D) float32 matrix.

    Uses ``np.frombuffer`` so each blob materializes via a zero-copy view
    that's then assigned into the matrix — ~1 ms for 1500×768 in practice.
    """
    blob_list = list(blobs)
    if not blob_list:
        return np.zeros((0, 0), dtype=np.float32)
    dim = len(blob_list[0]) // 4
    matrix = np.empty((len(blob_list), dim), dtype=np.float32)
    for index, blob in enumerate(blob_list):
        matrix[index] = np.frombuffer(blob, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def stack_normalized_anchors(anchors: list[list[float]]) -> np.ndarray:
    """Stack anchor vectors into an L2-normalized (M, D) float32 matrix."""
    matrix = np.asarray(anchors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def cosine_scores(content_blobs: list[bytes], anchor: list[float]) -> np.ndarray:
    """Per-row cosine of each stored embedding to a single ``anchor`` vector.

    Plain nearest-neighbor scoring for the explicit ``read_similar`` search
    tool and resolve-by-meaning — entries come back ranked so the model judges
    them, with no relevance-injection gate.
    """
    matrix = stack_normalized(content_blobs)  # (N, D)
    anchor_matrix = stack_normalized_anchors([anchor])  # (1, D)
    return (matrix @ anchor_matrix.T)[:, 0]  # (N,)
