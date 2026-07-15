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


def is_duplicate(
    candidate: EntrySide,
    existing: list[EntrySide],
    thresholds: DedupThresholds,
) -> EntrySide | None:
    """Return the first existing entry that ``candidate`` collides with under the
    dedup rule, or ``None`` if no match.  Returning the matched side (instead of
    bool) lets callers surface *which* existing entry blocked the write — the
    rejection message can then name it so the model can pivot to ``update_entry``
    when it has fresher info."""
    for side in existing:
        if _pair_is_duplicate(candidate, side, thresholds):
            return side
    return None


def _pair_is_duplicate(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> bool:
    """Apply the three-signal dedup rule to a single candidate/existing pair.

    Signals that can't be computed (missing keys, missing embeddings) are
    skipped. Fire if any one signal hits its strict threshold or any two
    signals hit their relaxed thresholds.
    """
    signals = _score_signals(candidate, existing, thresholds)
    if any(score >= strict for score, strict, _ in signals):
        return True
    relaxed_hits = sum(1 for score, _, relaxed in signals if score >= relaxed)
    return relaxed_hits >= 2


def _score_signals(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> list[tuple[float, float, float]]:
    """Return (score, strict_threshold, relaxed_threshold) for every applicable signal."""
    out: list[tuple[float, float, float]] = []
    if candidate.key is not None and existing.key is not None:
        out.append(
            (
                token_containment_ratio(candidate.key, existing.key),
                thresholds.key_tcr_strict,
                thresholds.key_tcr_relaxed,
            )
        )
    key_cos = _safe_cosine(candidate.key_vec, existing.key_vec)
    if key_cos is not None:
        out.append((key_cos, thresholds.key_sim_strict, thresholds.key_sim_relaxed))
    content_cos = _safe_cosine(candidate.content_vec, existing.content_vec)
    if content_cos is not None:
        out.append((content_cos, thresholds.content_sim_strict, thresholds.content_sim_relaxed))
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
