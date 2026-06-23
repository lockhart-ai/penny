"""Pure scoring primitives for the memory layer — no DB, no Memory classes.

Two families of shape-independent math the ``Memory`` objects compose:

  * dedup — the three-signal collision rule used by ``Collection.write`` and
    the ``exists`` probe (key TCR, key cosine, content cosine).
  * retrieval — embedding stacking, hybrid cosine+lexical ranking, the
    centrality-magnet penalty, and the adaptive cluster-strength cutoff used by
    ``read_similar`` / ``read_similar_hybrid``.

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
from similarity.lexical import idf, lexical_coverage, reciprocal_rank_fusion, tokens

from penny.constants import PennyConstants
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


def hybrid_scores(cos_matrix: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Combine a (N, M) cosine matrix into a per-row hybrid score.

    Hybrid = max(weighted_decay_over_history, cosine_to_current).  Anchors
    are oldest→newest, so weights go ``decay**(M-1) … decay**0`` and the
    last column is the current message.  With M=1 the weighted branch
    equals the current branch and ``maximum`` returns that single cosine.
    """
    anchor_count = cos_matrix.shape[1]
    weights = np.array(
        [decay ** (anchor_count - 1 - i) for i in range(anchor_count)],
        dtype=np.float32,
    )
    weighted = (cos_matrix * weights).sum(axis=1) / weights.sum()
    current = cos_matrix[:, -1]
    return np.maximum(weighted, current)


def centrality_via_centroid(matrix: np.ndarray) -> np.ndarray:
    """Per-row mean cosine to all OTHER rows, computed via the corpus centroid.

    Algebraically identical to the O(N²) loop ``mean_{j≠i}(cos(v_i, v_j))``:

        mean_{j≠i}(cos) = (N · v_i · centroid − 1) / (N − 1)

    where ``centroid = matrix.mean(axis=0)`` and rows are L2-normalized so
    ``v_i · v_i = 1``.  Cost is one ``mean`` and one matrix-vector product —
    O(N · D) per query, no precompute, no cache.

    Returns zeros for corpora of fewer than 2 rows (no neighbors to average).
    """
    n = matrix.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    centroid = matrix.mean(axis=0)
    return (n * (matrix @ centroid) - 1) / (n - 1)


def hybrid_rank_ids(
    content_blobs: list[bytes],
    contents: list[str],
    ids: list[int],
    anchors: list[list[float]],
    query_text: str,
) -> list[int]:
    """Fuse a cosine ranking and an IDF-lexical ranking via RRF, returning ids.

    Cosine is the best similarity across the conversation window (``max`` over
    anchors) so a strong hit on any turn counts; lexical coverage is the
    IDF-weighted fraction of the query's distinctive tokens each entry
    contains.  Inputs are parallel lists (blob/content/id per row).
    """
    matrix = stack_normalized(content_blobs)
    anchor_matrix = stack_normalized_anchors(anchors)
    best_cosine = (matrix @ anchor_matrix.T).max(axis=1)  # (N,) max over the window
    cosine_rank = [ids[i] for i in np.argsort(-best_cosine)]

    query_tokens = tokens(query_text)
    document_tokens = [tokens(content) for content in contents]
    idf_map = idf(document_tokens)
    coverage = np.array([lexical_coverage(query_tokens, doc, idf_map) for doc in document_tokens])
    coverage = _length_normalize(coverage, document_tokens)
    lexical_rank = [ids[i] for i in np.argsort(-coverage)]
    return reciprocal_rank_fusion([cosine_rank, lexical_rank])


def _length_normalize(coverage: np.ndarray, document_tokens: list[set[str]]) -> np.ndarray:
    """Damp lexical coverage by a sub-linear function of entry length.

    A long entry has a large token set, so it coincidentally contains more of
    any query's terms and wins the lexical leg on surface area alone — the
    long-document bias.  Dividing coverage by ``(1-b) + b*sqrt(len/avglen)``
    demotes those coincidental matches (modest coverage) while leaving genuinely
    on-topic long entries (near-full coverage + strong cosine) in place.  The
    penalty is ~flat — effectively inert — when entry lengths are uniform.
    """
    doc_len = np.array([len(doc) for doc in document_tokens], dtype=np.float32)
    mean_len = float(doc_len.mean()) if doc_len.size else 0.0
    if mean_len <= 0.0:
        return coverage
    b = PennyConstants.MEMORY_LEXICAL_LENGTH_B
    length_norm = (1.0 - b) + b * np.sqrt(doc_len / mean_len)
    return coverage / length_norm


def score_against_anchors(
    content_blobs: list[bytes],
    anchors: list[list[float]],
) -> np.ndarray:
    """Per-row ``max(weighted_decay, current_cos) - α·centrality`` for ranking.

    Stacks all candidate embeddings into an (N, D) matrix and all anchors into
    an (M, D) matrix, then a single matmul produces the full (N, M) cosine
    table.  The centrality-magnet penalty keeps generic boilerplate from
    leaking into unrelated queries.  Single-anchor reduces cleanly (M=1 → the
    weighted-decay branch is the lone cosine).  Returns the adjusted scores in
    row order (caller sorts).
    """
    matrix = stack_normalized(content_blobs)
    anchor_matrix = stack_normalized_anchors(anchors)
    cos_matrix = matrix @ anchor_matrix.T  # (N, M)
    return hybrid_scores(cos_matrix) - (
        PennyConstants.MEMORY_RELEVANT_CENTRALITY_PENALTY * centrality_via_centroid(matrix)
    )


def adaptive_cutoff(scores: list[float], floor: float) -> float | None:
    """Adaptive cutoff for similarity-ranked retrieval (scores sorted desc).

    With at least ``GATE_SAMPLE_SIZE`` candidates, applies a cluster-strength
    gate: if the head-mean / sample-mean ratio falls below ``CLUSTER_GATE``,
    returns ``None`` to suppress the result entirely (flat noise plateau, no
    real cluster).  Otherwise the cutoff combines a relative band against the
    cluster center with the absolute floor.

    Below the cold-start sample-size threshold, the gate is skipped and the
    larger of the configured absolute floor and the caller's ``floor`` is used.
    """
    if not scores:
        return None
    head_size = PennyConstants.MEMORY_RELEVANT_GATE_HEAD_SIZE
    sample_size = PennyConstants.MEMORY_RELEVANT_GATE_SAMPLE_SIZE
    absolute_floor = max(floor, PennyConstants.MEMORY_RELEVANT_ABSOLUTE_FLOOR)
    if len(scores) >= sample_size:
        head_mean = sum(scores[:head_size]) / head_size
        sample_mean = sum(scores[:sample_size]) / sample_size
        if (
            sample_mean <= 0
            or head_mean / sample_mean < PennyConstants.MEMORY_RELEVANT_CLUSTER_GATE
        ):
            return None
        return max(head_mean * PennyConstants.MEMORY_RELEVANT_RELATIVE_RATIO, absolute_floor)
    return absolute_floor
