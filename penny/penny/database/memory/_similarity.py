"""Pure scoring primitives for the memory layer — no DB, no Memory classes.

Shape-independent math the ``Memory`` objects compose: embedding stacking and
plain cosine nearest-neighbor scoring for the explicit ``read_similar`` search
and resolve-by-meaning.  (Entry identity is no longer scored here — the
three-signal dedup rule retired with the identity ladder,
``penny.database.memory.identity``.)

Everything here is a free function over plain values so it stays trivially
testable and reusable from both the entity classes and the registry.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from similarity.embeddings import (
    deserialize_embedding,
    serialize_embedding,
)


def maybe_serialize(vec: list[float] | None) -> bytes | None:
    return serialize_embedding(vec) if vec is not None else None


def maybe_deserialize(blob: bytes | None) -> list[float] | None:
    return deserialize_embedding(blob) if blob is not None else None


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
