"""Similarity primitives for penny (cosine similarity, TCR, dedup)."""

from similarity.dedup import DedupStrategy, is_embedding_duplicate
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    normalize_unicode,
    serialize_embedding,
    token_containment_ratio,
    tokenize_entity_name,
)

__all__ = [
    "DedupStrategy",
    "cosine_similarity",
    "deserialize_embedding",
    "find_similar",
    "is_embedding_duplicate",
    "normalize_unicode",
    "serialize_embedding",
    "token_containment_ratio",
    "tokenize_entity_name",
]
