"""Similarity primitives for penny (cosine similarity, TCR, dedup)."""

from similarity.dedup import DedupStrategy, JobSide, is_embedding_duplicate, is_same_job
from similarity.embeddings import (
    JOB_ROLE_WORDS,
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    is_token_containment,
    normalize_unicode,
    serialize_embedding,
    singularize,
    token_containment_ratio,
    tokenize_entity_name,
    tokenize_job_name,
)

__all__ = [
    "JOB_ROLE_WORDS",
    "DedupStrategy",
    "JobSide",
    "cosine_similarity",
    "deserialize_embedding",
    "find_similar",
    "is_embedding_duplicate",
    "is_same_job",
    "is_token_containment",
    "normalize_unicode",
    "serialize_embedding",
    "singularize",
    "token_containment_ratio",
    "tokenize_entity_name",
    "tokenize_job_name",
]
