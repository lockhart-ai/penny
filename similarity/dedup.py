"""Deduplication strategies using TCR and embedding similarity.

Composes the low-level primitives from embeddings.py.  All functions are
stateless — thresholds and vectors are passed as parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import NamedTuple

from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    is_token_containment,
    token_containment_ratio,
    tokenize_entity_name,
)


class DedupStrategy(StrEnum):
    """How to combine TCR and embedding signals for dedup."""

    EMBEDDING_ONLY = "embedding_only"
    TCR_AND_EMBEDDING = "tcr_and_embedding"
    TCR_OR_EMBEDDING = "tcr_or_embedding"


def is_embedding_duplicate(
    candidate_name: str,
    candidate_vec: list[float] | None,
    existing_items: list[tuple[str, bytes | None]],
    strategy: DedupStrategy,
    embedding_threshold: float,
    tcr_threshold: float = 0.0,
) -> int | None:
    """Check if a candidate is a semantic duplicate of any existing item.

    Args:
        candidate_name: Name/text of the candidate item.
        candidate_vec: Pre-computed embedding (None → no match possible).
        existing_items: List of (name, serialized_embedding_or_None).
        strategy: How to combine TCR and embedding signals.
        embedding_threshold: Cosine similarity threshold for embedding match.
        tcr_threshold: Token containment ratio threshold (ignored for
            EMBEDDING_ONLY).

    Returns:
        Index of the matching existing item, or None if no duplicate found.
    """
    candidate_tokens = tokenize_entity_name(candidate_name)

    for idx, (existing_name, existing_bytes) in enumerate(existing_items):
        tcr_pass = _check_tcr(
            candidate_name, candidate_tokens, existing_name, strategy, tcr_threshold
        )

        if strategy == DedupStrategy.TCR_AND_EMBEDDING and not tcr_pass:
            continue

        if candidate_vec is not None:
            embed_pass = _check_embedding(candidate_vec, existing_bytes, embedding_threshold)
        else:
            embed_pass = False

        if _is_match(strategy, tcr_pass, embed_pass):
            return idx

    return None


def _check_tcr(
    candidate_name: str,
    candidate_tokens: list[str],
    existing_name: str,
    strategy: DedupStrategy,
    tcr_threshold: float,
) -> bool:
    """Evaluate the TCR signal for one candidate–existing pair."""
    if strategy == DedupStrategy.EMBEDDING_ONLY:
        return False

    existing_tokens = tokenize_entity_name(existing_name)
    shorter_len = min(len(candidate_tokens), len(existing_tokens))

    # Single-token bypass: TCR is meaningless with one token (e.g. acronyms).
    if shorter_len <= 1 and strategy == DedupStrategy.TCR_AND_EMBEDDING:
        return True

    tcr = token_containment_ratio(candidate_name, existing_name)
    return tcr >= tcr_threshold


def _check_embedding(
    candidate_vec: list[float],
    existing_bytes: bytes | None,
    embedding_threshold: float,
) -> bool:
    """Evaluate the embedding similarity signal for one pair."""
    if existing_bytes is None:
        return False
    existing_vec = deserialize_embedding(existing_bytes)
    sim = cosine_similarity(candidate_vec, existing_vec)
    return sim >= embedding_threshold


def _is_match(strategy: DedupStrategy, tcr_pass: bool, embed_pass: bool) -> bool:
    """Combine TCR and embedding signals according to the strategy."""
    if strategy == DedupStrategy.EMBEDDING_ONLY:
        return embed_pass
    if strategy == DedupStrategy.TCR_AND_EMBEDDING:
        return embed_pass  # tcr_pass already enforced by caller skip
    # TCR_OR_EMBEDDING
    return tcr_pass or embed_pass


# ── Job identity (#1775) ─────────────────────────────────────────────────────
#
# Whether two watching jobs (Penny's collections) are the SAME job.  Measured
# against a corpus labelled by what actually happened in production, the answer
# is a two-tier rule with no tunable anywhere:
#
#   1. the same skill bound to the same parameters IS the same job, whatever the
#      two are called;
#   2. otherwise, strict token containment of the normalised names.
#
# Notably ABSENT: embedding similarity.  It is *anti*-correlated on the hard
# cases — the highest description cosine in the labelled set (0.90) belongs to a
# pair that must NOT merge (the same product listing, two model numbers), the
# lowest (0.24) to a real duplicate — because an embedding measures "same kind of
# thing" and every trap here is two collections of the same kind about different
# specifics.  Neither name nor description vectors participate.


class JobSide(NamedTuple):
    """One side of a job-identity comparison: what the job is CALLED (``name``) and
    what it DOES (``skill`` bound to ``params``).  ``skill``/``params`` are ``None``
    for inert storage — a collection with no routine attached yet."""

    name: str
    skill: str | None = None
    params: Mapping[str, str] | None = None


def is_same_job(candidate: JobSide, existing: JobSide) -> bool:
    """Whether ``candidate`` names the same job as ``existing`` — tier 1 (the same
    bound skill) or tier 2 (strict name containment); otherwise distinct.

    Field-wise over the POPULATED INTERSECTION only: a field set on one side and
    absent on the other contributes nothing in either direction, so an inert
    collection is never separated from an instantiated one just for lacking a
    skill, nor merged into one for it.
    """
    return _same_bound_skill(candidate, existing) or is_token_containment(
        candidate.name, existing.name
    )


def _same_bound_skill(candidate: JobSide, existing: JobSide) -> bool:
    """Tier 1: the same skill bound to the same parameters is the same job no matter
    what it is called — decisive exactly where names are unreliable, with no
    threshold.  Both sides must be populated: a missing skill or an empty binding
    is an absent field, never evidence of sameness."""
    if not candidate.skill or not existing.skill or not candidate.params or not existing.params:
        return False
    return candidate.skill == existing.skill and dict(candidate.params) == dict(existing.params)
