"""Embedding storage and similarity search utilities."""

from __future__ import annotations

import math
import re
import struct
import unicodedata


def serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a float vector to a compact binary blob for SQLite storage."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """Deserialize a binary blob back to a float vector."""
    count = len(data) // 4  # 4 bytes per float32
    return list(struct.unpack(f"<{count}f", data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(
    query: list[float],
    candidates: list[tuple[int, list[float]]],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Find the most similar candidates to a query embedding.

    Args:
        query: Query embedding vector
        candidates: List of (id, embedding) tuples to search
        top_k: Maximum number of results to return
        threshold: Minimum cosine similarity to include

    Returns:
        List of (id, similarity_score) tuples, sorted by descending similarity
    """
    scored = []
    for item_id, embedding in candidates:
        score = cosine_similarity(query, embedding)
        if score >= threshold:
            scored.append((item_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# Unicode punctuation that NFKD doesn't normalize to ASCII
_UNICODE_REPLACEMENTS = (
    ("\u2010", "-"),  # HYPHEN
    ("\u2011", "-"),  # NON-BREAKING HYPHEN
    ("\u2013", "-"),  # EN DASH
    ("\u2014", "-"),  # EM DASH
    ("\u2018", "'"),  # LEFT SINGLE QUOTATION MARK
    ("\u2019", "'"),  # RIGHT SINGLE QUOTATION MARK
    ("\u201c", '"'),  # LEFT DOUBLE QUOTATION MARK
    ("\u201d", '"'),  # RIGHT DOUBLE QUOTATION MARK
)


def normalize_unicode(text: str) -> str:
    """Normalize unicode variants to ASCII equivalents for token comparison.

    Applies NFKD decomposition, strips combining marks (accents), and replaces
    common unicode punctuation with ASCII equivalents.
    """
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    for old, new in _UNICODE_REPLACEMENTS:
        stripped = stripped.replace(old, new)
    return stripped


_TRAILING_YEAR_RE = re.compile(r"20\d{2}$")


def tokenize_entity_name(text: str) -> list[str]:
    """Tokenize an entity name for dedup comparison.

    Applies unicode normalization, lowercasing, separator normalization
    (underscores/hyphens → spaces), and trailing year stripping (2000-2099).
    """
    normalized = normalize_unicode(text).lower()
    # Normalize separators to spaces
    normalized = normalized.replace("_", " ").replace("-", " ")
    tokens = normalized.split()
    # Strip trailing year suffix (2000-2099): "agentica2026" → "agentica", "2026" → ""
    tokens = [_TRAILING_YEAR_RE.sub("", t) for t in tokens]
    # Filter empty tokens (pure year tokens become empty)
    return [t for t in tokens if t]


def token_containment_ratio(name_a: str, name_b: str) -> float:
    """Fraction of the shorter name's tokens found in the longer name.

    Returns 1.0 when the shorter name is a complete token-subset of the longer.
    Used as a fast lexical signal for entity deduplication.
    """
    tokens_a = set(tokenize_entity_name(name_a))
    tokens_b = set(tokenize_entity_name(name_b))
    shorter, longer = (
        (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    )
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


# ── Job-name normalisation (#1775) ───────────────────────────────────────────
#
# A *job* name (a collection watching something) is written two ways for the same
# thing: with the mechanism spelled out (``aurora-deck-2-monitor``) and without
# it (``aurora-deck-2``), singular and plural (``aurora-price`` /
# ``aurora-prices``), underscored and hyphenated.  Measured against the labelled
# corpus, folding those three axes is what lets a real duplicate REACH strict
# containment: raw containment caught 4/7, + role strip 5/7, + role strip and
# stemming 6/7 — each layer earning its place, all three at zero false merges.

# The CLOSED role-word vocabulary: words naming the MECHANISM that watches a
# subject rather than the subject itself.  Deliberately small and closed — every
# member is a word a collection name uses to say "…and I watch it", so dropping
# it leaves the subject behind.  Listed in canonical singular form (``singularize``
# runs first, so ``alerts`` / ``monitors`` / ``feeds`` need no separate entry);
# non-plural inflections (``monitoring``, ``watching``) are spelled out.
#
# Deliberately EXCLUDED: bare ``track`` and ``check``, which carry subject meaning
# of their own (a music track, a rent check) — only their mechanism-shaped
# relatives (``tracker`` / ``tracking`` / ``checker`` / ``checking``) are roles.
JOB_ROLE_WORDS: frozenset[str] = frozenset(
    {
        "monitor",
        "monitoring",
        "watch",
        "watcher",
        "watching",
        "tracker",
        "tracking",
        "scan",
        "scanner",
        "scanning",
        "alert",
        "alerting",
        "update",
        "feed",
        "notifier",
        "notification",
        "notify",
        "checker",
        "checking",
        "log",
        "latest",
    }
)

# Words that end in ``s`` without being plural — stripping the ``s`` would invent a
# different word (``news`` → ``new``), which is how a naive stemmer manufactures a
# false merge.  Endings already handled structurally (``-ss`` / ``-us`` / ``-is`` /
# ``-os``) need no entry here.
_PLURAL_INVARIANT: frozenset[str] = frozenset({"news", "series", "species"})

# ``-es`` after a sibilant is a plural marker on the whole ``es`` (``watches`` →
# ``watch``), unlike ``-es`` elsewhere (``prices`` → ``price``).
_SIBILANT_PLURALS = ("sses", "shes", "ches", "xes", "zes")

# Endings that are singular already, so a trailing ``s`` says nothing about number.
_SINGULAR_S_ENDINGS = ("ss", "us", "is", "os")

_MIN_STEMMABLE_LENGTH = 3


def singularize(token: str) -> str:
    """Naive singular form of one token — enough to converge the plural/singular
    twins a job name forks on (``prices``/``price``, ``decks``/``deck``), with no
    dictionary and no dependency.

    Conservative by construction: short tokens, known invariants (``news``), and
    already-singular ``s`` endings (``status``, ``analysis``) are returned
    untouched, because a wrong stem is a false merge and a missed stem is only a
    missed catch.
    """
    if len(token) <= _MIN_STEMMABLE_LENGTH or token in _PLURAL_INVARIANT:
        return token
    if token.endswith("ies"):
        return f"{token[:-3]}y"
    if token.endswith(_SIBILANT_PLURALS):
        return token[:-2]
    if token.endswith(_SINGULAR_S_ENDINGS):
        return token
    return token[:-1] if token.endswith("s") else token


def tokenize_job_name(text: str) -> list[str]:
    """Tokenize a *job* name for duplicate detection: :func:`tokenize_entity_name`'s
    normalisation (unicode variants, case, ``_``/``-``/whitespace separators,
    trailing years), then singularised, then stripped of the closed
    :data:`JOB_ROLE_WORDS` vocabulary — so ``aurora_deck_2_prices`` and
    ``aurora-deck-2-price-monitor`` both reduce to the same four tokens.
    """
    singular = (singularize(token) for token in tokenize_entity_name(text))
    return [token for token in singular if token not in JOB_ROLE_WORDS]


def is_token_containment(name_a: str, name_b: str) -> bool:
    """Whether one job name's tokens are wholly inside the other's.

    A **set relation, not a threshold** — there is nothing here to tune or drift.
    It is the discriminator the labelled corpus picked out: real duplicates are
    *extensions* (one name's tokens a subset of the other's), while the traps are
    *substitutions* — each side contributing a token the other lacks (``2``|``3``,
    ``times``|``fares``) — so neither side is contained in the other.

    A name that normalises to nothing (a bare role word like ``monitor``) is
    contained in nothing: an empty set is a subset of everything, which would
    make it a duplicate of every collection.
    """
    tokens_a = set(tokenize_job_name(name_a))
    tokens_b = set(tokenize_job_name(name_b))
    if not tokens_a or not tokens_b:
        return False
    return tokens_a <= tokens_b or tokens_b <= tokens_a
