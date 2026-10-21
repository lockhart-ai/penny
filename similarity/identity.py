"""Pure entry-identity rules — the deterministic steps of the identity ladder.

Two things a caller needs when deciding whether a new entry is the same as a
stored one, before any model is asked:

  * ``normalise_key`` / ``key_parts`` — the key normalisation the rules read:
    case, accents, punctuation, underscore/hyphen separators, a leading ``the``,
    a trailing parenthetical (``(band)``), and surrounding brackets.  The
    trailing parenthetical is split OFF first and normalised separately, so the
    number rule can read it: ``Kino's Journey (2003)`` is not ``Kino's Journey
    (2017)``, but ``Texhnolyze`` is still ``Texhnolyze (2003 anime)``.
  * ``pair_rule_verdict`` — the rules themselves, each settling only what it is
    sure of: keys whose MAIN part differs in a number token (digits, Roman
    numerals) are different entries; two parentheticals that both carry number
    tokens and differ are different entries; keys equal after normalising are
    the same entry.  Anything else returns ``None`` — undecided, left to the
    caller's next step (a shortlist and a decision model).  Content is
    deliberately not a rule: identical content under a different key is a
    question the decision model sees both keys for, never a repeat a
    deterministic step settles.
  * ``shortlist`` — rank stored entries against a candidate by cosine (key and
    content, best surface wins), best-first, and take the top ``k``.  Ranking
    only, no cutoff: every shortlisted entry is a candidate the caller still asks
    about.

Free functions over plain values, dependency-light like the rest of this
package, so the memory layer and a future second customer (collection-name
matching) can share one declaration of the rules instead of two drifting
copies.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import NamedTuple

from similarity.embeddings import cosine_similarity, normalize_unicode

# ── Key normalisation ────────────────────────────────────────────────────────

_LEADING_THE_RE = re.compile(r"^the\s+")
_TRAILING_PAREN_RE = re.compile(r"\s*\(([^()]*)\)$")
# Punctuation that is neither a word character nor whitespace — apostrophes,
# exclamation marks, dots, slashes — reads as noise for identity, so it splits
# into a word separator ("portugal. the man" ≈ "portugal the man").
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _unwrap_brackets(text: str) -> str:
    """Strip ONE layer of surrounding brackets — framing, not identity:
    ``[dune]`` or ``(dune)`` → ``dune``."""
    text = text.strip()
    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if text.startswith(opener) and text.endswith(closer) and len(text) > 2:
            return text[1:-1].strip()
    return text


def _split_parenthetical(key: str) -> tuple[str, str | None]:
    """Split a raw key into its main part and its trailing parenthetical
    (before normalisation).  Surrounding brackets come off first, so a
    whole-paren key like ``(dune)`` unwraps to the main part ``dune`` with no
    parenthetical; a parenthetical only counts when a non-empty main part
    precedes it."""
    unwrapped = _unwrap_brackets(key)
    match = _TRAILING_PAREN_RE.search(unwrapped)
    if match and unwrapped[: match.start()].strip():
        return unwrapped[: match.start()].rstrip(), match.group(1)
    return unwrapped, None


def _base_normalise(text: str) -> str:
    """Normalisation without the trailing-parenthetical handling: case, accents,
    punctuation, separators, a leading ``the``, surrounding brackets."""
    text = normalize_unicode(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = _unwrap_brackets(text)
    text = _LEADING_THE_RE.sub("", text)
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def normalise_key(key: str) -> str:
    """Reduce a key to the form identity is decided on: case-, accent-,
    punctuation- and separator-insensitive, without a leading ``the``, a
    trailing parenthetical, or surrounding brackets."""
    main, _ = _split_parenthetical(key)
    return _base_normalise(main)


def key_parts(key: str) -> tuple[str, str | None]:
    """A key split into its normalised MAIN part and its normalised trailing
    parenthetical, when it carries one — the two surfaces the number rule reads
    separately (``Kino's Journey (2003)`` → ``("kino s journey", "2003")``)."""
    main, paren = _split_parenthetical(key)
    return _base_normalise(main), (_base_normalise(paren) if paren is not None else None)


# ── Number tokens ────────────────────────────────────────────────────────────

_DIGIT_RUN_RE = re.compile(r"\d+")
_ROMAN_TOKEN_RE = re.compile(r"^[ivxlcdm]+$")
_ROMAN_TABLE = (
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
)


def _is_roman_numeral(token: str) -> bool:
    """Whether a token is a well-formed Roman numeral.

    Parses greedily and requires the round trip back to the canonical form, so
    tokens that merely look Roman (``im``, ``xm``) are rejected rather than read
    as numbers."""
    value = 0
    rest = token
    for amount, glyph in _ROMAN_TABLE:
        while rest.startswith(glyph):
            value += amount
            rest = rest[len(glyph) :]
    if rest:
        return False
    canonical = ""
    for amount, glyph in _ROMAN_TABLE:
        canonical += glyph * (value // amount)
        value %= amount
    return canonical == token


def number_tokens(key: str) -> list[str]:
    """The number tokens a normalised key part carries, as a sorted multiset:
    digit runs (including dates) and standalone well-formed Roman numerals.
    Two keys whose MAIN-part number tokens differ are different entries —
    ``2026-10-20`` is not ``2026-10-21``, ``Final Fantasy VI`` is not ``VII``;
    two PARENTHETICALS that both carry numbers and differ decide the same way
    (``Dune (1984)`` is not ``Dune (2021)``)."""
    tokens = [*_DIGIT_RUN_RE.findall(key)]
    tokens.extend(
        word
        for word in key.split()
        if _ROMAN_TOKEN_RE.match(word) and _is_roman_numeral(word)
    )
    return sorted(tokens)


# ── The rules ────────────────────────────────────────────────────────────────


class KeyRule(StrEnum):
    """What the deterministic rules say about one candidate/existing pair."""

    SAME = "same"
    DIFFERENT = "different"


def pair_rule_verdict(
    candidate_key: str,
    existing_key: str,
    candidate_content: str,
    existing_content: str,
) -> KeyRule | None:
    """The deterministic verdict for one pair, or ``None`` when the rules do not
    settle it.

    Each rule settles only what it is sure of, in order: keys that differ in a
    number token are DIFFERENT — decisively, since a date key is exactly what
    makes two digest entries different — read on the main part, and on the
    trailing parenthetical only when BOTH keys carry one with numbers in it (a
    number on just one side is a qualifier, not an identity: ``Texhnolyze`` is
    still ``Texhnolyze (2003 anime)``); keys equal after normalising are the
    SAME entry.  Content settles nothing here: identical content under a
    different key is the decision model's question, which sees both keys — a
    deterministic repeat rule would block two genuinely different things that
    happen to carry the same value."""
    cand_main, cand_paren = key_parts(candidate_key)
    exist_main, exist_paren = key_parts(existing_key)
    if number_tokens(cand_main) != number_tokens(exist_main):
        return KeyRule.DIFFERENT
    if cand_paren is not None and exist_paren is not None:
        cand_paren_tokens = number_tokens(cand_paren)
        exist_paren_tokens = number_tokens(exist_paren)
        if cand_paren_tokens and exist_paren_tokens and cand_paren_tokens != exist_paren_tokens:
            return KeyRule.DIFFERENT
    if cand_main == exist_main:
        return KeyRule.SAME
    return None


# ── Shortlist ────────────────────────────────────────────────────────────────


class ShortlistSide(NamedTuple):
    """One side of a shortlist comparison: the two embeddings a stored entry
    offers to rank against.  Missing vectors are skipped, never guessed."""

    key_vec: list[float] | None
    content_vec: list[float] | None


def shortlist(
    candidate_key_vec: list[float] | None,
    candidate_content_vec: list[float] | None,
    existing: list[ShortlistSide],
    k: int,
) -> list[int]:
    """Indices of the ``k`` best-ranking stored entries for a candidate, best
    first.

    Ranking only, no cutoff: the score is the best cosine either surface offers
    (a collision can live on the key or in the value), so a candidate with no
    embeddings shortlists nothing and every ranked entry remains a question for
    the caller's next step.  Ties keep store order."""
    scored: list[tuple[float, int]] = []
    for index, side in enumerate(existing):
        scores = [
            cosine
            for cosine in (
                _safe_cosine(candidate_key_vec, side.key_vec),
                _safe_cosine(candidate_content_vec, side.content_vec),
            )
            if cosine is not None
        ]
        if scores:
            scored.append((max(scores), index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scored[:k]]


def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return cosine_similarity(a, b)
