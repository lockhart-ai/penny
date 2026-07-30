"""Tests for the shared similarity and dedup module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from similarity.dedup import DedupStrategy, JobSide, is_embedding_duplicate, is_same_job

from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmResponseError
from penny.llm.similarity import embed_text

# ── embed_text ────────────────────────────────────────────────────────────────


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_returns_vector_on_success(self) -> None:
        client = AsyncMock()
        client.embed.return_value = [[1.0, 2.0, 3.0]]
        result = await embed_text(client, "hello")
        assert result == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self) -> None:
        client = AsyncMock()
        client.embed.side_effect = LlmResponseError("boom")
        result = await embed_text(client, "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_llm_exception_propagates(self) -> None:
        """Bugs in the embed pipeline must surface, not be swallowed as None."""
        client = AsyncMock()
        client.embed.side_effect = RuntimeError("unexpected bug")
        with pytest.raises(RuntimeError, match="unexpected bug"):
            await embed_text(client, "hello")


# ── is_embedding_duplicate ────────────────────────────────────────────────────


def _make_item(name: str, vec: list[float] | None) -> tuple[str, bytes | None]:
    """Helper to build (name, serialized_embedding) tuple."""
    return (name, serialize_embedding(vec) if vec else None)


class TestIsEmbeddingDuplicate:
    def test_none_candidate_vec_returns_none(self) -> None:
        items = [_make_item("foo", [1.0, 0.0])]
        result = is_embedding_duplicate("foo", None, items, DedupStrategy.EMBEDDING_ONLY, 0.8)
        assert result is None

    def test_embedding_only_match(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("different name", vec)]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 0

    def test_embedding_only_no_match(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        items = [_make_item("other", b)]
        result = is_embedding_duplicate("candidate", a, items, DedupStrategy.EMBEDDING_ONLY, 0.5)
        assert result is None

    def test_tcr_and_embedding_both_pass(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("star trek voyager", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_and_embedding_tcr_fails(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result is None

    def test_tcr_and_embedding_single_token_bypass(self) -> None:
        """Single-token names skip TCR requirement (e.g. acronyms)."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("clps", vec)]
        result = is_embedding_duplicate(
            "foo", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_tcr_only(self) -> None:
        """TCR passes but no embedding available — still a match in OR mode."""
        items: list[tuple[str, bytes | None]] = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", [1.0, 0.0], items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_no_candidate_vec_still_matches_tcr(self) -> None:
        """TCR passes with None candidate_vec — still a match in OR mode."""
        items: list[tuple[str, bytes | None]] = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", None, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_embedding_only(self) -> None:
        """TCR fails but embedding passes — still a match in OR mode."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "something else", vec, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_returns_first_match_index(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [
            _make_item("no match", [0.0, 1.0, 0.0]),
            _make_item("match", vec),
        ]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 1


# ── is_same_job — the labelled corpus (#1775) ────────────────────────────────
#
# The deterministic fixture the #1775 strategy was picked against, transposed onto
# the synthetic ``aurora-*`` / ``harbor-*`` families: every pair the corpus labels a
# DUPLICATE (the same one thing named two ways) and every pair it labels DISTINCT
# (the adversarial negatives — same subject, genuinely different thing).  Both
# directions are pinned, because a rule that only proves duplicates are caught is
# half a rule: the failure this fixture exists to prevent is a merge, not a miss.

# Labelled DUPLICATES — one job, two names.  Each line is the axis it forks on.
_DUPLICATE_PAIRS = [
    ("aurora-deck-2-price", "aurora-deck-2-prices"),  # plural
    ("aurora-deck-2-price", "aurora_deck_2_price"),  # underscore separators
    ("aurora-deck-2-price", "Aurora Deck 2 Price"),  # case + whitespace
    ("aurora-deck-2-price", "aurora‑deck‑2‑price"),  # unicode hyphens
    ("aurora-deck-2-price", "aurora-deck-2"),  # dropped qualifier
    ("aurora-deck-2-price", "aurora-deck-2-monitor"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-watcher"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-tracker"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-alerts"),  # role affix, plural
    ("aurora-deck-2-price", "aurora-deck-2-price-latest"),  # role affix
    ("aurora-deck-2-price", "aurora-prices"),  # catch-all vs. the one scoped from it
    ("aurora-deck-2-price", "aurora-price"),
    ("aurora-deck-2-price", "aurora-deck-price"),
    ("aurora-deck-2-price", "aurora-deck-prices"),
    ("aurora-deck-2-price", "aurora-decks"),
    ("aurora-deck-2-price", "aurora-decks-prices"),
    ("harbor-ferry-times", "harbor-ferry-time-monitor"),  # role affix + plural
]

# Labelled DISTINCT — the adversarial negatives.  Every one is a SUBSTITUTION: each
# side contributes a token the other lacks, and that token is the one carrying the
# meaning, which is exactly why no name-similarity threshold can separate them from
# a real duplicate and why containment (a set relation) can.
_DISTINCT_PAIRS = [
    ("aurora-deck-2-price", "aurora-deck-3-price"),  # different listing
    ("aurora-deck-2-price", "aurora-deck-2-condition"),  # different fact
    ("aurora-deck-2-price", "aurora-deck-2-seller-rating"),  # different fact
    ("aurora-deck-2-price", "boreal-deck-2-price"),  # different subject
    ("harbor-ferry-times", "harbor-ferry-fares"),  # different fact
    ("harbor-ferry-times", "harbor-bus-times"),  # different subject
    ("harbor-news-northwind", "harbor-news-southgale"),  # different source
    ("aurora-bids-notified", "aurora-bids-unnotified"),  # opposite halves of a pair
    ("aurora-deck-2-price", "harbor-ferry-times"),  # unrelated
    ("aurora-deck-2-price", "monitor"),  # a bare role word names no job
]

# Labelled duplicates the containment tier does NOT catch: one side says ``auction``
# where the other says ``price``, so it reads as a substitution.  Tier 3
# (near-containment + same-day + param overlap → ASK) is what reaches these, and it
# is deliberately out of scope for #1775 — pinned here so the gap is a recorded
# measurement rather than a surprise.
_KNOWN_MISSES = [
    ("aurora-deck-2-price", "aurora-deck-2-auction-watch"),
    ("aurora-deck-2-price", "aurora-deck-2-auction-watcher"),
]


def _same(name_a: str, name_b: str) -> bool:
    """``is_same_job`` on two bare names (no skill attached on either side)."""
    return is_same_job(JobSide(name=name_a), JobSide(name=name_b))


class TestIsSameJob:
    """The two-tier job-identity rule scored against the labelled corpus."""

    @pytest.mark.parametrize(("name_a", "name_b"), _DUPLICATE_PAIRS)
    def test_labelled_duplicates_are_caught(self, name_a: str, name_b: str) -> None:
        assert _same(name_a, name_b)
        assert _same(name_b, name_a)  # symmetric — order of arrival can't change it

    @pytest.mark.parametrize(("name_a", "name_b"), _DISTINCT_PAIRS)
    def test_labelled_negatives_are_never_merged(self, name_a: str, name_b: str) -> None:
        assert not _same(name_a, name_b)
        assert not _same(name_b, name_a)

    @pytest.mark.parametrize(("name_a", "name_b"), _KNOWN_MISSES)
    def test_substitution_duplicates_are_the_known_gap(self, name_a: str, name_b: str) -> None:
        """Recorded, not asserted-as-correct: these ARE duplicates, and tiers 1-2
        miss them.  Catching them is tier 3's job (ask, never auto-merge)."""
        assert not _same(name_a, name_b)

    def test_corpus_score(self) -> None:
        """The headline measurement: every negative held, and the misses are exactly
        the two substitution-shaped duplicates — so a regression in either direction
        shows up as a number, not as one parametrised line going red."""
        caught = sum(1 for pair in _DUPLICATE_PAIRS + _KNOWN_MISSES if _same(*pair))
        false_merges = sum(1 for pair in _DISTINCT_PAIRS if _same(*pair))
        assert (caught, false_merges) == (17, 0)

    def test_same_skill_and_params_is_the_same_job_whatever_the_names(self) -> None:
        """Tier 1: identical bound skills are one job even when the names share
        nothing — the case where names carry no usable signal at all."""
        params = {"url": "https://example.test/listing", "field": "price"}
        assert is_same_job(
            JobSide(name="aurora-deck-2-price", skill="Watch a page field", params=params),
            JobSide(name="harbor-ferry-times", skill="Watch a page field", params=params),
        )

    def test_same_skill_different_params_is_a_different_job(self) -> None:
        """Tier 1 is what separates the pair that defeats every name rule: the same
        routine bound to a DIFFERENT listing is a different job."""
        assert not is_same_job(
            JobSide(
                name="aurora-deck-2-price",
                skill="Watch a page field",
                params={"url": "https://example.test/deck-2"},
            ),
            JobSide(
                name="aurora-deck-3-price",
                skill="Watch a page field",
                params={"url": "https://example.test/deck-3"},
            ),
        )

    def test_skill_on_one_side_only_contributes_nothing(self) -> None:
        """Field-wise over the POPULATED INTERSECTION: an inert collection (no skill)
        is neither merged into an instantiated one for lacking a skill nor separated
        from it — tier 1 abstains and tier 2 decides on the names alone."""
        instantiated = JobSide(
            name="aurora-deck-2-price", skill="Watch a page field", params={"url": "u"}
        )
        assert is_same_job(JobSide(name="aurora-deck-2-prices"), instantiated)
        assert not is_same_job(JobSide(name="harbor-ferry-times"), instantiated)

    def test_unbound_skill_on_both_sides_falls_through_to_the_names(self) -> None:
        """A skill with no bindings says nothing about which job it is running, so an
        empty ``params`` is an absent field, not a match."""
        assert not is_same_job(
            JobSide(name="aurora-deck-2-price", skill="Watch a page field", params={}),
            JobSide(name="harbor-ferry-times", skill="Watch a page field", params={}),
        )
