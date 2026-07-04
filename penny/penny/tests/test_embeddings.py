"""Tests for embedding utilities and LlmClient.embed()."""

import math

import openai
import pytest
from similarity.embeddings import (
    find_similar,
    token_containment_ratio,
    tokenize_entity_name,
)

from penny.llm import LlmNotFoundError, LlmResponseError
from penny.llm.client import LlmClient
from penny.llm.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
)
from penny.tests.mocks.llm_patches import EMBED_DIM, deterministic_embed


class TestSerializeDeserialize:
    """Tests for embedding serialization round-trip."""

    def test_round_trip(self):
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = serialize_embedding(original)
        restored = deserialize_embedding(blob)
        assert len(restored) == len(original)
        for a, b in zip(original, restored, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_compact_size(self):
        embedding = [0.0] * 768
        blob = serialize_embedding(embedding)
        assert len(blob) == 768 * 4  # 4 bytes per float32

    def test_empty_vector(self):
        blob = serialize_embedding([])
        assert deserialize_embedding(blob) == []


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestTokenizeEntityName:
    """Tests for entity name tokenization with dedup normalization."""

    def test_basic_split(self):
        assert tokenize_entity_name("Stanford University") == ["stanford", "university"]

    def test_underscore_to_space(self):
        assert tokenize_entity_name("agentic_ai_summit") == ["agentic", "ai", "summit"]

    def test_hyphen_to_space(self):
        assert tokenize_entity_name("kef-ls50-meta") == ["kef", "ls50", "meta"]

    def test_year_token_stripped(self):
        assert tokenize_entity_name("aamas 2026") == ["aamas"]

    def test_year_suffix_stripped(self):
        assert tokenize_entity_name("agentica2026") == ["agentica"]

    def test_non_year_number_preserved(self):
        assert tokenize_entity_name("nvidia gb10") == ["nvidia", "gb10"]

    def test_pure_year_only_returns_empty(self):
        assert tokenize_entity_name("2025") == []

    def test_mixed_separators(self):
        assert tokenize_entity_name("etsi_ai-data conference 2026") == [
            "etsi",
            "ai",
            "data",
            "conference",
        ]

    def test_19xx_year_not_stripped(self):
        """Years before 2000 are not stripped (regex is 20xx only)."""
        assert tokenize_entity_name("event 1999") == ["event", "1999"]


class TestTokenContainmentRatio:
    """Tests for TCR with normalized tokenization."""

    def test_underscore_variant_matches(self):
        """agentic_ai_summit vs agentic ai summit → TCR = 1.0"""
        assert token_containment_ratio("agentic_ai_summit", "agentic ai summit") == 1.0

    def test_year_variant_matches(self):
        """etsi ai data conference 2026 vs etsi ai data conference → TCR = 1.0"""
        assert (
            token_containment_ratio("etsi ai data conference 2026", "etsi ai data conference")
            == 1.0
        )

    def test_year_suffix_matches(self):
        """agentica2026 vs agentica → TCR = 1.0"""
        assert token_containment_ratio("agentica2026", "agentica") == 1.0

    def test_abbreviation_partial_overlap(self):
        """applied ai conference vs applied ai conf → TCR = 2/3 ≈ 0.67"""
        tcr = token_containment_ratio("applied ai conference", "applied ai conf")
        assert tcr == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert token_containment_ratio("kef ls50", "nvidia rtx") == 0.0

    def test_empty_returns_zero(self):
        """Pure year vs pure year → both empty after normalization → 0.0"""
        assert token_containment_ratio("2026", "2025") == 0.0


class TestFindSimilar:
    """Tests for find_similar search function."""

    def test_returns_top_k(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            (1, [1.0, 0.0, 0.0]),  # identical
            (2, [0.9, 0.1, 0.0]),  # very similar
            (3, [0.0, 1.0, 0.0]),  # orthogonal
            (4, [0.5, 0.5, 0.0]),  # moderate
        ]
        results = find_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1  # Most similar first
        assert results[1][0] == 2

    def test_threshold_filters(self):
        query = [1.0, 0.0]
        candidates = [
            (1, [1.0, 0.0]),  # similarity = 1.0
            (2, [0.0, 1.0]),  # similarity = 0.0
            (3, [-1.0, 0.0]),  # similarity = -1.0
        ]
        results = find_similar(query, candidates, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_empty_candidates(self):
        assert find_similar([1.0], [], top_k=5) == []

    def test_descending_order(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            (1, [0.0, 1.0, 0.0]),
            (2, [0.5, 0.5, 0.0]),
            (3, [1.0, 0.0, 0.0]),
        ]
        results = find_similar(query, candidates)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestLlmClientEmbed:
    """Integration tests for LlmClient.embed() with mock."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_llm):
        expected = [[0.1, 0.2, 0.3, 0.4]]
        mock_llm.set_embed_handler(lambda model, input: expected)

        client = LlmClient(
            api_url="http://localhost:11434",
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("hello world")

        assert result == expected
        assert len(mock_llm.embed_requests) == 1
        assert mock_llm.embed_requests[0]["model"] == "nomic-embed-text"
        assert mock_llm.embed_requests[0]["input"] == "hello world"

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_llm):
        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_llm.set_embed_handler(lambda model, input: expected)

        client = LlmClient(
            api_url="http://localhost:11434",
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed(["a", "b", "c"])

        assert len(result) == 3
        assert result == expected

    @pytest.mark.asyncio
    async def test_embed_default_mock(self, mock_llm):
        """Default mock returns a deterministic, L2-normalised unit vector."""
        client = LlmClient(
            api_url="http://localhost:11434",
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("test")

        assert len(result) == 1
        assert len(result[0]) == EMBED_DIM
        assert result[0] == deterministic_embed("test")
        assert abs(sum(v * v for v in result[0]) - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_embed_404_raises_immediately_without_retry(self, mock_llm):
        """A 404 (model not found) must raise immediately — no retries."""
        call_count = 0

        def raising_handler(model: str, input: str | list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise LlmNotFoundError("model not found")

        mock_llm.set_embed_handler(raising_handler)

        client = LlmClient(
            api_url="http://localhost:11434",
            model="missing-model",
            max_retries=3,
            retry_delay=0.0,
        )
        with pytest.raises(LlmNotFoundError):
            await client.embed("hello")

        # Must have called embed exactly once — no retries on 404
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_embed_transient_error_retries(self, mock_llm):
        """Non-404 errors should still be retried up to max_retries."""
        call_count = 0

        def flaky_handler(model: str, input: str | list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise openai.OpenAIError("server error")

        mock_llm.set_embed_handler(flaky_handler)

        client = LlmClient(
            api_url="http://localhost:11434",
            model="some-model",
            max_retries=3,
            retry_delay=0.0,
        )
        with pytest.raises(LlmResponseError):
            await client.embed("hello")

        # Should have retried all 3 times
        assert call_count == 3
