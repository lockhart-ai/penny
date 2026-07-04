"""Tests for the shared similarity and dedup module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from similarity.dedup import DedupStrategy, is_embedding_duplicate

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
