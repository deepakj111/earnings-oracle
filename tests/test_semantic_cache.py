# tests/test_semantic_cache.py
"""
Tests for cache/semantic_cache.py — Embedding-Based Semantic Cache.

Tests cover:
  - Collection setup (create vs. skip if exists)
  - Cache hits and misses
  - Cache set (upsert to Qdrant)
  - Disabled cache (full no-op)
  - Error handling (fail-open)
  - GenerationResult round-trip serialization via from_dict
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.http import models as qmodels

from cache.semantic_cache import SemanticCache
from generation.models import GenerationResult
from observability.trace_models import SpanStatus

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_qdrant() -> MagicMock:
    """Mock QdrantClient with empty collections list."""
    client = MagicMock()
    collections = MagicMock()
    collections.collections = []
    client.get_collections.return_value = collections
    return client


@pytest.fixture()
def gen_result() -> GenerationResult:
    """Minimal valid GenerationResult for testing."""
    return GenerationResult(
        question="What was Apple's revenue?",
        answer="Apple's revenue was $94.9B in Q4 2024.",
        citations=[],
        model="gpt-4.1-nano",
        prompt_tokens=800,
        completion_tokens=120,
        total_tokens=920,
        context_chunks_used=5,
        context_tokens_used=3200,
        latency_seconds=1.5,
        grounded=True,
        retrieval_failed=False,
    )


def _mock_embedding() -> np.ndarray:
    """Return a deterministic fake embedding vector."""
    return np.random.default_rng(42).random(1024).astype(np.float32)


# ── Collection setup ──────────────────────────────────────────────────────────


class TestCollectionSetup:
    """Verify cache creates or skips Qdrant collection."""

    def test_creates_collection_if_missing(self, mock_qdrant: MagicMock) -> None:
        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            SemanticCache(mock_qdrant)
            mock_qdrant.create_collection.assert_called_once()
            _, kwargs = mock_qdrant.create_collection.call_args
            assert kwargs["collection_name"] == "test_cache"

    def test_skips_if_collection_exists(self, mock_qdrant: MagicMock) -> None:
        existing = MagicMock()
        existing.name = "test_cache"
        collections = MagicMock()
        collections.collections = [existing]
        mock_qdrant.get_collections.return_value = collections

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"

            SemanticCache(mock_qdrant)
            mock_qdrant.create_collection.assert_not_called()


# ── Cache GET ─────────────────────────────────────────────────────────────────


class TestCacheGet:
    """Verify cache lookup — hits, misses, and errors."""

    @patch("cache.semantic_cache.SemanticCache._get_embedder")
    def test_hit_returns_generation_result(
        self,
        mock_get_embedder: MagicMock,
        mock_qdrant: MagicMock,
        gen_result: GenerationResult,
    ) -> None:
        # Mock embedder
        embedder = MagicMock()
        embedder.embed.return_value = iter([_mock_embedding()])
        mock_get_embedder.return_value = embedder

        # Mock Qdrant search returning a hit above threshold
        hit = MagicMock()
        hit.score = 0.98
        hit.payload = {"result": gen_result.to_dict()}
        mock_qdrant.search.return_value = [hit]

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            cache = SemanticCache(mock_qdrant)
            result, span = cache.get("What was Apple's revenue?")

            assert result is not None
            assert result.answer == gen_result.answer
            assert result.model == gen_result.model
            assert span.cache_hit is True
            assert span.similarity_score == 0.98
            assert span.threshold_used == 0.95
            assert span.status == SpanStatus.OK

    @patch("cache.semantic_cache.SemanticCache._get_embedder")
    def test_miss_returns_none(
        self,
        mock_get_embedder: MagicMock,
        mock_qdrant: MagicMock,
    ) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = iter([_mock_embedding()])
        mock_get_embedder.return_value = embedder

        mock_qdrant.search.return_value = []

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            cache = SemanticCache(mock_qdrant)
            result, span = cache.get("What was Apple's revenue?")

            assert result is None
            assert span.cache_hit is False
            assert span.status == SpanStatus.OK

    @patch("cache.semantic_cache.SemanticCache._get_embedder")
    def test_error_returns_none_and_error_span(
        self,
        mock_get_embedder: MagicMock,
        mock_qdrant: MagicMock,
    ) -> None:
        """Cache errors should fail open — return None, not raise."""
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("connection lost")
        mock_get_embedder.return_value = embedder

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            cache = SemanticCache(mock_qdrant)
            result, span = cache.get("test")

            assert result is None
            assert span.status == SpanStatus.ERROR


# ── Cache SET ─────────────────────────────────────────────────────────────────


class TestCacheSet:
    """Verify cache inserts."""

    @patch("cache.semantic_cache.SemanticCache._get_embedder")
    def test_set_upserts_to_qdrant(
        self,
        mock_get_embedder: MagicMock,
        mock_qdrant: MagicMock,
        gen_result: GenerationResult,
    ) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = iter([_mock_embedding()])
        mock_get_embedder.return_value = embedder

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            cache = SemanticCache(mock_qdrant)
            cache.set("What was Apple's revenue?", gen_result)

            mock_qdrant.upsert.assert_called_once()
            _, kwargs = mock_qdrant.upsert.call_args
            assert kwargs["collection_name"] == "test_cache"
            assert kwargs["wait"] is False
            assert len(kwargs["points"]) == 1

            point = kwargs["points"][0]
            assert isinstance(point, qmodels.PointStruct)
            assert point.payload["result"]["answer"] == gen_result.answer

    @patch("cache.semantic_cache.SemanticCache._get_embedder")
    def test_set_swallows_errors(
        self,
        mock_get_embedder: MagicMock,
        mock_qdrant: MagicMock,
        gen_result: GenerationResult,
    ) -> None:
        """SET errors should be swallowed — never block the response."""
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("boom")
        mock_get_embedder.return_value = embedder

        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = True
            mock_settings.cache.collection_name = "test_cache"
            mock_settings.cache.similarity_threshold = 0.95

            cache = SemanticCache(mock_qdrant)
            # Should not raise
            cache.set("test", gen_result)


# ── Disabled cache ────────────────────────────────────────────────────────────


class TestDisabledCache:
    """Verify disabled cache is a complete no-op."""

    def test_disabled_skips_setup(self, mock_qdrant: MagicMock) -> None:
        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = False

            SemanticCache(mock_qdrant)
            mock_qdrant.create_collection.assert_not_called()

    def test_disabled_get_returns_none(self, mock_qdrant: MagicMock) -> None:
        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = False

            cache = SemanticCache(mock_qdrant)
            result, span = cache.get("test")
            assert result is None
            assert span.cache_hit is False

    def test_disabled_set_is_noop(
        self, mock_qdrant: MagicMock, gen_result: GenerationResult
    ) -> None:
        with patch("cache.semantic_cache.settings") as mock_settings:
            mock_settings.cache.enabled = False

            cache = SemanticCache(mock_qdrant)
            cache.set("test", gen_result)
            mock_qdrant.upsert.assert_not_called()


# ── GenerationResult round-trip ───────────────────────────────────────────────


class TestGenerationResultRoundTrip:
    """Verify from_dict correctly reconstructs what to_dict serializes."""

    def test_round_trip(self, gen_result: GenerationResult) -> None:
        serialized = gen_result.to_dict()
        restored = GenerationResult.from_dict(serialized)

        assert restored.question == gen_result.question
        assert restored.answer == gen_result.answer
        assert restored.model == gen_result.model
        assert restored.prompt_tokens == gen_result.prompt_tokens
        assert restored.completion_tokens == gen_result.completion_tokens
        assert restored.grounded == gen_result.grounded
        assert restored.retrieval_failed == gen_result.retrieval_failed
        assert restored.context_chunks_used == gen_result.context_chunks_used

    def test_from_dict_with_empty_dict(self) -> None:
        """Gracefully handle an empty payload (corrupt cache entry)."""
        result = GenerationResult.from_dict({})
        assert result.question == ""
        assert result.answer == ""
        assert result.citations == []
