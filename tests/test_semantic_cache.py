"""
Unit tests for retrieval/semantic_cache.py
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import settings
from generation.models import GenerationResult
from retrieval.semantic_cache import SemanticCache


@pytest.fixture
def mock_gen_result() -> GenerationResult:
    return GenerationResult(
        question="What was Apple's revenue?",
        answer="Apple reported revenue of $94.9B [1].",
        citations=[],
        grounded=True,
        retrieval_failed=False,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        latency_seconds=1.5,
        model="gpt-5-mini",
        context_tokens_used=500,
        context_chunks_used=2,
    )


@pytest.mark.asyncio
class TestSemanticCache:
    async def test_ensure_collection_creates_when_missing(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=False)
        cache.client.create_collection = AsyncMock()

        await cache._ensure_collection()
        cache.client.create_collection.assert_awaited_once()
        assert cache._initialized is True

    async def test_ensure_collection_skips_when_exists(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.create_collection = AsyncMock()

        await cache._ensure_collection()
        cache.client.create_collection.assert_not_awaited()
        assert cache._initialized is True

    async def test_set_and_get_cached_response_hit(self, mock_gen_result: GenerationResult) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.upsert = AsyncMock()

        dummy_vector = [0.1] * 1536
        await cache.set_cached_response(
            query="What was Apple's revenue?",
            query_vector=dummy_vector,
            result=mock_gen_result,
        )
        cache.client.upsert.assert_awaited_once()

        # Simulate cache HIT
        mock_point = MagicMock()
        mock_point.id = "test-point-id"
        mock_point.score = 0.99
        mock_point.payload = {
            "query": "What was Apple's revenue?",
            "created_at": time.time(),
            "tickers": ["AAPL"],
            "generation_result": mock_gen_result.to_json(),
        }

        mock_query_response = MagicMock()
        mock_query_response.points = [mock_point]
        cache.client.query_points = AsyncMock(return_value=mock_query_response)

        cached = await cache.get_cached_response(query_vector=dummy_vector, threshold=0.98)
        assert cached is not None
        assert cached.question == mock_gen_result.question
        assert cached.answer == mock_gen_result.answer
        assert cache._hits == 1
        assert cache.stats()["hit_rate"] == 1.0

    async def test_get_cached_response_miss(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)

        mock_query_response = MagicMock()
        mock_query_response.points = []
        cache.client.query_points = AsyncMock(return_value=mock_query_response)

        dummy_vector = [0.1] * 1536
        cached = await cache.get_cached_response(query_vector=dummy_vector, threshold=0.98)
        assert cached is None
        assert cache._misses == 1

    async def test_get_cached_response_expired_ttl(self, mock_gen_result: GenerationResult) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.delete = AsyncMock()

        # Create expired point (created 48 hours ago, default TTL is 24h)
        mock_point = MagicMock()
        mock_point.id = "expired-id"
        mock_point.score = 0.99
        mock_point.payload = {
            "query": "Old query",
            "created_at": time.time() - (48 * 3600),
            "tickers": ["AAPL"],
            "generation_result": mock_gen_result.to_json(),
        }

        mock_query_response = MagicMock()
        mock_query_response.points = [mock_point]
        cache.client.query_points = AsyncMock(return_value=mock_query_response)

        dummy_vector = [0.1] * 1536
        cached = await cache.get_cached_response(query_vector=dummy_vector, threshold=0.98)
        assert cached is None
        assert cache._evictions == 1
        cache.client.delete.assert_awaited_once()

    async def test_invalidate_ticker(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.delete = AsyncMock()

        count = await cache.invalidate_ticker("AAPL")
        assert count == 1
        cache.client.delete.assert_awaited_once()

    async def test_stats_metrics(self) -> None:
        cache = SemanticCache()
        cache._hits = 8
        cache._misses = 2
        cache._evictions = 1
        stats = cache.stats()
        assert stats["hits"] == 8
        assert stats["misses"] == 2
        assert stats["evictions"] == 1
        assert stats["total_lookups"] == 10
        assert stats["hit_rate"] == 0.8

    async def test_dimension_mismatch_recreates_collection(self) -> None:
        from types import SimpleNamespace

        cache = SemanticCache()
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True

        coll_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=settings.embedding.vector_dim + 100)
                )
            )
        )
        mock_client.get_collection.return_value = coll_info
        cache.client = mock_client

        await cache._ensure_collection()
        assert mock_client.delete_collection.await_count == 1
        assert mock_client.create_collection.await_count == 1

    async def test_get_cached_response_exception_handled(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.query_points = AsyncMock(side_effect=RuntimeError("Qdrant unavailable"))

        cached = await cache.get_cached_response([0.1] * 1536)
        assert cached is None
        assert cache._misses == 1

    async def test_set_cached_response_exception_handled(
        self, mock_gen_result: GenerationResult
    ) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.upsert = AsyncMock(side_effect=RuntimeError("Qdrant write failed"))

        # Should not raise exception
        await cache.set_cached_response("What was Apple's revenue?", [0.1] * 1536, mock_gen_result)

    async def test_invalidate_ticker_exception_handled(self) -> None:
        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)
        cache.client.delete = AsyncMock(side_effect=RuntimeError("Delete failed"))

        result = await cache.invalidate_ticker("AAPL")
        assert result == 0

    async def test_expired_entry_eviction_error_handled(self) -> None:
        from types import SimpleNamespace

        cache = SemanticCache()
        cache.client = MagicMock()
        cache.client.collection_exists = AsyncMock(return_value=True)

        expired_hit = SimpleNamespace(
            id="point-123",
            score=0.99,
            payload={
                "query": "old query",
                "created_at": time.time() - 999999,
                "generation_result": "{}",
            },
        )
        cache.client.query_points = AsyncMock(return_value=SimpleNamespace(points=[expired_hit]))
        cache.client.delete = AsyncMock(side_effect=RuntimeError("Eviction delete failed"))

        result = await cache.get_cached_response([0.1] * 1536)
        assert result is None
        assert cache._evictions == 1
