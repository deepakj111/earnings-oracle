"""
Semantic Caching Layer using Qdrant.

Short-circuits the LLM pipeline by returning cached GenerationResult payloads
if the new query is semantically identical (cosine similarity > 0.98) to a previous query.
Includes TTL expiration, ticker-based invalidation, and hit-rate telemetry.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from config import settings
from generation.models import GenerationResult


class SemanticCache:
    """
    Qdrant-backed semantic cache for pipeline query responses with TTL and invalidation.
    """

    COLLECTION_NAME = "semantic_cache"

    def __init__(self, qdrant_url: str = settings.infra.qdrant_url) -> None:
        self.client = AsyncQdrantClient(url=qdrant_url, timeout=10, check_compatibility=False)
        self._initialized = False
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def _ensure_collection(self) -> None:
        if self._initialized:
            return

        exists = await self.client.collection_exists(self.COLLECTION_NAME)
        if exists:
            try:
                coll_info = await self.client.get_collection(self.COLLECTION_NAME)
                size = getattr(coll_info.config.params.vectors, "size", None)
                if size is not None and size != settings.embedding.vector_dim:
                    logger.warning(
                        f"Semantic cache has vector dimension {size}, "
                        f"but current model expects {settings.embedding.vector_dim}. Recreating..."
                    )
                    await self.client.delete_collection(self.COLLECTION_NAME)
                    exists = False
            except Exception as exc:
                logger.warning(f"Could not verify semantic cache vector dimensions: {exc}")

        if not exists:
            logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME} for Semantic Caching")
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.embedding.vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        self._initialized = True

    async def get_cached_response(
        self, query_vector: list[float], threshold: float = 0.98
    ) -> GenerationResult | None:
        """
        Search for a semantically equivalent query in the cache.
        Returns the cached GenerationResult if similarity >= threshold and unexpired.
        """
        try:
            await self._ensure_collection()

            hits = await self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=1,
                with_payload=True,
                score_threshold=threshold,
            )

            if hits.points:
                hit = hits.points[0]
                payload = hit.payload or {}

                # Check TTL expiration
                created_at = float(payload.get("created_at", 0))
                ttl_hours = getattr(settings.infra, "cache_ttl_hours", 24)
                if created_at > 0 and (time.time() - created_at) > (ttl_hours * 3600):
                    logger.info(
                        f"Semantic Cache EXPIRED for query: '{payload.get('query')}' (TTL {ttl_hours}h)"
                    )
                    self._evictions += 1
                    self._misses += 1
                    try:
                        await self.client.delete(
                            collection_name=self.COLLECTION_NAME,
                            points_selector=[hit.id],
                        )
                    except Exception as del_err:
                        logger.debug(f"Failed to evict expired cache entry: {del_err}")
                    return None

                self._hits += 1
                logger.info(
                    f"Semantic Cache HIT! Score: {hit.score:.4f} for cached query: '{payload.get('query')}'"
                )
                payload_json = payload.get("generation_result")
                if payload_json:
                    data = json.loads(payload_json)
                    return GenerationResult.from_dict(data)
            else:
                self._misses += 1

        except Exception as e:
            logger.warning(f"Semantic cache retrieval failed: {e}")
            self._misses += 1

        return None

    async def set_cached_response(
        self, query: str, query_vector: list[float], result: GenerationResult
    ) -> None:
        """
        Save a generated response to the semantic cache with metadata and timestamp.
        """
        try:
            await self._ensure_collection()

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query))
            tickers = result.unique_tickers

            await self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload={
                            "query": query,
                            "created_at": time.time(),
                            "tickers": tickers,
                            "generation_result": result.to_json(),
                        },
                    )
                ],
            )
            logger.debug(
                f"Saved response to Semantic Cache for query: '{query}' (tickers={tickers})"
            )
        except Exception as e:
            logger.warning(f"Semantic cache upsert failed: {e}")

    async def invalidate_ticker(self, ticker: str) -> int:
        """
        Invalidate all cached responses referencing a specific company ticker.
        Should be invoked when new SEC filings are ingested.
        """
        try:
            await self._ensure_collection()
            t_upper = ticker.upper().strip()
            filter_cond = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tickers",
                        match=models.MatchValue(value=t_upper),
                    )
                ]
            )
            await self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=models.FilterSelector(filter=filter_cond),
            )
            logger.info(f"Invalidated semantic cache entries for ticker={t_upper}")
            return 1
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for ticker {ticker}: {e}")
            return 0

    def stats(self) -> dict[str, Any]:
        """Return cache hit rate, eviction, and lookup metrics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "total_lookups": total,
            "hit_rate": round(hit_rate, 4),
        }
