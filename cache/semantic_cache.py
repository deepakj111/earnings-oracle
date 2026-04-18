# cache/semantic_cache.py
"""
Embedding-Based Semantic Cache.

Intercepts identical or highly similar user queries to instantly return
cached generation results, bypassing the LLM and search layers entirely.

Design decisions:
  - Fail-open: if the cache is unavailable or errors, the pipeline proceeds
    normally. A cache failure should never block the user.
  - The cache uses the same embedding model as the retrieval layer (BAAI/bge-
    large-en-v1.5) so that semantic similarity is measured consistently.
  - Qdrant is used as the cache backend because it is already a dependency
    and supports efficient ANN search with cosine distance.
  - Cache writes use wait=False to avoid adding latency to the response path.
"""

from __future__ import annotations

import time
import uuid

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from generation.models import GenerationResult
from observability.trace_models import SemanticCacheSpan, SpanStatus


class SemanticCache:
    """
    Qdrant-backed semantic caching layer.

    Sits before Layer 2 (Query Transformation) in the pipeline. If the
    incoming question is semantically near-identical to a cached question,
    the stored GenerationResult is returned instantly — zero LLM calls,
    zero retrieval, zero cost.

    Thread-safety: each method creates its own span and performs independent
    Qdrant calls. No mutable shared state.
    """

    def __init__(self, qdrant_client: QdrantClient) -> None:
        self.qdrant = qdrant_client
        self.enabled = settings.cache.enabled
        self.collection = settings.cache.collection_name
        self.threshold = settings.cache.similarity_threshold
        self._embed_client: object | None = None

        if self.enabled:
            self._setup_collection()

    def _get_embedder(self) -> object:
        """Lazy-load the text embedder (same model as retrieval layer)."""
        if self._embed_client is None:
            from fastembed import TextEmbedding

            self._embed_client = TextEmbedding(model_name=settings.embedding.model)
        return self._embed_client

    def _setup_collection(self) -> None:
        """Ensure the cache collection exists in Qdrant."""
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == self.collection for c in collections)

            if not exists:
                logger.info(f"Creating semantic cache collection: {self.collection}")
                self.qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=1024,  # BAAI/bge-large-en-v1.5
                        distance=qmodels.Distance.COSINE,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=100000,
                    ),
                )
        except Exception as exc:
            logger.warning(f"Failed to setup semantic cache: {exc}")
            self.enabled = False

    # ── Public interface ───────────────────────────────────────────────────

    def get(self, question: str) -> tuple[GenerationResult | None, SemanticCacheSpan]:
        """
        Check if an answer for this question is already cached.

        Returns:
            Tuple of (GenerationResult if hit else None, SemanticCacheSpan).
        """
        span = SemanticCacheSpan(threshold_used=self.threshold)
        if not self.enabled:
            return None, span

        start_t = time.perf_counter()
        try:
            embedder = self._get_embedder()
            # fastembed yields a generator of numpy arrays
            embedding = next(embedder.embed([question]))
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=1,
                score_threshold=self.threshold,
            )

            latency = time.perf_counter() - start_t
            span.latency_seconds = latency

            if results:
                hit = results[0]
                span.cache_hit = True
                span.similarity_score = hit.score
                payload = hit.payload or {}

                # Reconstruct GenerationResult from the cached payload
                gen_result = GenerationResult.from_dict(payload.get("result", {}))

                logger.info(
                    f"[Cache Hit] sim={hit.score:.4f} | "
                    f"latency={latency * 1000:.1f}ms | "
                    f"q={question!r:.60}"
                )
                return gen_result, span

            # Cache miss
            span.cache_hit = False
            logger.debug(
                f"[Cache Miss] latency={time.perf_counter() - start_t:.4f}s | q={question!r:.60}"
            )
            return None, span

        except Exception as exc:
            logger.warning(f"Semantic cache GET failed (fail-open): {exc}")
            span.latency_seconds = time.perf_counter() - start_t
            span.status = SpanStatus.ERROR
            return None, span

    def set(self, question: str, result: GenerationResult) -> None:
        """
        Store a successful generation result in the cache.

        Uses wait=False for non-blocking writes — the pipeline
        returns the answer to the user immediately while Qdrant
        commits the cache entry in the background.
        """
        if not self.enabled:
            return

        try:
            embedder = self._get_embedder()
            embedding = next(embedder.embed([question]))
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

            payload = {"result": result.to_dict()}
            point_id = str(uuid.uuid4())

            self.qdrant.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
                wait=False,  # background insert — no latency added to response
            )
            logger.debug(f"[Cache Set] async insert queued for q={question!r:.60}")
        except Exception as exc:
            logger.warning(f"Semantic cache SET failed (fail-open): {exc}")
