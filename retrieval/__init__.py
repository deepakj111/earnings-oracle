"""
Layer 3 — Hybrid Retrieval.

Public API:

    from retrieval import retrieve
    from retrieval.models import MetadataFilter, RetrievalResult

    result = retrieve(
        query=transformed_query,          # TransformedQuery from Layer 2
        qdrant_client=client,
        metadata_filter=MetadataFilter(ticker="AAPL", year=2024),
    )
    print(result.summary())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from retrieval.models import MetadataFilter, RetrievalResult, SearchResult
from retrieval.reranker import rerank
from retrieval.searcher import search

if TYPE_CHECKING:
    from query.models import TransformedQuery


def retrieve(
    query: TransformedQuery,
    qdrant_client: QdrantClient,
    metadata_filter: MetadataFilter | None = None,
) -> RetrievalResult:
    """
    Full Layer 3 pipeline: search → rerank → RetrievalResult.

    Args:
        query           : TransformedQuery from Layer 2 (HyDE + multi-query + step-back)
        qdrant_client   : connected QdrantClient instance
        metadata_filter : optional ticker/year/quarter scoping

    Returns:
        RetrievalResult containing top-k ranked chunks ready for generation.
    """
    # 3a — Hybrid search + RRF + parent fetch
    candidates: list[SearchResult] = search(
        query=query,
        qdrant_client=qdrant_client,
        metadata_filter=metadata_filter,
    )

    # 3b — Cross-encoder reranking
    final: list[SearchResult] = rerank(
        query=query.original,
        candidates=candidates,
    )

    return RetrievalResult(
        query=query.original,
        results=final,
        reranked=True,
        total_candidates=len(candidates),
        metadata_filter=metadata_filter,
        failed_techniques=list(query.failed_techniques),
    )


__all__ = [
    "retrieve",
    "MetadataFilter",
    "RetrievalResult",
    "SearchResult",
]
