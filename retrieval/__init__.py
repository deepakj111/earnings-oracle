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

if TYPE_CHECKING:
    from query.models import TransformedQuery

# retrieval/__init__.py — add at the bottom of the file
from retrieval.reranker import warmup_reranker
from retrieval.searcher import (  # <-- Add _fetch_parent_texts here
    _fetch_parent_texts,
    search,
    warmup_bm25,
    warmup_embed_client,
)


def retrieve(
    query: TransformedQuery,
    qdrant_client: QdrantClient,
    metadata_filter: MetadataFilter | None = None,
) -> RetrievalResult:
    # 3a — Hybrid search + RRF (Returns small child chunks)
    candidates: list[SearchResult] = search(
        query=query,
        qdrant_client=qdrant_client,
        metadata_filter=metadata_filter,
    )

    # 3b — Cross-encoder reranking (Fast because it's only reading 128-token chunks)
    top_children: list[SearchResult] = rerank(
        query=query.original,
        candidates=candidates,
    )

    # 3c — Late Parent Fetch (Only fetching the final 5 parent chunks for the LLM)
    final_parents = _fetch_parent_texts(qdrant_client, top_children)

    return RetrievalResult(
        query=query.original,
        results=final_parents,
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
    "warmup_embed_client",
    "warmup_bm25",
    "warmup_reranker",
]
