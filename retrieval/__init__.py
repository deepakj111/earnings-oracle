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

from loguru import logger
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
    """Perform hybrid retrieval, cross-encoder reranking, and fetch context parent chunks."""
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

    # 3c — Knowledge Graph context injection (GraphRAG)
    try:
        from knowledge_graph.graph_retriever import graph_retrieve as _graph_retrieve

        graph_chunks, _graph_span = _graph_retrieve(
            question=query.original,
            existing_results=top_children,
            qdrant_client=qdrant_client,
        )
        if graph_chunks:
            top_children = top_children + graph_chunks
    except Exception as exc:
        logger.debug(f"Graph retrieval skipped (fail-open): {exc}")  # nosec B110

    # 3d — Late Parent Fetch (Only fetching the final 5 parent chunks for the LLM)
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
