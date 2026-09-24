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

from config import settings
from retrieval.models import MetadataFilter, RetrievalResult, SearchResult
from retrieval.reranker import rerank, warmup_reranker
from retrieval.searcher import (
    _fetch_parent_texts,
    search,
    warmup_bm25,
    warmup_embed_client,
)

if TYPE_CHECKING:
    from query.models import TransformedQuery


def retrieve(
    query: TransformedQuery,
    qdrant_client: QdrantClient,
    metadata_filter: MetadataFilter | None = None,
    pre_computed_query_vector: list[float] | None = None,
) -> RetrievalResult:
    """Perform hybrid retrieval, cross-encoder reranking, and fetch context parent chunks."""
    # 3a — Hybrid search + RRF (BM25 + Qdrant)
    candidates: list[SearchResult] = search(
        query=query,
        qdrant_client=qdrant_client,
        metadata_filter=metadata_filter,
        pre_computed_query_vector=pre_computed_query_vector,
    )

    # 3b — Fetch parent texts so cross-encoder evaluates full table context
    candidates_with_parents = _fetch_parent_texts(qdrant_client, candidates)

    # 3c — Cross-encoder reranking (evaluates full table text)
    top_children: list[SearchResult] = rerank(
        query=query.original,
        candidates=candidates_with_parents,
    )

    # 3d — Knowledge Graph context injection (GraphRAG)
    try:
        from knowledge_graph.graph_retriever import graph_retrieve as _graph_retrieve

        graph_chunks, _graph_span = _graph_retrieve(
            question=query.original,
            existing_results=top_children,
            qdrant_client=qdrant_client,
            metadata_filter=metadata_filter,
        )
        if graph_chunks:
            top_children = top_children + graph_chunks
    except Exception as exc:
        logger.debug(f"Graph retrieval skipped (fail-open): {exc}")

    # 3e — Structured SEC GAAP Facts Injection (Dual-Path Ground Truth)
    if metadata_filter and metadata_filter.ticker:
        try:
            from config.companies import CompanyRegistry
            from ingestion.facts_store import FactStore

            facts = FactStore.query(
                ticker=metadata_filter.ticker,
                fiscal_year=metadata_filter.year,
                quarter=metadata_filter.quarter,
            )
            if facts:
                facts_text = FactStore.format_as_context(facts)
                prof = CompanyRegistry.get_company(metadata_filter.ticker)
                company_name = prof.name if prof else facts[0].ticker
                facts_chunk = SearchResult(
                    chunk_id=f"fact_{metadata_filter.ticker}_{metadata_filter.year or 'all'}",
                    parent_id=None,
                    text=facts_text,
                    parent_text=facts_text,
                    rrf_score=1.0,
                    rerank_score=1.0,
                    ticker=metadata_filter.ticker,
                    company=company_name,
                    date=facts[0].period_end,
                    year=facts[0].fiscal_year,
                    quarter=facts[0].quarter,
                    fiscal_period=f"{facts[0].quarter} {facts[0].fiscal_year}",
                    section_title="Authoritative SEC Financial Statements",
                    doc_type="10-K/10-Q GAAP Facts",
                    source="facts",
                )
                top_children.insert(0, facts_chunk)
        except Exception as exc:
            logger.debug(f"Fact injection skipped: {exc}")

    return RetrievalResult(
        query=query.original,
        results=top_children,
        reranked=settings.reranker.enabled,
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
