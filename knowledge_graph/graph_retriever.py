# knowledge_graph/graph_retriever.py
"""
Graph-Fused Retrieval — Knowledge Graph Context Injection.

Runs after standard hybrid search + reranking to inject additional
context from the knowledge graph. This enables answering cross-document
and relational questions that pure vector search cannot handle.

Pipeline:
  1. Match entities from the user's question against the knowledge graph
  2. Traverse relationships to find related entities
  3. Collect chunk IDs associated with related entities
  4. Fetch those chunks from Qdrant as supplementary context
  5. Return as SearchResult objects with source="graph"

Design decisions:
  - Matching uses exact + fuzzy substring matching against entity names
  - Graph chunks are appended after reranked results (lower priority)
  - Chunks already present in reranked results are deduplicated
  - Max graph chunks is configurable (default 3)
  - Fail-open: graph errors never block the main retrieval pipeline
"""

from __future__ import annotations

import time

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from knowledge_graph.entity_store import EntityStore
from knowledge_graph.models import KnowledgeGraph
from observability.trace_models import GraphRetrievalSpan, SpanStatus
from retrieval.models import SearchResult

# Module-level cached graph (loaded once, reused across requests)
_cached_graph: KnowledgeGraph | None = None


def _load_graph() -> KnowledgeGraph:
    """Lazy-load the knowledge graph from disk (cached in module global)."""
    global _cached_graph
    if _cached_graph is None:
        store = EntityStore()
        _cached_graph = store.load()
    return _cached_graph


def _match_entities_from_question(
    question: str,
    graph: KnowledgeGraph,
) -> list[str]:
    """
    Match entity names from the user's question against the knowledge graph.

    Uses case-insensitive substring matching: if an entity's name
    (or any alias) appears in the question, it's considered a match.

    Returns a list of matched entity names (normalized lowercase).
    """
    question_lower = question.lower()
    matched: list[str] = []
    seen: set[str] = set()

    for entity in graph.entities.values():
        # Check entity name
        if entity.name in question_lower and entity.name not in seen:
            matched.append(entity.name)
            seen.add(entity.name)
            continue
        # Check aliases
        for alias in entity.aliases:
            if alias in question_lower and entity.name not in seen:
                matched.append(entity.name)
                seen.add(entity.name)
                break

    return matched


def _collect_related_chunk_ids(
    matched_entities: list[str],
    graph: KnowledgeGraph,
    existing_chunk_ids: set[str],
) -> list[str]:
    """
    Traverse the knowledge graph to find chunk IDs related to matched entities.

    1. Direct chunks: chunks where the matched entity was mentioned
    2. Related chunks: chunks of entities connected via relationships

    Deduplicates against existing_chunk_ids (already in retrieval results).
    """
    candidate_chunk_ids: list[str] = []
    seen: set[str] = set(existing_chunk_ids)

    for entity_name in matched_entities:
        # Direct entity chunks
        for chunk_id in graph.get_entity_chunk_ids(entity_name):
            if chunk_id not in seen:
                candidate_chunk_ids.append(chunk_id)
                seen.add(chunk_id)

        # Related entity chunks (one hop)
        for _rel, related_entity in graph.find_related(entity_name):
            if related_entity is None:
                continue
            for chunk_id in related_entity.chunk_ids:
                if chunk_id not in seen:
                    candidate_chunk_ids.append(chunk_id)
                    seen.add(chunk_id)

    return candidate_chunk_ids


def _fetch_chunks_by_ids(
    chunk_ids: list[str],
    qdrant_client: QdrantClient,
    max_chunks: int,
) -> list[SearchResult]:
    """Fetch specific chunks from Qdrant by their chunk_id field."""
    if not chunk_ids:
        return []

    # Only fetch up to max_chunks
    target_ids = chunk_ids[:max_chunks]

    try:
        scroll_result, _ = qdrant_client.scroll(
            collection_name=settings.embedding.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="chunk_id",
                        match=qmodels.MatchAny(any=target_ids),
                    )
                ]
            ),
            limit=max_chunks,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in scroll_result:
            if point.payload:
                results.append(
                    SearchResult.from_payload(
                        payload=point.payload,
                        rrf_score=0.0,  # graph chunks don't have RRF scores
                        source="graph",
                    )
                )
        return results
    except Exception as exc:
        logger.warning(f"Graph chunk fetch failed (fail-open): {exc}")
        return []


def graph_retrieve(
    question: str,
    existing_results: list[SearchResult],
    qdrant_client: QdrantClient,
) -> tuple[list[SearchResult], GraphRetrievalSpan]:
    """
    Graph-fused retrieval: inject relational context from the knowledge graph.

    Args:
        question: User's original question
        existing_results: Results from standard hybrid search + reranking
        qdrant_client: Qdrant client for fetching graph chunks

    Returns:
        Tuple of (additional graph SearchResults, GraphRetrievalSpan for tracing)
    """
    span = GraphRetrievalSpan()

    if not settings.knowledge_graph.retrieval_enabled:
        return [], span

    start_t = time.perf_counter()

    try:
        graph = _load_graph()
        if graph.entity_count == 0:
            span.latency_seconds = time.perf_counter() - start_t
            return [], span

        # 1. Match entities from question
        matched = _match_entities_from_question(question, graph)
        span.entities_matched = len(matched)
        span.matched_entity_names = matched[:10]  # cap for trace readability

        if not matched:
            span.latency_seconds = time.perf_counter() - start_t
            return [], span

        # 2. Traverse relationships to collect related chunk IDs
        existing_chunk_ids = {r.chunk_id for r in existing_results}
        related_chunk_ids = _collect_related_chunk_ids(matched, graph, existing_chunk_ids)
        span.relationships_traversed = len(related_chunk_ids)

        # 3. Fetch chunks from Qdrant
        max_chunks = settings.knowledge_graph.max_graph_chunks
        graph_results = _fetch_chunks_by_ids(related_chunk_ids, qdrant_client, max_chunks)
        span.chunks_injected = len(graph_results)

        span.latency_seconds = time.perf_counter() - start_t
        logger.info(
            f"[GraphRAG] entities={len(matched)} | "
            f"related_chunks={len(related_chunk_ids)} | "
            f"injected={len(graph_results)} | "
            f"latency={span.latency_seconds * 1000:.1f}ms"
        )
        return graph_results, span

    except Exception as exc:
        logger.warning(f"Graph retrieval failed (fail-open): {exc}")
        span.latency_seconds = time.perf_counter() - start_t
        span.status = SpanStatus.ERROR
        return [], span


def invalidate_cache() -> None:
    """Clear the cached graph (e.g., after re-ingestion)."""
    global _cached_graph
    _cached_graph = None
