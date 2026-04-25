"""
Layer 3a — Hybrid Retrieval: BM25 + Qdrant dense search with RRF fusion.

Pipeline:
  TransformedQuery (from Layer 2)
      │
      ├── hyde_document      → 1× Qdrant dense search
      ├── multi_queries[0..n]→ n× Qdrant dense + n× BM25
      └── stepback_query     → 1× Qdrant dense + 1× BM25
                │
                ▼
          Deduplicate by chunk_id (track all sources per chunk)
                │
                ▼
          RRF fusion — score = Σ 1/(k + rank_i) across all result lists
                │
                ▼
          Fetch parent chunk text from Qdrant (top_k_pre_rerank chunks)
                │
                ▼
          → list[SearchResult]  (consumed by reranker.py)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from retrieval.models import MetadataFilter, SearchResult

if TYPE_CHECKING:
    from query.models import TransformedQuery

# ── Module-level fastembed client (lazy-initialised on first search) ───────────
_embed_client: object | None = None


def _get_embed_client() -> object:
    global _embed_client
    if _embed_client is None:
        from fastembed import TextEmbedding

        _embed_client = TextEmbedding(model_name=settings.embedding.model)
        logger.info(f"Embedding model loaded for retrieval: {settings.embedding.model}")
    return _embed_client


# ── BM25 index helpers ─────────────────────────────────────────────────────────

_BM25_INDEX_PATH = Path("data/bm25_index.pkl")
_BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")

_bm25_index: object | None = None
_bm25_corpus: list[dict] | None = None


def _load_bm25() -> tuple[object, list[dict]]:
    """
    Lazy-load the BM25 index and corpus from disk.
    Called once on first BM25 search; results are cached in module globals.
    Raises FileNotFoundError with a clear message if ingestion has not been run.
    """
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None and _bm25_corpus is not None:
        return _bm25_index, _bm25_corpus

    if not _BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {_BM25_INDEX_PATH}. "
            "Run `poetry run python -m ingestion.pipeline` first."
        )
    if not _BM25_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {_BM25_CORPUS_PATH}. "
            "Run `poetry run python -m ingestion.pipeline` first."
        )

    with open(_BM25_INDEX_PATH, "rb") as f:
        _bm25_index = pickle.load(f)  # nosec B301
    with open(_BM25_CORPUS_PATH, "rb") as f:
        _bm25_corpus = pickle.load(f)  # nosec B301

    logger.info(f"BM25 index loaded: {len(_bm25_corpus)} corpus entries.")
    return _bm25_index, _bm25_corpus


# ── Embedding ──────────────────────────────────────────────────────────────────


def _embed(text: str) -> list[float]:
    """Embed a single text string using fastembed. Returns a flat float list."""
    client = _get_embed_client()
    vectors = list(client.embed([text]))  # type: ignore[attr-defined]
    return vectors[0].tolist()


# ── Qdrant filter builder ──────────────────────────────────────────────────────


def _build_qdrant_filter(mf: MetadataFilter | None) -> qmodels.Filter | None:
    """
    Convert a MetadataFilter into a Qdrant must-match filter.
    Returns None if no filter fields are set or filtering is disabled in config.
    """
    if mf is None or not settings.retrieval.metadata_filter_enabled:
        return None

    conditions: list[qmodels.FieldCondition] = []

    if mf.ticker:
        conditions.append(
            qmodels.FieldCondition(
                key="ticker",
                match=qmodels.MatchValue(value=mf.ticker),
            )
        )
    if mf.year:
        conditions.append(
            qmodels.FieldCondition(
                key="year",
                match=qmodels.MatchValue(value=mf.year),
            )
        )
    if mf.quarter:
        conditions.append(
            qmodels.FieldCondition(
                key="quarter",
                match=qmodels.MatchValue(value=mf.quarter),
            )
        )

    return qmodels.Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


# ── Qdrant dense search ────────────────────────────────────────────────────────


def _qdrant_search(
    client: QdrantClient,
    query_text: str,
    top_k: int,
    qdrant_filter: qmodels.Filter | None,
) -> list[dict]:
    """
    Search Qdrant with a single query string.
    Returns a list of payload dicts, ordered by cosine similarity (best first).
    """
    vector = _embed(query_text)

    hits = client.query_points(
        collection_name=settings.embedding.collection_name,
        query=vector,  # <-- Changed from query_vector to query
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    return [hit.payload for hit in hits.points if hit.payload]


# ── BM25 keyword search ────────────────────────────────────────────────────────


def _bm25_search(
    query_text: str,
    top_k: int,
    metadata_filter: MetadataFilter | None,
) -> list[dict]:
    bm25, corpus = _load_bm25()
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)  # type: ignore[attr-defined]

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for idx, _score in scored:
        if idx >= len(corpus):
            continue
        entry: dict = corpus[idx]

        if metadata_filter and settings.retrieval.metadata_filter_enabled:
            if metadata_filter.ticker and entry.get("ticker") != metadata_filter.ticker:
                continue
            if metadata_filter.year and entry.get("year") != metadata_filter.year:
                continue
            if metadata_filter.quarter and entry.get("quarter") != metadata_filter.quarter:
                continue

        results.append(entry)
        if len(results) >= top_k:
            break

    return results


# ── RRF fusion ─────────────────────────────────────────────────────────────────


def _rrf_fuse(
    result_lists: list[list[str]],
    all_payloads: dict[str, dict],
    k: int,
) -> list[tuple[str, float, str]]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Args:
        result_lists : list of ordered chunk_id lists (each is one search result)
        all_payloads : chunk_id → payload dict (built up during search)
        k            : RRF constant (default 60 from original paper)

    Returns:
        List of (chunk_id, rrf_score, source) sorted by rrf_score descending.
        source is "dense", "bm25", or "both".
    """
    rrf_scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for result_list_with_source in result_lists:
        chunk_ids, source_label = result_list_with_source
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in sources:
                sources[chunk_id] = set()
            sources[chunk_id].add(source_label)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        (chunk_id, score, "both" if len(sources[chunk_id]) > 1 else next(iter(sources[chunk_id])))
        for chunk_id, score in fused
    ]


# ── Parent chunk fetch ─────────────────────────────────────────────────────────


def _fetch_parent_texts(
    client: QdrantClient,
    results: list[SearchResult],
) -> list[SearchResult]:
    """
    For each SearchResult that has a parent_id, fetch the full parent chunk
    text from Qdrant and replace parent_text with it.

    Parent chunks contain ~512 tokens of context vs the child's ~128 tokens.
    This is the core of the parent/child architecture — retrieve small (child)
    for precision, read large (parent) for context in generation.

    Results without a parent_id (tables, standalone chunks) keep their own text.
    """
    if not settings.retrieval.parent_fetch_enabled:
        return results

    # Collect unique parent IDs that need fetching
    parent_ids_needed = {r.parent_id for r in results if r.parent_id is not None}

    if not parent_ids_needed:
        return results

    # Fetch all parent payloads in one batch call
    parent_map: dict[str, str] = {}
    try:
        scroll_result, _ = client.scroll(
            collection_name=settings.embedding.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="chunk_id",
                        match=qmodels.MatchAny(any=list(parent_ids_needed)),
                    )
                ]
            ),
            limit=len(parent_ids_needed) + 10,
            with_payload=True,
        )
        for point in scroll_result:
            if point.payload:
                cid = point.payload.get("chunk_id", "")
                text = point.payload.get("text", "")
                if cid:
                    parent_map[cid] = text
    except Exception as e:
        logger.warning(f"Parent fetch failed (using child text as fallback): {e}")
        return results

    # Patch parent_text onto each result
    for r in results:
        if r.parent_id and r.parent_id in parent_map:
            r.parent_text = parent_map[r.parent_id]

    fetched = sum(1 for r in results if r.parent_id and r.parent_id in parent_map)
    logger.debug(f"Parent fetch: {fetched}/{len(results)} chunks upgraded to parent context.")
    return results


# ── Public search function ─────────────────────────────────────────────────────


def search(
    query: TransformedQuery,
    qdrant_client: QdrantClient,
    metadata_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    """
    Run the full hybrid search for a TransformedQuery.

    1. Dense search  : hyde_document + all_retrieval_queries → Qdrant
    2. Sparse search : all_retrieval_queries → BM25
    3. RRF fusion    : merge and score all result lists
    4. Parent fetch  : replace child text with full parent context

    Returns top_k_pre_rerank SearchResults, sorted by RRF score descending.
    These are passed directly into reranker.rerank().
    """
    cfg = settings.retrieval
    top_k_dense = cfg.top_k_dense
    top_k_bm25 = cfg.top_k_bm25
    top_k_pre = settings.reranker.top_k_pre_rerank
    qdrant_filter = _build_qdrant_filter(metadata_filter)

    all_payloads: dict[str, dict] = {}
    rrf_input: list[tuple[list[str], str]] = []

    # ── 1. Dense search: HyDE document ────────────────────────────────────────
    if query.hyde_document:
        try:
            hits = _qdrant_search(qdrant_client, query.hyde_document, top_k_dense, qdrant_filter)
            ids = []
            for p in hits:
                cid = p.get("chunk_id", "")
                if cid:
                    all_payloads[cid] = p
                    ids.append(cid)
            rrf_input.append((ids, "dense"))
            logger.debug(f"HyDE dense search: {len(ids)} hits")
        except Exception as e:
            logger.warning(f"HyDE dense search failed: {e}")

    # ── 2. Dense + BM25 search: all_retrieval_queries ─────────────────────────
    for q_text in query.all_retrieval_queries:
        # Dense
        try:
            hits = _qdrant_search(qdrant_client, q_text, top_k_dense, qdrant_filter)
            ids = []
            for p in hits:
                cid = p.get("chunk_id", "")
                if cid:
                    all_payloads[cid] = p
                    ids.append(cid)
            rrf_input.append((ids, "dense"))
        except Exception as e:
            logger.warning(f"Dense search failed for query '{q_text[:60]}': {e}")

        # BM25
        try:
            hits = _bm25_search(q_text, top_k_bm25, metadata_filter)
            ids = []
            for entry in hits:
                cid = entry.get("chunk_id", "")
                if cid:
                    # BM25 corpus entries don't have all payload fields —
                    # merge with existing Qdrant payload if available
                    if cid not in all_payloads:
                        all_payloads[cid] = entry
                    ids.append(cid)
            rrf_input.append((ids, "bm25"))
        except Exception as e:
            logger.warning(f"BM25 search failed for query '{q_text[:60]}': {e}")

    if not rrf_input:
        logger.error("All search variants failed — returning empty results.")
        return []

    # ── 3. RRF fusion ──────────────────────────────────────────────────────────
    fused = _rrf_fuse(rrf_input, all_payloads, k=cfg.rrf_k_constant)  # type: ignore[arg-type]
    logger.info(
        f"RRF fusion: {len(all_payloads)} unique chunks → "
        f"top {min(top_k_pre, len(fused))} passed to reranker"
    )

    # Build SearchResult objects for top_k_pre_rerank candidates
    candidates: list[SearchResult] = []
    for chunk_id, rrf_score, source in fused[:top_k_pre]:
        payload = all_payloads.get(chunk_id, {})
        if not payload:
            continue
        candidates.append(SearchResult.from_payload(payload, rrf_score, source))

    return candidates


def warmup_embed_client() -> None:
    """Pre-load the fastembed model into memory. Safe to call multiple times."""
    _get_embed_client()


def warmup_bm25() -> None:
    """Pre-load the BM25 index from disk into memory. Safe to call multiple times."""
    _load_bm25()
