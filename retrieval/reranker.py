"""
Layer 3b — Cross-encoder reranking with FlashRank.

Takes the top_k_pre_rerank SearchResults from searcher.py and scores each
(query, parent_text) pair with a local cross-encoder model, returning only
the top_k_final results sorted by rerank score.

Model: ms-marco-MiniLM-L-12-v2
  - ~66MB ONNX model, fully local, no API cost
  - Trained on MS MARCO passage ranking
  - ~8-15ms for 20 candidates on CPU

When reranker.enabled = False (via RAG_RERANKER_ENABLED=false env var),
results are returned as-is sorted by RRF score — useful for ablation testing.
"""

from __future__ import annotations

from loguru import logger

from config import settings
from retrieval.models import SearchResult

# ── FlashRank client (lazy-initialised on first rerank call) ───────────────────
_ranker: object | None = None
_rerank_request_cls: type | None = None


def _get_ranker() -> tuple[object, object]:
    global _ranker, _rerank_request_cls
    if _ranker is None:
        from flashrank import Ranker, RerankRequest

        _ranker = Ranker(model_name=settings.reranker.model)
        _rerank_request_cls = RerankRequest
        logger.info(f"FlashRank reranker loaded: {settings.reranker.model}")
    return _ranker, _rerank_request_cls


# ── Public rerank function ─────────────────────────────────────────────────────


def rerank(
    query: str,
    candidates: list[SearchResult],
) -> list[SearchResult]:
    """Score search candidates against the original query using a cross-encoder model."""
    if not candidates:
        return []

    top_k_final = settings.retrieval.top_k_final

    if not settings.reranker.enabled:
        logger.debug("Reranker disabled — returning top-k by RRF score.")
        for r in candidates:
            r.rerank_score = r.rrf_score
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        return candidates[:top_k_final]

    passages = [
        {"id": i, "text": (r.parent_text or r.text)[:1200]} for i, r in enumerate(candidates)
    ]

    try:
        ranker, rerank_request_cls = _get_ranker()  # moved inside try
        request = rerank_request_cls(query=query, passages=passages)  # type: ignore[operator]
        reranked = ranker.rerank(request)  # type: ignore[attr-defined]
        id_to_score: dict[int, float] = {item["id"]: float(item["score"]) for item in reranked}

        if len(candidates) == 1:
            candidates[0].rerank_score = id_to_score.get(0, 0.0)
            return candidates

        raw_scores = [id_to_score.get(i, 0.0) for i in range(len(candidates))]
        min_ce, max_ce = min(raw_scores), max(raw_scores)
        range_ce = (max_ce - min_ce) if max_ce > min_ce else 1.0

        rrf_scores = [r.rrf_score for r in candidates]
        min_rrf, max_rrf = min(rrf_scores), max(rrf_scores)
        range_rrf = (max_rrf - min_rrf) if max_rrf > min_rrf else 1.0

        for i, result in enumerate(candidates):
            raw_ce = id_to_score.get(i, min_ce)
            norm_ce = (raw_ce - min_ce) / range_ce
            norm_rrf = (result.rrf_score - min_rrf) / range_rrf
            # Soft score interpolation: 65% cross-encoder + 35% RRF confidence
            result.rerank_score = 0.65 * norm_ce + 0.35 * norm_rrf

        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        top = candidates[:top_k_final]

        logger.info(
            f"Reranking: {len(candidates)} candidates → {len(top)} results "
            f"(top blended score: {top[0].rerank_score:.4f})"
        )
        return top

    except Exception as e:
        logger.warning(f"Reranking failed ({e}) — falling back to RRF order.")
        for r in candidates:
            r.rerank_score = r.rrf_score
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        return candidates[:top_k_final]


def warmup_reranker() -> None:
    """Pre-load the FlashRank cross-encoder into memory. Safe to call multiple times."""
    _get_ranker()
