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
    if not candidates:
        return []

    top_k_final = settings.retrieval.top_k_final

    if not settings.reranker.enabled:
        logger.debug("Reranker disabled — returning top-k by RRF score.")
        for r in candidates:
            r.rerank_score = r.rrf_score
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        return candidates[:top_k_final]

    passages = [{"id": i, "text": r.parent_text or r.text} for i, r in enumerate(candidates)]

    try:
        ranker, rerank_request_cls = _get_ranker()  # moved inside try
        request = rerank_request_cls(query=query, passages=passages)
        reranked = ranker.rerank(request)

        id_to_score: dict[int, float] = {item["id"]: float(item["score"]) for item in reranked}
        for i, result in enumerate(candidates):
            result.rerank_score = id_to_score.get(i, float("-inf"))

        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        top = candidates[:top_k_final]

        logger.info(
            f"Reranking: {len(candidates)} candidates → {len(top)} results "
            f"(top score: {top[0].rerank_score:.4f})"
        )
        return top

    except Exception as e:
        logger.warning(f"Reranking failed ({e}) — falling back to RRF order.")
        for r in candidates:
            r.rerank_score = r.rrf_score
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        return candidates[:top_k_final]
