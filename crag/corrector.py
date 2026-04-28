# crag/corrector.py
"""
Layer 5 — Corrective RAG (CRAG) orchestrator.

Decision flow:
  ┌────────────────────────────────────────────────────────────────┐
  │  GenerationResult (from Layer 4)                               │
  │         │                                                      │
  │  grounded=True AND grade_even_if_grounded=False?               │
  │         │ YES → return CORRECT immediately (fast path)         │
  │         │ NO  ↓                                               │
  │  Grade each retrieved chunk with RelevanceGrader               │
  │         │                                                      │
  │  relevance_ratio >= high_threshold → CORRECT                   │
  │  relevance_ratio <= low_threshold  → INCORRECT (web only)      │
  │  otherwise                         → AMBIGUOUS (web + local)   │
  │         │                                                      │
  │  CORRECT → return original result (no re-generation)           │
  │  INCORRECT / AMBIGUOUS → WebSearchClient.search()              │
  │         │                                                      │
  │  Build corrected RetrievalResult                               │
  │  INCORRECT : web results only                                  │
  │  AMBIGUOUS : relevant local chunks + web results               │
  │         │                                                      │
  │  Generator.generate() → new GenerationResult                   │
  │         │                                                      │
  │  Return CRAGResult with diagnostics                            │
  └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from hashlib import md5

from loguru import logger

from config import settings as _settings
from crag.grader import RelevanceGrader
from crag.models import CRAGAction, CRAGResult, WebSearchResult
from crag.web_search import WebSearchClient
from generation.generator import Generator
from generation.models import GenerationResult
from retrieval.models import RetrievalResult, SearchResult

_cfg = _settings.crag


# ── Helpers ────────────────────────────────────────────────────────────────────


def _web_to_search_result(w: WebSearchResult, idx: int) -> SearchResult:
    """
    Convert a WebSearchResult into a SearchResult so it flows through the
    existing Generator without modification.  Web results get chunk_id="web:<hash>".
    """
    url_hash = md5(w.url.encode(), usedforsecurity=False).hexdigest()[:8]  # nosec B324
    body = f"[WEB] {w.title}\n\n{w.snippet}"
    return SearchResult(
        chunk_id=f"web:{url_hash}",
        parent_id=None,
        text=body,
        parent_text=body,
        rrf_score=w.score,
        rerank_score=w.score,
        ticker="WEB",
        company=w.title[:50],
        date="",
        year=0,
        quarter="",
        fiscal_period="Web Search",
        section_title=w.title[:60],
        doc_type="web",
        source="web",
    )


def _build_corrected_result(
    original: RetrievalResult,
    relevant_local: list[SearchResult],
    web_results: list[WebSearchResult],
    action: CRAGAction,
) -> RetrievalResult:
    web_as_sr = [_web_to_search_result(w, i) for i, w in enumerate(web_results)]

    if action == CRAGAction.INCORRECT:
        combined = web_as_sr
    elif action == CRAGAction.AMBIGUOUS:
        combined = relevant_local + web_as_sr
    else:
        combined = original.results  # CORRECT — should not reach here

    return RetrievalResult(
        query=original.query,
        results=combined,
        reranked=False,
        total_candidates=len(combined),
        metadata_filter=original.metadata_filter,
        failed_techniques=original.failed_techniques,
    )


# ── Public corrector ───────────────────────────────────────────────────────────


class CRAGCorrector:
    """
    Layer 5: Corrective RAG.  Wraps the pipeline output with a quality loop.

    Stateless between calls — safe for concurrent use after initialisation.
    All heavy dependencies (grader, web client, generator) are lazy-loaded.

    Usage (standalone):
        corrector = CRAGCorrector()
        crag = corrector.correct(question, gen_result, retrieval_result)
        print(crag.final_result.answer)
        print(f"action={crag.action.value}  corrected={crag.was_corrected}")

    Usage (via pipeline):
        crag = pipeline.ask_with_crag("What was AAPL Q4 2024 revenue?")
    """

    def __init__(self) -> None:
        self._grader = RelevanceGrader()
        self._web = WebSearchClient()
        self._generator = Generator()
        logger.info(
            f"CRAGCorrector ready | "
            f"enabled={_cfg.enabled} | "
            f"thresholds=({_cfg.low_relevance_threshold}, {_cfg.high_relevance_threshold}) | "
            f"web={self._web.provider}"
        )

    def correct(
        self,
        question: str,
        generation_result: GenerationResult,
        retrieval_result: RetrievalResult,
    ) -> CRAGResult:
        """
        Run the CRAG correction loop.

        Args:
            question          : original user question (stripped)
            generation_result : output of Layer 4 generator
            retrieval_result  : output of Layer 3 retriever (for chunk grading)

        Returns:
            CRAGResult — action taken + final GenerationResult + diagnostics
        """
        t0 = time.perf_counter()

        # ── CRAG disabled ────────────────────────────────────────────────────
        if not _cfg.enabled:
            logger.debug("CRAG disabled — returning original result.")
            return CRAGResult(
                question=question,
                action=CRAGAction.CORRECT,
                original_result=generation_result,
                final_result=generation_result,
                latency_seconds=time.perf_counter() - t0,
            )

        # ── Fast path: already grounded and grading not forced ───────────────
        if generation_result.grounded and not _cfg.grade_even_if_grounded:
            logger.info("CRAG fast path: generation is grounded — action=CORRECT.")
            return CRAGResult(
                question=question,
                action=CRAGAction.CORRECT,
                original_result=generation_result,
                final_result=generation_result,
                latency_seconds=time.perf_counter() - t0,
            )

        # ── Empty retrieval → INCORRECT immediately ──────────────────────────
        chunks = retrieval_result.results
        if not chunks:
            logger.warning("CRAG: empty retrieval — INCORRECT, triggering web search.")
            grades = []
            action = CRAGAction.INCORRECT
        else:
            # ── Grade chunks ─────────────────────────────────────────────────
            t_grade = time.perf_counter()
            grades = self._grader.grade_chunks(question, chunks)
            grade_elapsed = time.perf_counter() - t_grade

            ratio = sum(g.relevant for g in grades) / len(grades)
            if ratio >= _cfg.high_relevance_threshold:
                action = CRAGAction.CORRECT
            elif ratio <= _cfg.low_relevance_threshold:
                action = CRAGAction.INCORRECT
            else:
                action = CRAGAction.AMBIGUOUS

            logger.info(
                f"CRAG grading: {sum(g.relevant for g in grades)}/{len(grades)} relevant | "
                f"ratio={ratio:.2f} → action={action.value} | {grade_elapsed:.2f}s"
            )

        # ── CORRECT: return original without re-generation ───────────────────
        if action == CRAGAction.CORRECT:
            return CRAGResult(
                question=question,
                action=action,
                original_result=generation_result,
                final_result=generation_result,
                relevance_grades=grades,
                latency_seconds=time.perf_counter() - t0,
            )

        # ── INCORRECT / AMBIGUOUS: web search + re-generate ──────────────────
        logger.info(f"CRAG web search (action={action.value})...")
        web_results = self._web.search(question, max_results=_cfg.web_search_max_results)

        # Collect relevant local chunks for AMBIGUOUS path
        id_map = {c.chunk_id: c for c in chunks}
        relevant_local: list[SearchResult] = [
            id_map[g.chunk_id] for g in grades if g.relevant and g.chunk_id in id_map
        ]

        corrected_retrieval = _build_corrected_result(
            original=retrieval_result,
            relevant_local=relevant_local,
            web_results=web_results,
            action=action,
        )

        if corrected_retrieval.is_empty:
            logger.warning("CRAG: corrected retrieval is empty — returning original result.")
            final = generation_result
        else:
            t_gen = time.perf_counter()
            final = self._generator.generate(
                question=question,
                retrieval_result=corrected_retrieval,
            )
            logger.info(f"CRAG re-generation: {time.perf_counter() - t_gen:.2f}s")

        total = time.perf_counter() - t0
        logger.info(
            f"CRAG complete | action={action.value} | web={len(web_results)} | total={total:.2f}s"
        )

        return CRAGResult(
            question=question,
            action=action,
            original_result=generation_result,
            final_result=final,
            relevance_grades=grades,
            web_results_used=web_results,
            web_search_triggered=True,
            latency_seconds=total,
        )
