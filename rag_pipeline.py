"""
End-to-end Financial RAG Pipeline.

Wires all four layers into a single callable interface:

  Layer 2 — Query Transformation  (HyDE + Multi-Query + Step-Back)
  Layer 3 — Hybrid Retrieval      (BM25 + Qdrant + RRF + FlashRank reranking)
  Layer 4 — Answer Generation     (LLM synthesis + grounded source citations)

──────────────────────────────────────────────────────────────────────────────
Quick-start:

    from rag_pipeline import FinancialRAGPipeline
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    pipeline = FinancialRAGPipeline(qdrant_client=client)

    # Structured answer with citations
    result = pipeline.ask("What was Apple's revenue in Q4 2024?")
    print(result.format_answer_with_citations())

    # Scoped to a specific ticker + year
    from retrieval.models import MetadataFilter
    result = pipeline.ask(
        question="What was NVIDIA's data center gross margin?",
        metadata_filter=MetadataFilter(ticker="NVDA", year=2024),
    )

    # Streaming (for UI layers)
    for token in pipeline.ask_streaming("What was Meta's ad revenue?"):
        print(token, end="", flush=True)

    # Full diagnostic dump (for debugging / notebooks)
    result, query_summary, retrieval_summary = pipeline.ask_verbose(
        "How did Apple's Services revenue trend across 2024?"
    )
    print(query_summary)
    print(retrieval_summary)
    print(result.format_answer_with_citations())

──────────────────────────────────────────────────────────────────────────────
Latency profile (typical, CPU-only):

  Layer 2 — Query transformation : ~0.8–1.2 s  (3 concurrent LLM calls)
  Layer 3 — Hybrid retrieval     : ~0.3–0.8 s  (BM25 + Qdrant + reranker)
  Layer 4 — Answer generation    : ~0.8–2.0 s  (single LLM call)
  ─────────────────────────────────────────────
  Total                          : ~2–4 s       end-to-end

──────────────────────────────────────────────────────────────────────────────
Thread-safety:

  All internal singletons (OpenAI client, embedding model, BM25 index,
  FlashRank reranker) are safe for concurrent reads after first initialisation.
  Parallel ask() calls across threads are supported.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
import contextlib
from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client import QdrantClient

from cache.semantic_cache import SemanticCache
from config import settings as _settings
from generation import Generator
from generation.models import GenerationResult
from observability.tracer import RAGTracer
from query import QueryTransformer
from query.router import QueryRouter
from retrieval import retrieve, warmup_bm25, warmup_embed_client, warmup_reranker
from retrieval.models import MetadataFilter, RetrievalResult

if TYPE_CHECKING:
    from crag.models import CRAGResult


class FinancialRAGPipeline:
    """
    Four-layer Financial RAG pipeline for SEC 8-K earnings filings.

    Composes:
      QueryTransformer  →  TransformedQuery   (HyDE + multi-query + step-back)
      retrieve()        →  RetrievalResult    (BM25 + Qdrant + RRF + rerank)
      Generator         →  GenerationResult   (LLM answer + citations)

    The pipeline is stateless between calls.  Internal models and API clients
    are lazy-loaded on first use and cached as module-level singletons.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        enable_query_cache: bool = True,
    ) -> None:
        self.qdrant_client = qdrant_client
        self._transformer = QueryTransformer(enable_cache=enable_query_cache)
        self._generator = Generator()
        self._router = QueryRouter()
        self._cache = SemanticCache(qdrant_client)

        # ── Observability: structured per-request tracing ──────────────────
        obs_cfg = _settings.observability
        self._tracer = RAGTracer(
            enabled=obs_cfg.tracing_enabled,
            output_dir=obs_cfg.trace_output_dir,
            persist_traces=obs_cfg.persist_traces,
            cost_alert_per_request_usd=obs_cfg.cost_alert_per_request_usd,
            cost_alert_per_session_usd=obs_cfg.cost_alert_per_session_usd,
        )

        logger.info("Pre-loading models into memory to prevent cold-start latency...")
        warmup_embed_client()
        warmup_bm25()

        if _settings.reranker.enabled:
            with contextlib.suppress(ImportError):
                warmup_reranker()  # Loads FlashRank cross-encoder

        logger.info(
            "FinancialRAGPipeline ready | "
            f"qdrant={_settings.infra.qdrant_url} | "
            f"transform_model={_settings.query_transform.model} | "
            f"generation_model={_settings.generation.model} | "
            f"reranker={'enabled' if _settings.reranker.enabled else 'disabled'} | "
            f"tracing={'enabled' if obs_cfg.tracing_enabled else 'disabled'}"
        )

    # ── Primary interface ─────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        metadata_filter: MetadataFilter | None = None,
        enable_routing: bool = True,
    ) -> GenerationResult:
        pipeline_start = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        logger.info(f"Pipeline.ask | {question!r:.80}")

        # ── Start trace ───────────────────────────────────────────────────────
        trace = self._tracer.start_trace(question=question)

        # ── Check semantic cache first ───────────────────────────────────────
        cached_result, cache_span = self._cache.get(question)
        self._tracer.record_semantic_cache(trace, cache_span)

        if cached_result:
            # Cache hit avoids all other layers
            trace.status = self._tracer.end_trace(
                trace,
                total_latency=time.perf_counter() - pipeline_start,
            ).status
            cached_result.trace_id = trace.trace_id
            cached_result.was_cached = True
            return cached_result

        if enable_routing:
            routing = self._router.route(question)
            logger.info(f"[Router] {routing.summary()}")

            if routing.should_refuse:
                from generation.models import GenerationResult

                return GenerationResult(
                    answer=(
                        "I can only answer questions about SEC 8-K earnings filings "
                        "for AAPL, NVDA, MSFT, AMZN, META, JPM, XOM, UNH, TSLA, and WMT. "
                        "Please ask a financial question about one of these companies."
                    ),
                    citations=[],
                    grounded=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    model=_settings.generation.model,
                    context_tokens_used=0,
                    chunks_used=0,
                )
        else:
            routing = None

        # ── Layer 2: Query Transformation ────────────────────────────────────
        t2 = time.perf_counter()
        transformed = self._transformer.transform(
            question,
            skip_hyde=(routing.skip_hyde if routing else False),
        )
        t2_elapsed = time.perf_counter() - t2

        logger.info(
            f"[L2] {len(transformed.multi_queries)} query variants | "
            f"degraded={transformed.failed_techniques} | {t2_elapsed:.2f}s"
        )

        # Record L2 span
        self._tracer.record_query_transform(
            trace,
            self._tracer.build_query_transform_span(
                latency=t2_elapsed,
                cache_hit=False,  # cache hit is internal to transformer
                multi_query_count=len(transformed.multi_queries),
                hyde_generated=(transformed.hyde_document != question),
                stepback_generated=(transformed.stepback_query != question),
                failed_techniques=list(transformed.failed_techniques),
            ),
        )

        # ── Layer 3: Hybrid Retrieval ─────────────────────────────────────────
        t3 = time.perf_counter()
        retrieval_result: RetrievalResult = retrieve(
            query=transformed,
            qdrant_client=self.qdrant_client,
            metadata_filter=metadata_filter,
        )
        t3_elapsed = time.perf_counter() - t3
        logger.info(
            f"[L3] {retrieval_result.total_candidates} candidates → "
            f"{len(retrieval_result.results)} results | "
            f"reranked={retrieval_result.reranked} | {t3_elapsed:.2f}s"
        )

        # Record L3 span
        self._tracer.record_retrieval(
            trace,
            self._tracer.build_retrieval_span(
                latency=t3_elapsed,
                total_candidates=retrieval_result.total_candidates,
                final_count=len(retrieval_result.results),
                reranked=retrieval_result.reranked,
                reranker_model=(_settings.reranker.model if _settings.reranker.enabled else ""),
                results=retrieval_result.results,
            ),
        )

        # ── Layer 4: Generation ────────────────────────────────────────────────
        t4 = time.perf_counter()
        result: GenerationResult = self._generator.generate(
            question=question,
            retrieval_result=retrieval_result,
        )
        t4_elapsed = time.perf_counter() - t4

        # Record L4 span
        gen_span = self._tracer.build_generation_span(
            latency=t4_elapsed,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            context_chunks=result.context_chunks_used,
            context_tokens=result.context_tokens_used,
            citation_count=len(result.citations),
            grounded=result.grounded,
            retrieval_failed=result.retrieval_failed,
        )
        self._tracer.record_generation(trace, gen_span)

        # Record the LLM call for generation
        self._tracer.record_llm_call(
            trace,
            caller="generation",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=t4_elapsed,
        )

        # Update semantic cache asynchronously
        self._cache.set(question, result)

        # ── Finalize trace ─────────────────────────────────────────────────────
        total = time.perf_counter() - pipeline_start
        self._tracer.end_trace(trace, total_latency=total)

        # Attach trace_id for request correlation
        result.trace_id = trace.trace_id

        logger.info(
            f"Pipeline complete | grounded={result.grounded} | "
            f"citations={len(result.citations)} | tokens={result.total_tokens} | "
            f"cost=${trace.total_cost_usd:.4f} | trace={trace.trace_id[:8]} | "
            f"total={total:.2f}s (L2={t2_elapsed:.2f}s L3={t3_elapsed:.2f}s L4={t4_elapsed:.2f}s)"
        )

        return result

    # ── Streaming variant ─────────────────────────────────────────────────────

    def ask_streaming(
        self,
        question: str,
        metadata_filter: MetadataFilter | None = None,
    ) -> Iterator[str]:
        """
        Streaming pipeline: run L2 + L3 synchronously, then stream L4 tokens.

        Designed for UI layers (Streamlit, Gradio, FastAPI Server-Sent Events).
        No structured GenerationResult is returned — use ask() when you need
        citation extraction, token counts, or the grounding flag.

        Args:
            question       : natural language question
            metadata_filter: optional ticker/year/quarter scoping

        Yields:
            str: raw text delta tokens from the LLM response

        Usage:
            for token in pipeline.ask_streaming("What was Apple's Q4 revenue?"):
                print(token, end="", flush=True)
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        logger.info(f"Pipeline.ask_streaming | {question!r:.80}")

        transformed = self._transformer.transform(question)
        retrieval_result = retrieve(
            query=transformed,
            qdrant_client=self.qdrant_client,
            metadata_filter=metadata_filter,
        )

        yield from self._generator.generate_streaming(
            question=question,
            retrieval_result=retrieval_result,
        )

    # ── Verbose diagnostic variant ────────────────────────────────────────────

    def ask_verbose(
        self,
        question: str,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[GenerationResult, str, str]:
        """
        Ask a question and return result + full diagnostic summaries.

        Useful for debugging, evaluation harnesses, and demo notebooks.

        Returns:
            (GenerationResult, query_transform_summary, retrieval_summary)

        Example:
            result, q_summary, r_summary = pipeline.ask_verbose(
                "How did NVIDIA's data center revenue change in Q3 2024?"
            )
            print("=== Query Transformation ===")
            print(q_summary)
            print("\\n=== Retrieval ===")
            print(r_summary)
            print("\\n=== Answer ===")
            print(result.format_answer_with_citations())

        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        transformed = self._transformer.transform(question)
        retrieval_result = retrieve(
            query=transformed,
            qdrant_client=self.qdrant_client,
            metadata_filter=metadata_filter,
        )
        result = self._generator.generate(
            question=question,
            retrieval_result=retrieval_result,
        )
        return result, transformed.summary(), retrieval_result.summary()

    # ── CRAG variant

    def ask_with_crag(
        self,
        question: str,
        metadata_filter: MetadataFilter | None = None,
    ) -> CRAGResult:
        """
        Full pipeline with CRAG correction loop.

        Runs L2 + L3 + L4 exactly as ask(), then passes the result through
        the CRAGCorrector which grades retrieved chunks, optionally triggers
        web-search fallback, and re-generates if necessary.

        Returns CRAGResult which wraps both the original and final
        GenerationResult alongside grading diagnostics.

        Args:
            question       : natural language financial question
            metadata_filter: optional ticker/year/quarter scoping

        Returns:
            CRAGResult with .final_result, .action, .was_corrected, etc.
        """
        from crag import CRAGCorrector

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        # Lazy-init: import is deferred to avoid loading crag at startup if unused.
        # Thread-safe: the GIL ensures only one thread constructs CRAGCorrector.
        if self._corrector is None:
            self._corrector = CRAGCorrector()

        pipeline_start = time.perf_counter()

        # L2
        transformed = self._transformer.transform(question)

        # L3
        retrieval_result: RetrievalResult = retrieve(
            query=transformed,
            qdrant_client=self.qdrant_client,
            metadata_filter=metadata_filter,
        )

        # L4
        generation_result = self._generator.generate(
            question=question,
            retrieval_result=retrieval_result,
        )

        # L5 — CRAG
        crag_result = self._corrector.correct(
            question=question,
            generation_result=generation_result,
            retrieval_result=retrieval_result,
        )

        total = time.perf_counter() - pipeline_start
        logger.info(
            f"Pipeline+CRAG complete | action={crag_result.action.value} | "
            f"corrected={crag_result.was_corrected} | total={total:.2f}s"
        )
        return crag_result
