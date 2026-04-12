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

from loguru import logger
from qdrant_client import QdrantClient

from config import settings as _settings
from generation import Generator
from generation.models import GenerationResult
from query import QueryTransformer
from retrieval import retrieve
from retrieval.models import MetadataFilter, RetrievalResult


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
        """
        Args:
            qdrant_client     : connected QdrantClient (caller owns lifecycle)
            enable_query_cache: cache QueryTransformer results in-memory
                                to avoid redundant API calls for repeated queries
                                (e.g. during evaluation, CRAG loops, UI demos)
        """
        self.qdrant_client = qdrant_client
        self._transformer = QueryTransformer(enable_cache=enable_query_cache)
        self._generator = Generator()
        logger.info(
            "FinancialRAGPipeline ready | "
            f"qdrant={_settings.infra.qdrant_url} | "
            f"transform_model={_settings.query_transform.model} | "
            f"generation_model={_settings.generation.model} | "
            f"reranker={'enabled' if _settings.reranker.enabled else 'disabled'}"
        )

    # ── Primary interface ─────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        metadata_filter: MetadataFilter | None = None,
    ) -> GenerationResult:
        """
        Ask a financial question and return a structured GenerationResult.

        Args:
            question       : natural language question about earnings / filings
            metadata_filter: optional ticker/year/quarter scoping
                             Example: MetadataFilter(ticker="AAPL", year=2024)

        Returns:
            GenerationResult with:
              .answer                     — synthesised LLM response with [N] citations
              .citations                  — list of Citation objects with source metadata
              .format_answer_with_citations() — pretty-printed answer + source list
              .to_dict() / .to_json()     — serialisable for API responses
              .grounded                   — False signals insufficient context
              .total_tokens               — for cost tracking
        """
        pipeline_start = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        logger.info(f"Pipeline.ask | {question!r:.80}")

        # ── Layer 2: Query Transformation ────────────────────────────────────
        t2 = time.perf_counter()
        transformed = self._transformer.transform(question)
        t2_elapsed = time.perf_counter() - t2
        logger.info(
            f"[L2] {len(transformed.multi_queries)} query variants | "
            f"degraded={transformed.failed_techniques} | {t2_elapsed:.2f}s"
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

        # ── Layer 4: Generation ────────────────────────────────────────────────
        t4 = time.perf_counter()
        result: GenerationResult = self._generator.generate(
            question=question,
            retrieval_result=retrieval_result,
        )
        t4_elapsed = time.perf_counter() - t4

        total = time.perf_counter() - pipeline_start
        logger.info(
            f"Pipeline complete | grounded={result.grounded} | "
            f"citations={len(result.citations)} | tokens={result.total_tokens} | "
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
            print(f"\\nTokens: {result.total_tokens} | Cost: ${result.cost_estimate_usd:.5f}")
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
