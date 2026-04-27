# observability/tracer.py
"""
Thread-safe RAG pipeline tracer with context-managed spans.

The RAGTracer is the central orchestrator for request-level observability.
It manages the lifecycle of a PipelineTrace and provides convenience methods
for recording layer spans and LLM calls.

Thread-safety model:
  - Each request gets its own PipelineTrace via Python contextvars
  - The tracer instance itself is stateless — safe for concurrent use
  - The CostTracker is shared across requests (thread-safe via Lock)

Usage (standalone):
    tracer = RAGTracer()
    trace = tracer.start_trace("What was AAPL revenue?")

    # Record layer spans
    tracer.record_query_transform(trace, QueryTransformSpan(...))
    tracer.record_retrieval(trace, RetrievalSpan(...))
    tracer.record_generation(trace, GenerationSpan(...))

    # Record individual LLM calls
    tracer.record_llm_call(trace, LLMCallSpan(...))

    # Finalize
    tracer.end_trace(trace, total_latency=3.14)
    print(trace.to_json())

Usage (via pipeline — see rag_pipeline.py):
    The pipeline calls tracer methods at each layer boundary.
    Traces are automatically finalized and persisted on request completion.

Persistence:
    Traces are optionally written to disk as JSON files in the configured
    trace output directory. This enables offline analysis, debugging, and
    cost reconciliation against OpenAI invoices.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from observability.cost_tracker import CostTracker, estimate_cost
from observability.trace_models import (
    CRAGSpan,
    GenerationSpan,
    GraphRetrievalSpan,
    LLMCallSpan,
    PipelineTrace,
    QueryTransformSpan,
    RetrievalSpan,
    SpanStatus,
)


class RAGTracer:
    """
    Production-grade tracer for the Financial RAG pipeline.

    Provides structured observability for every pipeline request:
      - Per-layer latency breakdown
      - LLM call tracking with token counts and cost
      - Retrieval quality diagnostics
      - Cost accumulation with alert thresholds
      - Optional trace persistence to disk

    Thread-safety: the tracer instance is safe for concurrent use.
    Each trace is an independent PipelineTrace object — no shared mutable state.
    """

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "data/traces",
        persist_traces: bool = True,
        cost_alert_per_request_usd: float = 0.10,
        cost_alert_per_session_usd: float = 5.00,
    ) -> None:
        self.enabled = enabled
        self._output_dir = Path(output_dir)
        self._persist = persist_traces
        self._cost_tracker = CostTracker(
            alert_per_request_usd=cost_alert_per_request_usd,
            alert_per_session_usd=cost_alert_per_session_usd,
        )

        if enabled:
            logger.info(
                f"RAGTracer ready | persist={persist_traces} | "
                f"output_dir={output_dir} | "
                f"cost_alerts=($"
                f"{cost_alert_per_request_usd:.2f}/req, "
                f"${cost_alert_per_session_usd:.2f}/session)"
            )

    # ── Trace lifecycle ────────────────────────────────────────────────────

    def start_trace(self, question: str, **metadata: object) -> PipelineTrace:
        """
        Create a new PipelineTrace for a request.

        Args:
            question: the user question being processed
            **metadata: arbitrary key-value pairs to attach to the trace

        Returns:
            A new PipelineTrace instance (caller owns lifetime).
        """
        if not self.enabled:
            return PipelineTrace(question=question)

        trace = PipelineTrace(
            question=question,
            metadata=dict(metadata),
        )
        logger.debug(f"Trace started | id={trace.trace_id[:8]} | q={question!r:.60}")
        return trace

    def end_trace(
        self,
        trace: PipelineTrace,
        total_latency: float,
        status: SpanStatus = SpanStatus.OK,
        error_message: str = "",
    ) -> PipelineTrace:
        """
        Finalize a trace: compute aggregates, record cost, optionally persist.

        Args:
            trace         : the PipelineTrace to finalize
            total_latency : end-to-end latency in seconds
            status        : overall pipeline status
            error_message : error details if status is ERROR

        Returns:
            The finalized PipelineTrace (same object, mutated in place).
        """
        if not self.enabled:
            return trace

        trace.total_latency_seconds = total_latency
        trace.status = status
        trace.error_message = error_message
        trace.finalize()

        # Record all costs in the session tracker
        costs = [c.cost for c in trace.llm_calls if c.cost is not None]
        if costs:
            self._cost_tracker.record_request_cost(costs)

        logger.info(f"Trace complete | {trace.summary()}")

        # Persist to disk
        if self._persist:
            self._persist_trace(trace)

        return trace

    # ── Layer span recording ───────────────────────────────────────────────

    def record_query_transform(self, trace: PipelineTrace, span: QueryTransformSpan) -> None:
        """Attach the Layer 2 query transformation span to a trace."""
        if not self.enabled:
            return
        trace.query_transform = span
        logger.debug(
            f"[Trace {trace.trace_id[:8]}] L2 recorded | "
            f"latency={span.latency_seconds:.3f}s | "
            f"degraded={span.techniques_failed}"
        )

    def record_retrieval(self, trace: PipelineTrace, span: RetrievalSpan) -> None:
        """Attach the Layer 3 retrieval span to a trace."""
        if not self.enabled:
            return
        trace.retrieval = span
        logger.debug(
            f"[Trace {trace.trace_id[:8]}] L3 recorded | "
            f"latency={span.latency_seconds:.3f}s | "
            f"candidates={span.total_unique_candidates} → "
            f"final={span.final_chunk_count}"
        )

    def record_graph_retrieval(self, trace: PipelineTrace, span: GraphRetrievalSpan) -> None:
        """Attach the Graph Retrieval span to a trace."""
        if not self.enabled:
            return
        trace.graph_retrieval = span
        logger.debug(
            f"[Trace {trace.trace_id[:8]}] GraphRAG recorded | "
            f"latency={span.latency_seconds:.3f}s | "
            f"entities={span.entities_matched} | "
            f"chunks_injected={span.chunks_injected}"
        )

    def record_generation(self, trace: PipelineTrace, span: GenerationSpan) -> None:
        """Attach the Layer 4 generation span to a trace."""
        if not self.enabled:
            return
        trace.generation = span
        logger.debug(
            f"[Trace {trace.trace_id[:8]}] L4 recorded | "
            f"latency={span.latency_seconds:.3f}s | "
            f"tokens={span.total_tokens} | "
            f"grounded={span.grounded}"
        )

    def record_crag(self, trace: PipelineTrace, span: CRAGSpan) -> None:
        """Attach the Layer 5 CRAG span to a trace."""
        if not self.enabled:
            return
        trace.crag = span
        logger.debug(
            f"[Trace {trace.trace_id[:8]}] L5 recorded | "
            f"action={span.action} | "
            f"corrected={span.was_corrected}"
        )

    def record_llm_call(
        self,
        trace: PipelineTrace,
        caller: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        status: SpanStatus = SpanStatus.OK,
        error_message: str = "",
    ) -> LLMCallSpan:
        """
        Record a single LLM API call within a trace.

        Automatically computes the cost estimate from the model pricing table.

        Args:
            trace             : parent PipelineTrace
            caller            : caller context (e.g. "query_transform/hyde")
            model             : model name (e.g. "gpt-4.1-nano")
            prompt_tokens     : input token count
            completion_tokens : output token count
            latency_seconds   : call latency
            status            : call status
            error_message     : error details if failed

        Returns:
            The created LLMCallSpan (also appended to trace.llm_calls).
        """
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        span = LLMCallSpan(
            caller=caller,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency_seconds,
            status=status,
            error_message=error_message,
            cost=cost,
        )

        if self.enabled:
            trace.llm_calls.append(span)

        return span

    # ── Convenience: build spans from pipeline results ─────────────────────

    @staticmethod
    def build_query_transform_span(
        latency: float,
        cache_hit: bool,
        multi_query_count: int,
        hyde_generated: bool,
        stepback_generated: bool,
        failed_techniques: list[str],
    ) -> QueryTransformSpan:
        """Build a QueryTransformSpan from pipeline layer outputs."""
        techniques = ["multi_query", "stepback"]
        if hyde_generated:
            techniques.append("hyde")

        status = SpanStatus.OK
        if failed_techniques:
            status = SpanStatus.DEGRADED

        return QueryTransformSpan(
            latency_seconds=latency,
            cache_hit=cache_hit,
            techniques_attempted=techniques,
            techniques_failed=failed_techniques,
            multi_query_count=multi_query_count,
            hyde_generated=hyde_generated,
            stepback_generated=stepback_generated,
            status=status,
        )

    @staticmethod
    def build_retrieval_span(
        latency: float,
        total_candidates: int,
        final_count: int,
        reranked: bool,
        reranker_model: str,
        results: list,
    ) -> RetrievalSpan:
        """Build a RetrievalSpan from retrieval layer outputs."""
        # Compute source distribution
        source_dist: dict[str, int] = {}
        top_rrf = 0.0
        top_rerank = 0.0

        for r in results:
            src = getattr(r, "source", "unknown")
            source_dist[src] = source_dist.get(src, 0) + 1
            rrf = getattr(r, "rrf_score", 0.0)
            rerank = getattr(r, "rerank_score", float("-inf"))
            if rrf > top_rrf:
                top_rrf = rrf
            if rerank > top_rerank:
                top_rerank = rerank

        return RetrievalSpan(
            latency_seconds=latency,
            total_unique_candidates=total_candidates,
            rrf_fused_count=total_candidates,
            reranked=reranked,
            reranker_model=reranker_model,
            final_chunk_count=final_count,
            top_rrf_score=top_rrf,
            top_rerank_score=top_rerank,
            source_distribution=source_dist,
        )

    @staticmethod
    def build_generation_span(
        latency: float,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        context_chunks: int,
        context_tokens: int,
        citation_count: int,
        grounded: bool,
        retrieval_failed: bool,
    ) -> GenerationSpan:
        """Build a GenerationSpan from generation layer outputs."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        return GenerationSpan(
            latency_seconds=latency,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_chunks_used=context_chunks,
            context_tokens_used=context_tokens,
            citation_count=citation_count,
            grounded=grounded,
            retrieval_failed=retrieval_failed,
            cost=cost,
        )

    @staticmethod
    def build_crag_span(
        latency: float,
        enabled: bool,
        action: str,
        relevance_ratio: float,
        chunks_graded: int,
        chunks_relevant: int,
        web_search_triggered: bool,
        web_results_count: int,
        was_corrected: bool,
    ) -> CRAGSpan:
        """Build a CRAGSpan from CRAG layer outputs."""
        return CRAGSpan(
            latency_seconds=latency,
            enabled=enabled,
            action=action,
            relevance_ratio=relevance_ratio,
            chunks_graded=chunks_graded,
            chunks_relevant=chunks_relevant,
            web_search_triggered=web_search_triggered,
            web_results_count=web_results_count,
            was_corrected=was_corrected,
        )

    # ── Cost tracker access ─────────────────────────────────────────────────

    @property
    def cost_tracker(self) -> CostTracker:
        """Access the session-level cost tracker."""
        return self._cost_tracker

    def cost_summary(self) -> str:
        """Human-readable session cost summary."""
        return self._cost_tracker.summary()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _persist_trace(self, trace: PipelineTrace) -> Path | None:
        """Write trace JSON to disk. Returns the file path or None on failure."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            ts = trace.timestamp.replace(":", "-").replace("+", "").split(".")[0]
            filename = f"trace_{ts}_{trace.trace_id[:8]}.json"
            path = self._output_dir / filename
            path.write_text(trace.to_json(), encoding="utf-8")
            logger.debug(f"Trace persisted → {path}")
            return path
        except Exception as exc:
            logger.warning(f"Failed to persist trace: {exc}")
            return None

    def get_persisted_traces(self, limit: int = 50) -> list[dict]:
        """
        Load recent traces from disk for analysis.

        Returns parsed trace dicts sorted by timestamp (newest first),
        limited to `limit` entries.
        """
        if not self._output_dir.exists():
            return []

        traces: list[dict] = []
        for path in sorted(self._output_dir.glob("trace_*.json"), reverse=True):
            if len(traces) >= limit:
                break
            try:
                with open(path, encoding="utf-8") as f:
                    traces.append(json.load(f))
            except Exception as exc:
                logger.debug(f"Skipping corrupt trace file {path}: {exc}")
                continue

        return traces
