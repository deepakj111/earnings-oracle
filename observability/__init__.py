# observability/__init__.py
"""
Production-grade LLM observability for the Financial RAG system.

Provides structured per-request tracing with:
  - End-to-end latency breakdown by pipeline layer
  - Token usage and cost tracking per LLM call
  - Retrieval quality diagnostics (candidates, reranked, cited)
  - Thread-safe context-managed spans via contextvars
  - JSON-serializable traces for offline analysis

Usage:
    from observability import RAGTracer

    tracer = RAGTracer()
    with tracer.start_trace(question="What was AAPL revenue?") as trace:
        # ... pipeline runs, spans are recorded automatically ...
        pass
    print(trace.to_json())
"""

from observability.cost_tracker import CostTracker
from observability.trace_models import (
    CostEstimate,
    CRAGSpan,
    GenerationSpan,
    LLMCallSpan,
    PipelineTrace,
    QueryTransformSpan,
    RetrievalSpan,
)
from observability.tracer import RAGTracer

__all__ = [
    "RAGTracer",
    "CostTracker",
    "PipelineTrace",
    "QueryTransformSpan",
    "RetrievalSpan",
    "GenerationSpan",
    "CRAGSpan",
    "LLMCallSpan",
    "CostEstimate",
]
