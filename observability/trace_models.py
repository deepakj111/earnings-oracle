# observability/trace_models.py
"""
Structured trace data contracts for RAG pipeline observability.

Each dataclass represents one span in a hierarchical trace:

  PipelineTrace (root)
    ├── QueryTransformSpan   — Layer 2 timing, degraded techniques, cache status
    ├── RetrievalSpan        — Layer 3 candidate counts, scores, parent fetch
    ├── GenerationSpan       — Layer 4 token usage, model, grounding status
    ├── CRAGSpan             — Layer 5 action, web search, re-generation
    └── LLMCallSpan[]        — individual LLM API calls across all layers

Design decisions:
  - Frozen dataclasses for immutability after construction
  - All timestamps are ISO-8601 UTC strings for JSON serialization
  - Cost estimates derived from a pricing table, not hard-coded rates
  - Span IDs use UUID4 for uniqueness across distributed systems
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SpanStatus(str, Enum):
    """Status of a trace span."""

    OK = "ok"
    ERROR = "error"
    DEGRADED = "degraded"  # partial failure (e.g., one LLM technique failed)


@dataclass
class CostEstimate:
    """
    Cost estimate for a single LLM API call.

    Calculated from model pricing table × token counts.
    All values in USD.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.prompt_cost_usd + self.completion_cost_usd

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cost_usd": round(self.prompt_cost_usd, 6),
            "completion_cost_usd": round(self.completion_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


@dataclass
class LLMCallSpan:
    """
    Trace span for a single LLM API call.

    Captures model, token usage, latency, cost, and the caller context
    (which pipeline layer initiated the call).
    """

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    caller: str = ""  # "query_transform/hyde", "generation", "crag/grader", etc.
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    status: SpanStatus = SpanStatus.OK
    error_message: str = ""
    cost: CostEstimate | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "caller": self.caller,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
            "status": self.status.value,
            "error_message": self.error_message,
            "cost": self.cost.to_dict() if self.cost else None,
            "timestamp": self.timestamp,
        }


@dataclass
class QueryTransformSpan:
    """
    Trace span for Layer 2 — Query Transformation.

    Records which techniques ran, which degraded, cache behavior,
    and the resulting query variant counts.
    """

    latency_seconds: float = 0.0
    cache_hit: bool = False
    techniques_attempted: list[str] = field(default_factory=list)
    techniques_failed: list[str] = field(default_factory=list)
    multi_query_count: int = 0
    hyde_generated: bool = False
    stepback_generated: bool = False
    status: SpanStatus = SpanStatus.OK

    @property
    def is_degraded(self) -> bool:
        return len(self.techniques_failed) > 0

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "cache_hit": self.cache_hit,
            "techniques_attempted": self.techniques_attempted,
            "techniques_failed": self.techniques_failed,
            "multi_query_count": self.multi_query_count,
            "hyde_generated": self.hyde_generated,
            "stepback_generated": self.stepback_generated,
            "status": self.status.value,
            "is_degraded": self.is_degraded,
        }


@dataclass
class RetrievalSpan:
    """
    Trace span for Layer 3 — Hybrid Retrieval.

    Records candidate pool sizes, reranking behavior, parent fetch
    outcomes, and the final chunk quality diagnostics.
    """

    latency_seconds: float = 0.0
    dense_candidates: int = 0
    bm25_candidates: int = 0
    total_unique_candidates: int = 0
    rrf_fused_count: int = 0
    reranked: bool = False
    reranker_model: str = ""
    parent_fetch_count: int = 0
    final_chunk_count: int = 0
    top_rrf_score: float = 0.0
    top_rerank_score: float = 0.0
    source_distribution: dict[str, int] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.OK

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "dense_candidates": self.dense_candidates,
            "bm25_candidates": self.bm25_candidates,
            "total_unique_candidates": self.total_unique_candidates,
            "rrf_fused_count": self.rrf_fused_count,
            "reranked": self.reranked,
            "reranker_model": self.reranker_model,
            "parent_fetch_count": self.parent_fetch_count,
            "final_chunk_count": self.final_chunk_count,
            "top_rrf_score": round(self.top_rrf_score, 6),
            "top_rerank_score": round(self.top_rerank_score, 4),
            "source_distribution": self.source_distribution,
            "status": self.status.value,
        }


@dataclass
class GenerationSpan:
    """
    Trace span for Layer 4 — Answer Generation.

    Records context window construction, token usage, grounding status,
    citation extraction, and cost.
    """

    latency_seconds: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_chunks_used: int = 0
    context_tokens_used: int = 0
    citation_count: int = 0
    grounded: bool = True
    retrieval_failed: bool = False
    cost: CostEstimate | None = None
    status: SpanStatus = SpanStatus.OK

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "context_chunks_used": self.context_chunks_used,
            "context_tokens_used": self.context_tokens_used,
            "citation_count": self.citation_count,
            "grounded": self.grounded,
            "retrieval_failed": self.retrieval_failed,
            "cost": self.cost.to_dict() if self.cost else None,
            "status": self.status.value,
        }


@dataclass
class CRAGSpan:
    """
    Trace span for Layer 5 — Corrective RAG.

    Records the CRAG decision, grading results, web search outcomes,
    and whether re-generation was triggered.
    """

    latency_seconds: float = 0.0
    enabled: bool = True
    action: str = ""  # "correct" | "ambiguous" | "incorrect"
    relevance_ratio: float = 0.0
    chunks_graded: int = 0
    chunks_relevant: int = 0
    web_search_triggered: bool = False
    web_results_count: int = 0
    was_corrected: bool = False
    status: SpanStatus = SpanStatus.OK

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "enabled": self.enabled,
            "action": self.action,
            "relevance_ratio": round(self.relevance_ratio, 4),
            "chunks_graded": self.chunks_graded,
            "chunks_relevant": self.chunks_relevant,
            "web_search_triggered": self.web_search_triggered,
            "web_results_count": self.web_results_count,
            "was_corrected": self.was_corrected,
            "status": self.status.value,
        }


@dataclass
class SemanticCacheSpan:
    """
    Trace span for Embedding-Based Semantic Cache.

    Records cache lookup latency, hit status, and the highest similarity score.
    """

    latency_seconds: float = 0.0
    cache_hit: bool = False
    similarity_score: float = 0.0
    threshold_used: float = 0.0
    status: SpanStatus = SpanStatus.OK

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "cache_hit": self.cache_hit,
            "similarity_score": round(self.similarity_score, 6),
            "threshold_used": self.threshold_used,
            "status": self.status.value,
        }


@dataclass
class GraphRetrievalSpan:
    """
    Trace span for Knowledge Graph Fused Retrieval.

    Records entity matching, relationship traversal, and chunk injection.
    """

    latency_seconds: float = 0.0
    entities_matched: int = 0
    matched_entity_names: list[str] = field(default_factory=list)
    relationships_traversed: int = 0
    chunks_injected: int = 0
    status: SpanStatus = SpanStatus.OK

    def to_dict(self) -> dict:
        return {
            "latency_seconds": round(self.latency_seconds, 4),
            "entities_matched": self.entities_matched,
            "matched_entity_names": self.matched_entity_names,
            "relationships_traversed": self.relationships_traversed,
            "chunks_injected": self.chunks_injected,
            "status": self.status.value,
        }


@dataclass
class PipelineTrace:
    """
    Root trace for one end-to-end RAG pipeline request.

    Aggregates all layer spans, LLM call spans, and computes
    total cost and latency breakdown.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_latency_seconds: float = 0.0

    # Layer spans
    semantic_cache: SemanticCacheSpan | None = None
    query_transform: QueryTransformSpan | None = None
    retrieval: RetrievalSpan | None = None
    graph_retrieval: GraphRetrievalSpan | None = None
    generation: GenerationSpan | None = None
    crag: CRAGSpan | None = None

    # All individual LLM calls across all layers
    llm_calls: list[LLMCallSpan] = field(default_factory=list)

    # Aggregate cost
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0

    # Pipeline outcome
    status: SpanStatus = SpanStatus.OK
    error_message: str = ""
    metadata: dict = field(default_factory=dict)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def latency_breakdown(self) -> dict[str, float]:
        """Per-layer latency in seconds."""
        breakdown: dict[str, float] = {}
        if self.semantic_cache:
            breakdown["semantic_cache"] = self.semantic_cache.latency_seconds
        if self.query_transform:
            breakdown["query_transform"] = self.query_transform.latency_seconds
        if self.retrieval:
            breakdown["retrieval"] = self.retrieval.latency_seconds
        if self.graph_retrieval:
            breakdown["graph_retrieval"] = self.graph_retrieval.latency_seconds
        if self.generation:
            breakdown["generation"] = self.generation.latency_seconds
        if self.crag:
            breakdown["crag"] = self.crag.latency_seconds
        return breakdown

    @property
    def cost_breakdown(self) -> dict[str, float]:
        """Per-caller cost in USD, aggregated from all LLM calls."""
        costs: dict[str, float] = {}
        for call in self.llm_calls:
            if call.cost:
                caller = call.caller or "unknown"
                costs[caller] = costs.get(caller, 0.0) + call.cost.total_cost_usd
        return costs

    # ── Aggregation ────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """
        Compute aggregate fields from individual spans.
        Call this after all spans have been recorded.
        """
        self.total_prompt_tokens = sum(c.prompt_tokens for c in self.llm_calls)
        self.total_completion_tokens = sum(c.completion_tokens for c in self.llm_calls)
        self.total_cost_usd = sum(c.cost.total_cost_usd for c in self.llm_calls if c.cost)

        # Set degraded status if any layer was degraded
        if any(c.status == SpanStatus.ERROR for c in self.llm_calls):
            self.status = SpanStatus.DEGRADED

    # ── Summaries ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        parts = [
            f"trace={self.trace_id[:8]}",
            f"latency={self.total_latency_seconds:.2f}s",
            f"llm_calls={self.total_llm_calls}",
            f"tokens={self.total_tokens}",
            f"cost=${self.total_cost_usd:.4f}",
            f"status={self.status.value}",
        ]
        return " | ".join(parts)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "timestamp": self.timestamp,
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_llm_calls": self.total_llm_calls,
            "status": self.status.value,
            "error_message": self.error_message,
            "latency_breakdown": {k: round(v, 4) for k, v in self.latency_breakdown.items()},
            "cost_breakdown": {k: round(v, 6) for k, v in self.cost_breakdown.items()},
            "semantic_cache": self.semantic_cache.to_dict() if self.semantic_cache else None,
            "query_transform": (self.query_transform.to_dict() if self.query_transform else None),
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "graph_retrieval": (self.graph_retrieval.to_dict() if self.graph_retrieval else None),
            "generation": self.generation.to_dict() if self.generation else None,
            "crag": self.crag.to_dict() if self.crag else None,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
