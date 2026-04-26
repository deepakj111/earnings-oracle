# api/metrics.py
"""
LLMOps Prometheus metrics for the Financial RAG API.

All metrics are registered in RAG_REGISTRY (not the default global registry)
so that:
  1. Tests can import RAG_REGISTRY and inspect it directly — no registry
     collision when multiple TestClient instances are created per test session.
  2. The /metrics endpoint serves only application-level metrics, not the
     default Python process metrics (gc, memory, etc.) unless desired.

Exported symbols used by the rest of the codebase:
  RAG_REGISTRY            — pass to generate_latest() in the /metrics route
  PrometheusMiddleware    — ASGI middleware; add via app.add_middleware()
  record_generation_result(result)
  record_retrieval_result(result)
  record_pipeline_latency(layer, seconds)
  record_crag_result(result)
"""

from __future__ import annotations

import time
import typing
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from generation.models import GenerationResult
    from retrieval.models import RetrievalResult

# ── Custom registry ────────────────────────────────────────────────────────────
# Using a dedicated registry (not prometheus_client.REGISTRY) prevents
# "Duplicated timeseries" errors when pytest creates multiple TestClient
# instances in the same process, each of which triggers create_app().
RAG_REGISTRY = CollectorRegistry(auto_describe=True)

# ── HTTP layer metrics ─────────────────────────────────────────────────────────
rag_http_requests_total = Counter(
    "rag_http_requests_total",
    "Total HTTP requests handled by the RAG API",
    ["endpoint", "method", "status_code"],
    registry=RAG_REGISTRY,
)

rag_http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=RAG_REGISTRY,
)

# ── LLM cost / token metrics ───────────────────────────────────────────────────
rag_llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "Cumulative LLM tokens consumed",
    ["model", "token_type"],
    registry=RAG_REGISTRY,
)

rag_llm_cost_usd_total = Counter(
    "rag_llm_cost_usd_total",
    "Cumulative estimated LLM cost in USD",
    ["model"],
    registry=RAG_REGISTRY,
)

# ── Retrieval metrics ──────────────────────────────────────────────────────────
rag_retrieval_candidates = Histogram(
    "rag_retrieval_candidates",
    "Number of candidate chunks entering the reranker",
    buckets=(5, 10, 20, 30, 50, 100),
    registry=RAG_REGISTRY,
)

rag_retrieval_results_returned = Histogram(
    "rag_retrieval_results_returned",
    "Number of chunks returned after reranking",
    buckets=(1, 2, 3, 5, 8, 10, 15, 20),
    registry=RAG_REGISTRY,
)

rag_context_tokens_used = Histogram(
    "rag_context_tokens_used",
    "Tokens consumed by the context window per query",
    buckets=(256, 512, 1024, 2048, 4096, 8192, 16384),
    registry=RAG_REGISTRY,
)

rag_grounded_responses_total = Counter(
    "rag_grounded_responses_total",
    "Responses classified as grounded vs ungrounded",
    ["grounded"],
    registry=RAG_REGISTRY,
)

rag_retrieval_failed_total = Counter(
    "rag_retrieval_failed_total",
    "Queries that returned zero retrieval results",
    registry=RAG_REGISTRY,
)

# ── CRAG metrics ───────────────────────────────────────────────────────────────
rag_crag_actions_total = Counter(
    "rag_crag_actions_total",
    "CRAG corrective actions taken (correct / ambiguous / incorrect)",
    ["action"],
    registry=RAG_REGISTRY,
)

# ── Pipeline layer latency ─────────────────────────────────────────────────────
rag_pipeline_latency_seconds = Histogram(
    "rag_pipeline_latency_seconds",
    "Per-layer pipeline latency in seconds",
    ["layer"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=RAG_REGISTRY,
)

# ── Path normalisation map ─────────────────────────────────────────────────────
# Maps raw request paths to clean endpoint label values.
# Unknown paths are bucketed as "/other" to limit cardinality.
_PATH_MAP: dict[str, str] = {
    "/query": "/query",
    "/query/": "/query",
    "/query/stream": "/query/stream",
    "/query/stream/": "/query/stream",
    "/health": "/health",
    "/health/": "/health",
    "/health/live": "/health/live",
    "/health/live/": "/health/live",
    "/health/ready": "/health/ready",
    "/health/ready/": "/health/ready",
    "/metrics": "/metrics",
    "/metrics/": "/metrics",
}


def _normalise_path(path: str) -> str:
    """Return a low-cardinality endpoint label for *path*."""
    return _PATH_MAP.get(path, "/other")


# ── ASGI middleware ────────────────────────────────────────────────────────────


class PrometheusMiddleware:
    """
    Zero-boilerplate ASGI middleware that records:
      - rag_http_requests_total (endpoint, method, status_code)
      - rag_http_request_duration_seconds (endpoint)

    Add to the FastAPI app via:
        app.add_middleware(PrometheusMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        endpoint = _normalise_path(path)

        status_code = 500
        start = time.perf_counter()

        async def send_with_status(message: typing.MutableMapping[str, typing.Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            duration = time.perf_counter() - start
            rag_http_requests_total.labels(
                endpoint=endpoint,
                method=method,
                status_code=str(status_code),
            ).inc()
            rag_http_request_duration_seconds.labels(endpoint=endpoint).observe(duration)


# ── Recording helpers ─────────────────────────────────────────────────────────


def record_generation_result(result: GenerationResult) -> None:
    """Push token, cost, grounding, and context metrics from a GenerationResult."""
    model = getattr(result, "model", "unknown")

    prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
    completion_tokens = getattr(result, "completion_tokens", 0) or 0

    grounded = getattr(result, "grounded", False)
    context_tokens = getattr(result, "context_tokens_used", None)
    retrieval_failed = getattr(result, "retrieval_failed", False)

    if prompt_tokens:
        rag_llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
    if completion_tokens:
        rag_llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)

    rag_grounded_responses_total.labels(grounded=str(grounded).lower()).inc()

    if retrieval_failed:
        rag_retrieval_failed_total.inc()

    if context_tokens is not None:
        rag_context_tokens_used.observe(context_tokens)


def record_retrieval_result(result: RetrievalResult) -> None:
    """Push retrieval candidate and result-count metrics."""
    total_candidates = getattr(result, "total_candidates", 0) or 0
    results = getattr(result, "results", []) or []

    rag_retrieval_candidates.observe(total_candidates)
    rag_retrieval_results_returned.observe(len(results))


def record_pipeline_latency(layer: str, seconds: float) -> None:
    """Record per-layer pipeline latency (call once per layer per request)."""
    rag_pipeline_latency_seconds.labels(layer=layer).observe(seconds)


def record_crag_result(result: typing.Any) -> None:
    """Push CRAG action counter from a CRAGResult."""
    action = result.action.value if hasattr(result.action, "value") else str(result.action)
    rag_crag_actions_total.labels(action=action).inc()
