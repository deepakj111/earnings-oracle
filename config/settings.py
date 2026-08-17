"""
Centralized configuration for the Financial RAG System.

All model names, token budgets, temperatures, and retry parameters
live here. Code modules import from this file — never from os.getenv directly.

Override any value by setting the corresponding env var in .env.
Env var names follow the pattern: RAG_<SECTION>_<KEY> (all uppercase).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"Config error: {key}={raw!r} is not a valid integer.") from None


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"Config error: {key}={raw!r} is not a valid float.") from None


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an env var. Accepts: 1/0, true/false, yes/no, on/off."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Layer 1: Query Routing ───────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryRouterConfig:
    """Configuration for query/router.py (Layer 1 — query intent classification)."""

    model: str = field(default_factory=lambda: _env_str("RAG_QUERY_ROUTER_MODEL", "gpt-5-mini"))
    temperature: float = field(default_factory=lambda: _env_float("RAG_QUERY_ROUTER_TEMP", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_QUERY_ROUTER_MAX_TOKENS", 2048))


# ── Layer 2: Query Transformation ─────────────────────────────────────────────


@dataclass(frozen=True)
class QueryTransformConfig:
    """Configuration for query/transformer.py (Layer 2 — HyDE + Multi-Query + Step-Back)."""

    model: str = field(default_factory=lambda: _env_str("RAG_QUERY_TRANSFORM_MODEL", "gpt-5-mini"))

    # Per-technique temperatures — intentionally different
    temperature_hyde: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_HYDE", 0.3)
    )
    temperature_multi_query: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_MULTI_QUERY", 0.7)
    )
    temperature_stepback: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_STEPBACK", 0.1)
    )

    # Max output tokens per technique
    max_tokens_hyde: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_HYDE", 4096)
    )
    max_tokens_multi_query: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_MULTI_QUERY", 4096)
    )
    max_tokens_stepback: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_STEPBACK", 4096)
    )

    # Retry / backoff
    max_retries: int = field(default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_RETRIES", 3))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_RETRY_DELAY", 1.0)
    )

    # In-memory LRU cache size
    cache_max_size: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_CACHE_SIZE", 256)
    )


# ── Layer 3: Reranking ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankerConfig:
    """
    Configuration for retrieval/reranker.py (Layer 3 — FlashRank cross-encoder).

    top_k_pre_rerank: number of RRF-fused candidates passed to the cross-encoder.
    enabled: set RAG_RERANKER_ENABLED=false to bypass reranking entirely.
    """

    model: str = field(
        default_factory=lambda: _env_str("RAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
    )
    top_k_pre_rerank: int = field(default_factory=lambda: _env_int("RAG_RERANKER_TOP_K_PRE", 20))
    enabled: bool = field(default_factory=lambda: _env_bool("RAG_RERANKER_ENABLED", True))


# ── Layer 4: Generation ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationConfig:
    """
    Configuration for generation/generator.py (Layer 4 — answer synthesis).

    max_context_tokens: hard cap on total tokens in the retrieved context block.
    """

    model: str = field(default_factory=lambda: _env_str("RAG_GENERATION_MODEL", "gpt-5"))
    temperature: float = field(default_factory=lambda: _env_float("RAG_GENERATION_TEMP", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_GENERATION_MAX_TOKENS", 4096))
    max_context_tokens: int = field(
        default_factory=lambda: _env_int("RAG_GENERATION_MAX_CONTEXT_TOKENS", 8192)
    )
    max_retries: int = field(default_factory=lambda: _env_int("RAG_GENERATION_MAX_RETRIES", 3))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("RAG_GENERATION_RETRY_DELAY", 1.0)
    )


# ── Ingestion / Embedding ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for ingestion/indexer.py (OpenAI API + Qdrant).
    Kept here so the retrieval layer can reference the same model name
    when embedding HyDE documents for dense search.
    """

    model: str = field(
        default_factory=lambda: _env_str("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    vector_dim: int = field(default_factory=lambda: _env_int("RAG_EMBEDDING_VECTOR_DIM", 1536))
    collection_name: str = field(
        default_factory=lambda: _env_str("RAG_QDRANT_COLLECTION", "company_filings")
    )
    upsert_batch_size: int = field(
        default_factory=lambda: _env_int("RAG_EMBEDDING_UPSERT_BATCH_SIZE", 50)
    )
    ingestion_batch_size: int = field(
        default_factory=lambda: _env_int("RAG_INGESTION_BATCH_SIZE", 100)
    )
    max_concurrency: int = field(
        default_factory=lambda: _env_int("RAG_INGESTION_MAX_CONCURRENCY", 4)
    )
    threads: int = field(default_factory=lambda: _env_int("RAG_EMBEDDING_THREADS", 0))


# ── Layer 3: Retrieval ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval/searcher.py (Layer 3 — BM25 + Qdrant + RRF)."""

    top_k_dense: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_DENSE", 25))
    top_k_bm25: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_BM25", 25))
    # Reduced 12 → 8: ablations showed Context Precision 0.245–0.37 with 12 chunks.
    # Fewer, higher-quality chunks improve LLM focus and reduce hallucination risk.
    # Override with RAG_RETRIEVAL_TOP_K_FINAL env var.
    top_k_final: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_FINAL", 8))
    rrf_k_constant: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_RRF_K", 60))
    parent_fetch_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_RETRIEVAL_PARENT_FETCH", True)
    )
    metadata_filter_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_RETRIEVAL_META_FILTER", True)
    )
    bm25_weight: float = field(
        default_factory=lambda: _env_float("RAG_RETRIEVAL_BM25_WEIGHT", 1.15)
    )
    dense_weight: float = field(
        default_factory=lambda: _env_float("RAG_RETRIEVAL_DENSE_WEIGHT", 1.0)
    )


# ── Infrastructure ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InfraConfig:
    """Infrastructure endpoints and secrets."""

    qdrant_url: str = field(default_factory=lambda: _env_str("QDRANT_URL", "http://localhost:6333"))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: _env_str("SEC_USER_AGENT", "Your Name your@email.com")
    )
    log_format: str = field(default_factory=lambda: _env_str("LOG_FORMAT", "text"))


# ── Layer 5: CRAG ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CRAGConfig:
    """Configuration for Layer 5 — Corrective RAG (crag/corrector.py & crag/grader.py)."""

    # Disabled by default: ablation studies (Arm 6) showed CRAG degrades performance
    # (-10pp Context Precision, -13pp Answer Relevancy) when the corpus is already
    # well-grounded (Faithfulness=1.0). Enable only for fallback experiments or when
    # dealing with domains where the corpus may be incomplete.
    enabled: bool = field(default_factory=lambda: _env_bool("RAG_CRAG_ENABLED", False))
    grader_model: str = field(
        default_factory=lambda: _env_str("RAG_CRAG_GRADER_MODEL", "gpt-5-mini")
    )
    grader_temperature: float = field(
        default_factory=lambda: _env_float("RAG_CRAG_GRADER_TEMP", 0.0)
    )
    grader_max_tokens: int = field(
        default_factory=lambda: _env_int("RAG_CRAG_GRADER_MAX_TOKENS", 2048)
    )
    high_relevance_threshold: float = field(
        default_factory=lambda: _env_float("RAG_CRAG_HIGH_THRESHOLD", 0.6)
    )
    low_relevance_threshold: float = field(
        default_factory=lambda: _env_float("RAG_CRAG_LOW_THRESHOLD", 0.2)
    )
    grade_even_if_grounded: bool = field(
        default_factory=lambda: _env_bool("RAG_CRAG_GRADE_IF_GROUNDED", False)
    )
    grader_max_workers: int = field(default_factory=lambda: _env_int("RAG_CRAG_GRADER_WORKERS", 2))
    web_search_max_results: int = field(
        default_factory=lambda: _env_int("RAG_CRAG_WEB_MAX_RESULTS", 4)
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────


@dataclass()  # mutable so monkeypatch can redirect output_dir in tests
class EvaluationConfig:
    """Configuration for evaluation/harness.py & metrics.py (LLMOps evaluation harness)."""

    model: str = field(default_factory=lambda: _env_str("RAG_EVAL_MODEL", "gpt-5-mini"))
    temperature: float = field(default_factory=lambda: _env_float("RAG_EVAL_TEMP", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_EVAL_MAX_TOKENS", 2048))
    max_workers: int = field(default_factory=lambda: _env_int("RAG_EVAL_MAX_WORKERS", 2))
    output_dir: str = field(
        default_factory=lambda: _env_str(
            "RAG_EVAL_OUTPUT_DIR",
            os.path.join(os.path.dirname(__file__), "..", "data", "eval_reports"),
        )
    )


# ── Observability / Tracing ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for observability/tracer.py (structured LLM tracing)."""

    tracing_enabled: bool = field(default_factory=lambda: _env_bool("RAG_TRACING_ENABLED", True))
    trace_output_dir: str = field(
        default_factory=lambda: _env_str(
            "RAG_TRACING_OUTPUT_DIR",
            os.path.join(os.path.dirname(__file__), "..", "data", "traces"),
        )
    )
    persist_traces: bool = field(default_factory=lambda: _env_bool("RAG_TRACING_PERSIST", False))
    cost_alert_per_request_usd: float = field(
        default_factory=lambda: _env_float("RAG_COST_ALERT_PER_REQUEST", 0.10)
    )
    cost_alert_per_session_usd: float = field(
        default_factory=lambda: _env_float("RAG_COST_ALERT_PER_SESSION", 5.00)
    )
    # Audit log — always-on structured JSON/JSONL output for every query
    audit_enabled: bool = field(default_factory=lambda: _env_bool("RAG_AUDIT_ENABLED", True))
    audit_log_dir: str = field(
        default_factory=lambda: _env_str(
            "RAG_AUDIT_LOG_DIR",
            os.path.join(os.path.dirname(__file__), "..", "data", "audit_logs"),
        )
    )


# ── Knowledge Graph ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeGraphConfig:
    """Configuration for GraphRAG knowledge graph extraction and retrieval."""

    extraction_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_KG_EXTRACTION_ENABLED", True)
    )
    retrieval_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_KG_RETRIEVAL_ENABLED", True)
    )
    extraction_model: str = field(
        default_factory=lambda: _env_str("RAG_KG_EXTRACTION_MODEL", "gpt-5-mini")
    )
    extraction_temperature: float = field(
        default_factory=lambda: _env_float("RAG_KG_EXTRACTION_TEMP", 1.0)
    )
    extraction_max_tokens: int = field(
        default_factory=lambda: _env_int("RAG_KG_EXTRACTION_MAX_TOKENS", 8192)
    )
    max_graph_chunks: int = field(default_factory=lambda: _env_int("RAG_KG_MAX_GRAPH_CHUNKS", 3))


# ── Root Settings (single import point for all modules) ───────────────────────


@dataclass(frozen=True)
class Settings:
    """
    Root config object. Import `settings` from `config` in any module:

        from config import settings

        settings.query_router.model           # Layer 1
        settings.query_transform.model        # Layer 2
        settings.retrieval.top_k_final        # Layer 3 search
        settings.reranker.enabled             # Layer 3 reranker
        settings.generation.max_context_tokens  # Layer 4
    """

    query_router: QueryRouterConfig = field(default_factory=QueryRouterConfig)
    query_transform: QueryTransformConfig = field(default_factory=QueryTransformConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    crag: CRAGConfig = field(default_factory=CRAGConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)

    def reload(self) -> None:
        """Reload all configuration sections from current os.environ."""
        object.__setattr__(self, "query_router", QueryRouterConfig())
        object.__setattr__(self, "query_transform", QueryTransformConfig())
        object.__setattr__(self, "generation", GenerationConfig())
        object.__setattr__(self, "embedding", EmbeddingConfig())
        object.__setattr__(self, "retrieval", RetrievalConfig())
        object.__setattr__(self, "reranker", RerankerConfig())
        object.__setattr__(self, "infra", InfraConfig())
        object.__setattr__(self, "crag", CRAGConfig())
        object.__setattr__(self, "evaluation", EvaluationConfig())
        object.__setattr__(self, "observability", ObservabilityConfig())
        object.__setattr__(self, "knowledge_graph", KnowledgeGraphConfig())

        import sys

        if "crag.corrector" in sys.modules:
            setattr(sys.modules["crag.corrector"], "_cfg", self.crag)  # noqa: B010

    def validate(self) -> None:
        if not self.infra.openai_api_key:
            raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
        if not self.infra.qdrant_url:
            raise OSError("QDRANT_URL is not set. Add it to your .env file.")
        if not self.infra.sec_user_agent or self.infra.sec_user_agent == "Your Name your@email.com":
            raise OSError(
                "SEC_USER_AGENT is not set. "
                "Add 'FirstName LastName email@example.com' to your .env file (SEC fair-use policy)."
            )


# Module-level singleton — imported by all other modules
settings = Settings()
