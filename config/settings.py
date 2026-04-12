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


# ── Layer 2: Query Transformation ─────────────────────────────────────────────


@dataclass(frozen=True)
class QueryTransformConfig:
    """
    Configuration for query/transformer.py (Layer 2 — HyDE + Multi-Query + Step-Back).

    Model tier rationale:
      gpt-5-nano — $0.05/1M input, $0.005/1M output.
      Query transformation prompts are short (~120–350 tokens input, ~80–250 output).
      At this scale, nano-tier models are more than sufficient for instruction-following
      tasks like rephrasing and hypothetical passage generation.
    """

    model: str = field(default_factory=lambda: _env_str("RAG_QUERY_TRANSFORM_MODEL", "gpt-5-nano"))

    # Per-technique temperatures — intentionally different
    # HyDE: moderate creativity for realistic-sounding passages
    # Multi-Query: higher variance so rephrasings actually differ
    # Step-Back: near-determinism — same question, same abstraction
    temperature_hyde: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_HYDE", 0.3)
    )
    temperature_multi_query: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_MULTI_QUERY", 0.7)
    )
    temperature_stepback: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_TEMP_STEPBACK", 0.1)
    )

    # Max output tokens per technique — kept tight to control cost
    max_tokens_hyde: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_HYDE", 250)
    )
    max_tokens_multi_query: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_MULTI_QUERY", 200)
    )
    max_tokens_stepback: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_TOKENS_STEPBACK", 100)
    )

    # Retry / backoff
    max_retries: int = field(default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_MAX_RETRIES", 3))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("RAG_QUERY_TRANSFORM_RETRY_DELAY", 1.0)
    )

    # In-memory LRU cache size (number of distinct queries to cache per session)
    cache_max_size: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_TRANSFORM_CACHE_SIZE", 256)
    )


# ── Layer 4: Generation (placeholder — populated when generation/ is built) ───


@dataclass(frozen=True)
class GenerationConfig:
    """
    Configuration for generation/generator.py (Layer 4 — answer synthesis).

    Model tier rationale:
      gpt-5-nano — $0.05/1M input, $0.005/1M output.
      Generation prompts are long (retrieved context = 2000–5000 tokens).
      Mini-tier provides better reasoning over dense financial text than nano.
    """

    model: str = field(default_factory=lambda: _env_str("RAG_GENERATION_MODEL", "gpt-5-nano"))
    temperature: float = field(default_factory=lambda: _env_float("RAG_GENERATION_TEMP", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_GENERATION_MAX_TOKENS", 1024))
    max_retries: int = field(default_factory=lambda: _env_int("RAG_GENERATION_MAX_RETRIES", 3))
    retry_base_delay_seconds: float = field(
        default_factory=lambda: _env_float("RAG_GENERATION_RETRY_DELAY", 1.0)
    )


# ── Ingestion / Embedding ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for ingestion/indexer.py (fastembed + Qdrant).
    Kept here so retrieval layer can reference the same model name
    when embedding HyDE documents for dense search.
    """

    model: str = field(
        default_factory=lambda: _env_str("RAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    )
    vector_dim: int = field(default_factory=lambda: _env_int("RAG_EMBEDDING_VECTOR_DIM", 1024))
    collection_name: str = field(
        default_factory=lambda: _env_str("RAG_QDRANT_COLLECTION", "earnings_transcripts")
    )
    upsert_batch_size: int = field(
        default_factory=lambda: _env_int("RAG_EMBEDDING_UPSERT_BATCH_SIZE", 50)
    )


# ── Retrieval ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Configuration for retrieval/searcher.py (Layer 3 — BM25 + Qdrant + RRF).
    Populated now so retrieval/ module can import these on implementation.
    """

    top_k_dense: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_DENSE", 10))
    top_k_bm25: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_BM25", 10))
    top_k_final: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_TOP_K_FINAL", 5))
    rrf_k_constant: int = field(default_factory=lambda: _env_int("RAG_RETRIEVAL_RRF_K", 60))


# ── Infrastructure ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InfraConfig:
    """Infrastructure endpoints and secrets. All four values come from .env only."""

    qdrant_url: str = field(default_factory=lambda: _env_str("QDRANT_URL", "http://localhost:6333"))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: _env_str("SEC_USER_AGENT", "Your Name your@email.com")
    )
    # Read by fastembed library automatically — surfaced here for documentation/validation only
    fastembed_cache_path: str = field(default_factory=lambda: _env_str("FASTEMBED_CACHE_PATH", ""))


# ── Root Settings (single import point for all modules) ───────────────────────


@dataclass(frozen=True)
class Settings:
    """
    Root config object. Import `settings` from `config` in any module:

        from config import settings
        model = settings.query_transform.model
    """

    query_transform: QueryTransformConfig = field(default_factory=QueryTransformConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

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
