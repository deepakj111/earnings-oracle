"""
API data contracts — Pydantic v2 request and response models.

Design principles:
  - All user-facing strings have Field(description=) for OpenAPI docs
  - Input models validate domain constraints (ticker allow-list, quarter enum)
  - Response models mirror the internal dataclasses but use JSON-safe types
  - Nested models are kept flat where possible — avoids unnecessary wrapping
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ── Domain constants ───────────────────────────────────────────────────────────

_VALID_TICKERS: frozenset[str] = frozenset(
    {"AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"}
)

_VALID_QUARTERS: frozenset[str] = frozenset({"Q1", "Q2", "Q3", "Q4"})


# ── Request models ─────────────────────────────────────────────────────────────


class MetadataFilterIn(BaseModel):
    """
    Optional scoping filter to restrict retrieval to a specific filing subset.

    All fields are individually optional — only set fields are applied.
    Example: {"ticker": "AAPL", "year": 2024} returns only Apple filings
    from fiscal year 2024 regardless of quarter.
    """

    ticker: str | None = Field(
        None,
        description=(
            "Company ticker. Supported: AAPL, NVDA, MSFT, AMZN, META, JPM, XOM, UNH, TSLA, WMT"
        ),
        examples=["AAPL"],
    )
    year: int | None = Field(
        None,
        ge=2020,
        le=2030,
        description="Fiscal year (2020–2030).",
        examples=[2024],
    )
    quarter: str | None = Field(
        None,
        description="Fiscal quarter — Q1, Q2, Q3, or Q4.",
        examples=["Q4"],
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.upper().strip()
        if v not in _VALID_TICKERS:
            raise ValueError(
                f"Unknown ticker '{v}'. Supported: {', '.join(sorted(_VALID_TICKERS))}"
            )
        return v

    @field_validator("quarter")
    @classmethod
    def validate_quarter(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.upper().strip()
        if v not in _VALID_QUARTERS:
            raise ValueError(f"Invalid quarter '{v}'. Must be one of: Q1, Q2, Q3, Q4.")
        return v


class AskRequest(BaseModel):
    """Request body for POST /query and POST /query/stream."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language financial question about SEC 8-K earnings filings.",
        examples=["What was Apple's total revenue in Q4 2024?"],
    )
    filter: MetadataFilterIn | None = Field(
        None,
        description=(
            "Optional scoping filter. Restricts retrieval to matching filings. "
            "Omit to search across all ingested companies and periods."
        ),
    )
    verbose: bool = Field(
        False,
        description=(
            "When true, the response includes query_summary (query transformation details) "
            "and retrieval_summary (ranked chunk diagnostics). Adds negligible overhead."
        ),
    )


# ── Response sub-models ────────────────────────────────────────────────────────


class CitationOut(BaseModel):
    """Metadata for one source reference used in the generated answer."""

    index: int = Field(
        ..., description="1-based citation index — matches [N] inline in the answer text."
    )
    ticker: str
    company: str
    date: str = Field(..., description="Filing date: YYYY-MM-DD")
    fiscal_period: str = Field(..., description="e.g. 'Q4 2024'")
    section_title: str = Field(..., description="Document section the chunk came from.")
    doc_type: str
    source: str = Field(
        ..., description="Retrieval system that surfaced this chunk: 'dense' | 'bm25' | 'both'"
    )
    rerank_score: float = Field(
        ..., description="FlashRank cross-encoder relevance score (higher = more relevant)."
    )
    excerpt: str = Field(..., description="First 250 characters of the source passage.")


class UsageOut(BaseModel):
    """OpenAI token usage and cost estimate for the generation step."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate_usd: float = Field(
        ..., description="Rough cost estimate in USD based on gpt-4o-mini pricing."
    )


class ContextOut(BaseModel):
    """Statistics about the retrieval context window passed to the LLM."""

    chunks_used: int = Field(..., description="Number of document chunks sent to the LLM.")
    tokens_used: int = Field(..., description="Token count of the full context block.")


# ── Primary response model ─────────────────────────────────────────────────────


class AskResponse(BaseModel):
    """
    Full pipeline output returned by POST /query.

    grounded=False signals the model could not ground its answer in the
    retrieved context — a CRAG layer should trigger a web-search fallback
    on this flag.
    """

    question: str
    answer: str = Field(..., description="LLM-synthesised answer with inline [N] citations.")
    citations: list[CitationOut]
    grounded: bool = Field(
        ...,
        description=(
            "False if the model signalled insufficient context in the retrieved documents. "
            "Use as a CRAG trigger for web-search fallback."
        ),
    )
    retrieval_failed: bool = Field(
        ..., description="True if zero documents were retrieved from the index."
    )
    model: str = Field(..., description="OpenAI model used for answer generation.")
    usage: UsageOut
    context: ContextOut
    latency_seconds: float = Field(..., description="End-to-end pipeline wall-clock time.")
    unique_tickers: list[str] = Field(
        ..., description="Deduplicated tickers cited in the answer, in citation order."
    )
    unique_sources: list[str] = Field(
        ...,
        description=(
            "Deduplicated 'TICKER fiscal_period' source labels, e.g. ['AAPL Q4 2024']. "
            "Ready to render in a UI sources list."
        ),
    )
    # Verbose diagnostics — None unless AskRequest.verbose=True
    query_summary: str | None = Field(
        None,
        description="Query transformation details: HyDE doc, multi-queries, step-back query.",
    )
    retrieval_summary: str | None = Field(
        None,
        description="Retrieval diagnostics: candidate counts, rerank scores, sources.",
    )


# ── Health models ─────────────────────────────────────────────────────────────


class ComponentStatus(BaseModel):
    status: str = Field(..., description="'ok' | 'error'")
    detail: str | None = Field(None, description="Human-readable status detail or error message.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status: 'healthy' | 'degraded' | 'unhealthy'")
    version: str
    uptime_seconds: float
    components: dict[str, ComponentStatus] = Field(..., description="Per-dependency health status.")
