"""
Layer 2b — Sub-Query Decomposition (2026 SOTA Agentic Multi-Hop Retrieval).

Decomposes complex, multi-part, or comparative financial questions into
atomic, self-contained sub-queries. Each sub-query targets an isolated entity,
fiscal period, or financial metric.

If the question is already atomic/simple, the decomposer returns the original
question directly, avoiding unnecessary LLM latency and context fragmentation.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from config.llm_client import aparse

_cfg = settings.query_transform

_DECOMPOSER_SYSTEM_PROMPT = """You are a financial query decomposition specialist for an SEC 10-K and 10-Q RAG system.
Your job is to analyze user queries and determine if they require multi-hop or comparative retrieval.

Rules:
1. If the user query asks for a single metric, definition, or fact for one company in one period, mark is_complex=False and provide an empty sub_queries list.
2. If the query compares two or more companies (e.g., Apple vs Microsoft), compares two or more distinct fiscal periods (e.g., Q1 2024 vs Q1 2023), or asks for multiple distinct financial metrics requiring different sections (e.g., revenue AND risk factors), mark is_complex=True and break it down into 2 to 4 atomic, self-contained sub-queries.
3. Every generated sub-query MUST be fully self-contained: explicitly name the company/ticker, fiscal year/quarter, and exact financial metric. Do NOT use pronouns like "it", "them", or "both".
4. Keep sub-queries focused, concise, and optimized for dense and BM25 search.
"""


class DecomposedQueryPlan(BaseModel):
    """Structured decomposition plan for complex queries."""

    is_complex: bool = Field(
        description="True if the query requires multi-hop, multi-entity, or multi-period retrieval; False if already atomic."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="List of 2 to 4 atomic, self-contained sub-queries if is_complex is True; empty if is_complex is False.",
    )
    reasoning: str = Field(
        description="Brief rationale explaining why decomposition was or was not applied."
    )


class QueryDecomposer:
    """Agentic query decomposition component for financial RAG."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or _cfg.model
        self._enabled = _cfg.decomposition_enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def _call_llm(self, question: str) -> DecomposedQueryPlan:
        """
        Call the LLM for structured query decomposition via config.llm_client (provider-agnostic).

        Retry logic (tenacity exponential backoff) is handled inside aparse().
        """
        return await aparse(
            messages=[
                {"role": "system", "content": _DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {question}"},
            ],
            schema=DecomposedQueryPlan,
            model=self._model,
            temperature=0.1,
            max_tokens=_cfg.max_tokens_decomposition,
        )

    async def decompose(self, question: str) -> tuple[bool, list[str], str]:
        """
        Analyze and optionally decompose a question into atomic sub-queries.

        Returns:
            (is_decomposed: bool, sub_queries: list[str], reasoning: str)
            If the question is atomic or decomposition is disabled, returns
            (False, [question], "Original atomic query retained").
        """
        clean_q = question.strip()
        if not clean_q:
            return False, [], "Empty query"

        if not self._enabled:
            return False, [clean_q], "Query decomposition disabled via configuration"

        try:
            plan = await self._call_llm(clean_q)
            if plan.is_complex and plan.sub_queries:
                # Filter out any degenerate empty strings
                valid_subs = [q.strip() for q in plan.sub_queries if q.strip()]
                if len(valid_subs) >= 2:
                    logger.info(
                        f"[Decomposer] Query decomposed into {len(valid_subs)} atomic sub-queries: "
                        f"{valid_subs} | Reason: {plan.reasoning}"
                    )
                    return True, valid_subs, plan.reasoning

            logger.info(f"[Decomposer] Query evaluated as atomic: {clean_q!r:.60}")
            return False, [clean_q], plan.reasoning
        except Exception as exc:
            logger.warning(
                f"[Decomposer] Decomposition failed, falling back to original query: {exc}"
            )
            return False, [clean_q], f"Fallback due to error: {exc}"


__all__ = ["QueryDecomposer", "DecomposedQueryPlan"]
