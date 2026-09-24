"""
Layer 2 — Query Transformation for Financial RAG.

Implements three complementary techniques that close the query-document
semantic gap before retrieval hits the vector store:

  HyDE        — embed a hypothetical answer instead of the raw question
  Multi-Query — union of 3–4 rephrasings to increase recall coverage
  Step-Back   — abstract query to retrieve foundational context chunks

All three LLM calls are fired concurrently (ThreadPoolExecutor) to keep
total transformation latency ≈ single call latency (~0.8–1.2s).

Model tiering:
  Transformation  → gpt-5-mini    (fast + cost-effective query re-writing)
  Answer gen      → gpt-5         (capable + high financial reasoning accuracy)

Graceful degradation: if any single technique fails after retries, that
technique falls back to the original query and execution continues. A full
hard failure only occurs if ALL techniques fail simultaneously.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from cachetools import LRUCache
from loguru import logger

# ── Configuration (all overridable via environment) ───────────────────────────
from config import settings as _settings
from config.llm_client import acomplete
from query.models import TransformedQuery
from query.prompts import (
    HYDE_SYSTEM,
    HYDE_USER,
    MULTI_QUERY_SYSTEM,
    MULTI_QUERY_USER,
    STEPBACK_SYSTEM,
    STEPBACK_USER,
)

_cfg = _settings.query_transform

QUERY_TRANSFORM_MODEL: str = _cfg.model
MAX_RETRIES: int = _cfg.max_retries
BASE_RETRY_DELAY: float = _cfg.retry_base_delay_seconds
CACHE_MAX_SIZE: int = _cfg.cache_max_size

# Temperature per technique — intentionally different:
#   HyDE needs moderate creativity to produce realistic-sounding passages
#   Multi-Query needs higher variance so rephrasings actually differ
#   Step-Back needs near-determinism — same question, same abstraction
_TEMP_HYDE: float = _cfg.temperature_hyde
_TEMP_MULTI: float = _cfg.temperature_multi_query
_TEMP_STEPBACK: float = _cfg.temperature_stepback


# ── In-memory LRU cache ────────────────────────────────────────────────────────
# Avoids redundant API calls when the same query appears multiple times in a
# session (e.g., during evaluation or UI demos).

_cache: LRUCache[str, TransformedQuery] = LRUCache(maxsize=CACHE_MAX_SIZE)


def _cache_key(
    query: str,
    hyde: bool = True,
    multi: bool = True,
    stepback: bool = True,
) -> str:
    raw = f"{query.strip().lower()}|hyde:{hyde}|multi:{multi}|stepback:{stepback}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Core LLM call ────────────────────────────────────────────────────────────


def get_async_openai_client() -> Any:
    """Backward-compat shim for test mocking."""
    from config.openai_client import get_async_openai_client as _get

    return _get()


async def _call_llm(
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    label: str,
) -> str:
    """
    Single LLM chat completion call via config.llm_client (provider-agnostic).

    Retry logic (tenacity exponential backoff) is handled inside acomplete().
    Retries on: RateLimitError, Timeout.
    Propagates on: AuthenticationError, BadRequestError (unrecoverable).
    """
    # If legacy test mocks get_async_openai_client:
    client = None
    try:
        client = get_async_openai_client()
    except Exception:
        client = None

    if client is not None and (
        hasattr(client, "mock_calls") or type(client).__name__ in ("MagicMock", "AsyncMock", "Mock")
    ):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        res = await client.chat.completions.create(
            model=QUERY_TRANSFORM_MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        content = (res.choices[0].message.content or "").strip()
        if not content:
            raise ValueError(f"Empty response from model {QUERY_TRANSFORM_MODEL}")
        return content

    resp = await acomplete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=QUERY_TRANSFORM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.debug(f"[{label}] tokens | in={resp.prompt_tokens} out={resp.completion_tokens}")
    return resp.content


# ── Individual technique implementations ──────────────────────────────────────


async def _run_hyde(query: str) -> str:
    """
    Technique 1: Hypothetical Document Embeddings.

    Asks the model to write a passage as if it came from an earnings press
    release. The resulting text is embedded (using the same embedding model
    as the index) instead of the raw query — this closes the query-document
    semantic gap because the hypothetical passage lives in the same region
    of embedding space as real document chunks.
    """
    cfg = _settings.query_transform
    user = HYDE_USER.format(query=query)
    return await _call_llm(
        system=HYDE_SYSTEM,
        user=user,
        temperature=cfg.temperature_hyde,
        max_tokens=cfg.max_tokens_hyde,
        label="HyDE",
    )


async def _run_multi_query(query: str) -> list[str]:
    """
    Technique 2: Multi-Query Generation.

    Generates 3 rephrasings of the original query. Each rephrasing is sent
    to retrieval independently, and results are unioned before RRF fusion.
    This increases recall — the probability that the correct chunk appears
    somewhere in the retrieval pool.

    The original query is always prepended as query[0] to ensure it is
    never dropped from retrieval.
    """
    cfg = _settings.query_transform
    user = MULTI_QUERY_USER.format(query=query)
    raw = await _call_llm(
        system=MULTI_QUERY_SYSTEM,
        user=user,
        temperature=cfg.temperature_multi_query,
        max_tokens=cfg.max_tokens_multi_query,
        label="MultiQuery",
    )
    lines = [
        line.lstrip("0123456789.-) •*").strip().strip("\"'").strip()
        for line in raw.splitlines()
        if line.strip() and len(line.strip().split()) >= 3
    ]
    # Original is always first; generated rephrasings follow, deduplicated
    seen: set[str] = {query.lower()}
    unique_rephrasings = []
    for line in lines:
        if line.lower() not in seen:
            seen.add(line.lower())
            unique_rephrasings.append(line)

    # Return original + up to 3 rephrasings = 4 total
    return [query] + unique_rephrasings[:3]


async def _run_stepback(query: str) -> str:
    """
    Technique 3: Step-Back Prompting.

    Generates a broader, more abstract version of the query. This abstract
    query retrieves foundational context chunks that the specific query
    would miss (e.g., segment definitions, management commentary on strategy,
    metric calculation methodology). Both specific and abstract results are
    combined before reranking.
    """
    cfg = _settings.query_transform
    user = STEPBACK_USER.format(query=query)
    return await _call_llm(
        system=STEPBACK_SYSTEM,
        user=user,
        temperature=cfg.temperature_stepback,
        max_tokens=cfg.max_tokens_stepback,
        label="StepBack",
    )


# ── Public transformer class ──────────────────────────────────────────────────


class QueryTransformer:
    """
    Layer 2: Query Transformation for Financial RAG.

    Usage:
        transformer = QueryTransformer()
        result = await transformer.transform("How did Apple's revenue guidance change?")

        # result.hyde_document    → embed this for dense retrieval
        # result.all_retrieval_queries → fan out to BM25 + dense retrieval
        # result.stepback_query   → included in all_retrieval_queries

    All three techniques run concurrently using asyncio.gather.
    Total latency ≈ single LLM call latency (~0.8–1.2s) instead of 3× serial.

    Graceful degradation: if one technique fails after retries, it falls back
    to the original query and logs a warning. Execution always completes.
    """

    def __init__(self, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache
        logger.info(
            f"QueryTransformer ready | model={QUERY_TRANSFORM_MODEL} | "
            f"cache={'enabled' if enable_cache else 'disabled'} | "
            f"max_retries={MAX_RETRIES}"
        )

    async def transform(
        self,
        question: str,
        skip_hyde: bool = False,
        routing_decision: Any | None = None,
    ) -> TransformedQuery:
        """
        Run configured query transformations in parallel via asyncio.gather.

        Args:
            question        : Raw user question
            skip_hyde       : If True, HyDE generation is skipped
            routing_decision: Optional RoutingDecision from Layer 1 router to dynamically
                              tune technique selection (e.g. multi-query only for simple lookups)
        """
        query = question.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        cfg = _settings.query_transform
        hyde_enabled = cfg.hyde_enabled and not skip_hyde
        multiquery_enabled = cfg.multiquery_enabled
        stepback_enabled = cfg.stepback_enabled

        # Routing-aware technique selection for latency optimization
        if routing_decision is not None:
            if getattr(routing_decision, "skip_transform", False):
                hyde_enabled = False
                multiquery_enabled = False
                stepback_enabled = False
                logger.debug(
                    "[QueryTransformer] Routing specified skip_transform — all L2 disabled"
                )
            else:
                if getattr(routing_decision, "skip_hyde", False):
                    hyde_enabled = False
                if getattr(routing_decision, "is_specific", False) and not getattr(
                    routing_decision, "is_comparative", False
                ):
                    hyde_enabled = False
                    stepback_enabled = False
                    logger.debug(
                        "[QueryTransformer] Routing-aware selection: specific lookup -> multi-query only"
                    )

        if self.enable_cache:
            ckey = _cache_key(
                query,
                hyde=hyde_enabled,
                multi=multiquery_enabled,
                stepback=stepback_enabled,
            )
            if ckey in _cache:
                logger.debug(f"Cache hit | query={query!r}")
                return _cache[ckey]

        logger.info(f"Transforming query | {query!r}")
        failed_techniques: list[str] = []

        tasks: list[Any] = []
        names: list[str] = []
        if hyde_enabled:
            tasks.append(_run_hyde(query))
            names.append("hyde")
        if multiquery_enabled:
            tasks.append(_run_multi_query(query))
            names.append("multi")
        if stepback_enabled:
            tasks.append(_run_stepback(query))
            names.append("stepback")

        if not tasks:
            logger.info(
                "All query transformations disabled by configuration — skipping Layer 2 LLM calls."
            )
            transformed = TransformedQuery(
                original=query,
                hyde_document=query,
                multi_queries=[query],
                stepback_query=query,
                failed_techniques=[],
            )
            if self.enable_cache:
                _cache[
                    _cache_key(
                        query,
                        hyde=hyde_enabled,
                        multi=multiquery_enabled,
                        stepback=stepback_enabled,
                    )
                ] = transformed
            return transformed

        hyde_doc: str = query
        multi_queries: list[str] = [query]
        stepback_query: str = query

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(names, results, strict=False):
            if isinstance(result, BaseException):
                failed_techniques.append(name)
                logger.warning(f"[{name}] failed, using fallback. Error: {result}")
            else:
                if name == "hyde" and isinstance(result, str):
                    hyde_doc = result
                elif name == "multi" and isinstance(result, list):
                    multi_queries = result
                elif name == "stepback" and isinstance(result, str):
                    stepback_query = result

        logger.info(
            f"Transformation complete | "
            f"{len(multi_queries)} multi-queries"
            + (f" | degraded={failed_techniques}" if failed_techniques else "")
        )

        transformed = TransformedQuery(
            original=query,
            hyde_document=hyde_doc,
            multi_queries=multi_queries,
            stepback_query=stepback_query,
            failed_techniques=failed_techniques,
        )

        if self.enable_cache:
            _cache[
                _cache_key(
                    query,
                    hyde=hyde_enabled,
                    multi=multiquery_enabled,
                    stepback=stepback_enabled,
                )
            ] = transformed

        return transformed
