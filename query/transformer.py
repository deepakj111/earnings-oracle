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
  Transformation  → gpt-4o-mini   ($0.15/1M in, $0.60/1M out)  cheap + fast
  Answer gen      → gpt-4.1       (future generation/ layer)    capable + accurate

Graceful degradation: if any single technique fails after retries, that
technique falls back to the original query and execution continues. A full
hard failure only occurs if ALL techniques fail simultaneously.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

# ── Configuration (all overridable via environment) ───────────────────────────
# ── NEW: import from config ────────────────────────────────────────────────────
from config import settings as _settings
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


# ── OpenAI client (lazy singleton, thread-safe after first init) ──────────────

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = _settings.infra.openai_api_key
        if not api_key:
            raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _openai_client = OpenAI(api_key=api_key, max_retries=0)
        logger.info(f"OpenAI client initialised | query model={QUERY_TRANSFORM_MODEL}")
    return _openai_client


# ── In-memory LRU cache ────────────────────────────────────────────────────────
# Avoids redundant API calls when the same query appears multiple times in a
# session (e.g., during evaluation, CRAG loops, or UI demos).

_cache: dict[str, TransformedQuery] = {}
_cache_insertion_order: list[str] = []


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _cache_get(key: str) -> TransformedQuery | None:
    return _cache.get(key)


def _cache_put(key: str, result: TransformedQuery) -> None:
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest = _cache_insertion_order.pop(0)
        _cache.pop(oldest, None)
    _cache[key] = result
    _cache_insertion_order.append(key)


# ── Core LLM call with exponential backoff ────────────────────────────────────


def _call_llm(
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    label: str,
) -> str:
    """
    Single OpenAI chat completion call with exponential backoff on transient errors.

    Retries on: RateLimitError, APITimeoutError, APIError (5xx).
    Propagates on: AuthenticationError, InvalidRequestError (unrecoverable).

    Args:
        system:      System prompt string.
        user:        User message string.
        temperature: Sampling temperature for this technique.
        max_tokens:  Upper bound on response length.
        label:       Technique name for log context ("HyDE", "MultiQuery", "StepBack").

    Returns:
        Stripped response text from the model.
    """
    client = _get_client()
    delay = BASE_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=QUERY_TRANSFORM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            text = text.strip()

            if not text:
                raise ValueError(f"[{label}] Model returned an empty response.")

            if response.usage:
                logger.debug(
                    f"[{label}] tokens | "
                    f"in={response.usage.prompt_tokens} "
                    f"out={response.usage.completion_tokens}"
                )
            return text

        except (RateLimitError, APITimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                f"[{label}] Transient error (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2.0

        except APIError as exc:
            # Retry on 5xx server errors, propagate on 4xx client errors
            if exc.status_code is not None and exc.status_code < 500:
                raise
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                f"[{label}] Server error {exc.status_code} (attempt {attempt}/{MAX_RETRIES}). "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"[{label}] All {MAX_RETRIES} retry attempts exhausted.")


# ── Individual technique implementations ──────────────────────────────────────


def _run_hyde(query: str) -> str:
    """
    Technique 1: Hypothetical Document Embeddings.

    Asks the model to write a passage as if it came from an earnings press
    release. The resulting text is embedded (using the same fastembed model
    as the index) instead of the raw query — this closes the query-document
    semantic gap because the hypothetical passage lives in the same region
    of embedding space as real document chunks.
    """
    user = HYDE_USER.format(query=query)
    return _call_llm(
        system=HYDE_SYSTEM,
        user=user,
        temperature=_TEMP_HYDE,
        max_tokens=_cfg.max_tokens_hyde,
        label="HyDE",
    )


def _run_multi_query(query: str) -> list[str]:
    """
    Technique 2: Multi-Query Generation.

    Generates 3 rephrasings of the original query. Each rephrasing is sent
    to retrieval independently, and results are unioned before RRF fusion.
    This increases recall — the probability that the correct chunk appears
    somewhere in the retrieval pool.

    The original query is always prepended as query[0] to ensure it is
    never dropped from retrieval.
    """
    user = MULTI_QUERY_USER.format(query=query)
    raw = _call_llm(
        system=MULTI_QUERY_SYSTEM,
        user=user,
        temperature=_TEMP_MULTI,
        max_tokens=_cfg.max_tokens_multi_query,
        label="MultiQuery",
    )
    lines = [
        line.lstrip("0123456789.-) •*").strip()
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


def _run_stepback(query: str) -> str:
    """
    Technique 3: Step-Back Prompting.

    Generates a broader, more abstract version of the query. This abstract
    query retrieves foundational context chunks that the specific query
    would miss (e.g., segment definitions, management commentary on strategy,
    metric calculation methodology). Both specific and abstract results are
    combined before reranking.
    """
    user = STEPBACK_USER.format(query=query)
    return _call_llm(
        system=STEPBACK_SYSTEM,
        user=user,
        temperature=_TEMP_STEPBACK,
        max_tokens=_cfg.max_tokens_stepback,
        label="StepBack",
    )


# ── Public transformer class ──────────────────────────────────────────────────


class QueryTransformer:
    """
    Layer 2: Query Transformation for Financial RAG.

    Usage:
        transformer = QueryTransformer()
        result = transformer.transform("How did Apple's revenue guidance change?")

        # result.hyde_document    → embed this with fastembed for dense retrieval
        # result.all_retrieval_queries → fan out to BM25 + dense retrieval
        # result.stepback_query   → included in all_retrieval_queries

    All three techniques run concurrently (ThreadPoolExecutor, 3 workers).
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

    def transform(self, query: str) -> TransformedQuery:
        """
        Apply HyDE, Multi-Query, and Step-Back concurrently to a single query.

        Args:
            query: The raw user query string (non-empty).

        Returns:
            TransformedQuery with all three technique outputs populated.

        Raises:
            ValueError: If query is empty.
            EnvironmentError: If OPENAI_API_KEY is missing.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        if self.enable_cache:
            cached = _cache_get(_cache_key(query))
            if cached is not None:
                logger.debug(f"Cache hit | query={query!r}")
                return cached

        logger.info(f"Transforming query | {query!r}")
        start = time.perf_counter()
        failed_techniques: list[str] = []

        # Fire all three techniques concurrently.
        # Each runs in its own thread — safe because OpenAI SDK is thread-safe.
        hyde_doc: str = query  # fallback: original query
        multi_queries: list[str] = [query]  # fallback: only original
        stepback_query: str = query  # fallback: original query

        tasks = {
            "hyde": _run_hyde,
            "multi": _run_multi_query,
            "stepback": _run_stepback,
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn, query): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "hyde":
                        hyde_doc = result
                    elif name == "multi":
                        multi_queries = result
                    else:
                        stepback_query = result
                except Exception as exc:
                    failed_techniques.append(name)
                    logger.warning(f"[{name}] failed, using fallback. Error: {exc}")

        elapsed = time.perf_counter() - start
        logger.info(
            f"Transformation complete | "
            f"{len(multi_queries)} multi-queries | "
            f"{elapsed:.2f}s" + (f" | degraded={failed_techniques}" if failed_techniques else "")
        )

        transformed = TransformedQuery(
            original=query,
            hyde_document=hyde_doc,
            multi_queries=multi_queries,
            stepback_query=stepback_query,
            failed_techniques=failed_techniques,
        )

        if self.enable_cache:
            _cache_put(_cache_key(query), transformed)

        return transformed
