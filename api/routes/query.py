# api/routes/query.py
"""
Query routes — POST /query and POST /query/stream.


Both routes delegate to the same FinancialRAGPipeline but differ in how
they return results:


  POST /query        — runs the full pipeline synchronously (in a thread) and
                       returns a fully structured AskResponse JSON body with
                       citations, token usage, grounding flag, and diagnostics.


  POST /query/stream — runs L2 (query transformation) + L3 (retrieval) in a
                       thread, then streams L4 (generation) tokens as
                       Server-Sent Events.  No structured citation data is
                       returned — use /query when you need metadata.


Threading model:
  FastAPI runs on an async event loop (uvicorn + asyncio).  The RAG pipeline
  is entirely synchronous (OpenAI SDK, BM25, FlashRank are all
  blocking calls).  We offload every pipeline call to a ThreadPoolExecutor so
  the event loop stays free to handle other requests concurrently.


  _THREAD_POOL is module-level with max_workers=4 — tune based on:
    - OpenAI rate limits (tokens-per-minute / requests-per-minute)
    - Available CPU cores for FlashRank
    - Expected concurrent users


SSE message format (POST /query/stream):
  data: {"token": "<text delta>"}\n\n   — one per LLM token
  data: {"error": "<message>"}\n\n      — if generation fails mid-stream
  data: [DONE]\n\n                      — always the final message


Metrics instrumentation (NEW):
  record_generation_result() is called after every successful ask() call.
  record_retrieval_result() is not called here because retrieval stats
  are embedded in GenerationResult (context_chunks_used, context_tokens_used).
  Streaming calls do NOT produce a GenerationResult, so LLM metrics are
  not recorded for stream requests (HTTP-level metrics still fire via middleware).


FIX 1 (Bug): asyncio.get_event_loop() was used in both ask() and ask_stream().
  In Python 3.10+ this raises DeprecationWarning inside an already-running loop.
  Replaced with asyncio.get_running_loop() which is always correct inside an
  async handler and raises RuntimeError (not silently degrades) if called
  outside an event loop.


FIX 2 (Bug): Producer future in _consume() was silently discarded.
  loop.run_in_executor() returns an asyncio.Future.  If _produce() raised
  before placing the sentinel on the queue, _consume() would block until the
  60-second timeout.  The future is now stored and awaited in the finally block
  so any exception propagates and the stream closes cleanly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from api.dependencies import get_pipeline
from api.metrics import record_generation_result  # ← NEW
from api.models import AskRequest, AskResponse, CitationOut, ContextOut, UsageOut
from generation.models import GenerationResult
from query.guardrails import QueryGuardrails
from rag_pipeline import FinancialRAGPipeline
from retrieval.models import MetadataFilter

router = APIRouter(redirect_slashes=False)


# ── Thread pool for blocking pipeline calls ────────────────────────────────────
# Named threads make profiling and debugging easier (visible in py-spy / pystack).
_THREAD_POOL = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="rag-worker",
)


# Typed dependency alias for cleaner route signatures
_Pipeline = Annotated[FinancialRAGPipeline, Depends(get_pipeline)]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _to_metadata_filter(req: AskRequest) -> MetadataFilter | None:
    """
    Convert the API-level MetadataFilterIn model to the retrieval-level
    MetadataFilter dataclass.  Returns None if no filter fields are set.
    """
    if req.filter is None:
        return None
    f = req.filter
    if not any([f.ticker, f.year, f.quarter]):
        return None
    return MetadataFilter(ticker=f.ticker, year=f.year, quarter=f.quarter)


def _serialise(
    result: GenerationResult,
    *,
    verbose: bool,
    query_summary: str | None,
    retrieval_summary: str | None,
    overall_latency: float | None = None,
) -> AskResponse:
    """
    Convert the internal GenerationResult dataclass into the public AskResponse.

    Verbose fields are included only when the caller requested them to keep
    the default response payload compact.
    """
    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[
            CitationOut(
                index=c.index,
                ticker=c.ticker,
                company=c.company,
                date=c.date,
                fiscal_period=c.fiscal_period,
                section_title=c.section_title,
                doc_type=c.doc_type,
                source=c.source,
                rerank_score=round(c.rerank_score, 4),
                excerpt=c.excerpt,
            )
            for c in result.citations
        ],
        grounded=result.grounded,
        confidence_score=round(
            getattr(result, "confidence_score", None)
            if getattr(result, "confidence_score", 1.0) != 1.0
            else getattr(result, "computed_confidence_score", 1.0),
            3,
        ),
        retrieval_failed=result.retrieval_failed,
        model=result.model,
        usage=UsageOut(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
        context=ContextOut(
            chunks_used=result.context_chunks_used,
            tokens_used=result.context_tokens_used,
        ),
        latency_seconds=round(
            overall_latency if overall_latency is not None else result.latency_seconds, 3
        ),
        unique_tickers=result.unique_tickers,
        unique_sources=result.unique_sources,
        calculations=getattr(result, "calculations", []),
        numerical_hallucination_warnings=getattr(result, "numerical_hallucination_warnings", []),
        citation_integrity_warnings=getattr(result, "citation_integrity_warnings", []),
        reflexion_attempts=getattr(result, "reflexion_attempts", 0),
        query_summary=query_summary if verbose else None,
        retrieval_summary=retrieval_summary if verbose else None,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=AskResponse,
    summary="Ask a financial question (structured response)",
    description=(
        "Run the full 4-layer Financial RAG pipeline and return a structured JSON "
        "answer with inline citations, token usage, and grounding diagnostics.\n\n"
        "**Pipeline layers executed:**\n"
        "- **L2 Query Transformation** — HyDE + multi-query + step-back "
        "(3 concurrent LLM calls, ~0.8–1.2 s)\n"
        "- **L3 Hybrid Retrieval** — BM25 + Qdrant dense search → RRF fusion → "
        "FlashRank cross-encoder reranking (~0.3–0.8 s)\n"
        "- **L4 Answer Generation** — GPT synthesis with grounded [N] citations and PAL Math (~0.8–1.8 s)\n\n"
        "**Total typical latency:** 1.5–2.5 s.\n\n"
        "Set `verbose=true` to receive `query_summary` and `retrieval_summary` "
        "diagnostic strings in the response — useful for debugging and evaluation."
    ),
    responses={
        200: {"description": "Answer synthesised with citations."},
        400: {"description": "Invalid request (empty question, unknown ticker, etc.)."},
        422: {"description": "Pydantic validation error."},
        429: {"description": "OpenAI rate limit exceeded — retry after 10 s."},
        503: {
            "description": (
                "BM25 index or Qdrant collection not found. "
                "Run `poetry run python -m ingestion.pipeline` first."
            )
        },
        504: {"description": "LLM API timeout after all retries."},
    },
)
async def ask(
    body: AskRequest,
    pipeline: _Pipeline,
    request: Request,
) -> AskResponse:
    """Respond to a financial question using the RAG pipeline."""
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        f"[{rid}] POST /query | "
        f"q={body.question!r:.80} | "
        f"filter={body.filter} | verbose={body.verbose}"
    )

    # ── Input Guardrails ───────────────────────────────────────────────────
    guardrails = QueryGuardrails()
    gr_result = guardrails.validate(body.question)
    if not gr_result.passed and gr_result.has_critical_violation:
        violation_messages = "; ".join(v.message for v in gr_result.violations)
        logger.warning(f"[{rid}] Query rejected by security guardrails: {violation_messages}")
        raise ValueError(f"Security guardrail rejected query: {violation_messages}")

    sanitized_question = gr_result.sanitized_query
    metadata_filter = _to_metadata_filter(body)

    if body.verbose:
        result, query_summary, retrieval_summary = await pipeline.ask_verbose(
            question=sanitized_question,
            metadata_filter=metadata_filter,
        )
    else:
        result = await pipeline.ask(
            question=sanitized_question,
            metadata_filter=metadata_filter,
            request_id=rid,
            endpoint="/query",
        )
        query_summary = retrieval_summary = None

    record_generation_result(result)

    logger.info(
        f"[{rid}] answer ready | grounded={result.grounded} | "
        f"citations={len(result.citations)} | tokens={result.total_tokens} | "
        f"latency={result.latency_seconds:.2f}s"
    )

    return _serialise(
        result,
        verbose=body.verbose,
        query_summary=query_summary,
        retrieval_summary=retrieval_summary,
    )


@router.post(
    "/stream",
    summary="Ask a financial question (Server-Sent Events stream)",
    description=(
        "Streaming variant of POST /query.  Runs L2 (query transformation) and "
        "L3 (retrieval) synchronously, then streams L4 (answer generation) tokens "
        "as Server-Sent Events for progressive UI rendering.\n\n"
        "**SSE message format:**\n"
        "```\n"
        'data: {"token": "Apple"}\n\n'
        'data: {"token": " reported"}\n\n'
        "data: [DONE]\n\n"
        "```\n\n"
        "**Error events** (stream still terminates with [DONE]):\n"
        "```\n"
        'data: {"error": "<message>"}\n\n'
        "data: [DONE]\n\n"
        "```\n\n"
        "**Note:** No citation metadata or token counts are available in streaming "
        "mode.  Use POST /query when you need structured output."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Event stream of answer tokens.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def ask_stream(
    body: AskRequest,
    pipeline: _Pipeline,
    request: Request,
) -> StreamingResponse:
    """Stream response tokens for a financial question."""
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"[{rid}] POST /query/stream | q={body.question!r:.80}")

    metadata_filter = _to_metadata_filter(body)

    async def _consume() -> AsyncGenerator[str, None]:
        """
        Async consumer: reads tokens directly and yields SSE data strings.
        """
        try:
            stream: Any = pipeline.ask_streaming(
                question=body.question,
                metadata_filter=metadata_filter,
                request_id=rid,
                endpoint="/query/stream",
            )
            if hasattr(stream, "__aiter__"):
                async for item in stream:
                    if isinstance(item, dict):
                        payload = json.dumps(item)
                    else:
                        payload = json.dumps({"token": item})
                    yield f"data: {payload}\n\n"
            else:
                for item in stream:
                    if isinstance(item, dict):
                        payload = json.dumps(item)
                    else:
                        payload = json.dumps({"token": item})
                    yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.error(f"[{rid}] Streaming pipeline error: {type(exc).__name__}: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _consume(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx response buffering for SSE
            "X-Request-ID": rid,
        },
    )


@router.post(
    "/cache/invalidate",
    summary="Invalidate semantic cache entries",
    description=(
        "Invalidate cached query responses in Qdrant. "
        "Provide a ticker (e.g. ?ticker=NVDA) to invalidate entries for a specific company, "
        "or omit to flush the entire semantic cache."
    ),
    responses={
        200: {"description": "Cache entries invalidated successfully."},
    },
)
async def invalidate_cache(
    pipeline: _Pipeline,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Invalidate semantic cache entries by ticker or flush the entire cache."""
    cache = getattr(pipeline, "_cache", None)
    if cache is None:
        return {"status": "skipped", "message": "Semantic cache not configured", "deleted": 0}

    if ticker:
        t_clean = ticker.strip().upper()
        count = await cache.invalidate_ticker(t_clean)
        return {"status": "success", "ticker": t_clean, "deleted": count}

    try:
        if await cache.client.collection_exists(cache.COLLECTION_NAME):
            await cache.client.delete_collection(cache.COLLECTION_NAME)
            cache._initialized = False
            await cache._ensure_collection()
        return {"status": "success", "message": "Semantic cache flushed completely"}
    except Exception as exc:
        logger.warning(f"Failed to flush semantic cache: {exc}")
        return {"status": "error", "message": str(exc)}
