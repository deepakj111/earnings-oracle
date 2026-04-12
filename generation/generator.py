"""
Layer 4 — Answer Synthesis with grounded source citations.

Pipeline:
  RetrievalResult (from Layer 3)
       │
       ▼
  build_context()
    · Deduplication by parent_id
    · Lost-in-the-middle valley reordering
    · Token budget enforcement (max_context_tokens)
       │
       ▼
  OpenAI chat completion  (non-streaming or streaming)
    · Financial analyst system prompt
    · Numbered context blocks [1]..[N]
    · Mandatory citation format contract
       │
       ▼
  Citation extraction
    · Regex scan for [N] patterns in answer text
    · Map each citation number → SearchResult metadata
    · Warn on out-of-range citation numbers (hallucinated citations)
       │
       ▼
  Grounding check
    · Phrase-matching heuristic for "not found / insufficient context" signals
    · Sets GenerationResult.grounded = False → CRAG / API can act on this
       │
       ▼
  GenerationResult
    · answer + citations + token usage + latency + cost estimate

Retry strategy:
  Uses tenacity with exponential backoff on transient OpenAI errors
  (RateLimitError, APITimeoutError).  Non-retriable 4xx errors propagate
  immediately.

Streaming variant:
  generate_streaming() yields raw text tokens as they arrive.
  No structured GenerationResult is produced during streaming — use
  generate() when citations and token counts are needed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from loguru import logger
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings as _settings
from generation.context_builder import build_context
from generation.models import Citation, GenerationResult
from generation.prompts import GENERATION_SYSTEM, GENERATION_USER, UNGROUNDED_PHRASES
from retrieval.models import RetrievalResult, SearchResult

_cfg = _settings.generation

# ── OpenAI client (lazy singleton) ────────────────────────────────────────────

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = _settings.infra.openai_api_key
        if not api_key:
            raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
        # max_retries=0 — tenacity handles our retry logic instead of the SDK default
        _openai_client = OpenAI(api_key=api_key, max_retries=0)
        logger.info(f"OpenAI generation client initialised | model={_cfg.model}")
    return _openai_client


# ── Citation extraction ────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _extract_citations(
    answer: str,
    citation_results: list[SearchResult],
) -> list[Citation]:
    """
    Scan the answer text for [N] citation patterns and map each unique
    citation number to its corresponding SearchResult.

    Args:
        answer          : raw answer text from the LLM
        citation_results: SearchResults in citation order — index 0 = block [1]

    Returns:
        List of Citation objects sorted by citation index, deduplicated.
        Out-of-range citation numbers are logged and skipped (never crash).
    """
    raw_indices = {int(m) for m in _CITATION_RE.findall(answer)}

    citations: list[Citation] = []
    for idx in sorted(raw_indices):
        result_index = idx - 1  # [1] → index 0, [2] → index 1, …
        if 0 <= result_index < len(citation_results):
            r = citation_results[result_index]
            excerpt = (r.parent_text or r.text)[:250].strip()
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=r.chunk_id,
                    parent_id=r.parent_id,
                    ticker=r.ticker,
                    company=r.company,
                    date=r.date,
                    fiscal_period=r.fiscal_period,
                    section_title=r.section_title,
                    doc_type=r.doc_type,
                    source=r.source,
                    rerank_score=r.rerank_score,
                    excerpt=excerpt,
                )
            )
        else:
            # The model cited a number that doesn't exist in the context —
            # a hallucinated citation.  Log a warning; do not crash.
            logger.warning(
                f"Hallucinated citation [{idx}] — "
                f"only {len(citation_results)} context chunks were provided."
            )

    return citations


# ── Grounding check ────────────────────────────────────────────────────────────


def _is_grounded(answer: str) -> bool:
    """
    Heuristic check: returns False if the answer signals insufficient context.
    Consumed by downstream routing — CRAG can trigger a web fallback on False.
    """
    lower = answer.lower()
    return not any(phrase in lower for phrase in UNGROUNDED_PHRASES)


# ── Core LLM call with tenacity retry ─────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(
        multiplier=_cfg.retry_base_delay_seconds,
        min=_cfg.retry_base_delay_seconds,
        max=30.0,
    ),
    stop=stop_after_attempt(_cfg.max_retries),
    reraise=True,
)
def _call_llm(prompt_messages: list[dict]) -> tuple[str, int, int]:
    """
    Single OpenAI chat completion call with tenacity retry on transient errors.

    Retries on : RateLimitError, APITimeoutError  (transient, back-off helps)
    Propagates : APIError 4xx, AuthenticationError  (unrecoverable — retry wastes money)

    Returns:
        (answer_text, prompt_tokens, completion_tokens)
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=_cfg.model,
            messages=prompt_messages,
            # temperature=_cfg.temperature,
            max_completion_tokens=_cfg.max_tokens,
        )
    except APIError as exc:
        # Retry on 5xx server errors; propagate on 4xx client errors immediately
        if exc.status_code is not None and exc.status_code < 500:
            raise
        raise  # tenacity will decide whether to retry based on exception type

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    if not answer:
        raise ValueError("Generation model returned an empty response.")

    return answer, prompt_tokens, completion_tokens


# ── Fallback answer ────────────────────────────────────────────────────────────

_NO_CONTEXT_ANSWER = (
    "No relevant documents were found in the knowledge base for this question. "
    "Please verify that the relevant earnings filings have been ingested "
    "(run `poetry run python -m ingestion.pipeline`), "
    "or try rephrasing your question."
)


# ── Public Generator class ─────────────────────────────────────────────────────


class Generator:
    """
    Layer 4: LLM answer synthesis with grounded source citations.

    Thread-safety: the Generator instance is stateless — it holds no mutable
    state.  The OpenAI client is a process-level singleton that is safe for
    concurrent use after first initialization.

    Usage (recommended — via module-level shortcut):
        from generation import generate
        result = generate(question="...", retrieval_result=result)

    Usage (direct):
        generator = Generator()
        result = generator.generate(question, retrieval_result)
        print(result.format_answer_with_citations())

    Streaming:
        for token in generator.generate_streaming(question, retrieval_result):
            print(token, end="", flush=True)
    """

    def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """
        Synthesise an answer from retrieved context with inline source citations.

        Args:
            question         : original user question (already stripped by caller)
            retrieval_result : output from Layer 3 (search + rerank)

        Returns:
            GenerationResult with answer, citations, token usage, and diagnostics.

        Raises:
            OSError            : OPENAI_API_KEY not set
            RateLimitError     : rate limit exceeded after all retries
            APITimeoutError    : API timeout after all retries
        """
        start = time.perf_counter()

        # ── Empty retrieval fast-path ──────────────────────────────────────────
        if retrieval_result.is_empty:
            logger.warning(
                "Generation called with empty RetrievalResult — returning no-context answer."
            )
            return GenerationResult(
                question=question,
                answer=_NO_CONTEXT_ANSWER,
                citations=[],
                model=_cfg.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                context_chunks_used=0,
                context_tokens_used=0,
                latency_seconds=time.perf_counter() - start,
                grounded=False,
                retrieval_failed=True,
            )

        # ── Build context window ───────────────────────────────────────────────
        context_text, citation_results, context_tokens = build_context(
            results=retrieval_result.results,
            max_context_tokens=_cfg.max_context_tokens,
        )
        logger.info(
            f"Context built | chunks={len(citation_results)} | tokens={context_tokens} | "
            f"query={question!r:.60}"
        )

        # ── Assemble prompt ────────────────────────────────────────────────────
        user_content = GENERATION_USER.format(
            context=context_text,
            question=question,
        )
        # Merge system and user into a single user message
        prompt_messages: list[dict] = [
            {"role": "user", "content": f"{GENERATION_SYSTEM}\n\n{user_content}"},
        ]

        # ── LLM call (tenacity-retried) ────────────────────────────────────────
        answer, prompt_tokens, completion_tokens = _call_llm(prompt_messages)

        # ── Post-process ───────────────────────────────────────────────────────
        citations = _extract_citations(answer, citation_results)
        grounded = _is_grounded(answer)
        latency = time.perf_counter() - start

        logger.info(
            f"Generation complete | "
            f"citations={len(citations)} | grounded={grounded} | "
            f"tokens={prompt_tokens}+{completion_tokens} | {latency:.2f}s"
        )

        return GenerationResult(
            question=question,
            answer=answer,
            citations=citations,
            model=_cfg.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            context_chunks_used=len(citation_results),
            context_tokens_used=context_tokens,
            latency_seconds=latency,
            grounded=grounded,
            retrieval_failed=False,
        )

    def generate_streaming(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> Iterator[str]:
        """
        Streaming variant: yield raw text tokens as they arrive from the LLM.

        Useful for UI layers (Streamlit, Gradio, FastAPI SSE) to render
        progressive output without waiting for the full response.

        Note: No GenerationResult is produced during streaming — citations
        and token counts are unavailable.  Use generate() for structured output.

        Args:
            question         : original user question
            retrieval_result : output from Layer 3

        Yields:
            str: raw text delta tokens (may be empty strings between chunks)

        Usage:
            for token in generator.generate_streaming(question, result):
                print(token, end="", flush=True)
            print()  # final newline
        """
        if retrieval_result.is_empty:
            yield _NO_CONTEXT_ANSWER
            return

        context_text, _citation_results, context_tokens = build_context(
            results=retrieval_result.results,
            max_context_tokens=_cfg.max_context_tokens,
        )
        logger.info(
            f"Streaming context | chunks={len(_citation_results)} | tokens={context_tokens}"
        )

        user_content = GENERATION_USER.format(
            context=context_text,
            question=question,
        )
        # Merge system and user into a single user message
        prompt_messages: list[dict] = [
            {"role": "user", "content": f"{GENERATION_SYSTEM}\n\n{user_content}"},
        ]

        client = _get_client()
        with client.chat.completions.create(
            model=_cfg.model,
            messages=prompt_messages,
            # temperature=_cfg.temperature,
            max_completion_tokens=_cfg.max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
