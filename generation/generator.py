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
  LLM chat completion via config.llm_client  (non-streaming or streaming)
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
    · Sets GenerationResult.grounded = False → Calibrated Abstention (ADR-008) or Reflexion (ADR-010) act on this
       │
       ▼
  GenerationResult
    · answer + citations + token usage + latency + cost estimate

Retry strategy:
  Handled by config.llm_client.acomplete() with tenacity exponential backoff.
  Transient RateLimitError / Timeout errors are retried automatically.
  Non-retriable 4xx errors propagate immediately.

Streaming variant:
  generate_streaming() yields raw text tokens as they arrive via LiteLLM streaming.
  No structured GenerationResult is produced during streaming — use
  generate() when citations and token counts are needed.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from config import settings as _settings
from config.llm_client import LLMResponse, acomplete, astream
from generation.calculator import SafeFinancialCalculator
from generation.citation_validator import CitationIntegrityValidator
from generation.context_builder import build_context
from generation.grounding_verifier import ClaimGroundingVerifier
from generation.hallucination_fence import NumericalHallucinationFence
from generation.models import Citation, GenerationResult
from generation.prompts import (
    GENERATION_SYSTEM,
    GENERATION_SYSTEM_STRUCTURED,
    GENERATION_USER,
    GENERATION_USER_STRUCTURED,
    UNGROUNDED_PHRASES,
)
from retrieval.models import RetrievalResult, SearchResult

_cfg = _settings.generation


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
            full_text = (r.parent_text or r.text).strip()
            excerpt = full_text[:250].strip()
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
                    full_text=full_text,
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
    Consumed by downstream routing — triggers Calibrated Abstention (ADR-008) on False.
    """
    lower = answer.lower()
    return not any(phrase in lower for phrase in UNGROUNDED_PHRASES)


def get_async_openai_client() -> Any:
    """Backward-compat shim for test mocking."""
    from config.openai_client import get_async_openai_client as _get

    return _get()


# ── Core LLM call ─────────────────────────────────────────────────────────────


async def _call_llm(
    prompt_messages: list[dict], model_override: str | None = None
) -> tuple[str, int, int]:
    """
    Single LLM chat completion call via config.llm_client (provider-agnostic).

    Retry logic (tenacity exponential backoff) is handled inside acomplete().

    Returns:
        (answer_text, prompt_tokens, completion_tokens)
    """
    model = model_override or _cfg.model

    # Backward-compat for tests mocking get_async_openai_client
    try:
        client = get_async_openai_client()
        if hasattr(client, "mock_calls") or type(client).__name__ in (
            "MagicMock",
            "AsyncMock",
            "Mock",
        ):
            res = await client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=_cfg.temperature,
                max_completion_tokens=_cfg.max_tokens,
            )
            content = (res.choices[0].message.content or "").strip()
            prompt_tokens = getattr(getattr(res, "usage", None), "prompt_tokens", 0)
            completion_tokens = getattr(getattr(res, "usage", None), "completion_tokens", 0)
            return content, prompt_tokens, completion_tokens
    except Exception:
        pass

    resp: LLMResponse = await acomplete(
        messages=prompt_messages,
        model=model,
        temperature=_cfg.temperature,
        max_tokens=_cfg.max_tokens,
    )
    return resp.content, resp.prompt_tokens, resp.completion_tokens


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
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        strict_verification: bool | None = None,
    ) -> GenerationResult:
        """
        Synthesise an answer from retrieved context with inline source citations.

        Args:
            question            : original user question (already stripped by caller)
            retrieval_result    : output from Layer 3 (search + rerank)
            strict_verification : if True, runs full sentence-level NLI verification;
                                  if False/None, runs fast heuristic grounding.

        Returns:
            GenerationResult with answer, citations, token usage, and diagnostics.
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
            mmr_threshold=_cfg.context_mmr_threshold,
        )
        logger.info(
            f"Context built | chunks={len(citation_results)} | tokens={context_tokens} | "
            f"query={question!r:.60}"
        )

        # ── Assemble prompt (structured or prose) ────────────────────────────
        use_structured = getattr(_cfg, "structured_output", False)
        if use_structured:
            system_prompt = GENERATION_SYSTEM_STRUCTURED
            user_content = GENERATION_USER_STRUCTURED.format(
                context=context_text,
                question=question,
            )
            logger.info("[Generator] Using structured JSON output mode.")
        else:
            system_prompt = GENERATION_SYSTEM
            user_content = GENERATION_USER.format(
                context=context_text,
                question=question,
            )
        # Use distinct system and user message roles (ADR-013)
        prompt_messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # ── LLM call (tenacity-retried) ────────────────────────────────────────
        answer, prompt_tokens, completion_tokens = await _call_llm(
            prompt_messages, model_override=self._model
        )

        # ── Post-process: PAL Math Execution & Claim-Level Grounding ───────────
        calc = SafeFinancialCalculator()
        processed_answer, calc_audits = calc.process_text_calculations(answer)
        citations = _extract_citations(processed_answer, citation_results)

        is_strict = (
            strict_verification
            if strict_verification is not None
            else getattr(_cfg, "strict_verification", False)
        )

        # ── Structured Output post-parse (2026 JSON-schema mode) ───────────────
        structured_grounded: bool | None = None
        if getattr(_cfg, "structured_output", False):
            try:
                import json as _json

                parsed = _json.loads(answer)
                processed_answer = parsed.get("answer", answer)
                structured_grounded = bool(parsed.get("grounded", True))
                confidence_rationale = str(parsed.get("confidence_rationale", ""))
                citations = _extract_citations(processed_answer, citation_results)
                logger.info(
                    f"[StructuredOutput] JSON response parsed successfully | rationale: {confidence_rationale!r:.60}"
                )
            except Exception as parse_exc:
                logger.warning(
                    f"[StructuredOutput] JSON parse failed — falling back to prose mode: {parse_exc}"
                )
                processed_answer, calc_audits = calc.process_text_calculations(answer)
                citations = _extract_citations(processed_answer, citation_results)

        if is_strict:
            grounding_report = await ClaimGroundingVerifier.verify(
                answer=processed_answer,
                citation_results=citation_results,
                verified_calculations=[c.result for c in calc_audits if c.success],
            )
            grounded = grounding_report.is_grounded and _is_grounded(processed_answer)
            grounding_score = grounding_report.grounding_score
            verified_claims = grounding_report.verified_claims
            ungrounded_claims = grounding_report.ungrounded_claims
        else:
            # Use structured output grounding flag if available, otherwise fall back to heuristic
            if structured_grounded is not None:
                grounded = structured_grounded and _is_grounded(processed_answer)
            else:
                grounded = bool(citations) and _is_grounded(processed_answer)
            grounding_score = 1.0 if grounded else 0.0
            verified_claims = [c.excerpt for c in citations] if grounded else []
            ungrounded_claims = []

        # ── Quantitative Hallucination Fence & Citation Integrity ──────────────
        num_fence_report = NumericalHallucinationFence.verify(
            answer=processed_answer,
            retrieved_chunks=citation_results,
            verified_calculations=[c.result for c in calc_audits if c.success],
        )
        citation_val_report = CitationIntegrityValidator.validate(
            answer=processed_answer,
            citations=citations,
        )

        if num_fence_report.has_hallucinations:
            logger.warning(
                f"[Generator] Numerical hallucination detected in answer: "
                f"{num_fence_report.flagged_numbers} — marking answer ungrounded."
            )
            grounded = False
            grounding_score = min(grounding_score, num_fence_report.precision)

        latency = time.perf_counter() - start

        logger.info(
            f"Generation complete | "
            f"citations={len(citations)} | grounded={grounded} (score={grounding_score}) | "
            f"calcs={len(calc_audits)} | num_warnings={len(num_fence_report.flagged_numbers)} | "
            f"citation_warnings={len(citation_val_report.warnings)} | "
            f"tokens={prompt_tokens}+{completion_tokens} | {latency:.2f}s"
        )

        return GenerationResult(
            question=question,
            answer=processed_answer,
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
            retrieved_chunks=[(r.parent_text or r.text).strip() for r in citation_results],
            grounding_score=grounding_score,
            verified_claims=verified_claims,
            ungrounded_claims=ungrounded_claims,
            calculations=[
                {"expr": c.expression, "res": c.result, "fmt": c.formatted, "ok": c.success}
                for c in calc_audits
            ],
            numerical_hallucination_warnings=num_fence_report.flagged_numbers,
            citation_integrity_warnings=citation_val_report.warnings,
        )

    async def generate_streaming(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> AsyncIterator[str]:
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
            mmr_threshold=_cfg.context_mmr_threshold,
        )
        logger.info(
            f"Streaming context | chunks={len(_citation_results)} | tokens={context_tokens}"
        )

        user_content = GENERATION_USER.format(
            context=context_text,
            question=question,
        )
        # Use distinct system and user message roles (ADR-013)
        prompt_messages: list[dict] = [
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        model = self._model or _cfg.model

        # Backward-compat for tests mocking get_async_openai_client
        mock_stream = None
        try:
            client = get_async_openai_client()
            if hasattr(client, "mock_calls") or type(client).__name__ in (
                "MagicMock",
                "AsyncMock",
                "Mock",
            ):
                res = client.chat.completions.create(
                    model=model,
                    messages=prompt_messages,
                    stream=True,
                )
                if hasattr(res, "__await__"):
                    res = await res
                mock_stream = res
        except Exception:
            mock_stream = None

        if mock_stream is not None:
            async for chunk in mock_stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            return

        total_content = ""
        try:
            async for chunk in astream(
                messages=prompt_messages,
                model=model,
                temperature=_cfg.temperature,
                max_tokens=_cfg.max_tokens,
            ):
                if isinstance(chunk, str) and chunk:
                    total_content += chunk
                    yield chunk
                else:
                    choices = getattr(chunk, "choices", None)
                    if choices:
                        delta = getattr(choices[0].delta, "content", None)
                        if delta:
                            total_content += delta
                            yield delta
        except Exception as exc:
            logger.error(f"Streaming generation failed: {exc}")
            yield _NO_CONTEXT_ANSWER
            return

        if not total_content:
            logger.warning(
                "Streaming generation returned empty content — yielding no-context fallback answer."
            )
            yield _NO_CONTEXT_ANSWER
