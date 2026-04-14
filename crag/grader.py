# crag/grader.py
"""
Layer 5a — LLM-based chunk relevance grading for CRAG.

The grader is a lightweight binary classifier: given a question and a retrieved
chunk, it returns a RelevanceGrade indicating whether the chunk contains
information directly relevant to answering the question.

Design decisions:
  - Uses gpt-4.1-nano (same tier as query transform — cheap, fast)
  - Structured JSON output for reliable regex-based parsing
  - Concurrent grading via ThreadPoolExecutor (one LLM call per chunk)
  - Fail-open: on any error, defaults to relevant=True so the pipeline
    continues rather than silently discarding potentially useful context

Grading criteria:
  true  — chunk contains the specific metric, time period, company, or
          statement being asked about
  false — wrong company, wrong period, different metric, or boilerplate text
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from openai import OpenAI

from config import settings as _settings
from crag.models import RelevanceGrade
from retrieval.models import SearchResult

# ── Prompts ─────────────────────────────────────────────────────────────────────

_GRADER_PROMPT = """\
You are a financial document relevance grader for a RAG retrieval system.

Given a financial question and a document chunk from an SEC 8-K earnings filing,
assess whether the document chunk contains information that DIRECTLY helps answer
the question.

Rules:
- Mark relevant=true ONLY if the chunk contains the specific metric, time period,
  company, or management statement being asked about
- Background context, boilerplate ("safe harbor"), or tangential figures = false
- Wrong company or wrong fiscal period = always false
- Be strict — partial matches should lean toward false

Respond with ONLY this JSON (no preamble, no explanation outside the JSON):
{{
  "relevant": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "one concise sentence"
}}

Question: {question}

Document chunk:
{chunk_text}

JSON:"""

# ── JSON parsing ──────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^}]+\}", re.DOTALL)


def _parse_response(raw: str, chunk_id: str) -> tuple[bool, float, str]:
    """
    Parse grader JSON response. Returns (relevant, score, reasoning).
    Falls back to (True, 0.5, reason) on any failure — fail-open by design.
    """
    text = raw.strip()

    def _fallback(why: str) -> tuple[bool, float, str]:
        logger.debug(f"Grader parse fallback ({why}) for chunk={chunk_id!r}")
        return True, 0.5, f"parse fallback: {why}"

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return _fallback("no JSON found")
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return _fallback("JSON extract failed")

    relevant = bool(data.get("relevant", True))
    raw_score = data.get("score", 0.5)
    score = max(0.0, min(1.0, float(raw_score)))
    reasoning = str(data.get("reasoning", ""))[:250]
    return relevant, score, reasoning


# ── OpenAI client singleton ────────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = _settings.infra.openai_api_key
        if not key:
            raise OSError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=key, max_retries=2)
    return _client


# ── Core single-chunk grading ──────────────────────────────────────────────────


def _grade_one(question: str, chunk: SearchResult) -> RelevanceGrade:
    """
    Grade one chunk. Never raises — fails open to relevant=True on any error.
    Caps chunk text at 800 chars to avoid wasting tokens on boilerplate.
    """
    chunk_text = ((chunk.parent_text or chunk.text) or "")[:800].strip()

    try:
        resp = _get_client().chat.completions.create(
            model=_settings.query_transform.model,  # reuse nano tier
            messages=[
                {
                    "role": "user",
                    "content": _GRADER_PROMPT.format(
                        question=question,
                        chunk_text=chunk_text,
                    ),
                }
            ],
            temperature=0.0,
            max_completion_tokens=128,
        )
        raw = (resp.choices[0].message.content or "").strip()
        relevant, score, reasoning = _parse_response(raw, chunk.chunk_id)
    except Exception as exc:
        logger.warning(f"Grader error for chunk={chunk.chunk_id!r}: {exc} — defaulting relevant")
        relevant, score, reasoning = True, 0.5, f"grader error: {type(exc).__name__}"

    return RelevanceGrade(
        chunk_id=chunk.chunk_id,
        relevant=relevant,
        score=score,
        reasoning=reasoning,
    )


# ── Public grader class ────────────────────────────────────────────────────────


class RelevanceGrader:
    """
    Concurrent LLM-based relevance grader for CRAG.

    Grades all chunks in parallel so total latency ≈ one LLM call
    (~200–400ms for gpt-4.1-nano) regardless of chunk count.

    Usage:
        grader = RelevanceGrader()
        grades = grader.grade_chunks("What was AAPL revenue?", retrieval.results)
        relevant = [c for c, g in zip(chunks, grades) if g.relevant]
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._workers = max_workers or _settings.crag.grader_max_workers

    def grade_single(self, question: str, chunk: SearchResult) -> RelevanceGrade:
        """Grade a single chunk. Useful in tests or sequential evaluation."""
        return _grade_one(question, chunk)

    def grade_chunks(
        self,
        question: str,
        chunks: list[SearchResult],
    ) -> list[RelevanceGrade]:
        """
        Grade all chunks concurrently. Returns grades in the same order as input.

        Thread-safe: each future is independent.
        Never raises: each grade independently fails open.

        Args:
            question: original user question
            chunks:   SearchResult list from Layer 3

        Returns:
            list[RelevanceGrade] in same order as chunks
        """
        if not chunks:
            return []

        n = len(chunks)
        grades: list[RelevanceGrade | None] = [None] * n

        with ThreadPoolExecutor(max_workers=min(self._workers, n)) as pool:
            fut_to_idx = {
                pool.submit(_grade_one, question, chunk): i for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    grades[idx] = fut.result()
                except Exception as exc:
                    logger.error(f"Unexpected grader future error idx={idx}: {exc}")
                    grades[idx] = RelevanceGrade(
                        chunk_id=chunks[idx].chunk_id,
                        relevant=True,
                        score=0.5,
                        reasoning="unexpected future error",
                    )

        result = [g for g in grades if g is not None]
        n_relevant = sum(1 for g in result if g.relevant)
        logger.info(
            f"RelevanceGrader: {n_relevant}/{len(result)} chunks relevant for q={question!r:.60}"
        )
        return result
