# evaluation/metrics.py
"""
LLM-based evaluation metrics for the Financial RAG system.

Implements four core RAG quality metrics directly via OpenAI API calls,
without depending on the Ragas library (which has frequent API changes).

Metric definitions:
  faithfulness      — Are all answer claims supported by the retrieved context?
                      Score = supported_claims / total_claims.  Range: 0–1.
  answer_relevancy  — Does the answer address the question being asked?
                      Score = 0 (off-topic) to 1 (fully addresses question).
  context_precision — What fraction of retrieved chunks are relevant to the query?
                      Score = relevant_retrieved / total_retrieved.
  context_recall    — Does the retrieved context cover the ground truth answer?
                      Score = covered_ground_truth_statements / total_statements.

All four prompts instruct the evaluator LLM to respond with structured JSON
for reliable programmatic parsing.  Falls back to score=0.5 on parse errors.

Usage:
    from evaluation.metrics import score_faithfulness, score_all

    ms = score_all(
        question="What was Apple's Q4 revenue?",
        answer="Apple reported $94.9B [1].",
        context_chunks=["Apple Q4 2024... revenue $94.9B..."],
        ground_truth="Apple reported $94.9 billion in Q4 FY2024.",
    )
    for m in ms:
        print(f"{m.metric}: {m.score:.3f}  — {m.reasoning}")
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from loguru import logger

from config import settings as _settings
from config.openai_client import get_openai_client
from evaluation.models import MetricScore

_eval_cfg = _settings.evaluation
_JSON_RE = re.compile(r"\{[^}]*\}", re.DOTALL)


def _call(prompt: str) -> str:
    resp = get_openai_client().chat.completions.create(
        model=_eval_cfg.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=256,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_score(raw: str, metric: str) -> tuple[float, str]:
    """
    Parse {"score": 0.8, "reasoning": "..."} from LLM response.
    Returns (0.5, "parse error") as fallback — never raises.
    """
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            logger.warning(f"[{metric}] no JSON in response: {text[:80]}")
            return 0.5, "parse error"
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return 0.5, "JSON extract failed"

    score = max(0.0, min(1.0, float(data.get("score", 0.5))))
    reasoning = str(data.get("reasoning", ""))[:250]
    return score, reasoning


# ── Metric prompts ─────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating whether a RAG-generated answer is faithful to its source context.

Question: {question}

Retrieved context:
{context}

Generated answer:
{answer}

Task: Identify every factual claim in the answer. For each claim, determine if it
is directly supported by the retrieved context (not prior knowledge).

Score 1.0 if every claim is supported. Score 0.0 if no claims are supported.
Score proportionally for partial support.

Respond with ONLY this JSON:
{{"score": 0.0 to 1.0, "reasoning": "one sentence explaining the score"}}

JSON:"""

_RELEVANCY_PROMPT = """\
You are evaluating whether a generated answer is relevant to the original question.

Question: {question}

Generated answer:
{answer}

Task: Does the answer directly address the question? Score:
  1.0 — fully and precisely answers what was asked
  0.7 — mostly answers but misses some aspect
  0.4 — partially relevant (tangential or incomplete)
  0.0 — completely off-topic or non-responsive

Respond with ONLY this JSON:
{{"score": 0.0 to 1.0, "reasoning": "one sentence explaining the score"}}

JSON:"""

_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved document chunks for answering a question.

Question: {question}

Retrieved chunks:
{chunks_numbered}

Task: For each chunk, determine if it contains information that DIRECTLY helps
answer the question. Score = (relevant chunks) / (total chunks).

Respond with ONLY this JSON:
{{"score": 0.0 to 1.0, "reasoning": "one sentence summarising which chunks were relevant"}}

JSON:"""

_RECALL_PROMPT = """\
You are evaluating whether retrieved context covers the information in a ground-truth answer.

Question: {question}

Ground truth answer:
{ground_truth}

Retrieved context:
{context}

Task: Identify the key factual statements in the ground truth. For each, check if the
retrieved context contains the same information. Score = covered_statements / total_statements.

Respond with ONLY this JSON:
{{"score": 0.0 to 1.0, "reasoning": "one sentence explaining what was or was not covered"}}

JSON:"""


# ── Public metric functions ────────────────────────────────────────────────────


def score_faithfulness(
    question: str,
    answer: str,
    context_chunks: list[str],
) -> MetricScore:
    """Are all claims in the answer supported by the retrieved context?"""
    context = "\n\n".join(f"[{i + 1}] {c[:600]}" for i, c in enumerate(context_chunks))
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    try:
        raw = _call(prompt)
        score, reasoning = _parse_score(raw, "faithfulness")
    except Exception as exc:
        logger.warning(f"faithfulness metric error: {exc}")
        score, reasoning = 0.5, f"metric error: {type(exc).__name__}"
    return MetricScore(metric="faithfulness", score=score, reasoning=reasoning)


def score_answer_relevancy(question: str, answer: str) -> MetricScore:
    """Does the answer directly address the question?"""
    prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
    try:
        raw = _call(prompt)
        score, reasoning = _parse_score(raw, "answer_relevancy")
    except Exception as exc:
        logger.warning(f"answer_relevancy metric error: {exc}")
        score, reasoning = 0.5, f"metric error: {type(exc).__name__}"
    return MetricScore(metric="answer_relevancy", score=score, reasoning=reasoning)


def score_context_precision(question: str, context_chunks: list[str]) -> MetricScore:
    """What fraction of retrieved chunks are relevant to the question?"""
    if not context_chunks:
        return MetricScore(metric="context_precision", score=0.0, reasoning="no context chunks")
    chunks_numbered = "\n\n".join(f"[{i + 1}] {c[:400]}" for i, c in enumerate(context_chunks))
    prompt = _PRECISION_PROMPT.format(question=question, chunks_numbered=chunks_numbered)
    try:
        raw = _call(prompt)
        score, reasoning = _parse_score(raw, "context_precision")
    except Exception as exc:
        logger.warning(f"context_precision metric error: {exc}")
        score, reasoning = 0.5, f"metric error: {type(exc).__name__}"
    return MetricScore(metric="context_precision", score=score, reasoning=reasoning)


def score_context_recall(
    question: str,
    context_chunks: list[str],
    ground_truth: str,
) -> MetricScore:
    """Does the retrieved context cover the key facts in the ground truth?"""
    if not context_chunks:
        return MetricScore(metric="context_recall", score=0.0, reasoning="no context chunks")
    context = "\n\n".join(f"[{i + 1}] {c[:600]}" for i, c in enumerate(context_chunks))
    prompt = _RECALL_PROMPT.format(question=question, ground_truth=ground_truth, context=context)
    try:
        raw = _call(prompt)
        score, reasoning = _parse_score(raw, "context_recall")
    except Exception as exc:
        logger.warning(f"context_recall metric error: {exc}")
        score, reasoning = 0.5, f"metric error: {type(exc).__name__}"
    return MetricScore(metric="context_recall", score=score, reasoning=reasoning)


def score_all(
    question: str,
    answer: str,
    context_chunks: list[str],
    ground_truth: str,
    metrics: list[str] | None = None,
) -> list[MetricScore]:
    """
    Compute all four metrics for a single (question, answer, context, truth) tuple.

    Args:
        metrics: subset of ["faithfulness","answer_relevancy",
                             "context_precision","context_recall"]
                 If None, all four are computed.

    Returns:
        list[MetricScore] in the requested order.
    """
    _all: dict[str, Callable[[], MetricScore]] = {
        "faithfulness": lambda: score_faithfulness(question, answer, context_chunks),
        "answer_relevancy": lambda: score_answer_relevancy(question, answer),
        "context_precision": lambda: score_context_precision(question, context_chunks),
        "context_recall": lambda: score_context_recall(question, context_chunks, ground_truth),
    }
    selected = metrics or list(_all.keys())
    return [_all[m]() for m in selected if m in _all]


def compute_all_metrics(
    question: str,
    answer: str,
    context_chunks: list[str],
    ground_truth: str,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute a dictionary of metric name → score for a single QA sample.

    Convenience wrapper used by the retrieval experiment framework to
    evaluate a single pipeline call without working with MetricScore objects.

    Delegates to ``score_all`` to avoid duplicating the metric dispatch table.

    Args:
        question       : The original user question
        answer         : The pipeline-generated answer
        context_chunks : List of retrieved chunk texts used for the answer
        ground_truth   : Expected factual answer (from golden dataset)
        metrics        : Subset of metrics to compute; defaults to all four

    Returns:
        dict mapping metric name → float score in [0, 1]
    """
    scored = score_all(question, answer, context_chunks, ground_truth, metrics)
    return {ms.metric: ms.score for ms in scored}
