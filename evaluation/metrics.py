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

import collections
import contextlib
import json
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from config import settings as _settings
from config.openai_client import get_openai_client
from evaluation.models import MetricScore

_eval_cfg = _settings.evaluation
_JSON_RE = re.compile(r"\{[^}]*\}", re.DOTALL)


def _call(prompt: str) -> str:
    client = get_openai_client()
    call_kwargs: dict[str, Any] = {
        "model": _eval_cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max(_eval_cfg.max_tokens, 4096),
        "response_format": {"type": "json_object"},
    }
    if _eval_cfg.temperature != 1.0 and not _eval_cfg.model.startswith(("gpt-5", "o1", "o3")):
        call_kwargs["temperature"] = _eval_cfg.temperature

    try:
        resp = client.chat.completions.create(**call_kwargs)
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            # Fallback for reasoning models if completion token budget was consumed by reasoning
            call_kwargs["max_completion_tokens"] = 8192
            resp = client.chat.completions.create(**call_kwargs)
            content = (resp.choices[0].message.content or "").strip()
        return content
    except Exception as exc:
        if "temperature" in str(exc).lower() and "temperature" in call_kwargs:
            call_kwargs.pop("temperature")
            resp = client.chat.completions.create(**call_kwargs)
            return (resp.choices[0].message.content or "").strip()
        else:
            raise


def _parse_score(raw: str, metric: str) -> tuple[float, str]:
    """
    Parse {"score": 0.8, "reasoning": "..."} from LLM response.
    Returns (0.5, "parse error") as fallback — never raises.
    """
    text = raw.strip()

    # 1. Direct JSON parse
    with contextlib.suppress(Exception):
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            reasoning = str(data.get("reasoning", ""))[:250]
            return score, reasoning

    # 2. Extract substring between first '{' and last '}'
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx : end_idx + 1]
        with contextlib.suppress(Exception):
            data = json.loads(json_str)
            if isinstance(data, dict) and "score" in data:
                score = max(0.0, min(1.0, float(data.get("score", 0.5))))
                reasoning = str(data.get("reasoning", ""))[:250]
                return score, reasoning

    # 3. Regex fallback for "score": X.X
    match = re.search(r'"score"\s*:\s*([0-1](?:\.\d+)?)', text)
    if match:
        with contextlib.suppress(Exception):
            score = float(match.group(1))
            return max(0.0, min(1.0, score)), "regex score fallback"

    logger.warning(f"[{metric}] no valid JSON in response: {text[:100]!r}")
    return 0.5, "parse error"


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
    context = "\n\n".join(f"[{i + 1}] {c[:3000].strip()}" for i, c in enumerate(context_chunks))
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
    chunks_numbered = "\n\n".join(
        f"[{i + 1}] {c[:2500].strip()}" for i, c in enumerate(context_chunks)
    )
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
    context = "\n\n".join(f"[{i + 1}] {c[:3000].strip()}" for i, c in enumerate(context_chunks))
    prompt = _RECALL_PROMPT.format(question=question, ground_truth=ground_truth, context=context)
    try:
        raw = _call(prompt)
        score, reasoning = _parse_score(raw, "context_recall")
    except Exception as exc:
        logger.warning(f"context_recall metric error: {exc}")
        score, reasoning = 0.5, f"metric error: {type(exc).__name__}"
    return MetricScore(metric="context_recall", score=score, reasoning=reasoning)


# ── Statistical & Text Similarity NLP Metrics ───────────────────────────────


def _normalize_text(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens for NLP metrics."""
    return re.findall(r"\w+", text.lower())


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def score_token_f1(answer: str, ground_truth: str) -> MetricScore:
    """SQuAD token-level Precision/Recall F1 score."""
    pred_tokens = _normalize_text(answer)
    gt_tokens = _normalize_text(ground_truth)
    if not pred_tokens or not gt_tokens:
        return MetricScore(metric="token_f1", score=0.0, reasoning="empty tokens")
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return MetricScore(metric="token_f1", score=0.0, reasoning="no token overlap")
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return MetricScore(
        metric="token_f1",
        score=round(f1, 4),
        reasoning=f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}",
    )


def _ngram_f1(pred_tokens: list[str], gt_tokens: list[str], n: int) -> float:
    if len(pred_tokens) < n or len(gt_tokens) < n:
        return 0.0
    pred_ngrams = collections.Counter(_get_ngrams(pred_tokens, n))
    gt_ngrams = collections.Counter(_get_ngrams(gt_tokens, n))
    overlap = sum((pred_ngrams & gt_ngrams).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(pred_ngrams.values())
    rec = overlap / sum(gt_ngrams.values())
    return (2 * prec * rec) / (prec + rec)


def _lcs_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    m, n = len(pred_tokens), len(gt_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gt_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / m
    rec = lcs_len / n
    return (2 * prec * rec) / (prec + rec)


def score_rouge_1(answer: str, ground_truth: str) -> MetricScore:
    pred_tokens = _normalize_text(answer)
    gt_tokens = _normalize_text(ground_truth)
    f1 = _ngram_f1(pred_tokens, gt_tokens, 1)
    return MetricScore(metric="rouge1_f1", score=round(f1, 4), reasoning=f"ROUGE-1 F1={f1:.4f}")


def score_rouge_2(answer: str, ground_truth: str) -> MetricScore:
    pred_tokens = _normalize_text(answer)
    gt_tokens = _normalize_text(ground_truth)
    f1 = _ngram_f1(pred_tokens, gt_tokens, 2)
    return MetricScore(metric="rouge2_f1", score=round(f1, 4), reasoning=f"ROUGE-2 F1={f1:.4f}")


def score_rouge_l(answer: str, ground_truth: str) -> MetricScore:
    pred_tokens = _normalize_text(answer)
    gt_tokens = _normalize_text(ground_truth)
    f1 = _lcs_f1(pred_tokens, gt_tokens)
    return MetricScore(metric="rougeL_f1", score=round(f1, 4), reasoning=f"ROUGE-L F1={f1:.4f}")


def score_bleu(answer: str, ground_truth: str) -> MetricScore:
    pred_tokens = _normalize_text(answer)
    gt_tokens = _normalize_text(ground_truth)
    if not pred_tokens or not gt_tokens:
        return MetricScore(metric="bleu_4", score=0.0, reasoning="empty tokens")
    c, r = len(pred_tokens), len(gt_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    precisions = []
    for n in range(1, 5):
        if len(pred_tokens) < n or len(gt_tokens) < n:
            precisions.append(0.0)
            continue
        p_ngrams = collections.Counter(_get_ngrams(pred_tokens, n))
        g_ngrams = collections.Counter(_get_ngrams(gt_tokens, n))
        overlap = sum((p_ngrams & g_ngrams).values())
        total = sum(p_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0.0)
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        s = sum(math.log(p) for p in precisions) / 4.0
        bleu = bp * math.exp(s)
    return MetricScore(
        metric="bleu_4", score=round(bleu, 4), reasoning=f"BLEU-4={bleu:.4f} (BP={bp:.2f})"
    )


def score_semantic_similarity(answer: str, ground_truth: str) -> MetricScore:
    """Cosine similarity of embeddings using text-embedding-3-small."""
    client = get_openai_client()
    try:
        res = client.embeddings.create(
            input=[answer[:1000], ground_truth[:1000]],
            model=_settings.embedding.model,
        )
        vec_ans = res.data[0].embedding
        vec_gt = res.data[1].embedding
        dot = sum(a * b for a, b in zip(vec_ans, vec_gt, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_ans))
        norm_b = math.sqrt(sum(b * b for b in vec_gt))
        sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        score = max(0.0, min(1.0, float(sim)))
        return MetricScore(
            metric="semantic_similarity", score=round(score, 4), reasoning=f"Cosine sim={score:.4f}"
        )
    except Exception as exc:
        logger.warning(f"semantic_similarity metric error: {exc}")
        return MetricScore(
            metric="semantic_similarity", score=0.5, reasoning=f"metric error: {exc}"
        )


def score_all(
    question: str,
    answer: str,
    context_chunks: list[str],
    ground_truth: str,
    metrics: list[str] | None = None,
) -> list[MetricScore]:
    """
    Compute LLM-as-a-judge and NLP statistical metrics for a single QA sample.

    Supported metrics:
      - faithfulness, answer_relevancy, context_precision, context_recall (LLM-as-a-Judge)
      - token_f1, rouge1_f1, rouge2_f1, rougeL_f1, bleu_4 (Statistical NLP)
      - semantic_similarity (Embedding Vector Cosine Sim)
    """
    _all: dict[str, Callable[[], MetricScore]] = {
        "faithfulness": lambda: score_faithfulness(question, answer, context_chunks),
        "answer_relevancy": lambda: score_answer_relevancy(question, answer),
        "context_precision": lambda: score_context_precision(question, context_chunks),
        "context_recall": lambda: score_context_recall(question, context_chunks, ground_truth),
        "token_f1": lambda: score_token_f1(answer, ground_truth),
        "rouge1_f1": lambda: score_rouge_1(answer, ground_truth),
        "rouge2_f1": lambda: score_rouge_2(answer, ground_truth),
        "rougeL_f1": lambda: score_rouge_l(answer, ground_truth),
        "bleu_4": lambda: score_bleu(answer, ground_truth),
        "semantic_similarity": lambda: score_semantic_similarity(answer, ground_truth),
    }
    selected = metrics or list(_all.keys())
    exec_map = {m: _all[m] for m in selected if m in _all}

    results: dict[str, MetricScore] = {}
    with ThreadPoolExecutor(
        max_workers=min(len(exec_map), 8), thread_name_prefix="eval_metric"
    ) as executor:
        futures = {executor.submit(fn): m for m, fn in exec_map.items()}
        for future in as_completed(futures):
            m = futures[future]
            try:
                results[m] = future.result()
            except Exception as exc:
                logger.warning(f"Metric calculation failed for {m}: {exc}")
                results[m] = MetricScore(metric_name=m, score=0.0, reasoning=f"Error: {exc}")

    # Return in original requested order
    return [results[m] for m in selected if m in results]


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
