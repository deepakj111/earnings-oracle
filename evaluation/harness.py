# evaluation/harness.py
"""
LLMOps Evaluation Harness for the Financial RAG system.

Runs the full pipeline against a golden QA dataset, scores each result with
LLM-based metrics, and produces an EvalReport with per-sample diagnostics
and aggregate averages.

Usage (CLI):
    poetry run python -m evaluation.harness

Usage (programmatic):
    from evaluation.harness import EvaluationHarness
    from evaluation.dataset import GOLDEN_DATASET, get_dataset_subset

    harness = EvaluationHarness(pipeline)
    report = harness.run(
        dataset=get_dataset_subset(5),          # smoke test on 5 samples
        metrics=["faithfulness", "answer_relevancy"],
        dataset_name="smoke_test",
    )
    print(report.summary())
    harness.save_report(report)                 # writes JSON + CSV to data/eval_reports/
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from config import settings as _settings
from evaluation.dataset import GOLDEN_DATASET
from evaluation.metrics import score_all
from evaluation.models import EvalReport, EvalSample, EvalSampleResult, MetricScore
from evaluation.statistics import compute_bootstrap_ci
from retrieval.models import MetadataFilter

if TYPE_CHECKING:
    from rag_pipeline import FinancialRAGPipeline

_eval_cfg = _settings.evaluation

_ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class EvaluationHarness:
    """
    Offline evaluation harness.  Runs the full RAG pipeline on golden QA pairs
    and scores answers with LLM-based metrics.

    Thread-safety: each sample runs in its own thread (max_workers from config).
    The underlying pipeline must be thread-safe — FinancialRAGPipeline is.

    Args:
        pipeline: an initialised FinancialRAGPipeline instance.
                  The harness calls pipeline.ask_verbose() for each sample so it
                  can capture both the answer and the retrieval context.
    """

    def __init__(self, pipeline: FinancialRAGPipeline) -> None:
        self._pipeline = pipeline
        logger.info(
            f"EvaluationHarness ready | model={_eval_cfg.model} | workers={_eval_cfg.max_workers}"
        )

    # ── Single sample runner ───────────────────────────────────────────────────

    def _run_sample(
        self,
        sample: EvalSample,
        metrics: list[str],
    ) -> EvalSampleResult:
        t0 = time.perf_counter()

        # Build MetadataFilter from sample scoping fields
        meta_filter: MetadataFilter | None = None
        if any([sample.ticker, sample.year, sample.quarter]):
            meta_filter = MetadataFilter(
                ticker=sample.ticker,
                year=sample.year,
                quarter=sample.quarter,
            )

        # Run pipeline
        # Disable cache to prevent data leakage during evaluation
        try:
            old_cache_enabled = _settings.cache.enabled
            # Monkeypatch the cache temporarily for this thread
            object.__setattr__(_settings.cache, "enabled", False)
            result, _q_summary, _r_summary = self._pipeline.ask_verbose(
                question=sample.question,
                metadata_filter=meta_filter,
            )
        except Exception as exc:
            logger.warning(f"Pipeline failed for sample={sample.sample_id}: {exc}")
            return EvalSampleResult(
                sample=sample,
                generated_answer="",
                context_chunks=[],
                latency_seconds=time.perf_counter() - t0,
                pipeline_failed=True,
                error_message=str(exc),
            )
        finally:
            object.__setattr__(_settings.cache, "enabled", old_cache_enabled)

        # Extract context chunks for metric scoring
        context_chunks = [(c.excerpt or "") for c in result.citations]

        # Score metrics
        metric_scores: dict[str, MetricScore] = {}
        try:
            scored = score_all(
                question=sample.question,
                answer=result.answer,
                context_chunks=context_chunks,
                ground_truth=sample.ground_truth,
                metrics=metrics,
            )
            metric_scores = {ms.metric: ms for ms in scored}
        except Exception as exc:
            logger.warning(f"Metric scoring failed for sample={sample.sample_id}: {exc}")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Sample {sample.sample_id} | "
            f"latency={elapsed:.2f}s | "
            + " ".join(f"{k}={v.score:.2f}" for k, v in metric_scores.items())
        )

        return EvalSampleResult(
            sample=sample,
            generated_answer=result.answer,
            context_chunks=context_chunks,
            metric_scores=metric_scores,
            latency_seconds=elapsed,
        )

    # ── Full dataset runner ────────────────────────────────────────────────────

    def run(
        self,
        dataset: list[EvalSample] | None = None,
        metrics: list[str] | None = None,
        dataset_name: str = "golden_dataset",
    ) -> EvalReport:
        """
        Run the evaluation harness on a dataset of golden QA samples.

        Args:
            dataset     : list of EvalSample. Defaults to the full GOLDEN_DATASET.
            metrics     : subset of the four metric names. Defaults to all four.
            dataset_name: label for the EvalReport (used in filenames).

        Returns:
            EvalReport with per-sample results and aggregate metric averages.
        """
        # Ensure reproducibility
        random.seed(42)
        np.random.seed(42)

        samples = GOLDEN_DATASET if dataset is None else dataset
        selected_metrics = metrics or _ALL_METRICS

        if not samples:
            logger.warning("EvaluationHarness.run called with empty dataset.")
            return EvalReport(
                dataset_name=dataset_name,
                n_samples=0,
                n_failed=0,
                metric_averages={},
                sample_results=[],
                total_latency_seconds=0.0,
            )

        logger.info(
            f"EvaluationHarness starting | "
            f"samples={len(samples)} | metrics={selected_metrics} | "
            f"workers={_eval_cfg.max_workers}"
        )
        t_total = time.perf_counter()
        sample_results: list[EvalSampleResult] = [None] * len(samples)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=_eval_cfg.max_workers) as pool:
            fut_to_idx = {
                pool.submit(self._run_sample, s, selected_metrics): i for i, s in enumerate(samples)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    sample_results[idx] = fut.result()
                except Exception as exc:
                    logger.error(f"Harness future error at idx={idx}: {exc}")
                    sample_results[idx] = EvalSampleResult(
                        sample=samples[idx],
                        generated_answer="",
                        context_chunks=[],
                        pipeline_failed=True,
                        error_message=str(exc),
                    )

        # Compute aggregate metric averages (exclude failed samples)
        successful = [r for r in sample_results if not r.pipeline_failed]
        metric_averages: dict[str, float] = {}
        metric_confidence_intervals: dict[str, tuple[float, float]] = {}
        for metric in selected_metrics:
            scores = [
                r.metric_scores[metric].score for r in successful if metric in r.metric_scores
            ]
            if scores:
                metric_averages[metric] = sum(scores) / len(scores)
                # Compute 95% Bootstrap CI dynamically
                metric_confidence_intervals[metric] = compute_bootstrap_ci(scores)
            else:
                metric_averages[metric] = 0.0
                metric_confidence_intervals[metric] = (0.0, 0.0)

        total_latency = time.perf_counter() - t_total
        n_failed = sum(1 for r in sample_results if r.pipeline_failed)

        report = EvalReport(
            dataset_name=dataset_name,
            n_samples=len(samples),
            n_failed=n_failed,
            metric_averages=metric_averages,
            metric_confidence_intervals=metric_confidence_intervals,
            sample_results=sample_results,
            total_latency_seconds=total_latency,
        )

        logger.info(
            f"EvaluationHarness complete | {len(samples)} samples | "
            f"{n_failed} failed | {total_latency:.1f}s\n"
            + "\n".join(f"  {k}: {v:.3f}" for k, v in metric_averages.items())
        )
        return report

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_report(self, report: EvalReport) -> tuple[Path, Path]:
        """
        Save the report to disk as both JSON and CSV.

        Returns:
            (json_path, csv_path) — absolute paths of the written files.
        """
        out_dir = Path(_eval_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp slug for unique filenames
        ts = report.timestamp.replace(":", "-").replace("+", "").split(".")[0]
        stem = f"{report.dataset_name}_{ts}"

        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.csv"

        json_path.write_text(report.to_json(), encoding="utf-8")
        csv_path.write_text(report.to_csv(), encoding="utf-8")

        logger.info(f"EvalReport saved: {json_path}, {csv_path}")
        return json_path, csv_path


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from qdrant_client import QdrantClient

    from rag_pipeline import FinancialRAGPipeline

    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--n", type=int, default=0, help="Number of samples (0 = all)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=_ALL_METRICS,
        choices=_ALL_METRICS,
        help="Metrics to compute",
    )
    parser.add_argument("--name", default="golden_dataset", help="Report name")
    args = parser.parse_args()

    _settings.validate()
    client = QdrantClient(url=_settings.infra.qdrant_url)
    pipeline = FinancialRAGPipeline(qdrant_client=client)
    harness = EvaluationHarness(pipeline)

    from evaluation.dataset import get_dataset_subset

    samples = get_dataset_subset(args.n) if args.n > 0 else None
    report = harness.run(dataset=samples, metrics=args.metrics, dataset_name=args.name)
    print(report.summary())
    harness.save_report(report)
