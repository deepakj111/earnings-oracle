# experiments/retrieval_experiment.py
"""
Retrieval Experiment Framework — controlled A/B ablation for retrieval config.

Allows comparing any two pipeline configurations against the golden eval dataset
and producing a structured diff report:  baseline vs. variant.

This is the scientific backbone of retrieval improvement decisions. Instead of
"I think top_k=8 is better," you run an experiment and commit the result.

Supported experiment axes:
  - top_k_final          : final chunks after reranking (e.g. 3 vs 5 vs 8)
  - reranker_enabled     : FlashRank ON vs OFF
  - rrf_k_constant       : RRF k=60 vs k=30 vs k=100
  - hyde_enabled         : HyDE ON vs OFF (via routing)
  - top_k_dense / bm25   : retrieval pool size

Usage:
    poetry run python -m experiments.retrieval_experiment \\
        --baseline '{"top_k_final": 5, "reranker_enabled": true}' \\
        --variant  '{"top_k_final": 8, "reranker_enabled": true}' \\
        --n 16 \\
        --name "top_k_ablation_5_vs_8"

    # Or programmatically:
    from experiments.retrieval_experiment import RetrievalExperiment, ExperimentConfig

    exp = RetrievalExperiment(pipeline_factory=make_pipeline)
    report = exp.run(
        baseline=ExperimentConfig(label="baseline", top_k_final=5),
        variant=ExperimentConfig(label="top_k_8",  top_k_final=8),
        n_samples=16,
    )
    print(report.diff_summary())
    report.save("data/experiments/")
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from evaluation.dataset import GOLDEN_DATASET, EvalSample
from evaluation.metrics import compute_all_metrics


@dataclass
class ExperimentConfig:
    """
    Configuration overrides for one arm of an experiment.

    All fields are optional; unspecified fields inherit from the live settings.
    Only the fields listed here are patched; everything else in the pipeline
    runs with its default configuration.
    """

    label: str = "unnamed"
    top_k_final: int | None = None
    top_k_dense: int | None = None
    top_k_bm25: int | None = None
    rrf_k_constant: int | None = None
    reranker_enabled: bool | None = None
    hyde_enabled: bool | None = None

    def to_env_patch(self) -> dict[str, str]:
        """Convert config fields to environment variable overrides."""
        patch: dict[str, str] = {}
        if self.top_k_final is not None:
            patch["RAG_RETRIEVAL_TOP_K_FINAL"] = str(self.top_k_final)
        if self.top_k_dense is not None:
            patch["RAG_RETRIEVAL_TOP_K_DENSE"] = str(self.top_k_dense)
        if self.top_k_bm25 is not None:
            patch["RAG_RETRIEVAL_TOP_K_BM25"] = str(self.top_k_bm25)
        if self.rrf_k_constant is not None:
            patch["RAG_RETRIEVAL_RRF_K"] = str(self.rrf_k_constant)
        if self.reranker_enabled is not None:
            patch["RAG_RERANKER_ENABLED"] = str(self.reranker_enabled).lower()
        return patch

    def diff_vs(self, other: ExperimentConfig) -> dict[str, tuple[Any, Any]]:
        """Return fields that differ between self and other as {field: (self_val, other_val)}."""
        diffs: dict[str, tuple[Any, Any]] = {}
        for f_name in (
            "top_k_final",
            "top_k_dense",
            "top_k_bm25",
            "rrf_k_constant",
            "reranker_enabled",
            "hyde_enabled",
        ):
            a, b = getattr(self, f_name), getattr(other, f_name)
            if a != b:
                diffs[f_name] = (a, b)
        return diffs


@dataclass
class ArmResult:
    """
    Evaluation results for one experiment arm (baseline or variant).
    """

    config: ExperimentConfig
    metric_scores: dict[str, float]
    sample_scores: list[dict[str, Any]]
    total_latency_s: float
    pipeline_errors: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def avg(self, metric: str) -> float:
        return self.metric_scores.get(metric, 0.0)


@dataclass
class ExperimentReport:
    """
    Full A/B experiment report comparing baseline vs. variant arm.
    """

    name: str
    baseline: ArmResult
    variant: ArmResult
    n_samples: int
    metrics_evaluated: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def delta(self, metric: str) -> float:
        """variant score - baseline score for a metric."""
        return self.variant.avg(metric) - self.baseline.avg(metric)

    def winner(self, metric: str) -> str:
        """Returns 'baseline', 'variant', or 'tie' for a given metric."""
        d = self.delta(metric)
        if abs(d) < 0.01:
            return "tie"
        return "variant" if d > 0 else "baseline"

    def diff_summary(self) -> str:
        lines = [
            f"\n{'=' * 65}",
            f"  EXPERIMENT: {self.name}",
            f"  Created  : {self.created_at}",
            f"  Samples  : {self.n_samples}",
            f"  Baseline : {self.baseline.config.label}",
            f"  Variant  : {self.variant.config.label}",
            f"{'=' * 65}",
            "",
            "  CONFIG DIFF:",
        ]
        diffs = self.baseline.config.diff_vs(self.variant.config)
        if diffs:
            for k, (bv, vv) in diffs.items():
                lines.append(f"    {k}: baseline={bv}  variant={vv}")
        else:
            lines.append("    (no config differences — identical configs)")
        lines.append("")
        lines.append("  METRIC SCORES:")
        lines.append(
            f"    {'Metric':<26} {'Baseline':>9} {'Variant':>9} {'Delta':>9} {'Winner':<10}"
        )
        lines.append(f"    {'-' * 64}")
        for m in self.metrics_evaluated:
            b_score = self.baseline.avg(m)
            v_score = self.variant.avg(m)
            delta = self.delta(m)
            winner = self.winner(m)
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"    {m:<26} {b_score:>9.4f} {v_score:>9.4f} {sign}{delta:>8.4f} {winner:<10}"
            )
        lines.append("")
        lines.append(
            f"  Baseline errors: {self.baseline.pipeline_errors}  "
            f"Variant errors: {self.variant.pipeline_errors}"
        )
        lines.append(
            f"  Baseline latency: {self.baseline.total_latency_s:.1f}s  "
            f"Variant latency: {self.variant.total_latency_s:.1f}s"
        )
        lines.append(f"{'=' * 65}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "n_samples": self.n_samples,
            "metrics_evaluated": self.metrics_evaluated,
            "baseline": {
                "config": asdict(self.baseline.config),
                "metric_scores": self.baseline.metric_scores,
                "total_latency_s": self.baseline.total_latency_s,
                "pipeline_errors": self.baseline.pipeline_errors,
                "timestamp": self.baseline.timestamp,
            },
            "variant": {
                "config": asdict(self.variant.config),
                "metric_scores": self.variant.metric_scores,
                "total_latency_s": self.variant.total_latency_s,
                "pipeline_errors": self.variant.pipeline_errors,
                "timestamp": self.variant.timestamp,
            },
        }

    def save(self, output_dir: str = "data/experiments") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        slug = self.name.replace(" ", "_").lower()
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(output_dir) / f"{slug}_{ts}.json"
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info(f"Experiment report saved → {path}")
        return str(path)


PipelineFactory = Callable[[], Any]


class RetrievalExperiment:
    """
    Runs controlled A/B experiments over retrieval configuration.

    The pipeline_factory callable must return a FinancialRAGPipeline-compatible
    object with an .ask(question) method. Environment variable overrides are
    applied before each arm runs by patching os.environ directly.

    After each arm, os.environ is restored to its pre-experiment state.
    """

    _METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(self, pipeline_factory: PipelineFactory) -> None:
        self._factory = pipeline_factory

    def run(
        self,
        baseline: ExperimentConfig,
        variant: ExperimentConfig,
        n_samples: int = 16,
        metrics: list[str] | None = None,
        name: str | None = None,
    ) -> ExperimentReport:
        """
        Run baseline and variant arms sequentially and return a diff report.

        Args:
            baseline  : Baseline configuration arm
            variant   : Variant configuration arm to compare against baseline
            n_samples : Number of samples from the golden dataset to evaluate
            metrics   : Subset of metrics to compute (defaults to all four)
            name      : Human-readable experiment name for report

        Returns:
            ExperimentReport with per-metric scores and diff table
        """
        metrics = metrics or self._METRICS
        exp_name = name or f"{baseline.label}_vs_{variant.label}"
        dataset = GOLDEN_DATASET[:n_samples]

        logger.info(f"Experiment '{exp_name}' starting | n={n_samples} samples | metrics={metrics}")
        logger.info(f"  baseline config: {asdict(baseline)}")
        logger.info(f"  variant  config: {asdict(variant)}")

        baseline_result = self._run_arm(baseline, dataset, metrics)
        variant_result = self._run_arm(variant, dataset, metrics)

        report = ExperimentReport(
            name=exp_name,
            baseline=baseline_result,
            variant=variant_result,
            n_samples=n_samples,
            metrics_evaluated=metrics,
        )
        logger.info(report.diff_summary())
        return report

    def _run_arm(
        self,
        config: ExperimentConfig,
        dataset: list[EvalSample],
        metrics: list[str],
    ) -> ArmResult:
        """
        Run a single experiment arm: patch env, build pipeline, evaluate, restore env.
        """
        env_patch = config.to_env_patch()
        original_env: dict[str, str | None] = {}

        for k, v in env_patch.items():
            original_env[k] = os.environ.get(k)
            os.environ[k] = v

        logger.info(f"Running arm '{config.label}' | env_patch={env_patch}")
        t_start = time.perf_counter()

        try:
            import config.settings as settings_module
            from config.settings import Settings

            new_settings = Settings()
            settings_module.settings = new_settings

            pipeline = self._factory()
            sample_scores: list[dict[str, Any]] = []
            errors = 0

            for sample in dataset:
                try:
                    result = pipeline.ask(sample.question)
                    scores = compute_all_metrics(
                        question=sample.question,
                        answer=result.answer,
                        context_chunks=[c.text for c in result.citations],
                        ground_truth=sample.ground_truth,
                        metrics=metrics,
                    )
                    sample_scores.append(
                        {
                            "sample_id": sample.id,
                            "question": sample.question,
                            "scores": scores,
                            "pipeline_failed": False,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"  Sample {sample.id} failed: {exc}")
                    errors += 1
                    sample_scores.append(
                        {
                            "sample_id": sample.id,
                            "question": sample.question,
                            "scores": {m: 0.0 for m in metrics},
                            "pipeline_failed": True,
                        }
                    )

            total_latency = time.perf_counter() - t_start

            agg: dict[str, float] = {}
            valid = [s for s in sample_scores if not s["pipeline_failed"]]
            for m in metrics:
                if valid:
                    agg[m] = sum(s["scores"].get(m, 0.0) for s in valid) / len(valid)
                else:
                    agg[m] = 0.0

            logger.info(
                f"Arm '{config.label}' complete | "
                f"errors={errors}/{len(dataset)} | "
                f"latency={total_latency:.1f}s | "
                f"scores={agg}"
            )
            return ArmResult(
                config=config,
                metric_scores=agg,
                sample_scores=sample_scores,
                total_latency_s=total_latency,
                pipeline_errors=errors,
            )

        finally:
            for k, original_v in original_env.items():
                if original_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = original_v

            import config.settings as settings_module

            settings_module.settings = Settings()


def _cli_main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run a retrieval A/B experiment against the golden eval dataset."
    )
    parser.add_argument("--baseline", type=str, required=True, help="JSON config for baseline arm")
    parser.add_argument("--variant", type=str, required=True, help="JSON config for variant arm")
    parser.add_argument("--n", type=int, default=16, help="Number of eval samples")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save", action="store_true", help="Save report to data/experiments/")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    )
    args = parser.parse_args()

    baseline_dict = json.loads(args.baseline)
    variant_dict = json.loads(args.variant)

    baseline_cfg = ExperimentConfig(label="baseline", **baseline_dict)
    variant_cfg = ExperimentConfig(label="variant", **variant_dict)

    from qdrant_client import QdrantClient

    from config import settings
    from rag_pipeline import FinancialRAGPipeline

    def make_pipeline() -> FinancialRAGPipeline:
        client = QdrantClient(url=settings.infra.qdrant_url)
        return FinancialRAGPipeline(qdrant_client=client)

    exp = RetrievalExperiment(pipeline_factory=make_pipeline)
    report = exp.run(
        baseline=baseline_cfg,
        variant=variant_cfg,
        n_samples=args.n,
        metrics=args.metrics,
        name=args.name,
    )

    print(report.diff_summary())

    if args.save:
        path = report.save()
        print(f"Saved → {path}")

    any_variant_wins = any(report.winner(m) == "variant" for m in args.metrics)
    sys.exit(0 if any_variant_wins else 1)


if __name__ == "__main__":
    _cli_main()
