# evaluation/models.py
"""
Data contracts for the Evaluation Harness.

  EvalSample      → one golden QA pair from the dataset
  MetricScore     → one metric's score + LLM reasoning for a single sample
  EvalSampleResult→ full evaluation output for a single pipeline run
  EvalReport      → aggregated results across the full dataset
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvalSample:
    """
    One golden QA pair used for offline evaluation.

    question    : the financial question to ask
    ground_truth: the expected answer (used for recall scoring)
    ticker      : optional — restricts retrieval to this company
    year        : optional — restricts retrieval to this fiscal year
    quarter     : optional — restricts retrieval to this fiscal quarter
    sample_id   : unique identifier for deduplication and logging
    """

    question: str
    ground_truth: str
    ticker: str | None = None
    year: int | None = None
    quarter: str | None = None
    sample_id: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id:
            import hashlib

            self.sample_id = hashlib.sha256(self.question.encode()).hexdigest()[:12]


@dataclass
class MetricScore:
    """
    LLM-produced score for a single metric on a single sample.

    score    : 0.0 to 1.0 (higher is better for all metrics)
    reasoning: one-sentence explanation from the evaluator LLM
    """

    metric: str
    score: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class EvalSampleResult:
    """
    Full evaluation output for one QA sample.

    pipeline_failed : True if the pipeline raised during the run.
    metric_scores   : {metric_name: MetricScore}
    """

    sample: EvalSample
    generated_answer: str
    context_chunks: list[str]
    metric_scores: dict[str, MetricScore] = field(default_factory=dict)
    latency_seconds: float = 0.0
    pipeline_failed: bool = False
    error_message: str = ""

    @property
    def score_for(self) -> dict[str, float]:
        """Convenience: {metric_name: score_float}"""
        return {k: v.score for k, v in self.metric_scores.items()}

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample.sample_id,
            "question": self.sample.question,
            "ground_truth": self.sample.ground_truth,
            "generated_answer": self.generated_answer,
            "pipeline_failed": self.pipeline_failed,
            "error_message": self.error_message,
            "latency_seconds": round(self.latency_seconds, 3),
            "metric_scores": {k: v.to_dict() for k, v in self.metric_scores.items()},
        }


@dataclass
class EvalReport:
    """
    Aggregated evaluation results across the full dataset.

    metric_averages: mean score per metric across all non-failed samples.
    timestamp      : ISO-8601 UTC string when the report was generated.
    """

    dataset_name: str
    n_samples: int
    n_failed: int
    metric_averages: dict[str, float]
    sample_results: list[EvalSampleResult]
    total_latency_seconds: float
    metric_confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    pipeline_version: str = "0.1.0"

    @property
    def pass_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return (self.n_samples - self.n_failed) / self.n_samples

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "n_samples": self.n_samples,
            "n_failed": self.n_failed,
            "pass_rate": round(self.pass_rate, 4),
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "metric_averages": {k: round(v, 4) for k, v in self.metric_averages.items()},
            "metric_confidence_intervals": {
                k: (round(v[0], 4), round(v[1], 4))
                for k, v in self.metric_confidence_intervals.items()
            },
            "sample_results": [r.to_dict() for r in self.sample_results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """
        Export a flat CSV with one row per sample × metric.
        Suitable for import into pandas, Google Sheets, or MLflow.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "sample_id",
                "question",
                "pipeline_failed",
                "latency_seconds",
                "metric",
                "score",
                "reasoning",
            ]
        )
        for r in self.sample_results:
            if r.pipeline_failed:
                writer.writerow(
                    [
                        r.sample.sample_id,
                        r.sample.question[:80],
                        True,
                        round(r.latency_seconds, 3),
                        "",
                        "",
                        r.error_message,
                    ]
                )
                continue
            for metric_name, ms in r.metric_scores.items():
                writer.writerow(
                    [
                        r.sample.sample_id,
                        r.sample.question[:80],
                        False,
                        round(r.latency_seconds, 3),
                        metric_name,
                        round(ms.score, 4),
                        ms.reasoning[:100],
                    ]
                )
        return buf.getvalue()

    def summary(self) -> str:
        lines = [
            f"=== EvalReport: {self.dataset_name} ===",
            f"Timestamp  : {self.timestamp}",
            f"Samples    : {self.n_samples} total, {self.n_failed} failed "
            f"(pass rate {self.pass_rate:.0%})",
            f"Latency    : {self.total_latency_seconds:.1f}s total",
            "Metrics    :",
        ]
        for metric, avg in self.metric_averages.items():
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            if metric in self.metric_confidence_intervals:
                low, high = self.metric_confidence_intervals[metric]
                ci_str = f"[95% CI: {low:.3f} - {high:.3f}]"
                lines.append(f"  {metric:<25} {bar}  {avg:.3f}  {ci_str}")
            else:
                lines.append(f"  {metric:<25} {bar}  {avg:.3f}")
        return "\n".join(lines)
