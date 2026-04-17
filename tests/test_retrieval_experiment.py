# tests/test_retrieval_experiment.py
"""
Tests for experiments/retrieval_experiment.py — A/B retrieval experiment framework.

Tests cover:
  - ExperimentConfig env patch generation
  - ExperimentConfig diff_vs() detection of changed fields
  - ExperimentReport delta() and winner() logic
  - ExperimentReport diff_summary() formatting
  - ExperimentReport to_dict() serialisation
  - ExperimentReport save() file creation
  - RetrievalExperiment.run() with mocked pipeline (no real LLM calls)
  - ArmResult metric aggregation
  - CLI entrypoint argument parsing (import-level smoke test)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.retrieval_experiment import (
    ArmResult,
    ExperimentConfig,
    ExperimentReport,
    RetrievalExperiment,
)


class TestExperimentConfig:
    def test_env_patch_top_k_final(self) -> None:
        cfg = ExperimentConfig(label="test", top_k_final=8)
        patch = cfg.to_env_patch()
        assert patch["RAG_RETRIEVAL_TOP_K_FINAL"] == "8"

    def test_env_patch_reranker_false(self) -> None:
        cfg = ExperimentConfig(label="no_rerank", reranker_enabled=False)
        patch = cfg.to_env_patch()
        assert patch["RAG_RERANKER_ENABLED"] == "false"

    def test_env_patch_empty_when_no_overrides(self) -> None:
        cfg = ExperimentConfig(label="default")
        assert cfg.to_env_patch() == {}

    def test_env_patch_multiple_fields(self) -> None:
        cfg = ExperimentConfig(label="multi", top_k_final=3, rrf_k_constant=30)
        patch = cfg.to_env_patch()
        assert "RAG_RETRIEVAL_TOP_K_FINAL" in patch
        assert "RAG_RETRIEVAL_RRF_K" in patch

    def test_diff_vs_detects_changed_field(self) -> None:
        baseline = ExperimentConfig(label="baseline", top_k_final=5)
        variant = ExperimentConfig(label="variant", top_k_final=8)
        diffs = baseline.diff_vs(variant)
        assert "top_k_final" in diffs
        assert diffs["top_k_final"] == (5, 8)

    def test_diff_vs_no_diff_when_identical(self) -> None:
        cfg = ExperimentConfig(label="same", top_k_final=5)
        assert cfg.diff_vs(cfg) == {}

    def test_diff_vs_reranker_toggle(self) -> None:
        baseline = ExperimentConfig(label="b", reranker_enabled=True)
        variant = ExperimentConfig(label="v", reranker_enabled=False)
        diffs = baseline.diff_vs(variant)
        assert "reranker_enabled" in diffs


class TestArmResult:
    def _make_arm(self, scores: dict[str, float], errors: int = 0) -> ArmResult:
        return ArmResult(
            config=ExperimentConfig(label="test"),
            metric_scores=scores,
            sample_scores=[],
            total_latency_s=10.0,
            pipeline_errors=errors,
        )

    def test_avg_returns_correct_score(self) -> None:
        arm = self._make_arm({"faithfulness": 0.82})
        assert arm.avg("faithfulness") == pytest.approx(0.82)

    def test_avg_returns_zero_for_missing_metric(self) -> None:
        arm = self._make_arm({})
        assert arm.avg("faithfulness") == 0.0


class TestExperimentReport:
    def _make_report(
        self,
        baseline_scores: dict[str, float],
        variant_scores: dict[str, float],
    ) -> ExperimentReport:
        baseline = ArmResult(
            config=ExperimentConfig(label="baseline", top_k_final=5),
            metric_scores=baseline_scores,
            sample_scores=[],
            total_latency_s=20.0,
            pipeline_errors=0,
        )
        variant = ArmResult(
            config=ExperimentConfig(label="variant", top_k_final=8),
            metric_scores=variant_scores,
            sample_scores=[],
            total_latency_s=22.0,
            pipeline_errors=0,
        )
        return ExperimentReport(
            name="test_exp",
            baseline=baseline,
            variant=variant,
            n_samples=5,
            metrics_evaluated=list(baseline_scores.keys()),
        )

    def test_delta_positive_when_variant_better(self) -> None:
        report = self._make_report({"faithfulness": 0.70}, {"faithfulness": 0.80})
        assert report.delta("faithfulness") == pytest.approx(0.10)

    def test_delta_negative_when_baseline_better(self) -> None:
        report = self._make_report({"faithfulness": 0.85}, {"faithfulness": 0.75})
        assert report.delta("faithfulness") < 0

    def test_winner_variant(self) -> None:
        report = self._make_report({"answer_relevancy": 0.70}, {"answer_relevancy": 0.82})
        assert report.winner("answer_relevancy") == "variant"

    def test_winner_baseline(self) -> None:
        report = self._make_report({"context_precision": 0.80}, {"context_precision": 0.60})
        assert report.winner("context_precision") == "baseline"

    def test_winner_tie_within_tolerance(self) -> None:
        report = self._make_report({"faithfulness": 0.80}, {"faithfulness": 0.805})
        assert report.winner("faithfulness") == "tie"

    def test_diff_summary_contains_metric_name(self) -> None:
        report = self._make_report({"faithfulness": 0.80}, {"faithfulness": 0.85})
        summary = report.diff_summary()
        assert "faithfulness" in summary

    def test_diff_summary_contains_baseline_label(self) -> None:
        report = self._make_report({"faithfulness": 0.8}, {"faithfulness": 0.9})
        assert "baseline" in report.diff_summary()

    def test_diff_summary_contains_variant_label(self) -> None:
        report = self._make_report({"faithfulness": 0.8}, {"faithfulness": 0.9})
        assert "variant" in report.diff_summary()

    def test_to_dict_has_required_keys(self) -> None:
        report = self._make_report({"faithfulness": 0.8}, {"faithfulness": 0.9})
        d = report.to_dict()
        assert "name" in d
        assert "baseline" in d
        assert "variant" in d
        assert "metrics_evaluated" in d

    def test_save_creates_file(self, tmp_path: Path) -> None:
        report = self._make_report({"faithfulness": 0.8}, {"faithfulness": 0.9})
        path = report.save(output_dir=str(tmp_path))
        assert Path(path).exists()
        with open(path) as fh:
            data = json.load(fh)
        assert data["name"] == "test_exp"

    def test_save_json_is_valid(self, tmp_path: Path) -> None:
        report = self._make_report(
            {"faithfulness": 0.8, "answer_relevancy": 0.9},
            {"faithfulness": 0.85, "answer_relevancy": 0.88},
        )
        path = report.save(output_dir=str(tmp_path))
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data["baseline"]["metric_scores"], dict)


class TestRetrievalExperimentRun:
    def test_run_returns_experiment_report(self) -> None:
        from generation.models import GenerationResult

        mock_result = MagicMock(spec=GenerationResult)
        mock_result.answer = "Apple revenue was $94.9B in Q4 2024."
        mock_result.citations = []

        mock_pipeline = MagicMock()
        mock_pipeline.ask.return_value = mock_result

        mock_scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.75,
            "context_recall": 0.80,
        }

        with patch(
            "experiments.retrieval_experiment.compute_all_metrics",
            return_value=mock_scores,
        ):
            with patch(
                "experiments.retrieval_experiment.GOLDEN_DATASET",
                [
                    MagicMock(id="q1", question="What was AAPL revenue?", ground_truth="$94.9B"),
                    MagicMock(id="q2", question="What was NVDA revenue?", ground_truth="$35.1B"),
                ],
            ):
                exp = RetrievalExperiment(pipeline_factory=lambda: mock_pipeline)
                report = exp.run(
                    baseline=ExperimentConfig(label="baseline", top_k_final=5),
                    variant=ExperimentConfig(label="variant", top_k_final=8),
                    n_samples=2,
                    metrics=["faithfulness", "answer_relevancy"],
                    name="unit_test_exp",
                )

        assert isinstance(report, ExperimentReport)
        assert report.name == "unit_test_exp"
        assert report.n_samples == 2

    def test_run_handles_pipeline_error_gracefully(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.ask.side_effect = RuntimeError("Qdrant unavailable")

        with patch(
            "experiments.retrieval_experiment.GOLDEN_DATASET",
            [MagicMock(id="q1", question="What was AAPL revenue?", ground_truth="$94.9B")],
        ):
            exp = RetrievalExperiment(pipeline_factory=lambda: mock_pipeline)
            report = exp.run(
                baseline=ExperimentConfig(label="b", top_k_final=5),
                variant=ExperimentConfig(label="v", top_k_final=8),
                n_samples=1,
                metrics=["faithfulness"],
                name="error_test",
            )

        assert report.baseline.pipeline_errors == 1
        assert report.variant.pipeline_errors == 1

    def test_env_restored_after_arm(self) -> None:
        original_val = os.environ.get("RAG_RETRIEVAL_TOP_K_FINAL")
        mock_pipeline = MagicMock()
        mock_pipeline.ask.return_value = MagicMock(answer="test", citations=[])

        with patch("experiments.retrieval_experiment.compute_all_metrics", return_value={}):
            with patch(
                "experiments.retrieval_experiment.GOLDEN_DATASET",
                [MagicMock(id="q1", question="test?", ground_truth="test")],
            ):
                exp = RetrievalExperiment(pipeline_factory=lambda: mock_pipeline)
                exp.run(
                    baseline=ExperimentConfig(label="b", top_k_final=3),
                    variant=ExperimentConfig(label="v", top_k_final=7),
                    n_samples=1,
                    metrics=["faithfulness"],
                )

        assert os.environ.get("RAG_RETRIEVAL_TOP_K_FINAL") == original_val
