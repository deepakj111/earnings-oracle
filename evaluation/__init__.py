# evaluation/__init__.py
"""
Evaluation harness for the Financial RAG system.

Public API:

    from evaluation import EvaluationHarness
    from evaluation.dataset import GOLDEN_DATASET, get_dataset_subset
    from evaluation.metrics import score_all
    from evaluation.models import EvalReport, EvalSample

    harness = EvaluationHarness(pipeline)
    report = harness.run(dataset=get_dataset_subset(5))
    print(report.summary())
    harness.save_report(report)

CLI:
    poetry run python -m evaluation.harness --n 5 --metrics faithfulness answer_relevancy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluation.models import EvalReport, EvalSample, EvalSampleResult, MetricScore

if TYPE_CHECKING:
    from evaluation.harness import EvaluationHarness


def __getattr__(name: str) -> Any:
    if name == "EvaluationHarness":
        from evaluation.harness import EvaluationHarness

        return EvaluationHarness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvaluationHarness",
    "EvalReport",
    "EvalSample",
    "EvalSampleResult",
    "MetricScore",
]
