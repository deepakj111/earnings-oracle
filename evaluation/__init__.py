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

from evaluation.harness import EvaluationHarness
from evaluation.models import EvalReport, EvalSample, EvalSampleResult, MetricScore

__all__ = [
    "EvaluationHarness",
    "EvalReport",
    "EvalSample",
    "EvalSampleResult",
    "MetricScore",
]
