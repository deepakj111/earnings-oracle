# experiments/__init__.py
"""
Retrieval experiment framework for controlled A/B ablation.

Usage:
    from experiments.retrieval_experiment import (
        RetrievalExperiment,
        ExperimentConfig,
        ExperimentReport,
    )
"""

from experiments.retrieval_experiment import (
    ArmResult,
    ExperimentConfig,
    ExperimentReport,
    RetrievalExperiment,
)

__all__ = [
    "RetrievalExperiment",
    "ExperimentConfig",
    "ExperimentReport",
    "ArmResult",
]
