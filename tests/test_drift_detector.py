"""
Unit tests for DriftDetector and MMD distribution shift detection.
"""

import numpy as np

from evaluation.drift_detector import DriftDetector


class TestDriftDetector:
    def test_identical_distributions_yield_near_zero_mmd(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 16).astype(np.float32)
        report = DriftDetector.evaluate_drift(
            current_embeddings=X,
            baseline_embeddings=X,
            threshold=0.05,
            kernel="rbf",
        )
        assert report.mmd_score < 1e-4
        assert report.drift_detected is False

    def test_shifted_distributions_detect_drift(self) -> None:
        np.random.seed(42)
        # Distribution P centered at 0
        X = np.random.randn(50, 16).astype(np.float32)
        # Distribution Q centered at 5.0 (major covariate shift)
        Y = (np.random.randn(50, 16) + 5.0).astype(np.float32)

        report = DriftDetector.evaluate_drift(
            current_embeddings=Y,
            baseline_embeddings=X,
            threshold=0.05,
            kernel="rbf",
        )
        assert report.mmd_score > 0.05
        assert report.drift_detected is True

    def test_cosine_kernel_drift_detection(self) -> None:
        np.random.seed(42)
        X = np.random.randn(30, 8).astype(np.float32)
        report = DriftDetector.evaluate_drift(
            current_embeddings=X,
            baseline_embeddings=X,
            threshold=0.05,
            kernel="cosine",
        )
        assert report.mmd_score < 1e-4
        assert report.drift_detected is False

    def test_empty_samples_handled_gracefully(self) -> None:
        mmd = DriftDetector.compute_mmd([], [])
        assert mmd == 0.0
