"""
Production-grade Embedding Distribution Drift Detection (MMD).

Detects covariate shift / semantic drift in user query distributions over time
by computing Maximum Mean Discrepancy (MMD) between production query embeddings
(from audit logs) and the baseline evaluation distribution (e.g., golden dataset).

Used to alert ML engineering teams when incoming financial questions diverge
significantly from the retrieval index and evaluation benchmarks — a core
requirement for production monitoring of financial RAG systems.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class DriftReport:
    """Statistical summary of query embedding distribution drift."""

    mmd_score: float
    p_estimate: float
    drift_detected: bool
    threshold: float
    current_sample_size: int
    baseline_sample_size: int
    kernel: str
    cosine_distance_mean: float = 0.0
    cosine_distance_std: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "ALERT: SIGNIFICANT DRIFT DETECTED" if self.drift_detected else "STABLE: NO DRIFT"
        return (
            f"\n{'=' * 55}\n"
            f"   EMBEDDING DISTRIBUTION DRIFT AUDIT REPORT\n"
            f"{'=' * 55}\n"
            f"Status           : {status}\n"
            f"MMD Score        : {self.mmd_score:.6f} (Threshold: {self.threshold:.4f})\n"
            f"Sample Counts    : Current={self.current_sample_size} | Baseline={self.baseline_sample_size}\n"
            f"Kernel Used      : {self.kernel}\n"
            f"Mean Cosine Dist : {self.cosine_distance_mean:.4f} ± {self.cosine_distance_std:.4f}\n"
            f"{'=' * 55}\n"
        )


class DriftDetector:
    """
    Computes Maximum Mean Discrepancy (MMD) between query embedding representations.
    """

    @staticmethod
    def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """
        Compute RBF kernel matrix between X (m, d) and Y (n, d).
        k(x, y) = exp(-gamma * ||x - y||^2)
        """
        if gamma is None:
            # Median heuristic for gamma
            dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
            dists = np.maximum(dists, 0.0)
            median_dist = float(np.median(dists))
            gamma = 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0 / X.shape[1]

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        dist_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    @staticmethod
    def cosine_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Normalized cosine similarity kernel matrix between X and Y."""
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
        return np.dot(X_norm, Y_norm.T)

    @classmethod
    def compute_mmd(
        cls,
        P: list[list[float]] | np.ndarray,
        Q: list[list[float]] | np.ndarray,
        kernel: str = "rbf",
        gamma: float | None = None,
    ) -> float:
        """
        Compute MMD^2 between two embedding sets P and Q.

        MMD^2(P, Q) = E[k(x, x')] - 2E[k(x, y)] + E[k(y, y')]
        """
        if len(P) == 0 or len(Q) == 0:
            return 0.0

        X = np.asarray(P, dtype=np.float32)
        Y = np.asarray(Q, dtype=np.float32)

        if X.size == 0 or Y.size == 0:
            return 0.0

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        m = X.shape[0]
        n = Y.shape[0]

        if m == 0 or n == 0 or X.shape[1] == 0:
            return 0.0

        if kernel == "rbf":
            K_XX = cls.rbf_kernel(X, X, gamma=gamma)
            K_YY = cls.rbf_kernel(Y, Y, gamma=gamma)
            K_XY = cls.rbf_kernel(X, Y, gamma=gamma)
        else:
            K_XX = cls.cosine_kernel(X, X)
            K_YY = cls.cosine_kernel(Y, Y)
            K_XY = cls.cosine_kernel(X, Y)

        # Unbiased or standard MMD calculation
        mmd_sq = float(np.mean(K_XX) - 2.0 * np.mean(K_XY) + np.mean(K_YY))
        return max(0.0, math.sqrt(max(0.0, mmd_sq)))

    @classmethod
    def evaluate_drift(
        cls,
        current_embeddings: list[list[float]] | np.ndarray,
        baseline_embeddings: list[list[float]] | np.ndarray,
        threshold: float = 0.05,
        kernel: str = "rbf",
    ) -> DriftReport:
        """
        Evaluate statistical drift between current production samples and baseline.
        """
        X = np.asarray(current_embeddings, dtype=np.float32)
        Y = np.asarray(baseline_embeddings, dtype=np.float32)

        mmd = cls.compute_mmd(X, Y, kernel=kernel)

        # Compute cosine distance summary
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
        sim_matrix = np.dot(X_norm, Y_norm.T)
        dist_matrix = 1.0 - sim_matrix
        cos_mean = float(np.mean(dist_matrix))
        cos_std = float(np.std(dist_matrix))

        drift_detected = mmd >= threshold

        return DriftReport(
            mmd_score=mmd,
            p_estimate=round(max(0.0, 1.0 - (mmd / max(threshold, 1e-6))), 4),
            drift_detected=drift_detected,
            threshold=threshold,
            current_sample_size=X.shape[0],
            baseline_sample_size=Y.shape[0],
            kernel=kernel,
            cosine_distance_mean=cos_mean,
            cosine_distance_std=cos_std,
        )


def load_embeddings_from_file(path: Path | str) -> list[list[float]]:
    """Helper to load JSON list of vectors or JSONL audit logs."""
    p = Path(path)
    if not p.exists():
        logger.error(f"File not found: {p}")
        return []

    vectors: list[list[float]] = []
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    vec = record.get("query_vector") or record.get("embedding")
                    if vec:
                        vectors.append(vec)
    else:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list):
                        vectors.append(item)
                    elif isinstance(item, dict) and "embedding" in item:
                        vectors.append(item["embedding"])
    return vectors


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate embedding distribution drift via MMD.")
    parser.add_argument(
        "--current", type=str, required=True, help="Path to current embeddings JSON/JSONL"
    )
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to baseline embeddings JSON/JSONL"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="MMD drift alert threshold (default 0.05)"
    )
    parser.add_argument(
        "--kernel", type=str, default="rbf", choices=["rbf", "cosine"], help="Kernel type"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional output path for JSON report"
    )

    args = parser.parse_args()

    curr_vecs = load_embeddings_from_file(args.current)
    base_vecs = load_embeddings_from_file(args.baseline)

    if not curr_vecs or not base_vecs:
        logger.error("Failed to load vectors from provided files.")
        return

    report = DriftDetector.evaluate_drift(
        current_embeddings=curr_vecs,
        baseline_embeddings=base_vecs,
        threshold=args.threshold,
        kernel=args.kernel,
    )

    print(report.summary())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mmd_score": report.mmd_score,
                    "drift_detected": report.drift_detected,
                    "threshold": report.threshold,
                    "current_sample_size": report.current_sample_size,
                    "baseline_sample_size": report.baseline_sample_size,
                },
                f,
                indent=2,
            )
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
