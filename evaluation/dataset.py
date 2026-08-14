# evaluation/dataset.py
"""
Golden QA dataset module for offline evaluation of the Financial RAG system.

Loads the sole single golden dataset file from `data/golden_dataset.json`,
generated from actual SEC filings in `data/company_filings/`.

Usage:
    from evaluation.dataset import GOLDEN_DATASET, get_dataset_by_ticker
    samples = get_dataset_by_ticker("NVDA")
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from evaluation.models import EvalSample

GOLDEN_DATASET_PATH = Path("data/golden_dataset.json")


def load_golden_dataset(custom_path: Path | None = None) -> list[EvalSample]:
    """
    Load the single golden evaluation dataset from `data/golden_dataset.json`.

    Returns a list of `EvalSample` objects.
    """
    path = custom_path or GOLDEN_DATASET_PATH
    if not path.exists():
        logger.warning(
            f"Golden dataset file not found at {path.resolve()}. "
            "Run `poetry run python -m scripts.generate_golden_dataset` to generate it."
        )
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            sample = EvalSample(
                sample_id=item.get("sample_id", ""),
                question=item.get("question", ""),
                ground_truth=item.get("ground_truth", ""),
                ticker=item.get("ticker"),
                year=item.get("year"),
                quarter=item.get("quarter"),
            )
            if sample.sample_id and sample.question and sample.ground_truth:
                samples.append(sample)
        return samples
    except Exception as exc:
        logger.error(f"Failed to load golden dataset from {path}: {exc}")
        return []


GOLDEN_DATASET: list[EvalSample] = load_golden_dataset()


def get_dataset_by_ticker(ticker: str) -> list[EvalSample]:
    """Return all samples matching a specific ticker."""
    dataset = load_golden_dataset()
    return [s for s in dataset if (s.ticker or "").upper() == ticker.upper()]


def get_dataset_subset(n: int) -> list[EvalSample]:
    """Return the first n samples from the golden dataset."""
    dataset = load_golden_dataset()
    return dataset[:n]
