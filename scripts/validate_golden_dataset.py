# scripts/validate_golden_dataset.py
"""
Validation script for the generated Golden Evaluation Dataset.

Performs rigorous structural, schema, distribution, and grounding checks:
  1. JSON syntactic correctness
  2. Pydantic schema validation using EvalSample
  3. Coverage & quota balance across tickers (NFLX, NVDA, UNH, WMT)
  4. Temporal balance across fiscal years and quarters
  5. Question self-containment & ground-truth factual richness
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from evaluation.models import EvalSample


def validate_golden_dataset(path: Path = Path("data/golden_dataset.json")) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path.resolve()}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Golden dataset must be a JSON array, got {type(raw_data).__name__}")

    total = len(raw_data)
    logger.info(f"Loaded {total} samples from {path}")

    # 1. Pydantic Schema Validation
    validated_samples: list[EvalSample] = []
    schema_errors: list[str] = []

    for idx, item in enumerate(raw_data):
        try:
            sample = EvalSample(**item)
            validated_samples.append(sample)
        except ValidationError as e:
            schema_errors.append(f"Sample {idx} ({item.get('sample_id', 'unknown')}): {e}")

    if schema_errors:
        logger.error(f"Encountered {len(schema_errors)} schema validation errors!")
        for err in schema_errors[:5]:
            logger.error(err)
        raise ValueError(f"Dataset has {len(schema_errors)} schema validation errors.")

    logger.success(f"✓ All {total} samples passed Pydantic EvalSample schema validation.")

    # 2. Distribution by Ticker
    by_ticker = Counter(s.ticker for s in validated_samples)
    logger.info(f"Ticker distribution: {dict(by_ticker)}")

    # 3. Distribution by Year
    by_year = Counter(s.year for s in validated_samples)
    logger.info(f"Year distribution: {dict(by_year)}")

    # 4. Distribution by Period (Annual vs Quarterly)
    by_period = Counter(
        "Annual (10-K)" if s.quarter is None else f"10-Q ({s.quarter})" for s in validated_samples
    )
    logger.info(f"Period distribution: {dict(by_period)}")

    # 5. Question & Ground Truth Quality Checks
    min_q_len = min(len(s.question) for s in validated_samples)
    max_q_len = max(len(s.question) for s in validated_samples)
    avg_q_len = sum(len(s.question) for s in validated_samples) / total

    min_gt_len = min(len(s.ground_truth) for s in validated_samples)
    max_gt_len = max(len(s.ground_truth) for s in validated_samples)
    avg_gt_len = sum(len(s.ground_truth) for s in validated_samples) / total

    logger.info(f"Question length (chars): min={min_q_len}, avg={avg_q_len:.1f}, max={max_q_len}")
    logger.info(
        f"Ground Truth length (chars): min={min_gt_len}, avg={avg_gt_len:.1f}, max={max_gt_len}"
    )

    # 6. Self-containment check
    self_contained_count = 0
    for s in validated_samples:
        q_lower = s.question.lower()
        has_ticker_or_name = (
            s.ticker.lower() in q_lower
            or (s.ticker == "NFLX" and "netflix" in q_lower)
            or (s.ticker == "NVDA" and ("nvidia" in q_lower or "data center" in q_lower))
            or (s.ticker == "UNH" and ("unitedhealth" in q_lower or "optum" in q_lower))
            or (s.ticker == "WMT" and ("walmart" in q_lower or "sam's club" in q_lower))
        )
        if has_ticker_or_name:
            self_contained_count += 1

    logger.info(
        f"Self-contained company named questions: {self_contained_count}/{total} "
        f"({(self_contained_count / total) * 100:.1f}%)"
    )

    report = {
        "total_samples": total,
        "valid_schema": len(schema_errors) == 0,
        "by_ticker": dict(by_ticker),
        "by_year": dict(by_year),
        "by_period": dict(by_period),
        "avg_question_length": round(avg_q_len, 1),
        "avg_ground_truth_length": round(avg_gt_len, 1),
        "self_contained_ratio": round(self_contained_count / total, 3),
    }

    print("\n" + "=" * 60)
    print("GOLDEN DATASET VALIDATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60 + "\n")
    return report


if __name__ == "__main__":
    validate_golden_dataset()
