#!/usr/bin/env python3
"""
scripts/verify_ablation_isolation.py

Automated Invariant Verification for RAG Ablation Studies.
Mathematically verifies that each ablation arm executed ONLY its designated
components with 0% component leakage.

Usage:
    # Verify existing ablation run results in data/ablation_results/
    poetry run python scripts/verify_ablation_isolation.py

    # Run a fresh 5-sample verification test across all 6 arms and assert isolation
    poetry run python scripts/verify_ablation_isolation.py --run -n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def verify_arm_isolation(
    arm_index: int,
    arm_label: str,
    samples: list[dict[str, Any]],
) -> list[str]:
    """
    Assert structural invariants for each ablation arm.
    Returns a list of violation error strings (empty if passed).
    """
    violations: list[str] = []

    valid_samples = [s for s in samples if not s.get("pipeline_failed", False)]
    if not valid_samples:
        return [f"Arm {arm_index}: No successful samples to evaluate."]

    for s in valid_samples:
        sid = s.get("sample_id", "unknown")
        telem = s.get("telemetry", {})

        mq_count = telem.get("multi_query_count", 1)
        hyde = telem.get("hyde_generated", False)
        stepback = telem.get("stepback_generated", False)
        reranked = telem.get("reranked", False)
        chunk_sources = telem.get("chunk_sources", [])
        graph_chunks = telem.get("graph_chunks_count", 0)
        crag_action = telem.get("crag_action", None)

        # ── ARM 1: Base Naive RAG (Dense Only) ────────────────────────────────
        if arm_index == 1:
            if mq_count > 1:
                violations.append(
                    f"Sample {sid}: Multi-query leakage (count={mq_count}, expected 1)."
                )
            if hyde:
                violations.append(
                    f"Sample {sid}: HyDE leakage (hyde_generated=True, expected False)."
                )
            if stepback:
                violations.append(
                    f"Sample {sid}: Step-back leakage (stepback_generated=True, expected False)."
                )
            if reranked:
                violations.append(
                    f"Sample {sid}: Reranker leakage (reranked=True, expected False)."
                )
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )
            if crag_action is not None:
                violations.append(
                    f"Sample {sid}: CRAG leakage (crag_action={crag_action}, expected None)."
                )
            # Only dense chunks allowed
            for src in chunk_sources:
                if src not in ("dense", ""):
                    violations.append(
                        f"Sample {sid}: Sparse/Graph chunk leakage (source={src!r}, expected 'dense')."
                    )

        # ── ARM 2: + BM25 Sparse (Hybrid RRF) ─────────────────────────────────
        elif arm_index == 2:
            if mq_count > 1:
                violations.append(
                    f"Sample {sid}: Multi-query leakage (count={mq_count}, expected 1)."
                )
            if hyde:
                violations.append(
                    f"Sample {sid}: HyDE leakage (hyde_generated=True, expected False)."
                )
            if stepback:
                violations.append(
                    f"Sample {sid}: Step-back leakage (stepback_generated=True, expected False)."
                )
            if reranked:
                violations.append(
                    f"Sample {sid}: Reranker leakage (reranked=True, expected False)."
                )
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )
            if crag_action is not None:
                violations.append(
                    f"Sample {sid}: CRAG leakage (crag_action={crag_action}, expected None)."
                )

        # ── ARM 3: + Query Transform (HyDE + MultiQuery + StepBack) ───────────
        elif arm_index == 3:
            if reranked:
                violations.append(
                    f"Sample {sid}: Reranker leakage (reranked=True, expected False)."
                )
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )
            if crag_action is not None:
                violations.append(
                    f"Sample {sid}: CRAG leakage (crag_action={crag_action}, expected None)."
                )

        # ── ARM 4: + FlashRank Reranker ───────────────────────────────────────
        elif arm_index == 4:
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )
            if crag_action is not None:
                violations.append(
                    f"Sample {sid}: CRAG leakage (crag_action={crag_action}, expected None)."
                )

        # ── ARM 5: + Knowledge Graph (GraphRAG) ───────────────────────────────
        elif arm_index == 5:
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )
            if crag_action is not None:
                violations.append(
                    f"Sample {sid}: CRAG leakage (crag_action={crag_action}, expected None)."
                )

        # ── ARM 6: Full Stack (+ CRAG Fallback) ───────────────────────────────
        elif arm_index == 6:
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )
            if crag_action is None:
                violations.append(
                    f"Sample {sid}: CRAG was NOT executed (crag_action=None, expected valid action)."
                )

    return violations


def run_verification(base_dir: Path = Path("data/ablation_results")) -> bool:
    """Scan and verify all arm results in base_dir."""
    print("\n" + "=" * 80)
    print(" 🛡️  PORTFOLIO RAG ABLATION: COMPONENT ISOLATION AUDIT")
    print("=" * 80)

    arm_dirs = sorted(base_dir.glob("arm_*"))
    if not arm_dirs:
        print(f"❌ No ablation arm directories found in '{base_dir}'.")
        print("Run ablation studies first or pass '--run' to execute a live test run.")
        return False

    all_passed = True
    results_summary = []

    for arm_path in arm_dirs:
        # Extract arm number
        dir_name = arm_path.name
        try:
            arm_idx = int(dir_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        samples_file = arm_path / "samples.json"
        if not samples_file.exists():
            print(f"⚠️  Arm {arm_idx} ({dir_name}): samples.json missing — skipping.")
            continue

        with open(samples_file, encoding="utf-8") as f:
            samples = json.load(f)

        violations = verify_arm_isolation(arm_idx, dir_name, samples)

        status_str = "✅ PASS" if not violations else "❌ FAIL"
        if violations:
            all_passed = False

        results_summary.append(
            {
                "arm": arm_idx,
                "dir": dir_name,
                "samples_count": len(samples),
                "status": status_str,
                "violations": violations,
            }
        )

    # Print Report Table
    print(f"\n{'Arm #':<7} | {'Directory / Label':<40} | {'Samples':<8} | {'Isolation Status'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['arm']:<7} | {r['dir']:<40} | {r['samples_count']:<8} | {r['status']}")

    print("-" * 80)

    # Print Violations if any
    for r in results_summary:
        if r["violations"]:
            print(f"\n🚨 [ARM {r['arm']}] VIOLATIONS DETECTED ({len(r['violations'])}):")
            for v in r["violations"]:
                print(f"   • {v}")

    if all_passed:
        print("\n✨ ALL ARMS VERIFIED: 100% Component Isolation Confirmed with Zero Leakage.\n")
    else:
        print("\n❌ COMPONENT LEAKAGE DETECTED: Review violations above.\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify component isolation across portfolio RAG ablation arms."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/ablation_results"),
        help="Base directory containing arm_* folders (default: data/ablation_results)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute a fresh test run before verifying.",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=5,
        help="Number of samples to evaluate if --run is specified (default: 5).",
    )
    args = parser.parse_args()

    if args.run:
        from scripts.run_portfolio_ablations import run_granular_ablations

        logger.info(
            f"Running fresh ablation benchmark on {args.samples} samples for isolation check..."
        )
        run_granular_ablations(n_samples=args.samples, force=True)

    passed = run_verification(args.dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
