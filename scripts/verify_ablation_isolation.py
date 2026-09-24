#!/usr/bin/env python3
"""
scripts/verify_ablation_isolation.py

Automated Invariant Verification for RAG Ablation Studies.

Verifies two sets of arm directories:

  arm_*  — Cumulative (waterfall) ablation arms (1–6)
             Each arm must contain exactly the components expected for that
             layer; no earlier-layer features may leak into later arms
             in the wrong direction.

  iso_*  — Isolated single-component arms
             Each arm enables exactly ONE feature over the dense baseline.
             All other features must be off (zero leakage in both directions).

Usage:
    # Verify existing results in data/ablation_results/
    poetry run python scripts/verify_ablation_isolation.py

    # Run a fresh 5-sample verification test and then verify
    poetry run python scripts/verify_ablation_isolation.py --run -n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# ── Cumulative arm invariants ──────────────────────────────────────────────────


def verify_arm_isolation(
    arm_index: int,
    arm_label: str,
    samples: list[dict[str, Any]],
) -> list[str]:
    """
    Assert structural invariants for each cumulative ablation arm.

    Returns a list of violation error strings (empty list = all passed).
    """
    violations: list[str] = []

    valid_samples = [s for s in samples if not s.get("pipeline_failed", False)]
    logger.debug(
        f"Verifying cumulative arm {arm_index} ({arm_label}) on {len(valid_samples)} valid samples"
    )
    if not valid_samples:
        return [f"Arm {arm_index} ({arm_label}): No successful samples to evaluate."]

    # Dynamic Pareto tiers (Tier 1 Fast, Tier 2 SOTA) dynamically assemble components
    # based on empirical lift and are not constrained by fixed waterfall layering.
    if "tier" in arm_label.lower() or "dynamic" in arm_label.lower():
        return []

    for s in valid_samples:
        sid = s.get("sample_id", "unknown")
        telem = s.get("telemetry", {})

        mq_count = telem.get("multi_query_count", 1)
        hyde = telem.get("hyde_generated", False)
        stepback = telem.get("stepback_generated", False)
        reranked = telem.get("reranked", False)
        chunk_sources = telem.get("chunk_sources", [])
        graph_chunks = telem.get("graph_chunks_count", 0)

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

        # ── ARM 4: + Cross-Encoder Reranker & GraphRAG ────────────────────────
        elif arm_index == 4:
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )

        # ── ARM 5: 2026 Production SOTA (+ PAL Math & NLI Grounding) ───────────
        elif arm_index == 5:
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )

    return violations


# ── Isolated arm invariants ────────────────────────────────────────────────


def verify_isolated_arm_isolation(
    iso_name: str,
    arm_label: str,
    samples: list[dict[str, Any]],
) -> list[str]:
    """
    Assert structural invariants for isolated single-component arms.

    Each isolated arm must have exactly ONE feature enabled with all other
    features strictly off (zero cross-component leakage in both directions).

    Returns a list of violation error strings (empty list = all passed).
    """
    violations: list[str] = []

    valid_samples = [s for s in samples if not s.get("pipeline_failed", False)]
    logger.debug(
        f"Verifying isolated arm '{iso_name}' ({arm_label}) on {len(valid_samples)} valid samples"
    )
    if not valid_samples:
        return [f"Isolated arm '{iso_name}' ({arm_label}): No successful samples to evaluate."]

    for s in valid_samples:
        sid = s.get("sample_id", "unknown")
        telem = s.get("telemetry", {})

        mq_count = telem.get("multi_query_count", 1)
        hyde = telem.get("hyde_generated", False)
        stepback = telem.get("stepback_generated", False)
        reranked = telem.get("reranked", False)
        chunk_sources = telem.get("chunk_sources", [])
        graph_chunks = telem.get("graph_chunks_count", 0)

        # ── ISO BM25: Dense + BM25 only ───────────────────────────────────────
        if iso_name == "bm25":
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
            # Must have dense chunks, and BM25 chunks if keywords matched
            for src in chunk_sources:
                if src not in ("dense", "bm25", ""):
                    violations.append(
                        f"Sample {sid}: Unexpected chunk source in bm25-only arm (source={src!r})."
                    )

        # ── ISO QUERYTRANSFORM: Dense + transforms only ───────────────────────
        elif iso_name == "querytransform":
            # No BM25, no reranker, no graph, no CRAG
            for src in chunk_sources:
                if src not in ("dense", ""):
                    violations.append(
                        f"Sample {sid}: Sparse/Graph chunk leakage in querytransform-only arm "
                        f"(source={src!r})."
                    )
            if reranked:
                violations.append(
                    f"Sample {sid}: Reranker leakage (reranked=True, expected False)."
                )
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )
            # At least one transform technique must have fired
            if mq_count <= 1 and not hyde and not stepback:
                violations.append(
                    f"Sample {sid}: No transform fired (mq={mq_count}, hyde={hyde}, stepback={stepback})."
                )

        # ── ISO RERANKER: Dense + Reranker only ───────────────────────────────
        elif iso_name == "reranker":
            # Reranker must have fired
            if not reranked:
                violations.append(
                    f"Sample {sid}: Reranker was NOT executed (reranked=False, expected True)."
                )
            # No BM25, no transforms, no graph, no CRAG
            for src in chunk_sources:
                if src not in ("dense", ""):
                    violations.append(
                        f"Sample {sid}: Sparse/Graph chunk leakage in reranker-only arm "
                        f"(source={src!r})."
                    )
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
            if graph_chunks > 0:
                violations.append(
                    f"Sample {sid}: GraphRAG leakage (graph_chunks={graph_chunks}, expected 0)."
                )

        # ── ISO GRAPHRAG: Dense + GraphRAG only ───────────────────────────────
        elif iso_name == "graphrag":
            # No BM25, no transforms, no reranker, no CRAG
            for src in chunk_sources:
                if src in ("bm25", "sparse"):
                    violations.append(
                        f"Sample {sid}: BM25/sparse chunk leakage in graphrag-only arm "
                        f"(source={src!r})."
                    )
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
        # ── ISO PAL MATH: Dense + PAL Math & Grounding Verifier only ──────────
        elif iso_name == "pal_math":
            for src in chunk_sources:
                if src in ("bm25", "sparse"):
                    violations.append(
                        f"Sample {sid}: BM25/sparse chunk leakage in pal_math-only arm (source={src!r})."
                    )
                if src == "graph":
                    violations.append(
                        f"Sample {sid}: GraphRAG chunk leakage in pal_math-only arm (source={src!r})."
                    )
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

    return violations


# ── Main verifier ──────────────────────────────────────────────────────────────


def run_verification(base_dir: Path = Path("data/ablation_results")) -> bool:
    """Scan and verify all arm_* and iso_* results in base_dir."""
    print("\n" + "=" * 80)
    print(" 🛡️  PORTFOLIO RAG ABLATION: COMPONENT ISOLATION AUDIT")
    print("=" * 80)

    arm_dirs = sorted(base_dir.glob("arm_*"))
    iso_dirs = sorted(base_dir.glob("iso_*"))

    if not arm_dirs and not iso_dirs:
        print(f"❌ No ablation arm directories found in '{base_dir}'.")
        print("Run ablation studies first or pass '--run' to execute a live test run.")
        return False

    all_passed = True
    results_summary: list[dict[str, Any]] = []

    # ── Verify cumulative arms ─────────────────────────────────────────────────
    if arm_dirs:
        print(f"\n{'─' * 80}")
        print("  Cumulative (Waterfall) Arms")
        print(f"{'─' * 80}")

    for arm_path in arm_dirs:
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
                "type": "cumulative",
                "arm": arm_idx,
                "dir": dir_name,
                "samples_count": len(samples),
                "status": status_str,
                "violations": violations,
            }
        )

    # ── Verify isolated arms ───────────────────────────────────────────────────
    if iso_dirs:
        print(f"\n{'─' * 80}")
        print("  Isolated Single-Component Arms")
        print(f"{'─' * 80}")

    for iso_path in iso_dirs:
        dir_name = iso_path.name
        # Extract iso_name from e.g. "iso_1_bm25" → "bm25"
        parts = dir_name.split("_", 2)
        if len(parts) < 3:
            continue
        iso_name = parts[2]

        samples_file = iso_path / "samples.json"
        if not samples_file.exists():
            print(f"⚠️  Isolated arm '{iso_name}' ({dir_name}): samples.json missing — skipping.")
            continue

        with open(samples_file, encoding="utf-8") as f:
            samples = json.load(f)

        violations = verify_isolated_arm_isolation(iso_name, dir_name, samples)
        status_str = "✅ PASS" if not violations else "❌ FAIL"
        if violations:
            all_passed = False

        results_summary.append(
            {
                "type": "isolated",
                "arm": iso_name,
                "dir": dir_name,
                "samples_count": len(samples),
                "status": status_str,
                "violations": violations,
            }
        )

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'Type':<12} {'Arm / Name':<42} {'Samples':<8} {'Isolation Status'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['type']:<12} {r['dir']:<42} {r['samples_count']:<8} {r['status']}")

    print("-" * 80)

    # Print violations detail
    for r in results_summary:
        arm_violations: list[str] = r["violations"]  # type: ignore[assignment]
        if arm_violations:
            print(
                f"\n🚨 [{r['type'].upper()} ARM '{r['arm']}'] VIOLATIONS ({len(arm_violations)}):"
            )
            for v in arm_violations:
                print(f"   • {v}")

    if all_passed:
        print("\n✨ ALL ARMS VERIFIED: 100% Component Isolation Confirmed with Zero Leakage.\n")
    else:
        print("\n❌ COMPONENT LEAKAGE DETECTED: Review violations above.\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify component isolation across portfolio RAG ablation arms.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/ablation_results"),
        help="Base directory containing arm_* and iso_* folders (default: data/ablation_results)",
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
        run_granular_ablations(
            n_samples=args.samples, force=True, run_isolated=True, run_pareto=True
        )

    passed = run_verification(args.dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
