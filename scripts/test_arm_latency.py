"""
Diagnostic script to run a single question across ablation arms to benchmark
true live query-to-answer pipeline latency.

Usage:
    poetry run python scripts/test_arm_latency.py [--arm 1] [--sample-idx 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config import settings
from experiments.retrieval_experiment import ExperimentConfig
from scripts.run_portfolio_ablations import get_baseline_arm, make_pipeline


def run_single_arm_test(
    arm_idx: int,
    cfg: ExperimentConfig,
    sample: dict[str, Any],
    update_sample_in_cache: bool = False,
) -> dict[str, Any]:
    logger.info(f"\n{'=' * 60}\nTesting Arm {arm_idx}: {cfg.label}\n{'=' * 60}")

    # 1. Apply environment overrides
    env_patch = cfg.to_env_patch()
    original_env: dict[str, str | None] = {}
    for k, v in env_patch.items():
        original_env[k] = os.environ.get(k)
        os.environ[k] = v

    settings.reload()

    # 2. Instantiate pipeline
    pipeline = make_pipeline()
    pipeline.ensure_ready()

    question = sample["question"]
    logger.info(f"Question ({sample['sample_id']}): {question}")

    # 3. Time pure query-to-answer pipeline latency
    t_start = time.perf_counter()
    call_res = pipeline.ask(question)
    result = asyncio.run(call_res) if asyncio.iscoroutine(call_res) else call_res
    pipeline_latency = round(time.perf_counter() - t_start, 3)

    # 4. Extract telemetry and spans
    trace = getattr(pipeline, "last_trace", None)

    l2_latency = trace.query_transform.latency_seconds if (trace and trace.query_transform) else 0.0
    l3_latency = trace.retrieval.latency_seconds if (trace and trace.retrieval) else 0.0
    l4_latency = trace.generation.latency_seconds if (trace and trace.generation) else 0.0
    graph_latency = (
        trace.graph_retrieval.latency_seconds if (trace and trace.graph_retrieval) else 0.0
    )

    # Restore env
    for k, orig in original_env.items():
        if orig is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = orig
    settings.reload()

    record = {
        "arm_idx": arm_idx,
        "label": cfg.label,
        "sample_id": sample["sample_id"],
        "pipeline_latency_s": pipeline_latency,
        "l2_latency_s": round(l2_latency, 3),
        "l3_latency_s": round(l3_latency, 3),
        "l4_latency_s": round(l4_latency, 3),
        "graph_latency_s": round(graph_latency, 3),
        "retrieved_chunks": len(result.retrieved_chunks)
        if result.retrieved_chunks
        else len(result.citations),
        "tokens_prompt": result.prompt_tokens,
        "tokens_completion": result.completion_tokens,
        "answer_snippet": result.answer[:150].replace("\n", " "),
    }

    # If update_sample_in_cache requested, update this sample in arm's samples.json
    if update_sample_in_cache:
        import re

        from scripts.run_portfolio_ablations import _BASE_OUT_DIR

        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", cfg.label.lower()).strip("_")
        arm_dir = _BASE_OUT_DIR / f"arm_{arm_idx}_{slug}"
        cache_file = arm_dir / "samples.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                cached_samples = json.load(f)
            found = False
            for s in cached_samples:
                if s.get("sample_id") == sample["sample_id"]:
                    s["generated_answer"] = result.answer
                    s["pipeline_latency_seconds"] = pipeline_latency
                    found = True
                    break
            if found:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_samples, f, indent=2)
                logger.info(
                    f"Updated sample {sample['sample_id']} in {cache_file} with live answer and latency."
                )

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pipeline latency on 1 question across arms.")
    parser.add_argument(
        "--arm", type=int, default=None, help="Specific arm to run (1-6). Default: all arms."
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of question in golden_dataset.json (default 0).",
    )
    parser.add_argument(
        "--random", action="store_true", help="Pick a random question instead of sample-idx."
    )
    parser.add_argument(
        "--update-cache", action="store_true", help="Overwrite this sample in arm's samples.json."
    )
    args = parser.parse_args()

    with open("data/golden_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.random:
        sample_idx = secrets.randbelow(len(dataset))
    else:
        sample_idx = args.sample_idx

    sample = dataset[sample_idx]
    logger.info(f"Selected Question [{sample_idx}]: {sample['sample_id']}")
    logger.info(f"Q: {sample['question']}")

    arms = [get_baseline_arm()]
    if args.arm is not None:
        target_indices = [args.arm]
    else:
        target_indices = list(range(1, len(arms) + 1))

    results = []
    for idx in target_indices:
        cfg = arms[idx - 1]
        rec = run_single_arm_test(idx, cfg, sample, update_sample_in_cache=args.update_cache)
        results.append(rec)

    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS FOR QUESTION: {sample['sample_id']}")
    print(f"Q: {sample['question'][:80]}...")
    print("=" * 90)
    print(
        f"{'Arm':<4} | {'Arm Label':<40} | {'Total (s)':<9} | {'L2 (s)':<7} | {'L3 (s)':<7} | {'L4 (s)':<7}"
    )
    print("-" * 81)
    for r in results:
        print(
            f"{r['arm_idx']:<4} | {r['label']:<40} | {r['pipeline_latency_s']:>8.3f}s | {r['l2_latency_s']:>6.3f}s | {r['l3_latency_s']:>6.3f}s | {r['l4_latency_s']:>6.3f}s"
        )
    print("=" * 81 + "\n")


if __name__ == "__main__":
    main()
