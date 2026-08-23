# scripts/run_portfolio_ablations.py
"""
Orchestrator script to generate authentic, granular component-by-component ablation metrics.

Evaluates 6 incremental RAG architecture arms:
1. Base Naive RAG (Dense Only)
2. + BM25 Sparse Search (+ Hybrid RRF)
3. + Query Transformation (+ HyDE & Multi-Query)
4. + Cross-Encoder Reranking (+ FlashRank)
5. + Knowledge Graph Context Injection (+ GraphRAG)
6. Full Stack (+ Corrective RAG / CRAG)

Prints two formatted Markdown tables:
- Absolute Metrics Table (Faithfulness, Relevancy, Precision, Recall, Latency)
- Incremental Component Marginal Lift Table (+X.XX improvement per layer)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from config import settings
from evaluation.dataset import load_golden_dataset
from experiments.retrieval_experiment import ExperimentConfig, RetrievalExperiment
from rag_pipeline import FinancialRAGPipeline


def make_pipeline() -> FinancialRAGPipeline:
    """Factory to create a new default pipeline for the experiment."""
    try:
        client = QdrantClient(url=settings.infra.qdrant_url, timeout=5, check_compatibility=False)
        cols = {c.name for c in client.get_collections().collections}
        if settings.embedding.collection_name in cols:
            pt_count = client.count(settings.embedding.collection_name).count
            logger.info(
                f"Connected to Qdrant at {settings.infra.qdrant_url} | "
                f"collection '{settings.embedding.collection_name}' has {pt_count} points"
            )
        else:
            logger.warning(
                f"Collection '{settings.embedding.collection_name}' not found at {settings.infra.qdrant_url}"
            )
    except Exception as exc:
        logger.warning(
            f"Could not connect to Qdrant at {settings.infra.qdrant_url} ({exc}) — falling back to local storage"
        )
        client = QdrantClient(path="data/qdrant_user_storage")

    return FinancialRAGPipeline(
        qdrant_client=client,
        enable_query_cache=False,
        generation_model="gpt-5-mini",
    )


def get_ablation_arms() -> list[ExperimentConfig]:
    """Return the list of 6 granular incremental ablation configs."""
    return [
        ExperimentConfig(
            label="1. Base Naive RAG (Dense Only)",
            top_k_bm25=0,
            hyde_enabled=False,
            multiquery_enabled=False,
            stepback_enabled=False,
            reranker_enabled=False,
            graphrag_enabled=False,
            use_crag=False,
        ),
        ExperimentConfig(
            label="2. + BM25 Sparse (Hybrid RRF)",
            top_k_bm25=25,
            hyde_enabled=False,
            multiquery_enabled=False,
            stepback_enabled=False,
            reranker_enabled=False,
            graphrag_enabled=False,
            use_crag=False,
        ),
        ExperimentConfig(
            label="3. + Query Transform (HyDE + MultiQuery + StepBack)",
            top_k_bm25=25,
            hyde_enabled=True,
            multiquery_enabled=True,
            stepback_enabled=True,
            reranker_enabled=False,
            graphrag_enabled=False,
            use_crag=False,
        ),
        ExperimentConfig(
            label="4. + FlashRank Reranker",
            top_k_bm25=25,
            hyde_enabled=True,
            multiquery_enabled=True,
            stepback_enabled=True,
            reranker_enabled=True,
            graphrag_enabled=False,
            use_crag=False,
        ),
        ExperimentConfig(
            label="5. + Knowledge Graph (GraphRAG)",
            top_k_bm25=25,
            hyde_enabled=True,
            multiquery_enabled=True,
            stepback_enabled=True,
            reranker_enabled=True,
            graphrag_enabled=True,
            use_crag=False,
        ),
        ExperimentConfig(
            label="6. Full Stack (+ CRAG Fallback)",
            top_k_bm25=25,
            hyde_enabled=True,
            multiquery_enabled=True,
            stepback_enabled=True,
            reranker_enabled=True,
            graphrag_enabled=True,
            use_crag=True,
        ),
    ]


def run_granular_ablations(
    n_samples: int = 0,
    target_arms: list[int] | None = None,
    force: bool = False,
) -> None:
    dataset = load_golden_dataset()
    if n_samples > 0:
        dataset = dataset[:n_samples]

    logger.info(
        f"=== Starting Granular Portfolio RAG Ablation (n={len(dataset)} samples across tickers: { {s.ticker for s in dataset} }) ==="
    )

    exp = RetrievalExperiment(pipeline_factory=make_pipeline)

    # Base output directory
    base_out_dir = Path("data/ablation_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Define the 6 Granular Incremental Arms ────────────────────────────────
    all_arms = get_ablation_arms()
    if target_arms:
        valid_arms = [a for a in target_arms if 1 <= a <= len(all_arms)]
        if not valid_arms:
            logger.error(
                f"No valid arms selected from {target_arms}. Available arms: 1 to {len(all_arms)}"
            )
            return
        selected_arms = [(idx, all_arms[idx - 1]) for idx in valid_arms]
    else:
        selected_arms = list(enumerate(all_arms, start=1))

    results_map = {}

    for idx, cfg in selected_arms:
        logger.info(f"\n{'=' * 70}\n🚀 RUNNING ARM {idx}: {cfg.label}\n{'=' * 70}")

        # Create subfolder for this arm
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", cfg.label.lower()).strip("_")
        arm_dir = base_out_dir / f"arm_{idx}_{slug}"
        arm_dir.mkdir(parents=True, exist_ok=True)

        arm_res = exp._run_arm(
            cfg, dataset=dataset, metrics=exp._METRICS, arm_dir=arm_dir, force_recompute=force
        )
        results_map[cfg.label] = arm_res

        # 1. Save samples.json (already checkpointed during _run_arm)
        samples_file = arm_dir / "samples.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(arm_res.sample_scores, f, indent=2)

        # 2. Save summary_metrics.json
        metric_stats = {}
        for m in exp._METRICS:
            vals = [
                s["scores"].get(m, 0.0) for s in arm_res.sample_scores if not s["pipeline_failed"]
            ]
            if vals:
                metric_stats[m] = {
                    "mean": round(statistics.mean(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "std_dev": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                }
            else:
                metric_stats[m] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std_dev": 0.0}

        summary_data = {
            "arm_label": cfg.label,
            "sample_count": len(dataset),
            "error_count": arm_res.pipeline_errors,
            "total_latency_seconds": round(arm_res.total_latency_s, 2),
            "avg_latency_seconds": round(arm_res.total_latency_s / max(1, len(dataset)), 2),
            "metric_summary": metric_stats,
        }
        with open(arm_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 3. Save arm_report.md
        arm_md = f"# Arm Report: {cfg.label}\n\n"
        arm_md += f"- **Evaluated Samples**: {len(dataset)}\n"
        arm_md += f"- **Errors**: {arm_res.pipeline_errors}\n"
        arm_md += f"- **Avg Latency**: {summary_data['avg_latency_seconds']}s\n\n"
        arm_md += "## Metric Scores Summary\n\n"
        arm_md += "| Metric | Mean | Min | Max | Std Dev |\n|:---|:---:|:---:|:---:|:---:|\n"
        for m, st in metric_stats.items():
            arm_md += f"| **{m}** | {st['mean']:.4f} | {st['min']:.4f} | {st['max']:.4f} | {st['std_dev']:.4f} |\n"

        with open(arm_dir / "arm_report.md", "w", encoding="utf-8") as f:
            f.write(arm_md)

        logger.info(
            f"Saved arm artifacts → {arm_dir}/ (samples.json, summary_metrics.json, arm_report.md)"
        )

    # ── Master Reports (Root data/ablation_results/) ──────────────────────────
    # Save ablation_summary.json
    all_summary = {
        "dataset_samples": len(dataset),
        "arms": {
            cfg.label: {
                "metrics": results_map[cfg.label].metric_scores,
                "avg_latency_s": round(
                    results_map[cfg.label].total_latency_s / max(1, len(dataset)), 2
                ),
                "pipeline_errors": results_map[cfg.label].pipeline_errors,
            }
            for _, cfg in selected_arms
        },
    }
    with open(base_out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    # ── Table 1: Absolute Performance Table ──────────────────────────────────
    print("\n\n" + "=" * 110)
    print(
        "### 📊 TABLE 1: Absolute RAG Architecture Metrics (LLM-as-a-Judge + Statistical NLP + Semantic Sim) ###"
    )
    print("=" * 110)
    headers = [
        "Arm",
        "Faithfulness",
        "Relevancy",
        "Precision",
        "Recall",
        "Token F1",
        "ROUGE-1",
        "ROUGE-L",
        "BLEU-4",
        "Semantic Sim",
        "Avg Latency",
    ]
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "|:" + ":|:".join(["----------------"] * len(headers)) + ":|"
    print(header_str)
    print(sep_str)

    master_md = "# Comprehensive RAG Ablation & Metric Report\n\n"
    master_md += f"**Evaluated Samples**: {len(dataset)} financial QA pairs\n\n"
    master_md += (
        "## Table 1: Absolute Pipeline Performance Metrics\n\n" + header_str + "\n" + sep_str + "\n"
    )

    for _, cfg in selected_arms:
        res = results_map[cfg.label]
        f_val = res.avg("faithfulness")
        a_val = res.avg("answer_relevancy")
        p_val = res.avg("context_precision")
        r_val = res.avg("context_recall")
        tf1_val = res.avg("token_f1")
        r1_val = res.avg("rouge1_f1")
        rl_val = res.avg("rougeL_f1")
        b4_val = res.avg("bleu_4")
        sim_val = res.avg("semantic_similarity")
        lat = res.total_latency_s / max(1, len(dataset))

        row_str = f"| **{cfg.label}** | {f_val:.3f} | {a_val:.3f} | {p_val:.3f} | {r_val:.3f} | {tf1_val:.3f} | {r1_val:.3f} | {rl_val:.3f} | {b4_val:.3f} | {sim_val:.3f} | {lat:.2f}s |"
        print(row_str)
        master_md += row_str + "\n"

    print("=" * 110)

    # ── Table 2: Incremental Marginal Component Lift Table ────────────────────
    if len(selected_arms) > 1 and selected_arms[0][0] == 1:
        print("\n\n" + "=" * 110)
        print("### 📈 TABLE 2: Incremental Component Marginal Lift (Layer-by-Layer) ###")
        print("=" * 110)
        headers_t2 = [
            "Added Component Layer",
            "Targeted Capability",
            "Delta Faithfulness",
            "Delta Precision",
            "Delta Recall",
            "Delta Token F1",
            "Delta ROUGE-1",
            "Delta Semantic Sim",
            "Latency Change",
        ]
        header_t2_str = "| " + " | ".join(headers_t2) + " |"
        sep_t2_str = "|:" + ":|:".join(["----------------"] * len(headers_t2)) + ":|"
        print(header_t2_str)
        print(sep_t2_str)

        master_md += (
            "\n## Table 2: Incremental Component Marginal Lift (Layer-by-Layer)\n\n"
            + header_t2_str
            + "\n"
            + sep_t2_str
            + "\n"
        )

        prev_res = None
        for loop_i, (orig_idx, cfg) in enumerate(selected_arms):
            res = results_map[cfg.label]
            if loop_i == 0:
                target = "Dense Vector Retrieval Baseline"
                d_f = d_p = d_r = d_tf1 = d_r1 = d_sim = "0.000"
                d_lat = "+0.00s"
            else:
                assert prev_res is not None
                d_f_val = res.avg("faithfulness") - prev_res.avg("faithfulness")
                d_p_val = res.avg("context_precision") - prev_res.avg("context_precision")
                d_r_val = res.avg("context_recall") - prev_res.avg("context_recall")
                d_tf1_val = res.avg("token_f1") - prev_res.avg("token_f1")
                d_r1_val = res.avg("rouge1_f1") - prev_res.avg("rouge1_f1")
                d_sim_val = res.avg("semantic_similarity") - prev_res.avg("semantic_similarity")
                d_lat_val = (res.total_latency_s - prev_res.total_latency_s) / max(1, len(dataset))

                d_f = f"+{d_f_val:.3f}" if d_f_val >= 0 else f"{d_f_val:.3f}"
                d_p = f"+{d_p_val:.3f}" if d_p_val >= 0 else f"{d_p_val:.3f}"
                d_r = f"+{d_r_val:.3f}" if d_r_val >= 0 else f"{d_r_val:.3f}"
                d_tf1 = f"+{d_tf1_val:.3f}" if d_tf1_val >= 0 else f"{d_tf1_val:.3f}"
                d_r1 = f"+{d_r1_val:.3f}" if d_r1_val >= 0 else f"{d_r1_val:.3f}"
                d_sim = f"+{d_sim_val:.3f}" if d_sim_val >= 0 else f"{d_sim_val:.3f}"
                d_lat = f"+{d_lat_val:.2f}s" if d_lat_val >= 0 else f"{d_lat_val:.2f}s"

                targets = [
                    "",
                    "Exact Table & Financial Keyword Matching",
                    "Query Expansion & HyDE Document Synthesis",
                    "Deep Cross-Encoder Re-ranking",
                    "Multi-Hop Entity Link Context Injection",
                    "Self-Correction & Web Search Fallback",
                ]
                target = targets[orig_idx - 1] if orig_idx - 1 < len(targets) else ""

            comp_name = cfg.label.split(". ", 1)[-1]
            row_t2_str = f"| **{comp_name}** | {target} | {d_f} | {d_p} | {d_r} | {d_tf1} | {d_r1} | {d_sim} | {d_lat} |"
            print(row_t2_str)
            master_md += row_t2_str + "\n"
            prev_res = res

        print("=" * 110 + "\n")

    with open(base_out_dir / "ablation_report.md", "w", encoding="utf-8") as f:
        f.write(master_md)

    logger.info(f"Master ablation report saved to {base_out_dir}/ablation_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 6-arm portfolio RAG ablation study.")
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=10,
        help="Number of dataset samples to evaluate (default: 10, pass 0 or use --all for the entire dataset)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate against ALL questions in the golden dataset (equivalent to -n 0)",
    )
    parser.add_argument(
        "--arm",
        "--arms",
        nargs="+",
        type=int,
        default=None,
        dest="arms",
        help="Specific arm number(s) to evaluate (1 to 6). Example: --arm 1 or --arms 1 2 3",
    )
    parser.add_argument(
        "--no-cache",
        "--force",
        action="store_true",
        dest="force",
        help="Ignore cached sample checkpoints and re-evaluate all samples from scratch.",
    )
    args = parser.parse_args()
    n_samples = 0 if args.all else args.n_samples
    run_granular_ablations(n_samples=n_samples, target_arms=args.arms, force=args.force)
