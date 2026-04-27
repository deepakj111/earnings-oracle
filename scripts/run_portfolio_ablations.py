"""
Orchestrator script to generate authentic metrics for the Portfolio README Table.

Runs evaluations on a small subset across four main RAG architecture variants:
1. Naive RAG: Dense only
2. Hybrid + GraphRAG: BM25 + Dense + GraphRAG
3. Hybrid + GraphRAG + Reranker: (Above) + Cross-Encoder
4. Full Stack (+ CRAG): (Above) + Corrective RAG

Prints a Markdown table at the end exactly matching the README's expected format.
"""

from qdrant_client import QdrantClient

from config import settings
from experiments.retrieval_experiment import ExperimentConfig, RetrievalExperiment
from rag_pipeline import FinancialRAGPipeline


def make_pipeline() -> FinancialRAGPipeline:
    """Factory to create a new default pipeline for the experiment."""
    client = QdrantClient(url=settings.infra.qdrant_url)
    return FinancialRAGPipeline(qdrant_client=client)


def main():
    """Execute evaluation variants sequentially and print the unified portfolio README table."""
    exp = RetrievalExperiment(pipeline_factory=make_pipeline)

    # Test on a small representative batch so it runs fast and cheap
    # (Update n_samples if you want to run the full dataset)
    n_samples = 10

    print(f"=== Running Portfolio RAG Ablation (n={n_samples} samples) ===")

    # -------------------------------------------------------------
    # Arm 1: Naive RAG (Dense only)
    # -------------------------------------------------------------
    naive_cfg = ExperimentConfig(
        label="Naive RAG (Dense only)",
        top_k_bm25=0,  # Disable sparse
        reranker_enabled=False,  # Disable reranking
        graphrag_enabled=False,  # Disable GraphRAG
        use_crag=False,  # Disable CRAG
    )

    # -------------------------------------------------------------
    # Arm 2: Hybrid + GraphRAG
    # -------------------------------------------------------------
    hybrid_cfg = ExperimentConfig(
        label="Hybrid + GraphRAG",
        top_k_bm25=10,  # Enable sparse
        reranker_enabled=False,  # Still no reranking
        graphrag_enabled=True,  # Enable GraphRAG
        use_crag=False,
    )

    # -------------------------------------------------------------
    # Arm 3: Hybrid + GraphRAG + Reranker
    # -------------------------------------------------------------
    reranker_cfg = ExperimentConfig(
        label="Hybrid + Reranker",
        top_k_bm25=10,
        reranker_enabled=True,  # Enable reranking
        graphrag_enabled=True,
        use_crag=False,
    )

    # -------------------------------------------------------------
    # Arm 4: Full Stack (+ CRAG)
    # -------------------------------------------------------------
    full_stack_cfg = ExperimentConfig(
        label="Full Stack (+ CRAG)",
        top_k_bm25=10,
        reranker_enabled=True,
        graphrag_enabled=True,
        use_crag=True,  # Full pipeline
    )

    configs = [naive_cfg, hybrid_cfg, reranker_cfg, full_stack_cfg]
    results_map = {}

    # Execute sequentially using the experiment runner
    for cfg in configs:
        print(f"\n--- Running: {cfg.label} ---")
        from evaluation.dataset import GOLDEN_DATASET

        dataset = GOLDEN_DATASET[:n_samples]

        arm_res = exp._run_arm(cfg, dataset=dataset, metrics=exp._METRICS)
        results_map[cfg.label] = arm_res

    print("\n\n" + "=" * 80)
    print("### Authentic Portfolio Evaluation Table ###")
    print("=" * 80)
    print("| Architecture Variant | Faithfulness | Answer Relevancy | Context Precision |")
    print("|----------------------|--------------|------------------|-------------------|")

    for cfg in configs:
        res = results_map[cfg.label]
        # We don't have bootstrap uncertainty logic natively coupled inside ArmResult cleanly
        # So we'll just print point estimates, but format them nicely
        f_score = res.avg("faithfulness")
        a_score = res.avg("answer_relevancy")
        c_score = res.avg("context_precision")
        print(f"| **{cfg.label}** | {f_score:.3f} | {a_score:.3f} | {c_score:.3f} |")

    print("=" * 80)


if __name__ == "__main__":
    main()
