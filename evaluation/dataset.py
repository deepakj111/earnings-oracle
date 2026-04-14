# evaluation/dataset.py
"""
Golden QA dataset for offline evaluation of the Financial RAG system.

Each EvalSample contains a question + ground_truth answer drawn from
real SEC 8-K earnings filings for the 10 supported companies.

The ground_truth answers are intentionally phrased to match what a well-
grounded RAG system should produce — formal financial register, exact figures
where publicly known, with appropriate hedging for figures that may vary
by filing format or rounding convention.

Usage:
    from evaluation.dataset import GOLDEN_DATASET, get_dataset_by_ticker
    samples = get_dataset_by_ticker("AAPL")
"""

from __future__ import annotations

from evaluation.models import EvalSample

# ── Golden QA pairs ────────────────────────────────────────────────────────────
# Figures are sourced from publicly available 8-K earnings press releases.
# They reflect the filings that the ingestion pipeline downloads from SEC EDGAR.

GOLDEN_DATASET: list[EvalSample] = [
    # ── Apple ──────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="AAPL_Q4_2024_revenue",
        question="What was Apple's total net sales in Q4 fiscal year 2024?",
        ground_truth=(
            "Apple reported total net sales of $94.9 billion for the fourth quarter "
            "of fiscal year 2024, representing a 6% increase year-over-year."
        ),
        ticker="AAPL",
        year=2024,
        quarter="Q4",
    ),
    EvalSample(
        sample_id="AAPL_Q4_2024_services",
        question="How much revenue did Apple's Services segment generate in Q4 2024?",
        ground_truth=(
            "Apple's Services segment generated revenue of approximately $25 billion "
            "in Q4 fiscal year 2024, setting an all-time quarterly record for the segment."
        ),
        ticker="AAPL",
        year=2024,
        quarter="Q4",
    ),
    EvalSample(
        sample_id="AAPL_Q4_2024_eps",
        question="What was Apple's diluted earnings per share in Q4 fiscal 2024?",
        ground_truth=(
            "Apple reported diluted earnings per share of $1.64 for the fourth quarter "
            "of fiscal year 2024."
        ),
        ticker="AAPL",
        year=2024,
        quarter="Q4",
    ),
    # ── NVIDIA ─────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="NVDA_Q3_FY2025_datacenter",
        question="What was NVIDIA's data center revenue in Q3 fiscal year 2025?",
        ground_truth=(
            "NVIDIA reported data center revenue of $30.8 billion for Q3 fiscal year 2025, "
            "representing year-over-year growth of 112%."
        ),
        ticker="NVDA",
        year=2024,
        quarter="Q3",
    ),
    EvalSample(
        sample_id="NVDA_Q3_FY2025_total",
        question="What was NVIDIA's total revenue for Q3 fiscal year 2025?",
        ground_truth=(
            "NVIDIA reported total revenue of $35.1 billion for Q3 fiscal year 2025, "
            "up 94% year-over-year."
        ),
        ticker="NVDA",
        year=2024,
        quarter="Q3",
    ),
    # ── Microsoft ──────────────────────────────────────────────────────────
    EvalSample(
        sample_id="MSFT_Q1_FY2025_cloud",
        question="What was Microsoft's Intelligent Cloud segment revenue in Q1 fiscal 2025?",
        ground_truth=(
            "Microsoft's Intelligent Cloud segment reported revenue of approximately "
            "$24.1 billion in Q1 fiscal year 2025, driven by Azure and other cloud services."
        ),
        ticker="MSFT",
        year=2024,
        quarter="Q1",
    ),
    EvalSample(
        sample_id="MSFT_Q1_FY2025_total",
        question="What was Microsoft's total revenue in Q1 fiscal year 2025?",
        ground_truth=(
            "Microsoft reported total revenue of $65.6 billion for Q1 fiscal year 2025, "
            "representing 16% growth year-over-year."
        ),
        ticker="MSFT",
        year=2024,
        quarter="Q1",
    ),
    # ── Amazon ─────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="AMZN_Q3_2024_aws",
        question="What was Amazon Web Services revenue in Q3 2024?",
        ground_truth=(
            "Amazon Web Services reported revenue of $27.5 billion in Q3 2024, "
            "growing 19% year-over-year."
        ),
        ticker="AMZN",
        year=2024,
        quarter="Q3",
    ),
    # ── Meta ───────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="META_Q3_2024_ad_revenue",
        question="What was Meta's advertising revenue in Q3 2024?",
        ground_truth=(
            "Meta reported advertising revenue of approximately $40 billion in Q3 2024, "
            "representing a 19% year-over-year increase."
        ),
        ticker="META",
        year=2024,
        quarter="Q3",
    ),
    EvalSample(
        sample_id="META_Q3_2024_dau",
        question="How many daily active people did Meta report for Q3 2024?",
        ground_truth=(
            "Meta reported 3.29 billion daily active people across its family of apps "
            "in Q3 2024, a 5% year-over-year increase."
        ),
        ticker="META",
        year=2024,
        quarter="Q3",
    ),
    # ── JPMorgan ───────────────────────────────────────────────────────────
    EvalSample(
        sample_id="JPM_Q3_2024_nii",
        question="What was JPMorgan Chase's net interest income in Q3 2024?",
        ground_truth=(
            "JPMorgan Chase reported net interest income of approximately $23.5 billion in Q3 2024."
        ),
        ticker="JPM",
        year=2024,
        quarter="Q3",
    ),
    # ── Tesla ──────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="TSLA_Q3_2024_revenue",
        question="What was Tesla's total revenue in Q3 2024?",
        ground_truth=(
            "Tesla reported total revenue of $25.2 billion in Q3 2024, "
            "an 8% increase year-over-year."
        ),
        ticker="TSLA",
        year=2024,
        quarter="Q3",
    ),
    EvalSample(
        sample_id="TSLA_Q3_2024_deliveries",
        question="How many vehicles did Tesla deliver in Q3 2024?",
        ground_truth=(
            "Tesla delivered approximately 463,000 vehicles in Q3 2024, "
            "representing a 6% year-over-year increase."
        ),
        ticker="TSLA",
        year=2024,
        quarter="Q3",
    ),
    # ── Walmart ────────────────────────────────────────────────────────────
    EvalSample(
        sample_id="WMT_Q3_FY2025_revenue",
        question="What was Walmart's total net sales in Q3 fiscal year 2025?",
        ground_truth=(
            "Walmart reported total net sales of approximately $169.6 billion "
            "in Q3 fiscal year 2025, growing 5% year-over-year."
        ),
        ticker="WMT",
        year=2024,
        quarter="Q3",
    ),
    # ── ExxonMobil ─────────────────────────────────────────────────────────
    EvalSample(
        sample_id="XOM_Q3_2024_earnings",
        question="What were ExxonMobil's earnings in Q3 2024?",
        ground_truth=("ExxonMobil reported earnings of approximately $8.6 billion in Q3 2024."),
        ticker="XOM",
        year=2024,
        quarter="Q3",
    ),
    # ── Adversarial: out-of-scope company (should return ungrounded) ───────
    EvalSample(
        sample_id="OOS_berkshire",
        question="What was Berkshire Hathaway's Q3 2024 operating earnings?",
        ground_truth=(
            "Berkshire Hathaway is not in the knowledge base. "
            "The system should indicate it cannot answer from the available documents."
        ),
        ticker=None,
        year=None,
        quarter=None,
    ),
]


def get_dataset_by_ticker(ticker: str) -> list[EvalSample]:
    """Return all samples for a specific ticker."""
    return [s for s in GOLDEN_DATASET if s.ticker == ticker]


def get_dataset_subset(n: int) -> list[EvalSample]:
    """Return the first n samples — useful for quick smoke-test runs."""
    return GOLDEN_DATASET[:n]
