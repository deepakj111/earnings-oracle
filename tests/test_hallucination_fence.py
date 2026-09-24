"""
Unit tests for NumericalHallucinationFence.
"""

from generation.hallucination_fence import (
    NumericalHallucinationFence,
)
from retrieval.models import SearchResult


def _make_dummy_result(text: str) -> SearchResult:
    return SearchResult(
        chunk_id="test_1",
        parent_id="p_1",
        text=text,
        parent_text=text,
        rrf_score=1.0,
        rerank_score=1.0,
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period="Q4 2024",
        section_title="Revenue",
        doc_type="10-Q",
        source="both",
    )


class TestNumericalHallucinationFence:
    def test_empty_answer_returns_empty_report(self) -> None:
        report = NumericalHallucinationFence.verify("", [])
        assert report.total_extracted == 0
        assert report.has_hallucinations is False
        assert report.precision == 1.0

    def test_extract_numbers_ignores_bare_years(self) -> None:
        text = "In 2024, revenue grew 12% to $94.9B compared to 2023."
        numbers = NumericalHallucinationFence.extract_numbers(text)
        assert "12%" in numbers
        assert "$94.9B" in numbers
        assert "2024" not in numbers
        assert "2023" not in numbers

    def test_grounded_numbers_pass(self) -> None:
        source = _make_dummy_result("Apple reported revenue of $94.9B, up 6% year-over-year.")
        answer = "Apple had revenue of $94.9B with 6% growth [1]."
        report = NumericalHallucinationFence.verify(answer, [source])

        assert report.has_hallucinations is False
        assert len(report.flagged_numbers) == 0
        assert len(report.verified_numbers) >= 2

    def test_ungrounded_number_flagged(self) -> None:
        source = _make_dummy_result("Apple reported revenue of $94.9B.")
        # $105.2B is completely made up
        answer = "Apple reported revenue of $105.2B in the quarter [1]."
        report = NumericalHallucinationFence.verify(answer, [source])

        assert report.has_hallucinations is True
        assert "$105.2B" in report.flagged_numbers
        assert report.precision == 0.0

    def test_verified_pal_math_accepted(self) -> None:
        source = _make_dummy_result("Old revenue was $100M and new revenue was $120M.")
        # Model computed 20% growth via PAL
        answer = "Revenue increased by 20.0% from $100M to $120M [1]."
        report = NumericalHallucinationFence.verify(
            answer=answer,
            retrieved_chunks=[source],
            verified_calculations=[20.0],
        )

        assert report.has_hallucinations is False
        assert len(report.flagged_numbers) == 0

    def test_negative_and_accounting_parentheses_grounded(self) -> None:
        source = _make_dummy_result(
            "Operating loss was ($540) million compared to net loss of -$1.2B in the prior period."
        )
        answer = "The company recorded an operating loss of ($540) million and a prior net loss of -$1.2B [1]."
        report = NumericalHallucinationFence.verify(
            answer=answer,
            retrieved_chunks=[source],
        )
        assert report.has_hallucinations is False
        assert len(report.flagged_numbers) == 0
        assert len(report.verified_numbers) >= 2

    def test_extract_numbers_captures_negatives_and_parentheses(self) -> None:
        text = "Net loss was -$450M, operating margin was (5.2)%, and EBITDA was ($1,234) million."
        numbers = NumericalHallucinationFence.extract_numbers(text)
        assert any("-450M" in n or "-$450M" in n for n in numbers)
        assert any("(5.2)%" in n or "5.2" in n for n in numbers)
        assert any("1,234" in n for n in numbers)

    def test_basis_points_extraction_and_grounding(self) -> None:
        source = _make_dummy_result("Gross margin expanded by 1.5% compared to the prior year.")
        answer = "The company reported gross margin expansion of 150 bps [1]."
        report = NumericalHallucinationFence.verify(
            answer=answer,
            retrieved_chunks=[source],
        )
        assert report.has_hallucinations is False
        assert len(report.flagged_numbers) == 0
        assert "150 bps" in report.verified_numbers

    def test_basis_points_matched_via_pal_math(self) -> None:
        source = _make_dummy_result("Operating margin was 12.0% in 2024 and 13.5% in 2025.")
        answer = "Operating margin spread improved by 150 bps [1]."
        report = NumericalHallucinationFence.verify(
            answer=answer,
            retrieved_chunks=[source],
            verified_calculations=[150.0],
        )
        assert report.has_hallucinations is False
        assert len(report.flagged_numbers) == 0
        assert "150 bps" in report.verified_numbers
