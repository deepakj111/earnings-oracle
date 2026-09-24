"""
Unit tests for CitationIntegrityValidator.
"""

from generation.citation_validator import (
    CitationIntegrityValidator,
)
from generation.models import Citation


def _make_citation(idx: int, ticker: str, period: str = "Q4 2024") -> Citation:
    return Citation(
        index=idx,
        chunk_id=f"c_{idx}",
        parent_id=f"p_{idx}",
        ticker=ticker,
        company="Test Corp",
        date="2024-10-31",
        fiscal_period=period,
        section_title="Financials",
        doc_type="10-Q",
        source="dense",
        rerank_score=0.95,
        excerpt="excerpt text",
    )


class TestCitationIntegrityValidator:
    def test_empty_answer_is_valid(self) -> None:
        report = CitationIntegrityValidator.validate("", [])
        assert report.is_valid is True
        assert report.total_citations == 0

    def test_matching_ticker_and_citation_passes(self) -> None:
        cit = _make_citation(1, "AAPL")
        answer = "Apple reported strong hardware sales in the holiday quarter [1]."
        report = CitationIntegrityValidator.validate(answer, [cit])
        assert report.is_valid is True
        assert len(report.warnings) == 0

    def test_mismatched_ticker_generates_warning(self) -> None:
        cit_aapl = _make_citation(1, "AAPL")
        # Sentence is about Microsoft, but citing Apple chunk
        answer = "Microsoft reported intelligent cloud revenue growth of 29% [1]."
        report = CitationIntegrityValidator.validate(answer, [cit_aapl])
        assert report.is_valid is False
        assert len(report.warnings) == 1
        assert "AAPL" in report.warnings[0]
        assert "MSFT" in report.warnings[0]

    def test_multi_company_comparative_citations(self) -> None:
        cit1 = _make_citation(1, "AAPL")
        cit2 = _make_citation(2, "NVDA")
        answer = (
            "Apple posted services revenue of $24.9B [1]. "
            "Meanwhile, NVIDIA grew data center revenue by 112% [2]."
        )
        report = CitationIntegrityValidator.validate(answer, [cit1, cit2])
        assert report.is_valid is True
        assert len(report.warnings) == 0

    def test_optum_unh_alias_matching(self) -> None:
        cit_unh = _make_citation(1, "UNH")
        answer = "Optum revenue grew by 14% year-over-year driven by Optum Health expansion [1]."
        report = CitationIntegrityValidator.validate(answer, [cit_unh])
        assert report.is_valid is True
        assert len(report.warnings) == 0
