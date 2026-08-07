"""
Tests for ingestion/metadata_extractor.py

Pure functions — no I/O, no mocks needed.
Coverage:
  - Quarter detection from prose text
  - Quarter fallback from filing month
  - Year/month parsing from date string
  - fiscal_period string format
  - Unknown ticker fallback
  - COMPANY_MAP correctness
"""

import pytest

from ingestion.metadata_extractor import (
    COMPANY_MAP,
    DocumentMetadata,
    _derive_fiscal_period,
    extract_metadata,
)

TICKER = "AAPL"
DATE = "2024-10-31"


class TestDeriveFiscalPeriod:
    def test_10k_filing_january(self) -> None:
        period, year, qtr = _derive_fiscal_period("10-K", "2025-01-27")
        assert period == "FY 2024"
        assert year == 2024
        assert qtr == "FY"

    def test_10q_filing_april(self) -> None:
        period, year, qtr = _derive_fiscal_period("10-Q", "2026-04-17")
        assert period == "Q1 2026"
        assert year == 2026
        assert qtr == "Q1"

    def test_10q_filing_july(self) -> None:
        period, year, qtr = _derive_fiscal_period("10-Q", "2026-07-17")
        assert period == "Q2 2026"
        assert year == 2026
        assert qtr == "Q2"


PROSE = "Sample filing text."


class TestExtractMetadata:
    def test_returns_document_metadata(self) -> None:
        result = extract_metadata(TICKER, DATE, PROSE, form_type="10-K")
        assert isinstance(result, DocumentMetadata)

    def test_ticker_preserved(self) -> None:
        result = extract_metadata(TICKER, DATE, PROSE)
        assert result.ticker == TICKER

    def test_date_preserved(self) -> None:
        result = extract_metadata(TICKER, DATE, PROSE)
        assert result.date == DATE

    def test_year_parsed_from_date(self) -> None:
        result = extract_metadata(TICKER, "2024-10-31", PROSE, form_type="10-Q")
        assert result.year == 2024

    def test_fiscal_period_format(self) -> None:
        result = extract_metadata(TICKER, "2024-10-31", PROSE, form_type="10-Q")
        assert result.fiscal_period == f"{result.quarter} {result.year}"

    def test_known_ticker_maps_to_company(self) -> None:
        result = extract_metadata("NVDA", DATE, PROSE)
        assert result.company == "NVIDIA"

    def test_unknown_ticker_uses_ticker_as_company(self) -> None:
        result = extract_metadata("XYZ", DATE, PROSE)
        assert result.company == "XYZ"

    def test_all_10_tickers_in_company_map(self) -> None:
        expected = {"AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"}
        assert expected.issubset(set(COMPANY_MAP.keys()))

    def test_malformed_date_does_not_crash(self) -> None:
        result = extract_metadata(TICKER, "2024", PROSE)
        assert result.year == 2024

    @pytest.mark.parametrize(
        "ticker,company",
        [
            ("AAPL", "Apple"),
            ("NVDA", "NVIDIA"),
            ("MSFT", "Microsoft"),
            ("TSLA", "Tesla"),
            ("WMT", "Walmart"),
        ],
    )
    def test_company_map_spot_check(self, ticker, company) -> None:
        result = extract_metadata(ticker, DATE, PROSE)
        assert result.company == company
