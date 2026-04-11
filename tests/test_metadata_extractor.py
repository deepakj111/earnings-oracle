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
    _detect_quarter,
    extract_metadata,
)

TICKER = "AAPL"
DATE = "2024-10-31"

EARNINGS_Q1 = "This press release covers first quarter fiscal 2024 results."
EARNINGS_Q2 = "Second quarter revenue reached 94 billion dollars."
EARNINGS_Q3 = "Q3 2024 marked record services revenue."
EARNINGS_Q4 = "Fourth quarter and full year results are reported herein."
PLAIN_PROSE = "Revenue was strong. No quarter mentioned explicitly here."


class TestDetectQuarter:
    def test_first_quarter_text(self):
        assert _detect_quarter(EARNINGS_Q1, 1) == "Q1"

    def test_second_quarter_text(self):
        assert _detect_quarter(EARNINGS_Q2, 4) == "Q2"

    def test_q3_abbreviation(self):
        assert _detect_quarter(EARNINGS_Q3, 7) == "Q3"

    def test_fourth_quarter_text(self):
        assert _detect_quarter(EARNINGS_Q4, 10) == "Q4"

    def test_fallback_to_month_january(self):
        assert _detect_quarter(PLAIN_PROSE, 1) == "Q1"

    def test_fallback_to_month_april(self):
        assert _detect_quarter(PLAIN_PROSE, 4) == "Q2"

    def test_fallback_to_month_august(self):
        assert _detect_quarter(PLAIN_PROSE, 8) == "Q3"

    def test_fallback_to_month_november(self):
        assert _detect_quarter(PLAIN_PROSE, 11) == "Q4"

    def test_case_insensitive(self):
        assert _detect_quarter("FIRST QUARTER results were strong.", 1) == "Q1"

    def test_only_first_3000_chars_scanned(self):
        # Quarter mention buried past 3000 chars should NOT be detected
        far_text = ("x " * 2000) + "first quarter results"
        result = _detect_quarter(far_text, 6)
        assert result == "Q2"  # falls back to month=6 → Q2


class TestExtractMetadata:
    def test_returns_document_metadata(self):
        result = extract_metadata(TICKER, DATE, EARNINGS_Q4)
        assert isinstance(result, DocumentMetadata)

    def test_ticker_preserved(self):
        result = extract_metadata(TICKER, DATE, EARNINGS_Q4)
        assert result.ticker == TICKER

    def test_date_preserved(self):
        result = extract_metadata(TICKER, DATE, EARNINGS_Q4)
        assert result.date == DATE

    def test_year_parsed_from_date(self):
        result = extract_metadata(TICKER, DATE, EARNINGS_Q4)
        assert result.year == 2024

    def test_quarter_detected_from_prose(self):
        result = extract_metadata(TICKER, DATE, EARNINGS_Q1)
        assert result.quarter == "Q1"

    def test_fiscal_period_format(self):
        result = extract_metadata(TICKER, "2024-10-31", EARNINGS_Q4)
        assert result.fiscal_period == f"{result.quarter} 2024"

    def test_known_ticker_maps_to_company(self):
        result = extract_metadata("NVDA", DATE, PLAIN_PROSE)
        assert result.company == "NVIDIA"

    def test_unknown_ticker_uses_ticker_as_company(self):
        result = extract_metadata("XYZ", DATE, PLAIN_PROSE)
        assert result.company == "XYZ"

    def test_all_10_tickers_in_company_map(self):
        expected = {"AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"}
        assert expected.issubset(set(COMPANY_MAP.keys()))

    def test_malformed_date_does_not_crash(self):
        result = extract_metadata(TICKER, "2024", PLAIN_PROSE)
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
    def test_company_map_spot_check(self, ticker, company):
        result = extract_metadata(ticker, DATE, PLAIN_PROSE)
        assert result.company == company
