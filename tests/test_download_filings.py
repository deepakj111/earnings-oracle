"""
Tests for ingestion/download_filings.py

All HTTP calls are mocked — we test parsing/picking logic only.
Coverage:
  - get_8k_filings filters by form type and date range
  - get_filing_documents parses HTML index table correctly
  - pick_best_document prefers EX-99.1 over 8-K body
  - pick_best_document falls back to 8-K body when no exhibit found
  - pick_best_document returns None when no document available
  - download_document writes file and returns path on success
  - download_document returns None on HTTP failure
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from ingestion.download_filings import (
    get_8k_filings,
    get_filing_documents,
    pick_best_document,
    download_document,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

# test_download_filings.py — fix the fixture at the top of the file

SEC_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form":          ["8-K",         "10-Q",        "8-K",         "8-K"],
            "filingDate":    ["2024-10-31",   "2024-08-01",  "2022-12-01",  "2026-01-01"],
            #                                                                ^^^^^^^^^^
            #                                                  was "2025-06-01" — inside range
            "accessionNumber": [
                "0001234567-24-000001",
                "0001234567-24-000002",
                "0001234567-22-000003",
                "0001234567-26-000004",
            ],
        }
    }
}

FILING_INDEX_HTML = """
<html><body>
<table>
  <tr>
    <td>1</td><td>Earnings Press Release</td>
    <td><a href="ex99_1.htm">ex99_1.htm</a></td>
    <td>EX-99.1</td><td>120KB</td>
  </tr>
  <tr>
    <td>2</td><td>Form 8-K</td>
    <td><a href="form8k.htm">form8k.htm</a></td>
    <td>8-K</td><td>20KB</td>
  </tr>
</table>
</body></html>
"""

FILING_INDEX_NO_EXHIBIT = """
<html><body>
<table>
  <tr>
    <td>1</td><td>Form 8-K</td>
    <td><a href="form8k.htm">form8k.htm</a></td>
    <td>8-K</td><td>20KB</td>
  </tr>
</table>
</body></html>
"""


def _mock_response(json_data=None, text="", status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ── get_8k_filings ────────────────────────────────────────────────────────────

class TestGet8kFilings:
    def _run(self, start="2023-01-01", end="2025-12-31"):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(json_data=SEC_SUBMISSIONS_JSON)):
            return get_8k_filings("0000320193", "AAPL", start, end)

    def test_returns_only_8k_forms(self):
        results = self._run()
        assert all(r.get("ticker") == "AAPL" for r in results)

    def test_filters_out_10q(self):
        results = self._run()
        # 10-Q at index 1 must not appear
        assert len(results) == 1

    def test_filters_by_date_range(self):
        results = self._run(start="2023-01-01", end="2025-12-31")
        # 2022-12-01 and 2025-06-01 both outside range
        assert len(results) == 1
        assert results[0]["date"] == "2024-10-31"

    def test_result_contains_required_keys(self):
        results = self._run()
        for r in results:
            assert "ticker" in r
            assert "cik" in r
            assert "date" in r
            assert "accession" in r

    def test_ticker_attached_to_result(self):
        results = self._run()
        assert results[0]["ticker"] == "AAPL"

    def test_empty_range_returns_empty(self):
        results = self._run(start="2020-01-01", end="2020-12-31")
        assert results == []


# ── get_filing_documents ──────────────────────────────────────────────────────

class TestGetFilingDocuments:
    def _run(self, html=FILING_INDEX_HTML, status=200):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(text=html, status_code=status)):
            return get_filing_documents("0000320193", "0001234567-24-000001")

    def test_returns_list(self):
        result = self._run()
        assert isinstance(result, list)

    def test_parses_exhibit_type(self):
        result = self._run()
        types = [d["type"] for d in result]
        assert "EX-99.1" in types

    def test_parses_8k_form_type(self):
        result = self._run()
        types = [d["type"] for d in result]
        assert "8-K" in types

    def test_parses_document_name(self):
        result = self._run()
        names = [d["name"] for d in result]
        assert "ex99_1.htm" in names

    def test_failed_index_returns_empty(self):
        result = self._run(status=404)
        assert result == []


# ── pick_best_document ────────────────────────────────────────────────────────

class TestPickBestDocument:
    def test_prefers_ex99_1_over_8k(self):
        docs = [
            {"name": "form8k.htm",  "type": "8-K",     "description": "form 8-k"},
            {"name": "ex99_1.htm",  "type": "EX-99.1", "description": "press release"},
        ]
        assert pick_best_document(docs) == "ex99_1.htm"

    def test_fallback_to_8k_body(self):
        docs = [
            {"name": "form8k.htm", "type": "8-K", "description": "form 8-k"},
        ]
        assert pick_best_document(docs) == "form8k.htm"

    def test_returns_none_when_no_document(self):
        docs = [
            {"name": "cover.htm", "type": "COVER", "description": "cover page"},
        ]
        assert pick_best_document(docs) is None

    def test_empty_list_returns_none(self):
        assert pick_best_document([]) is None

    def test_ex99_2_also_accepted(self):
        docs = [
            {"name": "ex99_2.htm", "type": "EX-99.2", "description": "tables"},
        ]
        assert pick_best_document(docs) == "ex99_2.htm"

    def test_description_keyword_match(self):
        docs = [
            {"name": "results.htm", "type": "OTHER",
             "description": "earnings press release"},
        ]
        assert pick_best_document(docs) == "results.htm"


# ── download_document ─────────────────────────────────────────────────────────

class TestDownloadDocument:
    def _filing_meta(self):
        return {"ticker": "AAPL", "date": "2024-10-31"}

    def test_returns_path_on_success(self, tmp_path):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(text="<html>content</html>")):
            result = download_document(
                "0000320193", "0001234567-24-000001",
                "ex99_1.htm", self._filing_meta(), str(tmp_path)
            )
        assert result is not None
        assert result.endswith(".htm")

    def test_file_written_to_disk(self, tmp_path):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(text="<html>earnings</html>")):
            download_document(
                "0000320193", "0001234567-24-000001",
                "ex99_1.htm", self._filing_meta(), str(tmp_path)
            )
        files = list(tmp_path.glob("*.htm"))
        assert len(files) == 1

    def test_returns_none_on_http_error(self, tmp_path):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(status_code=403)):
            result = download_document(
                "0000320193", "0001234567-24-000001",
                "ex99_1.htm", self._filing_meta(), str(tmp_path)
            )
        assert result is None

    def test_filename_contains_ticker_and_date(self, tmp_path):
        with patch("ingestion.download_filings.requests.get",
                   return_value=_mock_response(text="<html>content</html>")):
            result = download_document(
                "0000320193", "0001234567-24-000001",
                "ex99_1.htm", self._filing_meta(), str(tmp_path)
            )
        assert "AAPL" in result
        assert "2024-10-31" in result