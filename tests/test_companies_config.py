# tests/test_companies_config.py
"""
Unit and integration tests for external company configuration,
CompanyRegistry dynamic operations, API routes (/companies),
and download CLI argument handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config.companies import CompanyRegistry
from ingestion.download_filings import main as download_main


class TestCompaniesConfigFile:
    def test_companies_json_exists_and_valid(self) -> None:
        config_path = Path("config/companies.json")
        assert config_path.exists(), "config/companies.json must exist"

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 4

        # Check required keys in each company entry
        for item in data:
            assert "ticker" in item
            assert "name" in item
            assert "cik" in item
            assert "sector" in item
            assert "fiscal_year_end_month" in item
            assert 1 <= item["fiscal_year_end_month"] <= 12
            assert "aliases" in item
            assert isinstance(item["aliases"], list)


class TestCompanyRegistry:
    def setup_method(self) -> None:
        CompanyRegistry.initialize()

    def test_supported_tickers(self) -> None:
        tickers = CompanyRegistry.get_supported_tickers()
        for expected in ["NVDA", "WMT", "NFLX", "UNH"]:
            assert expected in tickers

    def test_default_portfolio_tickers(self) -> None:
        defaults = CompanyRegistry.get_default_portfolio_tickers()
        assert defaults == ["NFLX", "NVDA", "UNH", "WMT"]

    def test_alias_map_resolution(self) -> None:
        alias_map = CompanyRegistry.get_alias_map()

        # Primary tickers and lowercase
        assert alias_map.get("nvda") == "NVDA"
        assert alias_map.get("wmt") == "WMT"
        assert alias_map.get("unh") == "UNH"

        # Brand names and subsidiaries
        assert alias_map.get("optum") == "UNH"
        assert alias_map.get("unitedhealthcare") == "UNH"
        assert alias_map.get("geforce") == "NVDA"
        assert alias_map.get("aws") == "AMZN"
        assert alias_map.get("instagram") == "META"

    def test_runtime_company_registration(self) -> None:
        CompanyRegistry.register_company(
            ticker="SNOW",
            name="Snowflake Inc.",
            cik="0001640147",
            sector="Technology / Cloud Data",
            fiscal_year_end_month=1,
            aliases=["snowflake", "snowpark"],
        )
        try:
            prof = CompanyRegistry.get_company("SNOW")
            assert prof is not None
            assert prof.name == "Snowflake Inc."
            assert prof.fiscal_year_end_month == 1

            alias_map = CompanyRegistry.get_alias_map()
            assert alias_map.get("snowpark") == "SNOW"
            assert alias_map.get("snowflake") == "SNOW"
        finally:
            # Revert to standard config
            CompanyRegistry.initialize()

    def test_generic_fiscal_period_derivation(self) -> None:
        # Dec fiscal year end (NFLX)
        fy_year, qtr, period = CompanyRegistry.derive_fiscal_period("NFLX", "10-K", "2025-01-24")
        assert fy_year == 2024
        assert qtr == "FY"
        assert period == "FY 2024"

        # Jan fiscal year end (NVDA)
        fy_year, qtr, period = CompanyRegistry.derive_fiscal_period("NVDA", "10-K", "2025-02-26")
        assert fy_year == 2025
        assert qtr == "FY"
        assert period == "FY 2025"

        # Q1 for Jan fiscal year end (filing in May)
        fy_year, qtr, period = CompanyRegistry.derive_fiscal_period("NVDA", "10-Q", "2024-05-29")
        assert qtr == "Q1"
        assert fy_year == 2025


class TestCompaniesApiRoutes:
    @pytest.fixture
    def client(self) -> TestClient:
        app = create_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_list_companies(self, client: TestClient) -> None:
        resp = client.get("/companies")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        tickers = [c["ticker"] for c in data]
        assert "NVDA" in tickers
        assert "UNH" in tickers
        assert "WMT" in tickers
        assert "NFLX" in tickers

    def test_get_company_success(self, client: TestClient) -> None:
        resp = client.get("/companies/NVDA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "NVDA"
        assert data["name"] == "NVIDIA"
        assert data["fiscal_year_end_month"] == 1
        assert "geforce" in data["aliases"]

    def test_get_company_not_found(self, client: TestClient) -> None:
        resp = client.get("/companies/NONEXISTENT999")
        assert resp.status_code == 404
        assert "not configured" in resp.json()["detail"].lower()


class TestDownloadFilingsCli:
    @patch("ingestion.download_filings.get_company_filings", return_value=[])
    def test_download_tickers_flag(self, mock_fetch) -> None:
        download_main(["--tickers", "NVDA,WMT", "-o", "/tmp/test_filings"])
        # Verify get_company_filings was called for NVDA and WMT
        calls = mock_fetch.call_args_list
        called_tickers = {c.kwargs.get("ticker") for c in calls}
        assert called_tickers == {"NVDA", "WMT"}

    @patch("ingestion.download_filings.get_company_filings", return_value=[])
    def test_download_default_portfolio(self, mock_fetch) -> None:
        download_main(["-o", "/tmp/test_filings"])
        calls = mock_fetch.call_args_list
        called_tickers = {c.kwargs.get("ticker") for c in calls}
        default_portfolio = set(CompanyRegistry.get_default_portfolio_tickers())
        assert called_tickers == default_portfolio
