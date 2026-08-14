# config/companies.py
"""
Centralized Production-Grade Company & Fiscal Calendar Registry.

Extensible for any US-listed public company. Configures:
- Company metadata (Ticker, Name, CIK, Sector, Download Start Date)
- Fiscal Year End Month (1 = January, 12 = December, 9 = September, 6 = June, etc.)
- Dynamic Fiscal Period Derivation (Generic algorithm handling any corporate fiscal calendar)

Allows zero hardcoding of ticker names or filing dates in business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    cik: str = ""
    sector: str = "General"
    fiscal_year_end_month: int = (
        12  # 1 = January (WMT/NVDA), 12 = December (NFLX/UNH), 9 = Sept (AAPL), 6 = June (MSFT)
    )
    download_start_date: str = "2024-01-01"  # SEC EDGAR filing download start date

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker.upper(),
            "name": self.name,
            "cik": self.cik,
            "sector": self.sector,
            "fiscal_year_end_month": self.fiscal_year_end_month,
            "download_start_date": self.download_start_date,
        }


# Single source of truth for company metadata and fiscal configurations
_REGISTRY_COMPANIES: list[CompanyProfile] = [
    CompanyProfile(
        ticker="NVDA",
        name="NVIDIA",
        cik="0001045810",
        sector="Technology / Semiconductors",
        fiscal_year_end_month=1,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="WMT",
        name="Walmart",
        cik="0000104169",
        sector="Consumer Staples / Retail",
        fiscal_year_end_month=1,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="NFLX",
        name="Netflix",
        cik="0001065280",
        sector="Communication Services / Streaming",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="UNH",
        name="UnitedHealth Group",
        cik="0000731766",
        sector="Healthcare / Managed Care",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="AAPL",
        name="Apple",
        cik="0000320193",
        sector="Technology / Consumer Electronics",
        fiscal_year_end_month=9,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="MSFT",
        name="Microsoft",
        cik="0000789019",
        sector="Technology / Software",
        fiscal_year_end_month=6,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="AMZN",
        name="Amazon",
        cik="0001018724",
        sector="Consumer Discretionary / E-Commerce",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="META",
        name="Meta Platforms",
        cik="0001326801",
        sector="Communication Services / Interactive Media",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="JPM",
        name="JPMorgan Chase",
        cik="0000019617",
        sector="Financials / Banking",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="TSLA",
        name="Tesla",
        cik="0001318605",
        sector="Consumer Discretionary / Automotive",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
    CompanyProfile(
        ticker="XOM",
        name="ExxonMobil",
        cik="0000034088",
        sector="Energy / Oil & Gas",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
    ),
]


class CompanyRegistry:
    """
    Singleton registry managing company profiles and generic fiscal calendar logic.
    """

    _profiles: dict[str, CompanyProfile] = {}

    @classmethod
    def initialize(cls) -> None:
        """Initialize registry from internal company profiles."""
        cls._profiles.clear()
        for p in _REGISTRY_COMPANIES:
            cls._profiles[p.ticker.upper()] = p

    @classmethod
    def register_company(
        cls,
        ticker: str,
        name: str,
        cik: str = "",
        sector: str = "General",
        fiscal_year_end_month: int = 12,
        download_start_date: str = "2024-01-01",
    ) -> None:
        """Dynamically register a new company profile at runtime."""
        t_upper = ticker.upper().strip()
        prof = CompanyProfile(
            ticker=t_upper,
            name=name.strip(),
            cik=cik.strip(),
            sector=sector.strip(),
            fiscal_year_end_month=fiscal_year_end_month,
            download_start_date=download_start_date,
        )
        cls._profiles[t_upper] = prof
        logger.info(f"Registered company profile for {t_upper} ({name})")

    @classmethod
    def get_company(cls, ticker: str) -> CompanyProfile | None:
        if not cls._profiles:
            cls.initialize()
        return cls._profiles.get(ticker.upper().strip())

    @classmethod
    def get_company_map(cls) -> dict[str, str]:
        """Return dict mapping ticker -> company name."""
        if not cls._profiles:
            cls.initialize()
        return {ticker: prof.name for ticker, prof in cls._profiles.items()}

    @classmethod
    def get_supported_tickers(cls) -> list[str]:
        """Return list of supported tickers."""
        if not cls._profiles:
            cls.initialize()
        return sorted(cls._profiles.keys())

    @classmethod
    def derive_fiscal_period(
        cls,
        ticker: str,
        form_type: str,
        filing_date: str,
    ) -> tuple[int, str, str]:
        """
        Generic algorithm for deriving (fiscal_year, quarter_str, fiscal_period_str)
        for ANY company based on its configured fiscal year end month.

        No hardcoded ticker logic!
        """
        if not cls._profiles:
            cls.initialize()

        prof = cls.get_company(ticker)
        fy_end_month = prof.fiscal_year_end_month if prof else 12

        year = 2025
        month = 1

        if filing_date and "-" in filing_date:
            parts = filing_date.split("-")
            if len(parts) >= 1 and parts[0].isdigit():
                year = int(parts[0])
            if len(parts) >= 2 and parts[1].isdigit():
                month = int(parts[1])

        form_upper = form_type.upper().strip()

        # ── 10-K Filings (Annual) ──────────────────────────────────────────────
        if "10-K" in form_upper:
            if fy_end_month == 12:
                # Calendar year company (Dec 31 end): 10-K filed early Y covers FY Y-1
                fiscal_year = year - 1 if month <= 6 else year
            elif fy_end_month == 1:
                # Jan fiscal year end (WMT/NVDA): 10-K filed early Y covers FY Y
                fiscal_year = year if month <= 4 else year + 1
            else:
                # Generic fiscal year ends
                fiscal_year = year if month >= fy_end_month else year - 1

            quarter = "FY"
            fiscal_period = f"FY {fiscal_year}"
            return fiscal_year, quarter, fiscal_period

        # ── 10-Q Filings (Quarterly) ───────────────────────────────────────────
        if "10-Q" in form_upper:
            if fy_end_month == 12:
                # Calendar year company (Dec 31 end)
                fiscal_year = year
                if month in (4, 5, 6):
                    quarter = "Q1"
                elif month in (7, 8, 9):
                    quarter = "Q2"
                elif month in (10, 11, 12):
                    quarter = "Q3"
                else:
                    quarter = "Q4"
                    fiscal_year = year - 1
            elif fy_end_month == 1:
                # Jan fiscal year end (WMT/NVDA)
                if month in (4, 5, 6):
                    quarter = "Q1"
                    fiscal_year = year + 1
                elif month in (7, 8, 9):
                    quarter = "Q2"
                    fiscal_year = year + 1
                elif month in (10, 11, 12):
                    quarter = "Q3"
                    fiscal_year = year + 1
                else:
                    quarter = "Q4"
                    fiscal_year = year
            else:
                # Generic calculation relative to fiscal year start month
                fy_start_month = (fy_end_month % 12) + 1
                month_offset = (month - fy_start_month) % 12
                q_num = (month_offset // 3) + 1
                quarter = f"Q{q_num}"
                fiscal_year = year if month > fy_end_month else year + 1

            fiscal_period = f"{quarter} {fiscal_year}"
            return fiscal_year, quarter, fiscal_period

        # Fallback for non-standard forms
        quarter = "Q1"
        fiscal_year = year
        fiscal_period = f"{quarter} {fiscal_year}"
        return fiscal_year, quarter, fiscal_period


# Initialize on import
CompanyRegistry.initialize()
