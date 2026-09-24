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

import json
from dataclasses import dataclass, field
from pathlib import Path
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
    default_portfolio: bool = False
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker.upper(),
            "name": self.name,
            "cik": self.cik,
            "sector": self.sector,
            "fiscal_year_end_month": self.fiscal_year_end_month,
            "download_start_date": self.download_start_date,
            "default_portfolio": self.default_portfolio,
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompanyProfile:
        return cls(
            ticker=data["ticker"].upper().strip(),
            name=data["name"].strip(),
            cik=data.get("cik", "").strip(),
            sector=data.get("sector", "General").strip(),
            fiscal_year_end_month=int(data.get("fiscal_year_end_month", 12)),
            download_start_date=data.get("download_start_date", "2024-01-01").strip(),
            default_portfolio=bool(data.get("default_portfolio", False)),
            aliases=[str(a).strip().lower() for a in data.get("aliases", []) if str(a).strip()],
        )


# Fallback company profiles in case external configuration is unavailable
_REGISTRY_COMPANIES: list[CompanyProfile] = [
    CompanyProfile(
        ticker="NVDA",
        name="NVIDIA",
        cik="0001045810",
        sector="Technology / Semiconductors",
        fiscal_year_end_month=1,
        download_start_date="2024-01-01",
        default_portfolio=True,
        aliases=["nvidia", "geforce", "mellanox", "nvd"],
    ),
    CompanyProfile(
        ticker="WMT",
        name="Walmart",
        cik="0000104169",
        sector="Consumer Staples / Retail",
        fiscal_year_end_month=1,
        download_start_date="2024-01-01",
        default_portfolio=True,
        aliases=["walmart", "sam's club", "sams club", "wal-mart"],
    ),
    CompanyProfile(
        ticker="NFLX",
        name="Netflix",
        cik="0001065280",
        sector="Communication Services / Streaming",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=True,
        aliases=["netflix"],
    ),
    CompanyProfile(
        ticker="UNH",
        name="UnitedHealth Group",
        cik="0000731766",
        sector="Healthcare / Managed Care",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=True,
        aliases=[
            "optum",
            "united health",
            "unitedhealthcare",
            "united health group",
            "change healthcare",
        ],
    ),
    CompanyProfile(
        ticker="AAPL",
        name="Apple",
        cik="0000320193",
        sector="Technology / Consumer Electronics",
        fiscal_year_end_month=9,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["apple", "iphone", "ipad", "macbook"],
    ),
    CompanyProfile(
        ticker="MSFT",
        name="Microsoft",
        cik="0000789019",
        sector="Technology / Software",
        fiscal_year_end_month=6,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["microsoft", "azure", "windows", "xbox"],
    ),
    CompanyProfile(
        ticker="AMZN",
        name="Amazon",
        cik="0001018724",
        sector="Consumer Discretionary / E-Commerce",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["amazon", "aws", "prime video"],
    ),
    CompanyProfile(
        ticker="META",
        name="Meta Platforms",
        cik="0001326801",
        sector="Communication Services / Interactive Media",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["meta", "facebook", "instagram", "whatsapp", "oculus"],
    ),
    CompanyProfile(
        ticker="JPM",
        name="JPMorgan Chase",
        cik="0000019617",
        sector="Financials / Banking",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["jpmorgan", "jp morgan", "chase"],
    ),
    CompanyProfile(
        ticker="TSLA",
        name="Tesla",
        cik="0001318605",
        sector="Consumer Discretionary / Automotive",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["tesla"],
    ),
    CompanyProfile(
        ticker="XOM",
        name="ExxonMobil",
        cik="0000034088",
        sector="Energy / Oil & Gas",
        fiscal_year_end_month=12,
        download_start_date="2024-01-01",
        default_portfolio=False,
        aliases=["exxon", "exxonmobil", "mobil"],
    ),
]


class CompanyRegistry:
    """
    Singleton registry managing company profiles and generic fiscal calendar logic.
    Loads dynamically from config/companies.json or registered at runtime.
    """

    CONFIG_PATH: Path = Path(__file__).parent / "companies.json"
    _profiles: dict[str, CompanyProfile] = {}

    @classmethod
    def initialize(cls, config_path: Path | str | None = None) -> None:
        """Initialize registry from companies.json, falling back to internal defaults."""
        cls._profiles.clear()
        target_path = Path(config_path) if config_path else cls.CONFIG_PATH

        if target_path.exists():
            try:
                content = target_path.read_text(encoding="utf-8")
                raw_list = json.loads(content)
                if isinstance(raw_list, list):
                    for item in raw_list:
                        if isinstance(item, dict) and "ticker" in item and "name" in item:
                            prof = CompanyProfile.from_dict(item)
                            cls._profiles[prof.ticker] = prof
                    if cls._profiles:
                        logger.debug(
                            f"CompanyRegistry initialized with {len(cls._profiles)} companies from {target_path}"
                        )
                        return
            except Exception as exc:
                logger.warning(
                    f"Failed to load company registry from {target_path} ({exc}). Using internal defaults."
                )

        # Fallback to internal profiles
        for p in _REGISTRY_COMPANIES:
            cls._profiles[p.ticker.upper()] = p

    @classmethod
    def load_from_json(cls, config_path: Path | str | None = None) -> None:
        """Explicitly reload registry from a JSON configuration file."""
        cls.initialize(config_path)

    @classmethod
    def register_company(
        cls,
        ticker: str,
        name: str,
        cik: str = "",
        sector: str = "General",
        fiscal_year_end_month: int = 12,
        download_start_date: str = "2024-01-01",
        default_portfolio: bool = False,
        aliases: list[str] | None = None,
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
            default_portfolio=default_portfolio,
            aliases=[str(a).strip().lower() for a in (aliases or []) if str(a).strip()],
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
    def get_default_portfolio_tickers(cls) -> list[str]:
        """Return list of tickers designated as the default portfolio."""
        if not cls._profiles:
            cls.initialize()
        return sorted([t for t, p in cls._profiles.items() if p.default_portfolio])

    @classmethod
    def get_all_companies(cls) -> list[CompanyProfile]:
        """Return all registered CompanyProfile instances."""
        if not cls._profiles:
            cls.initialize()
        return list(cls._profiles.values())

    @classmethod
    def get_alias_map(cls) -> dict[str, str]:
        """
        Dynamically build mapping of lowercased entity names/aliases to uppercase tickers.
        Includes ticker, company name, filtered distinctive words, and all custom aliases.
        """
        if not cls._profiles:
            cls.initialize()

        mapping: dict[str, str] = {}
        for ticker, prof in cls._profiles.items():
            t_upper = ticker.upper()
            mapping[t_upper.lower()] = t_upper
            mapping[prof.name.lower()] = t_upper

            # Common prefixes/short words (e.g., 'netflix', 'walmart')
            for word in prof.name.lower().split():
                if len(word) >= 4 and word not in (
                    "corporation",
                    "company",
                    "group",
                    "inc.",
                    "holdings",
                    "platforms",
                    "services",
                ):
                    mapping[word] = t_upper

            # Distinctive configured aliases
            for alias in prof.aliases:
                clean_alias = alias.lower().strip()
                if clean_alias:
                    mapping[clean_alias] = t_upper

        return mapping

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
