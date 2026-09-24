"""
Fiscal Calendar Intelligence & Query Temporal Normalizer.

Bridges the gap between natural language temporal queries (e.g., 'calendar 2025',
'in 2024', 'FY26 Q3', 'last fiscal year') and exact SEC filing metadata (fiscal_year, quarter).
Handles off-calendar companies (NVDA, WMT Jan FY-end, AAPL Sep FY-end, MSFT Jun FY-end)
using the CompanyRegistry configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.companies import CompanyRegistry


@dataclass
class ResolvedPeriod:
    ticker: str | None
    fiscal_year: int | None
    quarter: str | None  # "Q1", "Q2", "Q3", "Q4", "FY", or None
    is_fiscal_explicit: bool = False
    is_calendar_explicit: bool = False
    raw_match: str = ""
    explanation: str = ""


class FiscalResolver:
    """
    Normalizes temporal query expressions into company-specific SEC filing periods.
    """

    # Matches explicit fiscal mentions: FY25, FY 2025, fiscal 2025, FY2026, 2025 fiscal year
    _FY_EXPLICIT_RE = re.compile(
        r"\b(?:FY|fiscal\s+year|fiscal)\s*['\"]?(20\d{2}|\d{2})\b|\b(20\d{2})\s+fiscal\b",
        re.IGNORECASE,
    )

    # Matches quarterly mentions: Q1, Q2, Q3, Q4, first quarter, fourth quarter, etc.
    _QUARTER_RE = re.compile(
        r"\b(Q[1-4]|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|1st\s+quarter|2nd\s+quarter|3rd\s+quarter|4th\s+quarter)\b",
        re.IGNORECASE,
    )

    # Matches calendar mentions: calendar 2025, CY2025, CY 25, or bare 4-digit year 2020-2030
    _CY_EXPLICIT_RE = re.compile(
        r"\b(?:CY|calendar\s+year|calendar)\s*['\"]?(20\d{2}|\d{2})\b",
        re.IGNORECASE,
    )
    _BARE_YEAR_RE = re.compile(r"\b(202[0-9]|2030)\b")

    _QUARTER_MAP = {
        "q1": "Q1",
        "first quarter": "Q1",
        "1st quarter": "Q1",
        "q2": "Q2",
        "second quarter": "Q2",
        "2nd quarter": "Q2",
        "q3": "Q3",
        "third quarter": "Q3",
        "3rd quarter": "Q3",
        "q4": "Q4",
        "fourth quarter": "Q4",
        "4th quarter": "Q4",
    }

    @classmethod
    def resolve(cls, query: str, ticker: str | None = None) -> ResolvedPeriod:
        """
        Analyze the query and resolve the intended fiscal year and quarter for the given ticker.
        """
        query_clean = query.strip()
        prof = CompanyRegistry.get_company(ticker) if ticker else None
        fy_end_month = prof.fiscal_year_end_month if prof else 12

        # 1. Parse Quarter
        quarter: str | None = None
        q_match = cls._QUARTER_RE.search(query_clean)
        if q_match:
            q_text = q_match.group(1).lower()
            quarter = cls._QUARTER_MAP.get(q_text, q_text.upper())

        # 2. Parse Explicit Fiscal Year: FY25, FY 2026, etc.
        fy_match = cls._FY_EXPLICIT_RE.search(query_clean)
        if fy_match:
            raw_yr = fy_match.group(1) or fy_match.group(2)
            yr = int(raw_yr) if len(raw_yr) == 4 else 2000 + int(raw_yr)
            return ResolvedPeriod(
                ticker=ticker,
                fiscal_year=yr,
                quarter=quarter,
                is_fiscal_explicit=True,
                raw_match=fy_match.group(0),
                explanation=f"Explicit fiscal year {yr} detected.",
            )

        # 3. Parse Explicit Calendar Year: CY2024, calendar 2025
        cy_match = cls._CY_EXPLICIT_RE.search(query_clean)
        if cy_match:
            raw_yr = cy_match.group(1)
            cy_year = int(raw_yr) if len(raw_yr) == 4 else 2000 + int(raw_yr)
            fiscal_year = cls._calendar_to_fiscal_year(cy_year, fy_end_month, quarter)
            return ResolvedPeriod(
                ticker=ticker,
                fiscal_year=fiscal_year,
                quarter=quarter,
                is_calendar_explicit=True,
                raw_match=cy_match.group(0),
                explanation=f"Explicit calendar year {cy_year} mapped to FY{fiscal_year} (FY end month: {fy_end_month}).",
            )

        # 4. Bare Year Mention: "in 2025", "2024 results"
        bare_match = cls._BARE_YEAR_RE.search(query_clean)
        if bare_match:
            year_val = int(bare_match.group(1))
            # For companies with January fiscal year-end (e.g. NVDA, WMT),
            # asking for "2025 revenue" typically refers to the filing covering calendar year 2025,
            # which is SEC Form 10-K for Fiscal Year 2026 (ended Jan 2026).
            # When looking up by year, map calendar year to the filing fiscal year if annual.
            if fy_end_month == 1:
                # NVDA/WMT: Calendar 2025 is FY2026
                fiscal_year = year_val + 1 if quarter is None else year_val + 1
            elif fy_end_month == 12:
                # Calendar year company (NFLX, UNH)
                fiscal_year = year_val
            else:
                fiscal_year = year_val

            return ResolvedPeriod(
                ticker=ticker,
                fiscal_year=fiscal_year,
                quarter=quarter,
                is_fiscal_explicit=False,
                is_calendar_explicit=False,
                raw_match=bare_match.group(0),
                explanation=f"Bare year {year_val} normalized to FY{fiscal_year} for {ticker or 'general'}.",
            )

        return ResolvedPeriod(
            ticker=ticker,
            fiscal_year=None,
            quarter=quarter,
            explanation="No explicit year identified.",
        )

    @classmethod
    def _calendar_to_fiscal_year(
        cls, calendar_year: int, fy_end_month: int, quarter: str | None
    ) -> int:
        """
        Map a calendar year to corporate fiscal year.
        If fy_end_month == 1 (Jan), FY2026 ends Jan 2026 and covers Feb 2025 - Jan 2026 (mostly calendar 2025).
        So calendar 2025 corresponds to FY2026.
        """
        if fy_end_month == 1:
            return calendar_year + 1
        elif fy_end_month <= 6:
            return calendar_year
        else:
            return calendar_year
