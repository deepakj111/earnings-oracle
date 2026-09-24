"""
Dual-Path Structured SEC Facts Store.

Stores and queries structured, verified GAAP financial facts (Revenues, Operating Income,
Net Income, Diluted EPS, Margins) extracted directly from SEC Form 10-K and 10-Q filings.
Provides authoritative quantitative ground truth for retrieval and PAL validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

_FACTS_FILE = Path("data/financial_facts.json")


@dataclass
class FinancialFact:
    ticker: str
    concept: str  # e.g. "Revenues", "OperatingIncomeLoss", "NetIncomeLoss", "DilutedEPS"
    label: str  # e.g. "Total Revenue", "Operating Income", "Net Income"
    fiscal_year: int
    quarter: str  # "Q1", "Q2", "Q3", "Q4", or "FY"
    period_end: str  # e.g. "2025-01-26"
    value: float  # Numerical value
    unit: str = "USD"
    scale: str = "millions"  # "millions", "thousands", "ratio"
    source_file: str = ""

    def formatted_display(self) -> str:
        if self.scale == "ratio" or self.unit == "per share":
            return f"${self.value:.2f}"
        return f"${self.value:,.1f} {self.scale} {self.unit}"


class FactStore:
    """
    Manages persistence and lookup of verified SEC financial facts.
    """

    _cached_facts: list[FinancialFact] | None = None

    @classmethod
    def load(cls, force_reload: bool = False) -> list[FinancialFact]:
        if not force_reload and cls._cached_facts is not None:
            return cls._cached_facts

        if not _FACTS_FILE.exists():
            cls._cached_facts = []
            return cls._cached_facts

        try:
            with open(_FACTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            facts = [FinancialFact(**item) for item in data]
            cls._cached_facts = facts
            logger.info(f"Loaded {len(facts)} verified financial facts from {_FACTS_FILE}")
            return facts
        except Exception as exc:
            logger.warning(f"Could not load facts file {_FACTS_FILE}: {exc}")
            cls._cached_facts = []
            return cls._cached_facts

    @classmethod
    def save(cls, facts: list[FinancialFact]) -> None:
        _FACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Deduplicate by key: (ticker, concept, fiscal_year, quarter)
        facts_map: dict[tuple[str, str, int, str], FinancialFact] = {}
        for f in facts:
            k = (f.ticker.upper(), f.concept.lower(), f.fiscal_year, (f.quarter or "").upper())
            facts_map[k] = f

        deduped = list(facts_map.values())
        data = [asdict(item) for item in deduped]
        with open(_FACTS_FILE, "w", encoding="utf-8") as file_out:
            json.dump(data, file_out, indent=2)

        cls._cached_facts = deduped
        logger.info(f"Saved {len(deduped)} financial facts to {_FACTS_FILE}")

    @classmethod
    def query(
        cls,
        ticker: str,
        fiscal_year: int | None = None,
        quarter: str | None = None,
        concept: str | None = None,
    ) -> list[FinancialFact]:
        """
        Query verified facts by ticker, year, quarter, and concept.
        """
        facts = cls.load()
        t_upper = ticker.upper().strip()
        q_upper = quarter.upper().strip() if quarter else None
        c_lower = concept.lower().strip() if concept else None

        matches = []
        for f in facts:
            if f.ticker.upper() != t_upper:
                continue
            if fiscal_year is not None and f.fiscal_year != fiscal_year:
                continue
            if q_upper is not None and (f.quarter or "").upper() != q_upper:
                continue
            if c_lower is not None and c_lower not in f.concept.lower():
                continue
            matches.append(f)

        return matches

    @classmethod
    def format_as_context(cls, facts: list[FinancialFact]) -> str:
        """
        Format a list of facts into a high-authority numbered context passage.
        """
        if not facts:
            return ""

        lines = ["--- [VERIFIED GAAP FINANCIAL FACTS] ---"]
        for f in facts:
            q_label = f.quarter or "FY"
            lines.append(
                f"• {f.ticker} ({f.fiscal_year} {q_label}): {f.label} = {f.formatted_display()} "
                f"[Period Ended: {f.period_end}]"
            )
        return "\n".join(lines)
