import re
from dataclasses import dataclass

COMPANY_MAP: dict[str, str] = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "JPM": "JPMorgan Chase",
    "XOM": "ExxonMobil",
    "UNH": "UnitedHealth Group",
    "TSLA": "Tesla",
    "WMT": "Walmart",
}

MONTH_TO_QUARTER: dict[int, str] = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q2", 5: "Q2", 6: "Q2",
    7: "Q3", 8: "Q3", 9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4",
}

QUARTER_PATTERNS: list[tuple[str, str]] = [
    (r"\bfirst quarter\b", "Q1"),
    (r"\bsecond quarter\b", "Q2"),
    (r"\bthird quarter\b", "Q3"),
    (r"\bfourth quarter\b", "Q4"),
    (r"\bQ1\b", "Q1"),
    (r"\bQ2\b", "Q2"),
    (r"\bQ3\b", "Q3"),
    (r"\bQ4\b", "Q4"),
]


@dataclass
class DocumentMetadata:
    ticker: str
    company: str
    date: str
    year: int
    quarter: str
    fiscal_period: str


def _detect_quarter(text: str, fallback_month: int) -> str:
    sample = text[:3000]
    for pattern, quarter in QUARTER_PATTERNS:
        if re.search(pattern, sample, re.IGNORECASE):
            return quarter
    return MONTH_TO_QUARTER.get(fallback_month, "Q1")


def extract_metadata(ticker: str, date: str, raw_text: str) -> DocumentMetadata:
    parts = date.split("-")
    year = int(parts[0]) if len(parts) >= 1 else 2024
    month = int(parts[1]) if len(parts) >= 2 else 1
    quarter = _detect_quarter(raw_text, month)

    return DocumentMetadata(
        ticker=ticker,
        company=COMPANY_MAP.get(ticker, ticker),
        date=date,
        year=year,
        quarter=quarter,
        fiscal_period=f"{quarter} {year}",
    )