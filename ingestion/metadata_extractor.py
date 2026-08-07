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
    "NFLX": "Netflix",
}


def register_company(ticker: str, name: str) -> None:
    """Dynamically register a new ticker and company name at runtime."""
    COMPANY_MAP[ticker.upper().strip()] = name.strip()


# ---------------------------------------------------------------------------
# Fiscal period derivation — form-type-aware
# ---------------------------------------------------------------------------
# SEC filing publication lag (approximate):
#   10-K  published Jan–Mar → covers the *prior* fiscal year (FY Y-1)
#   10-Q  published Apr–May → covers Q1 of the filing year
#   10-Q  published Jul–Aug → covers Q2 of the filing year
#   10-Q  published Oct–Nov → covers Q3 of the filing year
#
# Note: There is no separate Q4 10-Q filing; Q4 is covered by the annual 10-K.
#
# Month → (quarter_label, year_offset) for each form type:
#   year_offset = -1 means the period is in the year *before* the filing date.
# ---------------------------------------------------------------------------

_10K_FISCAL_PERIOD = "FY"  # Annual — no quarter subdivision

# For 10-Q filings, map the filing month to the fiscal quarter it covers.
# These ranges reflect the typical 40-45 day reporting window after quarter end.
_10Q_MONTH_TO_QUARTER: dict[int, tuple[str, int]] = {
    # month : (quarter, year_offset)
    1: ("Q3", -1),  # very late 10-Q for prior year Q3 (rare)
    2: ("Q3", -1),  # rare late filer
    3: ("Q4", -1),  # very late 10-Q for prior year Q4 (extremely rare; usually 10-K)
    4: ("Q1", 0),  # Q1 ends Mar 31, filed Apr
    5: ("Q1", 0),  # Q1 late filer
    6: ("Q2", 0),  # Q2 ends Jun 30, filed Jun (rare early filer)
    7: ("Q2", 0),  # Q2 ends Jun 30, filed Jul
    8: ("Q2", 0),  # Q2 late filer
    9: ("Q3", 0),  # Q3 ends Sep 30, filed Sep (rare early filer)
    10: ("Q3", 0),  # Q3 ends Sep 30, filed Oct
    11: ("Q3", 0),  # Q3 late filer
    12: ("Q4", 0),  # Q4 filed Dec (extremely rare; usually next Jan as 10-K)
}

# For 10-K filings, the filing month → fiscal year offset.
# Companies with non-December fiscal year ends may file at different months,
# but Jan–Mar is by far the most common window for calendar-year companies.
_10K_MONTH_TO_YEAR_OFFSET: dict[int, int] = {
    1: -1,
    2: -1,
    3: -1,  # Jan–Mar: covers prior calendar year (most common)
    4: -1,
    5: -1,
    6: -1,  # Apr–Jun: fiscal year ending in prior year (e.g. Apr fiscal year)
    7: 0,
    8: 0,
    9: 0,  # Jul–Sep: fiscal year ending mid-year (e.g. Jun fiscal year-end)
    10: 0,
    11: 0,
    12: 0,  # Oct–Dec: fiscal year ending in current year
}


@dataclass
class DocumentMetadata:
    ticker: str
    company: str
    date: str
    year: int
    quarter: str  # "Q1" | "Q2" | "Q3" | "Q4" | "FY"
    fiscal_period: str  # e.g. "Q1 2025" | "FY 2024"
    form_type: str = "unknown"  # "10-K" | "10-Q" | "unknown"


def _derive_fiscal_period(
    form_type: str,
    filing_date: str,
) -> tuple[str, int, str]:
    """
    Derive (fiscal_period, fiscal_year, quarter) from the SEC form type and filing date.

    This is the authoritative source of truth for period labeling.  It replaces the
    previous ``_detect_quarter()`` heuristic which scanned document text and
    systematically mis-labelled every document by one quarter.

    Args:
        form_type  : "10-K", "10-Q", or any string (falls back gracefully)
        filing_date: ISO date string "YYYY-MM-DD"

    Returns:
        (fiscal_period_str, fiscal_year_int, quarter_str)
        e.g. ("Q2 2025", 2025, "Q2") or ("FY 2024", 2024, "FY")
    """
    year = 2024
    month = 1

    if filing_date and "-" in filing_date:
        parts = filing_date.split("-")
        if len(parts) >= 1 and parts[0].isdigit():
            year = int(parts[0])
        if len(parts) >= 2 and parts[1].isdigit():
            month = int(parts[1])

    form_upper = form_type.upper().strip()

    if form_upper in ("10-K", "10-K/A"):
        # Annual report — fiscal year is typically the year before filing
        year_offset = _10K_MONTH_TO_YEAR_OFFSET.get(month, -1)
        fiscal_year = year + year_offset
        quarter = "FY"
        fiscal_period = f"FY {fiscal_year}"

    elif form_upper in ("10-Q", "10-Q/A"):
        # Quarterly report — derive fiscal quarter from filing month
        quarter, year_offset = _10Q_MONTH_TO_QUARTER.get(month, ("Q?", 0))
        fiscal_year = year + year_offset
        fiscal_period = f"{quarter} {fiscal_year}"

    else:
        # Unknown form type — fall back to filing month → quarter heuristic
        # (preserved for any non-standard form types)
        quarter_map = {
            1: "Q1",
            2: "Q1",
            3: "Q1",
            4: "Q2",
            5: "Q2",
            6: "Q2",
            7: "Q3",
            8: "Q3",
            9: "Q3",
            10: "Q4",
            11: "Q4",
            12: "Q4",
        }
        quarter = quarter_map.get(month, "Q1")
        fiscal_year = year
        fiscal_period = f"{quarter} {fiscal_year}"

    return fiscal_period, fiscal_year, quarter


def extract_metadata(
    ticker: str,
    date: str,
    raw_text: str,
    form_type: str = "unknown",
) -> DocumentMetadata:
    """
    Extract standard metadata from a financial filing.

    Args:
        ticker    : Company ticker symbol (e.g. "NFLX")
        date      : Filing date as ISO string "YYYY-MM-DD" (from filename)
        raw_text  : Full document text (no longer used for period detection,
                    kept for API compatibility)
        form_type : SEC form type from filename (e.g. "10-K", "10-Q").
                    Defaults to "unknown" for backwards compatibility.

    Returns:
        DocumentMetadata with correct fiscal_period, year, and quarter.
    """
    fiscal_period, fiscal_year, quarter = _derive_fiscal_period(form_type, date)

    return DocumentMetadata(
        ticker=ticker,
        company=COMPANY_MAP.get(ticker.upper(), ticker),
        date=date,
        year=fiscal_year,
        quarter=quarter,
        fiscal_period=fiscal_period,
        form_type=form_type,
    )
