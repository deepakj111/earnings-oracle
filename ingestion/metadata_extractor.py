from dataclasses import dataclass

from config.companies import CompanyRegistry

# Export COMPANY_MAP dynamically from central CompanyRegistry
COMPANY_MAP: dict[str, str] = CompanyRegistry.get_company_map()


def register_company(
    ticker: str,
    name: str,
    cik: str = "",
    sector: str = "General",
    fiscal_year_end_month: int = 12,
) -> None:
    """Dynamically register a new ticker and company profile at runtime."""
    CompanyRegistry.register_company(
        ticker=ticker,
        name=name,
        cik=cik,
        sector=sector,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    global COMPANY_MAP
    COMPANY_MAP = CompanyRegistry.get_company_map()


@dataclass
class DocumentMetadata:
    ticker: str
    company: str
    date: str
    year: int
    quarter: str  # "Q1" | "Q2" | "Q3" | "Q4" | "FY"
    fiscal_period: str  # e.g. "Q1 2025" | "FY 2024"
    form_type: str = "unknown"  # "10-K" | "10-Q" | "unknown"
    file_name: str = ""


def _derive_fiscal_period(
    form_type: str,
    filing_date: str,
    ticker: str = "",
) -> tuple[str, int, str]:
    """
    Derive (fiscal_period, fiscal_year, quarter) via central CompanyRegistry.
    """
    fiscal_year, quarter, fiscal_period = CompanyRegistry.derive_fiscal_period(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
    )
    return fiscal_period, fiscal_year, quarter


def extract_metadata(
    ticker: str,
    date: str,
    raw_text: str,
    form_type: str = "unknown",
    file_name: str = "",
) -> DocumentMetadata:
    """
    Extract standard metadata from a financial filing.
    """
    fiscal_period, fiscal_year, quarter = _derive_fiscal_period(form_type, date, ticker=ticker)

    company_map = CompanyRegistry.get_company_map()
    return DocumentMetadata(
        ticker=ticker.upper(),
        company=company_map.get(ticker.upper(), ticker),
        date=date,
        year=fiscal_year,
        quarter=quarter,
        fiscal_period=fiscal_period,
        form_type=form_type,
        file_name=file_name,
    )
