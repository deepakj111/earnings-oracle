from query.fiscal_resolver import FiscalResolver
from query.router import QueryIntent, QueryRouter


def test_explicit_fiscal_year():
    res = FiscalResolver.resolve("What was NVIDIA's revenue in FY25?", ticker="NVDA")
    assert res.fiscal_year == 2025
    assert res.is_fiscal_explicit is True

    res2 = FiscalResolver.resolve("Walmart fiscal year 2024 operating margin", ticker="WMT")
    assert res2.fiscal_year == 2024
    assert res2.is_fiscal_explicit is True


def test_quarter_extraction():
    res = FiscalResolver.resolve("What was Netflix revenue in Q3 2024?", ticker="NFLX")
    assert res.fiscal_year == 2024
    assert res.quarter == "Q3"

    res2 = FiscalResolver.resolve("UnitedHealth second quarter 2024 earnings", ticker="UNH")
    assert res2.fiscal_year == 2024
    assert res2.quarter == "Q2"


def test_calendar_to_fiscal_normalization():
    # NVDA fiscal year ends in January, so calendar year 2025 maps to FY2026
    res_nvda = FiscalResolver.resolve("What was NVIDIA revenue in 2025?", ticker="NVDA")
    assert res_nvda.fiscal_year == 2026

    # NFLX fiscal year ends in December, so calendar year 2024 maps to FY2024
    res_nflx = FiscalResolver.resolve("What was Netflix revenue in 2024?", ticker="NFLX")
    assert res_nflx.fiscal_year == 2024


def test_router_integration():
    router = QueryRouter()
    decision = router.route("What was NVIDIA revenue in FY25?")
    assert decision.intent == QueryIntent.FINANCIAL_SPECIFIC
    assert decision.detected_ticker == "NVDA"
    assert decision.detected_year == 2025
