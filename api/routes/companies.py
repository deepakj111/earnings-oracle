# api/routes/companies.py
"""
Company directory and metadata routes — /companies.

Provides dynamic discovery of configured SEC filers, fiscal year-end months,
and brand/entity aliases without hardcoding tickers in frontend or clients.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from config.companies import CompanyRegistry

router = APIRouter()


@router.get(
    "",
    include_in_schema=False,
)
@router.get(
    "/",
    summary="List supported companies",
    description=(
        "Returns metadata for all configured public companies, including ticker, "
        "full corporate name, SEC CIK, sector, and fiscal calendar specifications."
    ),
)
async def list_companies() -> list[dict[str, Any]]:
    """Return all configured companies in alphabetical order by ticker."""
    profiles = CompanyRegistry.get_all_companies()
    return sorted([p.to_dict() for p in profiles], key=lambda x: x["ticker"])


@router.get(
    "/{ticker}",
    summary="Get single company profile",
    description="Returns detailed company profile and fiscal metadata for a specific ticker.",
)
async def get_company(ticker: str) -> dict[str, Any]:
    """Return metadata for a single company by ticker, or 404 if not found."""
    prof = CompanyRegistry.get_company(ticker)
    if not prof:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{ticker.upper()}' is not configured in the registry.",
        )
    return prof.to_dict()
