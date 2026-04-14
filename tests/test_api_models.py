# tests/test_api_models.py
"""
Tests for api/models.py

Coverage:
  AskRequest       — question length bounds, verbose default, filter passthrough
  MetadataFilterIn — ticker allow-list (case-insensitive), quarter enum,
                     year range bounds, all-optional semantics
  CitationOut      — field presence
  AskResponse      — full shape including optional verbose fields
  ComponentStatus  — status enum-like values
  HealthResponse   — shape

These are pure Pydantic unit tests — no HTTP client required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.models import (
    AskRequest,
    AskResponse,
    CitationOut,
    ComponentStatus,
    ContextOut,
    HealthResponse,
    MetadataFilterIn,
    UsageOut,
)

# ─────────────────────────────────────────────────────────────────────────────
# AskRequest
# ─────────────────────────────────────────────────────────────────────────────


class TestAskRequest:
    def test_valid_minimal(self) -> None:
        req = AskRequest(question="Apple revenue?")
        assert req.question == "Apple revenue?"
        assert req.filter is None
        assert req.verbose is False

    def test_valid_with_all_fields(self) -> None:
        req = AskRequest(
            question="What was NVDA Q4 2024 revenue?",
            filter=MetadataFilterIn(ticker="NVDA", year=2024, quarter="Q4"),
            verbose=True,
        )
        assert req.verbose is True
        assert req.filter.ticker == "NVDA"

    def test_question_minimum_length(self) -> None:
        """min_length=3 — exactly 3 chars should pass."""
        req = AskRequest(question="Why")
        assert req.question == "Why"

    def test_question_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="Hi")
        errors = exc_info.value.errors()
        assert any("min_length" in str(e) or "string_too_short" in str(e["type"]) for e in errors)

    def test_question_at_maximum_length(self) -> None:
        """max_length=2000 — exactly 2000 chars should pass."""
        req = AskRequest(question="A" * 2000)
        assert len(req.question) == 2000

    def test_question_exceeds_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            AskRequest(question="A" * 2001)

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            AskRequest()  # type: ignore[call-arg]

    def test_verbose_defaults_to_false(self) -> None:
        req = AskRequest(question="Test question here?")
        assert req.verbose is False

    def test_verbose_explicit_true(self) -> None:
        req = AskRequest(question="Test question here?", verbose=True)
        assert req.verbose is True


# ─────────────────────────────────────────────────────────────────────────────
# MetadataFilterIn
# ─────────────────────────────────────────────────────────────────────────────


class TestMetadataFilterIn:
    # ── Ticker validation ──────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "ticker",
        ["AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"],
    )
    def test_valid_tickers(self, ticker: str) -> None:
        f = MetadataFilterIn(ticker=ticker)
        assert f.ticker == ticker

    def test_ticker_lowercased_is_uppercased(self) -> None:
        f = MetadataFilterIn(ticker="aapl")
        assert f.ticker == "AAPL"

    def test_ticker_mixed_case_normalised(self) -> None:
        f = MetadataFilterIn(ticker="nvdA")
        assert f.ticker == "NVDA"

    def test_unknown_ticker_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MetadataFilterIn(ticker="GOOG")
        assert "GOOG" in str(exc_info.value)

    def test_ticker_none_is_allowed(self) -> None:
        f = MetadataFilterIn(ticker=None)
        assert f.ticker is None

    def test_all_fields_none_is_valid(self) -> None:
        f = MetadataFilterIn()
        assert f.ticker is None
        assert f.year is None
        assert f.quarter is None

    # ── Quarter validation ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("quarter", ["Q1", "Q2", "Q3", "Q4"])
    def test_valid_quarters(self, quarter: str) -> None:
        f = MetadataFilterIn(quarter=quarter)
        assert f.quarter == quarter

    def test_quarter_lowercased_normalised(self) -> None:
        f = MetadataFilterIn(quarter="q3")
        assert f.quarter == "Q3"

    def test_invalid_quarter_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MetadataFilterIn(quarter="Q5")
        assert "Q5" in str(exc_info.value)

    def test_quarter_none_is_allowed(self) -> None:
        f = MetadataFilterIn(quarter=None)
        assert f.quarter is None

    # ── Year validation ────────────────────────────────────────────────────────

    def test_year_within_range(self) -> None:
        f = MetadataFilterIn(year=2024)
        assert f.year == 2024

    def test_year_at_lower_bound(self) -> None:
        f = MetadataFilterIn(year=2020)
        assert f.year == 2020

    def test_year_at_upper_bound(self) -> None:
        f = MetadataFilterIn(year=2030)
        assert f.year == 2030

    def test_year_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetadataFilterIn(year=2019)

    def test_year_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetadataFilterIn(year=2031)

    def test_year_none_is_allowed(self) -> None:
        f = MetadataFilterIn(year=None)
        assert f.year is None


# ─────────────────────────────────────────────────────────────────────────────
# CitationOut
# ─────────────────────────────────────────────────────────────────────────────


class TestCitationOut:
    def _make(self) -> CitationOut:
        return CitationOut(
            index=1,
            ticker="AAPL",
            company="Apple",
            date="2024-10-31",
            fiscal_period="Q4 2024",
            section_title="Revenue",
            doc_type="earnings_release",
            source="both",
            rerank_score=0.9821,
            excerpt="Apple reported $94.9B.",
        )

    def test_all_fields_present(self) -> None:
        c = self._make()
        assert c.ticker == "AAPL"
        assert c.index == 1
        assert c.rerank_score == 0.9821

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            CitationOut(
                index=1,
                ticker="AAPL",
                # company missing
                date="2024-10-31",
                fiscal_period="Q4 2024",
                section_title="Revenue",
                doc_type="earnings_release",
                source="dense",
                rerank_score=0.9,
                excerpt="...",
            )


# ─────────────────────────────────────────────────────────────────────────────
# AskResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestAskResponse:
    def _make(self) -> AskResponse:
        return AskResponse(
            question="Apple revenue?",
            answer="$94.9B [1].",
            citations=[],
            grounded=True,
            retrieval_failed=False,
            model="gpt-4.1-nano",
            usage=UsageOut(
                prompt_tokens=1000,
                completion_tokens=80,
                total_tokens=1080,
            ),
            context=ContextOut(chunks_used=3, tokens_used=2048),
            latency_seconds=2.5,
            unique_tickers=[],
            unique_sources=[],
        )

    def test_valid_construction(self) -> None:
        resp = self._make()
        assert resp.question == "Apple revenue?"
        assert resp.grounded is True

    def test_verbose_fields_default_none(self) -> None:
        resp = self._make()
        assert resp.query_summary is None
        assert resp.retrieval_summary is None

    def test_verbose_fields_can_be_set(self) -> None:
        resp = self._make()
        resp.query_summary = "qs"
        resp.retrieval_summary = "rs"
        assert resp.query_summary == "qs"


# ─────────────────────────────────────────────────────────────────────────────
# ComponentStatus / HealthResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthModels:
    def test_component_status_ok(self) -> None:
        cs = ComponentStatus(status="ok", detail="All good")
        assert cs.status == "ok"

    def test_component_status_detail_optional(self) -> None:
        cs = ComponentStatus(status="error")
        assert cs.detail is None

    def test_health_response_structure(self) -> None:
        hr = HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=120.5,
            components={
                "qdrant": ComponentStatus(status="ok"),
                "pipeline": ComponentStatus(status="ok"),
            },
        )
        assert hr.status == "healthy"
        assert hr.uptime_seconds == 120.5
        assert "qdrant" in hr.components
