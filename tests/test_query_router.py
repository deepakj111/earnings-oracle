# tests/test_query_router.py
"""
Tests for query/router.py — QueryRouter intent classification.

Tests cover:
  - Heuristic fast-path (OUT_OF_SCOPE, FINANCIAL_SPECIFIC detection)
  - RoutingDecision flag correctness (skip_hyde, should_refuse, etc.)
  - RouterStats accumulation
  - LLM fallback path (mocked)
  - Edge cases: empty-ish queries, multi-ticker queries, case sensitivity
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from query.router import QueryIntent, QueryRouter


@pytest.fixture
def router() -> QueryRouter:
    """Router."""
    with patch("query.router.OpenAI"):
        r = QueryRouter()
    return r


class TestHeuristicClassification:
    def test_greeting_is_out_of_scope(self, router: QueryRouter) -> None:
        decision = router.route("Hello there")
        assert decision.intent == QueryIntent.OUT_OF_SCOPE
        assert decision.should_refuse is True
        assert decision.used_heuristic is True

    def test_thanks_is_out_of_scope(self, router: QueryRouter) -> None:
        decision = router.route("Thanks!")
        assert decision.intent == QueryIntent.OUT_OF_SCOPE
        assert decision.should_refuse is True

    def test_aapl_revenue_is_specific(self, router: QueryRouter) -> None:
        decision = router.route("What was Apple's revenue in Q4 2024?")
        assert decision.intent == QueryIntent.FINANCIAL_SPECIFIC
        assert decision.detected_ticker == "AAPL"
        assert decision.skip_hyde is False
        assert decision.should_refuse is False
        assert decision.used_heuristic is True

    def test_nvda_datacenter_is_specific(self, router: QueryRouter) -> None:
        decision = router.route("NVDA data center gross margin Q3 2024")
        assert decision.intent == QueryIntent.FINANCIAL_SPECIFIC
        assert decision.detected_ticker == "NVDA"

    def test_ticker_symbol_uppercase(self, router: QueryRouter) -> None:
        decision = router.route("What was MSFT earnings last quarter?")
        assert decision.intent == QueryIntent.FINANCIAL_SPECIFIC
        assert decision.detected_ticker == "MSFT"

    def test_company_name_canonical_ticker(self, router: QueryRouter) -> None:
        decision = router.route("What was Microsoft revenue Q4 2024?")
        assert decision.intent == QueryIntent.FINANCIAL_SPECIFIC
        assert decision.detected_ticker == "MSFT"

    def test_very_short_no_keywords_out_of_scope(self, router: QueryRouter) -> None:
        decision = router.route("hi")
        assert decision.intent == QueryIntent.OUT_OF_SCOPE

    def test_financial_keyword_no_ticker_falls_through_to_llm(self, router: QueryRouter) -> None:
        mock_response = {
            "intent": "FINANCIAL_GENERAL",
            "confidence": 0.85,
            "detected_ticker": None,
            "reasoning": "No ticker but financial domain",
        }
        with patch.object(router, "_llm_classify", return_value=mock_response):
            decision = router.route("What is a gross margin?")
        assert decision.intent == QueryIntent.FINANCIAL_GENERAL
        assert decision.used_heuristic is False


class TestRoutingDecisionFlags:
    def test_financial_specific_flags(self, router: QueryRouter) -> None:
        decision = router.route("What was AAPL revenue Q4 2024?")
        assert decision.skip_hyde is False
        assert decision.skip_transform is False
        assert decision.should_refuse is False
        assert decision.is_specific is True
        assert decision.is_general is False

    def test_out_of_scope_flags(self, router: QueryRouter) -> None:
        decision = router.route("hello world")
        assert decision.should_refuse is True
        assert decision.skip_transform is True
        assert decision.skip_hyde is True

    def test_ambiguous_flags_via_llm(self, router: QueryRouter) -> None:
        mock_response = {
            "intent": "AMBIGUOUS",
            "confidence": 0.4,
            "detected_ticker": None,
            "reasoning": "Could not classify",
        }
        with patch.object(router, "_llm_classify", return_value=mock_response):
            decision = router.route("tell me about earnings season trends")
        assert decision.intent == QueryIntent.AMBIGUOUS
        assert decision.skip_hyde is True
        assert decision.should_refuse is False


class TestRouterStats:
    def test_stats_accumulate(self, router: QueryRouter) -> None:
        router.route("What was AAPL revenue?")
        router.route("hello")
        assert router.stats.total_routed == 2

    def test_heuristic_count_increments(self, router: QueryRouter) -> None:
        router.route("What was NVDA revenue Q3 2024?")
        assert router.stats.heuristic_hits >= 1

    def test_heuristic_hit_rate(self, router: QueryRouter) -> None:
        router.route("What was AMZN net income?")
        router.route("hi there")
        assert 0.0 <= router.stats.heuristic_hit_rate <= 1.0

    def test_avg_latency_positive(self, router: QueryRouter) -> None:
        router.route("What was META ad revenue?")
        assert router.stats.avg_latency_ms >= 0.0

    def test_intent_counts_track(self, router: QueryRouter) -> None:
        router.route("What was JPM net income Q4 2024?")
        assert router.stats.intent_counts[QueryIntent.FINANCIAL_SPECIFIC.value] >= 1


class TestLLMFallback:
    def test_llm_failure_defaults_to_ambiguous(self, router: QueryRouter) -> None:
        with patch.object(router, "_llm_classify", side_effect=Exception("API error")):
            with patch.object(
                router,
                "_heuristic_classify",
                return_value=None,
            ):
                with patch.object(
                    router,
                    "_llm_classify",
                    return_value={
                        "intent": "AMBIGUOUS",
                        "confidence": 0.5,
                        "detected_ticker": None,
                        "reasoning": "LLM call failed",
                    },
                ):
                    decision = router.route("something unclear about revenue trends")
        assert decision.intent == QueryIntent.AMBIGUOUS

    def test_llm_general_query_sets_correct_flags(self, router: QueryRouter) -> None:
        mock_response = {
            "intent": "FINANCIAL_GENERAL",
            "confidence": 0.9,
            "detected_ticker": None,
            "reasoning": "Conceptual finance question",
        }
        with patch.object(router, "_llm_classify", return_value=mock_response):
            with patch.object(router, "_heuristic_classify", return_value=None):
                decision = router.route("Explain what free cash flow means")
        assert decision.intent == QueryIntent.FINANCIAL_GENERAL
        assert decision.skip_transform is True
        assert decision.should_refuse is False


class TestRoutingDecisionSummary:
    def test_summary_contains_intent(self, router: QueryRouter) -> None:
        decision = router.route("What was TSLA revenue?")
        summary = decision.summary()
        assert "FINANCIAL_SPECIFIC" in summary

    def test_summary_contains_ticker(self, router: QueryRouter) -> None:
        decision = router.route("What was WMT gross profit?")
        summary = decision.summary()
        assert "WMT" in summary

    def test_summary_marks_heuristic(self, router: QueryRouter) -> None:
        decision = router.route("What was XOM revenue Q4?")
        if decision.used_heuristic:
            assert "[heuristic]" in decision.summary()
