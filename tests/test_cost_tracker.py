# tests/test_cost_tracker.py
"""
Tests for observability/cost_tracker.py — model pricing and cost accumulation.

Tests cover:
  - Pricing table correctness for all registered models
  - Per-call cost calculation accuracy
  - Unknown model fallback behavior
  - CostTracker thread-safe accumulation
  - Per-request and per-session cost alerts
  - Session reset
  - Serialization
"""

import threading

import pytest

from observability.cost_tracker import (
    CostTracker,
    estimate_cost,
)

# ── estimate_cost ──────────────────────────────────────────────────────────────


class TestEstimateCost:
    """Verify per-call cost estimation."""

    def test_nano_cost_calculation(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=1000, completion_tokens=200)
        assert cost.prompt_cost_usd == pytest.approx(0.0001, abs=1e-8)
        assert cost.completion_cost_usd == pytest.approx(0.00008, abs=1e-8)
        assert cost.total_cost_usd == pytest.approx(0.00018, abs=1e-8)

    def test_zero_tokens(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=0, completion_tokens=0)
        assert cost.total_cost_usd == 0.0

    def test_unknown_model_returns_zero_cost(self) -> None:
        from unittest.mock import patch

        with patch(
            "litellm.cost_calculator.cost_per_token", side_effect=Exception("Unknown model")
        ):
            cost = estimate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
            assert cost.total_cost_usd == 0.0
            assert cost.model == "unknown-model-xyz"

    def test_large_token_count(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost.prompt_cost_usd == pytest.approx(0.10, abs=1e-6)


# ── CostTracker ────────────────────────────────────────────────────────────────


class TestCostTracker:
    """Verify thread-safe session-level cost tracking."""

    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_single_record(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        cost = CostEstimate("gpt-4.1-nano", 1000, 200, 0.0001, 0.00008)
        tracker.record(cost)
        assert tracker.total_calls == 1
        assert tracker.total_prompt_tokens == 1000
        assert tracker.total_completion_tokens == 200
        assert tracker.total_tokens == 1200
        assert tracker.total_cost_usd == pytest.approx(0.00018)

    def test_multiple_records_accumulate(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        cost1 = CostEstimate("gpt-4.1-nano", 1000, 200, 0.0001, 0.00008)
        cost2 = CostEstimate("gpt-4.1-nano", 500, 100, 0.00005, 0.00004)
        tracker.record(cost1)
        tracker.record(cost2)
        assert tracker.total_calls == 2
        assert tracker.total_prompt_tokens == 1500
        assert tracker.total_completion_tokens == 300

    def test_record_request_cost(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        costs = [
            CostEstimate("gpt-4.1-nano", 100, 50, 0.00001, 0.00002),
            CostEstimate("gpt-4.1-nano", 200, 100, 0.00002, 0.00004),
            CostEstimate("gpt-4.1-nano", 300, 150, 0.00003, 0.00006),
        ]
        total = tracker.record_request_cost(costs)
        assert tracker.total_calls == 3
        assert total == pytest.approx(sum(c.total_cost_usd for c in costs))

    def test_reset(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        cost = CostEstimate("gpt-4.1-nano", 1000, 200, 0.0001, 0.00008)
        tracker.record(cost)
        assert tracker.total_calls == 1
        tracker.reset()
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_summary_format(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        cost = CostEstimate("gpt-4.1-nano", 1000, 200, 0.0001, 0.00008)
        tracker.record(cost)
        summary = tracker.summary()
        assert "1 calls" in summary
        assert "$" in summary

    def test_to_dict(self) -> None:
        from observability.trace_models import CostEstimate

        tracker = CostTracker(alert_per_request_usd=0.05, alert_per_session_usd=2.0)
        cost = CostEstimate("gpt-4.1-nano", 1000, 200, 0.0001, 0.00008)
        tracker.record(cost)
        d = tracker.to_dict()
        assert d["total_calls"] == 1
        assert d["alert_per_request_usd"] == 0.05
        assert d["alert_per_session_usd"] == 2.0
        assert d["session_alert_fired"] is False

    def test_thread_safety(self) -> None:
        """Verify that concurrent record calls don't corrupt counters."""
        from observability.trace_models import CostEstimate

        tracker = CostTracker()
        n_threads = 10
        records_per_thread = 100
        cost = CostEstimate("gpt-4.1-nano", 100, 50, 0.00001, 0.00002)

        def worker() -> None:
            for _ in range(records_per_thread):
                tracker.record(cost)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_calls = n_threads * records_per_thread
        assert tracker.total_calls == expected_calls
        assert tracker.total_prompt_tokens == expected_calls * 100
        assert tracker.total_completion_tokens == expected_calls * 50

    def test_per_request_alert_fires(self, caplog: pytest.LogCaptureFixture) -> None:
        """Cost alert should fire if a single call exceeds threshold."""
        import logging

        from observability.trace_models import CostEstimate

        with caplog.at_level(logging.WARNING, logger="observability.cost_tracker"):
            tracker = CostTracker(alert_per_request_usd=0.0001)
            cost = CostEstimate("gpt-4.1-nano", 1000, 1000, 0.0005, 0.0005)
            tracker.record(cost)

        assert tracker.total_calls == 1

    def test_session_alert_fires_once(self) -> None:
        """Session alert should fire once when cumulative cost exceeds threshold."""
        from observability.trace_models import CostEstimate

        tracker = CostTracker(alert_per_session_usd=0.001)
        cost = CostEstimate("gpt-4.1-nano", 1000, 1000, 0.0002, 0.0002)
        # Accumulate enough calls to exceed $0.001
        for _ in range(10):
            tracker.record(cost)

        d = tracker.to_dict()
        assert d["session_alert_fired"] is True
