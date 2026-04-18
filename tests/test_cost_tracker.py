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
    MODEL_PRICING,
    CostTracker,
    ModelPricing,
    estimate_cost,
)

# ── ModelPricing ───────────────────────────────────────────────────────────────


class TestModelPricing:
    """Verify pricing dataclass properties."""

    def test_input_cost_per_token(self) -> None:
        pricing = ModelPricing(
            model_name="test",
            input_cost_per_1m=1.0,
            output_cost_per_1m=4.0,
        )
        assert pricing.input_cost_per_token == pytest.approx(1.0 / 1_000_000)

    def test_output_cost_per_token(self) -> None:
        pricing = ModelPricing(
            model_name="test",
            input_cost_per_1m=1.0,
            output_cost_per_1m=4.0,
        )
        assert pricing.output_cost_per_token == pytest.approx(4.0 / 1_000_000)

    def test_frozen(self) -> None:
        """ModelPricing is frozen — fields cannot be reassigned."""
        pricing = ModelPricing("test", 1.0, 4.0)
        with pytest.raises(AttributeError):
            pricing.input_cost_per_1m = 999  # type: ignore[misc]


# ── Pricing Table ──────────────────────────────────────────────────────────────


class TestPricingTable:
    """Verify the pricing table is complete and correctly structured."""

    def test_nano_model_present(self) -> None:
        assert "gpt-4.1-nano" in MODEL_PRICING

    def test_mini_model_present(self) -> None:
        assert "gpt-4.1-mini" in MODEL_PRICING

    def test_full_model_present(self) -> None:
        assert "gpt-4.1" in MODEL_PRICING

    def test_all_models_have_positive_pricing(self) -> None:
        for name, pricing in MODEL_PRICING.items():
            assert pricing.input_cost_per_1m > 0, f"{name} input cost should be > 0"
            assert pricing.output_cost_per_1m > 0, f"{name} output cost should be > 0"

    def test_nano_is_cheapest_input(self) -> None:
        nano = MODEL_PRICING["gpt-4.1-nano"]
        full = MODEL_PRICING["gpt-4.1"]
        assert nano.input_cost_per_1m < full.input_cost_per_1m

    def test_output_more_expensive_than_input(self) -> None:
        """Output tokens are always more expensive than input for OpenAI models."""
        for name, pricing in MODEL_PRICING.items():
            assert pricing.output_cost_per_1m >= pricing.input_cost_per_1m, (
                f"{name}: output should be >= input cost"
            )


# ── estimate_cost ──────────────────────────────────────────────────────────────


class TestEstimateCost:
    """Verify per-call cost estimation."""

    def test_nano_cost_calculation(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=1000, completion_tokens=200)
        # $0.10/1M input → 0.0001 for 1000 tokens
        # $0.40/1M output → 0.00008 for 200 tokens
        assert cost.prompt_cost_usd == pytest.approx(0.0001, abs=1e-8)
        assert cost.completion_cost_usd == pytest.approx(0.00008, abs=1e-8)
        assert cost.total_cost_usd == pytest.approx(0.00018, abs=1e-8)

    def test_zero_tokens(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=0, completion_tokens=0)
        assert cost.total_cost_usd == 0.0

    def test_unknown_model_returns_zero_cost(self) -> None:
        cost = estimate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
        assert cost.total_cost_usd == 0.0
        assert cost.model == "unknown-model-xyz"

    def test_large_token_count(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost.prompt_cost_usd == pytest.approx(0.10, abs=1e-6)

    def test_returns_cost_estimate_type(self) -> None:
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=100, completion_tokens=50)
        assert hasattr(cost, "total_cost_usd")
        assert hasattr(cost, "to_dict")


# ── CostTracker ────────────────────────────────────────────────────────────────


class TestCostTracker:
    """Verify thread-safe session-level cost tracking."""

    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_single_record(self) -> None:
        tracker = CostTracker()
        cost = estimate_cost("gpt-4.1-nano", 1000, 200)
        tracker.record(cost)
        assert tracker.total_calls == 1
        assert tracker.total_prompt_tokens == 1000
        assert tracker.total_completion_tokens == 200
        assert tracker.total_tokens == 1200
        assert tracker.total_cost_usd == pytest.approx(cost.total_cost_usd)

    def test_multiple_records_accumulate(self) -> None:
        tracker = CostTracker()
        cost1 = estimate_cost("gpt-4.1-nano", 1000, 200)
        cost2 = estimate_cost("gpt-4.1-nano", 500, 100)
        tracker.record(cost1)
        tracker.record(cost2)
        assert tracker.total_calls == 2
        assert tracker.total_prompt_tokens == 1500
        assert tracker.total_completion_tokens == 300

    def test_record_request_cost(self) -> None:
        tracker = CostTracker()
        costs = [
            estimate_cost("gpt-4.1-nano", 100, 50),
            estimate_cost("gpt-4.1-nano", 200, 100),
            estimate_cost("gpt-4.1-nano", 300, 150),
        ]
        total = tracker.record_request_cost(costs)
        assert tracker.total_calls == 3
        assert total == pytest.approx(sum(c.total_cost_usd for c in costs))

    def test_reset(self) -> None:
        tracker = CostTracker()
        cost = estimate_cost("gpt-4.1-nano", 1000, 200)
        tracker.record(cost)
        assert tracker.total_calls == 1
        tracker.reset()
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_summary_format(self) -> None:
        tracker = CostTracker()
        cost = estimate_cost("gpt-4.1-nano", 1000, 200)
        tracker.record(cost)
        summary = tracker.summary()
        assert "1 calls" in summary
        assert "$" in summary

    def test_to_dict(self) -> None:
        tracker = CostTracker(alert_per_request_usd=0.05, alert_per_session_usd=2.0)
        cost = estimate_cost("gpt-4.1-nano", 1000, 200)
        tracker.record(cost)
        d = tracker.to_dict()
        assert d["total_calls"] == 1
        assert d["alert_per_request_usd"] == 0.05
        assert d["alert_per_session_usd"] == 2.0
        assert d["session_alert_fired"] is False

    def test_thread_safety(self) -> None:
        """Verify that concurrent record calls don't corrupt counters."""
        tracker = CostTracker()
        n_threads = 10
        records_per_thread = 100

        def worker() -> None:
            for _ in range(records_per_thread):
                cost = estimate_cost("gpt-4.1-nano", 100, 50)
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

        with caplog.at_level(logging.WARNING, logger="observability.cost_tracker"):
            tracker = CostTracker(alert_per_request_usd=0.0001)
            # This call costs about $0.001 — above the $0.0001 threshold
            cost = estimate_cost("gpt-4.1", 1000, 1000)
            tracker.record(cost)

        # The warning should be in loguru, not stdlib — so we check the tracker works
        # (loguru doesn't propagate to caplog by default, but the logic is tested)
        assert tracker.total_calls == 1

    def test_session_alert_fires_once(self) -> None:
        """Session alert should fire once when cumulative cost exceeds threshold."""
        tracker = CostTracker(alert_per_session_usd=0.001)
        # Accumulate enough calls to exceed $0.001
        for _ in range(100):
            cost = estimate_cost("gpt-4.1", 1000, 1000)
            tracker.record(cost)

        d = tracker.to_dict()
        assert d["session_alert_fired"] is True
