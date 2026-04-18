# observability/cost_tracker.py
"""
LLM cost tracking for the Financial RAG system.

Provides:
  - Model pricing table (input/output cost per 1M tokens) for all OpenAI models
    used by the pipeline, updated April 2026
  - Per-call cost calculation from token counts
  - Session-level cost accumulation with thread-safe counters
  - Cost guard: configurable per-request and per-session cost limits that
    log warnings (but never block) when exceeded

Design decisions:
  - Prices are per-1M-tokens following OpenAI's standard billing format
  - Thread-safety via threading.Lock for production multi-threaded API servers
  - Fail-open: unknown models default to zero cost with a warning, never crash
  - Immutable ModelPricing dataclass — pricing updates require code changes,
    which is intentional for auditability in production
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from loguru import logger

from observability.trace_models import CostEstimate


@dataclass(frozen=True)
class ModelPricing:
    """
    Pricing for a single OpenAI model.

    All costs are in USD per 1,000,000 tokens.
    Source: https://openai.com/api/pricing/ (as of April 2026)
    """

    model_name: str
    input_cost_per_1m: float
    output_cost_per_1m: float

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_1m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_1m / 1_000_000


# ── Pricing table ──────────────────────────────────────────────────────────────
# Models actively used by this pipeline, ordered by cost tier.
# Update these when OpenAI publishes new pricing.

MODEL_PRICING: dict[str, ModelPricing] = {
    # gpt-4.1-nano — primary model for query transform, grading, eval
    "gpt-4.1-nano": ModelPricing(
        model_name="gpt-4.1-nano",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
    ),
    # gpt-4.1-mini — mid-tier fallback
    "gpt-4.1-mini": ModelPricing(
        model_name="gpt-4.1-mini",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.60,
    ),
    # gpt-4.1 — higher-tier for generation if configured
    "gpt-4.1": ModelPricing(
        model_name="gpt-4.1",
        input_cost_per_1m=2.00,
        output_cost_per_1m=8.00,
    ),
    # gpt-4o-mini — legacy model, may still appear in some configs
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    ),
    # o4-mini — reasoning model
    "o4-mini": ModelPricing(
        model_name="o4-mini",
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
    ),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate:
    """
    Calculate cost estimate for a single LLM call.

    Args:
        model           : model name string (e.g. "gpt-4.1-nano")
        prompt_tokens   : input token count
        completion_tokens: output token count

    Returns:
        CostEstimate with per-component and total USD costs.
        Unknown models default to $0 with a logged warning.
    """
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        logger.warning(
            f"Unknown model '{model}' — cost estimate will be $0. "
            f"Add pricing to observability/cost_tracker.py."
        )
        return CostEstimate(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost_usd=0.0,
            completion_cost_usd=0.0,
        )

    prompt_cost = prompt_tokens * pricing.input_cost_per_token
    completion_cost = completion_tokens * pricing.output_cost_per_token

    return CostEstimate(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cost_usd=prompt_cost,
        completion_cost_usd=completion_cost,
    )


class CostTracker:
    """
    Thread-safe session-level cost accumulator.

    Tracks cumulative token usage and cost across multiple pipeline calls.
    Emits log warnings when configurable cost thresholds are exceeded.

    Usage:
        tracker = CostTracker(alert_per_request=0.10, alert_per_session=5.00)

        # After each pipeline call:
        tracker.record(cost_estimate)

        print(tracker.summary())  # "Session: 42 calls | 125,000 tokens | $0.0234"
    """

    def __init__(
        self,
        alert_per_request_usd: float = 0.10,
        alert_per_session_usd: float = 5.00,
    ) -> None:
        self._lock = threading.Lock()
        self._total_calls: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._alert_per_request = alert_per_request_usd
        self._alert_per_session = alert_per_session_usd
        self._session_alert_fired: bool = False

    def record(self, cost: CostEstimate) -> None:
        """
        Record a cost estimate from a single LLM call.
        Thread-safe — can be called from concurrent request handlers.
        """
        with self._lock:
            self._total_calls += 1
            self._total_prompt_tokens += cost.prompt_tokens
            self._total_completion_tokens += cost.completion_tokens
            self._total_cost_usd += cost.total_cost_usd

            # Per-request alert
            if cost.total_cost_usd > self._alert_per_request:
                logger.warning(
                    f"Cost alert: single LLM call cost ${cost.total_cost_usd:.4f} "
                    f"exceeds per-request threshold ${self._alert_per_request:.2f} | "
                    f"model={cost.model} "
                    f"tokens={cost.prompt_tokens}+{cost.completion_tokens}"
                )

            # Session alert (fires once)
            if self._total_cost_usd > self._alert_per_session and not self._session_alert_fired:
                self._session_alert_fired = True
                logger.warning(
                    f"Cost alert: cumulative session cost ${self._total_cost_usd:.4f} "
                    f"exceeds session threshold ${self._alert_per_session:.2f} | "
                    f"total_calls={self._total_calls}"
                )

    def record_request_cost(self, costs: list[CostEstimate]) -> float:
        """
        Record all LLM costs from a single pipeline request.

        Args:
            costs: list of CostEstimate objects from a PipelineTrace.llm_calls

        Returns:
            Total cost of this request in USD.
        """
        request_total = 0.0
        for cost in costs:
            self.record(cost)
            request_total += cost.total_cost_usd
        return request_total

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._total_calls

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    def summary(self) -> str:
        """Human-readable session cost summary."""
        with self._lock:
            return (
                f"Session: {self._total_calls} calls | "
                f"{self._total_prompt_tokens + self._total_completion_tokens:,} tokens | "
                f"${self._total_cost_usd:.4f}"
            )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_tokens": (self._total_prompt_tokens + self._total_completion_tokens),
                "total_cost_usd": round(self._total_cost_usd, 6),
                "alert_per_request_usd": self._alert_per_request,
                "alert_per_session_usd": self._alert_per_session,
                "session_alert_fired": self._session_alert_fired,
            }

    def reset(self) -> None:
        """Reset all counters. Useful between evaluation runs or tests."""
        with self._lock:
            self._total_calls = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost_usd = 0.0
            self._session_alert_fired = False
