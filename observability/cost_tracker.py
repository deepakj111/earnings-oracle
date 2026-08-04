# observability/cost_tracker.py
"""Simplified LLM cost tracking for the Financial RAG system."""

from __future__ import annotations

import threading

from loguru import logger

from observability.trace_models import CostEstimate

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10 / 1_000_000, 0.40 / 1_000_000),
    "gpt-4.1": (2.50 / 1_000_000, 10.00 / 1_000_000),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate:
    """Calculate USD cost estimate for an LLM call."""
    if model in MODEL_PRICING:
        prompt_rate, completion_rate = MODEL_PRICING[model]
        prompt_cost = prompt_tokens * prompt_rate
        completion_cost = completion_tokens * completion_rate
        return CostEstimate(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost_usd=prompt_cost,
            completion_cost_usd=completion_cost,
        )

    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_calculator.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        logger.debug(f"Cost estimation fallback to 0.0 for model '{model}': {exc}")
        return CostEstimate(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost_usd=0.0,
            completion_cost_usd=0.0,
        )

    return CostEstimate(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cost_usd=prompt_cost,
        completion_cost_usd=completion_cost,
    )


class CostTracker:
    """Thread-safe session-level token and cost accumulator."""

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
        with self._lock:
            self._total_calls += 1
            self._total_prompt_tokens += cost.prompt_tokens
            self._total_completion_tokens += cost.completion_tokens
            self._total_cost_usd += cost.total_cost_usd

            if cost.total_cost_usd > self._alert_per_request:
                logger.warning(
                    f"Cost alert: call ${cost.total_cost_usd:.4f} > limit ${self._alert_per_request:.2f}"
                )

            if self._total_cost_usd > self._alert_per_session and not self._session_alert_fired:
                self._session_alert_fired = True
                logger.warning(
                    f"Cost alert: session ${self._total_cost_usd:.4f} > limit ${self._alert_per_session:.2f}"
                )

    def record_request_cost(self, costs: list[CostEstimate]) -> float:
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
        with self._lock:
            total_tokens = self._total_prompt_tokens + self._total_completion_tokens
            return (
                f"Session: {self._total_calls} calls | "
                f"{total_tokens:,} tokens | "
                f"${self._total_cost_usd:.4f}"
            )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "total_cost_usd": round(self._total_cost_usd, 6),
                "alert_per_request_usd": self._alert_per_request,
                "alert_per_session_usd": self._alert_per_session,
                "session_alert_fired": self._session_alert_fired,
            }

    def reset(self) -> None:
        with self._lock:
            self._total_calls = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost_usd = 0.0
            self._session_alert_fired = False
