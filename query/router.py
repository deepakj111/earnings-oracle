# query/router.py
"""
Query Router — classifies incoming questions before they enter the pipeline.

Intent classification determines the routing tier:

  FINANCIAL_SPECIFIC   : Ticker + metric + period detected → full pipeline (L2-L5)
  FINANCIAL_GENERAL    : Financial domain but no specific entity → L2 + L3 + L4 (skip CRAG)
  OUT_OF_SCOPE         : Non-financial question → short-circuit with refusal
  AMBIGUOUS            : Uncertain → route to full pipeline with a low-confidence flag

This prevents running 3 concurrent LLM calls (HyDE + multi-query + step-back)
on queries like "hello" or "what is a P/E ratio?" — a meaningful cost reduction
in production where 40-60% of queries to financial assistants are off-domain or
trivially answerable without retrieval.

Routing adds ~50-80ms (one cheap LLM call with structured JSON output).
The break-even point vs skipping routing is ~2 queries per 100 being OUT_OF_SCOPE.

Usage:
    from query.router import QueryRouter, RoutingDecision, QueryIntent

    router = QueryRouter()
    decision = router.route("What was Apple's revenue in Q4 2024?")
    print(decision.intent)           # QueryIntent.FINANCIAL_SPECIFIC
    print(decision.skip_hyde)        # False — run full L2 transform
    print(decision.detected_ticker)  # "AAPL"
    print(decision.confidence)       # 0.95
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from openai import OpenAI

from config import settings as _settings

_TICKER_PATTERN = re.compile(
    r"\b(AAPL|NVDA|MSFT|AMZN|META|JPM|XOM|UNH|TSLA|WMT|"
    r"Apple|NVIDIA|Microsoft|Amazon|Meta|JPMorgan|ExxonMobil|UnitedHealth|Tesla|Walmart)\b",
    re.IGNORECASE,
)

_FINANCIAL_KEYWORDS = frozenset(
    [
        "revenue",
        "earnings",
        "profit",
        "loss",
        "margin",
        "eps",
        "ebitda",
        "guidance",
        "outlook",
        "quarterly",
        "annual",
        "fiscal",
        "gross",
        "operating",
        "net income",
        "cash flow",
        "dividend",
        "buyback",
        "repurchase",
        "segment",
        "8-k",
        "10-k",
        "sec",
        "filing",
        "q1",
        "q2",
        "q3",
        "q4",
        "fy",
        "yoy",
        "qoq",
    ]
)

_ROUTER_SYSTEM_PROMPT = """You are a query intent classifier for a financial RAG system that
answers questions about SEC 8-K earnings filings for 10 companies:
AAPL, NVDA, MSFT, AMZN, META, JPM, XOM, UNH, TSLA, WMT.

Classify the user query into one of these intents:

FINANCIAL_SPECIFIC: Question about a specific metric, company, or time period that
  requires looking up actual filing data. Examples: "What was Apple revenue Q4 2024?",
  "NVIDIA data center gross margin Q3?", "How did JPM net income change YoY?"

FINANCIAL_GENERAL: Financial domain question but conceptual/general, not about specific
  filing data. Examples: "What is EPS?", "How do I read a 10-K?", "Explain gross margin."

OUT_OF_SCOPE: Completely off-topic. Examples: "Write me a poem", "What is the weather?",
  "Tell me a joke", greetings, small talk.

AMBIGUOUS: Unclear whether the query needs RAG or not. Cannot confidently classify.

Respond ONLY with valid JSON:
{
  "intent": "FINANCIAL_SPECIFIC" | "FINANCIAL_GENERAL" | "OUT_OF_SCOPE" | "AMBIGUOUS",
  "confidence": float between 0.0 and 1.0,
  "detected_ticker": "AAPL" | null,
  "reasoning": "one sentence explanation"
}"""


class QueryIntent(str, Enum):
    FINANCIAL_SPECIFIC = "FINANCIAL_SPECIFIC"
    FINANCIAL_GENERAL = "FINANCIAL_GENERAL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class RoutingDecision:
    """
    Result of routing classification for a single query.

    Attributes:
        intent          : Classified intent category
        confidence      : Model confidence [0, 1]
        detected_ticker : Ticker symbol if detected in query, else None
        reasoning       : One-sentence explanation from the classifier
        skip_hyde       : True when HyDE is wasteful (general/ambiguous queries)
        skip_transform  : True when full L2 transform should be skipped (e.g. general)
        should_refuse   : True when query is entirely out of scope
        latency_ms      : Time taken for routing decision in milliseconds
        used_heuristic  : True if heuristic fast-path was used (no LLM call)
    """

    intent: QueryIntent
    confidence: float
    detected_ticker: str | None
    reasoning: str
    skip_hyde: bool
    skip_transform: bool
    should_refuse: bool
    latency_ms: float
    used_heuristic: bool = False

    @property
    def is_specific(self) -> bool:
        return self.intent == QueryIntent.FINANCIAL_SPECIFIC

    @property
    def is_general(self) -> bool:
        return self.intent == QueryIntent.FINANCIAL_GENERAL

    def summary(self) -> str:
        heuristic_tag = " [heuristic]" if self.used_heuristic else ""
        return (
            f"intent={self.intent.value} confidence={self.confidence:.2f} "
            f"ticker={self.detected_ticker or 'none'} "
            f"skip_transform={self.skip_transform} refuse={self.should_refuse} "
            f"latency={self.latency_ms:.0f}ms{heuristic_tag}"
        )


@dataclass
class RouterStats:
    """Accumulated routing statistics — useful for dashboards and cost analysis."""

    total_routed: int = 0
    heuristic_hits: int = 0
    llm_calls: int = 0
    intent_counts: dict[str, int] = field(default_factory=lambda: {i.value: 0 for i in QueryIntent})
    total_latency_ms: float = 0.0

    @property
    def heuristic_hit_rate(self) -> float:
        if self.total_routed == 0:
            return 0.0
        return self.heuristic_hits / self.total_routed

    @property
    def avg_latency_ms(self) -> float:
        if self.total_routed == 0:
            return 0.0
        return self.total_latency_ms / self.total_routed


class QueryRouter:
    """
    Classifies incoming queries before they enter the RAG pipeline.

    Two-stage classification:
    1. Heuristic fast-path  : regex + keyword matching (~0ms, no LLM cost)
       - Detects obvious OUT_OF_SCOPE (very short queries, greetings, no financial terms)
       - Detects obvious FINANCIAL_SPECIFIC (ticker + financial keyword present)
    2. LLM fallback         : gpt-4.1-nano structured JSON (~50-80ms)
       - Called only when heuristics are inconclusive

    Thread-safe: OpenAI client is stateless after construction.
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=_settings.infra.openai_api_key)
        self._model = _settings.query_transform.model
        self._stats = RouterStats()
        logger.info(f"QueryRouter initialised | model={self._model}")

    @property
    def stats(self) -> RouterStats:
        return self._stats

    def route(self, question: str) -> RoutingDecision:
        """
        Classify a question and return a RoutingDecision.

        Args:
            question: Raw user query string

        Returns:
            RoutingDecision with intent, confidence, and routing flags
        """
        t_start = time.perf_counter()
        question_clean = question.strip()

        heuristic_result = self._heuristic_classify(question_clean)
        if heuristic_result is not None:
            latency_ms = (time.perf_counter() - t_start) * 1000
            decision = self._build_decision(
                intent=heuristic_result[0],
                confidence=heuristic_result[1],
                detected_ticker=heuristic_result[2],
                reasoning=heuristic_result[3],
                latency_ms=latency_ms,
                used_heuristic=True,
            )
            self._update_stats(decision)
            logger.debug(f"Router [heuristic] | {decision.summary()}")
            return decision

        llm_result = self._llm_classify(question_clean)
        latency_ms = (time.perf_counter() - t_start) * 1000

        decision = self._build_decision(
            intent=QueryIntent(llm_result.get("intent", "AMBIGUOUS")),
            confidence=float(llm_result.get("confidence", 0.5)),
            detected_ticker=llm_result.get("detected_ticker"),
            reasoning=llm_result.get("reasoning", "LLM classification"),
            latency_ms=latency_ms,
            used_heuristic=False,
        )
        self._update_stats(decision)
        logger.debug(f"Router [llm] | {decision.summary()}")
        return decision

    def _heuristic_classify(
        self, question: str
    ) -> tuple[QueryIntent, float, str | None, str] | None:
        """
        Fast-path classification using regex and keyword matching.

        Returns a 4-tuple (intent, confidence, ticker, reasoning) if heuristics
        are conclusive, else None to fall through to LLM classification.
        """
        lower = question.lower()
        words = lower.split()

        if len(words) <= 2 and not any(kw in lower for kw in _FINANCIAL_KEYWORDS):
            return QueryIntent.OUT_OF_SCOPE, 0.9, None, "Too short and no financial keywords"

        greeting_patterns = ("hello", "hi ", "hey ", "thanks", "thank you", "what is your")
        if any(lower.startswith(p) for p in greeting_patterns) and len(words) < 6:
            return QueryIntent.OUT_OF_SCOPE, 0.95, None, "Greeting or small talk detected"

        ticker_match = _TICKER_PATTERN.search(question)
        has_financial_kw = any(kw in lower for kw in _FINANCIAL_KEYWORDS)

        if ticker_match and has_financial_kw:
            raw_ticker = ticker_match.group(0).upper()
            ticker_map = {
                "APPLE": "AAPL",
                "NVIDIA": "NVDA",
                "MICROSOFT": "MSFT",
                "AMAZON": "AMZN",
                "JPMORGAN": "JPM",
                "EXXONMOBIL": "XOM",
                "UNITEDHEALTH": "UNH",
                "TESLA": "TSLA",
                "WALMART": "WMT",
            }
            canonical = ticker_map.get(raw_ticker, raw_ticker)
            return (
                QueryIntent.FINANCIAL_SPECIFIC,
                0.92,
                canonical,
                f"Detected ticker {canonical} with financial keyword",
            )

        return None

    def _llm_classify(self, question: str) -> dict:
        """Call the LLM for structured classification. Returns parsed JSON dict."""
        import json

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"Router LLM call failed ({exc}), defaulting to AMBIGUOUS")
            return {
                "intent": "AMBIGUOUS",
                "confidence": 0.5,
                "detected_ticker": None,
                "reasoning": f"LLM call failed: {exc}",
            }

    def _build_decision(
        self,
        intent: QueryIntent,
        confidence: float,
        detected_ticker: str | None,
        reasoning: str,
        latency_ms: float,
        used_heuristic: bool,
    ) -> RoutingDecision:
        return RoutingDecision(
            intent=intent,
            confidence=confidence,
            detected_ticker=detected_ticker,
            reasoning=reasoning,
            skip_hyde=(intent != QueryIntent.FINANCIAL_SPECIFIC),
            skip_transform=(intent in (QueryIntent.OUT_OF_SCOPE, QueryIntent.FINANCIAL_GENERAL)),
            should_refuse=(intent == QueryIntent.OUT_OF_SCOPE),
            latency_ms=latency_ms,
            used_heuristic=used_heuristic,
        )

    def _update_stats(self, decision: RoutingDecision) -> None:
        self._stats.total_routed += 1
        self._stats.total_latency_ms += decision.latency_ms
        self._stats.intent_counts[decision.intent.value] += 1
        if decision.used_heuristic:
            self._stats.heuristic_hits += 1
        else:
            self._stats.llm_calls += 1
