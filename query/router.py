# query/router.py
"""
Query Router — classifies incoming questions before they enter the pipeline.

Intent classification determines the routing tier:

  FINANCIAL_SPECIFIC   : Ticker + metric + period detected → full pipeline (L2-L5)
  FINANCIAL_GENERAL    : Financial domain but no specific entity → L2 + L3 + L4
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
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from config import settings as _settings
from config.llm_client import parse


def get_openai_client() -> Any:
    """Backward-compat shim for test mocking."""
    from config.openai_client import get_openai_client as _get

    return _get()


def _build_ticker_resolver() -> tuple[re.Pattern, dict[str, str]]:
    """
    Dynamically build ticker matching pattern and normalization mapping
    from registered companies and aliases in CompanyRegistry.
    """
    from config.companies import CompanyRegistry

    alias_map = CompanyRegistry.get_alias_map()
    mapping: dict[str, str] = {}
    patterns: set[str] = set()

    for phrase, ticker in alias_map.items():
        t_upper = ticker.upper()
        p_upper = phrase.upper()
        mapping[t_upper] = t_upper
        mapping[p_upper] = t_upper
        mapping[phrase] = t_upper
        patterns.add(re.escape(phrase))
        patterns.add(re.escape(t_upper))

    sorted_patterns = sorted(patterns, key=len, reverse=True)
    combined = "|".join(sorted_patterns)
    regex = re.compile(r"\b(" + combined + r")\b", re.IGNORECASE)
    return regex, mapping


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
        "sales",
        "strategy",
        "capex",
        "debt",
        "cash",
        "balance sheet",
        "subscribers",
        "subscription",
        "risk",
        "risks",
        "competition",
        "cost",
        "expenses",
        "spend",
    ]
)

_ROUTER_SYSTEM_PROMPT = """You are a query intent classifier for a financial RAG system that
answers questions about SEC earnings filings.

Classify the user query into one of these intents:

FINANCIAL_SPECIFIC: Question about a specific metric, company, or time period that
  requires looking up actual filing data. Examples: "What was Apple revenue Q4 2024?",
  "NVIDIA data center gross margin Q3?", "How did JPM net income change YoY?"

FINANCIAL_GENERAL: Financial domain question but conceptual/general, not about specific
  filing data. Examples: "What is EPS?", "How do I read a 10-K?", "Explain gross margin."

OUT_OF_SCOPE: Completely off-topic. Examples: "Write me a poem", "What is the weather?",
  "Tell me a joke", greetings, small talk.

AMBIGUOUS: Unclear whether the query needs RAG or not. Cannot confidently classify.
"""


class RouterDecisionSchema(BaseModel):
    intent: QueryIntent = Field(description="The classified intent of the user's query.")
    confidence: float = Field(description="Confidence in the classification from 0.0 to 1.0.")
    detected_ticker: str | None = Field(
        default=None, description="The ticker symbol if detected, else None."
    )
    reasoning: str = Field(description="A short one sentence explanation of the reasoning.")


_COMPARATIVE_KEYWORDS = frozenset(
    [
        "compare",
        "comparison",
        "compared",
        "change",
        "growth",
        "increased",
        "decreased",
        "versus",
        "vs",
        "yoy",
        "qoq",
        "difference",
        "trend",
        "between",
    ]
)


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
        detected_year   : Year detected in query (e.g. 2024), else None
        detected_quarter: Quarter detected in query (e.g. Q1), else None
        is_comparative  : True if query compares metrics across periods or years
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
    detected_year: int | None = None
    detected_quarter: str | None = None
    is_comparative: bool = False
    refuse_probability: float = 0.0

    @property
    def is_specific(self) -> bool:
        return self.intent == QueryIntent.FINANCIAL_SPECIFIC

    @property
    def is_general(self) -> bool:
        return self.intent == QueryIntent.FINANCIAL_GENERAL

    def summary(self) -> str:
        heuristic_tag = " [heuristic]" if self.used_heuristic else ""
        comparative_tag = " [comparative]" if self.is_comparative else ""
        return (
            f"intent={self.intent.value} confidence={self.confidence:.2f} "
            f"ticker={self.detected_ticker or 'none'} "
            f"skip_transform={self.skip_transform} refuse={self.should_refuse} "
            f"refuse_prob={self.refuse_probability:.2f} "
            f"latency={self.latency_ms:.0f}ms{heuristic_tag}{comparative_tag}"
        )


# Public alias for API consistency
RoutingResult = RoutingDecision


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
        self._model = _settings.query_router.model
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
            h_intent, h_conf, h_ticker, h_reason, h_year, h_quarter, h_comp = heuristic_result
            decision = self._build_decision(
                question_clean,
                intent=h_intent,
                confidence=h_conf,
                detected_ticker=h_ticker,
                reasoning=h_reason,
                latency_ms=latency_ms,
                used_heuristic=True,
                detected_year=h_year,
                detected_quarter=h_quarter,
                is_comparative=h_comp,
            )
            self._update_stats(decision)
            logger.debug(f"Router [heuristic] | {decision.summary()}")
            return decision

        llm_result = self._llm_classify(question_clean)
        latency_ms = (time.perf_counter() - t_start) * 1000

        detected_ticker = llm_result.get("detected_ticker")
        from query.fiscal_resolver import FiscalResolver

        resolved_period = FiscalResolver.resolve(question_clean, ticker=detected_ticker)
        all_years = re.findall(r"\b(202[0-9])\b", question_clean)
        is_comparative = (
            any(kw in question_clean.lower() for kw in _COMPARATIVE_KEYWORDS)
            or len(set(all_years)) > 1
        )
        detected_year = resolved_period.fiscal_year
        detected_quarter = resolved_period.quarter

        decision = self._build_decision(
            question_clean,
            intent=QueryIntent(llm_result.get("intent", "AMBIGUOUS")),
            confidence=float(llm_result.get("confidence", 0.5)),
            detected_ticker=detected_ticker,
            reasoning=llm_result.get("reasoning", "LLM classification"),
            latency_ms=latency_ms,
            used_heuristic=False,
            detected_year=detected_year,
            detected_quarter=detected_quarter,
            is_comparative=is_comparative,
        )
        self._update_stats(decision)
        logger.debug(f"Router [llm] | {decision.summary()}")
        return decision

    def _heuristic_classify(
        self, question: str
    ) -> tuple[QueryIntent, float, str | None, str, int | None, str | None, bool] | None:
        """
        Fast-path classification using regex and keyword matching.

        Returns a 7-tuple (intent, confidence, ticker, reasoning, detected_year,
        detected_quarter, is_comparative) if heuristics are conclusive, else None to fall through
        to LLM classification.
        """
        lower = question.lower()
        words = lower.split()

        if len(words) <= 2 and not any(kw in lower for kw in _FINANCIAL_KEYWORDS):
            return (
                QueryIntent.OUT_OF_SCOPE,
                0.9,
                None,
                "Too short and no financial keywords",
                None,
                None,
                False,
            )

        ticker_pattern, ticker_map = _build_ticker_resolver()
        ticker_match = ticker_pattern.search(question)
        has_financial_kw = any(kw in lower for kw in _FINANCIAL_KEYWORDS)

        greeting_patterns = ("hello", "hi ", "hey ", "thanks", "thank you", "what is your")
        if (
            any(lower.startswith(p) for p in greeting_patterns)
            and len(words) < 6
            and not ticker_match
            and not has_financial_kw
        ):
            return (
                QueryIntent.OUT_OF_SCOPE,
                0.95,
                None,
                "Greeting or small talk detected",
                None,
                None,
                False,
            )

        out_of_scope_topics = (
            "recipe",
            "weather",
            "poem",
            "joke",
            "cook",
            "cake",
            "cookie",
            "chocolate",
        )
        if any(t in lower for t in out_of_scope_topics) and not has_financial_kw:
            return (
                QueryIntent.OUT_OF_SCOPE,
                0.95,
                None,
                "Out-of-scope general topic detected",
                None,
                None,
                False,
            )

        if ticker_match and has_financial_kw:
            raw_match = ticker_match.group(0).upper()
            canonical = ticker_map.get(raw_match, raw_match)

            from query.fiscal_resolver import FiscalResolver

            resolved_period = FiscalResolver.resolve(question, ticker=canonical)
            all_years = re.findall(r"\b(202[0-9])\b", question)
            is_comparative = (
                any(kw in lower for kw in _COMPARATIVE_KEYWORDS) or len(set(all_years)) > 1
            )

            return (
                QueryIntent.FINANCIAL_SPECIFIC,
                0.92,
                canonical,
                f"Detected ticker {canonical} with financial keyword",
                resolved_period.fiscal_year,
                resolved_period.quarter,
                is_comparative,
            )

        return None

    def _llm_classify(self, question: str) -> dict:
        """Call the LLM for structured classification. Returns parsed JSON dict."""
        try:
            result = parse(
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                schema=RouterDecisionSchema,
                model=self._model,
                temperature=_settings.query_router.temperature,
                max_tokens=_settings.query_router.max_tokens,
            )
            return result.model_dump()
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
        question: str,
        intent: QueryIntent,
        confidence: float,
        detected_ticker: str | None,
        reasoning: str,
        latency_ms: float,
        used_heuristic: bool,
        detected_year: int | None = None,
        detected_quarter: str | None = None,
        is_comparative: bool = False,
    ) -> RoutingDecision:
        # Strict HyDE Gating (2026 SOTA Financial RAG Standard):
        # HyDE synthesizes a hypothetical SEC filing passage. For queries that are already
        # well-grounded or anchored to specific entities/periods/filings, HyDE is wasteful
        # (adds 1-2s latency) and harmful (causes hallucination drift from exact SEC numbers).
        # HyDE is skipped if ANY anchor or high description is present:
        lower_q = question.lower()
        has_filing_token = bool(
            re.search(
                r"\b(?:form\s*)?10[-‑]?[kq]\b|\b8[-‑]?k\b|\bannual\s+report\b|\bproxy\b", lower_q
            )
        )
        is_well_described = len(question.strip().split()) >= 6

        skip_hyde = (
            intent != QueryIntent.FINANCIAL_SPECIFIC
            or detected_ticker is not None
            or detected_year is not None
            or detected_quarter is not None
            or has_filing_token
            or is_well_described
        )

        refuse_prob = (
            confidence
            if (intent == QueryIntent.OUT_OF_SCOPE)
            else (round(max(0.0, 1.0 - confidence), 3) if intent == QueryIntent.AMBIGUOUS else 0.0)
        )

        return RoutingDecision(
            intent=intent,
            confidence=confidence,
            detected_ticker=detected_ticker,
            reasoning=reasoning,
            skip_hyde=skip_hyde,
            skip_transform=(intent in (QueryIntent.OUT_OF_SCOPE, QueryIntent.FINANCIAL_GENERAL)),
            should_refuse=(intent == QueryIntent.OUT_OF_SCOPE),
            latency_ms=latency_ms,
            used_heuristic=used_heuristic,
            detected_year=detected_year,
            detected_quarter=detected_quarter,
            is_comparative=is_comparative,
            refuse_probability=refuse_prob,
        )

    def _update_stats(self, decision: RoutingDecision) -> None:
        self._stats.total_routed += 1
        self._stats.total_latency_ms += decision.latency_ms
        self._stats.intent_counts[decision.intent.value] += 1
        if decision.used_heuristic:
            self._stats.heuristic_hits += 1
        else:
            self._stats.llm_calls += 1
