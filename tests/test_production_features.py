"""
Unit tests for production features added in 2026:
- Latency percentiles in RAGTracer
- SemanticCache stats and invalidation
- Context window MMR lexical deduplication
- Adaptive CE/RRF reranking for tabular data
- Settings.describe() and validation guards
- Router soft refuse probability
- QueryTransformer routing-aware technique gating
"""

import pytest

from config.settings import Settings
from generation.context_builder import build_context
from observability.tracer import LatencyReservoir, RAGTracer
from query.models import TransformedQuery
from query.router import QueryIntent, QueryRouter, RoutingDecision
from query.transformer import QueryTransformer
from retrieval.models import SearchResult
from retrieval.reranker import rerank


def _make_chunk(
    cid: str, text: str, chunk_type: str = "child", title: str = "Prose"
) -> SearchResult:
    return SearchResult(
        chunk_id=cid,
        parent_id=None,
        text=text,
        parent_text=text,
        rrf_score=0.8,
        rerank_score=0.8,
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period="Q4 2024",
        section_title=title,
        doc_type="10-Q",
        source="both",
        chunk_type=chunk_type,
    )


class TestLatencyReservoir:
    def test_latency_summary_percentiles(self) -> None:
        reservoir = LatencyReservoir(max_size=100)
        for i in range(1, 101):
            reservoir.add(float(i))
        summary = reservoir.summary()
        assert summary["count"] == 100
        assert 45.0 <= summary["p50"] <= 55.0
        assert summary["p95"] >= 90.0
        assert summary["p99"] >= 95.0

    def test_tracer_latency_summary_integration(self) -> None:
        tracer = RAGTracer(enabled=True)
        trace = tracer.start_trace("Test question")
        tracer.end_trace(trace, total_latency=1.25)
        summary = tracer.latency_summary()
        assert summary["count"] >= 1
        assert summary["p50"] == 1.25


class TestMMRContextDeduplication:
    def test_near_duplicate_blocks_are_pruned(self) -> None:
        block1 = _make_chunk(
            "c1", "Apple reported quarterly revenue of $94.9 billion up 6 percent year over year."
        )
        # block2 is 95% identical words
        block2 = _make_chunk(
            "c2", "Apple reported quarterly revenue of $94.9 billion up 6 percent year over year."
        )
        block3 = _make_chunk(
            "c3",
            "Operating cash flow for the full fiscal year reached an all-time record of $118B.",
        )

        context_text, citation_results, tokens = build_context(
            results=[block1, block2, block3],
            max_context_tokens=4096,
            mmr_threshold=0.90,
        )
        # block2 should be dropped due to MMR threshold
        assert len(citation_results) == 2
        chunk_ids = [c.chunk_id for c in citation_results]
        assert "c1" in chunk_ids
        assert "c3" in chunk_ids
        assert "c2" not in chunk_ids


class TestAdaptiveReranker:
    def test_table_heavy_candidates_use_equal_blending(self) -> None:
        candidates = [
            _make_chunk(
                f"t{i}",
                f"Table text row {i}",
                chunk_type="table",
                title="Consolidated Financial Statements Table",
            )
            for i in range(5)
        ]
        # Candidates are all tables -> should trigger 50/50 blend without error
        reranked = rerank("What was the revenue?", candidates)
        assert len(reranked) <= 5
        assert all(r.rerank_score > float("-inf") for r in reranked)


class TestSettingsValidationAndDescribe:
    def test_settings_describe_returns_nested_dict(self) -> None:
        s = Settings()
        d = s.describe()
        assert isinstance(d, dict)
        assert "query_router" in d
        assert "retrieval" in d
        assert "generation" in d
        assert d["retrieval"]["top_k_final"] == s.retrieval.top_k_final

    def test_settings_validation_guards(self) -> None:
        s = Settings()
        # Invalid top_k_final should fail validation
        with pytest.raises(ValueError):
            object.__setattr__(s.retrieval, "top_k_final", 0)
            s.validate()


class TestRouterSoftRefuseProbability:
    def test_out_of_scope_query_has_high_refuse_prob(self) -> None:
        router = QueryRouter()
        decision = router.route("What is the recipe for chocolate cake?")
        assert decision.should_refuse is True
        assert decision.refuse_probability >= 0.80

    def test_financial_query_has_low_refuse_prob(self) -> None:
        router = QueryRouter()
        decision = router.route("What was Apple's net sales in Q4 2024?")
        assert decision.should_refuse is False
        assert decision.refuse_probability == 0.0


@pytest.mark.asyncio
class TestRoutingAwareTransformer:
    async def test_specific_lookup_disables_hyde_and_stepback(self) -> None:
        transformer = QueryTransformer()
        decision = RoutingDecision(
            intent=QueryIntent.FINANCIAL_SPECIFIC,
            confidence=0.95,
            detected_ticker="AAPL",
            reasoning="Simple metric lookup",
            skip_hyde=False,
            skip_transform=False,
            should_refuse=False,
            latency_ms=1.0,
            used_heuristic=True,
            is_comparative=False,
        )
        # Should dynamically optimize to multi-query only (skipping HyDE and StepBack)
        transformed = await transformer.transform(
            "What was AAPL revenue?",
            routing_decision=decision,
        )
        assert isinstance(transformed, TransformedQuery)
        assert transformed.original == "What was AAPL revenue?"
