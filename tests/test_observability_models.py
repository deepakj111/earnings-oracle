# tests/test_observability_models.py
"""
Tests for observability/trace_models.py — structured trace dataclasses.

Tests cover:
  - Construction and field defaults for all span types
  - Cost estimation calculations
  - JSON serialization round-trip
  - SpanStatus enum values
  - PipelineTrace aggregation and finalization
  - Latency and cost breakdown derived properties
"""

import json

import pytest

from observability.trace_models import (
    CostEstimate,
    CRAGSpan,
    GenerationSpan,
    LLMCallSpan,
    PipelineTrace,
    QueryTransformSpan,
    RetrievalSpan,
    SpanStatus,
)

# ── SpanStatus ─────────────────────────────────────────────────────────────────


class TestSpanStatus:
    """Verify SpanStatus enum values map correctly."""

    def test_ok_value(self) -> None:
        assert SpanStatus.OK.value == "ok"

    def test_error_value(self) -> None:
        assert SpanStatus.ERROR.value == "error"

    def test_degraded_value(self) -> None:
        assert SpanStatus.DEGRADED.value == "degraded"

    def test_string_comparison(self) -> None:
        assert SpanStatus.OK == "ok"

    def test_all_variants(self) -> None:
        assert len(SpanStatus) == 3


# ── CostEstimate ──────────────────────────────────────────────────────────────


class TestCostEstimate:
    """Verify cost calculation and serialization."""

    def test_total_cost(self) -> None:
        cost = CostEstimate(
            model="gpt-4.1-nano",
            prompt_tokens=1000,
            completion_tokens=200,
            prompt_cost_usd=0.0001,
            completion_cost_usd=0.00008,
        )
        assert cost.total_cost_usd == pytest.approx(0.00018, abs=1e-8)

    def test_zero_tokens(self) -> None:
        cost = CostEstimate(
            model="gpt-4.1-nano",
            prompt_tokens=0,
            completion_tokens=0,
            prompt_cost_usd=0.0,
            completion_cost_usd=0.0,
        )
        assert cost.total_cost_usd == 0.0

    def test_to_dict_contains_all_fields(self) -> None:
        cost = CostEstimate(
            model="gpt-4.1-nano",
            prompt_tokens=100,
            completion_tokens=50,
            prompt_cost_usd=0.00001,
            completion_cost_usd=0.00002,
        )
        d = cost.to_dict()
        assert "model" in d
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "prompt_cost_usd" in d
        assert "completion_cost_usd" in d
        assert "total_cost_usd" in d

    def test_to_dict_rounding(self) -> None:
        cost = CostEstimate(
            model="test",
            prompt_tokens=1,
            completion_tokens=1,
            prompt_cost_usd=0.00000123456789,
            completion_cost_usd=0.00000987654321,
        )
        d = cost.to_dict()
        # Should be rounded to 6 decimal places
        assert d["prompt_cost_usd"] == round(0.00000123456789, 6)


# ── LLMCallSpan ───────────────────────────────────────────────────────────────


class TestLLMCallSpan:
    """Verify LLM call span construction and serialization."""

    def test_defaults(self) -> None:
        span = LLMCallSpan()
        assert span.caller == ""
        assert span.model == ""
        assert span.prompt_tokens == 0
        assert span.completion_tokens == 0
        assert span.latency_seconds == 0.0
        assert span.status == SpanStatus.OK
        assert span.cost is None
        assert span.timestamp  # should have a default timestamp

    def test_total_tokens(self) -> None:
        span = LLMCallSpan(prompt_tokens=100, completion_tokens=50)
        assert span.total_tokens == 150

    def test_to_dict(self) -> None:
        span = LLMCallSpan(
            caller="query_transform/hyde",
            model="gpt-4.1-nano",
            prompt_tokens=200,
            completion_tokens=100,
            latency_seconds=0.456,
        )
        d = span.to_dict()
        assert d["caller"] == "query_transform/hyde"
        assert d["model"] == "gpt-4.1-nano"
        assert d["total_tokens"] == 300

    def test_span_id_is_unique(self) -> None:
        span1 = LLMCallSpan()
        span2 = LLMCallSpan()
        assert span1.span_id != span2.span_id

    def test_error_status(self) -> None:
        span = LLMCallSpan(
            status=SpanStatus.ERROR,
            error_message="Rate limit exceeded",
        )
        d = span.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] == "Rate limit exceeded"


# ── QueryTransformSpan ─────────────────────────────────────────────────────────


class TestQueryTransformSpan:
    """Verify query transform span construction."""

    def test_no_degradation(self) -> None:
        span = QueryTransformSpan(
            latency_seconds=0.85,
            techniques_attempted=["hyde", "multi_query", "stepback"],
            multi_query_count=4,
            hyde_generated=True,
            stepback_generated=True,
        )
        assert not span.is_degraded
        assert span.status == SpanStatus.OK

    def test_degraded_when_techniques_fail(self) -> None:
        span = QueryTransformSpan(
            techniques_failed=["hyde"],
        )
        assert span.is_degraded

    def test_cache_hit(self) -> None:
        span = QueryTransformSpan(cache_hit=True, latency_seconds=0.001)
        d = span.to_dict()
        assert d["cache_hit"] is True

    def test_to_dict_includes_is_degraded(self) -> None:
        span = QueryTransformSpan(techniques_failed=["multi_query"])
        d = span.to_dict()
        assert d["is_degraded"] is True


# ── RetrievalSpan ──────────────────────────────────────────────────────────────


class TestRetrievalSpan:
    """Verify retrieval span construction and serialization."""

    def test_defaults(self) -> None:
        span = RetrievalSpan()
        assert span.dense_candidates == 0
        assert span.bm25_candidates == 0
        assert span.reranked is False
        assert span.final_chunk_count == 0

    def test_with_source_distribution(self) -> None:
        span = RetrievalSpan(
            source_distribution={"dense": 3, "bm25": 1, "both": 1},
            final_chunk_count=5,
        )
        d = span.to_dict()
        assert d["source_distribution"]["dense"] == 3
        assert d["final_chunk_count"] == 5

    def test_score_rounding(self) -> None:
        span = RetrievalSpan(
            top_rrf_score=0.123456789,
            top_rerank_score=0.987654321,
        )
        d = span.to_dict()
        assert d["top_rrf_score"] == round(0.123456789, 6)
        assert d["top_rerank_score"] == round(0.987654321, 4)


# ── GenerationSpan ─────────────────────────────────────────────────────────────


class TestGenerationSpan:
    """Verify generation span construction and serialization."""

    def test_total_tokens(self) -> None:
        span = GenerationSpan(prompt_tokens=500, completion_tokens=200)
        assert span.total_tokens == 700

    def test_with_cost(self) -> None:
        cost = CostEstimate(
            model="gpt-4.1-nano",
            prompt_tokens=500,
            completion_tokens=200,
            prompt_cost_usd=0.00005,
            completion_cost_usd=0.00008,
        )
        span = GenerationSpan(cost=cost, model="gpt-4.1-nano")
        d = span.to_dict()
        assert d["cost"]["total_cost_usd"] == pytest.approx(0.00013, abs=1e-6)

    def test_retrieval_failed(self) -> None:
        span = GenerationSpan(retrieval_failed=True, grounded=False)
        d = span.to_dict()
        assert d["retrieval_failed"] is True
        assert d["grounded"] is False


# ── CRAGSpan ───────────────────────────────────────────────────────────────────


class TestCRAGSpan:
    """Verify CRAG span construction and serialization."""

    def test_correct_action(self) -> None:
        span = CRAGSpan(
            action="correct",
            relevance_ratio=0.8,
            chunks_graded=5,
            chunks_relevant=4,
        )
        d = span.to_dict()
        assert d["action"] == "correct"
        assert d["was_corrected"] is False

    def test_incorrect_action_with_web(self) -> None:
        span = CRAGSpan(
            action="incorrect",
            web_search_triggered=True,
            web_results_count=4,
            was_corrected=True,
        )
        d = span.to_dict()
        assert d["web_search_triggered"] is True
        assert d["was_corrected"] is True

    def test_disabled(self) -> None:
        span = CRAGSpan(enabled=False)
        d = span.to_dict()
        assert d["enabled"] is False


# ── PipelineTrace ──────────────────────────────────────────────────────────────


class TestPipelineTrace:
    """Verify root trace aggregation, serialization, and derived properties."""

    def _make_trace_with_spans(self) -> PipelineTrace:
        """Helper: create a PipelineTrace with all spans populated."""
        trace = PipelineTrace(question="What was AAPL revenue in Q4?")

        trace.query_transform = QueryTransformSpan(
            latency_seconds=0.85,
            multi_query_count=4,
            hyde_generated=True,
            stepback_generated=True,
        )
        trace.retrieval = RetrievalSpan(
            latency_seconds=0.45,
            total_unique_candidates=35,
            final_chunk_count=5,
            reranked=True,
        )
        trace.generation = GenerationSpan(
            latency_seconds=1.20,
            model="gpt-4.1-nano",
            prompt_tokens=2000,
            completion_tokens=400,
            citation_count=3,
            grounded=True,
        )

        # Add LLM calls
        trace.llm_calls.append(
            LLMCallSpan(
                caller="query_transform/hyde",
                model="gpt-4.1-nano",
                prompt_tokens=150,
                completion_tokens=200,
                latency_seconds=0.3,
                cost=CostEstimate(
                    model="gpt-4.1-nano",
                    prompt_tokens=150,
                    completion_tokens=200,
                    prompt_cost_usd=0.000015,
                    completion_cost_usd=0.00008,
                ),
            )
        )
        trace.llm_calls.append(
            LLMCallSpan(
                caller="generation",
                model="gpt-4.1-nano",
                prompt_tokens=2000,
                completion_tokens=400,
                latency_seconds=1.20,
                cost=CostEstimate(
                    model="gpt-4.1-nano",
                    prompt_tokens=2000,
                    completion_tokens=400,
                    prompt_cost_usd=0.0002,
                    completion_cost_usd=0.00016,
                ),
            )
        )
        return trace

    def test_trace_id_is_uuid(self) -> None:
        trace = PipelineTrace()
        assert len(trace.trace_id) == 36  # UUID4 length

    def test_timestamp_is_populated(self) -> None:
        trace = PipelineTrace()
        assert trace.timestamp  # non-empty ISO string

    def test_total_tokens_from_llm_calls(self) -> None:
        trace = self._make_trace_with_spans()
        trace.finalize()
        assert trace.total_prompt_tokens == 2150  # 150 + 2000
        assert trace.total_completion_tokens == 600  # 200 + 400
        assert trace.total_tokens == 2750

    def test_total_cost_from_llm_calls(self) -> None:
        trace = self._make_trace_with_spans()
        trace.finalize()
        expected = 0.000015 + 0.00008 + 0.0002 + 0.00016
        assert trace.total_cost_usd == pytest.approx(expected, abs=1e-8)

    def test_total_llm_calls(self) -> None:
        trace = self._make_trace_with_spans()
        assert trace.total_llm_calls == 2

    def test_latency_breakdown(self) -> None:
        trace = self._make_trace_with_spans()
        breakdown = trace.latency_breakdown
        assert "query_transform" in breakdown
        assert "retrieval" in breakdown
        assert "generation" in breakdown
        assert breakdown["query_transform"] == pytest.approx(0.85)

    def test_latency_breakdown_excludes_none_spans(self) -> None:
        trace = PipelineTrace()
        assert trace.latency_breakdown == {}

    def test_cost_breakdown_by_caller(self) -> None:
        trace = self._make_trace_with_spans()
        costs = trace.cost_breakdown
        assert "query_transform/hyde" in costs
        assert "generation" in costs

    def test_summary_format(self) -> None:
        trace = self._make_trace_with_spans()
        trace.finalize()
        trace.total_latency_seconds = 2.50
        summary = trace.summary()
        assert "trace=" in summary
        assert "latency=" in summary
        assert "cost=" in summary
        assert "status=" in summary

    def test_to_dict_json_round_trip(self) -> None:
        trace = self._make_trace_with_spans()
        trace.finalize()
        trace.total_latency_seconds = 2.50

        d = trace.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["question"] == "What was AAPL revenue in Q4?"
        assert parsed["total_llm_calls"] == 2
        assert len(parsed["llm_calls"]) == 2

    def test_to_json(self) -> None:
        trace = self._make_trace_with_spans()
        trace.finalize()
        trace.total_latency_seconds = 2.50

        json_str = trace.to_json()
        parsed = json.loads(json_str)
        assert "trace_id" in parsed
        assert "latency_breakdown" in parsed
        assert "cost_breakdown" in parsed

    def test_empty_trace(self) -> None:
        trace = PipelineTrace(question="test")
        trace.finalize()
        assert trace.total_tokens == 0
        assert trace.total_cost_usd == 0.0
        assert trace.total_llm_calls == 0

    def test_finalize_sets_degraded_on_error(self) -> None:
        trace = PipelineTrace()
        trace.llm_calls.append(
            LLMCallSpan(
                status=SpanStatus.ERROR,
                error_message="timeout",
            )
        )
        trace.finalize()
        assert trace.status == SpanStatus.DEGRADED

    def test_metadata_preserved(self) -> None:
        trace = PipelineTrace(
            question="test",
            metadata={"source": "api", "request_id": "abc123"},
        )
        d = trace.to_dict()
        assert d["metadata"]["source"] == "api"
        assert d["metadata"]["request_id"] == "abc123"
