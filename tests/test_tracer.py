# tests/test_tracer.py
"""
Tests for observability/tracer.py — RAGTracer lifecycle and integration.

Tests cover:
  - Trace creation and finalization
  - Layer span recording
  - LLM call recording with automatic cost estimation
  - Convenience span builder methods
  - Trace persistence to disk
  - Trace loading from disk
  - Disabled tracer (no-op behavior)
  - Cost tracker session integration
"""

import json
from pathlib import Path

import pytest

from observability.trace_models import (
    CRAGSpan,
    GenerationSpan,
    LLMCallSpan,
    PipelineTrace,
    QueryTransformSpan,
    RetrievalSpan,
    SpanStatus,
)
from observability.tracer import RAGTracer

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def tracer(tmp_path: Path) -> RAGTracer:
    """Create a tracer that persists to a temp directory."""
    return RAGTracer(
        enabled=True,
        output_dir=str(tmp_path / "traces"),
        persist_traces=True,
        cost_alert_per_request_usd=1.00,
        cost_alert_per_session_usd=10.00,
    )


@pytest.fixture()
def disabled_tracer(tmp_path: Path) -> RAGTracer:
    """Create a disabled tracer."""
    return RAGTracer(
        enabled=False,
        output_dir=str(tmp_path / "traces"),
    )


# ── Trace lifecycle ────────────────────────────────────────────────────────────


class TestTraceLifecycle:
    """Verify trace creation, finalization, and persistence."""

    def test_start_trace_returns_pipeline_trace(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("What was AAPL revenue?")
        assert isinstance(trace, PipelineTrace)
        assert trace.question == "What was AAPL revenue?"
        assert trace.trace_id  # non-empty UUID

    def test_start_trace_with_metadata(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace(
            "test question",
            source="api",
            request_id="abc123",
        )
        assert trace.metadata["source"] == "api"
        assert trace.metadata["request_id"] == "abc123"

    def test_end_trace_sets_latency(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        tracer.end_trace(trace, total_latency=2.50)
        assert trace.total_latency_seconds == 2.50

    def test_end_trace_sets_status(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        tracer.end_trace(
            trace,
            total_latency=1.0,
            status=SpanStatus.ERROR,
            error_message="timeout",
        )
        assert trace.status == SpanStatus.ERROR
        assert trace.error_message == "timeout"

    def test_end_trace_calls_finalize(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        tracer.record_llm_call(
            trace,
            caller="test",
            model="gpt-4.1-nano",
            prompt_tokens=100,
            completion_tokens=50,
            latency_seconds=0.5,
        )
        tracer.end_trace(trace, total_latency=1.0)
        assert trace.total_prompt_tokens == 100
        assert trace.total_completion_tokens == 50
        assert trace.total_cost_usd > 0


# ── Disabled tracer ────────────────────────────────────────────────────────────


class TestDisabledTracer:
    """Verify disabled tracer is a complete no-op."""

    def test_start_trace_returns_empty_trace(self, disabled_tracer: RAGTracer) -> None:
        trace = disabled_tracer.start_trace("test")
        assert isinstance(trace, PipelineTrace)
        assert trace.question == "test"

    def test_record_spans_are_noop(self, disabled_tracer: RAGTracer) -> None:
        trace = disabled_tracer.start_trace("test")
        disabled_tracer.record_query_transform(trace, QueryTransformSpan(latency_seconds=0.5))
        # Span is NOT attached when disabled
        assert trace.query_transform is None

    def test_end_trace_skips_persistence(self, disabled_tracer: RAGTracer, tmp_path: Path) -> None:
        trace = disabled_tracer.start_trace("test")
        disabled_tracer.end_trace(trace, total_latency=1.0)
        trace_dir = tmp_path / "traces"
        if trace_dir.exists():
            assert len(list(trace_dir.glob("*.json"))) == 0


# ── Layer span recording ──────────────────────────────────────────────────────


class TestLayerSpanRecording:
    """Verify spans are correctly attached to traces."""

    def test_record_query_transform(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = QueryTransformSpan(
            latency_seconds=0.85,
            multi_query_count=4,
            hyde_generated=True,
            stepback_generated=True,
        )
        tracer.record_query_transform(trace, span)
        assert trace.query_transform is span
        assert trace.query_transform.multi_query_count == 4

    def test_record_retrieval(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = RetrievalSpan(
            latency_seconds=0.45,
            total_unique_candidates=35,
            final_chunk_count=5,
            reranked=True,
        )
        tracer.record_retrieval(trace, span)
        assert trace.retrieval is span
        assert trace.retrieval.final_chunk_count == 5

    def test_record_generation(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = GenerationSpan(
            latency_seconds=1.20,
            model="gpt-4.1-nano",
            prompt_tokens=2000,
            completion_tokens=400,
        )
        tracer.record_generation(trace, span)
        assert trace.generation is span
        assert trace.generation.total_tokens == 2400

    def test_record_crag(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = CRAGSpan(
            action="correct",
            relevance_ratio=0.8,
            was_corrected=False,
        )
        tracer.record_crag(trace, span)
        assert trace.crag is span
        assert trace.crag.action == "correct"


# ── LLM call recording ────────────────────────────────────────────────────────


class TestLLMCallRecording:
    """Verify LLM call spans with auto cost estimation."""

    def test_record_llm_call_appends_to_trace(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = tracer.record_llm_call(
            trace,
            caller="query_transform/hyde",
            model="gpt-4.1-nano",
            prompt_tokens=150,
            completion_tokens=200,
            latency_seconds=0.3,
        )
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].caller == "query_transform/hyde"
        assert isinstance(span, LLMCallSpan)

    def test_auto_cost_estimation(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = tracer.record_llm_call(
            trace,
            caller="generation",
            model="gpt-4.1-nano",
            prompt_tokens=2000,
            completion_tokens=400,
            latency_seconds=1.2,
        )
        assert span.cost is not None
        assert span.cost.total_cost_usd > 0
        assert span.cost.model == "gpt-4.1-nano"

    def test_multiple_calls(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        tracer.record_llm_call(trace, "hyde", "gpt-4.1-nano", 100, 50, 0.3)
        tracer.record_llm_call(trace, "multi", "gpt-4.1-nano", 100, 80, 0.4)
        tracer.record_llm_call(trace, "generation", "gpt-4.1-nano", 2000, 400, 1.2)
        assert len(trace.llm_calls) == 3

    def test_error_llm_call(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        span = tracer.record_llm_call(
            trace,
            caller="generation",
            model="gpt-4.1-nano",
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=5.0,
            status=SpanStatus.ERROR,
            error_message="RateLimitError",
        )
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "RateLimitError"


# ── Convenience span builders ──────────────────────────────────────────────────


class TestSpanBuilders:
    """Verify static convenience methods for building spans."""

    def test_build_query_transform_span(self) -> None:
        span = RAGTracer.build_query_transform_span(
            latency=0.85,
            cache_hit=False,
            multi_query_count=4,
            hyde_generated=True,
            stepback_generated=True,
            failed_techniques=[],
        )
        assert span.latency_seconds == 0.85
        assert span.multi_query_count == 4
        assert span.status == SpanStatus.OK

    def test_build_query_transform_span_degraded(self) -> None:
        span = RAGTracer.build_query_transform_span(
            latency=0.85,
            cache_hit=False,
            multi_query_count=1,
            hyde_generated=False,
            stepback_generated=True,
            failed_techniques=["hyde"],
        )
        assert span.status == SpanStatus.DEGRADED
        assert span.is_degraded

    def test_build_retrieval_span(self) -> None:
        # Create mock results
        class MockResult:
            source = "dense"
            rrf_score = 0.05
            rerank_score = 0.92

        results = [MockResult(), MockResult()]
        span = RAGTracer.build_retrieval_span(
            latency=0.45,
            total_candidates=35,
            final_count=5,
            reranked=True,
            reranker_model="ms-marco-MiniLM-L-12-v2",
            results=results,
        )
        assert span.latency_seconds == 0.45
        assert span.total_unique_candidates == 35
        assert span.source_distribution == {"dense": 2}

    def test_build_generation_span(self) -> None:
        span = RAGTracer.build_generation_span(
            latency=1.20,
            model="gpt-4.1-nano",
            prompt_tokens=2000,
            completion_tokens=400,
            context_chunks=5,
            context_tokens=2500,
            citation_count=3,
            grounded=True,
            retrieval_failed=False,
        )
        assert span.latency_seconds == 1.20
        assert span.cost is not None
        assert span.cost.model == "gpt-4.1-nano"

    def test_build_crag_span(self) -> None:
        span = RAGTracer.build_crag_span(
            latency=0.60,
            enabled=True,
            action="ambiguous",
            relevance_ratio=0.4,
            chunks_graded=5,
            chunks_relevant=2,
            web_search_triggered=True,
            web_results_count=4,
            was_corrected=True,
        )
        assert span.action == "ambiguous"
        assert span.was_corrected is True
        assert span.web_results_count == 4


# ── Persistence ────────────────────────────────────────────────────────────────


class TestTracePersistence:
    """Verify trace JSON file writing and reading."""

    def test_end_trace_persists_to_disk(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test question")
        tracer.record_llm_call(trace, "test", "gpt-4.1-nano", 100, 50, 0.5)
        tracer.end_trace(trace, total_latency=1.0)

        trace_dir = Path(tracer._output_dir)
        assert trace_dir.exists()
        files = list(trace_dir.glob("trace_*.json"))
        assert len(files) == 1

        # Verify content is valid JSON
        with open(files[0]) as f:
            data = json.load(f)
        assert data["question"] == "test question"
        assert data["total_llm_calls"] == 1

    def test_get_persisted_traces(self, tracer: RAGTracer) -> None:
        # Create multiple traces
        for i in range(3):
            trace = tracer.start_trace(f"question {i}")
            tracer.record_llm_call(trace, "test", "gpt-4.1-nano", 100, 50, 0.5)
            tracer.end_trace(trace, total_latency=1.0)

        traces = tracer.get_persisted_traces(limit=10)
        assert len(traces) == 3
        # Should be sorted newest first
        for t in traces:
            assert "trace_id" in t

    def test_get_persisted_traces_limit(self, tracer: RAGTracer) -> None:
        for i in range(5):
            trace = tracer.start_trace(f"question {i}")
            tracer.end_trace(trace, total_latency=1.0)

        traces = tracer.get_persisted_traces(limit=2)
        assert len(traces) == 2

    def test_get_persisted_traces_empty_dir(self, tmp_path: Path) -> None:
        tracer = RAGTracer(
            enabled=True,
            output_dir=str(tmp_path / "nonexistent"),
            persist_traces=False,
        )
        traces = tracer.get_persisted_traces()
        assert traces == []

    def test_no_persist_when_disabled(self, tmp_path: Path) -> None:
        tracer = RAGTracer(
            enabled=True,
            output_dir=str(tmp_path / "traces"),
            persist_traces=False,
        )
        trace = tracer.start_trace("test")
        tracer.end_trace(trace, total_latency=1.0)
        trace_dir = tmp_path / "traces"
        if trace_dir.exists():
            assert len(list(trace_dir.glob("*.json"))) == 0


# ── Cost tracker integration ──────────────────────────────────────────────────


class TestCostTrackerIntegration:
    """Verify tracer correctly accumulates costs across requests."""

    def test_cost_accumulates_across_traces(self, tracer: RAGTracer) -> None:
        for _ in range(3):
            trace = tracer.start_trace("test")
            tracer.record_llm_call(trace, "gen", "gpt-4.1-nano", 1000, 200, 1.0)
            tracer.end_trace(trace, total_latency=1.5)

        assert tracer.cost_tracker.total_calls == 3
        assert tracer.cost_tracker.total_prompt_tokens == 3000
        assert tracer.cost_tracker.total_cost_usd > 0

    def test_cost_summary(self, tracer: RAGTracer) -> None:
        trace = tracer.start_trace("test")
        tracer.record_llm_call(trace, "gen", "gpt-4.1-nano", 1000, 200, 1.0)
        tracer.end_trace(trace, total_latency=1.0)

        summary = tracer.cost_summary()
        assert "1 calls" in summary
        assert "$" in summary


# ── Full pipeline trace simulation ─────────────────────────────────────────────


class TestFullPipelineTraceSimulation:
    """End-to-end simulation of a complete pipeline trace."""

    def test_full_pipeline_trace(self, tracer: RAGTracer) -> None:
        """Simulate all layers of the pipeline being traced."""
        trace = tracer.start_trace("What was Apple's Q4 2024 revenue?")

        # L2: Query Transform
        tracer.record_query_transform(
            trace,
            RAGTracer.build_query_transform_span(
                latency=0.85,
                cache_hit=False,
                multi_query_count=4,
                hyde_generated=True,
                stepback_generated=True,
                failed_techniques=[],
            ),
        )
        # Record individual LLM calls from L2
        tracer.record_llm_call(trace, "query_transform/hyde", "gpt-4.1-nano", 150, 200, 0.3)
        tracer.record_llm_call(trace, "query_transform/multi", "gpt-4.1-nano", 120, 180, 0.35)
        tracer.record_llm_call(trace, "query_transform/stepback", "gpt-4.1-nano", 100, 80, 0.2)

        # L3: Retrieval
        tracer.record_retrieval(
            trace,
            RAGTracer.build_retrieval_span(
                latency=0.45,
                total_candidates=35,
                final_count=5,
                reranked=True,
                reranker_model="ms-marco-MiniLM-L-12-v2",
                results=[],
            ),
        )

        # L4: Generation
        tracer.record_generation(
            trace,
            RAGTracer.build_generation_span(
                latency=1.20,
                model="gpt-4.1-nano",
                prompt_tokens=2800,
                completion_tokens=350,
                context_chunks=5,
                context_tokens=2500,
                citation_count=3,
                grounded=True,
                retrieval_failed=False,
            ),
        )
        tracer.record_llm_call(trace, "generation", "gpt-4.1-nano", 2800, 350, 1.2)

        # Finalize
        tracer.end_trace(trace, total_latency=2.50)

        # Verify aggregates
        assert trace.total_llm_calls == 4
        assert trace.total_prompt_tokens == 150 + 120 + 100 + 2800
        assert trace.total_completion_tokens == 200 + 180 + 80 + 350
        assert trace.total_cost_usd > 0
        assert trace.status == SpanStatus.OK

        # Verify latency breakdown
        breakdown = trace.latency_breakdown
        assert breakdown["query_transform"] == pytest.approx(0.85)
        assert breakdown["retrieval"] == pytest.approx(0.45)
        assert breakdown["generation"] == pytest.approx(1.20)

        # Verify JSON roundtrip
        json_str = trace.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_llm_calls"] == 4
        assert "cost_breakdown" in parsed
