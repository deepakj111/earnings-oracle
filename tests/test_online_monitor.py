"""
Unit tests for evaluation/online_monitor.py (Layer 6b — Online Production Quality Monitor).
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.online_monitor import OnlineQualityMetrics, OnlineQualityMonitor


def _make_trace(
    latency: float = 1.2,
    answer: str = "Apple reported $94.9B revenue in Q4 2024 [1].",
    grounded: bool = True,
    retrieval_failed: bool = False,
    citation_count: int = 1,
) -> dict:
    return {
        "trace_id": "test_trace_123",
        "total_latency_seconds": latency,
        "generation": {
            "answer": answer,
            "grounded": grounded,
            "retrieval_failed": retrieval_failed,
            "citation_count": citation_count,
        },
    }


class TestOnlineQualityMonitor:
    def test_empty_traces_returns_safe_metrics(self) -> None:
        monitor = OnlineQualityMonitor()
        metrics = monitor.evaluate_traces([])
        assert metrics.sample_count == 0
        assert metrics.alert_triggered is False
        assert "No traces" in metrics.alerts[0]

    def test_healthy_traces_meet_all_slis(self) -> None:
        monitor = OnlineQualityMonitor()
        traces = [_make_trace() for _ in range(10)]
        metrics = monitor.evaluate_traces(traces)
        assert metrics.sample_count == 10
        assert metrics.grounded_rate == 1.0
        assert metrics.citation_coverage_rate == 1.0
        assert metrics.alert_triggered is False
        assert len(metrics.alerts) == 0

    def test_alert_triggered_on_low_grounded_rate(self) -> None:
        monitor = OnlineQualityMonitor(min_grounded_threshold=0.90)
        traces = [
            _make_trace(grounded=True),
            _make_trace(grounded=False, answer="I do not have sufficient information."),
        ]
        metrics = monitor.evaluate_traces(traces)
        assert metrics.sample_count == 2
        assert metrics.grounded_rate == 0.5
        assert metrics.alert_triggered is True
        assert any("Grounded rate" in a for a in metrics.alerts)

    def test_alert_triggered_on_latency_sla_breach(self) -> None:
        monitor = OnlineQualityMonitor(max_p95_latency_seconds=3.0)
        traces = [_make_trace(latency=5.5) for _ in range(10)]
        metrics = monitor.evaluate_traces(traces)
        assert metrics.alert_triggered is True
        assert any("p95 latency" in a for a in metrics.alerts)

    def test_evaluate_audit_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "audit.jsonl"
        traces = [_make_trace(latency=1.0 * (i + 1)) for i in range(5)]
        with open(log_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

        monitor = OnlineQualityMonitor()
        metrics = monitor.evaluate_audit_log(log_file, sample_size=3)
        assert metrics.sample_count == 3
        assert metrics.latency_p50 > 0
        summary_text = metrics.summary()
        assert "ONLINE PRODUCTION RAG QUALITY AUDIT REPORT" in summary_text

    def test_append_quality_metric(self, tmp_path: Path) -> None:
        out_file = tmp_path / "online_quality.jsonl"
        monitor = OnlineQualityMonitor()
        metrics = OnlineQualityMetrics(sample_count=5, grounded_rate=1.0)
        monitor.append_quality_metric(metrics, output_file=out_file)
        assert out_file.exists()
        with open(out_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["sample_count"] == 5
            assert data["grounded_rate"] == 1.0
