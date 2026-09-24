"""
Layer 6b — Online Production Quality Monitor (2026 SOTA LLMOps).

Continuously audits live production RAG queries from structured audit logs
(data/audit_logs/audit.jsonl). Computes rolling faithfulness proxies,
citation integrity rates, refusal rates, and latency percentiles.

Alerts ML engineering when real-world production metrics breach quality SLIs
(e.g., faithfulness proxy < 0.90 or citation coverage < 0.95).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class OnlineQualityMetrics:
    """Summary metrics of online production query quality."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_count: int = 0
    grounded_rate: float = 0.0
    citation_coverage_rate: float = 0.0
    mean_citations_per_answer: float = 0.0
    faithfulness_proxy_score: float = 0.0
    refusal_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    alert_triggered: bool = False
    alerts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = (
            "ALERT: QUALITY THRESHOLD BREACHED" if self.alert_triggered else "HEALTHY: ALL SLIs MET"
        )
        lines = [
            f"\n{'=' * 60}",
            "   ONLINE PRODUCTION RAG QUALITY AUDIT REPORT",
            f"{'=' * 60}",
            f"Status                     : {status}",
            f"Timestamp                  : {self.timestamp}",
            f"Evaluated Samples          : {self.sample_count}",
            f"Grounded Rate              : {self.grounded_rate * 100:.1f}% (SLI >= 90.0%)",
            f"Citation Coverage Rate     : {self.citation_coverage_rate * 100:.1f}% (SLI >= 95.0%)",
            f"Faithfulness Proxy Score   : {self.faithfulness_proxy_score:.3f} (SLI >= 0.880)",
            f"Refusal Rate               : {self.refusal_rate * 100:.1f}%",
            f"Latency p50 / p95          : {self.latency_p50:.2f}s / {self.latency_p95:.2f}s",
            f"Alerts Triggered           : {len(self.alerts)}",
        ]
        if self.alerts:
            lines.append("Alert Details:")
            for a in self.alerts:
                lines.append(f"  - [CRITICAL] {a}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class OnlineQualityMonitor:
    """
    Production quality auditor for live financial RAG query streams.
    """

    def __init__(
        self,
        min_grounded_threshold: float = 0.90,
        min_citation_coverage_threshold: float = 0.95,
        min_faithfulness_proxy_threshold: float = 0.88,
        max_p95_latency_seconds: float = 4.0,
    ) -> None:
        self.min_grounded_threshold = min_grounded_threshold
        self.min_citation_coverage_threshold = min_citation_coverage_threshold
        self.min_faithfulness_proxy_threshold = min_faithfulness_proxy_threshold
        self.max_p95_latency_seconds = max_p95_latency_seconds

    def evaluate_traces(self, traces: list[dict[str, Any]]) -> OnlineQualityMetrics:
        """Evaluate a collection of parsed trace dictionaries."""
        if not traces:
            return OnlineQualityMetrics(
                sample_count=0,
                alert_triggered=False,
                alerts=["No traces provided for evaluation"],
            )

        grounded_count = 0
        with_citations_count = 0
        total_citations = 0
        refusal_count = 0
        latencies: list[float] = []
        faithfulness_proxies: list[float] = []

        for t in traces:
            lat = float(t.get("total_latency_seconds", 0.0))
            if lat > 0:
                latencies.append(lat)

            gen = t.get("generation") or {}
            answer = gen.get("answer", "")
            is_grounded = bool(gen.get("grounded", True))
            retrieval_failed = bool(gen.get("retrieval_failed", False))

            if is_grounded and not retrieval_failed:
                grounded_count += 1

            if "cannot answer" in answer.lower() or "blocked by safety" in answer.lower():
                refusal_count += 1

            # Citation extraction
            cites = _CITATION_RE.findall(answer)
            cite_count = len(cites)
            total_citations += cite_count
            has_citations = cite_count > 0 or gen.get("citation_count", 0) > 0
            if has_citations:
                with_citations_count += 1

            # Proxy faithfulness score: combination of grounded flag and citation presence
            proxy = 1.0 if (is_grounded and has_citations) else (0.5 if is_grounded else 0.0)
            faithfulness_proxies.append(proxy)

        n = len(traces)
        grounded_rate = round(grounded_count / n, 4)
        citation_coverage = round(with_citations_count / n, 4)
        mean_citations = round(total_citations / n, 2)
        refusal_rate = round(refusal_count / n, 4)
        mean_faithfulness = (
            round(float(np.mean(faithfulness_proxies)), 4) if faithfulness_proxies else 0.0
        )

        p50 = round(float(np.percentile(latencies, 50)), 3) if latencies else 0.0
        p95 = round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0

        alerts: list[str] = []
        if grounded_rate < self.min_grounded_threshold:
            alerts.append(
                f"Grounded rate {grounded_rate * 100:.1f}% below SLI threshold of {self.min_grounded_threshold * 100:.1f}%"
            )
        if citation_coverage < self.min_citation_coverage_threshold:
            alerts.append(
                f"Citation coverage {citation_coverage * 100:.1f}% below SLI threshold of {self.min_citation_coverage_threshold * 100:.1f}%"
            )
        if mean_faithfulness < self.min_faithfulness_proxy_threshold:
            alerts.append(
                f"Faithfulness proxy {mean_faithfulness:.3f} below SLI threshold of {self.min_faithfulness_proxy_threshold:.3f}"
            )
        if p95 > self.max_p95_latency_seconds:
            alerts.append(
                f"p95 latency {p95:.2f}s exceeds SLA maximum of {self.max_p95_latency_seconds:.2f}s"
            )

        alert_triggered = len(alerts) > 0
        if alert_triggered:
            logger.warning(f"[OnlineMonitor] SLI Alerts triggered: {alerts}")
        else:
            logger.info(f"[OnlineMonitor] Production quality audit passed across {n} requests.")

        return OnlineQualityMetrics(
            sample_count=n,
            grounded_rate=grounded_rate,
            citation_coverage_rate=citation_coverage,
            mean_citations_per_answer=mean_citations,
            faithfulness_proxy_score=mean_faithfulness,
            refusal_rate=refusal_rate,
            latency_p50=p50,
            latency_p95=p95,
            alert_triggered=alert_triggered,
            alerts=alerts,
        )

    def evaluate_audit_log(
        self,
        audit_log_path: Path | str,
        sample_size: int = 100,
    ) -> OnlineQualityMetrics:
        """Read recent entries from JSONL audit log and evaluate."""
        path = Path(audit_log_path)
        if not path.exists():
            logger.warning(f"Audit log file not found at {path}. Returning empty report.")
            return OnlineQualityMetrics(
                sample_count=0,
                alert_triggered=False,
                alerts=[f"Audit log file not found at {path}"],
            )

        traces: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
                # Read from end of file for most recent production requests
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traces.append(json.loads(line))
                        if len(traces) >= sample_size:
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.error(f"Failed to read audit log {path}: {exc}")
            return OnlineQualityMetrics(
                sample_count=0,
                alert_triggered=True,
                alerts=[f"Error reading audit log: {exc}"],
            )

        return self.evaluate_traces(traces)

    def append_quality_metric(
        self,
        metrics: OnlineQualityMetrics,
        output_file: Path | str = "data/online_quality.jsonl",
    ) -> None:
        """Persist rolling metric report to JSONL for Grafana/Prometheus ingestion."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit live production RAG query quality from audit logs."
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default="data/audit_logs/audit.jsonl",
        help="Path to JSONL audit log.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of recent requests to audit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/online_quality.jsonl",
        help="Path to append rolling metrics.",
    )
    args = parser.parse_args()

    monitor = OnlineQualityMonitor()
    report = monitor.evaluate_audit_log(args.audit_log, sample_size=args.sample_size)
    print(report.summary())
    monitor.append_quality_metric(report, output_file=args.output)


if __name__ == "__main__":
    main()
