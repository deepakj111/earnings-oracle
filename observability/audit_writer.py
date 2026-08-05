# observability/audit_writer.py
"""
Thread-safe query audit writer for the Financial RAG system.

Every completed pipeline request is persisted in two complementary formats:

1. Per-trace JSON
   Path : data/audit_logs/YYYY-MM-DD/trace_<timestamp>_<trace_id>.json
   Size : full detail — HyDE doc, all multi-queries, per-chunk scores, answer text
   Use  : deep inspection of a single request

2. Global JSONL (append-only)
   Path : data/audit_logs/audit.jsonl
   Size : one compact summary line per request
   Use  : grep / jq / pandas.read_json(lines=True) for batch analysis

Design decisions:
  - Thread-safe via a single threading.Lock for the JSONL append
  - Daily subdirectories keep the file count manageable
  - Per-day cap of 10,000 JSON files (oldest pruned on overflow)
  - JSONL never pruned — rotate externally if needed (logrotate or cron)
  - Never raises — all I/O errors are logged at WARNING and swallowed so
    a disk error never breaks a query response
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from observability.trace_models import PipelineTrace


class AuditWriter:
    """
    Persist pipeline traces to disk in JSON + JSONL format.

    Thread-safety: safe for concurrent use from any number of request threads.
    The per-trace JSON files are independent (no lock needed).
    The JSONL append uses a single Lock to prevent interleaved writes.

    Args:
        output_dir : base directory for audit output (default: data/audit_logs)
        max_files_per_day : prune oldest JSON files once this threshold is exceeded
    """

    def __init__(
        self,
        output_dir: str = "data/audit_logs",
        max_files_per_day: int = 10_000,
    ) -> None:
        self._base = Path(output_dir)
        self._max_per_day = max_files_per_day
        self._jsonl_path = self._base / "audit.jsonl"
        self._lock = threading.Lock()

        # Ensure base directory exists at startup
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            logger.info(f"AuditWriter ready | output_dir={output_dir}")
        except OSError as exc:
            logger.warning(f"AuditWriter: could not create output dir {output_dir}: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def write(self, trace: PipelineTrace) -> Path | None:
        """
        Persist a completed trace.

        Writes:
          - A full JSON file in the daily subdirectory
          - A compact summary line appended to the global JSONL

        Returns:
            Path of the written JSON file, or None on I/O failure.
        """
        try:
            json_path = self._write_json(trace)
            self._append_jsonl(trace)
            return json_path
        except Exception as exc:
            logger.warning(f"AuditWriter: unexpected error writing trace {trace.trace_id}: {exc}")
            return None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _daily_dir(self, trace: PipelineTrace) -> Path:
        """Return (and create) the YYYY-MM-DD subdirectory for this trace."""
        try:
            # Parse trace timestamp to get the date component
            ts = datetime.fromisoformat(trace.timestamp)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
        date_str = ts.strftime("%Y-%m-%d")
        day_dir = self._base / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir

    def _write_json(self, trace: PipelineTrace) -> Path:
        """Write a full trace JSON file. Prunes excess files if over cap."""
        day_dir = self._daily_dir(trace)

        # Build a filesystem-safe timestamp prefix
        try:
            ts_str = datetime.fromisoformat(trace.timestamp).strftime("%H%M%S")
        except (ValueError, TypeError):
            ts_str = datetime.now(timezone.utc).strftime("%H%M%S")

        filename = f"trace_{ts_str}_{trace.trace_id}.json"
        path = day_dir / filename

        path.write_text(trace.to_json(indent=2), encoding="utf-8")
        logger.debug(f"Audit trace written → {path}")

        # Prune if over the daily cap (keep newest)
        self._prune_daily(day_dir)

        return path

    def _prune_daily(self, day_dir: Path) -> None:
        """Remove oldest JSON files in a daily dir once over the cap."""
        try:
            files = sorted(
                day_dir.glob("trace_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,  # newest first
            )
            if len(files) > self._max_per_day:
                for old in files[self._max_per_day:]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
        except Exception as exc:
            logger.debug(f"AuditWriter: pruning failed for {day_dir}: {exc}")

    def _append_jsonl(self, trace: PipelineTrace) -> None:
        """
        Append a compact one-line JSON summary to the global JSONL file.

        The summary intentionally omits the full HyDE doc, per-chunk excerpts,
        and answer text to keep each line small and grep-friendly.
        Individual JSON files hold the full detail.
        """
        summary = self._build_summary(trace)
        line = json.dumps(summary, ensure_ascii=False) + "\n"

        with self._lock:
            try:
                with self._jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                logger.warning(f"AuditWriter: could not append to JSONL: {exc}")

    @staticmethod
    def _build_summary(trace: PipelineTrace) -> dict:
        """
        Build a compact summary record for the JSONL audit log.

        Contains all fields needed for trend analysis and alerting,
        but excludes large text blobs (HyDE doc, chunk excerpts, answer).
        """
        qt = trace.query_transform
        ret = trace.retrieval
        gen = trace.generation

        return {
            "schema_version": "1.0",
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "received_at": trace.timestamp,
            "endpoint": trace.endpoint,
            "question": trace.question,
            "filter": trace.applied_filter,
            "status": trace.status.value,
            "error_message": trace.error_message,
            # Timing
            "total_latency_seconds": round(trace.total_latency_seconds, 3),
            "latency_breakdown": {
                k: round(v, 3) for k, v in trace.latency_breakdown.items()
            },
            # Tokens & cost
            "total_tokens": trace.total_tokens,
            "total_prompt_tokens": trace.total_prompt_tokens,
            "total_completion_tokens": trace.total_completion_tokens,
            "total_cost_usd": round(trace.total_cost_usd, 6),
            "total_llm_calls": trace.total_llm_calls,
            # Layer summaries
            "query_transform": {
                "cache_hit": qt.cache_hit if qt else None,
                "techniques_attempted": qt.techniques_attempted if qt else [],
                "techniques_failed": qt.techniques_failed if qt else [],
                "multi_query_count": qt.multi_query_count if qt else 0,
                "hyde_generated": qt.hyde_generated if qt else False,
                "stepback_generated": qt.stepback_generated if qt else False,
            } if qt else None,
            "retrieval": {
                "total_candidates": ret.total_unique_candidates if ret else 0,
                "final_chunk_count": ret.final_chunk_count if ret else 0,
                "reranked": ret.reranked if ret else False,
                "reranker_model": ret.reranker_model if ret else "",
                "top_rerank_score": round(ret.top_rerank_score, 4) if ret else 0.0,
                "source_distribution": ret.source_distribution if ret else {},
            } if ret else None,
            "generation": {
                "model": gen.model if gen else "",
                "mode": gen.mode if gen else "structured",
                "prompt_tokens": gen.prompt_tokens if gen else 0,
                "completion_tokens": gen.completion_tokens if gen else 0,
                "context_chunks_used": gen.context_chunks_used if gen else 0,
                "citation_count": gen.citation_count if gen else 0,
                "grounded": gen.grounded if gen else True,
                "latency_seconds": round(gen.latency_seconds, 3) if gen else 0.0,
            } if gen else None,
        }
