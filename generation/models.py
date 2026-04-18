"""
Data contracts for Layer 4 — Generation.

Two dataclasses are produced by this layer:
  Citation        → a single grounded source reference extracted from the answer
  GenerationResult→ the full output of the generation pipeline, ready for the UI/API layer
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Citation:
    """
    A single source citation extracted from the generated answer.

    index       : 1-based citation number that appears inline in the answer as [N]
    chunk_id    : child chunk that was retrieved and matched by retrieval
    parent_id   : parent chunk whose text was passed to the LLM as context
    ticker      : company ticker (e.g. "AAPL")
    company     : full company name (e.g. "Apple")
    date        : filing date string "YYYY-MM-DD"
    fiscal_period: e.g. "Q4 2024"
    section_title: section header from the original document (e.g. "Revenue")
    doc_type    : "earnings_release" etc.
    source      : which retrieval system surfaced this chunk ("dense"|"bm25"|"both")
    rerank_score: FlashRank cross-encoder score — higher = more relevant
    excerpt     : first 250 chars of parent_text — used for answer verification / Ragas eval
    """

    index: int
    chunk_id: str
    parent_id: str | None
    ticker: str
    company: str
    date: str
    fiscal_period: str
    section_title: str
    doc_type: str
    source: str
    rerank_score: float
    excerpt: str

    @property
    def label(self) -> str:
        """Human-readable source label: 'AAPL Q4 2024 — Revenue'"""
        title = (self.section_title or "General")[:50]
        return f"{self.ticker} {self.fiscal_period} — {title}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "ticker": self.ticker,
            "company": self.company,
            "date": self.date,
            "fiscal_period": self.fiscal_period,
            "section_title": self.section_title,
            "doc_type": self.doc_type,
            "source": self.source,
            "rerank_score": round(self.rerank_score, 4),
            "excerpt": self.excerpt,
        }


@dataclass
class GenerationResult:
    """
    Final output of the full RAG pipeline (all four layers).

    answer           : synthesised LLM response with inline [N] citations
    citations        : structured metadata for each citation used in the answer
    grounded         : False if the model signalled it couldn't find relevant context
    retrieval_failed : True if the retrieval layer returned zero results
    context_chunks_used: how many chunks were actually sent to the LLM
    context_tokens_used: token cost of the context block (for cost tracking)
    """

    question: str
    answer: str
    citations: list[Citation]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_chunks_used: int
    context_tokens_used: int
    latency_seconds: float
    grounded: bool
    retrieval_failed: bool
    trace_id: str | None = None  # set by tracer for request correlation

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def unique_tickers(self) -> list[str]:
        """Deduplicated list of tickers cited in the answer, in citation order."""
        seen: set[str] = set()
        result: list[str] = []
        for c in self.citations:
            if c.ticker not in seen:
                seen.add(c.ticker)
                result.append(c.ticker)
        return result

    @property
    def unique_sources(self) -> list[str]:
        """
        Deduplicated list of 'TICKER fiscal_period' source labels, in citation order.
        Useful for display in a UI: "Sources: AAPL Q4 2024, NVDA Q3 2024".
        """
        seen: set[str] = set()
        result: list[str] = []
        for c in self.citations:
            label = f"{c.ticker} {c.fiscal_period}"
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_answer_with_citations(self) -> str:
        """
        Returns the answer text followed by a numbered source list.

        Example:
            Apple reported Services revenue of $26.3B in Q4 2024 [1],
            up 22% year-over-year [1][2].

            Sources:
              [1] AAPL Q4 2024 — Revenue  (rerank: 0.9821)
              [2] AAPL Q4 2024 — Financial Highlights  (rerank: 0.9412)
        """
        lines = [self.answer]
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for c in self.citations:
                lines.append(f"  [{c.index}] {c.label}  (rerank: {c.rerank_score:.4f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "context": {
                "chunks_used": self.context_chunks_used,
                "tokens_used": self.context_tokens_used,
            },
            "latency_seconds": round(self.latency_seconds, 3),
            "grounded": self.grounded,
            "retrieval_failed": self.retrieval_failed,
            "unique_tickers": self.unique_tickers,
            "unique_sources": self.unique_sources,
        }
        if self.trace_id:
            d["trace_id"] = self.trace_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> GenerationResult:
        """Reconstruct a GenerationResult from a serialized dict (e.g., from cache)."""
        usage = data.get("usage", {})
        context = data.get("context", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            citations=[Citation(**c) for c in data.get("citations", [])],
            model=data.get("model", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
            context_chunks_used=context.get("chunks_used", 0),
            context_tokens_used=context.get("tokens_used", 0),
            latency_seconds=data.get("latency_seconds", 0.0),
            grounded=data.get("grounded", True),
            retrieval_failed=data.get("retrieval_failed", False),
            trace_id=data.get("trace_id"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
