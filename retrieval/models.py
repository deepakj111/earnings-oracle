"""
Data contracts for Layer 3 — Retrieval.

Three dataclasses flow through this layer:
  MetadataFilter  → optional ticker/year/quarter scoping passed into search
  SearchResult    → single chunk after RRF fusion + parent fetch
  RetrievalResult → final output of the full retrieval pipeline (search + rerank)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetadataFilter:
    """
    Optional scoping filter applied to both Qdrant and BM25 searches.

    All fields are optional — only set fields are applied as filters.
    Example: MetadataFilter(ticker="AAPL", year=2024) scopes results
    to Apple filings from 2024, ignoring quarter entirely.
    """

    ticker: str | None = None
    year: int | None = None
    quarter: str | None = None  # "Q1" | "Q2" | "Q3" | "Q4"


@dataclass
class SearchResult:
    """
    A single retrieved chunk after RRF fusion and parent fetch.

    text        : child chunk text (128 tokens) — what was matched by retrieval
    parent_text : full parent chunk text (512 tokens) — what gets passed to the LLM
    rrf_score   : Reciprocal Rank Fusion score before reranking (higher = more relevant)
    rerank_score: FlashRank cross-encoder score (set after reranking; -inf before)
    source      : which retrieval system surfaced this chunk
    """

    chunk_id: str
    parent_id: str | None
    text: str
    parent_text: str
    rrf_score: float
    rerank_score: float
    ticker: str
    company: str
    date: str
    year: int
    quarter: str
    fiscal_period: str
    section_title: str
    doc_type: str
    source: str  # "dense" | "bm25" | "both"

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        rrf_score: float,
        source: str,
    ) -> SearchResult:
        """
        Construct a SearchResult from a Qdrant point payload dict.
        Provides safe defaults for every field so missing payload keys
        never crash the retrieval layer.
        """
        return cls(
            chunk_id=payload.get("chunk_id", ""),
            parent_id=payload.get("parent_id"),
            text=payload.get("text", ""),
            parent_text=payload.get("text", ""),  # overwritten by parent fetch
            rrf_score=rrf_score,
            rerank_score=float("-inf"),
            ticker=payload.get("ticker", ""),
            company=payload.get("company", ""),
            date=payload.get("date", ""),
            year=int(payload.get("year", 0)),
            quarter=payload.get("quarter", ""),
            fiscal_period=payload.get("fiscal_period", ""),
            section_title=payload.get("section_title", ""),
            doc_type=payload.get("doc_type", ""),
            source=source,
        )


@dataclass
class RetrievalResult:
    """
    Final output of the full retrieval pipeline (searcher + reranker).

    results         : top-k chunks sorted by rerank_score (or rrf_score if reranking disabled)
    reranked        : True if FlashRank cross-encoder was applied
    total_candidates: number of unique chunks that entered the reranker
    query           : the original user query (for traceability)
    """

    query: str
    results: list[SearchResult]
    reranked: bool
    total_candidates: int
    metadata_filter: MetadataFilter | None = None
    failed_techniques: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    def summary(self) -> str:
        lines = [
            f"Query       : {self.query}",
            f"Candidates  : {self.total_candidates}",
            f"Returned    : {len(self.results)}",
            f"Reranked    : {self.reranked}",
        ]
        if self.metadata_filter:
            f = self.metadata_filter
            parts = [p for p in [f.ticker, str(f.year) if f.year else None, f.quarter] if p]
            lines.append(f"Filter      : {' | '.join(parts)}")
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"  [{i}] {r.ticker} {r.date} | {r.section_title[:40]} "
                f"| rrf={r.rrf_score:.4f} rerank={r.rerank_score:.4f} src={r.source}"
            )
        return "\n".join(lines)
