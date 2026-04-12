from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransformedQuery:
    """
    Output of Layer 2 — Query Transformation.

    Carries all query variants produced by HyDE, Multi-Query, and Step-Back
    prompting. The retrieval layer consumes this dataclass directly, embedding
    hyde_document for dense search and fanning out multi_queries + stepback_query
    for union-based recall expansion.
    """

    original: str
    hyde_document: str
    multi_queries: list[str]
    stepback_query: str

    # Populated at runtime if any technique failed gracefully
    failed_techniques: list[str] = field(default_factory=list)

    @property
    def all_retrieval_queries(self) -> list[str]:
        """
        Full set of queries to send to the retrieval layer.
        Combines multi_queries (includes original) + stepback_query, deduplicated.
        Used for union-based BM25 + dense retrieval before RRF fusion.
        """
        seen: set[str] = set()
        result: list[str] = []
        for q in self.multi_queries + [self.stepback_query]:
            normalized = q.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(q.strip())
        return result

    def summary(self) -> str:
        lines = [
            f"Original      : {self.original}",
            f"HyDE doc      : {self.hyde_document[:120].strip()}...",
            f"Multi-queries : {len(self.multi_queries)} total",
        ]
        for i, q in enumerate(self.multi_queries, 1):
            lines.append(f"  [{i}] {q}")
        lines.append(f"Step-Back     : {self.stepback_query}")
        if self.failed_techniques:
            lines.append(f"Degraded      : {', '.join(self.failed_techniques)} (fallbacks used)")
        return "\n".join(lines)
