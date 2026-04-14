# crag/models.py
"""
Data contracts for Layer 5 — Corrective RAG (CRAG).

Three dataclasses produced by this layer:
  RelevanceGrade  → per-chunk LLM relevance judgement
  WebSearchResult → single result from the web-search fallback
  CRAGResult      → full CRAG output wrapping the final GenerationResult
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from generation.models import GenerationResult


class CRAGAction(str, Enum):
    """
    Outcome of the CRAG relevance assessment.

    CORRECT   — Retrieved docs are sufficiently relevant; use original answer.
    AMBIGUOUS — Partial relevance; combine local relevant chunks with web results.
    INCORRECT — No relevant docs found; replace with web-search results entirely.
    """

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class RelevanceGrade:
    """
    LLM-produced relevance judgement for a single retrieved chunk.

    chunk_id  : identifier of the graded chunk (matches SearchResult.chunk_id)
    relevant  : True if the chunk directly helps answer the question
    score     : confidence (0.0 = definitely irrelevant, 1.0 = definitely relevant)
    reasoning : one-sentence explanation from the grader LLM
    """

    chunk_id: str
    relevant: bool
    score: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "relevant": self.relevant,
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class WebSearchResult:
    """
    A single result from the web-search fallback.

    snippet is injected into the generation context as a numbered block,
    so it should be a clean readable passage (not raw HTML).
    """

    title: str
    url: str
    snippet: str
    score: float = 0.0

    def to_context_block(self, index: int) -> str:
        """Format as a numbered context block matching generation/context_builder.py format."""
        return f"--- [{index}] WEB | {self.title[:60]} ---\n{self.snippet}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet[:500],
            "score": round(self.score, 4),
        }


@dataclass
class CRAGResult:
    """
    Full output of the CRAG correction loop.

    final_result   : the GenerationResult after correction (identical to
                     original_result when action == CORRECT)
    was_corrected  : True if web search was triggered (action != CORRECT)
    relevance_ratio: fraction of retrieved chunks graded as relevant (0.0–1.0)
    """

    question: str
    action: CRAGAction
    original_result: GenerationResult
    final_result: GenerationResult
    relevance_grades: list[RelevanceGrade] = field(default_factory=list)
    web_results_used: list[WebSearchResult] = field(default_factory=list)
    web_search_triggered: bool = False
    latency_seconds: float = 0.0

    @property
    def was_corrected(self) -> bool:
        return self.action != CRAGAction.CORRECT

    @property
    def relevant_chunk_count(self) -> int:
        return sum(1 for g in self.relevance_grades if g.relevant)

    @property
    def relevance_ratio(self) -> float:
        if not self.relevance_grades:
            return 0.0
        return self.relevant_chunk_count / len(self.relevance_grades)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "action": self.action.value,
            "was_corrected": self.was_corrected,
            "web_search_triggered": self.web_search_triggered,
            "relevance_ratio": round(self.relevance_ratio, 4),
            "relevant_chunks": self.relevant_chunk_count,
            "total_chunks_graded": len(self.relevance_grades),
            "web_results_used": len(self.web_results_used),
            "latency_seconds": round(self.latency_seconds, 3),
            "final_answer": self.final_result.answer,
            "final_grounded": self.final_result.grounded,
            "final_citations": len(self.final_result.citations),
            "original_grounded": self.original_result.grounded,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
