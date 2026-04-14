# crag/__init__.py
"""
Layer 5 — Corrective RAG (CRAG).

Public API:

    from crag import CRAGCorrector
    from crag.models import CRAGAction, CRAGResult

    corrector = CRAGCorrector()
    result = corrector.correct(
        question="What was Apple's Q4 2024 revenue?",
        generation_result=gen_result,
        retrieval_result=retrieval_result,
    )

    print(result.final_result.format_answer_with_citations())
    print(f"CRAG action : {result.action.value}")
    print(f"Was corrected: {result.was_corrected}")
    print(f"Web results  : {len(result.web_results_used)}")
    print(result.to_json())
"""

from crag.corrector import CRAGCorrector
from crag.grader import RelevanceGrader
from crag.models import CRAGAction, CRAGResult, RelevanceGrade, WebSearchResult
from crag.web_search import WebSearchClient

__all__ = [
    "CRAGCorrector",
    "RelevanceGrader",
    "WebSearchClient",
    "CRAGAction",
    "CRAGResult",
    "RelevanceGrade",
    "WebSearchResult",
]
