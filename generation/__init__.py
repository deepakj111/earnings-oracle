"""
Layer 4 — Answer Generation.

Public API:

    from generation import generate, Generator
    from generation.models import GenerationResult, Citation

    # Module-level shortcut (uses shared singleton Generator)
    result = generate(
        question="What was Apple's revenue in Q4 2024?",
        retrieval_result=retrieval_result,
    )
    print(result.format_answer_with_citations())
    print(result.to_json())

    # Direct class usage (useful when you need streaming)
    generator = Generator()
    result = generator.generate(question, retrieval_result)

    for token in generator.generate_streaming(question, retrieval_result):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from generation.generator import Generator
from generation.models import Citation, GenerationResult

if TYPE_CHECKING:
    from retrieval.models import RetrievalResult

# ── Module-level singleton ────────────────────────────────────────────────────
# Shared Generator instance used by the generate() shortcut.
# The Generator class is stateless — this avoids re-initialising the OpenAI
# client on every call while keeping the import surface clean.

_generator: Generator | None = None


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def generate(
    question: str,
    retrieval_result: RetrievalResult,
) -> GenerationResult:
    """
    Module-level shortcut: generate an answer from a RetrievalResult.

    Equivalent to Generator().generate(question, retrieval_result).
    Uses a shared Generator singleton to avoid re-creating the OpenAI client.

    Args:
        question         : natural language question (non-empty)
        retrieval_result : output from retrieval.retrieve()

    Returns:
        GenerationResult with answer, inline citations, and diagnostics.
    """
    return _get_generator().generate(question, retrieval_result)


__all__ = [
    "generate",
    "Generator",
    "GenerationResult",
    "Citation",
]
