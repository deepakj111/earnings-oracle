"""
Context window construction for Layer 4 — Generation.

Responsibilities:
  1. Parent-deduplication — never send the same parent chunk twice (avoid
     redundant tokens when two child chunks share the same parent)
  2. Token budget enforcement — hard cap at max_context_tokens; never exceed
  3. Lost-in-the-middle mitigation — reorder chunks so the highest-relevance
     evidence occupies the positions with highest LLM attention (start + end)
  4. Numbered block formatting — produces [1], [2], ... blocks for citation
     mapping in generator.py

──────────────────────────────────────────────────────────────────────────────
Lost-in-the-Middle: Background
──────────────────────────────────────────────────────────────────────────────
Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts"
showed that LLM performance degrades significantly when the relevant document
is placed in the middle of a long context.  The attention pattern is U-shaped:
models attend strongly to the first and last few documents, poorly to the
middle.

Mitigation — "valley ordering":
  Input (sorted best→worst by rerank_score): [r1, r2, r3, r4, r5]

  Valley order:                              [r1, r3, r5, r4, r2]
  LLM attention position:                   [HIGH  ↑↑ ↓↓ ↑↑ HIGH]

  Even-indexed ranks go left  (first half, chronological order)
  Odd-indexed ranks go right  (second half, REVERSED order)
  → r1 at position 0 (highest attention), r2 at position -1 (second-highest)

  For k=5: positions of rank → [r1@0, r3@1, r5@2, r4@3, r2@4]
  For k=4: positions of rank → [r1@0, r3@1, r4@2, r2@3]
  For k=3: positions of rank → [r1@0, r3@1, r2@2]
  For k=2: unchanged           [r1@0, r2@1]
  For k=1: unchanged           [r1@0]
"""

from __future__ import annotations

import tiktoken

from retrieval.models import SearchResult

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


# ── Lost-in-the-middle mitigation ─────────────────────────────────────────────


def _valley_reorder(results: list[SearchResult]) -> list[SearchResult]:
    """
    Reorder a list of SearchResults (best-first) into valley ordering.

    Strategy:
      - Even indices (0, 2, 4, …) → left half in order
      - Odd  indices (1, 3, 5, …) → right half in REVERSE order

    This guarantees:
      - rank-1 (best)  at position 0  (highest LLM attention)
      - rank-2 (second) at last position (second-highest LLM attention)
      - rank-3 at position 1, rank-4 at second-to-last, etc.
    """
    if len(results) <= 2:
        return list(results)

    left: list[SearchResult] = []  # even-rank items → front of context
    right: list[SearchResult] = []  # odd-rank items  → back of context (reversed)

    for i, r in enumerate(results):
        if i % 2 == 0:
            left.append(r)
        else:
            right.append(r)

    # right is reversed so that rank-2 (first odd-rank item) ends up last
    return left + list(reversed(right))


# ── Block formatting ───────────────────────────────────────────────────────────


def _format_block(index: int, result: SearchResult) -> str:
    """
    Render a single SearchResult as a numbered context block.

    Format:
        --- [1] AAPL | Q4 2024 | Revenue ---
        <parent_text or child text>

    The header is intentionally terse — every token in the header adds to
    prompt cost.  The [N] marker is the contract that lets the generator
    produce inline citations [1], [2] etc.
    """
    section = (result.section_title or "—")[:60]
    header = f"--- [{index}] {result.ticker} | {result.fiscal_period} | {section} ---"
    body = (result.parent_text or result.text).strip()
    return f"{header}\n{body}"


# ── Public API ─────────────────────────────────────────────────────────────────


def build_context(
    results: list[SearchResult],
    max_context_tokens: int,
) -> tuple[str, list[SearchResult], int]:
    """
    Build a numbered context block from a list of SearchResults.

    Pipeline:
      1. Deduplicate by parent_id — keep highest-rerank_score result per parent
      2. Valley-reorder for lost-in-the-middle mitigation
      3. Greedy token-budget allocation — add blocks until budget exhausted
      4. Format as numbered [1]..[N] blocks

    Args:
        results            : SearchResults from Layer 3 (sorted best→worst by rerank_score)
        max_context_tokens : hard token budget for the entire context block

    Returns:
        context_text    : complete formatted context string with [N] blocks
        citation_results: SearchResults in citation order (citation_results[i-1] = block [i])
        token_count     : exact token count of context_text

    Notes:
        - citation_results[0] corresponds to [1] in the answer, etc. (1-based)
        - The returned list may be shorter than `results` if the budget is exhausted
        - If even the first chunk alone exceeds budget, it is hard-truncated to fit
    """
    if not results:
        return "", [], 0

    # ── 1. Deduplicate by parent_id ──────────────────────────────────────────
    # When two child chunks share a parent, only the higher-scoring one is kept.
    # This prevents the LLM from seeing the same 512-token parent block twice,
    # which would waste tokens and inflate citation diversity signals.
    seen_parents: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        key = r.parent_id or r.chunk_id  # tables / standalone chunks use chunk_id
        if key not in seen_parents:
            seen_parents.add(key)
            deduped.append(r)

    # ── 2. Valley reorder ────────────────────────────────────────────────────
    ordered = _valley_reorder(deduped)

    # ── 3. Greedy token-budget allocation ────────────────────────────────────
    blocks: list[str] = []
    citation_results: list[SearchResult] = []
    total_tokens = 0

    for i, result in enumerate(ordered, start=1):
        block = _format_block(i, result)
        block_tokens = _count_tokens(block)

        if total_tokens + block_tokens > max_context_tokens:
            if not blocks:
                # Edge case: even the first chunk is too large.
                # Hard-truncate to fill the entire budget rather than returning empty.
                token_ids = _ENC.encode(block, disallowed_special=())
                truncated = _ENC.decode(token_ids[:max_context_tokens])
                blocks.append(truncated)
                citation_results.append(result)
                total_tokens = max_context_tokens
            # Budget exhausted — stop adding further chunks.
            break

        blocks.append(block)
        citation_results.append(result)
        total_tokens += block_tokens

    context_text = "\n\n".join(blocks)
    return context_text, citation_results, total_tokens
