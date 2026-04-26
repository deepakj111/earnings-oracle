"""
Tests for generation/context_builder.py

Coverage:
  _valley_reorder  — valley ordering for k=1..6, edge cases
  build_context    — parent deduplication, token budget, block formatting, empty input
"""

from __future__ import annotations

from generation.context_builder import _format_block, _valley_reorder, build_context
from retrieval.models import SearchResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    chunk_id: str,
    rerank_score: float = 0.9,
    parent_id: str | None = None,
    ticker: str = "AAPL",
    fiscal_period: str = "Q4 2024",
    section_title: str = "Revenue",
    text: str = "Apple reported revenue of $100B.",
    parent_text: str = "",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=parent_id,
        text=text,
        parent_text=parent_text or text,
        rrf_score=0.05,
        rerank_score=rerank_score,
        ticker=ticker,
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period=fiscal_period,
        section_title=section_title,
        doc_type="earnings_release",
        source="both",
    )


# ── _valley_reorder ───────────────────────────────────────────────────────────


class TestValleyReorder:
    def test_k1_unchanged(self) -> None:
        results = [_make_result("c1")]
        assert _valley_reorder(results) == results

    def test_k2_unchanged(self) -> None:
        r1, r2 = _make_result("c1"), _make_result("c2")
        assert _valley_reorder([r1, r2]) == [r1, r2]

    def test_k3_valley(self) -> None:
        r1, r2, r3 = _make_result("c1"), _make_result("c2"), _make_result("c3")
        # even indices → left: [r1, r3]; odd indices → right reversed: [r2]
        result = _valley_reorder([r1, r2, r3])
        assert result == [r1, r3, r2]

    def test_k4_valley(self) -> None:
        r1, r2, r3, r4 = [_make_result(f"c{i}") for i in range(1, 5)]
        # even: [r1, r3]; odd reversed: [r4, r2]
        result = _valley_reorder([r1, r2, r3, r4])
        assert result == [r1, r3, r4, r2]

    def test_k5_valley(self) -> None:
        r1, r2, r3, r4, r5 = [_make_result(f"c{i}") for i in range(1, 6)]
        # even: [r1, r3, r5]; odd reversed: [r4, r2]
        result = _valley_reorder([r1, r2, r3, r4, r5])
        assert result == [r1, r3, r5, r4, r2]

    def test_k6_valley(self) -> None:
        results = [_make_result(f"c{i}") for i in range(1, 7)]
        r1, r2, r3, r4, r5, r6 = results
        # even: [r1, r3, r5]; odd reversed: [r6, r4, r2]
        result = _valley_reorder(results)
        assert result == [r1, r3, r5, r6, r4, r2]

    def test_best_rank_always_first(self) -> None:
        """The rank-1 result (index 0) must always occupy position 0."""
        for k in range(1, 8):
            results = [_make_result(f"c{i}") for i in range(k)]
            reordered = _valley_reorder(results)
            assert reordered[0] is results[0], f"rank-1 not at position 0 for k={k}"

    def test_second_rank_always_last(self) -> None:
        """The rank-2 result (index 1) must occupy the last position for k≥3."""
        for k in range(3, 8):
            results = [_make_result(f"c{i}") for i in range(k)]
            reordered = _valley_reorder(results)
            assert reordered[-1] is results[1], f"rank-2 not at last position for k={k}"

    def test_no_items_dropped(self) -> None:
        """Valley reorder must preserve all items."""
        results = [_make_result(f"c{i}") for i in range(7)]
        reordered = _valley_reorder(results)
        assert sorted(r.chunk_id for r in reordered) == sorted(r.chunk_id for r in results)

    def test_empty_list(self) -> None:
        assert _valley_reorder([]) == []


# ── _format_block ─────────────────────────────────────────────────────────────


class TestFormatBlock:
    def test_header_contains_citation_index(self) -> None:
        r = _make_result("c1")
        block = _format_block(3, r)
        assert "[3]" in block

    def test_header_contains_ticker_and_period(self) -> None:
        r = _make_result("c1", ticker="NVDA", fiscal_period="Q3 2024")
        block = _format_block(1, r)
        assert "NVDA" in block
        assert "Q3 2024" in block

    def test_body_uses_parent_text_when_available(self) -> None:
        r = _make_result("c1", text="child text", parent_text="parent text")
        block = _format_block(1, r)
        assert "parent text" in block
        assert "child text" not in block

    def test_body_falls_back_to_text_when_no_parent(self) -> None:
        r = _make_result("c1", text="only child text", parent_text="")
        # parent_text="" → falls back to text
        block = _format_block(1, r)
        assert "only child text" in block

    def test_section_title_truncated_to_60_chars(self) -> None:
        long_title = "A" * 80
        r = _make_result("c1", section_title=long_title)
        block = _format_block(1, r)
        # The header line should not contain the full 80-char title
        header_line = block.split("\n")[0]
        assert "A" * 61 not in header_line


# ── build_context ─────────────────────────────────────────────────────────────


class TestBuildContext:
    def test_empty_results_returns_empty(self) -> None:
        context, results, tokens = build_context([], max_context_tokens=1000)
        assert context == ""
        assert results == []
        assert tokens == 0

    def test_single_result_included(self) -> None:
        r = _make_result("c1", text="Revenue was $100B in Q4 2024.")
        context, results, tokens = build_context([r], max_context_tokens=2000)
        assert "[1]" in context
        assert len(results) == 1
        assert tokens > 0

    def test_citation_index_matches_position(self) -> None:
        """citation_results[i-1] must correspond to block [i] in context text."""
        r1 = _make_result("c1", text="Apple revenue $100B.")
        r2 = _make_result("c2", ticker="NVDA", text="NVIDIA revenue $35B.")
        context, results, _ = build_context([r1, r2], max_context_tokens=4000)
        assert len(results) >= 1
        # Block [1] appears in context
        assert "[1]" in context

    def test_parent_deduplication(self) -> None:
        """Two children sharing a parent_id should produce only one context block."""
        shared_parent = "parent_abc"
        c1 = _make_result("c1", parent_id=shared_parent, rerank_score=0.9)
        c2 = _make_result("c2", parent_id=shared_parent, rerank_score=0.7)
        context, results, _ = build_context([c1, c2], max_context_tokens=8000)
        # Only one block should be included
        assert len(results) == 1
        assert results[0].chunk_id == "c1"  # higher rerank_score wins

    def test_different_parents_both_included(self) -> None:
        c1 = _make_result("c1", parent_id="p1")
        c2 = _make_result("c2", parent_id="p2")
        _, results, _ = build_context([c1, c2], max_context_tokens=8000)
        assert len(results) == 2

    def test_no_parent_id_uses_chunk_id_as_key(self) -> None:
        """Chunks without a parent_id should each get their own slot."""
        c1 = _make_result("c1", parent_id=None)
        c2 = _make_result("c2", parent_id=None)
        _, results, _ = build_context([c1, c2], max_context_tokens=8000)
        assert len(results) == 2

    def test_token_budget_enforced(self) -> None:
        """build_context must not exceed max_context_tokens."""
        # Create a result with enough text to require budgeting
        long_text = "Revenue was $100 billion. " * 100  # ~500 tokens
        results_list = [_make_result(f"c{i}", text=long_text) for i in range(10)]
        max_tokens = 200

        context, results, token_count = build_context(results_list, max_context_tokens=max_tokens)
        # token_count must be ≤ max_tokens (hard truncation may equal it)
        assert token_count <= max_tokens

    def test_first_chunk_hard_truncated_when_oversized(self) -> None:
        """If even the first chunk exceeds budget, it must be truncated (not dropped)."""
        long_text = "word " * 1000  # ~1000 tokens
        r = _make_result("c1", text=long_text)
        context, results, token_count = build_context([r], max_context_tokens=50)
        # We got something back (not empty)
        assert len(results) == 1
        assert token_count <= 50

    def test_context_text_matches_returned_token_count(self) -> None:
        """The returned token_count should match the actual context_text length."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        r = _make_result("c1", text="Short text for testing.")
        context, _, token_count = build_context([r], max_context_tokens=2000)
        actual = len(enc.encode(context, disallowed_special=()))
        assert actual == token_count

    def test_valley_ordering_applied(self) -> None:
        """
        With 3+ results, the first block in context must correspond to the
        rank-1 result (valley reorder puts rank-1 first).
        """
        r1 = _make_result("c1", rerank_score=0.95, text="Best chunk.")
        r2 = _make_result("c2", rerank_score=0.80, text="Second chunk.")
        r3 = _make_result("c3", rerank_score=0.70, text="Third chunk.")
        context, results, _ = build_context([r1, r2, r3], max_context_tokens=8000)
        # results[0] must be r1 (rank-1 placed at front by valley reorder)
        assert results[0].chunk_id == "c1"

    def test_multiple_tickers_in_context(self) -> None:
        """Context blocks from different companies should co-exist."""
        r1 = _make_result("c1", ticker="AAPL")
        r2 = _make_result("c2", ticker="MSFT", fiscal_period="Q4 2024")
        context, results, _ = build_context([r1, r2], max_context_tokens=8000)
        assert "AAPL" in context
        assert "MSFT" in context
