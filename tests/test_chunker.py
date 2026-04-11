# tests/test_chunker.py

"""
Unit tests for the production-grade 2026 financial document chunking pipeline.

Coverage:
  - Structure-aware section splitting (headers, tables, prose)
  - Parent chunk construction (token budget, overlap, contextual prefix)
  - Child chunk construction (overlap, atomicity, parent linkage)
  - Table protection (never split, never merged, atomic child)
  - Metadata correctness on every chunk type
  - Chunk ID determinism (uuid5 stability)
  - Edge cases (empty input, single word, giant section)
  - Public API contract (create_parent_child_chunks)
"""

import pytest

from ingestion.chunker import (
    TABLE_LINE_THRESHOLD,
    Chunk,
    _contextual_prefix,
    _is_table_block,
    _make_chunk_id,
    _split_into_semantic_sections,
    _token_count,
    create_parent_child_chunks,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

TICKER = "AAPL"
DATE = "2024-10-31"
DOCTYPE = "earnings_release"

# A realistic minimal earnings release (enough words to pass the 10-word floor)
EARNINGS_PROSE = (
    "Revenue grew 6 percent year over year driven by strong iPhone demand across all geographies. "
    "Services reached a new all time high of 24.9 billion dollars representing 14 percent growth. "
    "The company returned 29 billion dollars to shareholders through buybacks and dividends this quarter."
)

OUTLOOK_PROSE = (
    "Outlook for the December quarter: revenue between 89 billion and 90 billion dollars. "
    "Gross margin expected between 46 and 47 percent. "
    "Operating expenses projected between 14.2 billion and 14.4 billion dollars."
)

MARKDOWN_TABLE = (
    "| Segment   | Revenue ($B) | YoY |\n"
    "|-----------|-------------|-----|\n"
    "| iPhone    | 46.2        | +6% |\n"
    "| Mac       | 7.7         | +2% |\n"
    "| iPad      | 7.0         | -8% |\n"
    "| Services  | 24.9        | +14%|\n"
    "| Wearables | 9.0         | -3% |"
)


def _long_prose(n_words: int = 600) -> str:
    word = "revenue"
    return " ".join([f"{word}{i}" for i in range(n_words)])


# ── _token_count ──────────────────────────────────────────────────────────────


class TestTokenCount:
    def test_empty_string_returns_zero(self):
        assert _token_count("") == 0

    def test_single_word(self):
        assert _token_count("hello") >= 1

    def test_longer_text_returns_more_tokens(self):
        short = _token_count("hello world")
        long = _token_count("hello world " * 50)
        assert long > short

    def test_financial_notation_handled(self):
        # Tickers, dollar signs, percentages — should not raise
        result = _token_count("AAPL revenue $94.9B (+6% YoY) beat $93.1B estimate")
        assert result > 0


# ── _is_table_block ───────────────────────────────────────────────────────────


class TestIsTableBlock:
    def test_markdown_table_detected(self):
        lines = MARKDOWN_TABLE.splitlines()
        assert _is_table_block(lines) is True

    def test_prose_not_detected_as_table(self):
        lines = EARNINGS_PROSE.splitlines()
        assert _is_table_block(lines) is False

    def test_empty_list_returns_false(self):
        assert _is_table_block([]) is False

    def test_single_pipe_line_below_threshold(self):
        # One pipe line is NOT a table — avoids false positives in prose
        lines = ["| some data |"]
        assert _is_table_block(lines) is False

    def test_exactly_threshold_lines_required(self):
        # TABLE_LINE_THRESHOLD lines minimum — just below should fail
        lines = ["| col1 | col2 |"] * (TABLE_LINE_THRESHOLD - 1)
        assert _is_table_block(lines) is False

    def test_mixed_block_below_40pct_not_table(self):
        # 1 pipe line out of 5 total = 20% — below 40% threshold
        lines = [
            "Revenue grew 6 percent this quarter.",
            "Services hit an all-time high.",
            "iPhone demand was strong.",
            "Mac sales were steady.",
            "| segment | revenue |",
        ]
        assert _is_table_block(lines) is False

    def test_blank_lines_ignored_in_ratio(self):
        lines = ["", "  ", "| col1 |", "| col2 |", "| col3 |", ""]
        assert _is_table_block(lines) is True


# ── _make_chunk_id ────────────────────────────────────────────────────────────


class TestMakeChunkId:
    def test_deterministic_same_inputs(self):
        id1 = _make_chunk_id(TICKER, DATE, 0)
        id2 = _make_chunk_id(TICKER, DATE, 0)
        assert id1 == id2

    def test_different_index_gives_different_id(self):
        assert _make_chunk_id(TICKER, DATE, 0) != _make_chunk_id(TICKER, DATE, 1)

    def test_different_ticker_gives_different_id(self):
        assert _make_chunk_id("AAPL", DATE, 0) != _make_chunk_id("NVDA", DATE, 0)

    def test_different_date_gives_different_id(self):
        assert _make_chunk_id(TICKER, "2024-01-01", 0) != _make_chunk_id(TICKER, "2025-01-01", 0)

    def test_suffix_appended(self):
        chunk_id = _make_chunk_id(TICKER, DATE, 0, "tbl")
        assert chunk_id.endswith("_tbl")

    def test_no_suffix_no_trailing_underscore(self):
        chunk_id = _make_chunk_id(TICKER, DATE, 0)
        assert not chunk_id.endswith("_")

    def test_id_contains_ticker_and_date(self):
        chunk_id = _make_chunk_id(TICKER, DATE, 0)
        assert TICKER in chunk_id
        assert DATE in chunk_id

    def test_uuid5_namespace_isolation(self):
        # Different companies on same date should never collide
        ids = {_make_chunk_id(t, DATE, 0) for t in ["AAPL", "NVDA", "MSFT", "AMZN", "META"]}
        assert len(ids) == 5


# ── _contextual_prefix ────────────────────────────────────────────────────────


class TestContextualPrefix:
    def test_contains_ticker(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Revenue")
        assert TICKER in prefix

    def test_contains_date(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Revenue")
        assert DATE in prefix

    def test_contains_doc_type(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Revenue")
        assert DOCTYPE in prefix

    def test_contains_section_title(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Revenue")
        assert "Revenue" in prefix

    def test_table_sentinel_not_included_in_prefix(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "__TABLE__")
        assert "__TABLE__" not in prefix

    def test_empty_section_title_omitted(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "")
        assert "Section:" not in prefix

    def test_ends_with_real_newline(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Revenue")
        # Must end with a real newline, NOT the literal string "\\n"
        assert prefix.endswith("\n\n"), (
            f"Prefix ends with {repr(prefix[-4:])} — expected real newlines, "
            "not escaped backslash-n literals"
        )

    def test_no_literal_backslash_n(self):
        prefix = _contextual_prefix(TICKER, DATE, DOCTYPE, "Outlook")
        assert "\\n" not in prefix, (
            "Prefix contains literal '\\\\n' string — newline escape bug not fixed"
        )


# ── _split_into_semantic_sections ────────────────────────────────────────────


class TestSplitIntoSemanticSections:
    def test_returns_list_of_tuples(self):
        result = _split_into_semantic_sections([EARNINGS_PROSE])
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_empty_input_returns_empty(self):
        assert _split_into_semantic_sections([]) == []

    def test_below_word_floor_filtered_out(self):
        # Under 10 words — should be dropped
        result = _split_into_semantic_sections(["too short"])
        assert result == []

    def test_section_header_triggers_split(self):
        text = EARNINGS_PROSE + "\nOutlook\n" + OUTLOOK_PROSE
        result = _split_into_semantic_sections([text])
        titles = [t for t, _ in result]
        assert any("Outlook" in t for t in titles), (
            "Expected 'Outlook' header to trigger a split but no section found with that title"
        )

    def test_markdown_header_triggers_split(self):
        text = EARNINGS_PROSE + "\n## Segment Results\n" + OUTLOOK_PROSE
        result = _split_into_semantic_sections([text])
        titles = [t for t, _ in result]
        assert any("Segment Results" in t for t in titles)

    def test_table_emits_table_sentinel(self):
        sections = [EARNINGS_PROSE + "\n" + MARKDOWN_TABLE]
        result = _split_into_semantic_sections(sections)
        sentinels = [t for t, _ in result if t == "__TABLE__"]
        assert len(sentinels) >= 1, "Markdown table should produce a __TABLE__ sentinel"

    def test_real_newlines_used_not_escaped(self):
        # Regression: splitting on "\\n" (literal) = no split at all
        # If the bug exists, a multi-line section comes back as one block
        text = EARNINGS_PROSE + "\nOutlook\n" + OUTLOOK_PROSE
        result = _split_into_semantic_sections([text])
        # At least 2 sections expected if newlines work correctly
        assert len(result) >= 2, (
            f"Got {len(result)} section(s) — expected >=2. "
            "Likely caused by splitting on literal '\\\\n' instead of real newline."
        )

    def test_prose_without_headers_stays_intact(self):
        result = _split_into_semantic_sections([EARNINGS_PROSE])
        assert len(result) == 1
        _, text = result[0]
        assert len(text.split()) >= 10


# ── create_parent_child_chunks (public API) ───────────────────────────────────


class TestCreateParentChildChunks:
    def _run(self, sections=None, ticker=TICKER, date=DATE, doc_type=DOCTYPE):
        if sections is None:
            sections = [EARNINGS_PROSE]
        return create_parent_child_chunks(ticker, date, sections, doc_type)

    def test_returns_list_of_chunks(self):
        chunks = self._run()
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_sections_returns_empty(self):
        assert create_parent_child_chunks(TICKER, DATE, [], DOCTYPE) == []

    def test_all_below_word_floor_returns_empty(self):
        assert create_parent_child_chunks(TICKER, DATE, ["too short"], DOCTYPE) == []

    def test_parent_chunks_present(self):
        chunks = self._run()
        parents = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parents) >= 1

    def test_child_chunks_present(self):
        chunks = self._run()
        children = [c for c in chunks if c.chunk_type == "child"]
        assert len(children) >= 1

    def test_every_child_has_valid_parent_id(self):
        chunks = self._run([_long_prose(300)])
        parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
        for child in [c for c in chunks if c.chunk_type == "child"]:
            assert child.parent_id in parent_ids, (
                f"Child {child.chunk_id} has parent_id={child.parent_id!r} "
                "which does not match any parent chunk_id"
            )

    def test_parent_parent_id_is_none(self):
        chunks = self._run()
        for parent in [c for c in chunks if c.chunk_type == "parent"]:
            assert parent.parent_id is None

    def test_ordering_parent_before_its_children(self):
        chunks = self._run([_long_prose(300)])
        seen_parents: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_type == "parent":
                seen_parents.add(chunk.chunk_id)
            elif chunk.chunk_type == "child":
                assert chunk.parent_id in seen_parents, (
                    f"Child {chunk.chunk_id} appeared before its parent in the list"
                )

    def test_chunk_ids_are_unique(self):
        chunks = self._run([_long_prose(300)])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_all_chunks_have_ticker(self):
        chunks = self._run()
        for c in chunks:
            assert c.ticker == TICKER

    def test_all_chunks_have_date(self):
        chunks = self._run()
        for c in chunks:
            assert c.date == DATE

    def test_all_chunks_have_doc_type(self):
        chunks = self._run()
        for c in chunks:
            assert c.doc_type == DOCTYPE

    def test_all_chunks_have_non_empty_text(self):
        chunks = self._run([_long_prose(300)])
        for c in chunks:
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"

    def test_parent_text_contains_contextual_prefix(self):
        chunks = self._run()
        for parent in [c for c in chunks if c.chunk_type == "parent"]:
            assert TICKER in parent.text
            assert DATE in parent.text

    def test_parent_text_uses_real_newlines(self):
        chunks = self._run([EARNINGS_PROSE + "\nOutlook\n" + OUTLOOK_PROSE])
        for parent in [c for c in chunks if c.chunk_type == "parent"]:
            assert "\\n" not in parent.text, (
                f"Parent {parent.chunk_id} contains literal '\\\\n' — newline bug not fixed"
            )

    def test_metadata_keys_present_on_parent(self):
        chunks = self._run()
        required_keys = {
            "ticker",
            "date",
            "doc_type",
            "chunk_index",
            "section",
            "has_overlap",
            "is_table",
        }
        for parent in [c for c in chunks if c.chunk_type == "parent"]:
            missing = required_keys - parent.metadata.keys()
            assert not missing, f"Parent {parent.chunk_id} missing metadata keys: {missing}"

    def test_metadata_keys_present_on_child(self):
        chunks = self._run()
        required_keys = {"ticker", "date", "doc_type", "parent_id", "child_index", "is_table"}
        for child in [c for c in chunks if c.chunk_type == "child"]:
            missing = required_keys - child.metadata.keys()
            assert not missing, f"Child {child.chunk_id} missing metadata keys: {missing}"

    def test_child_index_sequential_per_parent(self):
        chunks = self._run([_long_prose(300)])
        from collections import defaultdict

        children_by_parent: dict[str, list[int]] = defaultdict(list)
        for c in chunks:
            if c.chunk_type == "child":
                children_by_parent[c.parent_id].append(c.metadata["child_index"])
        for parent_id, indices in children_by_parent.items():
            assert indices == list(range(len(indices))), (
                f"Child indices for parent {parent_id} are not sequential: {indices}"
            )

    def test_chunk_ids_deterministic_across_runs(self):
        chunks1 = self._run([EARNINGS_PROSE])
        chunks2 = self._run([EARNINGS_PROSE])
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2, "Chunk IDs are not deterministic across runs"


# ── Table protection ──────────────────────────────────────────────────────────


class TestTableProtection:
    def _run_with_table(self):
        sections = [EARNINGS_PROSE, MARKDOWN_TABLE, OUTLOOK_PROSE]
        return create_parent_child_chunks(TICKER, DATE, sections, DOCTYPE)

    def test_table_chunk_type_present(self):
        chunks = self._run_with_table()
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) >= 1

    def test_table_chunk_is_table_in_metadata(self):
        chunks = self._run_with_table()
        for c in [c for c in chunks if c.chunk_type == "table"]:
            assert c.metadata.get("is_table") is True

    def test_table_never_merged_into_text_parent(self):
        chunks = self._run_with_table()
        for parent in [c for c in chunks if c.chunk_type == "parent"]:
            # Text parents must not contain table sentinel
            assert "__TABLE__" not in parent.text

    def test_table_child_has_same_text_as_parent(self):
        chunks = self._run_with_table()
        table_parents = {c.chunk_id: c for c in chunks if c.chunk_type == "table"}
        table_children = [
            c for c in chunks if c.chunk_type == "child" and c.metadata.get("is_table")
        ]
        for child in table_children:
            parent = table_parents.get(child.parent_id)
            assert parent is not None, f"Table child {child.chunk_id} has no matching table parent"
            assert child.text == parent.text, (
                "Table child text should be identical to table parent text"
            )

    def test_table_produces_exactly_one_child(self):
        chunks = self._run_with_table()
        table_parents = [c for c in chunks if c.chunk_type == "table"]
        for tp in table_parents:
            children = [c for c in chunks if c.chunk_type == "child" and c.parent_id == tp.chunk_id]
            assert len(children) == 1, (
                f"Table parent {tp.chunk_id} should produce exactly 1 child, got {len(children)}"
            )

    def test_table_section_title_is_financial_table(self):
        chunks = self._run_with_table()
        for c in [c for c in chunks if c.chunk_type == "table"]:
            assert c.section_title == "Financial Table"

    def test_surrounding_prose_not_contaminated_by_table(self):
        chunks = self._run_with_table()
        text_parents = [c for c in chunks if c.chunk_type == "parent"]
        for tp in text_parents:
            assert tp.metadata.get("is_table") is False


# ── Overlap behaviour ─────────────────────────────────────────────────────────


class TestOverlapBehaviour:
    def test_multiple_parents_generated_for_long_input(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [_long_prose(1200)], DOCTYPE)
        parents = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parents) >= 2, (
            f"Expected multiple parents for 1200-word input, got {len(parents)}"
        )

    def test_later_parents_may_have_overlap_flag(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [_long_prose(1200)], DOCTYPE)
        parents = [c for c in chunks if c.chunk_type == "parent"]
        overlap_flags = [p.metadata.get("has_overlap", False) for p in parents]
        # At least one parent after the first should carry overlap
        assert any(overlap_flags[1:]), (
            "No parent after the first has has_overlap=True — overlap logic may be broken"
        )

    def test_children_have_word_overlap_with_siblings(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [_long_prose(300)], DOCTYPE)
        children = [c for c in chunks if c.chunk_type == "child" and not c.metadata.get("is_table")]
        if len(children) < 2:
            pytest.skip("Not enough children to test overlap")
        for i in range(len(children) - 1):
            words_a = set(children[i].text.split())
            words_b = set(children[i + 1].text.split())
            shared = words_a & words_b
            assert len(shared) > 0, f"Children {i} and {i + 1} share no words — overlap is missing"


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_section_minimum_viable(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [EARNINGS_PROSE], DOCTYPE)
        assert len(chunks) >= 2  # at least 1 parent + 1 child

    def test_multiple_companies_produce_distinct_ids(self):
        tickers = ["AAPL", "NVDA", "MSFT"]
        all_ids: set[str] = set()
        for t in tickers:
            chunks = create_parent_child_chunks(t, DATE, [EARNINGS_PROSE], DOCTYPE)
            for c in chunks:
                all_ids.add(c.chunk_id)
        # No ID collision across different tickers
        total = sum(
            len(create_parent_child_chunks(t, DATE, [EARNINGS_PROSE], DOCTYPE)) for t in tickers
        )
        assert len(all_ids) == total

    def test_all_doc_types_accepted(self):
        for dt in ["10-K", "10-Q", "earnings_release", "transcript", "8-K"]:
            chunks = create_parent_child_chunks(TICKER, DATE, [EARNINGS_PROSE], dt)
            assert len(chunks) >= 1, f"doc_type={dt!r} produced no chunks"

    def test_section_with_only_whitespace_skipped(self):
        chunks = create_parent_child_chunks(TICKER, DATE, ["   \n\n   \t  "], DOCTYPE)
        assert chunks == []

    def test_very_long_single_section_splits_into_multiple_parents(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [_long_prose(2000)], DOCTYPE)
        parents = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parents) >= 3

    def test_table_only_input(self):
        chunks = create_parent_child_chunks(TICKER, DATE, [MARKDOWN_TABLE], DOCTYPE)
        # Should produce at least a table chunk + its child
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) >= 1
