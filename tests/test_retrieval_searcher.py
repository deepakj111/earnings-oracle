"""
Tests for retrieval/searcher.py — RRF fusion, filter building, BM25 search.

Dense search and parent fetch require live Qdrant — tested with mocks.
BM25 search is tested with an in-memory BM25 index to avoid disk dependency.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from retrieval.models import MetadataFilter
from retrieval.searcher import _bm25_search, _build_qdrant_filter, _rrf_fuse


class TestBuildQdrantFilter:
    def test_none_filter_returns_none(self) -> None:
        assert _build_qdrant_filter(None) is None

    def test_empty_metadata_filter_returns_none(self) -> None:
        assert _build_qdrant_filter(MetadataFilter()) is None

    def test_ticker_only_builds_filter(self) -> None:
        f = _build_qdrant_filter(MetadataFilter(ticker="AAPL"))
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "ticker"

    def test_year_only_builds_filter(self) -> None:
        f = _build_qdrant_filter(MetadataFilter(year=2024))
        assert f is not None
        assert f.must[0].key == "year"

    def test_quarter_only_builds_filter(self) -> None:
        f = _build_qdrant_filter(MetadataFilter(quarter="Q3"))
        assert f is not None
        assert f.must[0].key == "quarter"

    def test_all_fields_produce_three_conditions(self) -> None:
        f = _build_qdrant_filter(MetadataFilter(ticker="NVDA", year=2024, quarter="Q2"))
        assert f is not None
        assert len(f.must) == 3

    def test_filter_disabled_in_config_returns_none(self) -> None:
        with patch("retrieval.searcher.settings") as mock_settings:
            mock_settings.retrieval.metadata_filter_enabled = False
            result = _build_qdrant_filter(MetadataFilter(ticker="AAPL"))
        assert result is None


class TestRrfFuse:
    def test_single_list_scores_correctly(self) -> None:
        payloads = {"a": {}, "b": {}, "c": {}}
        result = _rrf_fuse([(["a", "b", "c"], "dense")], payloads, k=60)
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_higher_rank_gets_higher_score(self) -> None:
        payloads = {"a": {}, "b": {}}
        result = _rrf_fuse([(["a", "b"], "dense")], payloads, k=60)
        a_score = next(s for i, s, _ in result if i == "a")
        b_score = next(s for i, s, _ in result if i == "b")
        assert a_score > b_score

    def test_chunk_in_two_lists_gets_higher_score(self) -> None:
        payloads = {"shared": {}, "dense_only": {}, "bm25_only": {}}
        lists = [
            (["shared", "dense_only"], "dense"),
            (["shared", "bm25_only"], "bm25"),
        ]
        result = _rrf_fuse(lists, payloads, k=60)
        shared_score = next(s for i, s, _ in result if i == "shared")
        dense_only_score = next(s for i, s, _ in result if i == "dense_only")
        assert shared_score > dense_only_score

    def test_source_is_both_when_in_dense_and_bm25(self) -> None:
        payloads = {"x": {}}
        lists = [(["x"], "dense"), (["x"], "bm25")]
        result = _rrf_fuse(lists, payloads, k=60)
        assert result[0][2] == "both"

    def test_source_is_dense_when_only_in_dense(self) -> None:
        payloads = {"x": {}}
        result = _rrf_fuse([(["x"], "dense")], payloads, k=60)
        assert result[0][2] == "dense"

    def test_source_is_bm25_when_only_in_bm25(self) -> None:
        payloads = {"x": {}}
        result = _rrf_fuse([(["x"], "bm25")], payloads, k=60)
        assert result[0][2] == "bm25"

    def test_empty_input_returns_empty(self) -> None:
        result = _rrf_fuse([], {}, k=60)
        assert result == []

    def test_k_constant_affects_score_magnitude(self) -> None:
        payloads = {"a": {}}
        r_k60 = _rrf_fuse([(["a"], "dense")], payloads, k=60)
        r_k1 = _rrf_fuse([(["a"], "dense")], payloads, k=1)
        assert r_k1[0][1] > r_k60[0][1]


class TestBm25Search:
    def _make_bm25_fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write a minimal BM25 index + corpus to tmp_path and return paths."""
        from rank_bm25 import BM25Okapi

        corpus = [
            {
                "chunk_id": "AAPL_2024-10-31_0",
                "text": "Apple revenue grew in Q1 2024",
                "ticker": "AAPL",
                "year": 2024,
                "quarter": "Q1",
                "parent_id": None,
            },
            {
                "chunk_id": "NVDA_2024-08-28_0",
                "text": "NVIDIA data center revenue exceeded expectations",
                "ticker": "NVDA",
                "year": 2024,
                "quarter": "Q2",
                "parent_id": None,
            },
        ]
        texts = [entry["text"].lower().split() for entry in corpus]
        bm25 = BM25Okapi(texts)

        idx_path = tmp_path / "bm25_index.pkl"
        corpus_path = tmp_path / "bm25_corpus.pkl"
        with open(idx_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus, f)

        return idx_path, corpus_path

    def test_bm25_returns_relevant_result(self, tmp_path) -> None:
        idx_path, corpus_path = self._make_bm25_fixtures(tmp_path)
        with (
            patch("retrieval.searcher._BM25_INDEX_PATH", idx_path),
            patch("retrieval.searcher._BM25_CORPUS_PATH", corpus_path),
            patch("retrieval.searcher._bm25_index", None),
            patch("retrieval.searcher._bm25_corpus", None),
        ):
            results = _bm25_search("Apple revenue", top_k=5, metadata_filter=None)
        assert any("AAPL" in r.get("ticker", "") for r in results)

    def test_bm25_respects_top_k(self, tmp_path) -> None:
        idx_path, corpus_path = self._make_bm25_fixtures(tmp_path)
        with (
            patch("retrieval.searcher._BM25_INDEX_PATH", idx_path),
            patch("retrieval.searcher._BM25_CORPUS_PATH", corpus_path),
            patch("retrieval.searcher._bm25_index", None),
            patch("retrieval.searcher._bm25_corpus", None),
        ):
            results = _bm25_search("revenue", top_k=1, metadata_filter=None)
        assert len(results) <= 1

    def test_bm25_metadata_filter_applied(self, tmp_path) -> None:
        idx_path, corpus_path = self._make_bm25_fixtures(tmp_path)
        with (
            patch("retrieval.searcher._BM25_INDEX_PATH", idx_path),
            patch("retrieval.searcher._BM25_CORPUS_PATH", corpus_path),
            patch("retrieval.searcher._bm25_index", None),
            patch("retrieval.searcher._bm25_corpus", None),
        ):
            results = _bm25_search(
                "revenue", top_k=5, metadata_filter=MetadataFilter(ticker="AAPL")
            )
        assert all(r.get("ticker") == "AAPL" for r in results)

    def test_bm25_missing_index_raises_file_not_found(self) -> None:
        with (
            patch("retrieval.searcher._BM25_INDEX_PATH", Path("/nonexistent/bm25_index.pkl")),
            patch("retrieval.searcher._bm25_index", None),
            patch("retrieval.searcher._bm25_corpus", None),
        ):
            with pytest.raises(FileNotFoundError, match="BM25 index not found"):
                _bm25_search("test", top_k=5, metadata_filter=None)
