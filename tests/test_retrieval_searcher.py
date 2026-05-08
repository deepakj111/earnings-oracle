"""
Tests for retrieval/searcher.py — RRF fusion, filter building, BM25 search.

Dense search and parent fetch require live Qdrant — tested with mocks.
BM25 search is tested with an in-memory BM25 index to avoid disk dependency.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from query.models import TransformedQuery
from retrieval.models import MetadataFilter, SearchResult
from retrieval.searcher import (
    _bm25_search,
    _build_qdrant_filter,
    _embed,
    _fetch_parent_texts,
    _qdrant_search,
    _rrf_fuse,
    search,
    warmup_bm25,
    warmup_embed_client,
)


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


class TestQdrantSearch:
    def test_qdrant_search_returns_payloads(self) -> None:
        from qdrant_client import QdrantClient

        client = Mock(spec=QdrantClient)

        class MockHit:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

        class MockPoints:
            def __init__(self, points: list) -> None:
                self.points = points

        client.query_points.return_value = MockPoints([MockHit({"chunk_id": "c1"})])

        with patch("retrieval.searcher._embed", return_value=[0.1, 0.2]):
            res = _qdrant_search(client, "query", 5, None)
            assert len(res) == 1
            assert res[0]["chunk_id"] == "c1"


class TestFetchParentTexts:
    def test_disabled_in_config(self) -> None:
        with patch("retrieval.searcher.settings") as mock_settings:
            mock_settings.retrieval.parent_fetch_enabled = False
            results = [
                SearchResult.from_payload({"chunk_id": "c1", "text": "text", "parent_id": "p1"}, rrf_score=1.0, source="dense")
            ]
            assert _fetch_parent_texts(Mock(), results) == results
            assert results[0].parent_text == "text"

    def test_no_parents_needed(self) -> None:
        with patch("retrieval.searcher.settings") as mock_settings:
            mock_settings.retrieval.parent_fetch_enabled = True
            results = [
                SearchResult.from_payload({"chunk_id": "c1", "text": "text", "parent_id": None}, rrf_score=1.0, source="dense")
            ]
            assert _fetch_parent_texts(Mock(), results) == results

    def test_fetch_success(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Record

        client = Mock(spec=QdrantClient)
        # scroll returns (list of points, next_page_offset)
        client.scroll.return_value = (
            [
                Record(id="p1", payload={"chunk_id": "p1", "text": "parent 1 text"}),
                Record(id="p2", payload={"chunk_id": "p2", "text": "parent 2 text"}),
            ],
            None,
        )

        with patch("retrieval.searcher.settings") as mock_settings:
            mock_settings.retrieval.parent_fetch_enabled = True
            results = [
                SearchResult.from_payload({"chunk_id": "c1", "text": "t1", "parent_id": "p1"}, rrf_score=1.0, source="dense"),
                SearchResult.from_payload({"chunk_id": "c2", "text": "t2", "parent_id": "p2"}, rrf_score=0.9, source="bm25"),
                SearchResult.from_payload({"chunk_id": "c3", "text": "t3", "parent_id": "p3"}, rrf_score=0.8, source="dense"),
            ]

            res = _fetch_parent_texts(client, results)
            assert res[0].parent_text == "parent 1 text"
            assert res[1].parent_text == "parent 2 text"
            assert res[2].parent_text == "t3"  # Not found in returned records

    def test_fetch_exception_returns_original(self) -> None:
        from qdrant_client import QdrantClient

        client = Mock(spec=QdrantClient)
        client.scroll.side_effect = Exception("Network error")

        with patch("retrieval.searcher.settings") as mock_settings:
            mock_settings.retrieval.parent_fetch_enabled = True
            results = [
                SearchResult.from_payload({"chunk_id": "c1", "text": "text1", "parent_id": "p1"}, rrf_score=1.0, source="dense"),
            ]

            res = _fetch_parent_texts(client, results)
            assert res[0].parent_text == "text1"


class TestSearch:
    def test_search_all_variants_failed(self) -> None:
        query = TransformedQuery(
            original="test",
            hyde_document="hyde",
            multi_queries=["q1"],
            stepback_query="step",
        )
        from qdrant_client import QdrantClient

        client = Mock(spec=QdrantClient)

        with (
            patch("retrieval.searcher._qdrant_search") as mock_qsearch,
            patch("retrieval.searcher._bm25_search") as mock_bsearch,
        ):
            mock_qsearch.side_effect = Exception("qdrant fail")
            mock_bsearch.side_effect = Exception("bm25 fail")

            res = search(query, client)
            assert res == []

    def test_search_success(self) -> None:
        query = TransformedQuery(
            original="test",
            hyde_document="hyde",
            multi_queries=["q1"],
            stepback_query="step",
        )
        from qdrant_client import QdrantClient

        client = Mock(spec=QdrantClient)

        with (
            patch("retrieval.searcher._qdrant_search") as mock_qsearch,
            patch("retrieval.searcher._bm25_search") as mock_bsearch,
            patch("retrieval.searcher._fetch_parent_texts") as mock_fetch,
            patch("retrieval.searcher.settings") as mock_settings,
        ):
            mock_settings.retrieval.top_k_dense = 2
            mock_settings.retrieval.top_k_bm25 = 2
            mock_settings.retrieval.rrf_k_constant = 60
            mock_settings.reranker.top_k_pre_rerank = 5

            mock_qsearch.side_effect = [
                [{"chunk_id": "c1", "text": "t1"}],  # hyde
                [{"chunk_id": "c2", "text": "t2"}],  # q1 dense
            ]
            mock_bsearch.side_effect = [
                [{"chunk_id": "c2", "text": "t2", "bm25_score": 1.0}],  # q1 bm25
            ]
            mock_fetch.side_effect = lambda c, res: res

            res = search(query, client)
            assert len(res) == 2
            assert any(r.chunk_id == "c1" for r in res)
            assert any(r.chunk_id == "c2" for r in res)


def test_embed() -> None:
    class MockClient:
        def embed(self, texts: list[str]) -> list:
            import numpy as np

            return [np.array([0.1, 0.2])]

    with patch("retrieval.searcher._get_embed_client", return_value=MockClient()):
        assert _embed("test") == [0.1, 0.2]


def test_warmups() -> None:
    with patch("retrieval.searcher._get_embed_client") as m1:
        warmup_embed_client()
        m1.assert_called_once()

    with patch("retrieval.searcher._load_bm25") as m2:
        warmup_bm25()
        m2.assert_called_once()
