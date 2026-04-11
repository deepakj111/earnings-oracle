"""
Tests for ingestion/indexer.py

External dependencies (Gemini, Qdrant) are fully mocked.
We test the logic — batching, normalization, payload structure —
not the APIs themselves.
Coverage:
  - _get_embedding returns normalized float list
  - index_document only embeds children, not parents/tables
  - Qdrant upsert called with correct batch sizes
  - Point payload contains all required fields
  - BM25 text list grows correctly
  - init_qdrant skips collection creation if already exists
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from ingestion.chunker import Chunk
from ingestion.metadata_extractor import DocumentMetadata
from ingestion.indexer import (
    COLLECTION_NAME,
    UPSERT_BATCH_SIZE,
    VECTOR_DIM,
    _get_embedding,
    index_document,
    init_qdrant,
    setup_genai,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunk(chunk_type: str, index: int = 0, parent_id: str = None) -> Chunk:
    return Chunk(
        chunk_id      = f"AAPL_2024-10-31_abc12345_{index}",
        parent_id     = parent_id,
        ticker        = "AAPL",
        date          = "2024-10-31",
        doc_type      = "earnings_release",
        chunk_type    = chunk_type,
        text          = f"[Context: AAPL | earnings_release | 2024-10-31] Sample text {index}.",
        section_title = "Revenue",
        metadata      = {
            "ticker": "AAPL", "date": "2024-10-31",
            "doc_type": "earnings_release", "chunk_index": index,
            "section": "Revenue", "has_overlap": False, "is_table": False,
            "parent_id": parent_id, "child_index": index,
        },
    )


def _make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        ticker="AAPL", company="Apple", date="2024-10-31",
        year=2024, quarter="Q4", fiscal_period="Q4 2024",
    )


def _fake_embedding(dim: int = VECTOR_DIM) -> list[float]:
    arr = np.random.rand(dim).astype(np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


# ── setup_genai / _get_embedding ──────────────────────────────────────────────

class TestGetEmbedding:
    def test_returns_list_of_floats(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value.embeddings = [
            MagicMock(values=_fake_embedding())
        ]
        with patch("ingestion.indexer._genai_client", mock_client):
            result = _get_embedding("Revenue grew 6 percent.")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_correct_dimension(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value.embeddings = [
            MagicMock(values=_fake_embedding())
        ]
        with patch("ingestion.indexer._genai_client", mock_client):
            result = _get_embedding("Revenue grew 6 percent.")
        assert len(result) == VECTOR_DIM

    def test_output_is_unit_normalized(self):
        mock_client = MagicMock()
        raw = np.array([3.0, 4.0] + [0.0] * (VECTOR_DIM - 2), dtype=np.float32)
        mock_client.models.embed_content.return_value.embeddings = [
            MagicMock(values=raw.tolist())
        ]
        with patch("ingestion.indexer._genai_client", mock_client):
            result = _get_embedding("test")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_raises_if_client_not_initialized(self):
        with patch("ingestion.indexer._genai_client", None):
            with pytest.raises(AssertionError):
                _get_embedding("test")


# ── index_document ────────────────────────────────────────────────────────────

class TestIndexDocument:
    def _run(self, chunks, metadata=None, existing_bm25=None):
        if metadata is None:
            metadata = _make_metadata()
        if existing_bm25 is None:
            existing_bm25 = []

        mock_qdrant = MagicMock()
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value.embeddings = [
            MagicMock(values=_fake_embedding())
        ]

        with patch("ingestion.indexer._genai_client", mock_client):
            with patch("ingestion.indexer.time.sleep"):
                result = index_document(chunks, metadata, mock_qdrant, existing_bm25)

        return result, mock_qdrant, mock_client

    def test_returns_list(self):
        chunks = [_make_chunk("child", 0, "parent_0")]
        result, _, _ = self._run(chunks)
        assert isinstance(result, list)

    def test_only_children_are_embedded(self):
        parent = _make_chunk("parent", 0)
        child  = _make_chunk("child",  1, parent.chunk_id)
        table  = _make_chunk("table",  2)
        _, _, mock_client = self._run([parent, child, table])
        assert mock_client.models.embed_content.call_count == 1

    def test_bm25_texts_grows_by_child_count(self):
        children = [_make_chunk("child", i, "parent_0") for i in range(5)]
        result, _, _ = self._run(children, existing_bm25=[])
        assert len(result) == 5

    def test_bm25_appends_to_existing(self):
        existing = [["previous", "text"]]
        child = _make_chunk("child", 0, "parent_0")
        result, _, _ = self._run([child], existing_bm25=existing)
        assert len(result) == 2

    def test_qdrant_upsert_called(self):
        child = _make_chunk("child", 0, "parent_0")
        _, mock_qdrant, _ = self._run([child])
        assert mock_qdrant.upsert.called

    def test_upsert_uses_correct_collection(self):
        child = _make_chunk("child", 0, "parent_0")
        _, mock_qdrant, _ = self._run([child])
        call_kwargs = mock_qdrant.upsert.call_args_list[0][1]
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    def test_point_payload_has_required_fields(self):
        child = _make_chunk("child", 0, "parent_0")
        _, mock_qdrant, _ = self._run([child])
        points = mock_qdrant.upsert.call_args_list[0][1]["points"]
        payload = points[0].payload
        required = {"chunk_id", "parent_id", "text", "ticker",
                    "company", "date", "year", "quarter", "fiscal_period"}
        missing = required - payload.keys()
        assert not missing, f"Payload missing fields: {missing}"

    def test_point_payload_ticker_matches_metadata(self):
        child = _make_chunk("child", 0, "parent_0")
        meta  = _make_metadata()
        _, mock_qdrant, _ = self._run([child], metadata=meta)
        points = mock_qdrant.upsert.call_args_list[0][1]["points"]
        assert points[0].payload["ticker"] == "AAPL"

    def test_batching_for_large_input(self):
        n_children = UPSERT_BATCH_SIZE * 3 + 1
        children = [_make_chunk("child", i, "parent_0") for i in range(n_children)]
        _, mock_qdrant, _ = self._run(children)
        assert mock_qdrant.upsert.call_count == 4  # ceil(n / BATCH_SIZE)

    def test_no_children_produces_no_upsert(self):
        parent = _make_chunk("parent", 0)
        _, mock_qdrant, _ = self._run([parent])
        mock_qdrant.upsert.assert_not_called()


# ── init_qdrant ───────────────────────────────────────────────────────────────

class TestInitQdrant:
    def test_creates_collection_if_not_exists(self):
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        with patch("ingestion.indexer.QdrantClient", return_value=mock_client):
            init_qdrant("http://localhost:6333")
        mock_client.create_collection.assert_called_once()

    def test_skips_creation_if_collection_exists(self):
        mock_col = MagicMock()
        mock_col.name = COLLECTION_NAME
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [mock_col]
        with patch("ingestion.indexer.QdrantClient", return_value=mock_client):
            init_qdrant("http://localhost:6333")
        mock_client.create_collection.assert_not_called()

    def test_returns_qdrant_client(self):
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        with patch("ingestion.indexer.QdrantClient", return_value=mock_client):
            result = init_qdrant("http://localhost:6333")
        assert result is mock_client