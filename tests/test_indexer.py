from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.chunker import Chunk
from ingestion.indexer import (
    COLLECTION_NAME,
    UPSERT_BATCH_SIZE,
    VECTOR_DIM,
    _get_embedding,
    index_document,
)
from ingestion.metadata_extractor import DocumentMetadata


def _fake_embedding(val: float = 0.1) -> list[float]:
    raw = np.full(VECTOR_DIM, val, dtype=np.float32)
    norm = np.linalg.norm(raw)
    return (raw / norm).tolist()


def _make_chunk(chunk_type: str, index: int, parent_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"AAPL_2024-01-01_abc_{index}",
        parent_id=parent_id,
        ticker="AAPL",
        date="2024-01-01",
        doc_type="earnings_release",
        chunk_type=chunk_type,
        text="Revenue grew strongly in the quarter. Earnings per share rose significantly.",
        section_title="Financial Highlights",
        metadata={
            "ticker": "AAPL",
            "date": "2024-01-01",
            "doc_type": "earnings_release",
            "chunk_index": index,
            "section": "Financial Highlights",
            "has_overlap": False,
            "is_table": False,
        },
    )


def _make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        ticker="AAPL",
        company="Apple",
        date="2024-01-01",
        year=2024,
        quarter="Q1",
        fiscal_period="Q1 2024",
    )


def _mock_embed_model(vec: list[float] | None = None) -> MagicMock:
    """
    Return a MagicMock that mimics fastembed TextEmbedding.
    Each call to .embed([text]) must return a fresh iterable of numpy arrays.
    Using side_effect ensures a new iterator is created per call.
    """
    embedding = np.array(vec if vec is not None else _fake_embedding(), dtype=np.float32)
    mock = MagicMock()
    mock.embed.side_effect = lambda texts: iter([embedding for _ in texts])
    return mock


class TestGetEmbedding:
    def test_returns_list_of_floats(self):
        mock_model = _mock_embed_model()
        with patch("ingestion.indexer._embed_model", mock_model):
            result = _get_embedding("Revenue grew 6 percent.")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_correct_dimension(self):
        mock_model = _mock_embed_model()
        with patch("ingestion.indexer._embed_model", mock_model):
            result = _get_embedding("Revenue grew 6 percent.")
        assert len(result) == VECTOR_DIM

    def test_output_is_unit_normalized(self):
        raw = np.array([3.0, 4.0] + [0.0] * (VECTOR_DIM - 2), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.embed.side_effect = lambda texts: iter([raw for _ in texts])
        with patch("ingestion.indexer._embed_model", mock_model):
            result = _get_embedding("Revenue grew 6 percent.")
        norm = np.linalg.norm(np.array(result, dtype=np.float32))
        assert abs(norm - 1.0) < 1e-5

    def test_raises_if_client_not_initialized(self):
        with patch("ingestion.indexer._embed_model", None):
            with pytest.raises(AssertionError, match="setup_embedder"):
                _get_embedding("Revenue grew 6 percent.")


class TestIndexDocument:
    def _run(
        self,
        chunks: list,
        existing_bm25: list | None = None,
        existing_corpus: list | None = None,
        metadata: DocumentMetadata | None = None,
    ) -> tuple[list[list[str]], list[dict], MagicMock, MagicMock]:
        """
        Helper that calls index_document with mocked embedder and Qdrant client.

        Returns (bm25_texts, bm25_corpus, mock_qdrant, mock_embed_model)
        so each test can assert on any of the four outputs independently.
        """
        mock_model = _mock_embed_model()
        mock_qdrant = MagicMock()
        bm25_texts_in = list(existing_bm25) if existing_bm25 is not None else []
        bm25_corpus_in = list(existing_corpus) if existing_corpus is not None else []

        with patch("ingestion.indexer._embed_model", mock_model):
            bm25_texts, bm25_corpus = index_document(
                chunks,
                metadata or _make_metadata(),
                mock_qdrant,
                bm25_texts_in,
                bm25_corpus_in,
            )
        return bm25_texts, bm25_corpus, mock_qdrant, mock_model

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_tuple_of_two_lists(self):
        chunks = [_make_chunk("child", 0, "parent_0")]
        bm25_texts, bm25_corpus, _, _ = self._run(chunks)
        assert isinstance(bm25_texts, list)
        assert isinstance(bm25_corpus, list)

    def test_bm25_texts_and_corpus_always_same_length(self):
        children = [_make_chunk("child", i, "parent_0") for i in range(7)]
        bm25_texts, bm25_corpus, _, _ = self._run(children)
        assert len(bm25_texts) == len(bm25_corpus)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def test_only_children_are_embedded(self):
        parent = _make_chunk("parent", 0)
        child = _make_chunk("child", 1, parent.chunk_id)
        table = _make_chunk("table", 2)
        _, _, _, mock_model = self._run([parent, child, table])
        assert mock_model.embed.call_count == 1

    def test_no_children_produces_no_embed_calls(self):
        parent = _make_chunk("parent", 0)
        _, _, _, mock_model = self._run([parent])
        mock_model.embed.assert_not_called()

    # ── BM25 texts ────────────────────────────────────────────────────────────

    def test_bm25_texts_grows_by_child_count(self):
        children = [_make_chunk("child", i, "parent_0") for i in range(5)]
        bm25_texts, _, _, _ = self._run(children, existing_bm25=[])
        assert len(bm25_texts) == 5

    def test_bm25_texts_appends_to_existing(self):
        existing = [["previous", "text"]]
        child = _make_chunk("child", 0, "parent_0")
        bm25_texts, _, _, _ = self._run([child], existing_bm25=existing)
        assert len(bm25_texts) == 2
        assert bm25_texts[0] == ["previous", "text"]

    def test_bm25_texts_are_lowercase_token_lists(self):
        child = _make_chunk("child", 0, "parent_0")
        bm25_texts, _, _, _ = self._run([child])
        assert isinstance(bm25_texts[0], list)
        assert all(token == token.lower() for token in bm25_texts[0])

    # ── BM25 corpus ───────────────────────────────────────────────────────────

    def test_bm25_corpus_grows_by_child_count(self):
        children = [_make_chunk("child", i, "parent_0") for i in range(5)]
        _, bm25_corpus, _, _ = self._run(children)
        assert len(bm25_corpus) == 5

    def test_bm25_corpus_appends_to_existing(self):
        existing_corpus = [{"chunk_id": "old_chunk", "ticker": "MSFT"}]
        child = _make_chunk("child", 0, "parent_0")
        _, bm25_corpus, _, _ = self._run([child], existing_corpus=existing_corpus)
        assert len(bm25_corpus) == 2
        assert bm25_corpus[0]["chunk_id"] == "old_chunk"

    def test_bm25_corpus_entry_has_required_fields(self):
        child = _make_chunk("child", 0, "parent_0")
        _, bm25_corpus, _, _ = self._run([child])
        entry = bm25_corpus[0]
        required = {
            "chunk_id",
            "parent_id",
            "text",
            "ticker",
            "company",
            "date",
            "year",
            "quarter",
            "fiscal_period",
            "section_title",
        }
        assert required.issubset(entry.keys())

    def test_bm25_corpus_entry_values_match_metadata(self):
        child = _make_chunk("child", 0, "parent_0")
        meta = _make_metadata()
        _, bm25_corpus, _, _ = self._run([child], metadata=meta)
        entry = bm25_corpus[0]
        assert entry["ticker"] == "AAPL"
        assert entry["company"] == "Apple"
        assert entry["year"] == 2024
        assert entry["quarter"] == "Q1"
        assert entry["fiscal_period"] == "Q1 2024"

    def test_bm25_corpus_entry_chunk_id_matches_chunk(self):
        child = _make_chunk("child", 3, "parent_0")
        _, bm25_corpus, _, _ = self._run([child])
        assert bm25_corpus[0]["chunk_id"] == "AAPL_2024-01-01_abc_3"

    def test_bm25_corpus_entry_parent_id_matches_chunk(self):
        child = _make_chunk("child", 0, "AAPL_2024-01-01_abc_parent")
        _, bm25_corpus, _, _ = self._run([child])
        assert bm25_corpus[0]["parent_id"] == "AAPL_2024-01-01_abc_parent"

    def test_bm25_corpus_entry_section_title_stored(self):
        child = _make_chunk("child", 0, "parent_0")
        _, bm25_corpus, _, _ = self._run([child])
        assert bm25_corpus[0]["section_title"] == "Financial Highlights"

    def test_bm25_texts_and_corpus_index_alignment(self):
        # The i-th token list in bm25_texts must correspond to bm25_corpus[i].
        # Verify by checking that the text field in corpus tokenises to the same
        # tokens as bm25_texts[i].
        children = [_make_chunk("child", i, "parent_0") for i in range(3)]
        bm25_texts, bm25_corpus, _, _ = self._run(children)
        for tokens, entry in zip(bm25_texts, bm25_corpus, strict=False):
            expected_tokens = entry["text"].lower().split()
            assert tokens == expected_tokens

    # ── Qdrant upsert ─────────────────────────────────────────────────────────

    def test_qdrant_upsert_called(self):
        child = _make_chunk("child", 0, "parent_0")
        _, _, mock_qdrant, _ = self._run([child])
        mock_qdrant.upsert.assert_called_once()

    def test_upsert_uses_correct_collection(self):
        child = _make_chunk("child", 0, "parent_0")
        _, _, mock_qdrant, _ = self._run([child])
        call_kwargs = mock_qdrant.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    def test_point_payload_has_required_fields(self):
        child = _make_chunk("child", 0, "parent_0")
        _, _, mock_qdrant, _ = self._run([child])
        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        required = {
            "chunk_id",
            "parent_id",
            "text",
            "ticker",
            "company",
            "date",
            "year",
            "quarter",
            "fiscal_period",
            "section_title",
        }
        assert required.issubset(point.payload.keys())

    def test_point_payload_ticker_matches_metadata(self):
        child = _make_chunk("child", 0, "parent_0")
        meta = _make_metadata()
        _, _, mock_qdrant, _ = self._run([child], metadata=meta)
        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        assert point.payload["ticker"] == "AAPL"
        assert point.payload["company"] == "Apple"

    def test_point_payload_section_title_stored(self):
        child = _make_chunk("child", 0, "parent_0")
        _, _, mock_qdrant, _ = self._run([child])
        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        assert point.payload["section_title"] == "Financial Highlights"

    def test_batching_for_large_input(self):
        n_children = UPSERT_BATCH_SIZE * 3 + 1
        children = [_make_chunk("child", i, "parent_0") for i in range(n_children)]
        _, _, mock_qdrant, _ = self._run(children)
        assert mock_qdrant.upsert.call_count == 4  # ceil(151 / 50) = 4

    def test_no_children_produces_no_upsert(self):
        parent = _make_chunk("parent", 0)
        _, _, mock_qdrant, _ = self._run([parent])
        mock_qdrant.upsert.assert_not_called()
