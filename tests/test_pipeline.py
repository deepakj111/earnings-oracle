"""
Integration test for ingestion/pipeline.py

Everything external is mocked:
  - Gemini (embeddings)
  - Qdrant (upsert + collections)
  - File system is real (tmp_path)

We verify the pipeline calls the right components in the right order,
produces BM25 texts, and handles empty/skippable files gracefully.
"""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

VECTOR_DIM = 768
_BODY = " ".join(["revenue"] * 120)

VALID_HTML = f"""
<html><body>
<p>{_BODY}</p>
<p>Services reached a new all time high of 24 point 9 billion dollars representing
fourteen percent growth versus prior year comparable period of fiscal operations.</p>
</body></html>
"""

SHORT_HTML = "<html><body><p>Too short.</p></body></html>"


def _fake_embedding():
    arr = np.random.rand(VECTOR_DIM).astype(np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _mock_qdrant():
    mock = MagicMock()
    mock.get_collections.return_value.collections = []
    return mock


def _mock_genai():
    mock = MagicMock()
    mock.models.embed_content.return_value.embeddings = [MagicMock(values=_fake_embedding())]
    return mock


@pytest.fixture
def transcript_dir(tmp_path):
    d = tmp_path / "data" / "transcripts"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def bm25_path(tmp_path):
    return tmp_path / "data" / "bm25_index.pkl"


class TestRunPipeline:
    def _run(self, transcript_dir, bm25_path):
        mock_qdrant_client = _mock_qdrant()
        mock_genai_client = _mock_genai()

        with (
            patch("ingestion.pipeline.TRANSCRIPTS_DIR", transcript_dir),
            patch("ingestion.pipeline.BM25_INDEX_PATH", bm25_path),
            patch("ingestion.pipeline.init_qdrant", return_value=mock_qdrant_client),
            patch("ingestion.pipeline.setup_genai"),
            patch("ingestion.indexer._genai_client", mock_genai_client),
            patch("ingestion.indexer.time.sleep"),
        ):
            from ingestion.pipeline import run_pipeline

            run_pipeline()

        return mock_qdrant_client

    def test_pipeline_runs_without_error(self, transcript_dir, bm25_path):
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)

    def test_valid_file_triggers_qdrant_upsert(self, transcript_dir, bm25_path):
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        mock_qdrant = self._run(transcript_dir, bm25_path)
        assert mock_qdrant.upsert.called

    def test_short_file_is_skipped(self, transcript_dir, bm25_path):
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(SHORT_HTML, encoding="utf-8")
        mock_qdrant = self._run(transcript_dir, bm25_path)
        mock_qdrant.upsert.assert_not_called()

    def test_bm25_index_written_to_disk(self, transcript_dir, bm25_path):
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)
        assert bm25_path.exists()

    def test_bm25_file_is_valid_pickle(self, transcript_dir, bm25_path):
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)
        assert obj is not None

    def test_empty_transcripts_dir_produces_no_upsert(self, transcript_dir, bm25_path):
        mock_qdrant = self._run(transcript_dir, bm25_path)
        mock_qdrant.upsert.assert_not_called()

    def test_multiple_valid_files_all_indexed(self, transcript_dir, bm25_path):
        tickers = [("AAPL", "2024-10-31"), ("NVDA", "2024-07-15"), ("MSFT", "2024-04-30")]
        for ticker, date in tickers:
            (transcript_dir / f"{ticker}_{date}_0001234567.htm").write_text(
                VALID_HTML, encoding="utf-8"
            )
        mock_qdrant = self._run(transcript_dir, bm25_path)
        assert mock_qdrant.upsert.call_count >= 3
