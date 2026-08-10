import pickle
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from ingestion.pipeline import run_pipeline

VALID_HTML = """
<html><body>
<h2>Financial Highlights</h2>
<p>Apple Inc. reported record first quarter results for fiscal year 2024.
Revenue was $119.6 billion, up 2 percent year over year. Net income reached
$33.9 billion and diluted earnings per share were $2.18. Services revenue
set an all-time record of $23.1 billion. The board of directors has declared
a cash dividend of $0.24 per share. International sales accounted for 58 percent
of the quarter's revenue. Gross margin was 45.9 percent compared to 42.8 percent
in the year-ago quarter. Operating income was $40.4 billion. The company returned
over $27 billion to shareholders during the quarter through dividends and
share repurchases. Cash and marketable securities ended at $162.1 billion.</p>
<h2>Segment Results</h2>
<p>iPhone revenue was $69.7 billion. Mac revenue was $7.8 billion, up 1 percent.
iPad revenue was $7.0 billion. Wearables, Home and Accessories revenue was $11.9 billion.
Services revenue of $23.1 billion represents continued strong growth in the segment.
Retail and online stores together served millions of customers worldwide this quarter.
Operating expenses were $14.5 billion and research and development was $7.7 billion.</p>
</body></html>
"""

SHORT_HTML = "<html><body><p>Too short.</p></body></html>"


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """Transcript dir."""
    d = tmp_path / "data" / "company_filings"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def bm25_path(tmp_path: Path) -> Path:
    """Bm25 path."""
    return tmp_path / "data" / "bm25_index.pkl"


class TestRunPipeline:
    def _run(self, transcript_dir: Path, bm25_path: Path) -> MagicMock:
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        metrics_path = bm25_path.parent / "ingestion_metrics.json"
        bm25_corpus_path = bm25_path.parent / "bm25_corpus.pkl"

        with (
            patch("ingestion.pipeline.setup_embedder"),
            patch("ingestion.pipeline.TRANSCRIPTS_DIR", transcript_dir),
            patch("ingestion.pipeline.BM25_INDEX_PATH", bm25_path),
            patch("ingestion.pipeline.BM25_CORPUS_PATH", bm25_corpus_path),
            patch("ingestion.pipeline.INGESTION_METRICS_PATH", metrics_path),
            patch("ingestion.pipeline.init_qdrant", return_value=mock_qdrant),
            patch(
                "ingestion.pipeline.index_document",
                new_callable=AsyncMock,
                side_effect=lambda chunks, metadata, qdrant, timings=None: (
                    [["token"] for _ in chunks],
                    [
                        {"chunk_id": c.chunk_id, "text": c.text, "ticker": metadata.ticker}
                        for c in chunks
                    ],
                ),
            ),
        ):
            run_pipeline(fast=True)

        return mock_qdrant

    def test_pipeline_runs_without_error(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_10-K_2024-10-31_0001234567.htm").write_text(
            VALID_HTML, encoding="utf-8"
        )
        self._run(transcript_dir, bm25_path)

    def test_bm25_index_written_to_disk(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_10-K_2024-10-31_0001234567.htm").write_text(
            VALID_HTML, encoding="utf-8"
        )
        self._run(transcript_dir, bm25_path)
        assert bm25_path.exists()

    def test_bm25_file_is_valid_pickle(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_10-K_2024-10-31_0001234567.htm").write_text(
            VALID_HTML, encoding="utf-8"
        )
        self._run(transcript_dir, bm25_path)
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)  # nosec B301
        assert isinstance(obj, BM25Okapi)

    def test_empty_transcripts_dir_produces_no_upsert(self, transcript_dir, bm25_path) -> None:
        mock_qdrant = self._run(transcript_dir, bm25_path)
        mock_qdrant.upsert.assert_not_called()

    def test_metrics_saved_incrementally_across_runs(self, transcript_dir, bm25_path) -> None:
        import json

        metrics_path = bm25_path.parent / "ingestion_metrics.json"

        # First run with AAPL
        (transcript_dir / "AAPL_10-K_2024-10-31_0001234567.htm").write_text(
            VALID_HTML, encoding="utf-8"
        )
        self._run(transcript_dir, bm25_path)

        assert metrics_path.exists()
        with open(metrics_path, encoding="utf-8") as f:
            data1 = json.load(f)
        assert data1["summary"]["total_documents_processed"] == 1
        assert len(data1["documents"]) == 1
        assert data1["documents"][0]["file_name"] == "AAPL_10-K_2024-10-31_0001234567.htm"

        # Second run with MSFT
        (transcript_dir / "MSFT_10-Q_2024-04-30_0001234567.htm").write_text(
            VALID_HTML, encoding="utf-8"
        )
        self._run(transcript_dir, bm25_path)

        with open(metrics_path, encoding="utf-8") as f:
            data2 = json.load(f)
        assert data2["summary"]["total_documents_processed"] == 2
        assert len(data2["documents"]) == 2
        filenames = [doc["file_name"] for doc in data2["documents"]]
        assert "AAPL_10-K_2024-10-31_0001234567.htm" in filenames
        assert "MSFT_10-Q_2024-04-30_0001234567.htm" in filenames
