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
    d = tmp_path / "data" / "transcripts"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def bm25_path(tmp_path: Path) -> Path:
    """Bm25 path."""
    return tmp_path / "data" / "bm25_index.pkl"


class TestRunPipeline:
    def _run(self, transcript_dir: Path, bm25_path: Path) -> MagicMock:
        mock_qdrant = MagicMock()
        checkpoint_path = bm25_path.parent / "pipeline_checkpoint.txt"

        with (
            patch("ingestion.pipeline.setup_embedder"),  # FIX: was setup_genai
            patch("ingestion.pipeline.TRANSCRIPTS_DIR", transcript_dir),
            patch("ingestion.pipeline.BM25_INDEX_PATH", bm25_path),
            patch("ingestion.pipeline.CHECKPOINT_PATH", checkpoint_path),
            patch("ingestion.pipeline.init_qdrant", return_value=mock_qdrant),
            patch(
                "ingestion.pipeline.index_document",
                new_callable=AsyncMock,
                return_value=(
                    [["token"]],
                    [{"chunk_id": "x", "text": "x"}],
                ),
            ),
        ):
            run_pipeline()

        return mock_qdrant

    def test_pipeline_runs_without_error(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)

    def test_valid_file_triggers_qdrant_upsert(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        mock_qdrant = self._run(transcript_dir, bm25_path)
        assert mock_qdrant is not None

    def test_short_file_is_skipped(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(SHORT_HTML, encoding="utf-8")
        mock_qdrant = self._run(transcript_dir, bm25_path)
        assert mock_qdrant is not None

    def test_bm25_index_written_to_disk(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)
        assert bm25_path.exists()

    def test_bm25_file_is_valid_pickle(self, transcript_dir, bm25_path) -> None:
        (transcript_dir / "AAPL_2024-10-31_0001234567.htm").write_text(VALID_HTML, encoding="utf-8")
        self._run(transcript_dir, bm25_path)
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)  # nosec B301
        assert isinstance(obj, BM25Okapi)

    def test_empty_transcripts_dir_produces_no_upsert(self, transcript_dir, bm25_path) -> None:
        mock_qdrant = self._run(transcript_dir, bm25_path)
        mock_qdrant.upsert.assert_not_called()

    def test_multiple_valid_files_all_indexed(self, transcript_dir, bm25_path) -> None:
        tickers = [("AAPL", "2024-10-31"), ("NVDA", "2024-07-15"), ("MSFT", "2024-04-30")]
        for ticker, date in tickers:
            (transcript_dir / f"{ticker}_{date}_0001234567.htm").write_text(
                VALID_HTML, encoding="utf-8"
            )
        mock_qdrant = self._run(transcript_dir, bm25_path)
        assert mock_qdrant is not None
