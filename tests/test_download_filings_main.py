from unittest.mock import patch

from ingestion.download_filings import main


def test_main_execution_success():
    with (
        patch("ingestion.download_filings.os.makedirs") as mock_makedirs,
        patch("ingestion.download_filings.time.sleep"),
        patch("ingestion.download_filings.get_8k_filings") as mock_get_8k,
        patch("ingestion.download_filings.get_filing_documents") as mock_get_docs,
        patch("ingestion.download_filings.pick_best_document") as mock_pick,
        patch("ingestion.download_filings.download_document") as mock_download,
    ):
        mock_get_8k.return_value = [
            {"ticker": "AAPL", "cik": "123", "accession": "0001", "date": "2024"}
        ]
        mock_get_docs.return_value = [{"name": "doc1"}]
        mock_pick.return_value = "doc1"
        mock_download.return_value = "path/to/doc"

        with patch("ingestion.download_filings.COMPANIES", {"AAPL": "123"}):
            main()

        mock_makedirs.assert_called_once_with("data/transcripts", exist_ok=True)
        mock_get_8k.assert_called_with("123", "AAPL")
        mock_get_docs.assert_called_with("123", "0001")
        mock_pick.assert_called_once()
        mock_download.assert_called_once()


def test_main_execution_no_documents():
    with (
        patch("ingestion.download_filings.os.makedirs"),
        patch("ingestion.download_filings.time.sleep"),
        patch("ingestion.download_filings.get_8k_filings") as mock_get_8k,
        patch("ingestion.download_filings.get_filing_documents") as mock_get_docs,
        patch("ingestion.download_filings.pick_best_document") as mock_pick,
        patch("ingestion.download_filings.download_document") as mock_download,
    ):
        mock_get_8k.return_value = [
            {"ticker": "AAPL", "cik": "123", "accession": "0001", "date": "2024"}
        ]
        # No documents found
        mock_get_docs.return_value = []

        with patch("ingestion.download_filings.COMPANIES", {"AAPL": "123"}):
            main()

        mock_pick.assert_not_called()
        mock_download.assert_not_called()


def test_main_execution_no_best_document():
    with (
        patch("ingestion.download_filings.os.makedirs"),
        patch("ingestion.download_filings.time.sleep"),
        patch("ingestion.download_filings.get_8k_filings") as mock_get_8k,
        patch("ingestion.download_filings.get_filing_documents") as mock_get_docs,
        patch("ingestion.download_filings.pick_best_document") as mock_pick,
        patch("ingestion.download_filings.download_document") as mock_download,
    ):
        mock_get_8k.return_value = [
            {"ticker": "AAPL", "cik": "123", "accession": "0001", "date": "2024"}
        ]
        mock_get_docs.return_value = [{"name": "doc1"}]
        # No best document found
        mock_pick.return_value = None

        with patch("ingestion.download_filings.COMPANIES", {"AAPL": "123"}):
            main()

        mock_pick.assert_called_once()
        mock_download.assert_not_called()


def test_main_execution_download_failed():
    with (
        patch("ingestion.download_filings.os.makedirs"),
        patch("ingestion.download_filings.time.sleep"),
        patch("ingestion.download_filings.get_8k_filings") as mock_get_8k,
        patch("ingestion.download_filings.get_filing_documents") as mock_get_docs,
        patch("ingestion.download_filings.pick_best_document") as mock_pick,
        patch("ingestion.download_filings.download_document") as mock_download,
    ):
        mock_get_8k.return_value = [
            {"ticker": "AAPL", "cik": "123", "accession": "0001", "date": "2024"}
        ]
        mock_get_docs.return_value = [{"name": "doc1"}]
        mock_pick.return_value = "doc1"
        # Download returns None
        mock_download.return_value = None

        with patch("ingestion.download_filings.COMPANIES", {"AAPL": "123"}):
            main()

        mock_download.assert_called_once()
