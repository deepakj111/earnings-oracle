from pathlib import Path

from ingestion.state_manager import IngestionStateManager


def test_state_manager_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "test_state.db"
    mgr = IngestionStateManager(db_path=str(db_file))

    # New chunk should be reported as modified/unseen
    assert mgr.is_chunk_modified("chunk_1", "Apple reported $100B revenue.") is True

    # Single update
    mgr.update_chunk("chunk_1", "Apple reported $100B revenue.")
    assert mgr.is_chunk_modified("chunk_1", "Apple reported $100B revenue.") is False

    # Modified content should be detected
    assert mgr.is_chunk_modified("chunk_1", "Apple reported $105B revenue.") is True


def test_state_manager_batch_updates(tmp_path: Path) -> None:
    db_file = tmp_path / "test_state_batch.db"
    mgr = IngestionStateManager(db_path=str(db_file))

    batch = [
        ("c_1", "Text for chunk 1"),
        ("c_2", "Text for chunk 2"),
        ("c_3", "Text for chunk 3"),
    ]

    # Verify all are initially modified
    for cid, text in batch:
        assert mgr.is_chunk_modified(cid, text) is True

    # Batch update
    mgr.update_chunks(batch)

    # Verify none are modified now
    for cid, text in batch:
        assert mgr.is_chunk_modified(cid, text) is False

    # Empty batch is a no-op and doesn't fail
    mgr.update_chunks([])
