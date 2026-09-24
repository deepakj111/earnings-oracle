import hashlib
import sqlite3
from pathlib import Path


class IngestionStateManager:
    """
    Manages state for idempotent ingestion using SQLite.
    Tracks document-level and chunk-level SHA-256 hashes.
    """

    def __init__(self, db_path: str = "data/ingestion_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_hash TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_chunk_modified(self, chunk_id: str, chunk_text: str) -> bool:
        """
        Returns True if the chunk is new or its content has changed.
        """
        current_hash = self.compute_hash(chunk_text)
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_hash FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()

            if row is None:
                return True
            return row[0] != current_hash

    def update_chunk(self, chunk_id: str, chunk_text: str) -> None:
        """
        Updates the stored hash for a single chunk.
        """
        self.update_chunks([(chunk_id, chunk_text)])

    def update_chunks(self, chunks: list[tuple[str, str]]) -> None:
        """
        Batch update stored hashes for multiple chunks in a single transaction.
        """
        if not chunks:
            return
        records = [(cid, self.compute_hash(text)) for cid, text in chunks]
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO chunks (chunk_id, chunk_hash, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chunk_id) DO UPDATE SET
                chunk_hash = excluded.chunk_hash,
                updated_at = CURRENT_TIMESTAMP
                """,
                records,
            )
            conn.commit()
