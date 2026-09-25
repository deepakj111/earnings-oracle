"""
Persistent Chat Session Store for Multi-Tenant Enterprise Financial RAG.

Provides thread-safe, multi-user chat session persistence using SQLite.
Stores conversations, message history, grounding metadata, and citations.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from api.models import (
    ChatMessageRecord,
    SessionDetailOut,
    SessionSummaryOut,
)


class ChatSessionStore:
    """
    SQLite-backed repository for multi-user conversational chat history.
    """

    def __init__(self, db_path: str | Path = "data/chat_sessions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    citations_json TEXT,
                    meta_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id, updated_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id ASC)"
            )
            conn.commit()

    def create_session(
        self,
        user_id: str = "default_user",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default_tenant",
        session_id: str | None = None,
    ) -> str:
        """Create a new conversational session for a given user."""
        sid = session_id or f"session_{int(time.time() * 1000)}"
        now = time.time()
        session_title = title or "New Financial Analysis"
        meta_str = json.dumps(metadata or {})

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (session_id, user_id, tenant_id, title, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (sid, user_id, tenant_id, session_title, now, now, meta_str),
            )
            conn.commit()

        logger.debug(f"[ChatStore] Created session '{sid}' for user '{user_id}'")
        return sid

    def get_session(self, session_id: str, user_id: str | None = None) -> SessionDetailOut | None:
        """Retrieve full session detail and message history."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
            else:
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                return None

            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            message_rows = cursor.fetchall()

        messages: list[ChatMessageRecord] = []
        for row in message_rows:
            citations = json.loads(row["citations_json"] or "[]")
            meta = json.loads(row["meta_json"] or "{}")
            messages.append(
                ChatMessageRecord(
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["created_at"],
                    citations=citations,
                    metadata=meta,
                )
            )

        meta_dict = json.loads(session_row["metadata_json"] or "{}")
        return SessionDetailOut(
            session_id=session_row["session_id"],
            user_id=session_row["user_id"],
            title=session_row["title"],
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            messages=messages,
            metadata=meta_dict,
        )

    def list_sessions(self, user_id: str, limit: int = 50) -> list[SessionSummaryOut]:
        """List active sessions for a user, sorted by most recently updated."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.session_id, s.user_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE s.user_id = ?
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()

        return [
            SessionSummaryOut(
                session_id=row["session_id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def update_session_title(self, session_id: str, title: str, user_id: str | None = None) -> bool:
        """Update session title."""
        now = time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ? AND user_id = ?",
                    (title.strip(), now, session_id, user_id),
                )
            else:
                cursor.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title.strip(), now, session_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def delete_session(self, session_id: str, user_id: str | None = None) -> bool:
        """Delete session and its associated messages."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
            else:
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            session_deleted = cursor.rowcount > 0
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()
            return session_deleted

    def clear_session(self, session_id: str, user_id: str | None = None) -> bool:
        """Clear all messages from session without deleting session metadata."""
        now = time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
                if not cursor.fetchone():
                    return False
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
            conn.commit()
            return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "default_user",
        citations: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Add a message turn to a session, ensuring session exists."""
        now = time.time()
        citations_json = json.dumps(citations or [])
        meta_json = json.dumps(meta or {})

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure session exists
            cursor.execute(
                "SELECT session_id, title FROM sessions WHERE session_id = ?", (session_id,)
            )
            existing = cursor.fetchone()
            if not existing:
                # Auto-generate title from first user query
                auto_title = content[:60].strip() if role == "user" else "Financial Analysis"
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, tenant_id, title, created_at, updated_at, metadata_json)
                    VALUES (?, ?, 'default_tenant', ?, ?, ?, '{}')
                    """,
                    (session_id, user_id, auto_title, now, now),
                )
            else:
                cursor.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at, citations_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, citations_json, meta_json),
            )
            conn.commit()

    def get_recent_history(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        """Get recent conversational turns formatted for LLM query reformulation."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()

        # Reverse so order is chronological
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── Global store singleton ───────────────────────────────────────────────────
_global_chat_store: ChatSessionStore | None = None
_store_lock = threading.Lock()


def get_chat_store() -> ChatSessionStore:
    """Return the global ChatSessionStore singleton."""
    global _global_chat_store
    if _global_chat_store is None:
        with _store_lock:
            if _global_chat_store is None:
                _global_chat_store = ChatSessionStore()
    return _global_chat_store
