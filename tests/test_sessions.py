"""
Unit tests for Persistent ChatSessionStore and /sessions endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.chat_store import ChatSessionStore
from api.main import app


@pytest.fixture
def temp_store(tmp_path) -> ChatSessionStore:
    db_file = tmp_path / "test_chat.db"
    return ChatSessionStore(db_path=db_file)


class TestChatSessionStore:
    def test_create_and_get_session(self, temp_store: ChatSessionStore):
        sid = temp_store.create_session(
            user_id="analyst_1",
            title="Q4 Analysis",
            metadata={"source": "test"},
            tenant_id="tenant_a",
        )
        assert sid is not None
        assert sid.startswith("session_")

        detail = temp_store.get_session(sid, user_id="analyst_1")
        assert detail is not None
        assert detail.session_id == sid
        assert detail.user_id == "analyst_1"
        assert detail.title == "Q4 Analysis"
        assert detail.metadata == {"source": "test"}
        assert len(detail.messages) == 0

    def test_multi_user_isolation(self, temp_store: ChatSessionStore):
        s1 = temp_store.create_session(user_id="user_alice", title="Alice Session")
        s2 = temp_store.create_session(user_id="user_bob", title="Bob Session")

        alice_sessions = temp_store.list_sessions(user_id="user_alice")
        bob_sessions = temp_store.list_sessions(user_id="user_bob")

        assert len(alice_sessions) == 1
        assert alice_sessions[0].session_id == s1
        assert alice_sessions[0].title == "Alice Session"

        assert len(bob_sessions) == 1
        assert bob_sessions[0].session_id == s2
        assert bob_sessions[0].title == "Bob Session"

        # Bob cannot access Alice's session with user_id enforcement
        assert temp_store.get_session(s1, user_id="user_bob") is None
        # Accessing with Alice's user_id succeeds
        assert temp_store.get_session(s1, user_id="user_alice") is not None

    def test_add_messages_and_history(self, temp_store: ChatSessionStore):
        sid = temp_store.create_session(user_id="user_1", title="Conversation")

        temp_store.add_message(
            session_id=sid,
            role="user",
            content="What was Apple revenue?",
            user_id="user_1",
        )
        temp_store.add_message(
            session_id=sid,
            role="assistant",
            content="Apple revenue was $94.9B.",
            user_id="user_1",
            citations=[{"company": "AAPL", "doc_type": "10-K"}],
            meta={"confidence": 0.95},
        )

        detail = temp_store.get_session(sid, user_id="user_1")
        assert detail is not None
        assert len(detail.messages) == 2
        assert detail.messages[0].role == "user"
        assert detail.messages[0].content == "What was Apple revenue?"
        assert detail.messages[1].role == "assistant"
        assert len(detail.messages[1].citations) == 1
        assert detail.messages[1].metadata.get("confidence") == 0.95

        # Check recent history formatting
        recent = temp_store.get_recent_history(sid, limit=5)
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[1]["role"] == "assistant"

    def test_auto_create_session_on_message(self, temp_store: ChatSessionStore):
        auto_sid = "new_random_session_123"
        temp_store.add_message(
            session_id=auto_sid,
            role="user",
            content="How much did Nvidia grow data center revenue?",
            user_id="analyst_auto",
        )
        detail = temp_store.get_session(auto_sid)
        assert detail is not None
        assert detail.session_id == auto_sid
        assert "How much did Nvidia" in detail.title
        assert len(detail.messages) == 1

    def test_update_session_title(self, temp_store: ChatSessionStore):
        sid = temp_store.create_session(user_id="u1", title="Original Title")
        assert temp_store.update_session_title(sid, "Renamed Title", user_id="u1") is True

        detail = temp_store.get_session(sid)
        assert detail is not None
        assert detail.title == "Renamed Title"

    def test_clear_session(self, temp_store: ChatSessionStore):
        sid = temp_store.create_session(user_id="u1", title="Session to Clear")
        temp_store.add_message(sid, "user", "Message 1", "u1")
        temp_store.add_message(sid, "assistant", "Answer 1", "u1")

        assert temp_store.clear_session(sid, user_id="u1") is True
        detail = temp_store.get_session(sid)
        assert detail is not None
        assert len(detail.messages) == 0
        assert detail.title == "Session to Clear"

    def test_delete_session(self, temp_store: ChatSessionStore):
        sid = temp_store.create_session(user_id="u1", title="Session to Delete")
        temp_store.add_message(sid, "user", "Hello", "u1")

        assert temp_store.delete_session(sid, user_id="u1") is True
        assert temp_store.get_session(sid) is None

        # Repeat delete returns False
        assert temp_store.delete_session(sid, user_id="u1") is False


class TestSessionRoutes:
    @pytest.fixture(autouse=True)
    def setup_client(self, monkeypatch, tmp_path):
        test_store = ChatSessionStore(db_path=tmp_path / "route_test_sessions.db")
        monkeypatch.setattr("api.routes.sessions.get_chat_store", lambda: test_store)
        monkeypatch.setattr("api.routes.query.get_chat_store", lambda: test_store)
        self.store = test_store
        self.client = TestClient(app)

    def test_list_sessions_endpoint(self):
        self.store.create_session(user_id="test_user_api", title="Session API 1")
        response = self.client.get("/sessions", headers={"X-User-ID": "test_user_api"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Session API 1"

    def test_create_session_endpoint(self):
        payload = {"title": "Created via API", "metadata": {"tag": "fin"}}
        response = self.client.post(
            "/sessions",
            json=payload,
            headers={"X-User-ID": "test_user_api", "X-Tenant-ID": "corp_xyz"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Created via API"
        assert data["metadata"] == {"tag": "fin"}

    def test_get_session_endpoint(self):
        sid = self.store.create_session(user_id="test_user_api", title="Inspect Session")
        response = self.client.get(f"/sessions/{sid}", headers={"X-User-ID": "test_user_api"})
        assert response.status_code == 200
        assert response.json()["session_id"] == sid

    def test_rename_session_endpoint(self):
        sid = self.store.create_session(user_id="test_user_api", title="Old")
        patch_res = self.client.patch(
            f"/sessions/{sid}",
            json={"title": "New Title"},
            headers={"X-User-ID": "test_user_api"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["title"] == "New Title"

    def test_clear_session_endpoint(self):
        sid = self.store.create_session(user_id="test_user_api", title="Clear Me")
        self.store.add_message(sid, "user", "Q", "test_user_api")
        clear_res = self.client.post(
            f"/sessions/{sid}/clear", headers={"X-User-ID": "test_user_api"}
        )
        assert clear_res.status_code == 200
        detail = self.store.get_session(sid)
        assert detail is not None and len(detail.messages) == 0

    def test_delete_session_endpoint(self):
        sid = self.store.create_session(user_id="test_user_api", title="Delete Me")
        del_res = self.client.delete(f"/sessions/{sid}", headers={"X-User-ID": "test_user_api"})
        assert del_res.status_code == 204
        assert self.store.get_session(sid) is None

    def test_session_not_found(self):
        res = self.client.get("/sessions/nonexistent_12345")
        assert res.status_code == 404
