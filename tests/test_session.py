"""Tests for the SessionStore."""

from __future__ import annotations

import time

import pytest

from orchestrator.core.session import SessionStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test_sessions.db"
    s = SessionStore(db_path=str(db), max_messages=5)
    yield s
    s.close()


class TestSessionStore:

    def test_append_and_get(self, store):
        store.append("s1", "user", "hello")
        store.append("s1", "assistant", "hi there")
        msgs = store.get("s1")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_get_empty_session(self, store):
        assert store.get("nonexistent") == []

    def test_separate_sessions(self, store):
        store.append("s1", "user", "q1")
        store.append("s2", "user", "q2")
        assert len(store.get("s1")) == 1
        assert len(store.get("s2")) == 1
        assert store.get("s1")[0]["content"] == "q1"
        assert store.get("s2")[0]["content"] == "q2"

    def test_max_messages_limit(self, store):
        for i in range(10):
            store.append("s1", "user", f"msg-{i}")
        msgs = store.get("s1")
        assert len(msgs) == 5  # max_messages=5
        assert msgs[0]["content"] == "msg-5"
        assert msgs[-1]["content"] == "msg-9"

    def test_cleanup_removes_old(self, store):
        # Insert a message with a manually backdated timestamp
        old_time = time.time() - 7200  # 2 hours ago
        store._db.execute(
            "INSERT INTO sessions (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            ("s1", "user", "old msg", old_time),
        )
        store._db.commit()
        store.append("s1", "user", "new msg")

        deleted = store.cleanup(max_age_seconds=3600)  # 1 hour TTL
        assert deleted == 1
        msgs = store.get("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "new msg"

    def test_cleanup_returns_zero_when_nothing_expired(self, store):
        store.append("s1", "user", "fresh")
        assert store.cleanup(max_age_seconds=3600) == 0
