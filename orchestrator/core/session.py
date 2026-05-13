"""Session store — SQLite-backed conversation history with TTL."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".local" / "share" / "ai-orchestrator"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "sessions.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, created_at)
);
CREATE INDEX IF NOT EXISTS idx_sessions_access ON sessions (created_at);
"""


class SessionStore:
    """Lightweight SQLite session store for conversation history.

    Each session is a sequence of ``{role, content}`` messages keyed by
    ``session_id``.  Messages older than the configured TTL are pruned
    on explicit ``cleanup()`` calls.
    """

    def __init__(self, *, db_path: str | None = None, max_messages: int = 20) -> None:
        path = Path(os.path.expanduser(db_path)) if db_path else _DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._max_messages = max_messages
        log.debug("SessionStore opened at %s", path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> list[dict]:
        """Return the last ``max_messages`` messages for a session."""
        rows = self._db.execute(
            "SELECT role, content FROM sessions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        messages = [{"role": r, "content": c} for r, c in rows]
        return messages[-self._max_messages:]

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session."""
        self._db.execute(
            "INSERT INTO sessions (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self._db.commit()

    def cleanup(self, max_age_seconds: int) -> int:
        """Delete messages older than *max_age_seconds*. Returns rows deleted."""
        cutoff = time.time() - max_age_seconds
        cur = self._db.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
        self._db.commit()
        deleted = cur.rowcount
        if deleted:
            log.info("SessionStore: cleaned up %d expired messages", deleted)
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
