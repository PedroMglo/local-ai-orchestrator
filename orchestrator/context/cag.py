"""CAG context provider — reads pre-computed context packs from SQLite."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from orchestrator.config import get_settings
from orchestrator.context.base import ContextBlock, estimate_tokens

log = logging.getLogger(__name__)

# Intent mode → relevant pack types
_INTENT_PACKS: dict[str, list[str]] = {
    "general": ["config_environment"],
    "local": ["vault_summary", "config_environment", "rag_index_state"],
    "code": ["project_architecture", "repo_state", "knowledge_graph_summary", "pending_tasks"],
    "system": ["system_state", "local_services", "local_models"],
    "graph": ["knowledge_graph_summary", "project_architecture"],
    "local_and_graph": ["vault_summary", "knowledge_graph_summary", "project_architecture", "repo_state"],
    "system_and_local": ["system_state", "local_services", "config_environment"],
}


class CAGContextProvider:
    """Reads CAG packs from the RAG's SQLite database (read-only)."""

    def __init__(self, *, intent_hint: str = "general") -> None:
        self._intent_hint = intent_hint

    @property
    def name(self) -> str:
        return "cag"

    def _db_path(self) -> Path | None:
        raw = get_settings().context.cag.db_path
        if not raw:
            return None
        p = Path(raw).expanduser()
        return p if p.exists() else None

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        db = self._db_path()
        if db is None:
            return None

        pack_types = _INTENT_PACKS.get(self._intent_hint, ["config_environment"])
        now = time.time()

        rows = self._query_packs(db, pack_types, now)
        if rows is None:
            return None

        if not rows:
            return None

        parts: list[str] = []
        total_tokens = 0
        for pack_type, content in rows:
            tokens = estimate_tokens(content)
            if total_tokens + tokens > budget_tokens and parts:
                break
            parts.append(f"## {pack_type}\n{content}")
            total_tokens += tokens

        content = "\n\n".join(parts)
        return ContextBlock(
            source="cag",
            content=content,
            token_estimate=total_tokens,
        )

    def _query_packs(
        self, db: Path, pack_types: list[str], now: float
    ) -> list[tuple[str, str]] | None:
        """Query SQLite with one retry on database-locked errors."""
        for attempt in range(2):
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
                placeholders = ",".join("?" for _ in pack_types)
                rows = conn.execute(
                    f"SELECT pack_type, content FROM packs "
                    f"WHERE pack_type IN ({placeholders}) AND expires_at > ?",
                    (*pack_types, now),
                ).fetchall()
                conn.close()
                return rows
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc) and attempt == 0:
                    log.debug("CAG: database locked, retrying in 1s")
                    time.sleep(1)
                    continue
                log.debug("CAG: SQLite read failed: %s", exc)
                return None
            except Exception as exc:
                log.debug("CAG: SQLite read failed: %s", exc)
                return None
        return None

    def set_intent_hint(self, intent: str) -> None:
        """Update intent hint for pack selection."""
        self._intent_hint = intent

    def health(self) -> bool:
        db = self._db_path()
        if db is None or not db.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            conn.execute("SELECT 1 FROM packs LIMIT 1")
            conn.close()
            return True
        except Exception:
            return False
