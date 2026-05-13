"""RAG context provider — HTTP adapter to the obsidian-rag service."""

from __future__ import annotations

import logging
import time

import httpx

from orchestrator.config import get_settings
from orchestrator.context.base import ContextBlock, estimate_tokens

log = logging.getLogger(__name__)


class RAGContextProvider:
    """Obtém contexto via HTTP do RAG service (obsidian-rag API)."""

    def __init__(self) -> None:
        cfg = get_settings().rag
        self._url = cfg.url
        self._timeout = cfg.timeout
        # Circuit breaker state
        self._failures = 0
        self._threshold = cfg.circuit_breaker_threshold
        self._reset_seconds = cfg.circuit_breaker_reset
        self._open_until: float = 0.0

    @property
    def name(self) -> str:
        return "rag"

    def _is_circuit_open(self) -> bool:
        if self._failures >= self._threshold:
            if time.monotonic() < self._open_until:
                return True
            # Half-open: allow one attempt
            self._failures = self._threshold - 1
        return False

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = time.monotonic() + self._reset_seconds
            log.warning("RAG circuit breaker OPEN (will retry in %ds)", self._reset_seconds)

    def _record_success(self) -> None:
        self._failures = 0

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        if self._is_circuit_open():
            log.debug("RAG: circuit breaker open, skipping")
            return None

        top_k = min(10, max(3, budget_tokens // 400))

        try:
            # Query notes — exclude repo_doc to avoid project docs contaminating personal notes
            notes = self._query_collection(
                query, "/query", top_k=top_k, exclude_source_type="repo_doc"
            )
            # Query code
            code = self._query_collection(query, "/query/code", top_k=top_k)
        except Exception as exc:
            log.warning("RAG: request failed: %s", exc)
            self._record_failure()
            return None

        self._record_success()

        parts: list[str] = []
        if notes:
            parts.append("[NOTES]\n" + "\n---\n".join(notes) + "\n[/NOTES]")
        if code:
            parts.append("[CODE]\n" + "\n---\n".join(code) + "\n[/CODE]")

        if not parts:
            return None

        content = "\n\n".join(parts)
        return ContextBlock(
            source="rag",
            content=content,
            token_estimate=estimate_tokens(content),
        )

    def _query_collection(
        self,
        query: str,
        endpoint: str,
        *,
        top_k: int = 5,
        exclude_source_type: str | None = None,
    ) -> list[str]:
        payload: dict = {"query": query, "top_k": top_k, "min_score": 0.45}
        if exclude_source_type:
            payload["exclude_source_type"] = exclude_source_type
        resp = httpx.post(
            f"{self._url}{endpoint}",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        chunks: list[str] = []
        for r in results:
            title = r.get("note_title", "")
            section = r.get("section_header", "")
            text = r.get("text", "")
            score = r.get("score", 0)
            label = f"[{title} / {section}]" if section else f"[{title}]"
            chunks.append(f"{label} score={score:.2f}\n{text}")
        return chunks

    def health(self) -> bool:
        if self._is_circuit_open():
            return False
        try:
            resp = httpx.get(f"{self._url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
