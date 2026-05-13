"""Logs context provider — tail recent log files for errors."""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.context.base import ContextBlock, estimate_tokens

log = logging.getLogger(__name__)

# Common log locations
_LOG_LOCATIONS = [
    Path.home() / "ai-local" / "obsidian-rag" / "data" / "qdrant",
    Path.home() / "ai-local" / "orchestrator",
]


class LogsProvider:
    """Scans recent log files for error patterns."""

    @property
    def name(self) -> str:
        return "logs"

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        error_lines: list[str] = []

        for log_dir in _LOG_LOCATIONS:
            if not log_dir.is_dir():
                continue
            log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
            for lf in log_files:
                try:
                    text = lf.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines()[-100:]:
                        low = line.lower()
                        if "error" in low or "exception" in low or "traceback" in low:
                            error_lines.append(f"[{lf.name}] {line.strip()}")
                except OSError:
                    continue

        if not error_lines:
            return None

        # Truncate to budget
        content_lines: list[str] = ["# Recent Errors\n"]
        tokens = 10
        for line in error_lines[-50:]:
            t = estimate_tokens(line)
            if tokens + t > budget_tokens:
                content_lines.append("... [truncated]")
                break
            content_lines.append(line)
            tokens += t

        content = "\n".join(content_lines)
        return ContextBlock(
            source="logs",
            content=content,
            token_estimate=tokens,
        )

    def health(self) -> bool:
        return True
