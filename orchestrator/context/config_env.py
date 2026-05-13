"""Config/environment context provider."""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.config import get_settings
from orchestrator.context.base import ContextBlock, estimate_tokens
from orchestrator.core.security import safe_run

log = logging.getLogger(__name__)


class ConfigEnvProvider:
    """Provides config and environment summary."""

    @property
    def name(self) -> str:
        return "config"

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        cfg = get_settings()
        lines = ["# Environment Summary\n"]

        # Models
        lines.append("## Configured Models")
        lines.append(f"- Default: {cfg.models.default}")
        lines.append(f"- Fast: {cfg.models.fast}")
        lines.append(f"- Code: {cfg.models.code}")
        lines.append(f"- Deep: {cfg.models.deep}")
        lines.append(f"- Embedding: {cfg.models.embedding}")
        lines.append("")

        # Services
        lines.append("## Services")
        lines.append(f"- Ollama: {cfg.ollama.base_url}")
        lines.append(f"- RAG: {cfg.rag.url}")
        lines.append(f"- Orchestrator: {cfg.orchestrator.host}:{cfg.orchestrator.port}")
        lines.append("")

        # Repos
        if cfg.repos.paths:
            lines.append(f"## Repos ({len(cfg.repos.paths)})")
            for p in cfg.repos.paths:
                lines.append(f"- {Path(p).name}")
            lines.append("")

        # Ollama models (if available)
        ollama_out = safe_run(["ollama", "list"])
        if ollama_out:
            lines.append("## Installed Ollama Models")
            lines.append(f"```\n{ollama_out}\n```")

        content = "\n".join(lines)
        return ContextBlock(
            source="config",
            content=content,
            token_estimate=estimate_tokens(content),
        )

    def health(self) -> bool:
        return True
