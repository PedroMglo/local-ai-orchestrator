"""Context source routing — decides which providers to query per intent."""

from __future__ import annotations

import logging

from orchestrator.context.base import Complexity, Intent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source names (must match ContextProvider.name)
# ---------------------------------------------------------------------------
RAG = "rag"
CAG = "cag"
SYSTEM = "system"
REPO = "repo"
GRAPH = "graph"
CONFIG = "config"
LOGS = "logs"

# ---------------------------------------------------------------------------
# Routing table: intent → ordered list of sources
# ---------------------------------------------------------------------------
_ROUTE_TABLE: dict[Intent, list[str]] = {
    Intent.GENERAL: [],
    Intent.LOCAL: [RAG, CAG],
    Intent.CODE: [RAG, GRAPH, CAG],
    Intent.SYSTEM: [SYSTEM, CAG],
    Intent.GRAPH: [GRAPH, CAG],
    Intent.LOCAL_AND_GRAPH: [RAG, GRAPH, CAG],
    Intent.SYSTEM_AND_LOCAL: [SYSTEM, RAG, CAG],
    Intent.CLARIFY: [],
}


class ConfigContextRouter:
    """Routes intent to context provider names."""

    def route(self, intent: Intent, complexity: Complexity) -> list[str]:
        sources = list(_ROUTE_TABLE.get(intent, []))
        log.debug("ContextRouter: %s×%s → %s", intent.value, complexity.value, sources)
        return sources
