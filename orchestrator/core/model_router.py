"""Model selection based on intent × complexity × resource availability."""

from __future__ import annotations

import logging

from orchestrator.config import get_settings
from orchestrator.context.base import Complexity, Intent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing table: (intent, complexity) → model config key
# ---------------------------------------------------------------------------
# Keys: "default", "fast", "code", "deep"
_ROUTE_TABLE: dict[tuple[Intent, Complexity], str] = {
    # General
    (Intent.GENERAL, Complexity.SIMPLE): "fast",
    (Intent.GENERAL, Complexity.NORMAL): "default",
    (Intent.GENERAL, Complexity.COMPLEX): "default",
    (Intent.GENERAL, Complexity.DEEP): "deep",
    # Local / notes
    (Intent.LOCAL, Complexity.SIMPLE): "default",
    (Intent.LOCAL, Complexity.NORMAL): "default",
    (Intent.LOCAL, Complexity.COMPLEX): "default",
    (Intent.LOCAL, Complexity.DEEP): "deep",
    # Code
    (Intent.CODE, Complexity.SIMPLE): "code",
    (Intent.CODE, Complexity.NORMAL): "code",
    (Intent.CODE, Complexity.COMPLEX): "code",
    (Intent.CODE, Complexity.DEEP): "code",
    # System
    (Intent.SYSTEM, Complexity.SIMPLE): "fast",
    (Intent.SYSTEM, Complexity.NORMAL): "default",
    (Intent.SYSTEM, Complexity.COMPLEX): "default",
    (Intent.SYSTEM, Complexity.DEEP): "default",
    # Graph / architecture
    (Intent.GRAPH, Complexity.SIMPLE): "default",
    (Intent.GRAPH, Complexity.NORMAL): "default",
    (Intent.GRAPH, Complexity.COMPLEX): "deep",
    (Intent.GRAPH, Complexity.DEEP): "deep",
    # Combined
    (Intent.LOCAL_AND_GRAPH, Complexity.SIMPLE): "default",
    (Intent.LOCAL_AND_GRAPH, Complexity.NORMAL): "default",
    (Intent.LOCAL_AND_GRAPH, Complexity.COMPLEX): "deep",
    (Intent.LOCAL_AND_GRAPH, Complexity.DEEP): "deep",
    (Intent.SYSTEM_AND_LOCAL, Complexity.SIMPLE): "default",
    (Intent.SYSTEM_AND_LOCAL, Complexity.NORMAL): "default",
    (Intent.SYSTEM_AND_LOCAL, Complexity.COMPLEX): "default",
    (Intent.SYSTEM_AND_LOCAL, Complexity.DEEP): "default",
    # Clarify — fast response to ask for clarification
    (Intent.CLARIFY, Complexity.SIMPLE): "fast",
    (Intent.CLARIFY, Complexity.NORMAL): "fast",
    (Intent.CLARIFY, Complexity.COMPLEX): "fast",
    (Intent.CLARIFY, Complexity.DEEP): "fast",
}


class ConfigModelRouter:
    """Selects model from config based on intent × complexity routing table."""

    def select(self, intent: Intent, complexity: Complexity) -> str:
        cfg = get_settings().models
        key = _ROUTE_TABLE.get((intent, complexity), "default")

        model_map = {
            "default": cfg.default,
            "fast": cfg.fast,
            "code": cfg.code,
            "deep": cfg.deep,
        }

        model = model_map.get(key, cfg.default)
        log.debug("ModelRouter: %s×%s → %s (%s)", intent.value, complexity.value, model, key)
        return model
