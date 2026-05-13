"""Factory — builds a fully-wired Engine with all context providers registered."""

from __future__ import annotations

from orchestrator.context.cag import CAGContextProvider
from orchestrator.context.config_env import ConfigEnvProvider
from orchestrator.context.graph import GraphProvider
from orchestrator.context.logs import LogsProvider
from orchestrator.context.rag import RAGContextProvider
from orchestrator.context.repo import RepoProbeProvider
from orchestrator.context.system import SystemProbeProvider
from orchestrator.core.engine import Engine


def create_engine() -> Engine:
    """Create an Engine with all context providers registered."""
    engine = Engine()

    engine.register_provider(RAGContextProvider())
    engine.register_provider(CAGContextProvider())
    engine.register_provider(SystemProbeProvider())
    engine.register_provider(RepoProbeProvider())
    engine.register_provider(GraphProvider())
    engine.register_provider(ConfigEnvProvider())
    engine.register_provider(LogsProvider())

    return engine
