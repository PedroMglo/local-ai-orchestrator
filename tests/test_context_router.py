"""Tests for ContextRouter."""

from orchestrator.context.base import Complexity, Intent
from orchestrator.core.context_router import ConfigContextRouter


class TestContextRouter:
    def setup_method(self):
        self.router = ConfigContextRouter()

    def test_general_no_sources(self):
        sources = self.router.route(Intent.GENERAL, Complexity.NORMAL)
        assert sources == []

    def test_local_has_rag(self):
        sources = self.router.route(Intent.LOCAL, Complexity.NORMAL)
        assert "rag" in sources

    def test_local_has_cag(self):
        sources = self.router.route(Intent.LOCAL, Complexity.NORMAL)
        assert "cag" in sources

    def test_code_has_rag_and_graph(self):
        sources = self.router.route(Intent.CODE, Complexity.NORMAL)
        assert "rag" in sources
        assert "graph" in sources

    def test_system_has_system(self):
        sources = self.router.route(Intent.SYSTEM, Complexity.NORMAL)
        assert "system" in sources

    def test_graph_has_graph(self):
        sources = self.router.route(Intent.GRAPH, Complexity.NORMAL)
        assert "graph" in sources

    def test_combined_system_local(self):
        sources = self.router.route(Intent.SYSTEM_AND_LOCAL, Complexity.NORMAL)
        assert "system" in sources
        assert "rag" in sources

    def test_clarify_no_sources(self):
        sources = self.router.route(Intent.CLARIFY, Complexity.SIMPLE)
        assert sources == []
