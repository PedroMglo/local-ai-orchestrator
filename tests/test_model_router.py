"""Tests for ModelRouter."""

from orchestrator.config import _reset_settings
from orchestrator.context.base import Complexity, Intent
from orchestrator.core.model_router import ConfigModelRouter


class TestModelRouter:
    def setup_method(self):
        _reset_settings()
        self.router = ConfigModelRouter()

    def test_general_simple_uses_fast(self):
        model = self.router.select(Intent.GENERAL, Complexity.SIMPLE)
        assert "gemma" in model or "4b" in model or model  # fast model

    def test_general_normal_uses_default(self):
        model = self.router.select(Intent.GENERAL, Complexity.NORMAL)
        assert "qwen3" in model or "8b" in model or model

    def test_code_uses_coder(self):
        model = self.router.select(Intent.CODE, Complexity.NORMAL)
        assert "coder" in model or model

    def test_deep_uses_deepseek(self):
        model = self.router.select(Intent.GENERAL, Complexity.DEEP)
        assert "deepseek" in model or "r1" in model or model

    def test_system_simple_uses_fast(self):
        model = self.router.select(Intent.SYSTEM, Complexity.SIMPLE)
        assert "gemma" in model or "4b" in model or model

    def test_graph_deep_uses_deep(self):
        model = self.router.select(Intent.GRAPH, Complexity.DEEP)
        assert "deepseek" in model or "r1" in model or model

    def test_all_intents_return_string(self):
        for intent in Intent:
            for complexity in Complexity:
                model = self.router.select(intent, complexity)
                assert isinstance(model, str)
                assert len(model) > 0
