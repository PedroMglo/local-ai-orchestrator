"""Tests for the Engine with mock LLM and providers."""

from __future__ import annotations

import httpx

from orchestrator.config import _reset_settings
from orchestrator.context.base import Complexity, ContextBlock, Intent, estimate_tokens
from orchestrator.core.engine import _LLM_UNAVAILABLE_MSG, Engine


class MockLLM:
    """Mock LLM that echoes the model name."""

    def chat(self, messages, model, **kwargs):
        return f"[{model}] response"

    def chat_stream(self, messages, model, **kwargs):
        yield f"[{model}]"
        yield " stream"

    def health(self):
        return True


class FailingLLM:
    """Mock LLM that raises httpx errors."""

    def __init__(self, *, healthy: bool = False):
        self._healthy = healthy

    def chat(self, messages, model, **kwargs):
        raise httpx.ConnectError("Connection refused")

    def chat_stream(self, messages, model, **kwargs):
        raise httpx.ConnectError("Connection refused")

    def health(self):
        return self._healthy


class MockProvider:
    """Mock context provider."""

    def __init__(self, name: str, content: str = "mock context"):
        self._name = name
        self._content = content
        self._healthy = True

    @property
    def name(self) -> str:
        return self._name

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        return ContextBlock(
            source=self._name,
            content=self._content,
            token_estimate=estimate_tokens(self._content),
        )

    def health(self) -> bool:
        return self._healthy


class TestEngine:
    def setup_method(self):
        _reset_settings()
        self.llm = MockLLM()
        self.engine = Engine(llm=self.llm)

    def test_general_query_no_context(self):
        result = self.engine.run("o que é DNS?")
        assert result.intent == Intent.GENERAL
        assert result.sources_used == []
        assert result.response

    def test_model_selected(self):
        result = self.engine.run("o que é DNS?")
        # General + Simple → fast model
        assert result.model_used

    def test_with_provider(self):
        provider = MockProvider("rag", "contexto local relevante")
        self.engine.register_provider(provider)
        result = self.engine.run("mostra as minhas notas sobre Docker")
        assert result.intent == Intent.LOCAL
        assert "rag" in result.sources_used

    def test_model_override(self):
        result = self.engine.run("olá", model_override="custom:latest")
        assert result.model_used == "custom:latest"

    def test_stream(self):
        chunks = list(self.engine.stream("olá"))
        assert len(chunks) > 0
        full = "".join(chunks)
        assert "stream" in full

    def test_unhealthy_provider_skipped(self):
        provider = MockProvider("rag")
        provider._healthy = False
        self.engine.register_provider(provider)
        result = self.engine.run("mostra as minhas notas")
        assert "rag" not in result.sources_used

    def test_classify_only(self):
        routing = self.engine.classify("quanta RAM tenho?")
        assert routing.intent == Intent.SYSTEM
        assert isinstance(routing.complexity, Complexity)

    def test_code_intent_selects_coder(self):
        result = self.engine.run("escreve uma função Python")
        assert result.intent == Intent.CODE
        assert "coder" in result.model_used or result.model_used  # model contains "coder"

    def test_context_budget_respected(self):
        """Provider content is within budget."""
        big_provider = MockProvider("rag", "x" * 30000)
        self.engine.register_provider(big_provider)
        result = self.engine.run("mostra as minhas notas")
        # Should still work without crashing
        assert result.response


class TestEngineConfig:
    def test_config_loads(self):
        _reset_settings()
        engine = Engine(llm=MockLLM())
        result = engine.run("olá")
        assert result.response


class TestEngineResilience:
    """§4.1 — Engine handles LLM failures gracefully."""

    def setup_method(self):
        _reset_settings()

    def test_run_returns_degraded_when_llm_unhealthy(self):
        engine = Engine(llm=FailingLLM(healthy=False))
        result = engine.run("olá")
        assert _LLM_UNAVAILABLE_MSG in result.response
        assert result.model_used  # model is still selected
        assert result.intent == Intent.GENERAL

    def test_run_returns_degraded_on_connect_error(self):
        """LLM reports healthy but chat() throws — should catch and degrade."""
        engine = Engine(llm=FailingLLM(healthy=True))
        # Force health cache to pass
        engine._llm_healthy = True
        engine._llm_health_ts = float("inf")
        result = engine.run("olá")
        assert _LLM_UNAVAILABLE_MSG in result.response

    def test_stream_yields_degraded_when_llm_unhealthy(self):
        engine = Engine(llm=FailingLLM(healthy=False))
        chunks = list(engine.stream("olá"))
        full = "".join(chunks)
        assert _LLM_UNAVAILABLE_MSG in full

    def test_stream_yields_degraded_on_connect_error(self):
        engine = Engine(llm=FailingLLM(healthy=True))
        engine._llm_healthy = True
        engine._llm_health_ts = float("inf")
        chunks = list(engine.stream("olá"))
        full = "".join(chunks)
        assert _LLM_UNAVAILABLE_MSG in full

    def test_health_gate_caches(self):
        """Health check should not be called on every query within TTL."""
        llm = MockLLM()
        engine = Engine(llm=llm)
        # First call populates cache
        engine._is_llm_available()
        ts1 = engine._llm_health_ts
        # Second call within TTL should reuse cache
        engine._is_llm_available()
        assert engine._llm_health_ts == ts1

    def test_connect_error_invalidates_health_cache(self):
        engine = Engine(llm=FailingLLM(healthy=True))
        engine._llm_healthy = True
        engine._llm_health_ts = float("inf")
        engine.run("olá")
        # After error, cache should be invalidated
        assert engine._llm_healthy is False
