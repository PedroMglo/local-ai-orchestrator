"""Integration tests — require live services (Ollama, optionally RAG).

Skip with: pytest tests/test_integration.py -m integration
"""

from __future__ import annotations

import httpx
import pytest

# Mark all tests in this module
pytestmark = pytest.mark.integration


def _ollama_available() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _rag_available() -> bool:
    try:
        resp = httpx.get("http://localhost:8484/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
skip_no_rag = pytest.mark.skipif(not _rag_available(), reason="RAG service not running")


class TestClassificationLive:
    """Classification does not require LLM — always works."""

    def test_classify_general(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        r = engine.classify("o que é DNS?")
        assert r.intent.value == "general"

    def test_classify_system(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        r = engine.classify("quanta RAM tenho livre?")
        assert r.intent.value == "system"

    def test_classify_code(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        r = engine.classify("refactora esta função para async/await")
        assert r.intent.value == "code"

    def test_classify_local(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        r = engine.classify("o que dizem as minhas notas sobre Python?")
        assert r.intent.value == "local"


@skip_no_ollama
class TestOllamaLive:
    """Tests that require Ollama running."""

    def test_health(self):
        from orchestrator.llm.ollama import OllamaLLMClient
        client = OllamaLLMClient()
        assert client.health()

    def test_simple_generate(self):
        from orchestrator.llm.ollama import OllamaLLMClient
        client = OllamaLLMClient()
        resp = client.generate("Responde apenas: OK", "gemma3:4b")
        assert len(resp) > 0

    def test_engine_run_general(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        result = engine.run("Diz apenas 'olá'", model_override="gemma3:4b")
        assert len(result.response) > 0
        assert result.model_used == "gemma3:4b"
        assert result.intent.value == "general"

    def test_engine_stream(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        chunks = list(engine.stream("Diz 'teste'", model_override="gemma3:4b"))
        assert len(chunks) > 0
        full = "".join(chunks)
        assert len(full) > 0


@skip_no_ollama
class TestSystemProviderLive:
    """Tests system probe with live system."""

    def test_system_query_returns_context(self):
        from orchestrator.context.system import SystemProbeProvider
        provider = SystemProbeProvider()
        block = provider.get_context("quanta RAM tenho?")
        assert block is not None
        assert "Memory" in block.content

    def test_full_system_query(self):
        from orchestrator.factory import create_engine
        engine = create_engine()
        result = engine.run("quanta RAM tenho livre?", model_override="gemma3:4b")
        assert len(result.response) > 0
        assert "system" in result.sources_used


@skip_no_rag
class TestRAGProviderLive:
    """Tests that require RAG service running."""

    def test_rag_health(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        assert provider.health()

    def test_rag_query(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        block = provider.get_context("Python")
        # May return None if no results, but should not raise
        if block is not None:
            assert block.source == "rag"


class TestFallbackBehaviour:
    """Tests graceful degradation."""

    def test_rag_offline_returns_none(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        # Point to invalid URL
        provider._url = "http://localhost:19999"
        provider._timeout = 1
        result = provider.get_context("test")
        assert result is None

    def test_engine_works_without_rag(self):
        """Engine should work even if RAG is offline."""
        from orchestrator.context.system import SystemProbeProvider
        from orchestrator.core.engine import Engine

        engine = Engine()
        engine.register_provider(SystemProbeProvider())
        routing = engine.classify("o que é DNS?")
        assert routing.intent.value == "general"
