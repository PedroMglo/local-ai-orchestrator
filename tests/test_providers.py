"""Tests for context providers — mostly unit tests with mocked dependencies."""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import patch

from orchestrator.context.base import ContextBlock

# ── RAG Provider ──────────────────────────────────────────────────────────────

class TestRAGProvider:

    def test_circuit_breaker_opens(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        # Simulate failures
        for _ in range(provider._threshold):
            provider._record_failure()
        assert provider._is_circuit_open()

    def test_circuit_breaker_resets(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        for _ in range(provider._threshold):
            provider._record_failure()
        assert provider._is_circuit_open()
        provider._record_success()
        assert not provider._is_circuit_open()

    def test_returns_none_when_circuit_open(self):
        from orchestrator.context.rag import RAGContextProvider
        provider = RAGContextProvider()
        for _ in range(provider._threshold):
            provider._record_failure()
        provider._open_until = time.monotonic() + 9999
        result = provider.get_context("test query")
        assert result is None

    def test_name(self):
        from orchestrator.context.rag import RAGContextProvider
        assert RAGContextProvider().name == "rag"


# ── CAG Provider ──────────────────────────────────────────────────────────────

class TestCAGProvider:

    def test_returns_none_without_db(self):
        from orchestrator.context.cag import CAGContextProvider
        provider = CAGContextProvider()
        with patch.object(provider, "_db_path", return_value=None):
            result = provider.get_context("test")
        assert result is None

    def test_reads_valid_packs(self, tmp_path):
        from orchestrator.context.cag import CAGContextProvider

        db_file = tmp_path / "cag.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE packs (pack_type TEXT, content TEXT, expires_at REAL)")
        conn.execute(
            "INSERT INTO packs VALUES (?, ?, ?)",
            ("config_environment", "Ollama running on port 11434", time.time() + 3600),
        )
        conn.commit()
        conn.close()

        provider = CAGContextProvider(intent_hint="general")
        with patch.object(provider, "_db_path", return_value=db_file):
            result = provider.get_context("what config?")
        assert result is not None
        assert "config_environment" in result.content

    def test_set_intent_hint(self):
        from orchestrator.context.cag import CAGContextProvider
        provider = CAGContextProvider(intent_hint="general")
        provider.set_intent_hint("code")
        assert provider._intent_hint == "code"

    def test_name(self):
        from orchestrator.context.cag import CAGContextProvider
        assert CAGContextProvider().name == "cag"


# ── System Probe ──────────────────────────────────────────────────────────────

class TestSystemProbe:

    def test_name(self):
        from orchestrator.context.system import SystemProbeProvider
        assert SystemProbeProvider().name == "system"

    def test_detects_memory_subsystem(self):
        from orchestrator.context.system import _detect_subsystems
        subs = _detect_subsystems("quanta RAM tenho livre?")
        assert "memory" in subs

    def test_detects_gpu_subsystem(self):
        from orchestrator.context.system import _detect_subsystems
        subs = _detect_subsystems("mostra a VRAM da GPU")
        assert "gpu" in subs

    def test_health_always_true(self):
        from orchestrator.context.system import SystemProbeProvider
        assert SystemProbeProvider().health()

    def test_get_context_returns_block(self):
        from orchestrator.context.system import SystemProbeProvider
        provider = SystemProbeProvider()
        result = provider.get_context("quanta RAM tenho?")
        # May return None if 'free' not found, but should not raise
        if result is not None:
            assert isinstance(result, ContextBlock)


# ── Repo Probe ────────────────────────────────────────────────────────────────

class TestRepoProbe:

    def test_name(self):
        from orchestrator.context.repo import RepoProbeProvider
        assert RepoProbeProvider().name == "repo"

    def test_health_always_true(self):
        from orchestrator.context.repo import RepoProbeProvider
        assert RepoProbeProvider().health()


# ── Graph Provider ────────────────────────────────────────────────────────────

class TestGraphProvider:

    def test_name(self):
        from orchestrator.context.graph import GraphProvider
        assert GraphProvider().name == "graph"

    def test_reads_graph_json(self, tmp_path):
        from orchestrator.context.graph import GraphProvider

        # Create a graph.json
        graph_dir = tmp_path / "test-repo" / "graphify-out"
        graph_dir.mkdir(parents=True)
        graph_file = graph_dir / "graph.json"
        graph_file.write_text(json.dumps({
            "nodes": [
                {"id": "A", "label": "Module A", "source_file": "a.py"},
                {"id": "B", "label": "Module B", "source_file": "b.py"},
            ],
            "links": [
                {"source": "A", "target": "B"},
            ],
        }))

        provider = GraphProvider()
        # Mock graph paths
        with patch.object(provider, "_graph_paths", return_value=[("test-repo", graph_file)]):
            result = provider.get_context("architecture")
        assert result is not None
        assert "Module A" in result.content


# ── Config/Env Provider ──────────────────────────────────────────────────────

class TestConfigEnvProvider:

    def test_name(self):
        from orchestrator.context.config_env import ConfigEnvProvider
        assert ConfigEnvProvider().name == "config"

    def test_returns_block(self):
        from orchestrator.context.config_env import ConfigEnvProvider
        provider = ConfigEnvProvider()
        result = provider.get_context("config")
        assert result is not None
        assert "Models" in result.content


# ── Logs Provider ─────────────────────────────────────────────────────────────

class TestLogsProvider:

    def test_name(self):
        from orchestrator.context.logs import LogsProvider
        assert LogsProvider().name == "logs"

    def test_health_always_true(self):
        from orchestrator.context.logs import LogsProvider
        assert LogsProvider().health()
