"""Tests for the FastAPI application."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.app import app
from orchestrator.context.base import Complexity, Intent, OrchestratorResult, RoutingResult


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        with patch("orchestrator.api.app._get_engine") as mock_engine:
            engine = MagicMock()
            engine.health_report.return_value = {
                "ollama": True,
                "providers": {"rag": True},
                "all_ok": True,
            }
            mock_engine.return_value = engine
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ollama"] is True


class TestClassifyEndpoint:

    def test_classify(self, client):
        with patch("orchestrator.api.app._get_engine") as mock_engine:
            engine = MagicMock()
            engine.classify.return_value = RoutingResult(
                intent=Intent.LOCAL, complexity=Complexity.SIMPLE,
            )
            mock_engine.return_value = engine
            resp = client.post("/classify", json={"query": "o que são as minhas notas?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "local"
        assert data["complexity"] == "simple"


class TestQueryEndpoint:

    def test_query_non_stream(self, client):
        with patch("orchestrator.api.app._get_engine") as mock_engine:
            engine = MagicMock()
            engine.run.return_value = OrchestratorResult(
                response="Olá!",
                model_used="qwen3:8b",
                intent=Intent.GENERAL,
                complexity=Complexity.NORMAL,
                sources_used=[],
                context_tokens=0,
                latency_ms=42.5,
            )
            mock_engine.return_value = engine
            resp = client.post("/query", json={"query": "olá"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Olá!"
        assert data["model_used"] == "qwen3:8b"

    def test_query_validation_empty(self, client):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422


class TestAuthMiddleware:
    """API key authentication when configured."""

    def test_no_auth_when_key_empty(self, client):
        """All endpoints accessible when api_key is empty."""
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = ""
            mock_cfg.return_value = cfg
            with patch("orchestrator.api.app._get_engine") as mock_engine:
                engine = MagicMock()
                engine.health_report.return_value = {"ollama": True, "providers": {}, "all_ok": True}
                mock_engine.return_value = engine
                resp = client.get("/health")
        assert resp.status_code == 200

    def test_401_when_no_key_provided(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "secret123"
            mock_cfg.return_value = cfg
            resp = client.post("/classify", json={"query": "test"})
        assert resp.status_code == 401

    def test_401_when_wrong_key(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "correct-key"
            mock_cfg.return_value = cfg
            resp = client.post("/classify", json={"query": "test"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_passes_with_x_api_key(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "my-key"
            mock_cfg.return_value = cfg
            with patch("orchestrator.api.app._get_engine") as mock_engine:
                engine = MagicMock()
                engine.classify.return_value = RoutingResult(
                    intent=Intent.GENERAL, complexity=Complexity.SIMPLE,
                )
                mock_engine.return_value = engine
                resp = client.post(
                    "/classify",
                    json={"query": "test"},
                    headers={"X-API-Key": "my-key"},
                )
        assert resp.status_code == 200

    def test_passes_with_bearer_token(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "my-key"
            mock_cfg.return_value = cfg
            with patch("orchestrator.api.app._get_engine") as mock_engine:
                engine = MagicMock()
                engine.classify.return_value = RoutingResult(
                    intent=Intent.GENERAL, complexity=Complexity.SIMPLE,
                )
                mock_engine.return_value = engine
                resp = client.post(
                    "/classify",
                    json={"query": "test"},
                    headers={"Authorization": "Bearer my-key"},
                )
        assert resp.status_code == 200

    def test_health_exempt_from_auth(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "secret"
            mock_cfg.return_value = cfg
            with patch("orchestrator.api.app._get_engine") as mock_engine:
                engine = MagicMock()
                engine.health_report.return_value = {"ollama": True, "providers": {}, "all_ok": True}
                mock_engine.return_value = engine
                resp = client.get("/health")
        assert resp.status_code == 200

    def test_metrics_exempt_from_auth(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = "secret"
            mock_cfg.return_value = cfg
            resp = client.get("/metrics")
        assert resp.status_code == 200


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = ""
            mock_cfg.return_value = cfg
            resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_queries" in data
        assert "avg_latency_ms" in data

    def test_metrics_with_window(self, client):
        with patch("orchestrator.api.app.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.orchestrator.api_key = ""
            mock_cfg.return_value = cfg
            resp = client.get("/metrics?window=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window_seconds"] == 0
