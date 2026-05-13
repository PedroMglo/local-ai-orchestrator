"""Tests for the MetricsCollector."""

from __future__ import annotations

from enum import Enum
from unittest.mock import MagicMock

from orchestrator.core.metrics import MetricsCollector


class _Intent(Enum):
    LOCAL = "local"
    GENERAL = "general"


class _Complexity(Enum):
    NORMAL = "normal"
    SIMPLE = "simple"


def _make_result(*, latency_ms=100.0, intent="local", complexity="normal", model="qwen3:8b", context_tokens=500):
    r = MagicMock()
    r.latency_ms = latency_ms
    r.intent = _Intent(intent)
    r.complexity = _Complexity(complexity)
    r.model_used = model
    r.context_tokens = context_tokens
    return r


class TestMetricsCollector:

    def test_empty_summary(self):
        m = MetricsCollector()
        s = m.summary()
        assert s["total_queries"] == 0
        assert s["avg_latency_ms"] == 0

    def test_record_and_summary(self):
        m = MetricsCollector()
        m.record(_make_result(latency_ms=100.0))
        m.record(_make_result(latency_ms=200.0))
        s = m.summary(window_seconds=0)
        assert s["total_queries"] == 2
        assert s["avg_latency_ms"] == 150.0
        assert s["intent_distribution"] == {"local": 2}
        assert s["model_distribution"] == {"qwen3:8b": 2}

    def test_p95_latency(self):
        m = MetricsCollector()
        for i in range(100):
            m.record(_make_result(latency_ms=float(i + 1)))
        s = m.summary(window_seconds=0)
        assert s["total_queries"] == 100
        assert s["p95_latency_ms"] == 95.0

    def test_ring_buffer_maxlen(self):
        m = MetricsCollector(maxlen=5)
        for i in range(10):
            m.record(_make_result(latency_ms=float(i)))
        s = m.summary(window_seconds=0)
        assert s["total_queries"] == 5

    def test_window_filter(self):
        import time
        m = MetricsCollector()
        # Manually add an old record
        from orchestrator.core.metrics import _QueryRecord
        m._records.append(_QueryRecord(
            timestamp=time.time() - 600,
            latency_ms=999.0,
            intent="general",
            complexity="normal",
            model="old",
            context_tokens=0,
        ))
        m.record(_make_result(latency_ms=50.0))
        s = m.summary(window_seconds=300)
        assert s["total_queries"] == 1
        assert s["avg_latency_ms"] == 50.0

    def test_mixed_intents(self):
        m = MetricsCollector()
        m.record(_make_result(intent="local"))
        m.record(_make_result(intent="general"))
        m.record(_make_result(intent="local"))
        s = m.summary(window_seconds=0)
        assert s["intent_distribution"] == {"local": 2, "general": 1}
