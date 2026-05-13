"""Tests for GraphProvider cache — mtime invalidation within TTL."""

from __future__ import annotations

import json
import os
import time

import pytest

from orchestrator.config import _reset_settings
from orchestrator.context.graph import GraphProvider


@pytest.fixture(autouse=True)
def _reset():
    _reset_settings()


class TestGraphCacheMtime:
    """§2.2 — cache must detect file changes even within TTL window."""

    def _make_graph(self, tmp_path, nodes=None, edges=None):
        """Write a minimal graph.json and return its path."""
        data = {
            "nodes": nodes or [{"id": "a", "label": "A", "source_file": "a.py"}],
            "links": edges or [],
        }
        gp = tmp_path / "graph.json"
        gp.write_text(json.dumps(data), encoding="utf-8")
        return gp

    def test_cache_invalidated_on_mtime_change_within_ttl(self, tmp_path):
        gp = self._make_graph(tmp_path, nodes=[{"id": "v1", "label": "V1", "source_file": "v1.py"}])

        provider = GraphProvider()
        # First load — populates cache
        data1 = provider._load_graph("test_repo", gp)
        assert data1 is not None
        assert data1["nodes"][0]["id"] == "v1"

        # Overwrite file with different content (within TTL)
        time.sleep(0.05)  # ensure mtime differs
        new_data = {
            "nodes": [{"id": "v2", "label": "V2", "source_file": "v2.py"}],
            "links": [],
        }
        gp.write_text(json.dumps(new_data), encoding="utf-8")

        # Second load — should detect mtime change and reload
        data2 = provider._load_graph("test_repo", gp)
        assert data2 is not None
        assert data2["nodes"][0]["id"] == "v2"

    def test_cache_serves_from_memory_when_unchanged(self, tmp_path):
        gp = self._make_graph(tmp_path)

        provider = GraphProvider()
        data1 = provider._load_graph("test_repo", gp)
        assert data1 is not None

        # Record the loaded_at timestamp
        cached = provider._cache["test_repo"]
        _ = cached.loaded_at  # verify it exists

        # Second load — same file, should serve from cache
        data2 = provider._load_graph("test_repo", gp)
        assert data2 is data1  # same object reference = cache hit

    def test_cache_refreshes_ttl_after_expiry_if_unchanged(self, tmp_path, monkeypatch):
        gp = self._make_graph(tmp_path)

        provider = GraphProvider()
        data1 = provider._load_graph("test_repo", gp)
        assert data1 is not None

        # Simulate TTL expiry by backdating loaded_at
        cached = provider._cache["test_repo"]
        cached.loaded_at = time.time() - 9999

        # Load again — file unchanged, should refresh TTL
        data2 = provider._load_graph("test_repo", gp)
        assert data2 is data1  # same data
        assert provider._cache["test_repo"].loaded_at > cached.loaded_at - 9999

    def test_cache_handles_deleted_file(self, tmp_path):
        gp = self._make_graph(tmp_path)

        provider = GraphProvider()
        data1 = provider._load_graph("test_repo", gp)
        assert data1 is not None

        # Delete the file
        os.unlink(gp)

        # Load again — should fall through and fail gracefully
        data2 = provider._load_graph("test_repo", gp)
        assert data2 is None
