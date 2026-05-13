"""Graph context provider — reads Graphify graph.json from filesystem."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from orchestrator.config import get_settings
from orchestrator.context.base import ContextBlock, estimate_tokens

log = logging.getLogger(__name__)


class _CachedGraph:
    __slots__ = ("data", "mtime", "size", "loaded_at")

    def __init__(self, data: dict, mtime: float, size: int):
        self.data = data
        self.mtime = mtime
        self.size = size
        self.loaded_at = time.time()


class GraphProvider:
    """Provides knowledge graph context from Graphify output."""

    def __init__(self) -> None:
        self._cache: dict[str, _CachedGraph] = {}

    @property
    def name(self) -> str:
        return "graph"

    def _graph_paths(self) -> list[tuple[str, Path]]:
        """Find all graph.json files for configured repos."""
        cfg = get_settings()
        result: list[tuple[str, Path]] = []
        for repo_path in cfg.repos.paths:
            repo_name = Path(repo_path).name
            gp = cfg.graph.output_dir / repo_name / "graphify-out" / "graph.json"
            if gp.exists():
                result.append((repo_name, gp))
        return result

    def _load_graph(self, repo_name: str, path: Path) -> dict | None:
        cfg = get_settings()
        cached = self._cache.get(repo_name)

        if cached is not None:
            # Always check mtime/size first — detect changes even within TTL
            try:
                st = os.stat(path)
                if st.st_mtime == cached.mtime and st.st_size == cached.size:
                    age = time.time() - cached.loaded_at
                    if age >= cfg.graph.cache_ttl:
                        cached.loaded_at = time.time()  # refresh TTL
                    return cached.data
            except OSError:
                pass
            # File changed or stat failed — fall through to reload

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            st = os.stat(path)
            self._cache[repo_name] = _CachedGraph(data, st.st_mtime, st.st_size)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("GraphProvider: failed to load %s: %s", path, exc)
            return None

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        graph_paths = self._graph_paths()
        if not graph_paths:
            return None

        parts: list[str] = []
        total_tokens = 0

        for repo_name, path in graph_paths:
            data = self._load_graph(repo_name, path)
            if data is None:
                continue

            nodes = data.get("nodes", [])
            edges = data.get("links", data.get("edges", []))

            # Build summary
            lines = [f"## {repo_name} ({len(nodes)} nodes, {len(edges)} edges)"]

            # God nodes (high connectivity)
            degree: dict[str, int] = {}
            for edge in edges:
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                degree[src] = degree.get(src, 0) + 1
                degree[tgt] = degree.get(tgt, 0) + 1

            top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_nodes:
                lines.append("### Key nodes")
                node_map = {n.get("id", ""): n for n in nodes}
                for nid, deg in top_nodes:
                    node = node_map.get(nid, {})
                    label = node.get("label", nid)
                    sf = node.get("source_file", "")
                    lines.append(f"- {label} ({sf}) — {deg} connections")

            # Community summaries
            summaries_path = path.parent / "community_summaries.json"
            if summaries_path.exists():
                try:
                    with open(summaries_path, encoding="utf-8") as f:
                        summaries = json.load(f)
                    if summaries:
                        lines.append("### Communities")
                        for cid, summary in list(summaries.items())[:3]:
                            lines.append(f"- Community {cid}: {summary[:200]}")
                except (json.JSONDecodeError, OSError):
                    pass

            block = "\n".join(lines)
            block_tokens = estimate_tokens(block)
            if total_tokens + block_tokens > budget_tokens and parts:
                break
            parts.append(block)
            total_tokens += block_tokens

        if not parts:
            return None

        content = "\n\n".join(parts)
        return ContextBlock(
            source="graph",
            content=content,
            token_estimate=total_tokens,
        )

    def health(self) -> bool:
        return len(self._graph_paths()) > 0
