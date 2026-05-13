"""Configuração centralizada — carrega orchestrator.toml com suporte a env overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find orchestrator.toml, or check CWD."""
    current = Path(__file__).resolve().parent.parent
    if (current / "orchestrator.toml").exists():
        return current
    cwd = Path.cwd()
    if (cwd / "orchestrator.toml").exists():
        return cwd
    return Path.home() / "ai-local" / "orchestrator"


PROJECT_ROOT = _find_project_root()


def _load_toml() -> dict:
    path = PROJECT_ROOT / "orchestrator.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _env(section: str, key: str, default):
    """Check for env var ORC_{SECTION}_{KEY} (uppercase)."""
    env_key = f"ORC_{section.upper()}_{key.upper()}"
    val = os.environ.get(env_key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def _resolve_path(raw: str) -> Path:
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestratorConfig:
    host: str = "127.0.0.1"
    port: int = 8585
    api_key: str = ""


@dataclass(frozen=True)
class RAGConfig:
    url: str = "http://localhost:8484"
    timeout: int = 30
    health_interval: int = 30
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset: int = 60


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    max_concurrent_llm: int = 1


@dataclass(frozen=True)
class ModelsConfig:
    default: str = "qwen3:8b"
    fast: str = "gemma3:4b"
    code: str = "qwen2.5-coder:7b"
    deep: str = "deepseek-r1:8b"
    embedding: str = "bge-m3"


@dataclass(frozen=True)
class CAGConfig:
    db_path: str = ""


@dataclass(frozen=True)
class ContextConfig:
    token_budget: int = 6000
    provider_timeout: int = 10
    cag: CAGConfig = field(default_factory=CAGConfig)


@dataclass(frozen=True)
class ReposConfig:
    paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GraphConfig:
    output_dir: Path = field(default_factory=lambda: Path.home() / "ai-local" / "obsidian-rag" / "data" / "graphify")
    cache_ttl: int = 300


@dataclass(frozen=True)
class SecurityConfig:
    allowed_commands: frozenset[str] = frozenset({
        "free", "nvidia-smi", "df", "ps", "uptime", "nproc",
        "uname", "ip", "sensors", "git", "ollama", "docker",
        "cat", "wc", "head", "tail",
    })
    max_command_timeout: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "text"


@dataclass(frozen=True)
class SessionConfig:
    enabled: bool = False
    ttl_seconds: int = 3600
    db_path: str = ""
    max_messages: int = 20


@dataclass(frozen=True)
class Settings:
    orchestrator: OrchestratorConfig
    rag: RAGConfig
    ollama: OllamaConfig
    models: ModelsConfig
    context: ContextConfig
    repos: ReposConfig
    graph: GraphConfig
    security: SecurityConfig
    logging: LoggingConfig
    session: SessionConfig


def load_settings() -> Settings:
    raw = _load_toml()

    o = raw.get("orchestrator", {})
    orchestrator = OrchestratorConfig(
        host=_env("orchestrator", "host", o.get("host", "127.0.0.1")),
        port=_env("orchestrator", "port", o.get("port", 8585)),
        api_key=_env("orchestrator", "api_key", o.get("api_key", "")),
    )

    r = raw.get("rag", {})
    rag = RAGConfig(
        url=_env("rag", "url", r.get("url", "http://localhost:8484")),
        timeout=_env("rag", "timeout", r.get("timeout", 30)),
        health_interval=_env("rag", "health_interval", r.get("health_interval", 30)),
        circuit_breaker_threshold=_env("rag", "circuit_breaker_threshold", r.get("circuit_breaker_threshold", 3)),
        circuit_breaker_reset=_env("rag", "circuit_breaker_reset", r.get("circuit_breaker_reset", 60)),
    )

    ol = raw.get("ollama", {})
    ollama = OllamaConfig(
        base_url=_env("ollama", "base_url", ol.get("base_url", "http://localhost:11434")),
        max_concurrent_llm=_env("ollama", "max_concurrent_llm", ol.get("max_concurrent_llm", 1)),
    )

    m = raw.get("models", {})
    models = ModelsConfig(
        default=_env("models", "default", m.get("default", "qwen3:8b")),
        fast=_env("models", "fast", m.get("fast", "gemma3:4b")),
        code=_env("models", "code", m.get("code", "qwen2.5-coder:7b")),
        deep=_env("models", "deep", m.get("deep", "deepseek-r1:8b")),
        embedding=_env("models", "embedding", m.get("embedding", "bge-m3")),
    )

    cx = raw.get("context", {})
    cag_raw = cx.get("cag", {})
    cag = CAGConfig(
        db_path=_env("context", "cag_db_path", cag_raw.get("db_path", "")),
    )
    context = ContextConfig(
        token_budget=_env("context", "token_budget", cx.get("token_budget", 6000)),
        provider_timeout=_env("context", "provider_timeout", cx.get("provider_timeout", 10)),
        cag=cag,
    )

    rp = raw.get("repos", {})
    raw_paths = rp.get("paths", [])
    if isinstance(raw_paths, str):
        raw_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    repos = ReposConfig(
        paths=tuple(_resolve_path(p) for p in raw_paths),
    )

    g = raw.get("graph", {})
    graph = GraphConfig(
        output_dir=_resolve_path(_env("graph", "output_dir", g.get("output_dir", "~/ai-local/obsidian-rag/data/graphify"))),
        cache_ttl=_env("graph", "cache_ttl", g.get("cache_ttl", 300)),
    )

    s = raw.get("security", {})
    allowed = s.get("allowed_commands", list(SecurityConfig.allowed_commands))
    security = SecurityConfig(
        allowed_commands=frozenset(allowed),
        max_command_timeout=_env("security", "max_command_timeout", s.get("max_command_timeout", 5)),
    )

    lg = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=_env("logging", "level", lg.get("level", "INFO")),
        format=_env("logging", "format", lg.get("format", "text")),
    )

    se = raw.get("session", {})
    session = SessionConfig(
        enabled=_env("session", "enabled", se.get("enabled", False)),
        ttl_seconds=_env("session", "ttl_seconds", se.get("ttl_seconds", 3600)),
        db_path=_env("session", "db_path", se.get("db_path", "")),
        max_messages=_env("session", "max_messages", se.get("max_messages", 20)),
    )

    return Settings(
        orchestrator=orchestrator,
        rag=rag,
        ollama=ollama,
        models=models,
        context=context,
        repos=repos,
        graph=graph,
        security=security,
        logging=logging_cfg,
        session=session,
    )


# Module-level singleton (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _reset_settings() -> None:
    """Reset singleton — for testing."""
    global _settings
    _settings = None
