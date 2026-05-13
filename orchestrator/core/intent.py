"""Keyword-based intent classification with optional LLM fallback.

Extraído e adaptado de obsidian_rag/retrieval/router.py.
"""

from __future__ import annotations

import logging

from orchestrator.context.base import Intent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets (PT + EN)
# ---------------------------------------------------------------------------

_LOCAL_SIGNALS = frozenset({
    "meu", "minha", "meus", "minhas", "nosso", "nossa",
    "my", "our", "mine",
    "obsidian", "vault", "notas", "notes",
    "repo", "repositório", "repository", "projeto", "project",
    "ficheiro", "ficheiros", "file", "files",
    "código", "code", "script", "scripts",
    "configuração", "config", "setup",
    "documentos", "documents", "docs",
    "indexado", "indexed", "local",
    "pipeline", "codebase", "workspace",
    "modelfile", "modelfiles",
    "instalado", "instalados", "installed",
    "configurado", "configurados", "configured",
    "alias", "aliases", "funções", "functions",
})

_GRAPH_SIGNALS = frozenset({
    "depende", "dependência", "dependências", "depends", "dependency",
    "chama", "chamada", "calls", "called",
    "importa", "imports", "importação",
    "fluxo", "flow", "pipeline", "cadeia", "chain",
    "arquitectura", "arquitetura", "architecture", "structure", "estrutura",
    "impacto", "impact", "afeta", "affects",
    "relação", "relações", "relation", "relations", "relationship",
    "componente", "componentes", "component", "components",
    "módulo", "módulos", "module", "modules",
    "vizinhos", "neighbors", "neighbour",
    "comunidade", "community",
    "grafo", "graph",
    "upstream", "downstream", "montante", "jusante",
})

_SYSTEM_SIGNALS = frozenset({
    "ram", "memória", "memory", "vram",
    "gpu", "cpu", "processador", "processor",
    "disco", "disk", "storage", "armazenamento",
    "temperatura", "temperature", "temp",
    "processos", "processes",
    "sistema", "system",
    "kernel", "driver", "drivers",
    "nvidia", "amd", "cuda",
    "swap", "hardware",
    "máquina", "machine", "pc", "computador", "computer",
    "uptime", "carga", "load",
    "espaço", "space",
    "rede", "network", "ip",
    "bateria", "battery",
})

_CODE_SIGNALS = frozenset({
    "função", "função", "function", "method", "método",
    "classe", "class", "refactor", "refactora", "refatorar",
    "bug", "bugs", "debug", "debugging", "depurar",
    "implementa", "implement", "escreve", "write",
    "código", "code", "script", "programa", "program",
    "async", "await", "loop", "recursão", "recursion",
    "api", "endpoint", "rest", "grpc",
    "teste", "test", "testa", "testing",
    "compila", "compile", "build", "docker",
    "erro", "error", "exception", "traceback",
    "python", "javascript", "typescript", "rust", "go", "java",
    "sql", "bash", "shell", "zsh",
})

_SYSTEM_PATTERNS = (
    "quanto de ram", "quanta ram", "quanta memória", "memória livre",
    "espaço em disco", "espaço livre", "uso do disco",
    "temperatura do", "temperatura da",
    "está a usar gpu", "está a usar a gpu",
    "o que está a correr", "o que está a consumir",
    "processos activos", "processos ativos",
    "carga do sistema", "uso de cpu",
    "how much memory", "how much ram", "free memory", "free ram",
    "disk space", "disk usage", "free space",
    "temperature of", "gpu temperature", "cpu temperature",
    "using my gpu", "what is running", "what is using",
    "system load", "cpu usage", "memory usage",
)

_SYSTEM_FALSE_POSITIVES = (
    "machine learning", "system design", "system prompt",
    "operating system", "file system", "type system",
    "memory model", "memory management", "memory leak",
    "memory safety", "space complexity", "disk image",
    "network protocol", "network layer", "ip address",
    "load balancing", "load balancer",
)

_GRAPH_PATTERNS = (
    "como funciona", "como é que", "o que chama", "quem chama",
    "o que depende", "quem depende", "qual o fluxo", "qual é o fluxo",
    "que relação", "como se liga", "como interage",
    "o que acontece se mudar", "impacto de alterar",
    "este projeto", "este repo", "este módulo", "este pipeline",
    "o meu pipeline", "o meu repo", "o meu projeto",
    "how does", "what calls", "what depends", "call chain", "call flow",
    "depends on", "used by", "calls to",
    "this project", "this repo", "this module", "this pipeline",
    "my pipeline", "my repo", "my project",
)


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

class HeuristicIntentClassifier:
    """Keyword-based intent classification."""

    def classify(self, query: str, *, history: list[dict] | None = None) -> Intent:
        q_lower = query.lower()
        words = {w.strip(".,!?:;\"'()[]{}") for w in q_lower.split()}
        words.discard("")

        has_local = bool(words & _LOCAL_SIGNALS)
        has_graph = bool(words & _GRAPH_SIGNALS) or any(p in q_lower for p in _GRAPH_PATTERNS)
        has_system = bool(words & _SYSTEM_SIGNALS) or any(p in q_lower for p in _SYSTEM_PATTERNS)
        has_code = bool(words & _CODE_SIGNALS)

        # Suppress system false positives
        if has_system and any(fp in q_lower for fp in _SYSTEM_FALSE_POSITIVES):
            fp_words: set[str] = set()
            for fp in _SYSTEM_FALSE_POSITIVES:
                if fp in q_lower:
                    fp_words.update(fp.split())
            remaining = (words & _SYSTEM_SIGNALS) - fp_words
            remaining_patterns = any(
                p in q_lower for p in _SYSTEM_PATTERNS
                if not any(fp in q_lower and p in fp for fp in _SYSTEM_FALSE_POSITIVES)
            )
            has_system = bool(remaining) or remaining_patterns

        # Priority: system > code > graph > local > general
        if has_system and has_local:
            return Intent.SYSTEM_AND_LOCAL
        if has_system:
            return Intent.SYSTEM
        if has_code and not has_local and not has_graph:
            return Intent.CODE
        if has_local and has_graph:
            return Intent.LOCAL_AND_GRAPH
        if has_graph:
            project_hints = {"meu", "minha", "nosso", "my", "our", "repo", "projeto", "project"}
            if words & project_hints:
                return Intent.LOCAL_AND_GRAPH
            return Intent.GENERAL
        if has_local:
            return Intent.LOCAL
        if has_code:
            return Intent.CODE

        return Intent.GENERAL
