"""Context provider protocol and shared data types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using word-boundary heuristic.

    Uses ``re.findall(r'\\b\\w+\\b', text) * 1.3`` which is more accurate
    for mixed PT-PT/EN text and code than the naive ``len // 4`` approach.
    Returns at least 1 for non-empty text.
    """
    if not text:
        return 0
    return max(1, int(len(_WORD_RE.findall(text)) * 1.3))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """Classificação de intenção da query."""
    GENERAL = "general"
    LOCAL = "local"           # notas pessoais, vault, ficheiros
    CODE = "code"             # código, programação
    SYSTEM = "system"         # estado da máquina (RAM, GPU, disco)
    GRAPH = "graph"           # arquitectura, dependências, fluxo
    LOCAL_AND_GRAPH = "local_and_graph"
    SYSTEM_AND_LOCAL = "system_and_local"
    CLARIFY = "clarify"


class Complexity(str, Enum):
    """Classificação de complexidade da query."""
    SIMPLE = "simple"         # pergunta directa, resposta curta
    NORMAL = "normal"         # pergunta padrão
    COMPLEX = "complex"       # multi-part, boolean, relacional
    DEEP = "deep"             # raciocínio profundo, análise


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBlock:
    """Bloco de contexto retornado por um ContextProvider."""
    source: str               # nome do provider (ex: "rag", "system", "graph")
    content: str              # texto do contexto
    token_estimate: int = 0   # estimativa de tokens
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingResult:
    """Resultado do intent + complexity classification."""
    intent: Intent
    complexity: Complexity
    confidence: float = 0.0
    reason: str = ""
    method: str = "heuristic"  # "heuristic" | "llm"


@dataclass(frozen=True)
class OrchestratorResult:
    """Resultado final do Orquestrador."""
    response: str
    model_used: str
    intent: Intent
    complexity: Complexity
    sources_used: list[str] = field(default_factory=list)
    context_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ContextProvider(Protocol):
    """Interface para fontes de contexto."""

    @property
    def name(self) -> str: ...

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        """Obter contexto relevante para a query. Retorna None se indisponível."""
        ...

    def health(self) -> bool:
        """True se o provider está operacional."""
        ...


@runtime_checkable
class IntentClassifier(Protocol):
    """Classifica a intenção da query."""

    def classify(self, query: str, *, history: list[dict] | None = None) -> Intent: ...


@runtime_checkable
class ComplexityClassifier(Protocol):
    """Classifica a complexidade da query."""

    def classify(self, query: str) -> Complexity: ...


@runtime_checkable
class ModelRouter(Protocol):
    """Seleciona o modelo ideal."""

    def select(self, intent: Intent, complexity: Complexity) -> str: ...


@runtime_checkable
class ContextRouter(Protocol):
    """Decide que fontes de contexto consultar."""

    def route(self, intent: Intent, complexity: Complexity) -> list[str]: ...
