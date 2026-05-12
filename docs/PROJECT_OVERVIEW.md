# PROJECT OVERVIEW — ai-orchestrator

> **Versão:** 0.1.0
> **Última atualização:** 2026-05-12
> **Linguagem:** Python ≥ 3.11
> **Repositório:** `PedroMglo/local-ai-sys` · branch `dev-f2`

---

## Índice

1. [Objetivo principal](#1-objetivo-principal)
2. [Problema que resolve](#2-problema-que-resolve)
3. [Relação com obsidian-rag](#3-relação-com-obsidian-rag)
4. [Arquitectura geral](#4-arquitectura-geral)
5. [Fluxo de funcionamento](#5-fluxo-de-funcionamento)
6. [Componentes principais](#6-componentes-principais)
7. [Context Providers](#7-context-providers)
8. [Modelos e routing](#8-modelos-e-routing)
9. [API REST](#9-api-rest)
10. [CLI — comando `orc`](#10-cli--comando-orc)
11. [Shell integration — função `ai`](#11-shell-integration--função-ai)
12. [Configuração](#12-configuração)
13. [Ficheiros externos editados](#13-ficheiros-externos-editados)
14. [Testes](#14-testes)
15. [Dependências](#15-dependências)
16. [Como executar](#16-como-executar)
17. [Estado actual](#17-estado-actual)

---

## 1. Objetivo principal

O **ai-orchestrator** é a camada de inteligência entre o utilizador e os modelos de linguagem locais. Recebe uma query em linguagem natural, classifica a intenção e complexidade, agrega contexto das fontes relevantes (notas, código, sistema, grafo de conhecimento), seleciona automaticamente o modelo mais adequado e retorna a resposta.

É completamente separado do `obsidian-rag` — comunica com ele apenas via HTTP.

---

## 2. Problema que resolve

| Problema                                                                     | Solução                                                                 |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Seleção manual de modelo (o utilizador escolhia `ol coder`, `ol deep`, etc.) | Routing automático por intent × complexity                              |
| Fallback binário: RAG disponível ou Ollama directo                           | Fallback em 3 níveis: API → CLI → Ollama                                |
| Modelo hardcoded nos aliases shell                                           | Tabela de routing configurável em `orchestrator.toml`                   |
| RAG acoplado ao LLM (build_rag_context fazia tudo)                           | Separação de responsabilidades: RAG = retrieval, Orchestrator = decisão |
| Sem classificação de complexidade                                            | `ComplexityClassifier` com 4 níveis: simple/normal/complex/deep         |
| Sem contexto de sistema nas respostas                                        | `SystemProbeProvider` injeta RAM, GPU, disco, CPU em tempo real         |

---

## 3. Relação com obsidian-rag

```
┌─────────────────────────────────────────────┐
│  Shell (~/.zsh_custom.d/42-ai.zsh)          │
│  função ai() — entrada do utilizador        │
└─────────────┬───────────────────────────────┘
              │ HTTP ou CLI
              ▼
┌─────────────────────────────────────────────┐
│  ai-orchestrator  (localhost:8585)          │
│  ├─ IntentClassifier                        │
│  ├─ ComplexityClassifier                    │
│  ├─ ContextRouter → Context Providers       │
│  │   ├─ RAGContextProvider ──────────────┐  │
│  │   ├─ CAGContextProvider              │  │
│  │   ├─ SystemProbeProvider             │  │
│  │   ├─ RepoProbeProvider               │  │
│  │   ├─ GraphProvider                   │  │
│  │   └─ ConfigEnvProvider / LogsProvider │  │
│  ├─ ModelRouter                          │  │
│  └─ OllamaLLMClient                     │  │
└─────────────────────────────────────────┼──┘
                                          │ HTTP
              ┌───────────────────────────▼───┐
              │  obsidian-rag  (localhost:8484)│
              │  GET /health                  │
              │  POST /query                  │
              │  POST /query/code             │
              └───────────────────────────────┘
                            │
              ┌─────────────▼──────────┐
              │  Qdrant (localhost:6333)│
              └────────────────────────┘
```

O `obsidian-rag` **não é modificado** — continua como serviço autónomo. O orquestrador é puramente aditivo.

---

## 4. Arquitectura geral

### Pipeline sequencial (Engine.run)

```
query
  │
  ├─ HeuristicIntentClassifier     → Intent
  ├─ HeuristicComplexityClassifier → Complexity
  ├─ ConfigContextRouter           → [source names]
  ├─ _gather_context()             → [ContextBlock, ...]
  │    └─ (cada provider por ordem, com token budget)
  ├─ ConfigModelRouter             → model name
  ├─ _build_messages()             → [system, context, history, user]
  └─ OllamaLLMClient.chat()        → response string
```

### Estrutura de directórios

```
~/ai-local/orchestrator/
├── orchestrator.toml           # configuração principal
├── pyproject.toml              # metadata e deps
├── .venv/                      # venv isolado
├── docs/                       # esta pasta
├── orchestrator/
│   ├── config.py               # Settings (TOML + env ORC_*)
│   ├── factory.py              # create_engine() — wires all providers
│   ├── context/
│   │   ├── base.py             # Protocols + Enums + DataClasses
│   │   ├── rag.py              # RAGContextProvider
│   │   ├── cag.py              # CAGContextProvider
│   │   ├── system.py           # SystemProbeProvider
│   │   ├── repo.py             # RepoProbeProvider
│   │   ├── graph.py            # GraphProvider
│   │   ├── config_env.py       # ConfigEnvProvider
│   │   └── logs.py             # LogsProvider
│   ├── core/
│   │   ├── engine.py           # Engine (orquestração principal)
│   │   ├── intent.py           # HeuristicIntentClassifier
│   │   ├── complexity.py       # HeuristicComplexityClassifier
│   │   ├── model_router.py     # ConfigModelRouter
│   │   └── context_router.py   # ConfigContextRouter
│   ├── llm/
│   │   ├── base.py             # LLMClient Protocol
│   │   └── ollama.py           # OllamaLLMClient
│   ├── api/
│   │   ├── app.py              # FastAPI (POST /query, /classify, GET /health)
│   │   └── schemas.py          # Pydantic models
│   └── cli/
│       └── main.py             # orc CLI (ask, classify, serve, health, config)
└── tests/
    ├── conftest.py
    ├── test_intent.py          # 19 testes
    ├── test_complexity.py      # 11 testes
    ├── test_model_router.py    # 7 testes
    ├── test_context_router.py  # 8 testes
    ├── test_engine.py          # 10 testes
    ├── test_config.py          # 5 testes
    ├── test_providers.py       # 16 testes
    ├── test_api.py             # 4 testes
    ├── test_cli.py             # 4 testes
    └── test_integration.py     # 18 testes (requerem Ollama/RAG)
```

---

## 5. Fluxo de funcionamento

### Query completa (engine.run)

1. **Classificação de intent** — `HeuristicIntentClassifier` analisa keywords PT+EN e identifica o tipo de pergunta (geral, notas locais, código, sistema, grafo, combinado).
2. **Classificação de complexidade** — `HeuristicComplexityClassifier` avalia comprimento, sinais de raciocínio profundo e indicadores de código para escolher entre `simple/normal/complex/deep`.
3. **Context routing** — `ConfigContextRouter` consulta uma tabela estática `intent → [providers]` e retorna a lista de fontes a activar.
4. **Recolha de contexto** — `Engine._gather_context()` chama cada provider por ordem, respeitando o budget global de tokens (por defeito 6000). Providers falhos são ignorados sem falhar a query.
5. **Seleção de modelo** — `ConfigModelRouter` consulta a tabela `(intent, complexity) → model_key` e resolve para o nome de modelo configurado.
6. **Construção de mensagens** — `_build_messages()` monta: system prompt PT-PT, context blocks (tagged), histórico de conversa (opcional), query do utilizador.
7. **LLM call** — `OllamaLLMClient.chat()` chama `POST /api/chat` do Ollama com streaming opcional. Blocos `<think>` são removidos automaticamente.

### Streaming (engine.stream)

Mesmo pipeline até ao passo 6, depois `OllamaLLMClient.chat_stream()` itera sobre tokens via `httpx.stream`.

---

## 6. Componentes principais

### `orchestrator/context/base.py` — Tipos e Protocols

| Tipo                   | Descrição                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `Intent`               | Enum: `general`, `local`, `code`, `system`, `graph`, `local_and_graph`, `system_and_local`, `clarify`       |
| `Complexity`           | Enum: `simple`, `normal`, `complex`, `deep`                                                                 |
| `ContextBlock`         | Dataclass frozen: `source`, `content`, `token_estimate`, `metadata`                                         |
| `RoutingResult`        | Dataclass: `intent`, `complexity`, `confidence`, `reason`, `method`                                         |
| `OrchestratorResult`   | Dataclass: `response`, `model_used`, `intent`, `complexity`, `sources_used`, `context_tokens`, `latency_ms` |
| `ContextProvider`      | Protocol `@runtime_checkable`: `name`, `get_context()`, `health()`                                          |
| `IntentClassifier`     | Protocol: `classify(query, history)`                                                                        |
| `ComplexityClassifier` | Protocol: `classify(query)`                                                                                 |
| `ModelRouter`          | Protocol: `select(intent, complexity)`                                                                      |
| `ContextRouter`        | Protocol: `route(intent, complexity)`                                                                       |

### `orchestrator/core/intent.py` — IntentClassifier

Heurística keyword-based com sets PT+EN:

- `_LOCAL_SIGNALS` — notas, vault, obsidian, apontamentos, ficheiros
- `_CODE_SIGNALS` — função, código, bug, refactora, implementa, debug
- `_SYSTEM_SIGNALS` + `_SYSTEM_PATTERNS` — RAM, GPU, disco, processos, uptime
- `_GRAPH_SIGNALS` + `_GRAPH_PATTERNS` — arquitectura, dependências, grafo, mapa
- `_SYSTEM_FALSE_POSITIVES` — filtra "system design", "machine learning"

Lógica de combinação: LOCAL + GRAPH → `LOCAL_AND_GRAPH`, SYSTEM + LOCAL → `SYSTEM_AND_LOCAL`.

### `orchestrator/core/complexity.py` — ComplexityClassifier

| Nível     | Critérios                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| `simple`  | ≤ 3 palavras                                                                                                  |
| `normal`  | 4–10 palavras sem sinais especiais                                                                            |
| `complex` | > 10 palavras, ou operadores booleanos (" e ", " ou "), ou geração de código                                  |
| `deep`    | Sinais de análise profunda ("analisa", "compara", "explica em detalhe"), múltiplas perguntas (?…?), debugging |

### `orchestrator/core/model_router.py` — ModelRouter

Tabela `(Intent, Complexity) → config_key`:

| Intent          | SIMPLE  | NORMAL  | COMPLEX | DEEP    |
| --------------- | ------- | ------- | ------- | ------- |
| general         | fast    | default | default | deep    |
| local           | default | default | default | deep    |
| code            | code    | code    | code    | code    |
| system          | fast    | default | default | default |
| graph           | default | default | deep    | deep    |
| local_and_graph | default | default | deep    | deep    |
| clarify         | fast    | fast    | fast    | fast    |

### `orchestrator/core/context_router.py` — ContextRouter

Tabela `Intent → [source names]`:

| Intent           | Providers activados |
| ---------------- | ------------------- |
| general          | — (sem contexto)    |
| local            | rag, cag            |
| code             | rag, graph, cag     |
| system           | system, cag         |
| graph            | graph, cag          |
| local_and_graph  | rag, graph, cag     |
| system_and_local | system, rag, cag    |
| clarify          | —                   |

### `orchestrator/factory.py` — create_engine()

Wire-up de todos os providers na ordem correcta. Ponto central de composição.

---

## 7. Context Providers

| Provider              | Fonte                                                                              | Activado por intent           | Health check    |
| --------------------- | ---------------------------------------------------------------------------------- | ----------------------------- | --------------- |
| `RAGContextProvider`  | HTTP `localhost:8484/query` + `/query/code`                                        | local, code, system_and_local | `GET /health`   |
| `CAGContextProvider`  | SQLite read-only (cag.db do obsidian-rag)                                          | todos excepto general/clarify | ficheiro existe |
| `SystemProbeProvider` | Subprocessos: `free`, `nvidia-smi`, `df`, `ps`, `uptime`, `uname`, `ip`, `sensors` | system                        | sempre true     |
| `RepoProbeProvider`   | `git status/log` nos repos configurados                                            | (via context_router "repo")   | sempre true     |
| `GraphProvider`       | `graph.json` + `community_summaries.json` do Graphify                              | graph, code, local_and_graph  | ficheiro existe |
| `ConfigEnvProvider`   | `orchestrator.toml` + `ollama list`                                                | (via context_router "config") | sempre true     |
| `LogsProvider`        | Ficheiros `.log` em directórios configurados                                       | (via context_router "logs")   | sempre true     |

**Circuit breaker** (RAGContextProvider): após 3 falhas consecutivas, RAG é ignorado por 60 segundos. Configurable via `orchestrator.toml [rag]`.

**Budget**: cada provider recebe `budget_tokens = remaining` (token budget global menos o já consumido). Se o provider ultrapassar o budget, é truncado ou ignorado.

---

## 8. Modelos e routing

### Modelos configurados (orchestrator.toml)

| Chave       | Modelo             | Uso típico                                       |
| ----------- | ------------------ | ------------------------------------------------ |
| `default`   | `qwen3:8b`         | Generalista, bom PT-PT, notas                    |
| `fast`      | `gemma3:4b`        | Perguntas simples, baixa latência                |
| `code`      | `qwen2.5-coder:7b` | Código, debugging, refactoring                   |
| `deep`      | `deepseek-r1:8b`   | Análise profunda, arquitectura, chain-of-thought |
| `embedding` | `bge-m3`           | (usado pelo obsidian-rag, referenciado aqui)     |

### Override de modelo

```bash
orc ask -m gemma3:4b "resposta rápida"
ai -m deep "analisa a arquitectura"
```

---

## 9. API REST

Servidor FastAPI em `localhost:8585` (arranca com `orc serve`).

| Endpoint    | Método | Body           | Descrição                                                 |
| ----------- | ------ | -------------- | --------------------------------------------------------- |
| `/query`    | POST   | `QueryRequest` | Query completa com routing. Suporta `stream: true` (SSE). |
| `/classify` | POST   | `QueryRequest` | Classificação sem LLM. Retorna `intent` + `complexity`.   |
| `/health`   | GET    | —              | Estado de Ollama, RAG e todos os providers.               |

### QueryRequest

```json
{
  "query": "string",
  "model": "string | null",
  "stream": false,
  "history": [{ "role": "user", "content": "..." }]
}
```

### QueryResponse

```json
{
  "response": "string",
  "model_used": "qwen3:8b",
  "intent": "local",
  "complexity": "normal",
  "sources_used": ["rag", "cag"],
  "context_tokens": 1240,
  "latency_ms": 1823.4
}
```

---

## 10. CLI — comando `orc`

Entry point: `orc` (instalado via `pip install -e .`)

```bash
orc ask "query..."                   # query completa
orc ask -m qwen3:8b "..."            # override modelo
orc ask --stream "..."               # streaming de tokens
orc ask --debug "..."                # mostra intent/model/sources/latency em stderr
orc ask --json-output "..."          # output JSON completo

orc classify "query..."              # classificação sem LLM

orc serve                            # inicia FastAPI em :8585
orc health                           # estado de todos os componentes
orc config                           # mostra configuração actual
```

---

## 11. Shell integration — função `ai`

Definida em `~/.zsh_custom.d/42-ai.zsh`. Fallback automático em 3 níveis:

1. **API** — `curl localhost:8585/query` (se servidor a correr)
2. **CLI** — `$HOME/ai-local/orchestrator/.venv/bin/orc ask` (se venv existe)
3. **Ollama directo** — `ol` (sempre disponível)

```bash
ai "quanta RAM tenho?"               # routing automático
ai --classify "debug esta função"    # mostra intent+complexity
ai --debug "o que é DNS?"            # resposta + routing decisions em stderr
ai --json "explica Docker"           # output JSON completo
ai --direct "olá"                    # bypass orquestrador
ai -m deep "analisa a arquitectura"  # override modelo
echo "código" | ai "revê isto"       # stdin como contexto adicional
```

---

## 12. Configuração

### orchestrator.toml

```toml
[orchestrator]
host = "127.0.0.1"
port = 8585

[rag]
url = "http://localhost:8484"
timeout = 30
circuit_breaker_threshold = 3
circuit_breaker_reset = 60

[ollama]
base_url = "http://localhost:11434"

[models]
default = "qwen3:8b"
fast = "gemma3:4b"
code = "qwen2.5-coder:7b"
deep = "deepseek-r1:8b"
embedding = "bge-m3"

[context]
token_budget = 6000

[context.cag]
db_path = "~/ai-local/obsidian-rag/data/qdrant/cag.db"

[repos]
paths = []

[graph]
output_dir = "~/ai-local/obsidian-rag/data/graphify"
cache_ttl = 300

[security]
allowed_commands = ["free", "nvidia-smi", "df", "ps", "uptime", "nproc", "uname", "ip", "sensors", "git", "ollama"]
max_command_timeout = 5

[logging]
level = "INFO"
format = "text"
```

### Variáveis de ambiente (prefixo `ORC_`)

```bash
ORC_ORCHESTRATOR_PORT=8585
ORC_RAG_URL=http://localhost:8484
ORC_MODELS_DEFAULT=qwen3:8b
ORC_CONTEXT_TOKEN_BUDGET=8000
```

---

## 13. Ficheiros externos editados

Ficheiros **fora** de `~/ai-local/orchestrator/` criados ou modificados por este projecto:

| Ficheiro                    | Tipo       | Descrição                                                                                                                          |
| --------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `~/.zsh_custom.d/42-ai.zsh` | Modificado | Nova função `ai()` com routing inteligente e fallback 3 níveis. Funções existentes (`ol`, `aicode`, etc.) mantidas sem alterações. |
| `~/ai-local/.env`           | Criado     | Ficheiro de secrets centralizado (API keys). Não versionado.                                                                       |
| `~/ai-local/.gitignore`     | Modificado | Regras para `.env`, `__pycache__/`, `.venv/`, `*.db`.                                                                              |

### Paths lidos (read-only)

| Path                                         | Provider             | Conteúdo                                 |
| -------------------------------------------- | -------------------- | ---------------------------------------- |
| `~/ai-local/obsidian-rag/data/graphify/`     | `GraphProvider`      | `graph.json`, `community_summaries.json` |
| `~/ai-local/obsidian-rag/data/qdrant/cag.db` | `CAGContextProvider` | SQLite com context packs                 |
| `http://localhost:8484`                      | `RAGContextProvider` | API do obsidian-rag                      |
| `http://localhost:11434`                     | `OllamaLLMClient`    | API do Ollama                            |

---

## 14. Testes

```bash
# Todos os testes
pytest tests/ -v

# Só unitários (sem serviços externos)
pytest tests/ -v -m "not integration"

# Só integração (requer Ollama)
pytest tests/ -v -m integration
```

| Ficheiro                 | Testes  | Tipo       |
| ------------------------ | ------- | ---------- |
| `test_intent.py`         | 19      | unitário   |
| `test_complexity.py`     | 11      | unitário   |
| `test_model_router.py`   | 7       | unitário   |
| `test_context_router.py` | 8       | unitário   |
| `test_engine.py`         | 10      | unitário   |
| `test_config.py`         | 5       | unitário   |
| `test_providers.py`      | 16      | unitário   |
| `test_api.py`            | 4       | unitário   |
| `test_cli.py`            | 4       | unitário   |
| `test_integration.py`    | 18      | integração |
| **Total**                | **102** |            |

---

## 15. Dependências

### Produção

| Pacote              | Versão mín. | Uso                           |
| ------------------- | ----------- | ----------------------------- |
| `httpx`             | ≥ 0.27      | HTTP para Ollama e RAG        |
| `fastapi`           | ≥ 0.110     | API REST                      |
| `uvicorn[standard]` | ≥ 0.29      | ASGI server                   |
| `psutil`            | ≥ 5.9       | Métricas de sistema (reserva) |

### Desenvolvimento

`pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`

### Stdlib utilizada

`tomllib`, `sqlite3`, `subprocess`, `pathlib`, `dataclasses`, `enum`, `typing`, `logging`, `time`, `json`, `re`

---

## 16. Como executar

```bash
# 1. Criar venv e instalar
cd ~/ai-local/orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Verificar configuração
orc config

# 3. Verificar estado dos serviços
orc health

# 4. Testar classificação (não requer LLM)
orc classify "quanta RAM tenho?"
# → Intent: system | Complexity: normal

# 5. Query directa (requer Ollama)
orc ask "o que é DNS?"

# 6. Arrancar servidor API
orc serve
# → FastAPI em http://127.0.0.1:8585

# 7. Carregar shell helpers
source ~/.zsh_custom.d/42-ai.zsh
ai "olá"
```

---

## 17. Estado actual

| Fase                  | Estado | Descrição                                                   |
| --------------------- | ------ | ----------------------------------------------------------- |
| 0 — Scaffolding       | ✅     | Estrutura, configs, pyproject.toml                          |
| 1 — Core              | ✅     | Intent, Complexity, ModelRouter, ContextRouter, Engine, LLM |
| 2 — Context Providers | ✅     | 7 providers implementados e testados                        |
| 3 — API + CLI         | ✅     | FastAPI :8585, `orc` CLI completo                           |
| 4 — Shell integration | ✅     | `ai` com fallback 3 níveis, `--debug`, `--classify`         |
| 5 — Hardening         | ✅     | Circuit breaker, testes integração, documentação            |

**Próximos passos candidatos:**

- LLM fallback para classificação de intent (queries ambíguas)
- Métricas de latência persistentes (`/metrics` endpoint)
- Agentes com tool calling (ReAct loop)
- Suporte a múltiplos backends LLM (vLLM, LM Studio)
- Context providers adicionais (calendário, e-mail local, RSS)
