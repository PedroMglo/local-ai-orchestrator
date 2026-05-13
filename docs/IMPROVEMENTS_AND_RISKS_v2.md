# ROADMAP v0.5 → v1.0 — ai-orchestrator

> **Versão:** v2 (roadmap de evolução)
> **Criado em:** 2026-05-13
> **Base:** Estado actual v0.4 — 157 testes, 79% coverage, 7 providers, ruff/mypy limpos
> **Objectivo:** Levar o orchestrator de ferramenta de routing determinístico a agente autónomo com multi-backend

---

## Índice

1. [Estado actual (v0.4)](#1-estado-actual-v04)
2. [v0.5 — Multi-backend LLM](#2-v05--multi-backend-llm)
3. [v0.6 — Tool registry e schemas](#3-v06--tool-registry-e-schemas)
4. [v0.7 — Engine agentic (ReAct loop)](#4-v07--engine-agentic-react-loop)
5. [v0.8 — Context providers adicionais](#5-v08--context-providers-adicionais)
6. [v0.9 — Hardening e observabilidade](#6-v09--hardening-e-observabilidade)
7. [v1.0 — Release estável](#7-v10--release-estável)
8. [Dependências entre fases](#8-dependências-entre-fases)
9. [Riscos e mitigações](#9-riscos-e-mitigações)

---

## 1. Estado actual (v0.4)

### Arquitectura

```
User → CLI / API → Engine → [classify → context → model → LLM] → Response
                              ↑                          ↑
                        7 providers                 OllamaLLMClient
                   (RAG, CAG, System,               (único backend)
                    Repo, Graph, Config, Logs)
```

### LLMClient Protocol (actual)

```python
class LLMClient(Protocol):
    def generate(self, prompt, model, *, temperature, max_tokens, timeout) -> str
    def chat(self, messages, model, *, temperature, max_tokens, timeout) -> str
    def chat_stream(self, messages, model, *, temperature, max_tokens, timeout) -> Iterator[str]
    def health(self) -> bool
```

### Métricas

| Métrica              | Valor                                    |
| -------------------- | ---------------------------------------- |
| Testes unitários     | 157                                      |
| Coverage             | 79%                                      |
| Providers            | 7                                        |
| Backends LLM         | 1 (Ollama)                               |
| Intents suportados   | 8                                        |
| Modelos configurados | 5 (default, fast, code, deep, embedding) |

---

## 2. v0.5 — Multi-backend LLM

> **Tema:** Suportar backends LLM alternativos além do Ollama
> **Esforço:** Médio | **Risco:** Baixo

### Objectivo

Permitir que o orchestrator use qualquer servidor OpenAI-compatible (vLLM, LM Studio, llama.cpp server, text-generation-inference) como alternativa ao Ollama, sem alterar o fluxo do Engine.

### Tarefas

| #   | Tarefa                                                                                                         | Ficheiros                         | Complexidade |
| --- | -------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------ |
| 1   | Refactoring do config: secção `[llm]` com lista de backends (`type`, `base_url`, `models`, `priority`)         | `config.py`, `orchestrator.toml`  | Média        |
| 2   | Implementar `OpenAICompatibleLLMClient` — `chat()`, `chat_stream()`, `health()` via API `/v1/chat/completions` | `llm/openai_compat.py` (novo)     | Média        |
| 3   | `LLMRouter` — selecciona backend por modelo (cada modelo mapeado a um backend) com fallback automático         | `llm/router.py` (novo)            | Média        |
| 4   | Integrar `LLMRouter` no `factory.py` — substituir instanciação directa do `OllamaLLMClient`                    | `factory.py`                      | Baixa        |
| 5   | Actualizar `Engine.health_report()` para reportar estado de cada backend                                       | `core/engine.py`                  | Baixa        |
| 6   | Actualizar CLI `orc health` e `orc config` para mostrar backends                                               | `cli/main.py`                     | Baixa        |
| 7   | Testes: mock OpenAI-compatible server, fallback entre backends, health reporting                               | `tests/test_llm_router.py` (novo) | Média        |

### Config proposta

```toml
[[llm.backends]]
name = "ollama"
type = "ollama"
base_url = "http://localhost:11434"
models = ["qwen3:8b", "gemma3:4b", "qwen2.5-coder:7b", "deepseek-r1:8b"]
priority = 1

[[llm.backends]]
name = "vllm"
type = "openai"
base_url = "http://localhost:8000"
api_key = ""
models = ["meta-llama/Llama-3.1-8B-Instruct"]
priority = 2
```

### Critérios de conclusão

- [ ] `orc health` reporta estado de cada backend individualmente
- [ ] `orc ask "olá" -m meta-llama/Llama-3.1-8B-Instruct` usa o backend vLLM
- [ ] Se backend primário falhar, fallback automático para o secundário
- [ ] Testes unitários para `OpenAICompatibleLLMClient` e `LLMRouter`
- [ ] Retrocompatibilidade: config sem `[[llm.backends]]` continua a funcionar (usa Ollama)

---

## 3. v0.6 — Tool registry e schemas

> **Tema:** Expor cada ContextProvider como ferramenta invocável pelo LLM
> **Esforço:** Médio | **Risco:** Baixo
> **Depende de:** v0.5 (opcional, mas recomendado para function calling multi-backend)

### Objectivo

Criar um registo de tools com nome, descrição e schema JSON que o LLM possa invocar. Cada provider torna-se uma tool. Isto é a fundação para o loop agentic do v0.7.

### Tarefas

| #   | Tarefa                                                                                                         | Ficheiros                      | Complexidade |
| --- | -------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------ |
| 1   | Definir `ToolDefinition` dataclass (`name`, `description`, `parameters: dict`, `callable`)                     | `core/tools.py` (novo)         | Baixa        |
| 2   | Método `as_tool()` no protocolo `ContextProvider` — retorna `ToolDefinition`                                   | `context/base.py`              | Baixa        |
| 3   | Implementar `as_tool()` em cada provider (7 providers) — schema JSON com parâmetros (`query`, `budget_tokens`) | `context/*.py`                 | Média        |
| 4   | `ToolRegistry` — regista e resolve tools por nome, expõe lista para o LLM                                      | `core/tools.py`                | Baixa        |
| 5   | Integrar `ToolRegistry` no `Engine` — popular a partir dos providers registados                                | `core/engine.py`, `factory.py` | Baixa        |
| 6   | CLI `orc tools` — lista todas as tools registadas com descrição e schema                                       | `cli/main.py`                  | Baixa        |
| 7   | API endpoint `GET /tools` — retorna lista de tools para clientes                                               | `api/app.py`, `api/schemas.py` | Baixa        |
| 8   | Testes: tool registry, as_tool() para cada provider, CLI e API                                                 | `tests/test_tools.py` (novo)   | Média        |

### Schema exemplo (RAG tool)

```json
{
  "name": "rag_search",
  "description": "Pesquisa semântica nas notas Obsidian e código indexado",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Texto a pesquisar" },
      "budget_tokens": { "type": "integer", "default": 2000 }
    },
    "required": ["query"]
  }
}
```

### Critérios de conclusão

- [ ] `orc tools` lista 7+ tools com nome e descrição
- [ ] `GET /tools` retorna JSON schema de cada tool
- [ ] Cada provider implementa `as_tool()` com schema válido
- [ ] Testes cobrem registry, resolução e chamada de cada tool

---

## 4. v0.7 — Engine agentic (ReAct loop)

> **Tema:** Loop ReAct que permite ao LLM invocar tools e iterar
> **Esforço:** Alto | **Risco:** Médio
> **Depende de:** v0.6 (tool registry obrigatório)

### Objectivo

`Engine.run_agentic()` implementa um loop Reason-Act onde o LLM pode:

1. Raciocinar sobre a query
2. Invocar uma tool (provider)
3. Receber o resultado
4. Decidir se precisa de mais contexto ou se pode responder

### Tarefas

| #   | Tarefa                                                                                           | Ficheiros                      | Complexidade |
| --- | ------------------------------------------------------------------------------------------------ | ------------------------------ | ------------ |
| 1   | System prompt agentic com instruções de tool calling (formato Ollama/OpenAI)                     | `prompts/agentic.py` (novo)    | Média        |
| 2   | Parser de tool calls do output do LLM (JSON extraction do `<tool_call>` ou function_call)        | `core/tool_parser.py` (novo)   | Média        |
| 3   | `Engine.run_agentic()` — loop com max_iterations (default 5), invoca tools, reinjecta resultados | `core/engine.py`               | Alta         |
| 4   | `Engine.stream_agentic()` — variante streaming com chunks intermediários                         | `core/engine.py`               | Alta         |
| 5   | Guard rails: max_iterations, max_tool_calls, timeout total, budget de tokens cumulativo          | `core/engine.py`               | Média        |
| 6   | CLI `orc ask --agentic` — activa o modo agentic                                                  | `cli/main.py`                  | Baixa        |
| 7   | API: campo `agentic: bool` no `QueryRequest`                                                     | `api/app.py`, `api/schemas.py` | Baixa        |
| 8   | Testes: mock LLM que retorna tool calls, verifica iteração, limites, fallback                    | `tests/test_agentic.py` (novo) | Alta         |

### Fluxo do loop agentic

```
1. Build messages (system + user query + tool definitions)
2. Call LLM
3. Parse response:
   a. Se contém tool_call → executar tool → adicionar resultado às messages → goto 2
   b. Se é resposta final → retornar
   c. Se max_iterations atingido → retornar resposta parcial com warning
4. Return OrchestratorResult com metadata (iterations, tools_used, total_tokens)
```

### Guard rails

| Limite                 | Default | Configurável        |
| ---------------------- | ------- | ------------------- |
| `max_iterations`       | 5       | `orchestrator.toml` |
| `max_tool_calls`       | 10      | `orchestrator.toml` |
| `agentic_timeout`      | 60s     | `orchestrator.toml` |
| `agentic_token_budget` | 12000   | `orchestrator.toml` |

### Critérios de conclusão

- [ ] `orc ask --agentic "que alterações fiz nos meus projectos esta semana?"` invoca RAG + Repo automaticamente
- [ ] LLM decide sozinho quais tools invocar com base na query
- [ ] Loop termina em ≤ 5 iterações para queries normais
- [ ] Fallback para `run()` se o modelo não suportar function calling
- [ ] Testes com mock LLM cobrem: 0 tools, 1 tool, múltiplas tools, max_iterations

---

## 5. v0.8 — Context providers adicionais

> **Tema:** Novos providers para calendário, RSS e e-mail local
> **Esforço:** Médio | **Risco:** Baixo
> **Depende de:** v0.6 (cada provider deve implementar `as_tool()`)

### Objectivo

Expandir o universo de contexto disponível ao LLM com fontes pessoais adicionais.

### Tarefas

| #   | Tarefa                                                                                       | Ficheiros                                                                    | Complexidade |
| --- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------ |
| 1   | `CalendarProvider` — lê `.ics` do Thunderbird/GNOME Calendar, filtra eventos ±7 dias         | `context/calendar.py` (novo)                                                 | Média        |
| 2   | `RSSProvider` — lê feeds OPML/URLs configurados, cache com TTL, últimas N entradas           | `context/rss.py` (novo)                                                      | Média        |
| 3   | `EmailProvider` — lê Maildir/mbox local (read-only), filtra por data e query                 | `context/email.py` (novo)                                                    | Média        |
| 4   | Registar novos providers em `factory.py` (opt-in via config)                                 | `factory.py`, `config.py`                                                    | Baixa        |
| 5   | Config: secções `[calendar]`, `[rss]`, `[email]` em `orchestrator.toml`                      | `config.py`, `orchestrator.toml`                                             | Baixa        |
| 6   | `as_tool()` para cada novo provider                                                          | `context/calendar.py`, `context/rss.py`, `context/email.py`                  | Baixa        |
| 7   | Testes unitários com fixtures para cada provider                                             | `tests/test_calendar.py`, `tests/test_rss.py`, `tests/test_email.py` (novos) | Média        |
| 8   | Intent routing: actualizar `_ROUTE_TABLE` para incluir novos providers em intents relevantes | `core/context_router.py`                                                     | Baixa        |

### Config proposta

```toml
[calendar]
enabled = false
ics_path = "~/.local/share/gnome-calendar/local.calendar"
window_days = 7

[rss]
enabled = false
opml_path = ""
feeds = []
cache_ttl = 3600
max_entries = 20

[email]
enabled = false
maildir_path = "~/Mail"
max_age_days = 30
```

### Critérios de conclusão

- [ ] `orc ask "que eventos tenho esta semana?"` retorna dados do calendário (se configurado)
- [ ] `orc ask "últimas notícias dos meus feeds"` retorna entradas RSS
- [ ] `orc health` reporta estado dos novos providers
- [ ] Todos os providers são opt-in (disabled por defeito)
- [ ] Testes passam sem dependências externas (fixtures/mocks)

---

## 6. v0.9 — Hardening e observabilidade

> **Tema:** Estabilização, observabilidade e preparação para release
> **Esforço:** Médio | **Risco:** Baixo
> **Depende de:** v0.7 + v0.8

### Objectivo

Tornar o sistema production-ready (local) com logging estruturado, métricas detalhadas do loop agentic, e cobertura de testes ≥ 85%.

### Tarefas

| #   | Tarefa                                                                                     | Ficheiros                   | Complexidade |
| --- | ------------------------------------------------------------------------------------------ | --------------------------- | ------------ |
| 1   | Logging estruturado (JSON) — opt-in via `logging.format = "json"`                          | `config.py`, logger setup   | Média        |
| 2   | Métricas agentic: `iterations_per_query`, `tools_invoked`, `agentic_vs_direct` ratio       | `core/metrics.py`           | Média        |
| 3   | Endpoint `/metrics` expandido com dados agentic e per-backend                              | `api/app.py`                | Baixa        |
| 4   | `orc doctor` — diagnóstico completo (config, backends, providers, modelos, disco, versões) | `cli/main.py`               | Média        |
| 5   | Coverage ≥ 85% — identificar e cobrir gaps                                                 | `tests/`                    | Média        |
| 6   | Testes de integração automatizados (com Ollama real, marcados `@pytest.mark.integration`)  | `tests/test_integration.py` | Média        |
| 7   | Documentação: `PROJECT_OVERVIEW.md` v2, guia de configuração multi-backend                 | `docs/`                     | Baixa        |
| 8   | Validação de `orchestrator.toml` no startup — erro claro se config inválida                | `config.py`                 | Baixa        |

### Critérios de conclusão

- [ ] `orc doctor` reporta diagnóstico completo num único comando
- [ ] `/metrics` inclui dados agentic (quando usado)
- [ ] Coverage ≥ 85%
- [ ] Logging JSON funcional com `logging.format = "json"`
- [ ] Config inválida produz mensagem de erro clara (não stack trace)

---

## 7. v1.0 — Release estável

> **Tema:** Consolidação, polish e release oficial
> **Esforço:** Baixo | **Risco:** Baixo
> **Depende de:** v0.9

### Objectivo

Marcar o projecto como feature-complete para uso pessoal avançado com documentação final.

### Tarefas

| #   | Tarefa                                                              | Ficheiros                        | Complexidade |
| --- | ------------------------------------------------------------------- | -------------------------------- | ------------ |
| 1   | Bump versão para `1.0.0` em `api/app.py` e `pyproject.toml`         | `api/app.py`, `pyproject.toml`   | Baixa        |
| 2   | `CHANGELOG.md` — resumo de todas as versões v0.1 → v1.0             | `docs/CHANGELOG.md` (novo)       | Baixa        |
| 3   | README actualizado com arquitectura final, exemplos de uso, config  | `README.md`                      | Média        |
| 4   | `PROJECT_OVERVIEW.md` final — arquitectura v1.0, diagrama, decisões | `docs/PROJECT_OVERVIEW.md`       | Média        |
| 5   | Fechar todos os items em `IMPROVEMENTS_AND_RISKS.md`                | `docs/IMPROVEMENTS_AND_RISKS.md` | Baixa        |
| 6   | Tag git `v1.0.0` e GitHub Release                                   | —                                | Baixa        |

### Critérios de conclusão

- [ ] Todos os testes passam (unitários + integração)
- [ ] Coverage ≥ 85%
- [ ] `orc --version` retorna `1.0.0`
- [ ] Documentação completa e actualizada
- [ ] Sem items abertos no `IMPROVEMENTS_AND_RISKS.md` (excluindo nice-to-have)

---

## 8. Dependências entre fases

```
v0.4 (actual) ─── concluído
  │
  ├── v0.5 Multi-backend LLM
  │     │
  │     └── v0.6 Tool registry ──── v0.7 ReAct loop
  │                                       │
  ├── v0.8 Novos providers ──────────────┘
  │                                       │
  └─────────────────── v0.9 Hardening ────┘
                            │
                         v1.0 Release
```

**Notas:**

- v0.5 e v0.8 podem ser paralelizados (independentes)
- v0.6 depende conceptualmente de v0.5 (function calling precisa de backends que o suportem)
- v0.7 depende obrigatoriamente de v0.6 (precisa do tool registry)
- v0.9 é consolidação — depende de tudo anterior
- v1.0 é polish — depende de v0.9

---

## 9. Riscos e mitigações

| Risco                                                        | Impacto | Probabilidade | Mitigação                                                                                          |
| ------------------------------------------------------------ | ------- | ------------- | -------------------------------------------------------------------------------------------------- |
| Modelos locais não suportam function calling fiável          | Alto    | Média         | Parser robusto com fallback para regex; suporte a `<tool_call>` e JSON inline; modo agentic opt-in |
| vLLM/LM Studio API incompatível com standard OpenAI          | Médio   | Baixa         | Testar com endpoints reais; adaptar headers/campos por backend                                     |
| Providers adicionais (email, calendar) com formatos variados | Baixo   | Média         | Suportar apenas formatos comuns (ICS, Maildir); documentar limitações                              |
| Loop agentic infinito ou lento                               | Alto    | Baixa         | Guard rails rígidos (max_iterations=5, timeout=60s, token budget cumulativo)                       |
| Complexidade do sistema aumenta significativamente           | Médio   | Média         | Manter separação clara: cada fase é um PR auto-contido; testes obrigatórios                        |
| Cold start com múltiplos backends                            | Baixo   | Baixa         | Health check lazy; backends inicializados apenas quando invocados                                  |

---

> **Última actualização:** 2026-05-13
