# IMPROVEMENTS AND RISKS — ai-orchestrator

> **Versão:** 0.1.0
> **Última atualização:** 2026-05-12
> **Âmbito:** Análise crítica de limitações, riscos, dívida técnica e roadmap

---

## Índice

1. [Limitações da arquitectura actual](#1-limitações-da-arquitectura-actual)
2. [Problemas técnicos conhecidos](#2-problemas-técnicos-conhecidos)
3. [Dívida técnica](#3-dívida-técnica)
4. [Riscos operacionais](#4-riscos-operacionais)
5. [Segurança](#5-segurança)
6. [Performance](#6-performance)
7. [Melhorias recomendadas](#7-melhorias-recomendadas)
8. [Roadmap sugerido](#8-roadmap-sugerido)

---

## 1. Limitações da arquitectura actual

### 1.1 Classificação de intent apenas heurística

| Campo                  | Detalhe                                                        |
| ---------------------- | -------------------------------------------------------------- |
| **Prioridade**         | Média                                                          |
| **Impacto**            | Médio — queries ambíguas ou mistas podem ser mal classificadas |
| **Complexidade**       | Média                                                          |
| **Ficheiros afetados** | `orchestrator/core/intent.py`                                  |

O `HeuristicIntentClassifier` usa sets de keywords PT+EN. Funciona bem para queries directas mas falha em:

- Perguntas em inglês fora do vocabulário configurado
- Queries onde o contexto de conversa anterior altera o intent
- Seguimento de conversa ("e quanto custa?" após query sobre GPU)

**Solução proposta:** Adicionar LLM fallback (gemma3:4b com timeout 3s) quando a heurística retorna `GENERAL` e a query tem > 5 palavras — idêntico ao padrão já existente no `obsidian-rag/retrieval/router.py`.

---

### 1.2 Pipeline sequencial sem paralelismo nos providers

| Campo                  | Detalhe                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| **Prioridade**         | Baixa                                                                  |
| **Impacto**            | Médio — latência aumenta linearmente com o número de providers activos |
| **Complexidade**       | Baixa                                                                  |
| **Ficheiros afetados** | `orchestrator/core/engine.py` — `_gather_context()`                    |

Os context providers são chamados sequencialmente. Para intents como `LOCAL_AND_GRAPH` (rag + graph + cag), os pedidos HTTP ao RAG e a leitura do graph.json poderiam ser paralelos.

**Solução proposta:** `asyncio.gather()` ou `ThreadPoolExecutor` em `_gather_context()`, com timeout por provider.

---

### 1.3 Sem pipeline agentic (ReAct / tool calling)

| Campo                  | Detalhe                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| **Prioridade**         | Baixa (v0.2 candidato)                                                 |
| **Impacto**            | Limitação de capacidade — o LLM não pode pedir mais contexto ou iterar |
| **Complexidade**       | Alta                                                                   |
| **Ficheiros afetados** | `orchestrator/core/engine.py`                                          |

O fluxo actual é linear e determinístico: classify → context → model → response. O LLM não tem capacidade de pedir ferramentas, fazer follow-up queries ao RAG, ou iterar com raciocínio próprio.

**Solução proposta:** Loop agentic opcional em `Engine.run_agentic()` com tool definitions para cada provider, usando o Ollama function calling (disponível em qwen2.5+ e qwen3).

---

### 1.4 Sem histórico de conversa persistente

| Campo                  | Detalhe                                                  |
| ---------------------- | -------------------------------------------------------- |
| **Prioridade**         | Média                                                    |
| **Impacto**            | Médio — cada query é stateless; sem memória de sessão    |
| **Complexidade**       | Baixa                                                    |
| **Ficheiros afetados** | `orchestrator/api/app.py`, `orchestrator/api/schemas.py` |

O campo `history` existe na API mas não é persistido — cabe ao cliente enviar o histórico a cada request. Não há conceito de sessão no servidor.

**Solução proposta:** Session store opcional em SQLite com TTL (similar ao CAG), com `session_id` na `QueryRequest`.

---

## 2. Problemas técnicos conhecidos

### 2.1 Estimativa de tokens por divisão de caracteres

| Campo                  | Detalhe                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| **Prioridade**         | Baixa                                                                |
| **Impacto**            | Baixo — pode sub/sobreestimar budget; raro causar problemas práticos |
| **Ficheiros afetados** | `orchestrator/context/base.py`, todos os providers                   |

`token_estimate = len(content) // 4` é uma heurística grosseira. Para texto PT-PT com acentos e código com símbolos, o ratio real pode ser 3–5 chars/token.

**Solução proposta:** Adoptar a estimativa do `obsidian-rag` (`re.findall(r'\b\w+\b', text)` × 1.3) ou usar `tiktoken` para modelos compatíveis.

---

### 2.2 GraphProvider sem invalidação por mtime no primeiro load

| Campo                  | Detalhe                                                    |
| ---------------------- | ---------------------------------------------------------- |
| **Prioridade**         | Baixa                                                      |
| **Impacto**            | Baixo — pode servir graph.json stale após sync do Graphify |
| **Ficheiros afetados** | `orchestrator/context/graph.py`                            |

O cache em memória verifica `mtime` apenas nas revalidações. No primeiro load não há mtime baseline — se o ficheiro for actualizado antes do `cache_ttl` expirar, a mudança não é detectada imediatamente.

**Solução proposta:** Guardar `mtime` no load inicial e comparar sempre antes de servir do cache.

---

### 2.3 SystemProbeProvider sem timeout agregado

| Campo                  | Detalhe                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| **Prioridade**         | Baixa                                                               |
| **Impacto**            | Baixo — casos extremos onde vários comandos demoram perto do máximo |
| **Ficheiros afetados** | `orchestrator/context/system.py`                                    |

Cada comando tem `timeout = security.max_command_timeout` (5s). Com todos os subsistemas activos (memory + gpu + disk + cpu + processes + system + network + temperature), o pior caso é 8 × 5s = 40s antes de retornar.

**Solução proposta:** Timeout global para `_gather_context()` no engine, ou `ThreadPoolExecutor` com deadline por provider.

---

### 2.4 CLI `orc health` acede a atributos privados do engine

| Campo                  | Detalhe                                               |
| ---------------------- | ----------------------------------------------------- |
| **Prioridade**         | Baixa                                                 |
| **Impacto**            | Baixo — frágil a refactorings internos do Engine      |
| **Ficheiros afetados** | `orchestrator/cli/main.py`, `orchestrator/api/app.py` |

`engine._providers` e `engine._llm` são usados directamente em `health()`. Quebra encapsulamento.

**Solução proposta:** Método `Engine.health_report() -> dict[str, bool]` público.

---

## 3. Dívida técnica

### 3.1 Sem coverage report configurado

Os 102 testes passam mas não há relatório de coverage activo. O `pyproject.toml` tem `pytest-cov` como dep de dev mas sem configuração `[tool.coverage]`.

**Acção:** Adicionar `addopts = "--cov=orchestrator --cov-report=term-missing"` ao `[tool.pytest.ini_options]`.

---

### 3.2 Sem linting/formatting automático (CI)

`ruff` e `mypy` estão nas deps de dev mas não há CI configurado (GitHub Actions) nem pre-commit hooks.

**Acção:** Adicionar `.github/workflows/test.yml` com `ruff check`, `mypy`, `pytest -m "not integration"`.

---

### 3.3 `ConfigEnvProvider` chama `ollama list` sem whitelist

`ollama` está na `security.allowed_commands` mas a chamada em `config_env.py` usa `subprocess.run` directamente sem passar pelo `_safe_run()` do `system.py`.

**Acção:** Extrair `_safe_run()` para `orchestrator/core/security.py` e reutilizar em todos os providers.

---

### 3.4 `orc ask --stream` não mostra debug info

O modo streaming no CLI não suporta `--debug` (seria necessário separar o canal de debug do stream de tokens).

**Acção:** Emitir debug para stderr antes de iniciar o stream (já disponível para o modo não-stream).

---

## 4. Riscos operacionais

### 4.1 Ollama como single point of failure

Se o Ollama parar, todas as queries falham — incluindo as de intent `GENERAL` que não precisam de contexto local. Não há fallback de LLM.

**Mitigação actual:** `OllamaLLMClient.health()` retorna `False` e o `orc health` reporta. A função `ai` no shell faz fallback para `ol` (que também usa Ollama).

**Mitigação futura:** Suporte a backends alternativos (vLLM, LM Studio, OpenAI-compatible API local).

---

### 4.2 CAG database partilhada com obsidian-rag

O `CAGContextProvider` lê o SQLite do `obsidian-rag` em modo read-only (`?mode=ro`). Se o obsidian-rag estiver a fazer vacuum, migrate ou rebuild da base, o lock pode causar timeout.

**Mitigação:** Timeout de 5s no connect SQLite. Retorna `None` graciosamente se falhar.

---

### 4.3 Sem autenticação na API

`POST /query` em `localhost:8585` não tem autenticação. Qualquer processo local pode fazer queries — incluindo tools/agentes maliciosos.

**Mitigação actual:** Bind apenas em `127.0.0.1` (não exposto na rede).

**Mitigação futura:** API key via header `X-API-Key` (placeholder `ORC_API_KEY` já existe em `~/ai-local/.env`).

---

### 4.4 Sem rate limiting

Queries múltiplas simultâneas sobrecarregam o Ollama (que processa 1 request de cada vez em GPU). Sem queue ou rate limiting, os requests ficam em fila no servidor sem feedback ao cliente.

**Mitigação futura:** `asyncio.Semaphore` com max_concurrent_llm_calls configurable.

---

## 5. Segurança

### 5.1 Whitelist de comandos do sistema

O `SystemProbeProvider` e `RepoProbeProvider` só executam comandos que estão em `security.allowed_commands` no `orchestrator.toml`. Por defeito: `free`, `nvidia-smi`, `df`, `ps`, `uptime`, `nproc`, `uname`, `ip`, `sensors`, `git`, `ollama`.

**Sem sudo** — nenhum comando com escalação de privilégios. **Timeout máximo** de 5s por comando.

### 5.2 Secrets

- `~/ai-local/.env` — não versionado (`.gitignore`)
- Nunca em `orchestrator.toml` (versionado)
- `ORC_API_KEY` reservado para autenticação futura

### 5.3 Injecção via context

Os context blocks injectados no prompt LLM vêm de fontes locais controladas (SQLite, ficheiros, git, subprocessos whitelisted, HTTP local). Risco de prompt injection via conteúdo de notas Obsidian ou output de git — baixo mas não nulo.

**Mitigação:** Os context blocks são envolvidos em tags `[SOURCE]...[/SOURCE]` para delimitar claramente o contexto do prompt do utilizador.

---

## 6. Performance

### Latência esperada (sem cold start de modelo)

| Cenário                                 | Latência típica |
| --------------------------------------- | --------------- |
| Classificação apenas (`orc classify`)   | < 10ms          |
| Intent GENERAL, gemma3:4b               | 1–3s            |
| Intent LOCAL com RAG, qwen3:8b          | 3–8s            |
| Intent CODE com graph, qwen2.5-coder:7b | 4–10s           |
| Intent DEEP, deepseek-r1:8b             | 10–30s          |

### Cold start (modelo não carregado em VRAM)

Adicionar 5–15s ao primeiro request após idle (OLLAMA_KEEP_ALIVE expirado). Mitigado com `aiwarm` no shell.

### Uso de recursos do orquestrador

- **RAM**: ~50MB idle (FastAPI + providers inicializados)
- **CPU**: negligível fora de requests
- **Disco**: sem escrita (read-only nos providers)

---

## 7. Melhorias recomendadas

### Prioridade Alta

| Melhoria                                | Ficheiros                                     | Esforço     |
| --------------------------------------- | --------------------------------------------- | ----------- |
| LLM fallback na classificação de intent | `core/intent.py`                              | Médio       |
| `Engine.health_report()` público        | `core/engine.py`, `cli/main.py`, `api/app.py` | Baixo       |
| Coverage report no pytest               | `pyproject.toml`                              | Muito baixo |
| `_safe_run()` centralizado              | novo `core/security.py`                       | Baixo       |

### Prioridade Média

| Melhoria                           | Ficheiros                      | Esforço |
| ---------------------------------- | ------------------------------ | ------- |
| Paralelismo em `_gather_context()` | `core/engine.py`               | Médio   |
| Session store opcional             | `api/app.py`, `api/schemas.py` | Médio   |
| API key authentication             | `api/app.py`                   | Baixo   |
| Timeout global por provider        | `core/engine.py`               | Baixo   |

### Prioridade Baixa

| Melhoria                       | Ficheiros            | Esforço |
| ------------------------------ | -------------------- | ------- |
| Estimativa de tokens melhorada | todos os providers   | Baixo   |
| GitHub Actions CI              | `.github/workflows/` | Baixo   |
| Rate limiting LLM              | `api/app.py`         | Médio   |
| Métricas `/metrics` endpoint   | `api/app.py`         | Médio   |

---

## 8. Roadmap sugerido

### v0.2 — Robustez

- [ ] LLM fallback para classificação de intent ambígua
- [ ] `Engine.health_report()` como interface pública
- [ ] `_safe_run()` centralizado em `core/security.py`
- [ ] Coverage report activo (`--cov=orchestrator`)
- [ ] GitHub Actions: `ruff` + `mypy` + `pytest -m "not integration"`

### v0.3 — Performance

- [ ] `_gather_context()` paralelo com `ThreadPoolExecutor`
- [ ] Timeout global por provider (configurável)
- [ ] Rate limiting para requests LLM concorrentes

### v0.4 — Funcionalidades

- [ ] Session store SQLite com TTL (`session_id` na API)
- [ ] API key authentication (`X-API-Key`)
- [ ] `orc ask --stream --debug` (debug em stderr antes do stream)
- [ ] Endpoint `/metrics` com latências históricas

### v1.0 — Agentes

- [ ] `Engine.run_agentic()` com ReAct loop
- [ ] Tool definitions para cada ContextProvider
- [ ] Suporte a backends LLM alternativos (OpenAI-compatible)
- [ ] Context providers adicionais (calendário, RSS, e-mail local)
