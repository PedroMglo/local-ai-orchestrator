# IMPROVEMENTS AND RISKS — ai-orchestrator

> **Versão:** 0.4.0
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

### 1.1 Classificação de intent apenas heurística ✅ _Resolvido em v0.2_

| Campo                  | Detalhe                                                        |
| ---------------------- | -------------------------------------------------------------- |
| **Prioridade**         | Média                                                          |
| **Impacto**            | Médio — queries ambíguas ou mistas podem ser mal classificadas |
| **Complexidade**       | Média                                                          |
| **Ficheiros afetados** | `orchestrator/core/engine.py`, `orchestrator/core/intent.py`   |

O `HeuristicIntentClassifier` usa sets de keywords PT+EN. Funciona bem para queries directas mas falha em:

- Perguntas em inglês fora do vocabulário configurado
- Queries onde o contexto de conversa anterior altera o intent
- Seguimento de conversa ("e quanto custa?" após query sobre GPU)

**Implementado (v0.2):** `Engine._llm_intent_fallback()` chama o modelo `fast` (gemma3:4b) com timeout de 3s quando a heurística retorna `GENERAL` e a query tem > 5 palavras. O fallback é encapsulado em `Engine._classify_intent()` — o `HeuristicIntentClassifier` mantém-se puro e sem dependências de I/O. Parse via regex sobre os valores do enum `Intent`; qualquer erro retorna graciosamente `GENERAL`.

---

### 1.2 Pipeline sequencial sem paralelismo nos providers ✅ _Resolvido em v0.3_

| Campo                  | Detalhe                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| **Prioridade**         | Baixa                                                                    |
| **Impacto**            | Médio — latência aumentava linearmente com o número de providers activos |
| **Complexidade**       | Baixa                                                                    |
| **Ficheiros afetados** | `orchestrator/core/engine.py` — `_gather_context()`                      |

Os context providers eram chamados sequencialmente. Para intents como `LOCAL_AND_GRAPH` (rag + graph + cag), os pedidos HTTP ao RAG e a leitura do graph.json podiam ser paralelos.

**Implementado (v0.3):** `_gather_context()` usa `ThreadPoolExecutor(max_workers=len(active))` com `future.result(timeout=cfg.context.provider_timeout)` individual por provider. Resultados são reordenados pela prioridade original de `sources` e truncados ao budget. Providers lentos ou falhos são ignorados graciosamente. Latência do gather é logged.

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

### 1.4 Sem histórico de conversa persistente ✅ _Resolvido em v0.4_

| Campo                  | Detalhe                                                  |
| ---------------------- | -------------------------------------------------------- |
| **Prioridade**         | Média                                                    |
| **Impacto**            | Médio — cada query era stateless; sem memória de sessão  |
| **Complexidade**       | Baixa                                                    |
| **Ficheiros afetados** | `orchestrator/api/app.py`, `orchestrator/api/schemas.py` |

O campo `history` existia na API mas não era persistido — cabia ao cliente enviar o histórico a cada request. Não havia conceito de sessão no servidor.

**Implementado (v0.4):** Novo módulo `orchestrator/core/session.py` com `SessionStore` — SQLite em modo WAL, tabela `sessions`, TTL configurável (`session.ttl_seconds`, default 3600). O `session_id` foi adicionado a `QueryRequest` e `QueryResponse`. Quando a feature está activa (`session.enabled = true`), a API gera UUID para novas sessões, carrega histórico de sessões existentes, e persiste user/assistant messages após cada query. Limite de 20 mensagens por sessão (`session.max_messages`). Cleanup de sessões expiradas no startup.

---

## 2. Problemas técnicos conhecidos

### 2.1 Estimativa de tokens por divisão de caracteres ✅ _Resolvido em v0.4_

| Campo                  | Detalhe                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| **Prioridade**         | Baixa                                                                |
| **Impacto**            | Baixo — pode sub/sobreestimar budget; raro causar problemas práticos |
| **Ficheiros afetados** | `orchestrator/context/base.py`, todos os providers                   |

`token_estimate = len(content) // 4` era uma heurística grosseira. Para texto PT-PT com acentos e código com símbolos, o ratio real pode ser 3–5 chars/token.

**Implementado (v0.4):** Nova função `estimate_tokens(text)` em `context/base.py` usando `re.findall(r'\b\w+\b', text) * 1.3` — mais precisa para texto misto PT-PT/EN e código. Adoptada em todos os providers (`cag`, `config`, `rag`, `repo`, `system`, `graph`, `logs`) e no `Engine._estimate_tokens()`. Retorna mínimo 1 para texto não-vazio, 0 para texto vazio.

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

### 2.3 SystemProbeProvider sem timeout agregado ✅ _Mitigado em v0.3_

| Campo                  | Detalhe                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| **Prioridade**         | Baixa                                                               |
| **Impacto**            | Baixo — casos extremos onde vários comandos demoram perto do máximo |
| **Ficheiros afetados** | `orchestrator/context/system.py`                                    |

Cada comando tem `timeout = security.max_command_timeout` (5s). Com todos os subsistemas activos (memory + gpu + disk + cpu + processes + system + network + temperature), o pior caso era 8 × 5s = 40s antes de retornar.

**Mitigado (v0.3):** `_gather_context()` agora executa cada provider com `provider_timeout` individual (default 10s) via `ThreadPoolExecutor`. O `SystemProbeProvider` como um todo está limitado a 10s (configurável via `context.provider_timeout` em `orchestrator.toml`), independentemente de quantos subcomandos execute internamente.

---

### 2.4 CLI `orc health` acede a atributos privados do engine ✅ _Resolvido em v0.2_

| Campo                  | Detalhe                                               |
| ---------------------- | ----------------------------------------------------- |
| **Prioridade**         | Baixa                                                 |
| **Impacto**            | Baixo — frágil a refactorings internos do Engine      |
| **Ficheiros afetados** | `orchestrator/cli/main.py`, `orchestrator/api/app.py` |

`engine._providers` e `engine._llm` eram usados directamente em `health()`. Quebrava encapsulamento.

**Implementado (v0.2):** `Engine.health_report() -> dict[str, Any]` encapsula o acesso a `self._llm.health()` e `self._providers`. Retorna `{"ollama": bool, "providers": dict[str, bool], "all_ok": bool}`. CLI e API actualizados para usar este método exclusivamente.

---

## 3. Dívida técnica

### 3.1 Sem coverage report configurado ✅ _Resolvido em v0.2_

Os testes passavam mas não havia relatório de coverage activo. O `pyproject.toml` tinha `pytest-cov` como dep de dev mas sem configuração `[tool.coverage]`.

**Implementado (v0.2):** `addopts = "--cov=orchestrator --cov-report=term-missing"` em `[tool.pytest.ini_options]` e secção `[tool.coverage.run]` com `source = ["orchestrator"]`. Coverage actual: **71%** (88 testes unitários).

---

### 3.2 Sem linting/formatting automático (CI) ✅ _Resolvido em v0.2_

`ruff` e `mypy` estavam nas deps de dev mas não havia CI configurado.

**Implementado (v0.2):** `.github/workflows/orchestrator-ci.yml` com 3 jobs: `lint` (`ruff check` + `mypy`) → `test` (py3.11 + py3.12, `pytest -m "not integration" --cov-fail-under=30`) + `cli-smoke` (paralelo após lint). Path filter `orchestrator/**` evita trigger em mudanças do `obsidian-rag`.

---

### 3.3 `ConfigEnvProvider` chama `ollama list` sem whitelist ✅ _Resolvido em v0.2_

`ollama` estava na `security.allowed_commands` mas `config_env.py` chamava `subprocess.run` directamente, sem passar pela whitelist.

**Implementado (v0.2):** Novo módulo `orchestrator/core/security.py` com função pública `safe_run(cmd, *, timeout)` que: verifica `allowed_commands`, respeita `max_command_timeout`, usa `shell=False` explícito, captura `TimeoutExpired`, `FileNotFoundError` e `OSError`. Adoptado em `context/system.py` (removida `_safe_run()` local) e `context/config_env.py`. `context/repo.py` mantém `subprocess.run` com `cwd=` (necessário para git), mas verifica whitelist inline.

---

### 3.4 `orc ask --stream` não mostra debug info ✅ _Resolvido em v0.4_

O modo streaming no CLI não suportava `--debug` (era necessário separar o canal de debug do stream de tokens).

**Implementado (v0.4):** Quando `--stream --debug` são usados em conjunto, o CLI chama `engine.classify()` primeiro para obter routing decisions (intent, complexity), determina o modelo e sources, emite o bloco de debug para stderr, e só depois inicia o stream para stdout. Latência e context_tokens não são mostrados (indisponíveis antes do stream completar).

---

## 4. Riscos operacionais

### 4.1 Ollama como single point of failure ✅ _Mitigado em v0.4_

Se o Ollama parar, todas as queries falham — incluindo as de intent `GENERAL` que não precisam de contexto local. Não há fallback de LLM.

**Mitigação anterior:** `OllamaLLMClient.health()` retorna `False` e o `orc health` reporta. A função `ai` no shell faz fallback para `ol` (que também usa Ollama).

**Implementado (v0.4):** Health-gate com cache de 5s (`_is_llm_available()`) em `Engine` — se o Ollama estiver unreachable, retorna resposta degradada imediatamente sem tentar o request. `Engine.run()` e `Engine.stream()` protegidos com `try/except` para `httpx.ConnectError`, `httpx.TimeoutException` e `httpx.HTTPStatusError`. Em caso de falha, retorna `OrchestratorResult` com mensagem de aviso ao utilizador e invalida o cache de health para fast-fail nos requests seguintes. 6 testes de resiliência adicionados em `tests/test_engine.py`.

**Risco residual:** Sem backends LLM alternativos — se o Ollama estiver down, todas as queries retornam mensagem degradada. Suporte multi-backend planeado para v1.0.

---

### 4.2 CAG database partilhada com obsidian-rag ✅ _Mitigado em v0.4_

O `CAGContextProvider` lê o SQLite do `obsidian-rag` em modo read-only (`?mode=ro`). Se o obsidian-rag estiver a fazer vacuum, migrate ou rebuild da base, o lock pode causar timeout.

**Mitigação anterior:** Timeout de 5s no connect SQLite. Retorna `None` graciosamente se falhar.

**Implementado (v0.4):** `health()` agora executa `SELECT 1 FROM packs LIMIT 1` com timeout de 2s em vez de apenas verificar existência do ficheiro — detecta corrupção e locks. `get_context()` delega a query a `_query_packs()` que implementa retry (1x) com 1s de espera em caso de `sqlite3.OperationalError` com mensagem "locked". Erros não-locked falham imediatamente.

---

### 4.3 Sem autenticação na API ✅ _Resolvido em v0.4_

`POST /query` em `localhost:8585` não tinha autenticação. Qualquer processo local podia fazer queries — incluindo tools/agentes maliciosos.

**Mitigação anterior:** Bind apenas em `127.0.0.1` (não exposto na rede).

**Implementado (v0.4):** Middleware HTTP em `api/app.py` que valida API key via header `X-API-Key` ou `Authorization: Bearer <key>`. Configurável em `orchestrator.toml` (`orchestrator.api_key`) ou env var `ORC_ORCHESTRATOR_API_KEY`. Quando vazia (default), a autenticação está desactivada. Comparação timing-safe com `secrets.compare_digest`. Paths isentos: `/health`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`.

---

### 4.4 Sem rate limiting ✅ _Resolvido em v0.3_

Queries múltiplas simultâneas sobrecarregavam o Ollama (que processa 1 request de cada vez em GPU). Sem queue ou rate limiting, os requests ficavam em fila no servidor sem feedback ao cliente.

**Implementado (v0.3):** `asyncio.Semaphore(cfg.ollama.max_concurrent_llm)` na API — default 1 slot. O endpoint `/query` adquire o semaphore antes de chamar `engine.run()` ou `engine.stream()`. Se todas as slots estão ocupadas, retorna HTTP 429 com `Retry-After: 5`. Endpoints `/classify` e `/health` não são afectados.

---

## 5. Segurança

### 5.1 Whitelist de comandos do sistema ✅ _Implementado em v0.1, reforçado em v0.2_

O `SystemProbeProvider` e `RepoProbeProvider` só executam comandos que estão em `security.allowed_commands` no `orchestrator.toml`. Por defeito: `free`, `nvidia-smi`, `df`, `ps`, `uptime`, `nproc`, `uname`, `ip`, `sensors`, `git`, `ollama`.

**Sem sudo** — nenhum comando com escalação de privilégios. **Timeout máximo** de 5s por comando. Centralizado em `safe_run()` (`core/security.py`) desde v0.2.

### 5.2 Secrets ✅ _Implementado em v0.1, auth em v0.4_

- `~/ai-local/.env` — não versionado (`.gitignore`)
- Nunca em `orchestrator.toml` (versionado)
- `ORC_ORCHESTRATOR_API_KEY` — autenticação via env var ou config (implementado em v0.4, timing-safe com `secrets.compare_digest`)

### 5.3 Injecção via context ✅ _Mitigado em v0.1, reforçado em v0.4_

Os context blocks injectados no prompt LLM vêm de fontes locais controladas (SQLite, ficheiros, git, subprocessos whitelisted, HTTP local). Risco de prompt injection via conteúdo de notas Obsidian ou output de git — baixo mas não nulo.

**Mitigação (v0.1):** Os context blocks são envolvidos em tags `[SOURCE]...[/SOURCE]` para delimitar claramente o contexto do prompt do utilizador.

**Reforçado (v0.4):** Todo o conteúdo de providers é passado por `sanitize_context()` em `_build_messages()` — remove caracteres de controlo e trunca a 32K caracteres antes de injecção no prompt.

### 5.4 Sanitização de input ✅ _Implementado em v0.4_

Módulo `orchestrator/core/sanitize.py` com validadores para todos os inputs externos:

| Função                       | Acção                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sanitize_query(text)`       | Remove control chars, strip whitespace, limita a 4096 chars                                                                                      |
| `validate_history(history)`  | Valida estrutura de dicts (`role` ∈ {user, assistant, system}, `content` não-vazio), remove keys extra, limita a 50 mensagens, sanitiza conteúdo |
| `validate_session_id(sid)`   | Aceita apenas UUIDs v4 válidos — previne SQL injection e path traversal                                                                          |
| `validate_model_name(model)` | Apenas `[a-zA-Z0-9.:/_-]`, max 128 chars — previne shell injection via model override                                                            |
| `sanitize_context(text)`     | Remove control chars do output dos providers, trunca a 32K antes de prompt injection                                                             |

Aplicado em: `api/app.py` (endpoints `/query` e `/classify`), `core/engine.py` (`_build_messages()`).

### 5.5 Security headers ✅ _Implementado em v0.4_

Middleware HTTP em `api/app.py` adiciona headers de segurança a todas as respostas:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Cache-Control: no-store`

### 5.6 Error handler ✅ _Implementado em v0.4_

Exception handler genérico em `api/app.py` captura excepções não tratadas e retorna `500 Internal server error` sem expor stack traces, nomes de ficheiros ou detalhes internos da aplicação. Erros são logados internamente para debugging.

### 5.7 Testes de segurança

37 testes em `tests/test_security.py` cobrem:

- Sanitização de texto (control chars, null bytes, unicode)
- Validação de queries (length, whitespace, empty)
- Validação de histórico (roles, missing content, extra keys, caps)
- Validação de session_id (UUID, SQL injection)
- Validação de model name (shell injection, length)
- Sanitização de contexto (control chars, truncation)
- Security headers na API (X-Content-Type-Options, X-Frame-Options, Cache-Control)
- Error handler não expõe internals

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

### Concluído em v0.2

| Melhoria                                | Ficheiros                                     | Estado |
| --------------------------------------- | --------------------------------------------- | ------ |
| LLM fallback na classificação de intent | `core/engine.py`                              | ✅     |
| `Engine.health_report()` público        | `core/engine.py`, `cli/main.py`, `api/app.py` | ✅     |
| Coverage report no pytest               | `pyproject.toml`                              | ✅     |
| `safe_run()` centralizado               | `core/security.py` (novo)                     | ✅     |
| GitHub Actions CI                       | `.github/workflows/orchestrator-ci.yml`       | ✅     |

### Concluído em v0.3

| Melhoria                                | Ficheiros                     | Estado |
| --------------------------------------- | ----------------------------- | ------ |
| Paralelismo em `_gather_context()`      | `core/engine.py`              | ✅     |
| Timeout individual por provider         | `core/engine.py`, `config.py` | ✅     |
| Rate limiting LLM (`asyncio.Semaphore`) | `api/app.py`, `config.py`     | ✅     |

### Concluído em v0.4

| Melhoria                          | Ficheiros                                    | Estado |
| --------------------------------- | -------------------------------------------- | ------ |
| Session store SQLite com TTL      | `core/session.py`, `api/app.py`, `config.py` | ✅     |
| API key authentication            | `api/app.py`, `config.py`                    | ✅     |
| `orc ask --stream --debug`        | `cli/main.py`                                | ✅     |
| Endpoint `/metrics` com latências | `core/metrics.py`, `api/app.py`              | ✅     |
| Health-gate + LLM error handling  | `core/engine.py`                             | ✅     |
| CAG robust health + retry         | `context/cag.py`                             | ✅     |
| Estimativa de tokens melhorada    | `context/base.py`, todos os providers        | ✅     |
| Input sanitisation layer          | `core/sanitize.py`, `api/app.py`             | ✅     |
| Security headers middleware       | `api/app.py`                                 | ✅     |
| Error handler (info leak protect) | `api/app.py`                                 | ✅     |
| Context sanitisation              | `core/sanitize.py`, `core/engine.py`         | ✅     |

### Prioridade Média

_Sem items pendentes._

### Prioridade Baixa

| Melhoria                      | Ficheiros        | Esforço |
| ----------------------------- | ---------------- | ------- |
| Pipeline agentic (ReAct loop) | `core/engine.py` | Alto    |
| Backends LLM alternativos     | `llm/`           | Médio   |

---

## 8. Roadmap sugerido

### v0.2 — Robustez ✅ _Completo — 2026-05-12_

- [x] LLM fallback para classificação de intent ambígua
- [x] `Engine.health_report()` como interface pública
- [x] `safe_run()` centralizado em `core/security.py`
- [x] Coverage report activo (`--cov=orchestrator`) — actual: 71%
- [x] GitHub Actions: `ruff` + `mypy` + `pytest -m "not integration"`

### v0.3 — Performance ✅ _Completo — 2026-05-12_

- [x] `_gather_context()` paralelo com `ThreadPoolExecutor`
- [x] Timeout individual por provider (configurável: `context.provider_timeout`)
- [x] Rate limiting para requests LLM concorrentes (`asyncio.Semaphore`, `ollama.max_concurrent_llm`)

### v0.4 — Funcionalidades ✅ _Completo — 2026-05-13_

- [x] Session store SQLite com TTL (`session_id` na API, opt-in via `session.enabled`)
- [x] API key authentication (`X-API-Key` / `Authorization: Bearer`, timing-safe)
- [x] `orc ask --stream --debug` (debug em stderr antes do stream)
- [x] Endpoint `/metrics` com latências históricas (in-memory ring buffer)
- [x] Health-gate + LLM error handling (`Engine.run()` / `stream()` — §4.1)
- [x] CAG robust health + retry (`_query_packs()` — §4.2)
- [x] Estimativa de tokens melhorada (`estimate_tokens()` — §2.1)
- [x] Input sanitisation layer (`core/sanitize.py` — §5.4)
- [x] Security headers middleware (`api/app.py` — §5.5)
- [x] Error handler — info leak protection (`api/app.py` — §5.6)
- [x] Context sanitisation antes de prompt injection (`_build_messages()` — §5.3)

### v1.0 — Agentes

- [ ] `Engine.run_agentic()` com ReAct loop
- [ ] Tool definitions para cada ContextProvider
- [ ] Suporte a backends LLM alternativos (OpenAI-compatible)
- [ ] Context providers adicionais (calendário, RSS, e-mail local)
