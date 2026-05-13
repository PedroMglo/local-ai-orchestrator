# AI Orchestrator

Orquestrador de LLMs local — routing inteligente de queries para o modelo e contexto certos.

## Arquitectura

```
Shell (ai) → Orchestrator → Intent + Complexity → Context Providers → Model Selection → LLM → Response
                                                    ├── RAG (notas Obsidian via HTTP)
                                                    ├── CAG (packs pré-computados SQLite)
                                                    ├── System (RAM, GPU, disco, CPU)
                                                    ├── Repo (git state dos projectos)
                                                    ├── Graph (knowledge graph Graphify)
                                                    ├── Config (modelos, serviços)
                                                    └── Logs (erros recentes)
```

## Quick Start

```bash
# Instalar
cd ~/ai-local/orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# CLI
orc classify "quanta RAM tenho?"       # → intent=system, complexity=normal
orc ask "o que dizem as minhas notas?" # → resposta com contexto RAG
orc config                             # → configuração actual
orc health                             # → estado dos componentes

# Servidor API
orc serve                              # → FastAPI em localhost:8585

# Shell (depois de source ~/.zsh_custom.d/42-ai.zsh)
ai quanta RAM tenho?                   # → routing automático
ai --classify debug esta função        # → mostra intent+complexity
ai -m deep analisa a arquitectura      # → forçar deepseek-r1
ai --json explica Docker               # → output JSON completo
ai --direct olá                        # → bypass orquestrador
```

## Modelos

| Atalho    | Modelo           | Uso                 |
| --------- | ---------------- | ------------------- |
| default   | qwen3:8b         | Generalista, PT-PT  |
| fast      | gemma3:4b        | Respostas rápidas   |
| code      | qwen2.5-coder:7b | Código              |
| deep      | deepseek-r1:8b   | Raciocínio profundo |
| embedding | bge-m3           | Embeddings          |

## Routing

### Intent Detection (heurístico, PT+EN)

| Intent  | Triggers                                     |
| ------- | -------------------------------------------- |
| LOCAL   | "notas", "vault", "obsidian", "apontamentos" |
| CODE    | "função", "código", "debug", "refactora"     |
| SYSTEM  | "RAM", "GPU", "disco", "processos"           |
| GRAPH   | "arquitectura", "dependências", "grafo"      |
| GENERAL | Sem keywords específicas                     |

### Model Selection (Intent × Complexity)

|         | SIMPLE  | NORMAL  | COMPLEX | DEEP |
| ------- | ------- | ------- | ------- | ---- |
| GENERAL | fast    | default | default | deep |
| LOCAL   | fast    | default | default | deep |
| CODE    | code    | code    | code    | deep |
| SYSTEM  | fast    | default | default | deep |
| GRAPH   | default | default | deep    | deep |

## Configuração

Ficheiro `orchestrator.toml` + variáveis de ambiente `ORC_*`:

```bash
ORC_MODELS_DEFAULT=qwen3:8b
ORC_RAG_URL=http://localhost:8484
ORC_ORCHESTRATOR_PORT=8585
```

## API

| Endpoint    | Método | Descrição                  |
| ----------- | ------ | -------------------------- |
| `/query`    | POST   | Query completa com routing |
| `/classify` | POST   | Classificação sem LLM      |
| `/health`   | GET    | Estado dos componentes     |

## Testes

```bash
pytest tests/ -v                           # todos (102 testes)
pytest tests/ -v -m "not integration"      # só unitários (88 testes)
pytest tests/ -v -m integration            # só integração (requer Ollama/RAG)
```

## Fallback (3 níveis)

1. **API** — Orchestrator HTTP (`:8585`)
2. **CLI** — `orc ask` via venv
3. **Ollama** — `ol` directo (sem contexto)

## Shell helpers — `ol` e família

Funções definidas em `~/.zsh_custom.d/42-ai.zsh`. Para activar no terminal actual:

```bash
source ~/.zsh_custom.d/42-ai.zsh
```

### `ol` — chat principal com memória de sessão

```bash
# Uso básico
ol "explica o que é um pipe em Linux"          # qwen3:8b (default)
ol "o que dizem as minhas notas sobre Docker?" # usa RAG + sessão automática

# Escolher modelo
ol gemma "resposta rápida"                     # gemma3:4b (mais rápido)
ol coder "refactora esta função para async"    # qwen2.5-coder:7b
ol deep "analisa os prós e contras"            # deepseek-r1:8b
ol qwen "qualquer pergunta"                    # qwen3:8b (explícito)

# Via pipe
cat script.py | ol coder "encontra bugs"
echo "log de erros" | ol "o que falhou?"

# Sessões (memória entre perguntas no mesmo terminal)
ol "quantos nodes existem no vault?"           # cria sessão automática
ol "e quantas tags?"                           # continua com contexto ✅
ol --new                                       # limpar sessão e começar do zero
```

A sessão é guardada em `$_OL_SESSION` (variável de ambiente do terminal). Cada terminal tem a sua própria sessão. O contexto persiste em SQLite até o TTL de 3600s expirar.

### `olfast` — atalho para gemma3:4b

```bash
olfast "qual é a porta default do PostgreSQL?"  # resposta em ~1s
olfast "diferença entre INNER e LEFT JOIN"
```

Equivalente a `ol gemma`. Ideal para perguntas rápidas e factuais.

### `aicode` — assistente de código

```bash
aicode "escreve uma função Python para ler Parquet com DuckDB"
cat app.py | aicode "encontra bugs neste código"
aicode "converte este bash para Python" < script.sh
```

Equivalente a `ol coder`. Usa `qwen2.5-coder:7b`.

### `aiask` — pergunta concisa (máximo 3 frases)

```bash
aiask "diferença entre git rebase e merge"
aiask "qual é a porta default do Redis?"
```

### `ai` — routing inteligente com fallback 3 níveis

O `ai` usa o orchestrator completo (com RAG, Graph, System probes e routing por intent):

```bash
ai quanta RAM tenho livre?              # → system probe + gemma3
ai o que dizem as minhas notas         # → RAG + qwen3
ai debug esta função Python            # → RAG + graph + coder
ai -m deep analisa esta arquitectura   # → forçar deepseek-r1
ai --classify qual é o uptime          # → mostra intent+complexity
ai --debug "quanto espaço em disco?"   # → mostra routing decisions
ai --json "explica Docker"             # → output JSON completo
ai --direct "olá"                      # → bypass orchestrator, usa ol
cat ficheiro.py | ai "revê isto"       # → stdin como contexto
```

| Flag          | Descrição                                            |
| ------------- | ---------------------------------------------------- |
| `-m <modelo>` | Forçar modelo específico                             |
| `--classify`  | Só classificar, sem LLM                              |
| `--debug`     | Mostrar intent, complexity, model, sources, latência |
| `--json`      | Output JSON completo                                 |
| `--direct`    | Bypass orchestrator, chama `ol` directamente         |

### `aiwarm` — pré-aquecer modelo na GPU

```bash
aiwarm           # qwen3:8b (default)
aiwarm gemma     # gemma3:4b
aiwarm coder     # qwen2.5-coder:7b
aiwarm deep      # deepseek-r1:8b
```

Carrega o modelo na VRAM em background para que a primeira query não tenha latência de arranque.

### `aistatus` — estado do Ollama e VRAM

```bash
aistatus   # modelos carregados + uso de VRAM (nvidia-smi)
aimodels   # lista todos os modelos instalados
```

### Tabela de referência rápida

| Comando          | Modelo             | Sessão        | RAG/Providers       |
| ---------------- | ------------------ | ------------- | ------------------- |
| `ol "..."`       | qwen3:8b           | ✅ automática | ✅ via orchestrator |
| `ol gemma "..."` | gemma3:4b          | ✅ automática | ✅ via orchestrator |
| `ol coder "..."` | qwen2.5-coder:7b   | ✅ automática | ✅ via orchestrator |
| `ol deep "..."`  | deepseek-r1:8b     | ✅ automática | ✅ via orchestrator |
| `olfast "..."`   | gemma3:4b          | ✅ automática | ✅ via orchestrator |
| `aicode "..."`   | qwen2.5-coder:7b   | ✅ automática | ✅ via orchestrator |
| `ai "..."`       | routing automático | ✅ automática | ✅ completo         |
| `ol --new`       | —                  | 🔄 reset      | —                   |

## Ficheiros externos editados

Ficheiros fora de `~/ai-local/orchestrator/` que foram criados ou modificados por este projecto:

| Ficheiro                    | Tipo       | Descrição                                                                                                                                                                                  |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `~/.zsh_custom.d/42-ai.zsh` | Modificado | Nova função `ai` com routing inteligente e fallback 3 níveis. Funções existentes (`ol`, `aicode`, `aiask`, `olfast`, `aiwarm`, `aistatus`, `aimodels`, `aiembed`) mantidas sem alterações. |
| `~/ai-local/.env`           | Criado     | Ficheiro de secrets centralizado (API keys). Protegido pelo `.gitignore` — nunca versionado.                                                                                               |
| `~/ai-local/.gitignore`     | Modificado | Adicionadas regras para `.env`, `__pycache__/`, `.venv/`, `*.db`, entre outros.                                                                                                            |

### Dependências de leitura (read-only)

O orquestrador lê mas **não modifica** os seguintes paths externos:

| Path                                         | Provider             | Uso                                                                        |
| -------------------------------------------- | -------------------- | -------------------------------------------------------------------------- |
| `~/ai-local/obsidian-rag/data/graphify/`     | `GraphProvider`      | Leitura de `graph.json` e `community_summaries.json`                       |
| `~/ai-local/obsidian-rag/data/qdrant/cag.db` | `CAGContextProvider` | Leitura de packs pré-computados (SQLite read-only)                         |
| `http://localhost:8484`                      | `RAGContextProvider` | Queries HTTP ao serviço RAG (endpoints `/query`, `/query/code`, `/health`) |
| `http://localhost:11434`                     | `OllamaLLMClient`    | Comunicação com Ollama (endpoints `/api/chat`, `/api/generate`)            |
| Repos em `orchestrator.toml [repos]`         | `RepoProbeProvider`  | Leitura de estado git (`git status`, `git log`)                            |
