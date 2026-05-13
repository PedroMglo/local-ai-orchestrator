# Bug Report — Smoke Conversations & Integration (2026-05-13)

Executado com: `python tests/smoke_conversations.py --model gemma3:4b --verbose`
Resultado: **12/12 PASS** (todos os testes passaram formalmente, mas com anomalias detectadas)

---

## BUG-001 · `<unusedXXXX>` tokens no output do LLM

**Severidade:** HIGH  
**Testes afectados:** T05 (CAG context packs), T04 (1ª run sem ORC)

### Descrição

O LLM (gemma3:4b) gera tokens especiais do vocabulário interno (`<unused3467>：`, `<unused2401>的笔记中说：`) no início das respostas quando recebe contexto do CAG/RAG. Estes tokens indicam que o modelo confunde os separadores do contexto injectado com marcadores de idioma (o `:` fullwidth `：` é lido como separador CJK).

### Evidência

```
T05 Resposta: <unused3467>：  **Notas e Configuração do Projeto** …
T04 Resposta (run-1): <unused2401>的笔记中说：  "Module docstring…"
```

### Causa provável

O CAG/RAG formata o contexto com o padrão `## título\nconteúdo`. Quando o conteúdo inclui código Python com comments, docstrings ou símbolos especiais, o gemma3:4b (modelo multilíngue com tokenizador BPE) interpreta a fronteira do bloco como input CJK e "activa" unused tokens do vocabulário.

### Acção recomendada

1. Adicionar no system prompt a instrução explícita: `"Respond only in European Portuguese (PT-PT). Never use CJK characters."`
2. Sanitizar o contexto antes de injectar: strip de caracteres unicode não-latin no `_CONTEXT_INSTRUCTION` do engine.
3. Validar se o problema persiste com `qwen3:8b` (modelo default) — provavelmente não reproduz.

---

## BUG-002 · Graph intent não detectado para query de arquitectura

**Severidade:** MEDIUM  
**Testes afectados:** T06

### Descrição

A query `"mostra a arquitectura e dependências do projecto obsidian-rag"` foi classificada como `general` em vez de `graph` ou `local_and_graph`. O engine não injectou contexto de grafo, e o LLM respondeu com conhecimento genérico de training data (inventando a arquitectura em vez de usar o grafo local).

### Evidência

```
T06  Intent: general  Sources: (none)  Context: 0 tokens  Latency: 22014ms
```

### Causa provável

O `HeuristicIntentClassifier` não detectou os sinais de grafo porque a query inclui o nome de projecto (`obsidian-rag`) que dilui os padrões. O fallback LLM classificou como `general` porque o gemma3:4b não conhece os intents locais do sistema.

### Acção recomendada

Adicionar ao heurístico os padrões de "projecto local + arquitectura":

```python
# intent.py — adicionar a _GRAPH_PATTERNS
"arquitectura.*projecto", "dependências.*projecto", "projecto.*arquitectura"
```

Ou aumentar o peso de `obsidian-rag` / nomes de repos configurados como sinais de intent `local_and_graph`.

---

## BUG-003 · Código injectado em queries de refactoring sem função fornecida

**Severidade:** LOW  
**Testes afectados:** T03

### Descrição

Para a query `"refactora esta função Python para usar async/await"`, o engine injectou 2826 tokens de contexto RAG+graph+CAG com código arbitrário dos repositórios, mesmo não havendo função concreta para refactorizar. O LLM perguntou correctamente pela função mas desperdiçou o budget de contexto com chunks irrelevantes.

### Evidência

```
T03  Sources: rag, graph, cag  Context: 2826 tokens
Resposta: "please provide the Python function you want me to refactor"
```

### Causa provável

O router classifica `code` como `[rag, graph, cag]` sem verificar se há código inline na query. O RAG retorna chunks de código dos repos com score ~0.50 (acima do threshold de 0.45).

### Acção recomendada

Para queries de código sem conteúdo inline, o router deve reduzir o budget RAG ou usar `min_score: 0.60` nas queries de código. Alternativamente, detectar "sem código na query" e omitir RAG.

---

## BUG-004 · Intent `code` para query de conceito Python via API (T11)

**Severidade:** LOW / INFO  
**Testes afectados:** T11

### Descrição

A query `"o que é uma variável em Python?"` foi classificada como `code` pelo LLM fallback (via HTTP API), quando deveria ser `general`. O sistema respondeu correctamente mas com overhead desnecessário de providers de contexto.

### Evidência

```
T11  Intent: code  Latency: 23462ms
```

### Causa provável

O LLM fallback (`gemma3:4b`) sobre-generaliza — "Python" + "variável" → `code`. O heurístico não deveria ter delegado ao LLM para esta query (4 palavras simples).

### Acção recomendada

Reduzir o threshold de words para o LLM fallback de `> 5` para `> 7`, ou adicionar `"o que é"` como padrão de intent `general` antes do fallback.

---

## BUG-005 · Latências elevadas em queries com contexto RAG (T03, T06, T07, T11, T12)

**Severidade:** INFO  
**Testes afectados:** T03 (21s), T06 (22s), T07 (32s), T11 (23s), T12 (20s)

### Descrição

Queries com contexto ou multi-turn têm latência entre 20-32 segundos com gemma3:4b. O modelo default é `qwen3:8b` — espera-se latência ainda superior.

### Causa

Limitação do hardware (CPU/GPU disponível). O semáforo `max_concurrent_llm=1` está correcto. Context de 2826 tokens aumenta o tempo de pré-fill significativamente.

### Acção recomendada

- Monitorizar com o endpoint `/metrics` em produção
- Para sessões interactivas usar `gemma3:4b` com budget reduzido (`token_budget: 3000`)
- Streaming (`T08: 245ms` até primeiro token) já está funcional — preferir para UX

---

## NOTA POSITIVA — Session Cache funcional (T11 + T12)

O sistema de sessões SQLite funciona correctamente:

- `session_id` UUID gerado e retornado pela API ✓
- Turno 2 usou o histórico do turno 1 (resposta coerente sobre lista em Python, em sequência de variável) ✓
- `session_id` consistente entre turnos ✓
- TTL e cleanup configurados (3600s) ✓

---

## Fix já aplicado nesta sessão

**`orchestrator/context/rag.py`** — `RAGContextProvider._query_collection`  
Corrigido: a query `/query` (notas pessoais) agora passa `exclude_source_type: "repo_doc"` para evitar que docs de projectos contaminem respostas sobre notas pessoais (causa raiz do bug original reportado).

```python
# Antes
notes = self._query_collection(query, "/query", top_k=top_k)

# Depois
notes = self._query_collection(
    query, "/query", top_k=top_k, exclude_source_type="repo_doc"
)
```

---

_Gerado automaticamente — NOS Coding Agent — 2026-05-13_
