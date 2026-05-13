#!/usr/bin/env python3
"""
Smoke tests de conversação — valida integração completa
Ollama + Orchestrator + RAG + CAG

Usa o modelo mais rápido disponível (gemma3:4b por defeito).

Uso:
    python tests/smoke_conversations.py
    python tests/smoke_conversations.py --model gemma3:4b
    python tests/smoke_conversations.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Adicionar o root do projecto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers de output ─────────────────────────────────────────────────────────

RESET  = "\033[0m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def _h1(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")


def _h2(title: str) -> None:
    print(f"\n{CYAN}── {title} {'─' * (55 - len(title))}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _info(label: str, value: str) -> None:
    print(f"  {DIM}{label:<14}{RESET} {value}")


def _response(text: str, max_chars: int = 200) -> None:
    preview = text[:max_chars].replace("\n", " ").strip()
    suffix = "…" if len(text) > max_chars else ""
    print(f"  {DIM}Resposta:{RESET} {preview}{suffix}")


# ── Resultado ─────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float
    intent: str = ""
    complexity: str = ""
    model_used: str = ""
    sources: list[str] = field(default_factory=list)
    context_tokens: int = 0
    error: str = ""
    response_preview: str = ""


# ── Testes ───────────────────────────────────────────────────────────────────

def run_test(
    name: str,
    query: str,
    engine,
    *,
    model_override: str | None = None,
    history: list[dict] | None = None,
    expect_intent: str | None = None,
    expect_sources: list[str] | None = None,
    verbose: bool = False,
) -> TestResult:
    _h2(name)
    _info("Query:", f'"{query[:80]}"')

    t0 = time.perf_counter()
    try:
        result = engine.run(query, model_override=model_override, history=history)
        latency = (time.perf_counter() - t0) * 1000

        checks: list[tuple[bool, str]] = []

        # Resposta não vazia
        checks.append((bool(result.response.strip()), "Resposta não vazia"))

        # Intent esperado
        if expect_intent:
            ok = result.intent.value == expect_intent
            checks.append((ok, f"Intent = {expect_intent} (got: {result.intent.value})"))

        # Sources esperadas (subset)
        if expect_sources:
            for src in expect_sources:
                ok = src in result.sources_used
                checks.append((ok, f"Source '{src}' presente"))

        all_ok = all(c[0] for c in checks)

        _info("Intent:", result.intent.value)
        _info("Complexity:", result.complexity.value)
        _info("Model:", result.model_used)
        _info("Sources:", ", ".join(result.sources_used) if result.sources_used else "(none)")
        _info("Context:", f"{result.context_tokens} tokens")
        _info("Latency:", f"{latency:.0f}ms")

        for ok, label in checks:
            if ok:
                _ok(label)
            else:
                _fail(label)

        if verbose or not all_ok:
            _response(result.response)

        return TestResult(
            name=name,
            passed=all_ok,
            latency_ms=latency,
            intent=result.intent.value,
            complexity=result.complexity.value,
            model_used=result.model_used,
            sources=result.sources_used,
            context_tokens=result.context_tokens,
            response_preview=result.response[:200],
        )

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        _fail(f"Excepção: {exc}")
        return TestResult(name=name, passed=False, latency_ms=latency, error=str(exc))


# ── Suite principal ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests de conversação")
    parser.add_argument("--model", default="gemma3:4b", help="Modelo a usar (default: gemma3:4b)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar resposta completa")
    args = parser.parse_args()

    _h1("Smoke Tests — Ollama + Orchestrator + RAG + CAG")
    print(f"  Modelo rápido: {BOLD}{args.model}{RESET}")
    print(f"  Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Verificar serviços ────────────────────────────────────────────────────
    _h2("1. Verificação de serviços")

    import httpx

    services = {
        "Ollama": "http://localhost:11434/api/tags",
        "RAG":    "http://localhost:8484/health",
        "ORC API": "http://localhost:8585/health",
    }
    service_state: dict[str, bool] = {}
    for name, url in services.items():
        try:
            r = httpx.get(url, timeout=3)
            ok = r.status_code == 200
        except Exception:
            ok = False
        service_state[name] = ok
        ((_ok if ok else _fail)(f"{name}: {'UP' if ok else 'DOWN'}"))

    if not service_state["Ollama"]:
        print(f"\n{RED}FATAL: Ollama não está disponível. Abortar.{RESET}")
        return 1

    # ── Criar engine ──────────────────────────────────────────────────────────
    _h2("2. Inicialização do Engine")
    try:
        from orchestrator.config import _reset_settings
        _reset_settings()
        from orchestrator.factory import create_engine
        engine = create_engine()
        _ok("Engine criado com todos os providers")
    except Exception as exc:
        _fail(f"Falha ao criar engine: {exc}")
        return 1

    results: list[TestResult] = []

    # ── Teste 1: Query geral (sem contexto, modelo fast) ──────────────────────
    r = run_test(
        "T01 · Query geral simples",
        "o que é DNS?",
        engine,
        model_override=args.model,
        expect_intent="general",
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 2: Classificação de sistema ─────────────────────────────────────
    r = run_test(
        "T02 · Detecção de intent: sistema",
        "quanta RAM tenho disponível?",
        engine,
        model_override=args.model,
        expect_intent="system",
        expect_sources=["system"],
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 3: Detecção de intent: código ───────────────────────────────────
    r = run_test(
        "T03 · Detecção de intent: código",
        "refactora esta função Python para usar async/await",
        engine,
        model_override=args.model,
        expect_intent="code",
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 4: Query local (via RAG se disponível) ───────────────────────────
    rag_up = service_state.get("RAG", False)
    r = run_test(
        "T04 · Query local (RAG)" + (" [RAG DOWN — sem contexto]" if not rag_up else ""),
        "o que dizem as minhas notas sobre Python?",
        engine,
        model_override=args.model,
        expect_intent="local",
        expect_sources=["rag"] if rag_up else None,
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 5: CAG — packs do context cache ─────────────────────────────────
    r = run_test(
        "T05 · CAG — context packs",
        "mostra as minhas notas e configuração do projecto",
        engine,
        model_override=args.model,
        expect_intent="local",
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 6: Query de grafo / arquitectura ────────────────────────────────
    r = run_test(
        "T06 · Detecção de intent: grafo",
        "mostra a arquitectura e dependências do projecto obsidian-rag",
        engine,
        model_override=args.model,
        expect_intent=None,  # pode ser graph ou local_and_graph
        verbose=args.verbose,
    )
    results.append(r)

    # ── Teste 7: Multi-turn (histórico de conversa) ───────────────────────────
    _h2("T07 · Multi-turn — conversa com histórico")
    _info("Turno 1:", "o que é uma função Python?")

    t0 = time.perf_counter()
    try:
        r1 = engine.run("o que é uma função Python?", model_override=args.model)
        _info("Resposta 1:", r1.response[:100].replace("\n", " ") + "…")

        history = [
            {"role": "user", "content": "o que é uma função Python?"},
            {"role": "assistant", "content": r1.response},
        ]

        _info("Turno 2:", "e um decorator?")
        r2 = engine.run("e um decorator?", model_override=args.model, history=history)
        latency = (time.perf_counter() - t0) * 1000

        _info("Resposta 2:", r2.response[:100].replace("\n", " ") + "…")
        _info("Model:", r2.model_used)
        _info("Latency:", f"{latency:.0f}ms (2 turnos)")

        # Verifica coerência: resposta 2 deve mencionar algo relacionado com Python/decorator
        keywords = {"decorator", "função", "python", "wrapper", "function", "@"}
        response_lower = r2.response.lower()
        coherent = any(k in response_lower for k in keywords)
        _ok("Resposta coerente com histórico") if coherent else _fail("Resposta sem contexto do histórico")

        results.append(TestResult(
            name="T07 · Multi-turn",
            passed=bool(r1.response) and bool(r2.response) and coherent,
            latency_ms=latency,
            intent=r2.intent.value,
            model_used=r2.model_used,
            response_preview=r2.response[:200],
        ))
    except Exception as exc:
        _fail(f"Excepção: {exc}")
        results.append(TestResult(name="T07 · Multi-turn", passed=False, latency_ms=0, error=str(exc)))

    # ── Teste 8: Streaming ────────────────────────────────────────────────────
    _h2("T08 · Streaming de tokens")
    t0 = time.perf_counter()
    try:
        chunks: list[str] = []
        for chunk in engine.stream("diz apenas 'streaming OK'", model_override=args.model):
            chunks.append(chunk)
        latency = (time.perf_counter() - t0) * 1000
        full = "".join(chunks)
        _info("Chunks:", str(len(chunks)))
        _info("Resposta:", full[:100].replace("\n", " "))
        _info("Latency:", f"{latency:.0f}ms")
        _ok("Streaming funcional") if chunks else _fail("Sem chunks recebidos")
        results.append(TestResult(
            name="T08 · Streaming",
            passed=bool(chunks),
            latency_ms=latency,
            model_used=args.model,
            response_preview=full[:200],
        ))
    except Exception as exc:
        _fail(f"Excepção: {exc}")
        results.append(TestResult(name="T08 · Streaming", passed=False, latency_ms=0, error=str(exc)))

    # ── Teste 9: RAG directo via HTTP ─────────────────────────────────────────
    if rag_up:
        _h2("T09 · RAG HTTP — query directa ao provider")
        try:
            from orchestrator.context.rag import RAGContextProvider
            prov = RAGContextProvider()
            t0 = time.perf_counter()
            block = prov.get_context("Python machine learning")
            latency = (time.perf_counter() - t0) * 1000
            _info("Latency:", f"{latency:.0f}ms")
            _info("Block:", f"source={block.source}, tokens≈{block.token_estimate}" if block else "None (sem resultados)")
            _ok("RAG provider responde") if block is not None else _ok("RAG responde (sem resultados para esta query)")
            results.append(TestResult(
                name="T09 · RAG HTTP",
                passed=True,
                latency_ms=latency,
                sources=["rag"] if block else [],
                context_tokens=block.token_estimate if block else 0,
            ))
        except Exception as exc:
            _fail(f"Excepção: {exc}")
            results.append(TestResult(name="T09 · RAG HTTP", passed=False, latency_ms=0, error=str(exc)))
    else:
        _h2("T09 · RAG HTTP [SKIPPED — RAG DOWN]")

    # ── Teste 10: CAG directo via SQLite ──────────────────────────────────────
    _h2("T10 · CAG SQLite — leitura de packs")
    try:
        from orchestrator.context.cag import CAGContextProvider
        prov = CAGContextProvider(intent_hint="local")
        t0 = time.perf_counter()
        block = prov.get_context("configuração do projecto")
        latency = (time.perf_counter() - t0) * 1000
        _info("Latency:", f"{latency:.0f}ms")
        if block:
            _info("Block:", f"source={block.source}, tokens≈{block.token_estimate}")
            _info("Preview:", block.content[:80].replace("\n", " ") + "…")
            _ok("CAG leu packs do SQLite")
        else:
            _info("Resultado:", "Nenhum pack válido (possível TTL expirado)")
            _ok("CAG responde sem erro (packs expirados)")
        results.append(TestResult(
            name="T10 · CAG SQLite",
            passed=True,
            latency_ms=latency,
            context_tokens=block.token_estimate if block else 0,
        ))
    except Exception as exc:
        _fail(f"Excepção: {exc}")
        results.append(TestResult(name="T10 · CAG SQLite", passed=False, latency_ms=0, error=str(exc)))

    # ── Teste 11: Session cache via HTTP API ──────────────────────────────────
    orc_up = service_state.get("ORC API", False)
    if orc_up:
        import uuid as _uuid
        session_id = str(_uuid.uuid4())
        _h2("T11 · Session cache — turno 1 (criar sessão)")
        t0 = time.perf_counter()
        try:
            r_s1 = httpx.post(
                "http://localhost:8585/query",
                json={
                    "query": "o que é uma variável em Python?",
                    "model": args.model,
                    "session_id": session_id,
                },
                timeout=60,
            )
            r_s1.raise_for_status()
            data_s1 = r_s1.json()
            latency = (time.perf_counter() - t0) * 1000
            _info("Session ID:", session_id[:16] + "…")
            _info("Intent:", data_s1.get("intent", "?"))
            _info("Latency:", f"{latency:.0f}ms")
            _info("Resposta:", data_s1.get("response", "")[:100].replace("\n", " ") + "…")
            _ok("Sessão criada com sucesso")
            results.append(TestResult(
                name="T11 · Session turno 1",
                passed=True,
                latency_ms=latency,
                intent=data_s1.get("intent", ""),
                model_used=data_s1.get("model_used", ""),
                response_preview=data_s1.get("response", "")[:200],
            ))
        except Exception as exc:
            _fail(f"Excepção: {exc}")
            results.append(TestResult(name="T11 · Session turno 1", passed=False, latency_ms=0, error=str(exc)))
            session_id = None  # skip T12

        _h2("T12 · Session cache — turno 2 (continuidade)")
        if session_id:
            t0 = time.perf_counter()
            try:
                r_s2 = httpx.post(
                    "http://localhost:8585/query",
                    json={
                        "query": "e o que é uma lista?",
                        "model": args.model,
                        "session_id": session_id,
                    },
                    timeout=60,
                )
                r_s2.raise_for_status()
                data_s2 = r_s2.json()
                latency = (time.perf_counter() - t0) * 1000
                _info("Session ID:", session_id[:16] + "…")
                _info("Latency:", f"{latency:.0f}ms")
                _info("Resposta:", data_s2.get("response", "")[:120].replace("\n", " ") + "…")

                # Verificar coerência — a resposta deve mencionar Python ou lista
                resp_lower = data_s2.get("response", "").lower()
                coherent = any(k in resp_lower for k in {"lista", "list", "python", "elemento", "append", "colecção"})
                if coherent:
                    _ok("Resposta coerente com histórico de sessão (session cache funcional)")
                else:
                    _fail("Resposta não mostra evidência do histórico — session cache pode não estar a funcionar")

                # Verificar que o session_id retornado bate certo
                returned_id = data_s2.get("session_id")
                id_match = returned_id == session_id
                _ok("session_id consistente") if id_match else _fail(f"session_id devolvido diferente: {returned_id}")

                results.append(TestResult(
                    name="T12 · Session turno 2",
                    passed=coherent and id_match,
                    latency_ms=latency,
                    intent=data_s2.get("intent", ""),
                    model_used=data_s2.get("model_used", ""),
                    response_preview=data_s2.get("response", "")[:200],
                ))
            except Exception as exc:
                _fail(f"Excepção: {exc}")
                results.append(TestResult(name="T12 · Session turno 2", passed=False, latency_ms=0, error=str(exc)))
        else:
            _info("Skipped:", "turno 1 falhou, não é possível testar continuidade")
    else:
        _h2("T11/T12 · Session cache [SKIP — ORC API DOWN]")
        _info("Razão:", "Servidor :8585 não está disponível (executa 'orc serve')")

    # ── Sumário ───────────────────────────────────────────────────────────────
    _h1("Sumário")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_latency = sum(r.latency_ms for r in results)

    print(f"  Testes:   {BOLD}{len(results)}{RESET}")
    print(f"  Passou:   {GREEN}{passed}{RESET}")
    print(f"  Falhou:   {RED if failed else DIM}{failed}{RESET}")
    print(f"  Latência: {total_latency:.0f}ms total\n")

    # Tabela
    col_w = 32
    print(f"  {'Teste':<{col_w}} {'Estado':<10} {'Intent':<18} {'Latency':>8}")
    print(f"  {'-' * col_w} {'-' * 9} {'-' * 17} {'-' * 8}")
    for r in results:
        estado = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
        intent = r.intent or "—"
        print(f"  {r.name:<{col_w}} {estado:<10} {intent:<18} {r.latency_ms:>7.0f}ms")
        if r.error:
            print(f"    {RED}└─ {r.error[:70]}{RESET}")

    print()

    if service_state.get("ORC API"):
        print(f"  {DIM}Servidor API :8585 está UP — podes fazer: ai --debug 'o que é DNS?'{RESET}")
    else:
        print(f"  {YELLOW}⚠ Servidor API :8585 não está UP — executa 'orc serve' para iniciá-lo{RESET}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
