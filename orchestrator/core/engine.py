"""Orchestrator engine — main entry point for query processing.

Flow:
  1. IntentClassifier → Intent
  2. ComplexityClassifier → Complexity
  3. ContextRouter → list of source names
  4. ContextProviders → ContextBlocks (with budget)
  5. ModelRouter → model name
  6. LLMClient → response
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Iterator

import httpx

from orchestrator.config import get_settings
from orchestrator.context.base import (
    ContextBlock,
    Intent,
    OrchestratorResult,
    RoutingResult,
)
from orchestrator.core.complexity import HeuristicComplexityClassifier
from orchestrator.core.context_router import ConfigContextRouter
from orchestrator.core.intent import HeuristicIntentClassifier
from orchestrator.core.model_router import ConfigModelRouter
from orchestrator.core.sanitize import sanitize_context
from orchestrator.llm.base import LLMClient
from orchestrator.llm.ollama import OllamaLLMClient

log = logging.getLogger(__name__)

# System prompt for the orchestrator
_SYSTEM_PROMPT = (
    "Tu és um assistente inteligente e versátil. "
    "Respondes de forma clara, directa e precisa sobre qualquer tema. "
    "Usas português de Portugal (PT-PT). "
    "Não assumes que todas as perguntas são sobre um domínio específico. "
    "Quando não tens certeza, dizes honestamente que não sabes."
)

_CONTEXT_INSTRUCTION = (
    "The context below was retrieved from the user's local knowledge base. "
    "Use it to answer accurately. If the context does not contain relevant "
    "information, answer from your general knowledge and state that clearly. "
    "Never fabricate information that is not in the context."
)


def _estimate_tokens(text: str) -> int:
    """Delegate to the canonical estimator in context.base."""
    from orchestrator.context.base import estimate_tokens
    return estimate_tokens(text)


_LLM_UNAVAILABLE_MSG = (
    "⚠️ O serviço LLM (Ollama) não está disponível de momento. "
    "Verifique o estado com `orc health` e certifique-se de que o Ollama está a correr."
)

# How long to cache the LLM health check result (seconds)
_LLM_HEALTH_CACHE_TTL = 5.0


class Engine:
    """Main orchestration engine."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        providers: dict[str, object] | None = None,
    ) -> None:
        self._intent_clf = HeuristicIntentClassifier()
        self._complexity_clf = HeuristicComplexityClassifier()
        self._model_router = ConfigModelRouter()
        self._context_router = ConfigContextRouter()
        self._llm = llm or OllamaLLMClient()
        self._providers: dict[str, object] = providers or {}
        # Cached LLM health state
        self._llm_healthy: bool = True
        self._llm_health_ts: float = 0.0

    def _is_llm_available(self) -> bool:
        """Check LLM health with a short-lived cache to avoid per-query overhead."""
        now = time.monotonic()
        if now - self._llm_health_ts < _LLM_HEALTH_CACHE_TTL:
            return self._llm_healthy
        self._llm_healthy = self._llm.health()
        self._llm_health_ts = now
        if not self._llm_healthy:
            log.warning("Engine: LLM health check failed — Ollama unavailable")
        return self._llm_healthy

    def register_provider(self, provider: object) -> None:
        """Register a ContextProvider by its name."""
        name = getattr(provider, "name", None)
        if name is None:
            raise ValueError("Provider must have a 'name' attribute")
        self._providers[name] = provider

    def health_report(self) -> dict[str, Any]:
        """Return health status of all components.

        Returns:
            A dict with keys ``ollama`` (bool), ``providers`` (dict[str, bool]),
            and ``all_ok`` (bool — True only when Ollama is healthy).
        """
        providers_health: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                providers_health[name] = bool(getattr(provider, "health", lambda: True)())
            except Exception:
                providers_health[name] = False

        ollama_ok = self._llm.health()
        return {
            "ollama": ollama_ok,
            "providers": providers_health,
            "all_ok": ollama_ok,
        }

    def classify(self, query: str, *, history: list[dict] | None = None) -> RoutingResult:
        """Classify intent and complexity without executing."""
        intent = self._classify_intent(query, history=history)
        complexity = self._complexity_clf.classify(query)
        return RoutingResult(intent=intent, complexity=complexity)

    def _llm_intent_fallback(self, query: str) -> Intent | None:
        """Ask the fast LLM to classify intent when the heuristic returns GENERAL.

        Returns an ``Intent`` value if the LLM responds with a recognised label,
        or ``None`` on any failure (timeout, parse error, LLM unavailable).
        """
        cfg = get_settings()
        model = cfg.models.fast
        _VALID = "|".join(i.value for i in Intent)
        _PATTERN = re.compile(rf"\b({_VALID})\b", re.IGNORECASE)

        prompt = (
            f"Classify the following query into exactly ONE of these intents: "
            f"{', '.join(i.value for i in Intent)}.\n"
            f"Reply with ONLY the intent name, nothing else.\n"
            f"Query: {query}"
        )
        messages = [
            {"role": "system", "content": "You are a precise intent classifier. Reply with a single word."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self._llm.chat(messages, model, temperature=0.0, max_tokens=16, timeout=3.0)
            m = _PATTERN.search(raw.strip())
            if m:
                return Intent(m.group(1).upper())
        except Exception as exc:  # noqa: BLE001
            log.debug("Engine: LLM intent fallback failed: %s", exc)
        return None

    def _classify_intent(self, query: str, *, history: list[dict] | None = None) -> Intent:
        """Classify intent using the heuristic; fall back to LLM for ambiguous queries.

        The LLM fallback is triggered only when:
        - the heuristic returns ``Intent.GENERAL`` (no strong local signals), AND
        - the query has more than 5 words (short queries are clear enough).
        """
        intent = self._intent_clf.classify(query, history=history)
        if intent is Intent.GENERAL and len(query.split()) > 5:
            fallback = self._llm_intent_fallback(query)
            if fallback is not None:
                log.debug("Engine: intent heuristic=GENERAL llm_fallback=%s", fallback.value)
                return fallback
        return intent

    def _gather_context(
        self, query: str, sources: list[str], *, budget: int = 6000
    ) -> list[ContextBlock]:
        """Query registered providers in parallel and collect context blocks.

        Each provider runs in a separate thread with an individual timeout
        (``cfg.context.provider_timeout``). Results are re-ordered by the
        original ``sources`` priority and truncated to the token budget.
        """
        cfg = get_settings()
        provider_timeout = cfg.context.provider_timeout

        # Filter to available healthy providers (preserving order)
        active: list[tuple[str, object]] = []
        for source_name in sources:
            provider = self._providers.get(source_name)
            if provider is None:
                log.debug("Engine: provider %r not registered, skipping", source_name)
                continue
            if not getattr(provider, "health", lambda: True)():
                log.warning("Engine: provider %r unhealthy, skipping", source_name)
                continue
            active.append((source_name, provider))

        if not active:
            return []

        # Run providers in parallel
        results: dict[str, ContextBlock | None] = {}
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {
                executor.submit(
                    provider.get_context, query, budget_tokens=budget  # type: ignore[union-attr]
                ): name
                for name, provider in active
            }
            for future in futures:
                name = futures[future]
                try:
                    block = future.result(timeout=provider_timeout)
                    if block is not None and block.content:
                        results[name] = block
                except FuturesTimeoutError:
                    log.warning("Engine: provider %r timed out after %ds", name, provider_timeout)
                except Exception as exc:
                    log.warning("Engine: provider %r failed: %s", name, exc)

        gather_ms = (time.perf_counter() - t0) * 1000
        log.debug("Engine: parallel gather_context took %.0fms (%d providers)", gather_ms, len(active))

        # Re-order by original source priority and enforce budget
        blocks: list[ContextBlock] = []
        remaining = budget
        for source_name in sources:
            block = results.get(source_name)
            if block is None:
                continue
            if block.token_estimate <= remaining:
                blocks.append(block)
                remaining -= block.token_estimate
            else:
                break

        return blocks

    def _build_messages(
        self,
        query: str,
        context_blocks: list[ContextBlock],
        *,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """Build the message list for the LLM."""
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if context_blocks:
            context_parts = []
            for block in context_blocks:
                safe_content = sanitize_context(block.content)
                context_parts.append(f"[{block.source.upper()}]\n{safe_content}\n[/{block.source.upper()}]")
            context_text = "\n\n".join(context_parts)
            messages.append({
                "role": "system",
                "content": f"{_CONTEXT_INSTRUCTION}\n\n{context_text}",
            })

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})
        return messages

    def run(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        model_override: str | None = None,
    ) -> OrchestratorResult:
        """Full orchestration: classify → context → model → LLM → result."""
        t0 = time.perf_counter()
        cfg = get_settings()

        # 1. Classify
        intent = self._classify_intent(query, history=history)
        complexity = self._complexity_clf.classify(query)
        log.info("Engine: intent=%s complexity=%s", intent.value, complexity.value)

        # 2. Context routing
        sources = self._context_router.route(intent, complexity)

        # 3. Gather context
        blocks = self._gather_context(query, sources, budget=cfg.context.token_budget)

        # 4. Model selection
        model = model_override or self._model_router.select(intent, complexity)
        log.info("Engine: model=%s sources=%s blocks=%d", model, sources, len(blocks))

        # 5. Build messages and call LLM
        messages = self._build_messages(query, blocks, history=history)

        # Health-gate: fail fast if Ollama is known-down
        if not self._is_llm_available():
            latency = (time.perf_counter() - t0) * 1000
            return OrchestratorResult(
                response=_LLM_UNAVAILABLE_MSG,
                model_used=model,
                intent=intent,
                complexity=complexity,
                sources_used=[b.source for b in blocks],
                context_tokens=sum(b.token_estimate for b in blocks),
                latency_ms=latency,
            )

        try:
            response = self._llm.chat(messages, model)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            log.error("Engine: LLM call failed: %s", exc)
            # Invalidate health cache so next query also fast-fails
            self._llm_healthy = False
            self._llm_health_ts = time.monotonic()
            latency = (time.perf_counter() - t0) * 1000
            return OrchestratorResult(
                response=_LLM_UNAVAILABLE_MSG,
                model_used=model,
                intent=intent,
                complexity=complexity,
                sources_used=[b.source for b in blocks],
                context_tokens=sum(b.token_estimate for b in blocks),
                latency_ms=latency,
            )

        latency = (time.perf_counter() - t0) * 1000
        context_tokens = sum(b.token_estimate for b in blocks)

        return OrchestratorResult(
            response=response,
            model_used=model,
            intent=intent,
            complexity=complexity,
            sources_used=[b.source for b in blocks],
            context_tokens=context_tokens,
            latency_ms=latency,
        )

    def stream(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        model_override: str | None = None,
    ) -> Iterator[str]:
        """Streaming variant: yields text chunks."""
        cfg = get_settings()

        intent = self._classify_intent(query, history=history)
        complexity = self._complexity_clf.classify(query)
        sources = self._context_router.route(intent, complexity)
        blocks = self._gather_context(query, sources, budget=cfg.context.token_budget)
        model = model_override or self._model_router.select(intent, complexity)
        messages = self._build_messages(query, blocks, history=history)

        # Health-gate: fail fast if Ollama is known-down
        if not self._is_llm_available():
            yield _LLM_UNAVAILABLE_MSG
            return

        try:
            yield from self._llm.chat_stream(messages, model)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            log.error("Engine: LLM stream failed: %s", exc)
            self._llm_healthy = False
            self._llm_health_ts = time.monotonic()
            yield _LLM_UNAVAILABLE_MSG
