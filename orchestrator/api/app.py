"""FastAPI application — orchestrator HTTP server on port 8585."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from orchestrator.api.schemas import (
    ClassifyResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from orchestrator.config import get_settings
from orchestrator.core.metrics import metrics
from orchestrator.core.sanitize import (
    sanitize_query,
    validate_history,
    validate_model_name,
    validate_session_id,
)
from orchestrator.core.session import SessionStore
from orchestrator.factory import create_engine

log = logging.getLogger(__name__)

# Module-level engine — created at startup
_engine = None
# Semaphore limiting concurrent LLM calls (Ollama processes 1 at a time on GPU)
_llm_semaphore: asyncio.Semaphore | None = None
# Session store — initialised in lifespan if enabled
_session_store: SessionStore | None = None

# Paths exempt from API key authentication
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/metrics", "/docs", "/openapi.json", "/redoc"})


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _llm_semaphore, _session_store
    cfg = get_settings()
    _llm_semaphore = asyncio.Semaphore(cfg.ollama.max_concurrent_llm)

    # Session store
    if cfg.session.enabled:
        _session_store = SessionStore(
            db_path=cfg.session.db_path or None,
            max_messages=cfg.session.max_messages,
        )
        _session_store.cleanup(cfg.session.ttl_seconds)
        log.info("Session store enabled (ttl=%ds)", cfg.session.ttl_seconds)

    log.info(
        "Orchestrator starting on %s:%d (max_concurrent_llm=%d)",
        cfg.orchestrator.host, cfg.orchestrator.port, cfg.ollama.max_concurrent_llm,
    )
    _get_engine()
    yield
    if _session_store is not None:
        _session_store.close()
    log.info("Orchestrator shutting down")


app = FastAPI(
    title="AI Orchestrator",
    version="0.4.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce API key on all endpoints except health, metrics, and docs."""
    cfg = get_settings()
    api_key = cfg.orchestrator.api_key
    if api_key and request.url.path not in _AUTH_EXEMPT_PATHS:
        # Accept X-API-Key header or Authorization: Bearer <key>
        provided = request.headers.get("X-API-Key", "")
        if not provided:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                provided = auth.removeprefix("Bearer ").strip()
        if not provided or not secrets.compare_digest(provided, api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API key"},
            )
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — return 500 without leaking internals."""
    log.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


async def _acquire_llm_slot() -> None:
    """Try to acquire the LLM semaphore without waiting.

    Raises HTTP 429 if all slots are busy.
    """
    if _llm_semaphore is None:
        return  # no semaphore yet (e.g. during tests without lifespan)
    if not _llm_semaphore._value:  # noqa: SLF001 — fast non-blocking check
        raise HTTPException(
            status_code=429,
            detail="LLM busy — all slots occupied, try again shortly",
            headers={"Retry-After": "5"},
        )
    await _llm_semaphore.acquire()


def _release_llm_slot() -> None:
    if _llm_semaphore is not None:
        _llm_semaphore.release()


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Main query endpoint — classify, gather context, select model, call LLM."""
    # --- Input sanitisation ---
    clean_query = sanitize_query(req.query)
    if not clean_query:
        raise HTTPException(status_code=422, detail="Query is empty after sanitisation")
    history = validate_history(req.history)
    model = validate_model_name(req.model)
    session_id = validate_session_id(req.session_id)

    await _acquire_llm_slot()
    try:
        engine = _get_engine()
        cfg = get_settings()

        # Resolve history — session store takes priority if enabled
        if _session_store is not None and cfg.session.enabled:
            import uuid as _uuid

            if session_id is None:
                session_id = str(_uuid.uuid4())
            stored = _session_store.get(session_id)
            if stored:
                history = stored + (history or [])

        if req.stream:
            def event_stream():
                try:
                    for chunk in engine.stream(
                        clean_query,
                        history=history,
                        model_override=model,
                    ):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    _release_llm_slot()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = engine.run(clean_query, history=history, model_override=model)

        # Record metrics
        metrics.record(result)

        # Persist to session store
        if _session_store is not None and cfg.session.enabled and session_id:
            _session_store.append(session_id, "user", clean_query)
            _session_store.append(session_id, "assistant", result.response)

        return QueryResponse(
            response=result.response,
            model_used=result.model_used,
            intent=result.intent.value,
            complexity=result.complexity.value,
            sources_used=result.sources_used,
            context_tokens=result.context_tokens,
            latency_ms=round(result.latency_ms, 1),
            session_id=session_id,
        )
    finally:
        # Release only for non-stream (stream releases in event_stream generator)
        if not req.stream:
            _release_llm_slot()


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: QueryRequest):
    """Classify only — no LLM call."""
    clean_query = sanitize_query(req.query)
    if not clean_query:
        raise HTTPException(status_code=422, detail="Query is empty after sanitisation")
    engine = _get_engine()
    history = validate_history(req.history)
    routing = engine.classify(clean_query, history=history)
    return ClassifyResponse(
        intent=routing.intent.value,
        complexity=routing.complexity.value,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — reports status of all components."""
    engine = _get_engine()
    report = engine.health_report()

    return HealthResponse(
        status="ok" if report["all_ok"] else "degraded",
        ollama=report["ollama"],
        rag=report["providers"].get("rag", False),
        providers=report["providers"],
    )


@app.get("/metrics")
def get_metrics(window: int = 300):
    """Query metrics — latency stats and intent/model distribution.

    Args:
        window: Time window in seconds (default 300 = 5 min). 0 = all time.
    """
    return metrics.summary(window_seconds=window)


def run_server():
    """Run the server programmatically (for CLI use)."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "orchestrator.api.app:app",
        host=cfg.orchestrator.host,
        port=cfg.orchestrator.port,
        log_level=cfg.logging.level.lower(),
    )
