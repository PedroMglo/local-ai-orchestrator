"""API schemas — Pydantic models for request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    model: str | None = Field(None, description="Override model selection")
    stream: bool = Field(False, description="Stream response as SSE")
    history: list[dict] | None = Field(None, description="Conversation history")
    session_id: str | None = Field(None, description="Session ID for conversation continuity")


class ClassifyResponse(BaseModel):
    intent: str
    complexity: str


class QueryResponse(BaseModel):
    response: str
    model_used: str
    intent: str
    complexity: str
    sources_used: list[str]
    context_tokens: int
    latency_ms: float
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    rag: bool
    providers: dict[str, bool]
