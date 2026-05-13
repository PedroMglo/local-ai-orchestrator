"""LLMClient protocol — backend-agnostic interface for LLM generation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Backend-agnostic LLM interface."""

    def generate(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> str: ...

    def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> str: ...

    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        """Yield chunks of text as they arrive. Optional — may raise NotImplementedError."""
        ...

    def health(self) -> bool: ...
