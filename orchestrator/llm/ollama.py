"""Ollama implementation of LLMClient."""

from __future__ import annotations

import json
import logging
import re
from typing import Iterator

import httpx

from orchestrator.config import get_settings

log = logging.getLogger(__name__)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLMClient:
    """LLMClient backed by an Ollama server."""

    def __init__(self, *, base_url: str | None = None) -> None:
        self._base_url = base_url or get_settings().ollama.base_url

    def generate(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> str:
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _THINK_PATTERN.sub("", raw).strip()

    def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> str:
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        return _THINK_PATTERN.sub("", raw).strip()

    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> Iterator[str]:
        """Stream chat tokens from Ollama."""
        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=httpx.Timeout(connect=5.0, read=timeout, write=30.0, pool=10.0),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    def health(self) -> bool:
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
