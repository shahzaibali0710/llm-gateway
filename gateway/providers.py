"""Model providers.

Ollama is spoken to through its OpenAI-compatible ``/v1/chat/completions``
endpoint, so pointing ``base_url`` at vLLM or SGLang works unchanged.
Anthropic uses the native Messages API. Every failure surfaces as
``ProviderError`` so the router can fall back to the next model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config, ProviderConfig

Messages = list[dict[str, str]]


class ProviderError(Exception):
    """A provider could not produce a completion (network, HTTP or payload error)."""


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int
    output_tokens: int


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) for providers that report no usage."""
    return max(1, len(text) // 4)


class Provider:
    #: True when prompts leave the trust boundary and must be PII-redacted.
    external: bool = True

    async def complete(self, model: str, messages: Messages, max_tokens: int | None) -> Completion:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class _HTTPProvider(Provider):
    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.external = cfg.external
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            resp = await self.client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.cfg.name}: {type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(f"{self.cfg.name}: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.cfg.name}: response was not JSON") from exc


class OllamaProvider(_HTTPProvider):
    """Any OpenAI-compatible server: Ollama, vLLM, SGLang, llama.cpp server."""

    async def complete(self, model: str, messages: Messages, max_tokens: int | None) -> Completion:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = await self._post(f"{self.cfg.base_url}/v1/chat/completions", payload, {})
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.cfg.name}: unexpected response shape") from exc
        usage = data.get("usage") or {}
        return Completion(
            content=content,
            input_tokens=int(
                usage.get("prompt_tokens") or estimate_tokens(" ".join(m["content"] for m in messages))
            ),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(content)),
        )


class AnthropicProvider(_HTTPProvider):
    API_VERSION = "2023-06-01"
    DEFAULT_MAX_TOKENS = 1024  # the Messages API requires max_tokens

    async def complete(self, model: str, messages: Messages, max_tokens: int | None) -> Completion:
        api_key = os.environ.get(self.cfg.api_key_env or "ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(f"{self.cfg.name}: {self.cfg.api_key_env or 'ANTHROPIC_API_KEY'} is not set")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": api_key, "anthropic-version": self.API_VERSION}
        data = await self._post(f"{self.cfg.base_url}/v1/messages", payload, headers)
        try:
            content = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data["usage"]
            return Completion(content, int(usage["input_tokens"]), int(usage["output_tokens"]))
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"{self.cfg.name}: unexpected response shape") from exc


class MockProvider(Provider):
    """Deterministic offline provider. The model id ``fail`` always raises."""

    def __init__(self, cfg: ProviderConfig | None = None) -> None:
        self.external = cfg.external if cfg else False

    async def complete(self, model: str, messages: Messages, max_tokens: int | None) -> Completion:
        if model == "fail":
            raise ProviderError("mock: simulated provider failure")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        content = f"[mock:{model}] {last_user[:500]}"
        return Completion(
            content=content,
            input_tokens=estimate_tokens(" ".join(m["content"] for m in messages)),
            output_tokens=estimate_tokens(content),
        )


def build_providers(config: Config) -> dict[str, Provider]:
    built: dict[str, Provider] = {}
    for name, pc in config.providers.items():
        if pc.type == "ollama":
            built[name] = OllamaProvider(pc)
        elif pc.type == "anthropic":
            built[name] = AnthropicProvider(pc)
        else:
            built[name] = MockProvider(pc)
    return built
