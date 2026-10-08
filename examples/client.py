"""Tiny helper for talking to the gateway.

Environment:
  GATEWAY_URL  base URL of the gateway   (default http://localhost:8080)
  GATEWAY_KEY  your team's API key       (default newsroom-dev-key)

Because the gateway is OpenAI-compatible, the official OpenAI SDK works too,
just point it at the gateway and use model="auto" (or a route/model name):

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8080/v1", api_key="newsroom-dev-key")
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Foreslå en tittel til denne saken: ..."}],
    )
    print(resp.choices[0].message.content)
    print(resp.model_extra["gateway"])   # route, reason, latency_s, cost_usd, ...
"""

from __future__ import annotations

import os
from typing import Any

import httpx

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080").rstrip("/")
GATEWAY_KEY = os.environ.get("GATEWAY_KEY", "newsroom-dev-key")


def chat(
    prompt: str | list[dict[str, str]],
    model: str = "auto",
    system: str | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """POST one chat completion; returns the full response JSON (incl. the ``gateway`` block)."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    body: dict[str, Any] = {"model": model, "messages": messages, "cache": cache}
    if max_tokens:
        body["max_tokens"] = max_tokens
    resp = httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def text(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    out = chat("Si hei på norsk.")
    print(text(out))
    print(out["gateway"])
