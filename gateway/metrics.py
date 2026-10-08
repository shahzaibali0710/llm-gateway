"""Prometheus metrics plus the in-memory per-team usage behind GET /v1/usage.

Each app owns its own registry so several apps (e.g. in tests) can coexist in
one process without duplicate-registration errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_requests",
            "Chat requests by outcome.",
            ["team", "route", "model", "status"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "gateway_tokens",
            "Tokens processed.",
            ["team", "model", "direction"],
            registry=self.registry,
        )
        self.cost = Counter(
            "gateway_cost_usd",
            "Spend in USD.",
            ["team", "model"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "gateway_latency_seconds",
            "Provider call latency.",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "gateway_fallbacks",
            "Times a model failed and the next one in the chain was tried.",
            ["route", "from_model"],
            registry=self.registry,
        )
        self.cache_hits = Counter(
            "gateway_cache_hits", "Responses served from cache.", registry=self.registry
        )
        self.pii_redactions = Counter(
            "gateway_pii_redactions",
            "PII spans redacted before a prompt left the cluster.",
            ["type"],
            registry=self.registry,
        )
        self._usage: dict[str, dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cache_hits": 0}
        )

    def record_request(self, team: str, route: str, model: str, status: str) -> None:
        self.requests.labels(team, route, model, status).inc()
        self._usage[team]["requests"] += 1

    def record_success(self, team: str, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.tokens.labels(team, model, "input").inc(input_tokens)
        self.tokens.labels(team, model, "output").inc(output_tokens)
        self.cost.labels(team, model).inc(cost)
        u = self._usage[team]
        u["input_tokens"] += input_tokens
        u["output_tokens"] += output_tokens
        u["cost_usd"] += cost

    def record_cache_hit(self, team: str) -> None:
        self.cache_hits.inc()
        self._usage[team]["cache_hits"] += 1

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            team: {**u, "cost_usd": round(u["cost_usd"], 8)} for team, u in sorted(self._usage.items())
        }

    def render(self) -> bytes:
        return generate_latest(self.registry)
