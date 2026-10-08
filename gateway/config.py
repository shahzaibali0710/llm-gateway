"""Configuration loading and validation.

The whole gateway is driven by one YAML file (see ``config.yaml``). Values may
contain ``${VAR}`` or ``${VAR:-default}`` which are expanded from the
environment, so the same file works on a laptop, in compose and in Kubernetes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROVIDER_TYPES = ("ollama", "anthropic", "mock")
RESERVED_ROUTE_NAMES = ("auto",)

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """The configuration file is malformed or inconsistent."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str
    base_url: str = ""
    api_key_env: str = ""
    timeout_s: float = 120.0
    # External providers receive PII-redacted prompts. Defaults to True for
    # everything except ollama and mock, i.e. anything that leaves the cluster.
    external: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str  # the id the provider knows it by
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_1m + output_tokens * self.output_cost_per_1m
        ) / 1_000_000


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    ttl_seconds: float = 300.0
    max_entries: int = 1000


@dataclass(frozen=True)
class GuardrailConfig:
    max_input_chars: int = 20000
    blocked_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoRouteConfig:
    complex_context_chars: int = 4000
    complex_turn_chars: int = 600


@dataclass(frozen=True)
class Config:
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routes: dict[str, tuple[str, ...]]
    api_keys: dict[str, str]  # api key -> team name
    cache: CacheConfig = field(default_factory=CacheConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    autoroute: AutoRouteConfig = field(default_factory=AutoRouteConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = _expand(raw or {})

        providers: dict[str, ProviderConfig] = {}
        for name, p in (raw.get("providers") or {}).items():
            ptype = p.get("type")
            if ptype not in PROVIDER_TYPES:
                raise ConfigError(f"provider {name!r}: type must be one of {PROVIDER_TYPES}")
            providers[name] = ProviderConfig(
                name=name,
                type=ptype,
                base_url=(p.get("base_url") or "").rstrip("/"),
                api_key_env=p.get("api_key_env", ""),
                timeout_s=float(p.get("timeout_s", 120)),
                external=bool(p.get("external", ptype not in ("ollama", "mock"))),
            )

        models: dict[str, ModelConfig] = {}
        for name, m in (raw.get("models") or {}).items():
            if m.get("provider") not in providers:
                raise ConfigError(f"model {name!r}: unknown provider {m.get('provider')!r}")
            if not m.get("model"):
                raise ConfigError(f"model {name!r}: 'model' (provider-side id) is required")
            models[name] = ModelConfig(
                name=name,
                provider=m["provider"],
                model=str(m["model"]),
                input_cost_per_1m=float(m.get("input_cost_per_1m", 0)),
                output_cost_per_1m=float(m.get("output_cost_per_1m", 0)),
            )

        routes: dict[str, tuple[str, ...]] = {}
        for name, chain in (raw.get("routes") or {}).items():
            if name in RESERVED_ROUTE_NAMES or name in models:
                raise ConfigError(f"route {name!r} clashes with a reserved name or a model name")
            if not chain:
                raise ConfigError(f"route {name!r} has an empty chain")
            for model_name in chain:
                if model_name not in models:
                    raise ConfigError(f"route {name!r}: unknown model {model_name!r}")
            routes[name] = tuple(chain)
        for needed in ("simple", "complex"):
            if needed not in routes:
                raise ConfigError(f"routes must define {needed!r} (used by model='auto')")

        api_keys = {str(k): str(v) for k, v in (raw.get("api_keys") or {}).items()}
        if not api_keys:
            raise ConfigError("api_keys must contain at least one key")

        c = raw.get("cache") or {}
        g = raw.get("guardrails") or {}
        a = raw.get("autoroute") or {}
        return cls(
            providers=providers,
            models=models,
            routes=routes,
            api_keys=api_keys,
            cache=CacheConfig(
                enabled=bool(c.get("enabled", True)),
                ttl_seconds=float(c.get("ttl_seconds", 300)),
                max_entries=int(c.get("max_entries", 1000)),
            ),
            guardrails=GuardrailConfig(
                max_input_chars=int(g.get("max_input_chars", 20000)),
                blocked_patterns=tuple(g.get("blocked_patterns") or ()),
            ),
            autoroute=AutoRouteConfig(
                complex_context_chars=int(a.get("complex_context_chars", 4000)),
                complex_turn_chars=int(a.get("complex_turn_chars", 600)),
            ),
        )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from ``path``, ``$GATEWAY_CONFIG`` or ``./config.yaml``."""
    path = Path(path or os.environ.get("GATEWAY_CONFIG") or "config.yaml")
    with path.open(encoding="utf-8") as fh:
        return Config.from_dict(yaml.safe_load(fh))
