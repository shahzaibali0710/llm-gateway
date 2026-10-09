"""Route resolution and the transparent auto-router.

``model="auto"`` is decided by a handful of regexes on the latest user turn.
It is deliberately simple and explainable: every decision returns a
human-readable reason that ends up in the response, the logs and the evals.

Rule order matters (first match wins):
  0. SENSITIVE pattern in the latest user turn -> sensitive (fail closed, see below)
  1. HARD pattern in the latest user turn      -> complex
  2. total context > complex_context_chars     -> complex
  3. EASY short-output task                    -> simple (even if the pasted article is long)
  4. latest user turn > complex_turn_chars     -> complex
  5. otherwise                                 -> simple

The ``sensitive`` route must have no local fallback: requests that need a safe
refusal (e.g. looking up a private person's address) are answered by the
strong model or fail with a 502. They are never handed to a small local model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import AutoRouteConfig, Config

# Requests that must be refused or handled carefully (eval: doxxing case). They
# fail closed: routed to the `sensitive` route, which has no local fallback.
SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:home\s+address|hjemmeadressen?|adressen?|address|personnummer|fødselsnummer|hvor\s+bor)\b",
    )
)

# Tasks that need reasoning, long-form generation or code.
HARD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bexplain\s+why\b",
        r"\banaly[sz](?:e|es|ed|ing|is)\b",
        r"\bcompar(?:e|es|ed|ing|ison)\b",
        r"\bstep[\s-]by[\s-]step\b",
        r"\bprove\b",
        r"\bderiv(?:e|es|ation)\b",
        r"\bdebug(?:ging)?\b",
        r"\b(?:write|draft)\s+(?:me\s+)?(?:(?:an?|the|some)\s+)?(?:\w+\s+)?(?:code|function|script|email|essay|report)\b",
        r"\bplan(?:s|ning)?\b",
        r"\bstrateg(?:y|ies|ic)\b",
        r"\bresearch\b",
        r"\bsummari[sz]e\s+this\b",
        r"```",
        # Calculations: small local models often get arithmetic wrong (eval: percentage case)
        r"\b(?:prosent|percent(?:age)?|beregn|calculate|regn\s+ut)\b",
        r"\d\s?%",
        # Rewriting with constraints: small models don't follow the instruction (eval: neutral rewrite)
        r"\b(?:rewrite|omskriv|skriv\s+om|neutral(?:ly)?|nøytral(?:t)?)\b",
    )
)

# Short-output tasks: cheap even when the input document is long.
EASY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bclassify\b",
        r"\bcategori[sz]e\b",
        r"\btag\b",
        r"\bheadline\b",
        r"\btitle\b",
        r"\btittel\b",
        r"\btranslate\b",
        r"\boversett\b",
        r"\bextract\b",
        r"\blist\s+the\b",
    )
)


class UnknownRouteError(KeyError):
    """The requested model/route name is neither ``auto``, a route nor a model."""


@dataclass(frozen=True)
class Resolution:
    route: str  # route name (or model name for direct addressing)
    reason: str
    chain: tuple[str, ...]  # ordered model names to try


def _latest_user_turn(messages: Sequence[Mapping[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def classify(messages: Sequence[Mapping[str, str]], cfg: AutoRouteConfig | None = None) -> tuple[str, str]:
    """Return ``(route, reason)`` for ``model="auto"``."""
    cfg = cfg or AutoRouteConfig()
    turn = _latest_user_turn(messages)
    total = sum(len(m["content"]) for m in messages)

    for pat in SENSITIVE_PATTERNS:
        hit = pat.search(turn)
        if hit:
            return "sensitive", f"sensitive request pattern {hit.group(0).strip()!r} (fail closed, no local fallback)"
    for pat in HARD_PATTERNS:
        hit = pat.search(turn)
        if hit:
            return "complex", f"hard task pattern {hit.group(0).strip()!r} in latest user turn"
    if total > cfg.complex_context_chars:
        return "complex", f"context {total} chars > {cfg.complex_context_chars}"
    for pat in EASY_PATTERNS:
        hit = pat.search(turn)
        if hit:
            return "simple", f"short-output task {hit.group(0).strip()!r} (context {total} chars)"
    if len(turn) > cfg.complex_turn_chars:
        return "complex", f"latest user turn {len(turn)} chars > {cfg.complex_turn_chars}"
    return "simple", "short request, no hard-task pattern"


def resolve(requested: str, messages: Sequence[Mapping[str, str]], config: Config) -> Resolution:
    """Turn the ``model`` field into an ordered fallback chain."""
    if requested == "auto":
        route, reason = classify(messages, config.autoroute)
        return Resolution(route, reason, config.routes[route])
    if requested in config.routes:
        return Resolution(requested, f"explicit route {requested!r}", config.routes[requested])
    if requested in config.models:
        # Every model is addressable directly as a single-model route (used by evals).
        return Resolution(requested, f"explicit model {requested!r}", (requested,))
    raise UnknownRouteError(requested)


def is_known(requested: str, config: Config) -> bool:
    return requested == "auto" or requested in config.routes or requested in config.models
