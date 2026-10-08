"""Input guardrails and PII redaction.

Guardrails fail *closed*: a blocked or oversized request is rejected (400).
Redaction is applied only to prompts bound for external providers; local
models see the original text because it never leaves the cluster.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping, Sequence

from .config import GuardrailConfig


class GuardrailError(Exception):
    """Request rejected by a guardrail (mapped to HTTP 400)."""


# Order matters: more specific / longer patterns first so they are not
# partially eaten by a looser one (e.g. a card number by the phone pattern).
PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){12,15}\d(?!\d)"), "[CARD]"),
    # Norwegian fødselsnummer: ddmmyy + 5 digits.
    ("fnr", re.compile(r"(?<!\d)\d{6}[ ]?\d{5}(?!\d)"), "[FNR]"),
    # Norwegian phone: 8 digits starting 2-9, optional +47 / 0047 prefix.
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:(?:\+|00)47[ -]?)?(?:[2-9]\d{2}[ -]?\d{2}[ -]?\d{3}|[2-9]\d(?:[ -]?\d{2}){3})(?!\d)"
        ),
        "[PHONE]",
    ),
)


class Guardrails:
    def __init__(self, cfg: GuardrailConfig) -> None:
        self.max_input_chars = cfg.max_input_chars
        self._blocked = [re.compile(p) for p in cfg.blocked_patterns]

    def check(self, messages: Sequence[Mapping[str, str]]) -> None:
        total = sum(len(m["content"]) for m in messages)
        if total > self.max_input_chars:
            raise GuardrailError(f"input too long: {total} chars (max {self.max_input_chars})")
        for m in messages:
            for pat in self._blocked:
                if pat.search(m["content"]):
                    raise GuardrailError("request blocked by content policy (prompt-injection pattern)")


def redact_text(text: str) -> tuple[str, Counter[str]]:
    found: Counter[str] = Counter()
    for kind, pattern, token in PII_PATTERNS:
        text, n = pattern.subn(token, text)
        if n:
            found[kind] += n
    return text, found


def redact_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Return a redacted copy of ``messages`` plus per-type redaction counts."""
    total: Counter[str] = Counter()
    out: list[dict[str, str]] = []
    for m in messages:
        content, found = redact_text(m["content"])
        total.update(found)
        out.append({**m, "content": content})
    return out, total
