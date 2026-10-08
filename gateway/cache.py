"""Exact-match response cache with TTL and LRU eviction."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence


def make_key(route: str, messages: Sequence[Mapping[str, str]], max_tokens: int | None) -> str:
    blob = json.dumps([route, list(messages), max_tokens], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.enabled = enabled
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        entry = self._data.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at > self.ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)  # mark as recently used
        return value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._data[key] = (self._clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)  # evict least recently used

    def __len__(self) -> int:
        return len(self._data)
