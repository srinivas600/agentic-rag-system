"""
In-memory cache that mirrors the Redis async interface.
Used in dev mode when Redis is unavailable.
"""
from __future__ import annotations

import json
import time
from typing import Any


class MemoryCache:
    """Drop-in replacement for redis.asyncio.Redis with basic operations."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._lists: dict[str, list[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    def _is_expired(self, key: str) -> bool:
        if key in self._expiry and time.time() > self._expiry[key]:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> str | None:
        if self._is_expired(key):
            return None
        return self._store.get(key)

    async def set(self, key: str, value: str, **kwargs) -> None:
        self._store[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._expiry[key] = time.time() + ttl

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)
            self._expiry.pop(k, None)
            self._lists.pop(k, None)

    async def ping(self) -> bool:
        return True

    async def rpush(self, key: str, *values: str) -> int:
        if key not in self._lists:
            self._lists[key] = []
        self._lists[key].extend(values)
        return len(self._lists[key])

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    async def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    async def expire(self, key: str, ttl: int) -> None:
        self._expiry[key] = time.time() + ttl

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        if key not in self._hashes:
            self._hashes[key] = {}
        current = int(self._hashes[key].get(field, "0"))
        new_val = current + amount
        self._hashes[key][field] = str(new_val)
        return new_val

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if key not in self._zsets:
            self._zsets[key] = {}
        self._zsets[key].update(mapping)
        return len(mapping)

    async def zrangebyscore(
        self, key: str, min_score: str, max_score: str, withscores: bool = False
    ) -> list:
        zset = self._zsets.get(key, {})
        items = sorted(zset.items(), key=lambda x: x[1])
        if withscores:
            return items
        return [member for member, _ in items]

    async def close(self) -> None:
        pass


_memory_cache: MemoryCache | None = None


def get_memory_cache() -> MemoryCache:
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCache()
    return _memory_cache
