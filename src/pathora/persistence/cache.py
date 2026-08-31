"""Redis usage (Section 30): cache, rate limiting, distributed locks.

Falls back to an in-process implementation when REDIS_URL is unset so local runs
and CI need no Redis. The fallback is explicitly single-process and says so.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from pathora.config import Settings, get_settings


class InMemoryCache:
    backend = "memory"

    def __init__(self) -> None:
        self._values: dict[str, tuple[float | None, str]] = {}
        self._hits: dict[str, list[float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at is not None and expires_at < time.time():
            self._values.pop(key, None)
            return None
        return json.loads(payload)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = 3600) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        self._values[key] = (expires_at, json.dumps(value, default=str))

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < window_seconds]
        if len(hits) >= limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    @asynccontextmanager
    async def lock(self, key: str, *, timeout: int = 30):
        lock = self._locks.setdefault(key, asyncio.Lock())
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        try:
            yield
        finally:
            lock.release()


class RedisCache:  # pragma: no cover - requires a live Redis
    backend = "redis"

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError("pip install 'pathora-ai[infra]' to use Redis") from exc
        self._redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        payload = await self._redis.get(key)
        return json.loads(payload) if payload else None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = 3600) -> None:
        await self._redis.set(key, json.dumps(value, default=str), ex=ttl_seconds)

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, window_seconds)
        return count <= limit

    @asynccontextmanager
    async def lock(self, key: str, *, timeout: int = 30):
        async with self._redis.lock(f"lock:{key}", timeout=timeout):
            yield


def build_cache(settings: Settings | None = None):
    settings = settings or get_settings()
    return RedisCache(settings) if settings.redis_url else InMemoryCache()
