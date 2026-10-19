"""缓存抽象：Redis（可选）+ 进程内 TTL 缓存降级。"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str, ttl: int) -> int: ...
    async def aclose(self) -> None: ...


class InMemoryTTLCache:
    def __init__(self, max_items: int = 10000):
        self._data: dict[str, tuple[float | None, Any]] = {}
        self.max_items = max_items

    def _alive(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return False
        exp, _ = item
        if exp is not None and exp < time.time():
            self._data.pop(key, None)
            return False
        return True

    async def get(self, key):
        return self._data[key][1] if self._alive(key) else None

    async def set(self, key, value, ttl=None):
        if len(self._data) >= self.max_items:
            self._data.pop(next(iter(self._data)))
        self._data[key] = (time.time() + ttl if ttl else None, value)

    async def delete(self, key):
        self._data.pop(key, None)

    async def incr(self, key, ttl):
        if self._alive(key):
            exp, v = self._data[key]
            self._data[key] = (exp, v + 1)
            return v + 1
        self._data[key] = (time.time() + ttl, 1)
        return 1

    async def aclose(self):
        self._data.clear()


class RedisCache:
    def __init__(self, url: str):
        import redis.asyncio as redis  # 可选依赖

        self.r = redis.from_url(url, decode_responses=True)

    async def ping(self):
        await self.r.ping()

    async def get(self, key):
        v = await self.r.get(key)
        return json.loads(v) if v is not None else None

    async def set(self, key, value, ttl=None):
        await self.r.set(key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl)

    async def delete(self, key):
        await self.r.delete(key)

    async def incr(self, key, ttl):
        n = await self.r.incr(key)
        if n == 1:
            await self.r.expire(key, ttl)
        return n

    async def aclose(self):
        await self.r.aclose()
