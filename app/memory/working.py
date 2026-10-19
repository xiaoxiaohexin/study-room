"""工作记忆（Working Memory）：纯内存 RAM + TTL 自动过期 + 容量控制。

打分公式：(TF-IDF/字面重合 * 0.7 + 关键词 * 0.3) * 时间衰减 * (0.8 + importance * 0.4)
"""

from __future__ import annotations

import asyncio
import math
import re
import time

from app.memory.base import BaseMemory, MemoryItem, ScoredMemory
from app.observability.logging import get_logger

log = get_logger(__name__)


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return {w for w in words if len(w) > 1}


class WorkingMemory(BaseMemory):
    def __init__(self, capacity: int = 50, ttl_seconds: int = 1800):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        # user_id -> list[MemoryItem]
        self._store: dict[str, list[MemoryItem]] = {}
        self._lock = asyncio.Lock()

    def _purge_expired_locked(self, user_id: str) -> None:
        now = time.time()
        items = self._store.get(user_id, [])
        valid = [it for it in items if (now - it.created_at) <= self.ttl_seconds]
        if len(valid) != len(items):
            log.info(
                "purged_expired_working_memories",
                user_id=user_id,
                before=len(items),
                after=len(valid),
            )
        self._store[user_id] = valid

    async def add(self, item: MemoryItem) -> str:
        async with self._lock:
            self._purge_expired_locked(item.user_id)
            items = self._store.setdefault(item.user_id, [])
            if len(items) >= self.capacity:
                # 淘汰最老的一条
                popped = items.pop(0)
                log.info("working_memory_capacity_evicted", user_id=item.user_id, evicted_id=popped.id)
            items.append(item)
            log.info(
                "working_memory_added",
                user_id=item.user_id,
                memory_id=item.id,
                importance=item.importance,
                content_len=len(item.content),
            )
            return item.id

    async def retrieve(self, query: str, user_id: str, top_k: int = 5) -> list[ScoredMemory]:
        async with self._lock:
            self._purge_expired_locked(user_id)
            items = self._store.get(user_id, [])
            if not items:
                return []

            query_kw = _extract_keywords(query)
            now = time.time()
            scored: list[ScoredMemory] = []

            for it in items:
                it_kw = _extract_keywords(it.content)
                overlap = len(query_kw & it_kw)
                kw_score = overlap / (len(query_kw) or 1)
                sim_score = overlap / (math.sqrt((len(query_kw) or 1) * (len(it_kw) or 1)) or 1)

                age_hours = max(0.0, (now - it.created_at) / 3600.0)
                recency_decay = math.exp(-0.2 * age_hours)
                importance_factor = 0.8 + (it.importance * 0.4)

                final_score = (sim_score * 0.7 + kw_score * 0.3) * recency_decay * importance_factor

                scored.append(
                    ScoredMemory(
                        item=it,
                        score=float(final_score),
                        breakdown={
                            "sim": sim_score,
                            "kw": kw_score,
                            "recency": recency_decay,
                            "importance": importance_factor,
                        },
                    )
                )

            scored.sort(key=lambda s: s.score, reverse=True)
            res = scored[:top_k]
            log.debug("working_memory_retrieved", user_id=user_id, count=len(res))
            return res

    async def remove(self, ids: list[str]) -> None:
        id_set = set(ids)
        async with self._lock:
            for uid, items in self._store.items():
                self._store[uid] = [it for it in items if it.id not in id_set]
        log.info("working_memories_removed", count=len(ids))

    async def list_items(self, user_id: str, limit: int = 50) -> list[MemoryItem]:
        async with self._lock:
            self._purge_expired_locked(user_id)
            return list(reversed(self._store.get(user_id, [])))[:limit]
