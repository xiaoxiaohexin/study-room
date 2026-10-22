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


def _normalize_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    s = str(session_id).strip()
    return s if s else None


class WorkingMemory(BaseMemory):
    def __init__(self, capacity: int = 50, ttl_seconds: int = 1800):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        # (user_id, session_id) -> list[MemoryItem]
        self._store: dict[tuple[str, str | None], list[MemoryItem]] = {}
        self._lock = asyncio.Lock()

    def _purge_expired_locked(self, user_id: str, session_id: str | None) -> None:
        now = time.time()
        key = (user_id, session_id)
        items = self._store.get(key, [])
        if not items:
            return
        valid = [it for it in items if (now - it.created_at) <= self.ttl_seconds]
        if len(valid) != len(items):
            log.info(
                "purged_expired_working_memories",
                user_id=user_id,
                session_id=session_id,
                before=len(items),
                after=len(valid),
            )
        self._store[key] = valid

    def _purge_all_user_expired_locked(self, user_id: str) -> None:
        for u, s in list(self._store.keys()):
            if u == user_id:
                self._purge_expired_locked(u, s)

    async def add(self, item: MemoryItem) -> str:
        norm_sess = _normalize_session_id(item.session_id)
        key = (item.user_id, norm_sess)
        async with self._lock:
            self._purge_expired_locked(item.user_id, norm_sess)
            items = self._store.setdefault(key, [])
            if len(items) >= self.capacity:
                # 淘汰该 session 内部最老的一条
                popped = items.pop(0)
                log.info(
                    "working_memory_capacity_evicted",
                    user_id=item.user_id,
                    session_id=norm_sess,
                    evicted_id=popped.id,
                )
            items.append(item)
            log.info(
                "working_memory_added",
                user_id=item.user_id,
                session_id=norm_sess,
                memory_id=item.id,
                importance=item.importance,
                content_len=len(item.content),
            )
            return item.id

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[ScoredMemory]:
        norm_sess = _normalize_session_id(session_id)
        async with self._lock:
            if norm_sess is not None:
                self._purge_expired_locked(user_id, norm_sess)
                items = self._store.get((user_id, norm_sess), [])
            else:
                self._purge_all_user_expired_locked(user_id)
                # 优先检查未绑定会话的 legacy bucket
                items = self._store.get((user_id, None), [])
                if not items:
                    # 遗留无 session_id 调用兼容：跨会话聚合
                    items = [
                        it
                        for (u, _), it_list in self._store.items()
                        if u == user_id
                        for it in it_list
                    ]

            if not items:
                return []

            query_kw = _extract_keywords(query)
            now = time.time()
            scored: list[ScoredMemory] = []

            for it in items:
                it_kw = _extract_keywords(it.content)
                sub_matches = sum(1 for w in query_kw if w in it.content.lower())
                overlap = max(len(query_kw & it_kw), sub_matches)
                kw_score = overlap / (len(query_kw) or 1)
                sim_score = overlap / (math.sqrt((len(query_kw) or 1) * (len(it_kw) or 1)) or 1)

                age_hours = max(0.0, (now - it.created_at) / 3600.0)
                recency_decay = math.exp(-0.2 * age_hours)
                importance_factor = 0.8 + (it.importance * 0.4)

                final_score = (sim_score * 0.7 + kw_score * 0.3) * recency_decay * importance_factor
                if final_score <= 0.0:
                    continue

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
            log.debug("working_memory_retrieved", user_id=user_id, session_id=session_id, count=len(res))
            return res

    async def remove(
        self,
        ids: list[str],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        id_set = set(ids)
        norm_sess = _normalize_session_id(session_id)
        async with self._lock:
            for key, items in list(self._store.items()):
                if user_id is not None and key[0] != user_id:
                    continue
                if session_id is not None and key[1] != norm_sess:
                    continue
                self._store[key] = [it for it in items if it.id not in id_set]
        log.info("working_memories_removed", count=len(ids), user_id=user_id, session_id=session_id)

    async def list_items(
        self,
        user_id: str,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        norm_sess = _normalize_session_id(session_id)
        async with self._lock:
            if norm_sess is not None:
                self._purge_expired_locked(user_id, norm_sess)
                return list(reversed(self._store.get((user_id, norm_sess), [])))[:limit]

            self._purge_all_user_expired_locked(user_id)
            unsessioned = self._store.get((user_id, None), [])
            if unsessioned:
                return list(reversed(unsessioned))[:limit]
            all_user_items = [
                it
                for (u, _), it_list in self._store.items()
                if u == user_id
                for it in it_list
            ]
            return sorted(all_user_items, key=lambda x: x.created_at, reverse=True)[:limit]

    async def clear(self, user_id: str | None = None, session_id: str | None = None) -> None:
        norm_sess = _normalize_session_id(session_id)
        async with self._lock:
            if user_id is None:
                self._store.clear()
            elif session_id is None:
                keys_to_del = [k for k in self._store if k[0] == user_id]
                for k in keys_to_del:
                    del self._store[k]
            else:
                self._store.pop((user_id, norm_sess), None)
