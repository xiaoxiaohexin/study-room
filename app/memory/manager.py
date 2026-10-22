"""记忆管理器（MemoryManager）：管理四类记忆，执行并行跨类型检索、分值归一化、Top-K 聚合与生命周期控制。"""

from __future__ import annotations

import asyncio
from typing import Any

from app.memory.base import BaseMemory, MemoryItem, ScoredMemory
from app.memory.lifecycle import MemoryLifecycle
from app.observability.logging import get_logger

log = get_logger(__name__)


class MemoryManager:
    def __init__(
        self,
        working: BaseMemory | None = None,
        episodic: BaseMemory | None = None,
        semantic: BaseMemory | None = None,
        perceptual: BaseMemory | None = None,
        lifecycle: MemoryLifecycle | None = None,
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.perceptual = perceptual
        self.lifecycle = lifecycle
        self._stores: dict[str, BaseMemory] = {}
        if working is not None:
            self._stores["working"] = working
        if episodic is not None:
            self._stores["episodic"] = episodic
        if semantic is not None:
            self._stores["semantic"] = semantic
        if perceptual is not None:
            self._stores["perceptual"] = perceptual

    async def add(
        self,
        type: str,
        content: str,
        user_id: str,
        session_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        store = self._stores.get(type)
        if not store:
            raise ValueError(f"Unknown memory type: {type}")
        item = MemoryItem(
            user_id=user_id,
            session_id=session_id,
            type=type,  # type: ignore
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        mem_id = await store.add(item)
        log.info("memory_saved", type=type, user_id=user_id, memory_id=mem_id, importance=importance)
        return mem_id

    async def retrieve_all(
        self,
        queries: list[str],
        user_id: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[ScoredMemory]:
        """对扩展出的多个查询词，跨四类记忆并行检索，并按权重与归一化分值聚合返回 Top-K。"""
        if not queries:
            return []

        log.info(
            "memory_retrieve_all_started",
            user_id=user_id,
            session_id=session_id,
            query_count=len(queries),
            top_k=top_k,
        )
        tasks = []
        for q in queries:
            if self.working is not None:
                tasks.append(self.working.retrieve(q, user_id, top_k=top_k, session_id=session_id))
            if self.episodic is not None:
                tasks.append(self.episodic.retrieve(q, user_id, top_k=top_k, session_id=session_id))
            if self.semantic is not None:
                tasks.append(self.semantic.retrieve(q, user_id, top_k=top_k, session_id=session_id))
            if self.perceptual is not None:
                tasks.append(self.perceptual.retrieve(q, user_id, top_k=top_k, session_id=session_id))

        if not tasks:
            return []

        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        merged: dict[str, ScoredMemory] = {}
        for res in results_nested:
            if isinstance(res, Exception):
                log.warning("memory_retrieve_task_error", error=str(res))
                continue
            for sm in res:
                if sm.item.id not in merged or sm.score > merged[sm.item.id].score:
                    merged[sm.item.id] = sm

        sorted_list = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        top_results = sorted_list[:top_k]
        log.info(
            "memory_retrieve_all_finished",
            user_id=user_id,
            session_id=session_id,
            total_found=len(merged),
            returned=len(top_results),
        )
        return top_results

    async def list_items(
        self,
        user_id: str,
        type: str | None = None,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        if type and type in self._stores:
            return await self._stores[type].list_items(user_id, limit=limit, session_id=session_id)
        items: list[MemoryItem] = []
        for s in self._stores.values():
            items.extend(await s.list_items(user_id, limit=limit, session_id=session_id))
        return sorted(items, key=lambda x: x.created_at, reverse=True)[:limit]

    async def consolidate(self, user_id: str, session_id: str | None = None) -> dict[str, int]:
        if self.lifecycle is not None:
            return await self.lifecycle.consolidate(user_id, session_id=session_id)
        return {"working_to_episodic": 0, "episodic_to_semantic": 0}

    async def forget(self, user_id: str, strategy: str = "importance", **kwargs) -> int:
        if self.lifecycle is not None:
            return await self.lifecycle.forget(user_id, strategy=strategy, **kwargs)
        return 0
