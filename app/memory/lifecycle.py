"""记忆生命周期：记忆整合强化（Consolidate）与智能遗忘（Forget）。"""

from __future__ import annotations

import time

from app.memory.base import BaseMemory, MemoryItem
from app.observability.logging import get_logger

log = get_logger(__name__)


class MemoryLifecycle:
    def __init__(
        self,
        working: BaseMemory,
        episodic: BaseMemory,
        semantic: BaseMemory,
        consolidate_working_thresh: float = 0.7,
        consolidate_episodic_thresh: float = 0.8,
        forget_importance_thresh: float = 0.2,
        forget_max_age_days: int = 30,
        forget_capacity: int = 1000,
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.consolidate_working_thresh = consolidate_working_thresh
        self.consolidate_episodic_thresh = consolidate_episodic_thresh
        self.forget_importance_thresh = forget_importance_thresh
        self.forget_max_age_days = forget_max_age_days
        self.forget_capacity = forget_capacity

    async def consolidate(self, user_id: str) -> dict[str, int]:
        """将高重要度的工作记忆沉淀到情景记忆，将高重要度的情景记忆提炼到语义记忆。"""
        log.info("memory_consolidate_started", user_id=user_id)
        w_to_e_count = 0
        e_to_s_count = 0

        # 1. Working -> Episodic
        w_items = await self.working.list_items(user_id, limit=100)
        to_episodic = [it for it in w_items if it.importance >= self.consolidate_working_thresh]
        for it in to_episodic:
            new_item = MemoryItem(
                user_id=it.user_id,
                session_id=it.session_id,
                type="episodic",
                content=it.content,
                importance=it.importance,
                metadata={"source_working_id": it.id, **it.metadata},
            )
            await self.episodic.add(new_item)
            w_to_e_count += 1
        if to_episodic:
            await self.working.remove([it.id for it in to_episodic])

        # 2. Episodic -> Semantic
        e_items = await self.episodic.list_items(user_id, limit=100)
        to_semantic = [it for it in e_items if it.importance >= self.consolidate_episodic_thresh]
        for it in to_semantic:
            new_item = MemoryItem(
                user_id=it.user_id,
                session_id=it.session_id,
                type="semantic",
                content=it.content,
                importance=it.importance,
                metadata={"source_episodic_id": it.id, **it.metadata},
            )
            await self.semantic.add(new_item)
            e_to_s_count += 1

        result = {"working_to_episodic": w_to_e_count, "episodic_to_semantic": e_to_s_count}
        log.info("memory_consolidate_completed", user_id=user_id, **result)
        return result

    async def forget(
        self,
        user_id: str,
        strategy: str = "importance",
        threshold: float | None = None,
        max_age_days: int | None = None,
        capacity: int | None = None,
    ) -> int:
        """根据策略遗忘情景记忆。"""
        log.info("memory_forget_started", user_id=user_id, strategy=strategy)
        items = await self.episodic.list_items(user_id, limit=2000)
        now = time.time()
        to_remove_ids: list[str] = []

        if strategy == "importance":
            thresh = threshold if threshold is not None else self.forget_importance_thresh
            to_remove_ids = [it.id for it in items if it.importance < thresh]
        elif strategy == "age":
            days = max_age_days if max_age_days is not None else self.forget_max_age_days
            max_age_sec = days * 86400
            to_remove_ids = [it.id for it in items if (now - it.created_at) > max_age_sec]
        elif strategy == "capacity":
            cap = capacity if capacity is not None else self.forget_capacity
            if len(items) > cap:
                # 按重要性升序和时间升序排序淘汰
                sorted_items = sorted(items, key=lambda x: (x.importance, x.created_at))
                to_remove_ids = [it.id for it in sorted_items[: len(items) - cap]]

        if to_remove_ids:
            await self.episodic.remove(to_remove_ids)

        log.info(
            "memory_forget_completed", user_id=user_id, removed_count=len(to_remove_ids), strategy=strategy
        )
        return len(to_remove_ids)
