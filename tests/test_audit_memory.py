"""审计 P1-6 回归：记忆整合幂等，语义记忆跨进程可枚举、可删除。"""

from __future__ import annotations

from pathlib import Path

from app.llm.embedding import HashingEmbedder
from app.memory.base import MemoryItem
from app.memory.episodic import EpisodicMemory
from app.memory.lifecycle import MemoryLifecycle
from app.memory.semantic import SemanticMemory
from app.memory.working import WorkingMemory
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore, SQLiteVectorStore


async def test_consolidate_does_not_duplicate_semantic_memory(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "mem.db")
    await sqlite.initialize()
    try:
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        wm = WorkingMemory(capacity=10)
        em = EpisodicMemory(sqlite=sqlite, vector_store=vec, embedder=embedder)
        sm = SemanticMemory(vector_store=vec, graph_store=NullGraphStore(), embedder=embedder)
        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_episodic_thresh=0.8,
        )

        await em.add(
            MemoryItem(user_id="u1", type="episodic", content="用户偏好先推导再刷题", importance=0.9)
        )
        await lifecycle.consolidate("u1")
        await lifecycle.consolidate("u1")
        await lifecycle.consolidate("u1")

        semantic_items = await sm.list_items("u1", limit=100)
        assert len(semantic_items) == 1
        assert semantic_items[0].metadata.get("source_episodic_id")
    finally:
        await sqlite.aclose()


async def test_semantic_memory_is_enumerable_after_restart(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "sem.db")
    await sqlite.initialize()
    try:
        embedder = HashingEmbedder(dim=128)
        vec_first = SQLiteVectorStore(sqlite)
        sm_first = SemanticMemory(
            vector_store=vec_first, graph_store=NullGraphStore(), embedder=embedder
        )
        item = MemoryItem(user_id="u1", type="semantic", content="牛顿第二定律 F=ma", importance=0.9)
        await sm_first.add(item)

        # 模拟进程重启：新的向量库缓存与新的语义记忆实例
        vec_second = SQLiteVectorStore(sqlite)
        sm_second = SemanticMemory(
            vector_store=vec_second, graph_store=NullGraphStore(), embedder=embedder
        )
        items = await sm_second.list_items("u1", limit=10)
        assert len(items) == 1
        assert "牛顿第二定律" in items[0].content

        # 删除同样必须落到持久层
        await sm_second.remove([item.id], user_id="u1")
        assert await sm_second.list_items("u1", limit=10) == []
    finally:
        await sqlite.aclose()