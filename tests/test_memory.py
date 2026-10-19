import pytest
from pathlib import Path
from app.memory.base import MemoryItem
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory
from app.memory.lifecycle import MemoryLifecycle
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore
from app.storage.graph import NullGraphStore
from app.llm.embedding import HashingEmbedder


@pytest.mark.asyncio
async def test_working_memory_ttl_and_retrieve():
    wm = WorkingMemory(capacity=10, ttl_seconds=2)
    item = MemoryItem(
        user_id="u1",
        type="working",
        content="用户提到了明天下午三点开会讨论产品设计",
        importance=0.8,
    )
    await wm.add(item)
    results = await wm.retrieve(query="开会", user_id="u1")
    assert len(results) == 1
    assert "开会" in results[0].item.content


@pytest.mark.asyncio
async def test_memory_lifecycle_consolidate_and_forget(tmp_path: Path):
    db_file = tmp_path / "test_agent.db"
    sqlite = SQLiteStore(db_file)
    await sqlite.initialize()

    vec = InMemoryVectorStore()
    embedder = HashingEmbedder(dim=128)
    graph = NullGraphStore()

    wm = WorkingMemory(capacity=10, ttl_seconds=3600)
    em = EpisodicMemory(sqlite=sqlite, vector_store=vec, embedder=embedder)
    sm = SemanticMemory(vector_store=vec, graph_store=graph, embedder=embedder)

    lifecycle = MemoryLifecycle(
        working=wm,
        episodic=em,
        semantic=sm,
        consolidate_working_thresh=0.7,
        consolidate_episodic_thresh=0.85,
        forget_importance_thresh=0.3,
    )

    # 1. 向 WorkingMemory 写入一条高重要性记忆
    await wm.add(MemoryItem(user_id="u1", type="working", content="重要决策：采用架构A", importance=0.8))
    # 写入一条低重要性记忆
    await wm.add(MemoryItem(user_id="u1", type="working", content="闲聊：今天天气不错", importance=0.2))

    # 触发 consolidate
    res = await lifecycle.consolidate("u1")
    assert res["working_to_episodic"] == 1

    # 验证已被转移至 EpisodicMemory
    e_items = await em.list_items("u1")
    assert len(e_items) == 1
    assert "架构A" in e_items[0].content

    # 2. 测试遗忘机制
    await em.add(MemoryItem(user_id="u1", type="episodic", content="无关紧要的琐事", importance=0.1))
    removed = await lifecycle.forget("u1", strategy="importance")
    assert removed == 1

    await sqlite.aclose()
