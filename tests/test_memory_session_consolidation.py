"""Phase P1-B tests: session-scoped memory consolidation and scoped removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.embedding import HashingEmbedder
from app.memory.base import MemoryItem
from app.memory.episodic import EpisodicMemory
from app.memory.lifecycle import MemoryLifecycle
from app.memory.manager import MemoryManager
from app.memory.semantic import SemanticMemory
from app.memory.working import WorkingMemory
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore


@pytest.fixture
async def memory_bundle(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "memory_scope.db")
    await sqlite.initialize()
    vector = InMemoryVectorStore()
    embedder = HashingEmbedder(dim=256)
    working = WorkingMemory(capacity=20, ttl_seconds=3600)
    episodic = EpisodicMemory(sqlite=sqlite, vector_store=vector, embedder=embedder)
    semantic = SemanticMemory(vector_store=vector, graph_store=NullGraphStore(), embedder=embedder)
    lifecycle = MemoryLifecycle(working=working, episodic=episodic, semantic=semantic)
    manager = MemoryManager(
        working=working, episodic=episodic, semantic=semantic, lifecycle=lifecycle
    )
    yield manager, working, episodic, sqlite
    await sqlite.aclose()


class TestSessionScopedConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_only_promotes_current_session(self, memory_bundle):
        manager, working, episodic, _ = memory_bundle
        user = "student_a"

        item_a = MemoryItem(
            user_id=user,
            session_id="sess_a",
            type="working",
            content="会话 A 的高价值工具结果",
            importance=0.9,
        )
        item_b = MemoryItem(
            user_id=user,
            session_id="sess_b",
            type="working",
            content="会话 B 的高价值工具结果",
            importance=0.9,
        )
        await working.add(item_a)
        await working.add(item_b)

        result = await manager.consolidate(user, session_id="sess_a")
        assert result["working_to_episodic"] == 1

        remaining_working = await working.list_items(user, limit=10)
        assert [it.id for it in remaining_working] == [item_b.id]

        episodic_a = await episodic.list_items(user, session_id="sess_a")
        assert len(episodic_a) == 1
        assert "会话 A" in episodic_a[0].content
        episodic_b = await episodic.list_items(user, session_id="sess_b")
        assert episodic_b == []

    @pytest.mark.asyncio
    async def test_manager_list_items_respects_session_filter(self, memory_bundle):
        manager, working, _, _ = memory_bundle
        user = "student_b"
        await working.add(
            MemoryItem(user_id=user, session_id="s1", type="working", content="s1 item", importance=0.6)
        )
        await working.add(
            MemoryItem(user_id=user, session_id="s2", type="working", content="s2 item", importance=0.6)
        )

        items_s1 = await manager.list_items(user, type="working", session_id="s1")
        assert [it.session_id for it in items_s1] == ["s1"]
        all_items = await manager.list_items(user, type="working")
        assert len(all_items) == 2


class TestScopedRemoval:
    @pytest.mark.asyncio
    async def test_working_remove_respects_session_scope(self, memory_bundle):
        _, working, _, _ = memory_bundle
        user = "student_c"
        item_a = MemoryItem(
            user_id=user, session_id="sess_a", type="working", content="A", importance=0.5
        )
        item_b = MemoryItem(
            user_id=user, session_id="sess_b", type="working", content="B", importance=0.5
        )
        await working.add(item_a)
        await working.add(item_b)

        # Wrong session: must not delete item B.
        await working.remove([item_b.id], user_id=user, session_id="sess_a")
        assert len(await working.list_items(user, session_id="sess_b")) == 1

        # Correct session: deletes it.
        await working.remove([item_b.id], user_id=user, session_id="sess_b")
        assert await working.list_items(user, session_id="sess_b") == []

    @pytest.mark.asyncio
    async def test_episodic_remove_respects_user_scope(self, memory_bundle):
        _, _, episodic, _ = memory_bundle
        item = MemoryItem(
            user_id="owner", session_id="s1", type="episodic", content="用户记忆内容", importance=0.8
        )
        await episodic.add(item)

        # Other user cannot delete it.
        await episodic.remove([item.id], user_id="intruder")
        assert len(await episodic.list_items("owner")) == 1

        await episodic.remove([item.id], user_id="owner")
        assert await episodic.list_items("owner") == []
