"""Adversarial Empirical Challenge Test Suite for Milestone M2: Backend Memory Layered Isolation.

Authored by Empirical Challenger 1.
Empirically stress-tests:
1. High-concurrency async writes across multiple sessions of the same user.
2. Capacity flooding in session A ensuring session B retains its oldest items un-evicted.
3. Boundary session_ids: None, empty string "", spaces, whitespace, unicode with emojis, 256-character IDs.
4. Retrieval isolation: verifying 0 items returned across session boundaries for scratchpad tools.
5. TTL expiration and capacity eviction interplay.
6. Cross-user session isolation with identical session IDs.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from app.llm.embedding import HashingEmbedder
from app.memory.base import MemoryItem
from app.memory.episodic import EpisodicMemory
from app.memory.manager import MemoryManager
from app.memory.working import WorkingMemory
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore
from app.tools.base import ToolContext
from app.tools.builtin.memory_tool import MemoryTool


@pytest.fixture
async def adv_m2_sqlite(tmp_path: Path):
    db_file = tmp_path / "adv_m2_challenger1.db"
    store = SQLiteStore(db_file)
    await store.initialize()
    yield store
    await store.aclose()


# ============================================================================
# 1. High-Concurrency Stress Across Multiple Sessions of the Same User
# ============================================================================

class TestAdvWorkingMemoryConcurrency:
    """Stress tests high-concurrency async writes, reads, and listings across sessions."""

    @pytest.mark.asyncio
    async def test_adv_concurrent_writes_multi_session(self):
        """20 concurrent async workers writing 600 items across 10 distinct sessions of the same user."""
        wm = WorkingMemory(capacity=100, ttl_seconds=3600)
        user_id = "user_concurrent_stress"
        num_sessions = 10
        items_per_session = 60
        sessions = [f"sess_{i}" for i in range(num_sessions)]

        async def worker_write(sess: str, start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                item = MemoryItem(
                    user_id=user_id,
                    session_id=sess,
                    type="working",
                    content=f"Concurrent payload session {sess} item {i} keyword_{sess}_{i}",
                    importance=0.5,
                )
                await wm.add(item)
                await asyncio.sleep(0.0001)

        async def worker_read(sess: str):
            for _ in range(10):
                await wm.retrieve(query=f"keyword_{sess}", user_id=user_id, top_k=5, session_id=sess)
                await wm.list_items(user_id=user_id, limit=20, session_id=sess)
                await asyncio.sleep(0.0002)

        # Launch concurrent writers: 2 writers per session = 20 writers, each writing 30 items
        tasks = []
        for sess in sessions:
            tasks.append(worker_write(sess, 0, 30))
            tasks.append(worker_write(sess, 30, 30))
            tasks.append(worker_read(sess))

        await asyncio.gather(*tasks)

        # Verification: inspect internal store and public APIs
        all_item_ids_by_session: dict[str, set[str]] = {}
        for sess in sessions:
            items = await wm.list_items(user_id=user_id, limit=100, session_id=sess)
            assert len(items) == items_per_session, (
                f"Session {sess} expected {items_per_session} items, got {len(items)}"
            )
            item_ids = {it.id for it in items}
            all_item_ids_by_session[sess] = item_ids

            # Verify every item in this session belongs exclusively to this session
            for it in items:
                assert it.session_id == sess
                assert it.user_id == user_id
                assert f"session {sess}" in it.content

        # Verify pairwise ID disjointness across all sessions (zero bleed)
        for i in range(num_sessions):
            for j in range(i + 1, num_sessions):
                sess_i = sessions[i]
                sess_j = sessions[j]
                overlap = all_item_ids_by_session[sess_i] & all_item_ids_by_session[sess_j]
                assert len(overlap) == 0, f"ID bleed detected between {sess_i} and {sess_j}: {overlap}"

    @pytest.mark.asyncio
    async def test_adv_concurrent_read_write_clear_race(self):
        """Simultaneous add, retrieve, list, and clear operations across sessions must not raise or corrupt state."""
        wm = WorkingMemory(capacity=20, ttl_seconds=3600)
        user_id = "user_race_test"

        async def writer(sess: str):
            for i in range(40):
                await wm.add(MemoryItem(
                    user_id=user_id,
                    session_id=sess,
                    type="working",
                    content=f"Race content {sess} #{i}",
                    importance=0.7,
                ))

        async def reader(sess: str):
            for _ in range(40):
                await wm.retrieve(query="Race", user_id=user_id, top_k=5, session_id=sess)
                await wm.list_items(user_id=user_id, limit=10, session_id=sess)

        async def clearer(sess: str):
            for _ in range(5):
                await asyncio.sleep(0.001)
                await wm.clear(user_id=user_id, session_id=sess)

        # Run concurrent readers, writers, and clearers on different sessions
        await asyncio.gather(
            writer("sess_active_1"),
            writer("sess_active_2"),
            writer("sess_cleared"),
            reader("sess_active_1"),
            reader("sess_active_2"),
            reader("sess_cleared"),
            clearer("sess_cleared"),
        )

        # Active session 1 must still have its items up to capacity
        items_1 = await wm.list_items(user_id=user_id, session_id="sess_active_1")
        assert len(items_1) == 20
        for it in items_1:
            assert it.session_id == "sess_active_1"


# ============================================================================
# 2. Capacity Flooding & Eviction Independence
# ============================================================================

class TestAdvCapacityFlooding:
    """Stress tests capacity eviction per session and verifies session B remains intact."""

    @pytest.mark.asyncio
    async def test_adv_flooding_session_a_preserves_session_b_oldest(self):
        """Capacity flooding in session A ensures session B retains its oldest items un-evicted."""
        wm = WorkingMemory(capacity=5, ttl_seconds=3600)
        user_id = "user_flood_test"

        # 1. Session B writes 5 items
        oldest_b_item = MemoryItem(
            user_id=user_id,
            session_id="session_B",
            type="working",
            content="Session B oldest pristine anchor: determinant value is 42",
            importance=0.9,
        )
        await wm.add(oldest_b_item)

        for i in range(1, 5):
            await wm.add(MemoryItem(
                user_id=user_id,
                session_id="session_B",
                type="working",
                content=f"Session B intermediate state #{i}",
                importance=0.5,
            ))

        items_b_before = await wm.list_items(user_id=user_id, session_id="session_B")
        assert len(items_b_before) == 5
        assert any(it.id == oldest_b_item.id for it in items_b_before)

        # 2. Session A now floods 100 items (20x the capacity of 5)
        for i in range(100):
            await wm.add(MemoryItem(
                user_id=user_id,
                session_id="session_A",
                type="working",
                content=f"Session A flood payload #{i} matrix determinant calculation",
                importance=0.4,
            ))

        # 3. Assert Session A has strictly capped at 5 items (oldest 95 evicted)
        items_a_after = await wm.list_items(user_id=user_id, session_id="session_A")
        assert len(items_a_after) == 5
        # The remaining items in Session A should be items 95..99
        for it in items_a_after:
            assert it.session_id == "session_A"
            assert "Session A flood payload" in it.content

        # 4. Assert Session B still retains ALL 5 items, including its oldest item!
        items_b_after = await wm.list_items(user_id=user_id, session_id="session_B")
        assert len(items_b_after) == 5
        b_ids = [it.id for it in items_b_after]
        assert oldest_b_item.id in b_ids, "Oldest item in session B was unexpectedly evicted by session A flood!"

        # 5. Retrieval isolation: querying Session B for 'determinant' must return oldest_b_item
        res_b = await wm.retrieve(query="determinant", user_id=user_id, top_k=5, session_id="session_B")
        assert len(res_b) >= 1
        assert res_b[0].item.id == oldest_b_item.id

        # 6. Querying Session A for 'pristine anchor' must return 0 items
        res_a = await wm.retrieve(query="pristine anchor", user_id=user_id, top_k=5, session_id="session_A")
        assert len(res_a) == 0

    @pytest.mark.asyncio
    async def test_adv_fifo_eviction_strictly_in_session(self):
        """Validates that FIFO eviction operates strictly in insertion order within the session."""
        wm = WorkingMemory(capacity=3, ttl_seconds=3600)
        user_id = "user_fifo"
        sess = "session_fifo"

        item1 = MemoryItem(user_id=user_id, session_id=sess, type="working", content="Item 1 alpha", importance=0.5)
        item2 = MemoryItem(user_id=user_id, session_id=sess, type="working", content="Item 2 beta", importance=0.5)
        item3 = MemoryItem(user_id=user_id, session_id=sess, type="working", content="Item 3 gamma", importance=0.5)
        await wm.add(item1)
        await wm.add(item2)
        await wm.add(item3)

        # Capacity full (3)
        items = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items) == 3

        # Add 4th item -> item 1 should be evicted
        item4 = MemoryItem(user_id=user_id, session_id=sess, type="working", content="Item 4 delta", importance=0.5)
        await wm.add(item4)

        items_after = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items_after) == 3
        ids = [it.id for it in items_after]
        assert item1.id not in ids, "Item 1 was not evicted under FIFO"
        assert item2.id in ids
        assert item3.id in ids
        assert item4.id in ids


# ============================================================================
# 3. TTL Expiration & Capacity Interplay
# ============================================================================

class TestAdvTtlExpiration:
    """Stress tests TTL expiration and its interaction with capacity eviction."""

    @pytest.mark.asyncio
    async def test_adv_ttl_expired_items_purged_before_capacity(self):
        """Expired items are purged before capacity check, preventing premature eviction of active items."""
        wm = WorkingMemory(capacity=3, ttl_seconds=1)
        user_id = "user_ttl_cap"
        sess = "session_ttl"

        # Add 3 items
        await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Old item 1", importance=0.5))
        await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Old item 2", importance=0.5))
        await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Old item 3", importance=0.5))

        # Wait for TTL expiration
        await asyncio.sleep(1.1)

        # Now add 1 new fresh item
        fresh = MemoryItem(user_id=user_id, session_id=sess, type="working", content="Fresh brand new item", importance=0.8)
        await wm.add(fresh)

        # Verify only the fresh item exists in the store
        items = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items) == 1
        assert items[0].id == fresh.id
        assert items[0].content == "Fresh brand new item"

        # Retrieval for 'item' returns only fresh item
        res = await wm.retrieve(query="item", user_id=user_id, top_k=5, session_id=sess)
        assert len(res) == 1
        assert res[0].item.id == fresh.id

    @pytest.mark.asyncio
    async def test_adv_ttl_cross_session_expiration_independence(self):
        """Expiration in session A must not purge unexpired items in session B."""
        wm = WorkingMemory(capacity=10, ttl_seconds=1)
        user_id = "user_ttl_indep"

        # Session A items with created_at set in the past (expired)
        past_time = time.time() - 100
        await wm.add(MemoryItem(
            user_id=user_id,
            session_id="session_A",
            type="working",
            content="Session A past item",
            importance=0.5,
            created_at=past_time,
        ))

        # Session B items with current timestamp (fresh)
        fresh_b = MemoryItem(
            user_id=user_id,
            session_id="session_B",
            type="working",
            content="Session B fresh item",
            importance=0.8,
            created_at=time.time(),
        )
        await wm.add(fresh_b)

        # Query session A -> expired item purged, 0 items returned
        res_a = await wm.retrieve(query="past", user_id=user_id, session_id="session_A")
        assert len(res_a) == 0

        # Query session B -> fresh item still intact
        res_b = await wm.retrieve(query="fresh", user_id=user_id, session_id="session_B")
        assert len(res_b) == 1
        assert res_b[0].item.id == fresh_b.id


# ============================================================================
# 4. Boundary Session IDs
# ============================================================================

class TestAdvBoundarySessionIds:
    """Stress tests boundary session_ids: None, empty string, spaces, unicode, 256-character IDs."""

    @pytest.mark.asyncio
    async def test_adv_boundary_session_id_variants(self):
        """Comprehensive verification across boundary session_id variants."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        user_id = "user_boundaries"

        # Boundary IDs to test
        unicode_id = "会话_微积分_📐_Integral_∑dx_π"
        long_256_id = "s" * 256
        special_chars_id = "sess;DROP TABLE sessions;--!@#$%^&*()_+"

        # 1. Add items under distinct boundary IDs
        item_unicode = MemoryItem(
            user_id=user_id,
            session_id=unicode_id,
            type="working",
            content="Unicode session calculus formula proof",
            importance=0.8,
        )
        item_long = MemoryItem(
            user_id=user_id,
            session_id=long_256_id,
            type="working",
            content="Extremely long 256 session payload string",
            importance=0.7,
        )
        item_special = MemoryItem(
            user_id=user_id,
            session_id=special_chars_id,
            type="working",
            content="SQL injection attempt and special symbols safe storage",
            importance=0.9,
        )

        await wm.add(item_unicode)
        await wm.add(item_long)
        await wm.add(item_special)

        # 2. Verify retrieval matching
        res_u = await wm.retrieve("calculus", user_id=user_id, session_id=unicode_id)
        assert len(res_u) == 1
        assert res_u[0].item.id == item_unicode.id

        res_l = await wm.retrieve("payload", user_id=user_id, session_id=long_256_id)
        assert len(res_l) == 1
        assert res_l[0].item.id == item_long.id

        res_s = await wm.retrieve("injection", user_id=user_id, session_id=special_chars_id)
        assert len(res_s) == 1
        assert res_s[0].item.id == item_special.id

        # 3. Cross-session isolation between boundary IDs
        # Querying unicode session for long_256 content must return 0
        cross_res_1 = await wm.retrieve("payload", user_id=user_id, session_id=unicode_id)
        assert len(cross_res_1) == 0

        # Querying 256-char session for SQL injection content must return 0
        cross_res_2 = await wm.retrieve("injection", user_id=user_id, session_id=long_256_id)
        assert len(cross_res_2) == 0

        # 4. Off-by-one 255-char session ID must return 0
        res_off_by_one = await wm.retrieve("payload", user_id=user_id, session_id="s" * 255)
        assert len(res_off_by_one) == 0

    @pytest.mark.asyncio
    async def test_adv_boundary_none_empty_and_spaces_normalization(self):
        """Tests None, empty string, and whitespace string session_ids."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        user_id = "user_null_spaces"

        # Item with None session_id
        item_none = MemoryItem(
            user_id=user_id,
            session_id=None,
            type="working",
            content="Unsessioned None item content keyword_target",
            importance=0.6,
        )
        await wm.add(item_none)

        # Retrieve with None session_id
        res_none = await wm.retrieve("keyword_target", user_id=user_id, session_id=None)
        assert len(res_none) == 1
        assert res_none[0].item.id == item_none.id

        # Retrieve with empty string "" session_id (normalized to None)
        res_empty = await wm.retrieve("keyword_target", user_id=user_id, session_id="")
        assert len(res_empty) == 1
        assert res_empty[0].item.id == item_none.id

        # Retrieve with whitespace "   \t\n" session_id (normalized to None)
        res_ws = await wm.retrieve("keyword_target", user_id=user_id, session_id="   \t\n")
        assert len(res_ws) == 1
        assert res_ws[0].item.id == item_none.id

        # But querying with an explicit session "session_explicit" MUST return 0 items!
        res_explicit = await wm.retrieve("keyword_target", user_id=user_id, session_id="session_explicit")
        assert len(res_explicit) == 0, "Explicit session must NOT see unsessioned None memories!"

    @pytest.mark.asyncio
    async def test_adv_same_session_id_different_users(self):
        """Different users sharing the identical session_id string must be completely isolated."""
        wm = WorkingMemory(capacity=3, ttl_seconds=3600)
        shared_session = "course_algebra_common_session"

        # User 1 writes 3 items
        await wm.add(MemoryItem(user_id="user_1", session_id=shared_session, type="working", content="User 1 private secret", importance=0.8))
        await wm.add(MemoryItem(user_id="user_1", session_id=shared_session, type="working", content="User 1 item 2", importance=0.5))
        await wm.add(MemoryItem(user_id="user_1", session_id=shared_session, type="working", content="User 1 item 3", importance=0.5))

        # User 2 writes 1 item
        user2_item = MemoryItem(user_id="user_2", session_id=shared_session, type="working", content="User 2 distinct data", importance=0.9)
        await wm.add(user2_item)

        # User 1 floods 20 items into shared_session
        for i in range(20):
            await wm.add(MemoryItem(user_id="user_1", session_id=shared_session, type="working", content=f"User 1 flood {i}", importance=0.3))

        # User 2's item must remain 100% intact
        u2_items = await wm.list_items(user_id="user_2", session_id=shared_session)
        assert len(u2_items) == 1
        assert u2_items[0].id == user2_item.id

        # User 2 searching for 'secret' must return 0
        u2_search = await wm.retrieve("secret", user_id="user_2", session_id=shared_session)
        assert len(u2_search) == 0


# ============================================================================
# 5. Retrieval Isolation & Scratchpad Tool Cross-Session Zero-Leakage
# ============================================================================

class TestAdvScratchpadToolIsolation:
    """Stress tests MemoryTool scratchpad isolation across session boundaries."""

    @pytest.mark.asyncio
    async def test_adv_scratchpad_tool_cross_session_zero_items(self, adv_m2_sqlite: SQLiteStore):
        """Scratchpad tool execution output stored in Session A returns EXACTLY 0 items in Session B."""
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        em = EpisodicMemory(sqlite=adv_m2_sqlite, vector_store=vstore, embedder=embedder)
        mem_mgr = MemoryManager(working=wm, episodic=em)

        user_id = "student_coder"
        sess_a = "coding_session_alpha"
        sess_b = "coding_session_beta"

        # 1. Simulate Agent tool execution in Session A writing to working memory
        tool_scratchpad_content = (
            "执行工具 python_runner 结果: SECRET_DATABASE_AUTH_TOKEN_XY998811 "
            "optimization matrix determinant: 3.1415926535"
        )
        await mem_mgr.add(
            type="working",
            content=tool_scratchpad_content,
            user_id=user_id,
            session_id=sess_a,
            importance=0.6,
        )

        tool = MemoryTool(memory_manager=mem_mgr)

        # 2. Session B invokes memory_tool to search for exact secret keyword from Session A
        ctx_b = ToolContext(user_id=user_id, session_id=sess_b)
        result_b = await tool.run(
            arguments={"action": "search", "query": "SECRET_DATABASE_AUTH_TOKEN_XY998811"},
            context=ctx_b,
        )

        assert result_b.ok is True
        assert result_b.output == "未检索到相关记忆。", (
            f"Scratchpad memory leaked to session B! Got: {result_b.output}"
        )

        # Search for 'determinant' in Session B
        result_b_math = await tool.run(
            arguments={"action": "search", "query": "determinant"},
            context=ctx_b,
        )
        assert result_b_math.ok is True
        assert result_b_math.output == "未检索到相关记忆。"

        # 3. Session A invokes memory_tool: MUST retrieve its own scratchpad memory
        ctx_a = ToolContext(user_id=user_id, session_id=sess_a)
        result_a = await tool.run(
            arguments={"action": "search", "query": "SECRET_DATABASE_AUTH_TOKEN_XY998811"},
            context=ctx_a,
        )
        assert result_a.ok is True
        assert "SECRET_DATABASE_AUTH_TOKEN_XY998811" in result_a.output
        assert "[WORKING]" in result_a.output

        # 4. Direct WorkingMemory.retrieve validation: verifying 0 items returned across boundary
        direct_cross = await wm.retrieve(
            query="SECRET_DATABASE_AUTH_TOKEN_XY998811",
            user_id=user_id,
            top_k=5,
            session_id=sess_b,
        )
        assert len(direct_cross) == 0, "Direct WorkingMemory.retrieve leaked items across session boundary!"

    @pytest.mark.asyncio
    async def test_adv_memory_manager_retrieve_all_multi_query_isolation(self, adv_m2_sqlite: SQLiteStore):
        """MemoryManager.retrieve_all with multiple queries across session boundaries returns 0 working items."""
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        em = EpisodicMemory(sqlite=adv_m2_sqlite, vector_store=vstore, embedder=embedder)
        mem_mgr = MemoryManager(working=wm, episodic=em)

        user_id = "u_multi_query"
        sess_1 = "sess_1_crypto"
        sess_2 = "sess_2_history"

        # Session 1 scratchpad tool states
        await mem_mgr.add(
            type="working",
            content="执行工具 crypto_tool: RSA-4096 public modulus generated successfully",
            user_id=user_id,
            session_id=sess_1,
            importance=0.6,
        )

        # Session 2 multi-query retrieve_all targeting Session 1's terms
        queries = ["RSA-4096", "public modulus", "crypto_tool"]
        results = await mem_mgr.retrieve_all(
            queries=queries,
            user_id=user_id,
            top_k=10,
            session_id=sess_2,
        )

        # Zero working items must be returned
        working_items = [r for r in results if r.item.type == "working"]
        assert len(working_items) == 0, f"retrieve_all leaked working items: {working_items}"


# ============================================================================
# 6. Additional Adversarial Edge Cases: Clear Isolation & Extreme Capacity
# ============================================================================

class TestAdvClearAndExtremeCapacity:
    """Stress tests clear isolation across sessions and extreme capacity limits."""

    @pytest.mark.asyncio
    async def test_adv_clear_scoped_to_session(self):
        """wm.clear(user_id, session_id) clears only target session, other sessions unpurged."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        user_id = "user_clear_test"

        await wm.add(MemoryItem(user_id=user_id, session_id="sess_1", type="working", content="Item in sess 1", importance=0.5))
        await wm.add(MemoryItem(user_id=user_id, session_id="sess_2", type="working", content="Item in sess 2", importance=0.5))

        # Clear only sess_1
        await wm.clear(user_id=user_id, session_id="sess_1")

        # sess_1 must be empty
        s1_items = await wm.list_items(user_id=user_id, session_id="sess_1")
        assert len(s1_items) == 0

        # sess_2 must remain intact
        s2_items = await wm.list_items(user_id=user_id, session_id="sess_2")
        assert len(s2_items) == 1
        assert s2_items[0].content == "Item in sess 2"

    @pytest.mark.asyncio
    async def test_adv_extreme_capacity_singleton(self):
        """WorkingMemory(capacity=1) strictly maintains a single-item buffer with FIFO eviction."""
        wm = WorkingMemory(capacity=1, ttl_seconds=3600)
        user_id = "user_singleton"
        sess = "sess_single"

        # Add item 1
        id1 = await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Single item 1", importance=0.5))
        items1 = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items1) == 1
        assert items1[0].id == id1

        # Add item 2 -> item 1 must be evicted
        id2 = await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Single item 2", importance=0.5))
        items2 = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items2) == 1
        assert items2[0].id == id2

        # Add item 3 -> item 2 must be evicted
        id3 = await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="Single item 3", importance=0.5))
        items3 = await wm.list_items(user_id=user_id, session_id=sess)
        assert len(items3) == 1
        assert items3[0].id == id3

    @pytest.mark.asyncio
    async def test_adv_empty_and_single_char_query_resilience(self):
        """Querying with empty text or single punctuation/characters must not raise exceptions."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        user_id = "user_safe_query"
        sess = "sess_safe"

        await wm.add(MemoryItem(user_id=user_id, session_id=sess, type="working", content="正常工作记忆数据", importance=0.5))

        # Test safe empty queries
        assert await wm.retrieve("", user_id=user_id, session_id=sess) == []
        assert await wm.retrieve("   ", user_id=user_id, session_id=sess) == []
        assert await wm.retrieve("!", user_id=user_id, session_id=sess) == []
        assert await wm.retrieve("a", user_id=user_id, session_id=sess) == []
        # Multi-char query should retrieve
        res = await wm.retrieve("工作记忆", user_id=user_id, session_id=sess)
        assert len(res) == 1

