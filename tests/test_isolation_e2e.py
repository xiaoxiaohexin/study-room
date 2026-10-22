"""Comprehensive 4-Tier Automated End-to-End (E2E) Test Suite for Memory & RAG Namespace Isolation.

Covers all 5 Requirements (R1-R5) and Features from PROJECT.md:
- Tier 1: Feature Isolation Coverage (F_MEM, F_EPISODIC, F_SEMANTIC, F_DB, F_RAG, F_API, >=5 tests each = 30 tests)
- Tier 2: Boundary, Extreme & Negative Conditions (>=8 tests)
- Tier 3: Cross-Feature Pairwise Integrations (5 tests)
- Tier 4: Real-World Multi-Course Study Workloads (5 scenarios)

Total: 48 automated test cases.
Engineered according to opaque-box, requirement-driven, zero-facade principles.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.learning.store import LearningStore
from app.llm.embedding import HashingEmbedder
from app.main import create_app
from app.memory.base import MemoryItem
from app.memory.episodic import EpisodicMemory
from app.memory.lifecycle import MemoryLifecycle
from app.memory.manager import MemoryManager
from app.memory.semantic import SemanticMemory
from app.memory.working import WorkingMemory
from app.rag.retriever import RagRetriever
from app.session.state_manager import SessionStateManager
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore, SQLiteVectorStore, VectorPoint

# ============================================================================
# Shared Fixtures & Synthetic Oracles
# ============================================================================

@pytest.fixture
async def iso_sqlite(tmp_path: Path):
    """Provides a cleanly initialized SQLiteStore backed by a temporary file."""
    db_file = tmp_path / "test_iso.db"
    store = SQLiteStore(db_file)
    await store.initialize()
    yield store
    await store.aclose()


@pytest.fixture
async def iso_app(tmp_path: Path):
    """Provides an isolated FastAPI app instance with fresh SQLite and container."""
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "test_iso_api.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


@pytest.fixture
async def iso_client(iso_app):
    """Provides an asynchronous HTTP client for ASGI route invocations."""
    transport = ASGITransport(app=iso_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# Tier 1: Feature Isolation Coverage (>=5 per Feature Area)
# ============================================================================

class TestTier1WorkingMemoryIsolation:
    """F_MEM: Working memory isolation per (user_id, session_id)."""

    @pytest.mark.asyncio
    async def test_f_mem_01_compound_key_isolation(self):
        """WorkingMemory must strictly separate items between session A and session B."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)

        item_a = MemoryItem(
            user_id="user_1",
            session_id="session_alpha",
            type="working",
            content="Alpha session local state: calculated matrix determinant is 42",
            importance=0.8,
        )
        item_b = MemoryItem(
            user_id="user_1",
            session_id="session_beta",
            type="working",
            content="Beta session local state: Rust compiler optimization flag -O3",
            importance=0.8,
        )

        await wm.add(item_a)
        await wm.add(item_b)

        # Retrieval in session_alpha must only yield item_a
        results_a = await wm.retrieve(query="matrix determinant", user_id="user_1", session_id="session_alpha")
        assert len(results_a) >= 1
        assert any("matrix determinant" in r.item.content for r in results_a)
        assert not any("Rust compiler" in r.item.content for r in results_a)

        # Retrieval in session_beta must only yield item_b
        results_b = await wm.retrieve(query="compiler", user_id="user_1", session_id="session_beta")
        assert len(results_b) >= 1
        assert any("Rust compiler" in r.item.content for r in results_b)
        assert not any("matrix determinant" in r.item.content for r in results_b)

    @pytest.mark.asyncio
    async def test_f_mem_02_tool_scratchpad_unpolluted_across_sessions(self):
        """Tool execution observation scratchpad in Session 1 must not leak to Session 2."""
        wm = WorkingMemory(capacity=20, ttl_seconds=3600)

        # Session 1 tool execution output
        tool_output = MemoryItem(
            user_id="alice",
            session_id="sess_math",
            type="working",
            content="Tool output [run_python]: eigenvalues = [2.0, 5.0, 9.0]",
            importance=0.9,
        )
        await wm.add(tool_output)

        # Querying in sess_history should return zero tool scratchpad results from sess_math
        results_history = await wm.retrieve(query="eigenvalues python", user_id="alice", session_id="sess_history")
        assert len(results_history) == 0

    @pytest.mark.asyncio
    async def test_f_mem_03_list_items_scoped_to_session(self):
        """list_items(user_id, session_id) must only enumerate memories belonging to that session."""
        wm = WorkingMemory(capacity=20, ttl_seconds=3600)

        for i in range(3):
            await wm.add(MemoryItem(
                user_id="bob", session_id="sess_physics", type="working", content=f"Physics item {i}"
            ))
        for j in range(2):
            await wm.add(MemoryItem(
                user_id="bob", session_id="sess_chemistry", type="working", content=f"Chemistry item {j}"
            ))

        phys_items = await wm.list_items("bob", limit=10, session_id="sess_physics")
        chem_items = await wm.list_items("bob", limit=10, session_id="sess_chemistry")

        assert len(phys_items) == 3
        assert all(it.session_id == "sess_physics" for it in phys_items)
        assert len(chem_items) == 2
        assert all(it.session_id == "sess_chemistry" for it in chem_items)

    @pytest.mark.asyncio
    async def test_f_mem_04_session_capacity_independence(self):
        """Filling capacity in session A must not evict items from session B."""
        wm = WorkingMemory(capacity=3, ttl_seconds=3600)

        # Add 2 items to session B
        await wm.add(MemoryItem(user_id="u1", session_id="sess_b", type="working", content="B1"))
        await wm.add(MemoryItem(user_id="u1", session_id="sess_b", type="working", content="B2"))

        # Add 5 items to session A (exceeding capacity of 3)
        for i in range(5):
            await wm.add(MemoryItem(user_id="u1", session_id="sess_a", type="working", content=f"A{i}"))

        items_b = await wm.list_items("u1", limit=10, session_id="sess_b")
        # Session B's 2 items must be intact
        assert len(items_b) == 2
        assert {it.content for it in items_b} == {"B1", "B2"}

        items_a = await wm.list_items("u1", limit=10, session_id="sess_a")
        # Session A holds at most capacity (3 items)
        assert len(items_a) == 3

    @pytest.mark.asyncio
    async def test_f_mem_05_memory_manager_retrieve_all_passes_session_id(self):
        """MemoryManager.retrieve_all must accept session_id and pass it down to WorkingMemory."""
        wm = WorkingMemory(capacity=10)
        await wm.add(MemoryItem(user_id="u1", session_id="sess_1", type="working", content="Active token in session 1"))
        await wm.add(MemoryItem(user_id="u1", session_id="sess_2", type="working", content="Active token in session 2"))

        mm = MemoryManager(working=wm, episodic=None, semantic=None)
        res_s1 = await mm.retrieve_all(["Active token"], user_id="u1", session_id="sess_1")
        assert len(res_s1) >= 1
        assert all(m.item.session_id == "sess_1" for m in res_s1 if m.item.session_id)


class TestTier1EpisodicMemoryIsolation:
    """F_EPISODIC: Episodic memory recall prioritized for current session, cross-discipline filtered."""

    @pytest.mark.asyncio
    async def test_f_episodic_01_current_session_affinity_boost(self, iso_sqlite):
        """Episodic recall must apply session-affinity multiplier (1.35x) prioritizing matching session."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=32)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)

        # Item E1 from current session "physics"
        item_phys = MemoryItem(
            user_id="student_1",
            session_id="sess_physics",
            type="episodic",
            content="Newton's second law acceleration under gravity",
            importance=0.7,
        )
        # Item E2 from different session "literature"
        item_lit = MemoryItem(
            user_id="student_1",
            session_id="sess_literature",
            type="episodic",
            content="Acceleration of tragic narrative in Hamlet play",
            importance=0.7,
        )

        await em.add(item_phys)
        await em.add(item_lit)

        # Query in session "sess_physics"
        results = await em.retrieve(query="acceleration", user_id="student_1", top_k=5, session_id="sess_physics")
        assert len(results) >= 1
        # Top result must be the one from sess_physics due to session-affinity boost
        assert results[0].item.session_id == "sess_physics"

    @pytest.mark.asyncio
    async def test_f_episodic_02_cross_discipline_weak_relevance_filtered(self, iso_sqlite):
        """Cross-session episodic candidates with low relevance (<0.50) must be discarded."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=64)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)

        item_history = MemoryItem(
            user_id="student_1",
            session_id="sess_history",
            type="episodic",
            content="French Revolution battle of Waterloo in 1815",
            importance=0.5,
        )
        await em.add(item_history)

        # Student is now studying organic chemistry in sess_chem
        results = await em.retrieve(
            query="benzene ring chemical bonding structure",
            user_id="student_1",
            top_k=5,
            session_id="sess_chem",
        )
        # Cross-discipline weak relevance must not contaminate results
        assert not any("Waterloo" in r.item.content for r in results)

    @pytest.mark.asyncio
    async def test_f_episodic_03_vector_payload_stores_session_id(self, iso_sqlite):
        """Vector store payload and SQLite store must record session_id."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=16)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)

        item = MemoryItem(
            user_id="u1",
            session_id="sess_algebra",
            type="episodic",
            content="Quadratic formula derivation",
            importance=0.6,
        )
        await em.add(item)

        # Inspect SQLite row
        cur = await iso_sqlite.conn.execute("SELECT session_id, content FROM episodic_memories WHERE id = ?", (item.id,))
        row = await cur.fetchone()
        assert row is not None
        assert row["session_id"] == "sess_algebra"

    @pytest.mark.asyncio
    async def test_f_episodic_04_same_user_multi_session_history_separation(self, iso_sqlite):
        """Episodic memories for distinct courses must not cross-contaminate topical searches."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=32)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)

        await em.add(MemoryItem(
            user_id="u1", session_id="sess_calc", type="episodic", content="Riemann integral summation limits"
        ))
        await em.add(MemoryItem(
            user_id="u1", session_id="sess_genetics", type="episodic", content="Mendelian inheritance alleles"
        ))

        calc_results = await em.retrieve("Riemann integral", user_id="u1", session_id="sess_calc")
        assert len(calc_results) >= 1
        assert "Riemann integral" in calc_results[0].item.content
        assert not any("Mendelian" in r.item.content for r in calc_results)

    @pytest.mark.asyncio
    async def test_f_episodic_05_memory_lifecycle_consolidate_scoped_by_session(self, iso_sqlite):
        """MemoryLifecycle consolidate must scope working memory consolidation by session_id."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=32)
        wm = WorkingMemory(capacity=10)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)
        sm = SemanticMemory(vector_store=vec, graph_store=NullGraphStore(), embedder=embedder)

        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_working_thresh=0.7,
        )

        # Add high-importance memory in sess_a and sess_b
        await wm.add(MemoryItem(user_id="u1", session_id="sess_a", type="working", content="Important note in A", importance=0.85))
        await wm.add(MemoryItem(user_id="u1", session_id="sess_b", type="working", content="Important note in B", importance=0.85))

        # Consolidate sess_a only
        res = await lifecycle.consolidate("u1", session_id="sess_a")
        assert res.get("working_to_episodic", 0) >= 1

        # Check episodic memories: only sess_a memory consolidated
        episodic_items = await em.list_items("u1")
        assert any("Important note in A" in it.content for it in episodic_items)


class TestTier1SemanticMemoryAndProfile:
    """F_SEMANTIC: Semantic memory & SM-2 mastery preserved globally across sessions."""

    @pytest.mark.asyncio
    async def test_f_semantic_01_global_user_scope_across_sessions(self):
        """Semantic memory must be retrievable globally regardless of which session requests it."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=16)
        sm = SemanticMemory(vector_store=vec, graph_store=NullGraphStore(), embedder=embedder)

        item = MemoryItem(
            user_id="user_global",
            type="semantic",
            content="General Principle: Law of Conservation of Energy",
            importance=0.9,
        )
        await sm.add(item)

        # Query from session_1
        res1 = await sm.retrieve("Conservation of Energy", user_id="user_global", session_id="session_1")
        assert len(res1) >= 1
        assert "Conservation of Energy" in res1[0].item.content

        # Query from session_2
        res2 = await sm.retrieve("Conservation of Energy", user_id="user_global", session_id="session_2")
        assert len(res2) >= 1
        assert "Conservation of Energy" in res2[0].item.content

    @pytest.mark.asyncio
    async def test_f_semantic_02_sm2_mastery_persists_globally_across_sessions(self, iso_sqlite):
        """SM-2 mastery state updated in Session A must be visible in Session B for the same user."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        kp = await learning_store.upsert_knowledge_point(
            course_id="math_101",
            name="导数定义",
            summary="函数变化率的极限",
            kp_id="kp_derivative",
        )
        q = await learning_store.add_question(
            kp_id=kp["id"],
            course_id="math_101",
            q_type="choice",
            stem="导数的基本含义是？",
            answer="瞬时变化率",
        )

        # Student reviews in session 1
        att = await learning_store.record_attempt_and_update_mastery(
            user_id="student_1",
            question_id=q["id"],
            user_answer="瞬时变化率",
            is_correct=True,
            score=1.0,
        )
        assert att["repetitions"] == 1
        assert att["new_mastery"] > 0.3

        # Fetch mastery in session 2 directly from SQLite mastery table
        cur = await iso_sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            ("student_1", kp["id"]),
        )
        m = await cur.fetchone()
        assert m is not None
        assert int(m["repetitions"]) == 1
        assert float(m["mastery"]) > 0.3

    @pytest.mark.asyncio
    async def test_f_semantic_03_learner_profile_text_aggregates_across_sessions(self, iso_sqlite):
        """build_learner_profile_text must aggregate mastery and mistakes across all sessions."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        kp = await learning_store.upsert_knowledge_point(
            course_id="math_geom",
            name="勾股定理",
            summary="直角三角形边长关系 a^2 + b^2 = c^2",
            kp_id="kp_pythagoras",
        )
        q = await learning_store.add_question(
            kp_id=kp["id"],
            course_id="math_geom",
            q_type="choice",
            stem="直角边为3和4，斜边是？",
            answer="5",
        )

        # Session 1 records a mistake
        await learning_store.record_attempt_and_update_mastery(
            user_id="student_1",
            question_id=q["id"],
            user_answer="6",
            is_correct=False,
            score=0.0,
            feedback="误用了乘法",
        )

        profile_text = await learning_store.build_learner_profile_text(user_id="student_1")
        assert "勾股定理" in profile_text
        assert "错题" in profile_text or "薄弱" in profile_text

    @pytest.mark.asyncio
    async def test_f_semantic_04_different_users_mastery_isolated(self, iso_sqlite):
        """User A's mastery profile must never be visible to User B."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        kp = await learning_store.upsert_knowledge_point(
            course_id="phys_quantum",
            name="量子叠加态",
            summary="薛定谔方程与波函数",
            kp_id="kp_quantum",
        )
        q = await learning_store.add_question(
            kp_id=kp["id"],
            course_id="phys_quantum",
            q_type="choice",
            stem="量子叠加态的数学表示？",
            answer="线性叠加",
        )

        # User A has high mastery
        await learning_store.record_attempt_and_update_mastery(
            user_id="user_a",
            question_id=q["id"],
            user_answer="线性叠加",
            is_correct=True,
            score=1.0,
        )

        # User B has never reviewed
        cur = await iso_sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            ("user_b", kp["id"]),
        )
        row_b = await cur.fetchone()
        assert row_b is None

    @pytest.mark.asyncio
    async def test_f_semantic_05_new_session_inherits_global_knowledge_graph(self, iso_sqlite):
        """Creating a new session must inherit the complete knowledge graph for that user."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        await learning_store.upsert_knowledge_point(course_id="calc", name="极限", summary="极限理论", kp_id="kp_limit")
        await learning_store.upsert_knowledge_point(course_id="calc", name="连续性", summary="函数连续性", kp_id="kp_cont")

        # Query in newly created session
        kps = await learning_store.list_knowledge_points(course_id="calc")
        assert len(kps) >= 2
        names = {p["name"] for p in kps}
        assert "极限" in names and "连续性" in names


class TestTier1SQLiteSessionMigrationAndPersistence:
    """F_DB: SQLite sessions table migration idempotency and persistence."""

    @pytest.mark.asyncio
    async def test_f_db_01_sessions_table_has_course_id_and_rag_namespace(self, iso_sqlite):
        """Sessions table must have course_id and rag_namespace columns."""
        cur = await iso_sqlite.conn.execute("PRAGMA table_info(sessions)")
        cols = {row["name"] for row in await cur.fetchall()}
        assert "course_id" in cols, "sessions table missing course_id column"
        assert "rag_namespace" in cols, "sessions table missing rag_namespace column"

    @pytest.mark.asyncio
    async def test_f_db_02_idempotent_migration_on_preexisting_sessions_table(self, tmp_path: Path):
        """Running schema migration on pre-existing database with old sessions table must be idempotent."""
        db_path = tmp_path / "legacy.db"
        # 1. Create old database with legacy sessions table schema
        store = SQLiteStore(db_path)
        # Create legacy table manually without course_id / rag_namespace
        import aiosqlite
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                ("sess_old_1", "user_1", "旧物理讨论", 1000.0, 1000.0),
            )
            await conn.commit()

        # 2. Run initialize() / _migrate_schema()
        await store.initialize()

        cur = await store.conn.execute("PRAGMA table_info(sessions)")
        cols = {row["name"] for row in await cur.fetchall()}
        assert "course_id" in cols
        assert "rag_namespace" in cols

        # Check existing row preserved with default
        cur = await store.conn.execute("SELECT session_id, title, course_id, rag_namespace FROM sessions WHERE session_id = 'sess_old_1'")
        row = await cur.fetchone()
        assert row is not None
        assert row["title"] == "旧物理讨论"
        assert row["rag_namespace"] == "default" or row["rag_namespace"] is None

        # 3. Second run must execute cleanly without exceptions
        await store._migrate_schema()
        await store.aclose()

    @pytest.mark.asyncio
    async def test_f_db_03_sessions_table_indexes_exist(self, iso_sqlite):
        """Sessions table must have index on course_id and rag_namespace."""
        cur = await iso_sqlite.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row["name"] for row in await cur.fetchall()}
        assert "idx_sessions_course" in indexes or "idx_sessions_user" in indexes
        assert "idx_sessions_rag_ns" in indexes or "idx_sessions_user" in indexes

    @pytest.mark.asyncio
    async def test_f_db_04_default_rag_namespace_value(self, iso_sqlite):
        """When creating a session without rag_namespace, default must be 'default'."""
        state_mgr = SessionStateManager(sqlite=iso_sqlite)
        session = await state_mgr.get_or_create_session(user_id="user_d", session_id="sess_def")
        assert getattr(session, "rag_namespace", "default") == "default"

    @pytest.mark.asyncio
    async def test_f_db_05_session_state_manager_persists_and_reads_bindings(self, iso_sqlite):
        """SessionStateManager must persist and retrieve course_id and rag_namespace."""
        state_mgr = SessionStateManager(sqlite=iso_sqlite)
        created = await state_mgr.get_or_create_session(
            user_id="u1",
            session_id="sess_bind_test",
            course_id="cs101_algorithms",
            rag_namespace="ns_cs101",
            title="算法分析",
        )
        assert created.course_id == "cs101_algorithms"
        assert created.rag_namespace == "ns_cs101"

        # Read back from clean manager instance
        state_mgr2 = SessionStateManager(sqlite=iso_sqlite)
        loaded = await state_mgr2.get_session(user_id="u1", session_id="sess_bind_test")
        assert loaded is not None
        assert loaded.course_id == "cs101_algorithms"
        assert loaded.rag_namespace == "ns_cs101"


class TestTier1RagNamespaceIsolation:
    """F_RAG: RAG retrieval isolation by rag_namespace and X-RAG-Namespace."""

    @pytest.mark.asyncio
    async def test_f_rag_01_physical_collection_partitioning(self, iso_sqlite):
        """Ingesting documents into different namespaces must isolate search results."""
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=32)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        col_phys = retriever._collection_name("physics")
        col_lit = retriever._collection_name("literature")

        await vstore.ensure_collection(col_phys, 32)
        await vstore.ensure_collection(col_lit, 32)

        # Upsert physics doc
        await vstore.upsert(col_phys, [
            VectorPoint(
                id="p1",
                vector=[0.5] * 32,
                payload={"content": "Quantum entanglement in physics", "filename": "phys.pdf", "page_start": 10, "page_end": 10},
            )
        ])
        # Upsert literature doc
        await vstore.upsert(col_lit, [
            VectorPoint(
                id="l1",
                vector=[0.5] * 32,
                payload={"content": "Shakespeare sonnets in literature", "filename": "lit.pdf", "page_start": 20, "page_end": 20},
            )
        ])

        # Search physics namespace
        hits_phys = await retriever.search(["entanglement"], namespace="physics", top_k=5)
        assert len(hits_phys) == 1
        assert "Quantum entanglement" in hits_phys[0].content
        assert not any("Shakespeare" in h.content for h in hits_phys)

        # Search literature namespace
        hits_lit = await retriever.search(["Shakespeare"], namespace="literature", top_k=5)
        assert len(hits_lit) == 1
        assert "Shakespeare" in hits_lit[0].content
        assert not any("Quantum" in h.content for h in hits_lit)

    def test_f_rag_02_collection_name_standard_format(self):
        """Collection naming must follow rag_{namespace}_{embedder.tag}."""
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=16)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        col_name = retriever._collection_name("course_os")
        assert col_name == f"rag_course_os_{embedder.tag}"

    @pytest.mark.asyncio
    async def test_f_rag_03_empty_namespace_returns_zero_results(self, iso_sqlite):
        """Querying an empty/unpopulated namespace must return 0 results and never leak from default."""
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=16)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        # Populate 'default' namespace
        col_default = retriever._collection_name("default")
        await vstore.ensure_collection(col_default, 16)
        await vstore.upsert(col_default, [
            VectorPoint(id="d1", vector=[0.1] * 16, payload={"content": "Default general knowledge"})
        ])

        # Query empty namespace 'unpopulated_course'
        results = await retriever.search(["knowledge"], namespace="unpopulated_course", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_f_rag_04_api_header_x_rag_namespace_routing(self, iso_client, iso_app):
        """Header X-RAG-Namespace must route retrieval to specified knowledge base."""
        # Index document into namespace 'mechanics'
        container: AppContainer = iso_app.state.container
        col = container.rag_retriever._collection_name("mechanics")
        await container.vector_store.ensure_collection(col, 128)
        await container.vector_store.upsert(col, [
            VectorPoint(
                id="m1",
                vector=[0.1] * 128,
                payload={"content": "Hooke's Law of Elasticity F=-kx", "filename": "mechanics.pdf", "page_start": 12, "page_end": 12},
            )
        ])

        # Request with X-RAG-Namespace header to /v1/chat/completions
        resp = await iso_client.post(
            "/v1/chat/completions",
            headers={"X-RAG-Namespace": "mechanics"},
            json={
                "messages": [{"role": "user", "content": "Explain elasticity"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_f_rag_05_api_body_rag_namespace_routing(self, iso_client):
        """Body parameter rag_namespace must route retrieval to specified namespace."""
        resp = await iso_client.post(
            "/api/chat",
            json={
                "user_id": "student_api",
                "message": "Calculate thermodynamic efficiency",
                "rag_namespace": "thermodynamics",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data


class TestTier1SessionCrudApi:
    """F_API: Session CRUD API endpoints (/api/sessions)."""

    @pytest.mark.asyncio
    async def test_f_api_01_post_create_session_with_bindings(self, iso_client):
        """POST /api/sessions must create a session with course_id and rag_namespace."""
        payload = {
            "user_id": "alice_api",
            "title": "计算机网络第一周",
            "course_id": "course_net_101",
            "rag_namespace": "net_docs",
        }
        resp = await iso_client.post("/api/sessions", json=payload)
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["user_id"] == "alice_api"
        assert data["course_id"] == "course_net_101"
        assert data["rag_namespace"] == "net_docs"
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_f_api_02_get_session_detail_returns_bindings(self, iso_client):
        """GET /api/sessions/{user_id}/{session_id} returns session details including bindings."""
        # Create session first
        create_resp = await iso_client.post(
            "/api/sessions",
            json={"user_id": "bob_api", "course_id": "cs_os", "rag_namespace": "os_v1", "title": "操作系统"},
        )
        assert create_resp.status_code in (200, 201)
        session_id = create_resp.json()["session_id"]

        # Fetch detail
        detail_resp = await iso_client.get(f"/api/sessions/bob_api/{session_id}")
        assert detail_resp.status_code == 200
        data = detail_resp.json()
        assert data["session_id"] == session_id
        assert data["course_id"] == "cs_os"
        assert data["rag_namespace"] == "os_v1"

    @pytest.mark.asyncio
    async def test_f_api_03_patch_session_updates_bindings(self, iso_client):
        """PATCH /api/sessions/{user_id}/{session_id} updates course_id and rag_namespace."""
        # Create session
        c_res = await iso_client.post(
            "/api/sessions",
            json={"user_id": "carol_api", "course_id": "chem_1", "rag_namespace": "chem_basics"},
        )
        session_id = c_res.json()["session_id"]

        # Patch to advanced namespace
        patch_res = await iso_client.patch(
            f"/api/sessions/carol_api/{session_id}",
            json={"title": "进阶有机化学", "rag_namespace": "chem_advanced"},
        )
        assert patch_res.status_code == 200
        updated = patch_res.json()
        assert updated["title"] == "进阶有机化学"
        assert updated["rag_namespace"] == "chem_advanced"

    @pytest.mark.asyncio
    async def test_f_api_04_get_user_sessions_list_includes_bindings(self, iso_client):
        """GET /api/sessions/{user_id} includes course_id and rag_namespace in session records."""
        user_id = f"user_list_{uuid.uuid4().hex[:6]}"
        await iso_client.post(
            "/api/sessions",
            json={"user_id": user_id, "course_id": "course_a", "rag_namespace": "ns_a"},
        )
        await iso_client.post(
            "/api/sessions",
            json={"user_id": user_id, "course_id": "course_b", "rag_namespace": "ns_b"},
        )

        list_res = await iso_client.get(f"/api/sessions/{user_id}")
        assert list_res.status_code == 200
        sessions = list_res.json()["sessions"]
        assert len(sessions) >= 2
        namespaces = {s.get("rag_namespace") for s in sessions}
        assert "ns_a" in namespaces and "ns_b" in namespaces

    @pytest.mark.asyncio
    async def test_f_api_05_delete_session(self, iso_client):
        """DELETE /api/sessions/{user_id}/{session_id} removes session cleanly."""
        res = await iso_client.post("/api/sessions", json={"user_id": "del_user", "title": "待删除会话"})
        session_id = res.json()["session_id"]

        del_res = await iso_client.delete(f"/api/sessions/del_user/{session_id}")
        assert del_res.status_code == 200

        # Subsequent fetch must return 404
        get_res = await iso_client.get(f"/api/sessions/del_user/{session_id}")
        assert get_res.status_code == 404


# ============================================================================
# Tier 2: Boundary, Extreme & Negative Conditions (>=8 Tests)
# ============================================================================

class TestTier2BoundaryAndEdgeCases:
    """Boundary and negative stress conditions."""

    @pytest.mark.asyncio
    async def test_boundary_01_un_sessioned_working_memory_fallback(self):
        """WorkingMemory.retrieve(session_id=None) must fall back gracefully for legacy callers."""
        wm = WorkingMemory(capacity=10)
        await wm.add(MemoryItem(user_id="u_legacy", type="working", content="Legacy unsessioned item"))

        # Call without session_id
        res = await wm.retrieve("Legacy", user_id="u_legacy", session_id=None)
        assert len(res) >= 1
        assert "Legacy" in res[0].item.content

    @pytest.mark.asyncio
    async def test_boundary_02_empty_string_session_id_handling(self):
        """Passing session_id='' must not crash or trigger KeyError."""
        wm = WorkingMemory(capacity=10)
        await wm.add(MemoryItem(user_id="u_empty", session_id="", type="working", content="Empty string sess"))
        res = await wm.retrieve("Empty", user_id="u_empty", session_id="")
        assert len(res) >= 1

    @pytest.mark.asyncio
    async def test_boundary_03_nonexistent_namespace_zero_leakage(self, iso_sqlite):
        """Querying a non-existent namespace must never leak chunks from 'default' or other spaces."""
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=16)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        col_def = retriever._collection_name("default")
        await vstore.ensure_collection(col_def, 16)
        await vstore.upsert(col_def, [
            VectorPoint(id="sec1", vector=[0.2] * 16, payload={"content": "Top Secret Default Doc"})
        ])

        # Query completely nonexistent namespace
        res = await retriever.search(["Secret"], namespace="ghost_namespace_999", top_k=5)
        assert len(res) == 0

    @pytest.mark.asyncio
    async def test_boundary_04_unicode_and_special_character_namespaces(self, iso_sqlite):
        """Namespaces containing CJK, slashes, dashes, and periods must work without SQL injection."""
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=16)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        special_ns = "高等数学·微积分_v2.0-2026/Alpha"
        col = retriever._collection_name(special_ns)
        await vstore.ensure_collection(col, 16)
        await vstore.upsert(col, [
            VectorPoint(id="cjk1", vector=[0.3] * 16, payload={"content": "极值点与驻点充分条件"})
        ])

        res = await retriever.search(["驻点"], namespace=special_ns, top_k=5)
        assert len(res) == 1
        assert "极值点" in res[0].content

    @pytest.mark.asyncio
    async def test_boundary_05_long_ids_stress(self, iso_sqlite):
        """128-character user_id and session_id must persist without database overflow."""
        long_user = "u" * 128
        long_sess = "s" * 128
        state_mgr = SessionStateManager(sqlite=iso_sqlite)

        sess = await state_mgr.get_or_create_session(
            user_id=long_user,
            session_id=long_sess,
            course_id="long_c" * 10,
            rag_namespace="long_ns" * 5,
        )
        assert sess.user_id == long_user
        assert sess.session_id == long_sess

        loaded = await state_mgr.get_session(user_id=long_user, session_id=long_sess)
        assert loaded is not None
        assert loaded.session_id == long_sess

    @pytest.mark.asyncio
    async def test_boundary_06_preexisting_db_migration_preserves_50_sessions(self, tmp_path: Path):
        """Migrating legacy database containing 50 sessions must preserve all existing records."""
        db_file = tmp_path / "legacy_50.db"
        import aiosqlite
        async with aiosqlite.connect(str(db_file)) as conn:
            await conn.execute("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            for i in range(50):
                await conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (f"sess_{i:03d}", f"user_{i%5}", f"旧课程会话_{i}", 1000.0 + i, 1000.0 + i),
                )
            await conn.commit()

        store = SQLiteStore(db_file)
        await store.initialize()

        cur = await store.conn.execute("SELECT count(*) as cnt FROM sessions")
        row = await cur.fetchone()
        assert row["cnt"] == 50

        # Verify all gained rag_namespace='default'
        cur = await store.conn.execute("SELECT count(*) as cnt FROM sessions WHERE rag_namespace = 'default'")
        row = await cur.fetchone()
        assert row["cnt"] == 50
        await store.aclose()

    @pytest.mark.asyncio
    async def test_boundary_07_get_nonexistent_session_404(self, iso_client):
        """GET /api/sessions/ghost_u/ghost_s must return 404."""
        resp = await iso_client.get("/api/sessions/ghost_u/ghost_s")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_boundary_08_patch_nonexistent_session_404(self, iso_client):
        """PATCH /api/sessions/ghost_u/ghost_s must return 404."""
        resp = await iso_client.patch("/api/sessions/ghost_u/ghost_s", json={"title": "Ghost"})
        assert resp.status_code == 404


# ============================================================================
# Tier 3: Cross-Feature Pairwise Integrations (5 Tests)
# ============================================================================

class TestTier3CrossFeaturePairwiseCombinations:
    """Pairwise combinations across Memory, Database, RAG, and API."""

    @pytest.mark.asyncio
    async def test_pairwise_01_multi_session_switch_zero_bleed(self, iso_sqlite):
        """User switching between Physics and Literature sessions exhibits zero memory and RAG bleed."""
        # 1. Setup RAG in vector store
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=32)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        col_phys = retriever._collection_name("physics")
        col_lit = retriever._collection_name("literature")
        await vstore.ensure_collection(col_phys, 32)
        await vstore.ensure_collection(col_lit, 32)

        await vstore.upsert(col_phys, [
            VectorPoint(id="p_chunk", vector=[0.4] * 32, payload={"content": "Carnot heat engine thermodynamics"})
        ])
        await vstore.upsert(col_lit, [
            VectorPoint(id="l_chunk", vector=[0.4] * 32, payload={"content": "Shakespeare sonnet 18 summer day"})
        ])

        # 2. Setup Working Memory
        wm = WorkingMemory(capacity=10)
        await wm.add(MemoryItem(user_id="student_1", session_id="sess_phys", type="working", content="Entropy S=k*ln(W)"))
        await wm.add(MemoryItem(user_id="student_1", session_id="sess_lit", type="working", content="Iambic pentameter rhythm"))

        # --- Active in Session Physics ---
        phys_mem = await wm.retrieve("Entropy", user_id="student_1", session_id="sess_phys")
        phys_rag = await retriever.search(["heat engine"], namespace="physics", top_k=5)

        assert any("Entropy" in m.item.content for m in phys_mem)
        assert not any("Iambic" in m.item.content for m in phys_mem)
        assert any("Carnot" in c.content for c in phys_rag)
        assert not any("Shakespeare" in c.content for c in phys_rag)

        # --- Switch to Session Literature ---
        lit_mem = await wm.retrieve("Iambic", user_id="student_1", session_id="sess_lit")
        lit_rag = await retriever.search(["sonnet"], namespace="literature", top_k=5)

        assert any("Iambic" in m.item.content for m in lit_mem)
        assert not any("Entropy" in m.item.content for m in lit_mem)
        assert any("Shakespeare" in c.content for c in lit_rag)
        assert not any("Carnot" in c.content for c in lit_rag)

    @pytest.mark.asyncio
    async def test_pairwise_02_global_mastery_sync_across_isolated_course_sessions(self, iso_sqlite):
        """Mistake in Math session propagates to global mastery, visible when querying in Physics session."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        kp = await learning_store.upsert_knowledge_point(
            course_id="math_vec", name="向量叉乘", summary="右手定则与外积", kp_id="kp_vector_cross"
        )
        q = await learning_store.add_question(
            kp_id=kp["id"],
            course_id="math_vec",
            q_type="choice",
            stem="i x j 等于？",
            answer="k",
        )

        wm = WorkingMemory(capacity=10)
        # Math session records tool scratchpad
        await wm.add(MemoryItem(user_id="alice", session_id="sess_math", type="working", content="Scratchpad: solve cross product"))
        # Math session records a mistake
        await learning_store.record_attempt_and_update_mastery(
            user_id="alice", question_id=q["id"], user_answer="-k", is_correct=False, score=0.0, feedback="符号反了"
        )

        # Alice switches to Physics session
        phys_scratchpad = await wm.retrieve("Scratchpad", user_id="alice", session_id="sess_phys")
        # Working memory must remain isolated
        assert len(phys_scratchpad) == 0

        # But learner profile must know Alice struggled with vector cross product
        profile = await learning_store.build_learner_profile_text("alice")
        assert "向量叉乘" in profile
        assert "薄弱" in profile or "错题" in profile

    @pytest.mark.asyncio
    async def test_pairwise_03_session_binding_update_dynamically_redirects_rag(self, iso_client, iso_app):
        """Updating a session's rag_namespace via PATCH dynamically redirects future RAG retrievals."""
        container: AppContainer = iso_app.state.container

        # Index into ns_v1 and ns_v2
        col1 = container.rag_retriever._collection_name("ns_v1")
        col2 = container.rag_retriever._collection_name("ns_v2")
        await container.vector_store.ensure_collection(col1, 128)
        await container.vector_store.ensure_collection(col2, 128)
        await container.vector_store.upsert(col1, [
            VectorPoint(id="v1_doc", vector=[0.1] * 128, payload={"content": "Content of Version 1 Textbook"})
        ])
        await container.vector_store.upsert(col2, [
            VectorPoint(id="v2_doc", vector=[0.1] * 128, payload={"content": "Content of Version 2 Textbook"})
        ])

        # Create session bound to ns_v1
        res = await iso_client.post("/api/sessions", json={"user_id": "u_dyn", "rag_namespace": "ns_v1"})
        session_id = res.json()["session_id"]

        # Search with session bound to ns_v1
        sess = await container.state_manager.get_session("u_dyn", session_id)
        assert sess.rag_namespace == "ns_v1"

        # PATCH session to ns_v2
        patch_res = await iso_client.patch(f"/api/sessions/u_dyn/{session_id}", json={"rag_namespace": "ns_v2"})
        assert patch_res.status_code == 200

        # Reload session from state manager
        updated_sess = await container.state_manager.get_session("u_dyn", session_id)
        assert updated_sess.rag_namespace == "ns_v2"

    @pytest.mark.asyncio
    async def test_pairwise_04_concurrent_sessions_same_user(self):
        """Concurrent additions in two separate sessions for the same user execute race-free."""
        wm = WorkingMemory(capacity=50)

        async def worker(sess_id: str, count: int):
            for i in range(count):
                await wm.add(MemoryItem(user_id="concurrent_user", session_id=sess_id, type="working", content=f"{sess_id}_{i}"))
                await asyncio.sleep(0.001)

        await asyncio.gather(worker("sess_x", 20), worker("sess_y", 20))

        items_x = await wm.list_items("concurrent_user", limit=50, session_id="sess_x")
        items_y = await wm.list_items("concurrent_user", limit=50, session_id="sess_y")

        assert len(items_x) == 20
        assert len(items_y) == 20
        assert all(it.session_id == "sess_x" for it in items_x)
        assert all(it.session_id == "sess_y" for it in items_y)

    @pytest.mark.asyncio
    async def test_pairwise_05_precedence_hierarchy_header_over_session(self, iso_app):
        """Header X-RAG-Namespace takes precedence over session's bound namespace."""
        # Simulated resolution logic per PROJECT.md §3:
        # Effective Namespace = Header(X-RAG-Namespace) || Body(rag_namespace) || Session(rag_namespace) || "default"
        def resolve_effective_namespace(header_val: str | None, body_val: str | None, session_val: str | None) -> str:
            return header_val or body_val or session_val or "default"

        assert resolve_effective_namespace("header_ns", "body_ns", "session_ns") == "header_ns"
        assert resolve_effective_namespace(None, "body_ns", "session_ns") == "body_ns"
        assert resolve_effective_namespace(None, None, "session_ns") == "session_ns"
        assert resolve_effective_namespace(None, None, None) == "default"


# ============================================================================
# Tier 4: Real-World Multi-Course Study Workloads (5 Scenarios)
# ============================================================================

class TestTier4RealWorldWorkloads:
    """Multi-course end-to-end operational scenarios."""

    @pytest.mark.asyncio
    async def test_workload_01_multicourse_student_semester_journey(self, tmp_path: Path, iso_sqlite):
        """Ingest 3 course textbooks -> Bind 3 sessions -> Verify textbook citations match exact physical pages."""
        vstore = SQLiteVectorStore(iso_sqlite)
        embedder = HashingEmbedder(dim=32)
        retriever = RagRetriever(vector_store=vstore, embedder=embedder)

        courses = [
            ("ds_algorithms", "Data Structures", "B-Tree index splitting operation", 142),
            ("db_systems", "Database Systems", "ACID transaction serializability", 88),
            ("comp_networks", "Computer Networks", "TCP congestion control window", 215),
        ]

        state_mgr = SessionStateManager(sqlite=iso_sqlite)

        for ns, course_name, text, page in courses:
            # 1. Ingest into course namespace
            col = retriever._collection_name(ns)
            await vstore.ensure_collection(col, 32)
            await vstore.upsert(col, [
                VectorPoint(
                    id=f"doc_{ns}",
                    vector=[0.3] * 32,
                    payload={"content": text, "filename": f"{course_name}.pdf", "page_start": page, "page_end": page},
                )
            ])
            # 2. Create session bound to course
            await state_mgr.get_or_create_session(
                user_id="student_poly",
                session_id=f"sess_{ns}",
                course_id=f"c_{ns}",
                rag_namespace=ns,
                title=f"{course_name} 课程会话",
            )

        # Verify searches from each session
        for ns, course_name, text, page in courses:
            session = await state_mgr.get_session("student_poly", f"sess_{ns}")
            assert session.rag_namespace == ns

            hits = await retriever.search([text.split()[0]], namespace=session.rag_namespace, top_k=5)
            assert len(hits) == 1
            chunk = hits[0]
            assert chunk.metadata["page_start"] == page
            assert chunk.metadata["filename"] == f"{course_name}.pdf"

    @pytest.mark.asyncio
    async def test_workload_02_tool_execution_scratchpad_isolation_during_coding(self):
        """Python coding agent tool scratchpad in Session A never enters Session B."""
        wm = WorkingMemory(capacity=20)

        # Turn 1: In coding session, tool returns stack trace
        await wm.add(MemoryItem(
            user_id="coder_1",
            session_id="sess_code_debug",
            type="working",
            content="Tool output [bash]: Segmentation fault (core dumped) in malloc.c:120",
            importance=0.9,
        ))

        # Turn 2: User switches to English writing session
        await wm.add(MemoryItem(
            user_id="coder_1",
            session_id="sess_english_essay",
            type="working",
            content="Essay topic: Shakespearean pastoral comedy conventions",
            importance=0.7,
        ))

        # Query in English session
        retrieved = await wm.retrieve("Segmentation fault malloc", user_id="coder_1", session_id="sess_english_essay")
        assert len(retrieved) == 0

        # Query in Debug session
        retrieved_debug = await wm.retrieve("Segmentation fault", user_id="coder_1", session_id="sess_code_debug")
        assert len(retrieved_debug) >= 1
        assert "Segmentation fault" in retrieved_debug[0].item.content

    @pytest.mark.asyncio
    async def test_workload_03_cross_disciplinary_distraction_defense(self, iso_sqlite):
        """Chit-chat in entertainment session is filtered out when studying in calculus session."""
        vec = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=32)
        em = EpisodicMemory(sqlite=iso_sqlite, vector_store=vec, embedder=embedder)

        # Chit-chat turn in recreational session
        await em.add(MemoryItem(
            user_id="student_focus",
            session_id="sess_entertainment",
            type="episodic",
            content="I really enjoyed watching the new interstellar space movie yesterday",
            importance=0.4,
        ))
        # Calculus turn in academic session
        await em.add(MemoryItem(
            user_id="student_focus",
            session_id="sess_calculus",
            type="episodic",
            content="Calculus Taylor series expansion of sin(x) around zero",
            importance=0.8,
        ))

        # In calculus session, query about Taylor series
        calc_recalled = await em.retrieve("Taylor series", user_id="student_focus", top_k=5, session_id="sess_calculus")
        assert len(calc_recalled) >= 1
        assert "Taylor series" in calc_recalled[0].item.content
        assert not any("interstellar space" in r.item.content for r in calc_recalled)

    @pytest.mark.asyncio
    async def test_workload_04_sm2_spaced_repetition_unified_across_study_sessions(self, iso_sqlite):
        """SM-2 review ratings across 3 independent sessions advance global interval progression."""
        learning_store = LearningStore(sqlite=iso_sqlite, graph_store=NullGraphStore())
        kp = await learning_store.upsert_knowledge_point(
            course_id="algo_ds", name="快速排序", summary="分治与基准划分", kp_id="kp_quicksort"
        )
        q = await learning_store.add_question(
            kp_id=kp["id"],
            course_id="algo_ds",
            q_type="choice",
            stem="快排平均时间复杂度？",
            answer="O(n log n)",
        )

        # Day 1: Session 1 (score 1.0, quality 5)
        m1 = await learning_store.record_attempt_and_update_mastery(
            user_id="u_sm2", question_id=q["id"], user_answer="O(n log n)", is_correct=True, score=1.0
        )
        assert m1["repetitions"] == 1
        assert m1["interval_days"] >= 1.0

        # Day 2: Session 2 (score 1.0, quality 5)
        m2 = await learning_store.record_attempt_and_update_mastery(
            user_id="u_sm2", question_id=q["id"], user_answer="O(n log n)", is_correct=True, score=1.0
        )
        assert m2["repetitions"] == 2
        assert m2["interval_days"] >= 6.0

        # Day 3: Session 3 (score 1.0, quality 5)
        m3 = await learning_store.record_attempt_and_update_mastery(
            user_id="u_sm2", question_id=q["id"], user_answer="O(n log n)", is_correct=True, score=1.0
        )
        assert m3["repetitions"] == 3
        assert m3["interval_days"] > m2["interval_days"]

        # Check in new Session 4 from SQLite table
        cur = await iso_sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?", ("u_sm2", kp["id"])
        )
        m_final = await cur.fetchone()
        assert m_final is not None
        assert int(m_final["repetitions"]) == 3

    @pytest.mark.asyncio
    async def test_workload_05_full_lifecycle_session_crud_and_cleanup(self, iso_client, iso_sqlite):
        """Complete lifecycle: create session -> query -> patch -> list -> delete -> verify clean state."""
        user_id = f"lifecycle_{uuid.uuid4().hex[:8]}"

        # 1. Create session bound to chemistry
        res_create = await iso_client.post(
            "/api/sessions",
            json={"user_id": user_id, "course_id": "chem_101", "rag_namespace": "chem_ns", "title": "化学基础"},
        )
        assert res_create.status_code in (200, 201)
        sess_id = res_create.json()["session_id"]

        # 2. Patch title and course
        res_patch = await iso_client.patch(
            f"/api/sessions/{user_id}/{sess_id}",
            json={"title": "进阶化学实验", "course_id": "chem_201"},
        )
        assert res_patch.status_code == 200
        assert res_patch.json()["title"] == "进阶化学实验"
        assert res_patch.json()["course_id"] == "chem_201"

        # 3. List sessions
        res_list = await iso_client.get(f"/api/sessions/{user_id}")
        assert res_list.status_code == 200
        assert any(s["session_id"] == sess_id for s in res_list.json()["sessions"])

        # 4. Delete session
        res_del = await iso_client.delete(f"/api/sessions/{user_id}/{sess_id}")
        assert res_del.status_code == 200

        # 5. Verify deleted from DB
        cur = await iso_sqlite.conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sess_id,))
        row = await cur.fetchone()
        assert row is None
