"""Adversarial Empirical Challenge Test Suite for Milestone M2: Backend Memory Layered Isolation.

Authored by Empirical Challenger 2.
Adversarially stress-tests:
1. Fine-grained relevance scores near 0.50 cutoff (EpisodicMemory):
   - Mathematically calibrated cosine similarities near 0.50 threshold (0.4966 vs 0.5000 vs 0.5031)
   - Ensuring cross-session candidates < 0.50 are dropped while >= 0.50 survive
   - Ensuring current-session candidates below 0.50 are NOT dropped
   - Validating session_affinity multipliers (1.35x vs 0.65x vs 1.0x) in breakdown
2. Multi-session history separation under identical or similar keywords:
   - Identical keyword query current-session affinity boost and ranking inversion
   - Cross-topic noise immunity with real embedder
   - Working memory identical keywords zero-leakage guarantee
   - Candidate pool capacity stress against distractor crowding
3. MemoryLifecycle consolidation scoping:
   - consolidate(user_id, session_id="sess_1") strictly consolidates & purges sess_1 working memories
   - sess_2 working memories remain 100% untouched and unpurged
   - Episodic to Semantic consolidation scoping per session
   - Unscoped session_id=None consolidation fallback
4. Concurrency & cross-user isolation stress:
   - Concurrent eviction independence across sessions
   - Cross-user same session_id isolation
   - Whitespace and empty string session_id robustness and cross-store normalization difference
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import numpy as np
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


class ControlledDenseEmbedder:
    """Deterministic dense embedder enabling exact cosine control without external network."""

    def __init__(self, dim: int = 4, tag: str = "controlled_dense_4"):
        self.dim = dim
        self.tag = tag
        self._custom_vectors: dict[str, list[float]] = {}

    def register_vector(self, text: str, vector: list[float]) -> None:
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized = (np.asarray(vector, dtype=np.float32) / norm).tolist()
        else:
            normalized = vector
        self._custom_vectors[text] = normalized

    async def embed(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            if t in self._custom_vectors:
                res.append(self._custom_vectors[t])
            else:
                # Default unit vector along axis 0
                res.append([1.0, 0.0, 0.0, 0.0])
        return res


@pytest.fixture
async def adv_sqlite(tmp_path: Path):
    db_file = tmp_path / "adv_m2_challenge.db"
    store = SQLiteStore(db_file)
    await store.initialize()
    yield store
    await store.aclose()


@pytest.fixture
def controlled_embedder():
    return ControlledDenseEmbedder(dim=4, tag="test_dense_4")


@pytest.fixture
async def controlled_episodic(adv_sqlite: SQLiteStore, controlled_embedder: ControlledDenseEmbedder):
    vstore = InMemoryVectorStore()
    mem = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=controlled_embedder)
    await mem._ensure_store()
    return mem, controlled_embedder


# ============================================================================
# Group 1: Fine-Grained Relevance Scores Near 0.50 Cutoff (EpisodicMemory)
# ============================================================================

class TestEpisodicMemoryBoundaryCutoff:
    """Stress-tests the mathematical cutoff of 0.50 and session affinity multipliers."""

    @pytest.mark.asyncio
    async def test_boundary_just_below_0_50_cutoff_dropped(self, controlled_episodic):
        """Cross-session memory with final_score just under 0.50 (e.g. 0.4966) MUST be dropped."""
        mem, embedder = controlled_episodic
        query_text = "target query for boundary test"
        doc_text = "cross session document slightly below cutoff"

        # Query vector: [1.0, 0.0, 0.0, 0.0]
        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])
        # Cosine = 0.4100
        # sim_score = (0.41 + 1) / 2 = 0.705
        # base_score = (0.705 * 0.8 + 0.2) * (0.8 + 0.5 * 0.4) = 0.764 * 1.0 = 0.764
        # final_score = 0.764 * 0.65 = 0.4966 < 0.50
        cos_val = 0.4100
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc_text, [cos_val, sin_val, 0.0, 0.0])

        item = MemoryItem(
            user_id="user_adv_1",
            session_id="session_other",
            type="episodic",
            content=doc_text,
            importance=0.5,
        )
        await mem.add(item)

        # Retrieve with current session "session_current" -> candidate is cross-session
        results = await mem.retrieve(query=query_text, user_id="user_adv_1", top_k=5, session_id="session_current")
        assert len(results) == 0, f"Expected 0 results because final_score < 0.50, got {len(results)}"

    @pytest.mark.asyncio
    async def test_boundary_exact_0_50_cutoff_survives(self, controlled_episodic):
        """Cross-session memory with final_score exactly 0.50 MUST survive (since cutoff is final_score < 0.50)."""
        mem, embedder = controlled_episodic
        query_text = "exact boundary query"
        doc_text = "cross session document exactly at cutoff"

        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])
        # Cosine = 11 / 26 ≈ 0.4230769
        # sim_score = (11/26 + 1)/2 = 37/52 ≈ 0.711538
        # base_score = 37/52 * 0.8 + 0.2 = 10/13 ≈ 0.7692307
        # final_score = 10/13 * 0.65 = 0.5000000
        cos_val = 11.0 / 26.0
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc_text, [cos_val, sin_val, 0.0, 0.0])

        item = MemoryItem(
            user_id="user_adv_2",
            session_id="session_other",
            type="episodic",
            content=doc_text,
            importance=0.5,
        )
        await mem.add(item)

        results = await mem.retrieve(query=query_text, user_id="user_adv_2", top_k=5, session_id="session_current")
        assert len(results) == 1, "Candidate with final_score == 0.50 must NOT be dropped by `< 0.50` check!"
        assert pytest.approx(results[0].score, abs=1e-4) == 0.50
        assert results[0].breakdown["session_affinity"] == 0.65

    @pytest.mark.asyncio
    async def test_boundary_just_above_0_50_survives(self, controlled_episodic):
        """Cross-session memory with final_score just above 0.50 (e.g. 0.5031) MUST survive."""
        mem, embedder = controlled_episodic
        query_text = "query above cutoff"
        doc_text = "cross session document slightly above cutoff"

        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])
        # Cosine = 0.4350
        # sim_score = (0.435 + 1) / 2 = 0.7175
        # base_score = (0.7175 * 0.8 + 0.2) * 1.0 = 0.774
        # final_score = 0.774 * 0.65 = 0.5031 >= 0.50
        cos_val = 0.4350
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc_text, [cos_val, sin_val, 0.0, 0.0])

        item = MemoryItem(
            user_id="user_adv_3",
            session_id="session_other",
            type="episodic",
            content=doc_text,
            importance=0.5,
        )
        await mem.add(item)

        results = await mem.retrieve(query=query_text, user_id="user_adv_3", top_k=5, session_id="session_current")
        assert len(results) == 1
        assert pytest.approx(results[0].score, abs=1e-3) == 0.5031
        assert results[0].breakdown["session_affinity"] == 0.65

    @pytest.mark.asyncio
    async def test_current_session_score_below_0_50_not_dropped(self, controlled_episodic):
        """Current session memories are not subject to the cross-session 0.50 drop threshold."""
        mem, embedder = controlled_episodic
        query_text = "weak current session query"
        doc_text = "current session document with low similarity"

        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])
        # Very low cosine similarity: -0.6
        # sim_score = (-0.6 + 1) / 2 = 0.20
        # base_score = (0.20 * 0.8 + 0.2) * 1.0 = 0.36
        # final_score = 0.36 * 1.35 = 0.486 (< 0.50)
        cos_val = -0.6
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc_text, [cos_val, sin_val, 0.0, 0.0])

        item = MemoryItem(
            user_id="user_adv_4",
            session_id="session_current",
            type="episodic",
            content=doc_text,
            importance=0.5,
        )
        await mem.add(item)

        results = await mem.retrieve(query=query_text, user_id="user_adv_4", top_k=5, session_id="session_current")
        assert len(results) == 1, "Current session memory with score 0.486 should NOT be discarded!"
        assert pytest.approx(results[0].score, abs=1e-3) == 0.486
        assert results[0].breakdown["session_affinity"] == 1.35

    @pytest.mark.asyncio
    async def test_importance_scaling_cutoff_interaction(self, controlled_episodic):
        """Verifies that high importance enables cross-session memory to survive lower similarity, while low importance causes it to be discarded."""
        mem, embedder = controlled_episodic
        query = "importance scaling query"
        doc_high = "high importance doc"
        doc_low = "low importance doc"

        embedder.register_vector(query, [1.0, 0.0, 0.0, 0.0])
        # Moderate cosine = 0.30
        # sim_score = (0.30 + 1) / 2 = 0.65
        # For high importance (1.0):
        # importance_factor = 0.8 + 1.0 * 0.4 = 1.2
        # base_score = (0.65 * 0.8 + 0.2) * 1.2 = 0.72 * 1.2 = 0.864
        # final_score = 0.864 * 0.65 = 0.5616 (>= 0.50 -> SURVIVES)
        #
        # For low importance (0.1):
        # importance_factor = 0.8 + 0.1 * 0.4 = 0.84
        # base_score = (0.65 * 0.8 + 0.2) * 0.84 = 0.72 * 0.84 = 0.6048
        # final_score = 0.6048 * 0.65 = 0.39312 (< 0.50 -> DROPPED)
        cos_val = 0.30
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc_high, [cos_val, sin_val, 0.0, 0.0])
        embedder.register_vector(doc_low, [cos_val, sin_val, 0.0, 0.0])

        await mem.add(MemoryItem(user_id="user_adv_imp", session_id="sess_cross", type="episodic", content=doc_high, importance=1.0))
        await mem.add(MemoryItem(user_id="user_adv_imp", session_id="sess_cross", type="episodic", content=doc_low, importance=0.1))

        results = await mem.retrieve(query=query, user_id="user_adv_imp", top_k=5, session_id="sess_target")
        assert len(results) == 1
        assert results[0].item.content == doc_high
        assert pytest.approx(results[0].score, abs=1e-3) == 0.5616

    @pytest.mark.asyncio
    async def test_session_id_none_applies_neutral_factor(self, controlled_episodic):
        """When session_id is None, session_factor must be exactly 1.0 and no cutoff is applied."""
        mem, embedder = controlled_episodic
        query = "neutral session query"
        doc = "neutral session doc"

        embedder.register_vector(query, [1.0, 0.0, 0.0, 0.0])
        # Very low cosine: -0.8
        # sim_score = (-0.8 + 1)/2 = 0.10
        # base_score = (0.10 * 0.8 + 0.2) * 1.0 = 0.28
        # When session_id is None: final_score = base_score * 1.0 = 0.28
        cos_val = -0.8
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc, [cos_val, sin_val, 0.0, 0.0])

        await mem.add(MemoryItem(user_id="user_adv_none", session_id="any_sess", type="episodic", content=doc, importance=0.5))

        results = await mem.retrieve(query=query, user_id="user_adv_none", top_k=5, session_id=None)
        assert len(results) == 1
        assert pytest.approx(results[0].score, abs=1e-3) == 0.28
        assert results[0].breakdown["session_affinity"] == 1.0

    @pytest.mark.asyncio
    async def test_session_id_empty_string_applies_neutral_factor(self, controlled_episodic):
        """When session_id is empty string '', session_factor must be exactly 1.0 and no cutoff is applied."""
        mem, embedder = controlled_episodic
        query = "empty str session query"
        doc = "empty str session doc"

        embedder.register_vector(query, [1.0, 0.0, 0.0, 0.0])
        cos_val = -0.8
        sin_val = math.sqrt(1.0 - cos_val**2)
        embedder.register_vector(doc, [cos_val, sin_val, 0.0, 0.0])

        await mem.add(MemoryItem(user_id="user_adv_empty", session_id="any_sess", type="episodic", content=doc, importance=0.5))

        results = await mem.retrieve(query=query, user_id="user_adv_empty", top_k=5, session_id="")
        assert len(results) == 1
        assert pytest.approx(results[0].score, abs=1e-3) == 0.28
        assert results[0].breakdown["session_affinity"] == 1.0


# ============================================================================
# Group 2: Multi-Session History Separation Under Identical or Similar Keywords
# ============================================================================

class TestMultiSessionHistorySeparation:
    """Stress-tests history separation under identical keywords and multi-session competition."""

    @pytest.mark.asyncio
    async def test_identical_keywords_affinity_boost_and_ranking_inversion(self, controlled_episodic):
        """When two sessions have memories with identical content/keywords, current session wins by exactly 1.35 / 0.65 ratio."""
        mem, embedder = controlled_episodic
        query_text = "Python asyncio event loop tasks"
        doc_a = "Session A: Python asyncio event loop tasks and scheduling"
        doc_b = "Session B: Python asyncio event loop tasks and scheduling"

        # Give both identical vector similarity
        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])
        embedder.register_vector(doc_a, [0.8, 0.6, 0.0, 0.0])
        embedder.register_vector(doc_b, [0.8, 0.6, 0.0, 0.0])

        # Add both with identical importance and creation time
        now = time.time()
        item_a = MemoryItem(id="mem_a", user_id="u_inv", session_id="sess_A", type="episodic", content=doc_a, importance=0.8, created_at=now)
        item_b = MemoryItem(id="mem_b", user_id="u_inv", session_id="sess_B", type="episodic", content=doc_b, importance=0.8, created_at=now)
        await mem.add(item_a)
        await mem.add(item_b)

        # 1. Query from session A -> doc_a MUST be rank 1, doc_b MUST be rank 2
        res_a = await mem.retrieve(query=query_text, user_id="u_inv", top_k=5, session_id="sess_A")
        assert len(res_a) == 2
        assert res_a[0].item.id == "mem_a"
        assert res_a[1].item.id == "mem_b"
        ratio_a = res_a[0].score / res_a[1].score
        assert pytest.approx(ratio_a, abs=1e-4) == 1.35 / 0.65

        # 2. Query from session B -> doc_b MUST be rank 1, doc_a MUST be rank 2 (Clean Inversion)
        res_b = await mem.retrieve(query=query_text, user_id="u_inv", top_k=5, session_id="sess_B")
        assert len(res_b) == 2
        assert res_b[0].item.id == "mem_b"
        assert res_b[1].item.id == "mem_a"
        ratio_b = res_b[0].score / res_b[1].score
        assert pytest.approx(ratio_b, abs=1e-4) == 1.35 / 0.65

    @pytest.mark.asyncio
    async def test_working_memory_identical_keywords_strict_zero_leakage(self):
        """WorkingMemory must enforce complete zero-leakage between sessions even with identical text."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        user_id = "user_wm_leakage"

        content_shared = "tool_execution_result: docker compose up -d postgres"
        item_s1 = MemoryItem(user_id=user_id, session_id="sess_1", type="working", content=content_shared, importance=0.9)
        item_s2 = MemoryItem(user_id=user_id, session_id="sess_2", type="working", content=content_shared, importance=0.9)

        await wm.add(item_s1)
        await wm.add(item_s2)

        # Query in sess_1
        res_1 = await wm.retrieve(query="docker compose postgres", user_id=user_id, top_k=10, session_id="sess_1")
        assert len(res_1) == 1
        assert res_1[0].item.session_id == "sess_1"
        assert res_1[0].item.id == item_s1.id

        # Query in sess_2
        res_2 = await wm.retrieve(query="docker compose postgres", user_id=user_id, top_k=10, session_id="sess_2")
        assert len(res_2) == 1
        assert res_2[0].item.session_id == "sess_2"
        assert res_2[0].item.id == item_s2.id

    @pytest.mark.asyncio
    async def test_candidate_pool_expansion_protects_current_session_from_crowding(self, adv_sqlite: SQLiteStore):
        """Tests that when 15 cross-session items exist, the candidate pool expansion (max(top_k * 4, 20)) prevents current session items from being starved."""
        vstore = InMemoryVectorStore()
        embedder = ControlledDenseEmbedder(dim=4, tag="test_dense_pool")
        mem = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)
        await mem._ensure_store()

        query_text = "machine learning gradient descent"
        embedder.register_vector(query_text, [1.0, 0.0, 0.0, 0.0])

        # Add 15 cross-session items with high cosine (0.75)
        # base_score = ((0.75+1)/2 * 0.8 + 0.2) * 1.0 = (0.875*0.8+0.2) = 0.90
        # cross_score = 0.90 * 0.65 = 0.585
        cos_cross = 0.75
        sin_cross = math.sqrt(1.0 - cos_cross**2)
        for i in range(15):
            text_i = f"cross_session_candidate_{i}"
            embedder.register_vector(text_i, [cos_cross, sin_cross, 0.0, 0.0])
            await mem.add(MemoryItem(user_id="user_crowd", session_id="sess_other", type="episodic", content=text_i, importance=0.5))

        # Add 2 current-session items with slightly lower cosine (0.60)
        # base_score = ((0.60+1)/2 * 0.8 + 0.2) * 1.0 = (0.80*0.8+0.2) = 0.84
        # current_score = 0.84 * 1.35 = 1.134
        cos_curr = 0.60
        sin_curr = math.sqrt(1.0 - cos_curr**2)
        text_c1 = "current_session_item_1"
        text_c2 = "current_session_item_2"
        embedder.register_vector(text_c1, [cos_curr, sin_curr, 0.0, 0.0])
        embedder.register_vector(text_c2, [cos_curr, sin_curr, 0.0, 0.0])
        await mem.add(MemoryItem(user_id="user_crowd", session_id="sess_current", type="episodic", content=text_c1, importance=0.5))
        await mem.add(MemoryItem(user_id="user_crowd", session_id="sess_current", type="episodic", content=text_c2, importance=0.5))

        # Retrieve top_k=5 with session_id="sess_current"
        # Since candidate_k = max(5 * 4, 20) = 20, all 17 items (15 cross + 2 current) are examined!
        results = await mem.retrieve(query=query_text, user_id="user_crowd", top_k=5, session_id="sess_current")
        assert len(results) == 5
        # The two current session items MUST be at top 1 and 2 due to 1.35x boost!
        assert results[0].item.content in (text_c1, text_c2)
        assert results[1].item.content in (text_c1, text_c2)
        assert results[0].score > results[2].score


# ============================================================================
# Group 3: MemoryLifecycle Consolidation Scoping
# ============================================================================

class TestLifecycleConsolidationScoping:
    """Stress-tests that consolidate(user_id, session_id="sess_1") only consolidates and purges sess_1."""

    @pytest.mark.asyncio
    async def test_consolidate_scoped_purges_only_target_session_working_memory(self, adv_sqlite: SQLiteStore):
        """Crucial test: consolidate(user_id, session_id='sess_1') purges working memories from sess_1 while leaving sess_2 100% intact."""
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)
        sm = SemanticMemory(vector_store=vstore, graph_store=NullGraphStore(), embedder=embedder)
        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_working_thresh=0.7,
            consolidate_episodic_thresh=0.8,
        )

        user_id = "u_lifecycle_scope"

        # Populate sess_1: 2 high importance (0.9, 0.8), 1 low importance (0.4)
        w1_1 = MemoryItem(id="w1_1", user_id=user_id, session_id="sess_1", type="working", content="sess_1 high item 1", importance=0.9)
        w1_2 = MemoryItem(id="w1_2", user_id=user_id, session_id="sess_1", type="working", content="sess_1 high item 2", importance=0.8)
        w1_3 = MemoryItem(id="w1_3", user_id=user_id, session_id="sess_1", type="working", content="sess_1 low item 3", importance=0.4)
        await wm.add(w1_1)
        await wm.add(w1_2)
        await wm.add(w1_3)

        # Populate sess_2: 2 high importance (0.95, 0.85), 1 low importance (0.3)
        w2_1 = MemoryItem(id="w2_1", user_id=user_id, session_id="sess_2", type="working", content="sess_2 high item 1", importance=0.95)
        w2_2 = MemoryItem(id="w2_2", user_id=user_id, session_id="sess_2", type="working", content="sess_2 high item 2", importance=0.85)
        w2_3 = MemoryItem(id="w2_3", user_id=user_id, session_id="sess_2", type="working", content="sess_2 low item 3", importance=0.3)
        await wm.add(w2_1)
        await wm.add(w2_2)
        await wm.add(w2_3)

        # Pre-condition check
        assert len(await wm.list_items(user_id=user_id, session_id="sess_1")) == 3
        assert len(await wm.list_items(user_id=user_id, session_id="sess_2")) == 3

        # Execute scoped consolidation for sess_1 ONLY
        res = await lifecycle.consolidate(user_id=user_id, session_id="sess_1")

        # 1. Exactly 2 items consolidated
        assert res["working_to_episodic"] == 2

        # 2. In sess_1: high importance items (w1_1, w1_2) must be purged, only w1_3 remains
        remaining_s1 = await wm.list_items(user_id=user_id, session_id="sess_1")
        assert len(remaining_s1) == 1
        assert remaining_s1[0].id == "w1_3"

        # 3. In sess_2: MUST BE 100% UNTOUCHED! (All 3 items survive)
        remaining_s2 = await wm.list_items(user_id=user_id, session_id="sess_2")
        assert len(remaining_s2) == 3
        s2_ids = {it.id for it in remaining_s2}
        assert s2_ids == {"w2_1", "w2_2", "w2_3"}, f"sess_2 working memories were corrupted: {s2_ids}"

        # 4. Check episodic memories created: only 2 items created, all having session_id == 'sess_1'
        ep_items_s1 = await em.list_items(user_id=user_id, session_id="sess_1")
        assert len(ep_items_s1) == 2
        for it in ep_items_s1:
            assert it.session_id == "sess_1"

        # No episodic memories for sess_2
        ep_items_s2 = await em.list_items(user_id=user_id, session_id="sess_2")
        assert len(ep_items_s2) == 0

    @pytest.mark.asyncio
    async def test_consolidate_scoped_episodic_to_semantic(self, adv_sqlite: SQLiteStore):
        """Verifies that episodic to semantic consolidation respects session_id filter."""
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)
        sm = SemanticMemory(vector_store=vstore, graph_store=NullGraphStore(), embedder=embedder)
        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_working_thresh=0.7,
            consolidate_episodic_thresh=0.8,
        )

        user_id = "u_lifecycle_e_to_s"
        # Add high importance episodic memories directly
        e1 = MemoryItem(id="e1", user_id=user_id, session_id="sess_chem", type="episodic", content="Chemical valence bond theory", importance=0.9)
        e2 = MemoryItem(id="e2", user_id=user_id, session_id="sess_phys", type="episodic", content="Quantum wave packet collapse", importance=0.9)
        await em.add(e1)
        await em.add(e2)

        # Consolidate sess_chem only
        res = await lifecycle.consolidate(user_id=user_id, session_id="sess_chem")
        assert res["episodic_to_semantic"] == 1

        sem_items = await sm.list_items(user_id=user_id)
        assert len(sem_items) == 1
        assert "Chemical valence" in sem_items[0].content
        assert sem_items[0].metadata.get("source_episodic_id") == "e1"

    @pytest.mark.asyncio
    async def test_consolidate_without_session_id_aggregates_all(self, adv_sqlite: SQLiteStore):
        """When session_id is None, consolidate processes all user working memories."""
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)
        sm = SemanticMemory(vector_store=vstore, graph_store=NullGraphStore(), embedder=embedder)
        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_working_thresh=0.7,
            consolidate_episodic_thresh=0.8,
        )

        user_id = "u_lifecycle_all"
        await wm.add(MemoryItem(id="w_a", user_id=user_id, session_id="sess_A", type="working", content="sess A note", importance=0.85))
        await wm.add(MemoryItem(id="w_b", user_id=user_id, session_id="sess_B", type="working", content="sess B note", importance=0.85))

        res = await lifecycle.consolidate(user_id=user_id, session_id=None)
        assert res["working_to_episodic"] == 2
        # Both sessions purged of high importance
        assert len(await wm.list_items(user_id=user_id, session_id="sess_A")) == 0
        assert len(await wm.list_items(user_id=user_id, session_id="sess_B")) == 0

    @pytest.mark.asyncio
    async def test_memory_manager_consolidate_passthrough(self, adv_sqlite: SQLiteStore):
        """MemoryManager.consolidate correctly propagates session_id down to MemoryLifecycle."""
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        wm = WorkingMemory(capacity=50, ttl_seconds=3600)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)
        sm = SemanticMemory(vector_store=vstore, graph_store=NullGraphStore(), embedder=embedder)
        lifecycle = MemoryLifecycle(
            working=wm,
            episodic=em,
            semantic=sm,
            consolidate_working_thresh=0.7,
            consolidate_episodic_thresh=0.8,
        )
        mgr = MemoryManager(working=wm, episodic=em, semantic=sm, lifecycle=lifecycle)

        user_id = "u_mgr_consolidate"
        await wm.add(MemoryItem(id="w1", user_id=user_id, session_id="sess_x", type="working", content="sess X item", importance=0.9))
        await wm.add(MemoryItem(id="w2", user_id=user_id, session_id="sess_y", type="working", content="sess Y item", importance=0.9))

        res = await mgr.consolidate(user_id=user_id, session_id="sess_x")
        assert res["working_to_episodic"] == 1
        assert len(await wm.list_items(user_id=user_id, session_id="sess_x")) == 0
        assert len(await wm.list_items(user_id=user_id, session_id="sess_y")) == 1


# ============================================================================
# Group 4: Concurrency & Cross-User Isolation Stress
# ============================================================================

class TestConcurrencyAndUserIsolationStress:
    """Stress tests concurrent operations and user-boundary security."""

    @pytest.mark.asyncio
    async def test_working_memory_concurrent_cross_session_eviction_independence(self):
        """Stress-tests that concurrent writes to session A and session B do not evict each other when capacity is small."""
        capacity = 3
        wm = WorkingMemory(capacity=capacity, ttl_seconds=3600)
        user_id = "user_concurrent_stress"

        async def writer(session_id: str, count: int):
            for i in range(count):
                item = MemoryItem(
                    id=f"{session_id}_item_{i}",
                    user_id=user_id,
                    session_id=session_id,
                    type="working",
                    content=f"Payload {i} in session {session_id}",
                    importance=0.5,
                )
                await wm.add(item)
                await asyncio.sleep(0.001)

        # Concurrently add 10 items to sess_alpha and 10 items to sess_beta
        await asyncio.gather(writer("sess_alpha", 10), writer("sess_beta", 10))

        # Each session must retain exactly its own last `capacity` (3) items
        items_alpha = await wm.list_items(user_id=user_id, session_id="sess_alpha")
        items_beta = await wm.list_items(user_id=user_id, session_id="sess_beta")

        assert len(items_alpha) == capacity
        assert len(items_beta) == capacity

        for it in items_alpha:
            assert it.session_id == "sess_alpha"
            assert "sess_alpha" in it.id
        for it in items_beta:
            assert it.session_id == "sess_beta"
            assert "sess_beta" in it.id

    @pytest.mark.asyncio
    async def test_cross_user_same_session_id_strict_isolation(self, adv_sqlite: SQLiteStore):
        """User Alice and User Bob both use session_id='default'; their memories must remain strictly isolated."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)

        # Working memory add
        await wm.add(MemoryItem(user_id="alice", session_id="default", type="working", content="Alice secret formula", importance=0.8))
        await wm.add(MemoryItem(user_id="bob", session_id="default", type="working", content="Bob secret recipe", importance=0.8))

        # Episodic memory add
        await em.add(MemoryItem(user_id="alice", session_id="default", type="episodic", content="Alice episodic memory", importance=0.8))
        await em.add(MemoryItem(user_id="bob", session_id="default", type="episodic", content="Bob episodic memory", importance=0.8))

        # Retrieve for Alice
        wm_alice = await wm.retrieve(query="secret", user_id="alice", session_id="default")
        assert len(wm_alice) == 1
        assert "Alice" in wm_alice[0].item.content
        assert "Bob" not in wm_alice[0].item.content

        em_alice = await em.retrieve(query="episodic", user_id="alice", session_id="default")
        assert len(em_alice) == 1
        assert em_alice[0].item.user_id == "alice"

    @pytest.mark.asyncio
    async def test_whitespace_and_empty_session_id_behavior(self, adv_sqlite: SQLiteStore):
        """Documents and tests behavior of empty string vs whitespace session_id across memory layers."""
        wm = WorkingMemory(capacity=10, ttl_seconds=3600)
        vstore = InMemoryVectorStore()
        embedder = HashingEmbedder(dim=128)
        em = EpisodicMemory(sqlite=adv_sqlite, vector_store=vstore, embedder=embedder)

        # 1. Empty string session_id '' is handled gracefully without errors
        await wm.add(MemoryItem(user_id="u_empty_str", session_id="", type="working", content="empty string working", importance=0.5))
        await em.add(MemoryItem(user_id="u_empty_str", session_id="", type="episodic", content="empty string episodic", importance=0.5))

        wm_empty = await wm.retrieve(query="empty", user_id="u_empty_str", session_id="")
        assert len(wm_empty) == 1

        em_empty = await em.retrieve(query="empty", user_id="u_empty_str", session_id="")
        assert len(em_empty) == 1
        # In episodic memory, empty string '' activates neutral weighting 1.0
        assert em_empty[0].breakdown["session_affinity"] == 1.0

        # 2. Whitespace-only session_id '   ':
        # WorkingMemory normalizes '   ' to None via _normalize_session_id
        await wm.add(MemoryItem(user_id="u_ws_str", session_id="   ", type="working", content="whitespace working", importance=0.5))
        wm_ws = await wm.retrieve(query="whitespace", user_id="u_ws_str", session_id="   ")
        assert len(wm_ws) == 1
        # EpisodicMemory accepts '   ' literally without error
        await em.add(MemoryItem(user_id="u_ws_str", session_id="   ", type="episodic", content="whitespace episodic", importance=0.5))
        em_ws = await em.retrieve(query="whitespace", user_id="u_ws_str", session_id="   ")
        assert len(em_ws) == 1
